import argparse
import glob
import os
import sys

import h5py

DEFAULT_CKPT_DIR = '???.pt'
DEFAULT_SAVE_DIR = '???'

def parse_args():
    parser = argparse.ArgumentParser(description='Export base KV cache for post-stage training')
    parser.add_argument(
        '--ckpt',
        dest='ckpt_dir',
        default=os.environ.get('BASE_CKPT_DIR', DEFAULT_CKPT_DIR),
        help='Base checkpoint path, e.g. /path/to/base_run/snapshot/90000.pt',
    )
    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        default=os.environ.get('CACHE_SAVE_DIR', DEFAULT_SAVE_DIR),
        help='Directory to save exported KV cache files',
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=int(os.environ.get('CACHE_BATCH_SIZE', 64)),
        help='Frames per forward batch when exporting KV cache',
    )
    return parser.parse_args()

def collect_task_names(task_cfg_list):
    task_names = []
    for scene_item in task_cfg_list:
        for _, names in scene_item.items():
            task_names.extend(names)
    return task_names

def to_tensor_batch(frames_np, aug):
    return torch.stack([aug(frames_np[i]) for i in range(len(frames_np))], dim=0)

def export_cache_chunk(model, obs_chunk, task_emb, norm_stats):
    device = model.device
    B = obs_chunk[model.pixel_keys[0]].shape[0]

    assert model.use_language
    lang_emb = torch.as_tensor(task_emb, device=device).float().unsqueeze(0).unsqueeze(0)
    lang_features = lang_emb.repeat(B, model.history_len, 1)
    lang_features = model.language_projector(lang_features)
    lang_features = lang_features.reshape(B * model.history_len, -1)

    features = []
    for key in model.pixel_keys:
        raw_img = obs_chunk[key].to(device)
        img_feature = model.img_encoder(raw_img, lang=lang_features)
        img_feature = model.img_feature_projector(img_feature)
        features.append(img_feature)

    if model.use_proprio:
        proprio = obs_chunk[model.proprio_key].to(device).float()
        if norm_stats is not None:
            min_action = torch.as_tensor(norm_stats['min'], device=device).float()
            max_action = torch.as_tensor(norm_stats['max'], device=device).float()
            proprio = (proprio - min_action) / (max_action - min_action + 1e-5)

        proprio = model.proprio_projector(proprio)
        if proprio.shape[1] == 1:
            proprio = proprio.squeeze(1)
        features.append(proprio)

    features = torch.cat(features, dim=-1).view(B, -1, model.hidden_dim)

    prompt_features = lang_features.view(B, -1, model.hidden_dim)
    features = torch.cat([prompt_features, features], dim=-2)

    action_features, Kcache, Vcache = model.backbone.forward_to_save_KVcache(features)
    return (
        action_features.detach().cpu().to(torch.float32),
        Kcache.detach().cpu().to(torch.float32),
        Vcache.detach().cpu().to(torch.float32),
    )

def export_episode_cache_batched(model, episode_path, task_name, episode_id, base_save_dir, norm_stats, aug, batch_size):
    with h5py.File(episode_path, 'r') as data:
        num_frames = data[model.pixel_keys[0]].shape[0]
        task_emb = data['task_emb'][()]
        print(f'Caching data: {task_name} | episode_id: {episode_id} | frames:{num_frames} | batch_size:{batch_size}')

        action_features_chunks = []
        kcache_chunks = []
        vcache_chunks = []

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            obs_chunk = {
                key: to_tensor_batch(data[key][start:end], aug)
                for key in model.pixel_keys
            }
            if model.use_proprio:
                obs_chunk[model.proprio_key] = torch.from_numpy(data[model.proprio_key][start:end])

            with torch.no_grad():
                action_features, Kcache, Vcache = export_cache_chunk(model, obs_chunk, task_emb, norm_stats)

            action_features_chunks.append(action_features)
            kcache_chunks.append(Kcache)
            vcache_chunks.append(Vcache)
            print(f'  batch [{start}:{end}] done')

        action_features = torch.cat(action_features_chunks, dim=0)
        Kcache = torch.cat(kcache_chunks, dim=0)
        Vcache = torch.cat(vcache_chunks, dim=0)

    save_path = os.path.join(base_save_dir, task_name, f"cache_{episode_id}.hdf5")
    os.makedirs(os.path.join(base_save_dir, task_name), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('Kcache', data=Kcache.numpy(), compression='gzip')
        f.create_dataset('Vcache', data=Vcache.numpy(), compression='gzip')
        f.create_dataset('action_features', data=action_features.numpy(), compression='gzip')
    print(f'KV cache saved to {save_path}')

args = parse_args()
ckpt_dir = os.path.abspath(args.ckpt_dir)
save_dir = os.path.abspath(args.save_dir)
batch_size = args.batch_size

if batch_size <= 0:
    raise ValueError(f'--batch-size must be positive, got {batch_size}')

if not os.path.isfile(ckpt_dir):
    raise FileNotFoundError(f'ckpt not found: {ckpt_dir}')

snapshot_dir = os.path.dirname(ckpt_dir)
if os.path.basename(snapshot_dir) != 'snapshot':
    raise ValueError(f'ckpt should be inside a snapshot directory, got: {ckpt_dir}')

base_run_dir = os.path.dirname(snapshot_dir)
stats_dir = os.path.join(base_run_dir, 'stats.hdf5')
config_dir = os.path.join(base_run_dir, '.hydra', 'config.yaml')

if not os.path.isfile(stats_dir):
    raise FileNotFoundError(f'stats file not found: {stats_dir}')
if not os.path.isfile(config_dir):
    raise FileNotFoundError(f'config file not found: {config_dir}')

os.makedirs(save_dir, exist_ok=True)

import omegaconf

with open(config_dir, 'r') as f:
    config = omegaconf.OmegaConf.load(f)

base_dataset_dir = config.dataloader.bc_dataset.path
task_names = collect_task_names(config.suite.task.tasks)
print(f'base_dataset_dir: {base_dataset_dir}')
print(f'task_names: {task_names}')

traversal_dict = {}
for task_name in task_names:
    episode_dict = {}
    task_dir = os.path.join(base_dataset_dir, task_name)
    hdf5_files = glob.glob(os.path.join(task_dir, '*.hdf5'))
    for hdf5_file in hdf5_files:
        episode_id = hdf5_file.split('episode_')[1].split('.hdf5')[0]
        episode_dict[int(episode_id)] = hdf5_file
    traversal_dict[task_name] = episode_dict

import hydra
import torch

sys.path.append('./')
model = hydra.utils.instantiate(config['agent'])
model.load_snapshot(ckpt_dir)

model = model.cuda()

min_action = None
max_action = None
with h5py.File(stats_dir, 'r') as data:
    for key in data.keys():
        if 'min' in key:
            min_action = data[key][()]
        if 'max' in key:
            max_action = data[key][()]
stats_dict = {
    'min': min_action,
    'max': max_action,
}

from torchvision import transforms

aug = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
for task_name, episode_dict in traversal_dict.items():
    for episode_id, episode_path in episode_dict.items():
        export_episode_cache_batched(
            model=model,
            episode_path=episode_path,
            task_name=task_name,
            episode_id=episode_id,
            base_save_dir=save_dir,
            norm_stats=stats_dict,
            aug=aug,
            batch_size=batch_size,
        )
