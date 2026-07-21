import random
import numpy as np
import pickle as pkl
from pathlib import Path
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        cache_path,
        use_img_head,
        imghead_data_cache_path,
        suite,
        scenes,
        tasks,
        sample_interval,
        max_his_cache,

        num_future_queries,
        num_past_queries,
        action_dim,
        action_arrange_type,
        test_mode,
        spilt_datasets_for_multi_gpu,
        work_dir=None,
        base_stats_path=None,
        rank=0,
        world_size=1,
    ):

        self.test_mode = test_mode
        self.rank = rank
        self.world_size = world_size
        self.path = path
        self.cache_path = cache_path
        self.work_dir = work_dir
        self.sample_interval = sample_interval
        self.max_his_cache = max_his_cache
        self.use_img_head = use_img_head
        self.imghead_data_cache_path = imghead_data_cache_path
        self.spilt_datasets_for_multi_gpu = spilt_datasets_for_multi_gpu
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}
        task_names = []
        for scene in scenes:
            task_names.extend([task_name for task_name in tasks[scene]])
        subfolders = [f for f in Path(path).iterdir() if f.is_dir()]

        self._paths = subfolders
        if task_names is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split("/")[-1]
                if task in task_names:
                    idx = task_names.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths
        self.actions = []
        self._max_episode_len = 0
        self._num_samples = 0

        self.num_future_queries = num_future_queries
        self.num_past_queries = num_past_queries
        self.action_dim = action_dim
        self.action_arrange_type = action_arrange_type
        self.num_queries = num_future_queries + num_past_queries

        all_items = []

        for _path_idx in self._paths:
            data_dir = self._paths[_path_idx]
            data_dir = os.path.join(data_dir)
            all_items.extend([str(p) for p in Path(data_dir).rglob("*.hdf5")])

        if self.spilt_datasets_for_multi_gpu:
            items = [item for i, item in enumerate(all_items) if i % world_size == rank]
        else:
            items = all_items
        print(f"have {len(all_items)} hdf5 files, rank {rank} handles {len(items)} files")
        self.items = items

        for item in items:
            with h5py.File(item, "r") as data:
                action=data["qpos"][()]
            episode_len = action.shape[0]
            self._max_episode_len = max(
                self._max_episode_len,
                episode_len
            )
            self._num_samples += episode_len

        with h5py.File(base_stats_path, "r") as data:
            self.stats = {
                "min": data["min"][()],
                "max": data["max"][()],
            }
        if rank == 0:
            print(f'actions min and max: {self.stats["min"]}, {self.stats["max"]}')

            with h5py.File(self.work_dir / "stats.hdf5", "w") as f:
                f.create_dataset("min", data=self.stats["min"])
                f.create_dataset("max", data=self.stats["max"])
        self.len_episodes = len(self.items)

        self._episodes = []
        for item in items:
            parts = item.lstrip("/").split("/")
            task_name_item = parts[-2]
            episode_id_item = parts[-1].replace('episode_','cache_')
            cache_item = os.path.join(cache_path, task_name_item, episode_id_item)
            print(f'adding {task_name_item}, {episode_id_item}')
            with h5py.File(item, "r") as data:
                episode = dict(
                    action=data["qpos"][()],
                    )
            with h5py.File(cache_item, "r") as data:
                episode["Kcache"] = data["Kcache"][()]
                episode["Vcache"] = data["Vcache"][()]
                episode["action_features"] = data["action_features"][()]

            if self.use_img_head:
                imghead_data_cache_item = os.path.join(self.imghead_data_cache_path, task_name_item, f"{episode_id_item}")
                with h5py.File(imghead_data_cache_item, "r") as data:
                    episode["z"] = data["z"][()]

            self._episodes.append(episode)
            if self.test_mode:
                break

    def _preprocess(self, x):

        return (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5)

    def _sample_episode(self):
        idx = random.randint(0, self.len_episodes - 1)
        if self.test_mode:
            idx = 0
        episode = self._episodes[idx]
        return episode, idx

    def _sample(self):
        episodes,idx = self._sample_episode()

        actions = episodes["action"]

        last_action = actions[-1:]
        actions = np.concatenate((actions[1:], last_action), axis=0)

        Kcache = episodes["Kcache"]
        Vcache = episodes["Vcache"]
        action_features = episodes["action_features"]

        num_frames = len(actions)
        sample_idx = np.random.randint(0, num_frames - 1)

        num_actions = self.num_queries
        act_future = torch.zeros((self.num_future_queries, self.action_dim))
        actions = torch.from_numpy(actions)
        act_future[: min(num_frames, sample_idx + self.num_future_queries) - sample_idx] = actions[sample_idx : sample_idx + self.num_future_queries]
        last_action = actions[-1]
        if sample_idx+self.num_future_queries > num_frames:
            act_future[num_frames-sample_idx : ] = last_action
        action_is_valid_future = torch.ones((self.num_future_queries))

        if self.use_img_head:
            z = torch.from_numpy(episodes["z"])
            current_z = z[sample_idx]
            last_z = z[-1:]

            z = torch.cat((z[1:], last_z), axis=0)
            assert z.shape[0] == num_frames, "z and action shapes are inconsistent"
            if sample_idx+self.num_future_queries > num_frames:

                period1 = z[sample_idx:num_frames]
                period2 = last_z.repeat(self.num_future_queries-len(period1), 1,1)
                z_future = torch.cat((period1, period2), axis=0)
            else:
                z_future = z[sample_idx : sample_idx + self.num_future_queries]

            if self.num_past_queries > 0:
                if sample_idx >= self.num_past_queries:
                    z_past = z[sample_idx - self.num_past_queries : sample_idx]
                else:
                    period1 = z[0:sample_idx]
                    period2 = last_z.repeat(self.num_past_queries-len(period1), 1)
                    z_past = torch.cat((period2, period1), axis=0)
                sampled_z = torch.cat((z_past, z_future), axis=0)
            else:
                sampled_z = z_future

        if self.num_past_queries > 0:
            if sample_idx >= self.num_past_queries:
                action_past = actions[sample_idx - self.num_past_queries : sample_idx]
                action_is_valid_past = torch.ones((self.num_past_queries))
            else:
                action_past = torch.zeros((self.num_past_queries,self.action_dim))

                if sample_idx == 0:
                    action_is_valid_past = torch.zeros((self.num_past_queries))
                else:
                    action_past[-sample_idx:] = actions[0:sample_idx]
                    action_is_valid_past = torch.zeros((self.num_past_queries))
                    action_is_valid_past[-sample_idx:] = 1

            action = torch.cat((action_past, act_future), axis=0)
            act_is_valid = torch.cat((action_is_valid_past, action_is_valid_future), axis=0)
        else:
            action = act_future
            act_is_valid = action_is_valid_future

        act_is_valid = act_is_valid.unsqueeze(-1).repeat(1, self.action_dim)

        if self.action_arrange_type == 'joint_chunk':
            action = einops.rearrange(action, "chunk d -> d chunk")
        elif self.action_arrange_type == 'chunk_joint':
            action = action
        elif self.action_arrange_type == 'chunkjoint':
            action = einops.rearrange(action, "chunk d -> (chunk d)")
        else:
            raise ValueError(f"Invalid action arrange type: {self.action_arrange_type}")

        sampled_action = action

        sampled_action_features = action_features[sample_idx]

        idx_list = list(range(0, sample_idx + 1, self.sample_interval))

        if len(idx_list) > self.max_his_cache:
            idx_list = idx_list[-self.max_his_cache:]

        idx_list = idx_list[::-1]

        sampled_Kcache = []
        sampled_Vcache = []
        for cache_idx in idx_list:
            sampled_Kcache.append(Kcache[cache_idx])
            sampled_Vcache.append(Vcache[cache_idx])
        sampled_Kcache = np.stack(sampled_Kcache,axis=0)
        sampled_Vcache = np.stack(sampled_Vcache,axis=0)

        sampled_Kcache = torch.from_numpy(sampled_Kcache)
        sampled_Vcache = torch.from_numpy(sampled_Vcache)

        pad_len = self.max_his_cache - len(idx_list)
        if pad_len > 0:
            cache_shape = list(Kcache.shape)
            cache_shape[0] = pad_len
            sampled_Kcache = torch.cat([sampled_Kcache, torch.zeros(cache_shape)], dim=0)
            sampled_Vcache = torch.cat([sampled_Vcache, torch.zeros(cache_shape)], dim=0)

        pad_len_tensor = torch.ones(self.max_his_cache, dtype=torch.float32)
        if pad_len > 0:
            pad_len_tensor[-pad_len:] = 0

        sampled_action_features = torch.from_numpy(sampled_action_features)

        if self.use_img_head:
            return {
                "Kcache": sampled_Kcache,
                "Vcache": sampled_Vcache,
                "action_features": sampled_action_features,
                "actions": self._preprocess(sampled_action),
                "action_is_valid": act_is_valid,
                "cache_is_valid": pad_len_tensor,
                "correspond_img_latents": sampled_z,
                "current_img_latents": current_z,
            }
        else:
            return {
                "Kcache": sampled_Kcache,
                "Vcache": sampled_Vcache,
                "action_features": sampled_action_features,
                "actions": self._preprocess(sampled_action),
                "action_is_valid": act_is_valid,
                "cache_is_valid": pad_len_tensor,
            }

    def get_norm_stats(self):
        return self.stats

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self.len_episodes
