# PAM Code Public

[Project Page](https://tinda24.github.io/pam/) | [Paper](https://arxiv.org/pdf/2512.24638)

This repository contains the training code for a two-stage vision-language robot policy with a historical memory router. The base stage trains a policy from demonstrations, then the export stage writes backbone KV-cache features for each episode, and the post stage trains memory-augmented components on top of the frozen or partially frozen base policy.
The codebase is organized around Hydra configs and PyTorch DDP training. 

## Repository Layout

```text
agent/                  Policy wrappers for base and post-stage training
cfgs/                   Hydra configuration files
modules/                Model components: image encoders, backbone, action heads, memory router
read_data/              Dataset readers for HDF5 base/post datasets
scripts/                Training and cache-export entry points
tools/                  Utility scripts for data inspection and analysis
utils/                  Logging, replay buffer, plotting, and training helpers
repos/                  Placeholder for optional third-party dependencies
conda_env.yaml          Reference Conda environment
```

## Environment

Create the reference Conda environment:

```bash
conda env create -f conda_env.yaml
conda activate unigen
```

Install any local or platform-specific packages required by your simulator or dataset stack after the environment is created. CUDA, PyTorch, MuJoCo/DM Control, and HDF5 support should match the machine where training is run.

## Third-Party Repositories

```bash
mkdir -p repos

git clone https://github.com/real-stanford/diffusion_policy.git repos/diffusion_policy

git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git /tmp/CleanDiffuser
cp -r /tmp/CleanDiffuser/cleandiffuser repos/cleandiffuser
```

## Data Preparation

Training expects HDF5 demonstrations laid out by task name. The configs point `datapath` to the root directory containing per-task folders:

```text
<datapath>/
  default_task/
    episode_0.hdf5
    episode_1.hdf5
  another_task/
    episode_0.hdf5
```

Each episode should contain the observation keys configured in `cfgs/suite/manipulation.yaml`, such as `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `qpos`, and `task_emb`.

Before running, update or override these config fields:

- `root_dir`: project/output root
- `datapath`: dataset root
- `multi_gpu`: GPU ids used by DDP
- `local_host_id`: distributed training port
- `suite/task`: task config under `cfgs/suite/task/`

## Training Script

Run the complete base-export-post pipeline:

```bash
bash scripts/pipeline.sh \
  --title my_experiment \
  --task-name default_task \
  --base-gpus 0 \
  --post-gpus 0 \
  --export-gpu 0 \
  --base-steps 100000 \
  --post-steps 100000 \
  --cache-dir ./cache/my_experiment_base_cache
```

The script will:

1. Train the base policy with `scripts/train_ddp_base.py`.
2. Export base KV cache files with `scripts/export_cache/export_base_cache.py`.
3. Train the post-stage policy with `scripts/train_ddp_post.py`.

Outputs are written under:

```text
checkpoints/<base_title>/<run_tag>/
checkpoints_post/<post_title>/<run_tag>/
cache/<title>_base_cache/
```

## Manual Commands

Base training:

```bash
python scripts/train_ddp_base.py \
  root_dir=$(pwd) \
  datapath=/path/to/hdf5_data \
  suite/task=default_task \
  multi_gpu=[0] \
  local_host_id=12251
```

Export base KV cache:

```bash
python scripts/export_cache/export_base_cache.py \
  --ckpt /path/to/base_run/snapshot/90000.pt \
  --save-dir /path/to/cache_dir
```

Post-stage training:

```bash
python scripts/train_ddp_post.py \
  root_dir=$(pwd) \
  datapath=/path/to/hdf5_data \
  cache_path=/path/to/cache_dir \
  base_dir=/path/to/base_run \
  round=90000 \
  base_weight_dir=/path/to/base_run/snapshot/90000.pt \
  base_config_dir=/path/to/base_run/.hydra/config.yaml \
  base_stats_path=/path/to/base_run/stats.hdf5 \
  multi_gpu=[0] \
  local_host_id=12252
```

## Configuration Notes

- Top-level configs live in `cfgs/base.yaml` and `cfgs/post.yaml`.
- Task lists are in `cfgs/suite/task/`.
- Base agent settings are in `cfgs/agent/dynamic_actor_base.yaml`.
- Post agent settings are in `cfgs/agent/dynamic_actor_post.yaml`.
- The post stage copies architecture-critical fields from the base run config so the memory stage stays aligned with the trained base policy.


## Acknowledgements

This project builds on ideas, code, models, and benchmarks from the broader robot learning and generative modeling community. In particular, we thank the authors and maintainers of:

- Diffusion Policy, whose implementation patterns and robot policy components can be fetched from `https://github.com/real-stanford/diffusion_policy`.
- CleanDiffuser, which provides diffusion-model utilities used by several action and image head modules and can be fetched from `https://github.com/CleanDiffuserTeam/CleanDiffuser`.
- DINOv2, used as a visual representation backbone in the image encoder configs.
- Public robot manipulation benchmarks and datasets used for evaluation and development.
- PyTorch, Hydra, Hugging Face, and the open-source ecosystem that makes this training stack possible.

If you use this repository, please also cite the upstream projects, datasets, and pretrained models that your experiments depend on.
