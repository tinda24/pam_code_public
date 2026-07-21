USED_YAML = "base.yaml"
import warnings
import random
import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import hydra
import cv2
import numpy as np
import utils.agent_utils as utils
from utils.logger_ddp import Logger
from utils.draw_training_CSV_to_PNG import plot_losses
from utils.replay_buffer import make_expert_replay_loader
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
from omegaconf import OmegaConf

import shutil
def disk_usage_early_warning():
    total, used, free = shutil.disk_usage(".")
    free = free / (1024**3)
    if free < 12:
        return True
    else:
        return False

def make_optimizer(cfg,agent):
    optim = torch.optim.AdamW(
        agent.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
    )
    return optim

class WorkspaceIL:
    def __init__(self, cfg, rank=0, world_size=1, process_rank=None):
        self.work_dir = Path.cwd()
        if process_rank is None:
            process_rank = rank % world_size
        self.rank = rank
        self.relative_rank = process_rank
        self.world_size = world_size
        print(f"rank={rank} (process_rank={self.relative_rank}) | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"current_device={torch.cuda.current_device()} | name={torch.cuda.get_device_name(torch.cuda.current_device())}")
        if self.relative_rank == 0:
            print(f"workspace: {self.work_dir}")
        self.cfg = cfg

        self.cfg.dataloader.bc_dataset.work_dir = self.work_dir

        utils.set_seed_everywhere(cfg.seed + self.relative_rank)
        self.device = torch.device(f'cuda:{rank}')

        self.agent = hydra.utils.instantiate(self.cfg.agent).to(self.device)
        self.agent.load_snapshot()

        with open(self.work_dir / "full_config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)

        dataset_config = dict(self.cfg.dataloader.bc_dataset)
        dataset_config['rank'] = self.relative_rank
        dataset_config['world_size'] = world_size

        dataset_iterable = hydra.utils.call(dataset_config)
        self.expert_replay_loader = make_expert_replay_loader(dataset_iterable, self.cfg.batch_size)
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.logger = Logger(self.work_dir, use_tb=True, rank=self.relative_rank)

        if world_size > 1:
            self.agent = DDP(self.agent, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        self.optimizer = make_optimizer(cfg,self.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.log_every_steps, 1)
        save_every_step = utils.Every(self.cfg.save_every_steps, 1)
        fig_every_step = utils.Every(self.cfg.fig_every_steps, 1)

        metrics = None
        early_warning_available_flag = True
        while train_until_step(self.global_step):

            agent = self.agent.module if isinstance(self.agent, DDP) else self.agent
            metrics = agent.update(self.expert_replay_iter, self.global_step)

            loss = metrics["actor_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            dist.barrier()
            if save_every_step(self.global_step) and self.rank == self.cfg.multi_gpu[0]:
                self.save_snapshot()

            if early_warning_available_flag and disk_usage_early_warning() and self.rank == self.cfg.multi_gpu[0]:
                self.save_snapshot()
                print(f"@@@  disk usage is less than 20GB, save snapshot")
                print(f"@@@  disk usage is less than 20GB, save snapshot")
                early_warning_available_flag = False

            if self.global_step % 2000 == 0:
                early_warning_available_flag = True

            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent
        new_payload = agent.save_snapshot()
        payload.update(new_payload)

        with snapshot.open("wb") as f:
            torch.save(payload, f)

def setup_distributed(rank, world_size, gpu_ids, local_host_id):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(local_host_id)

    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()

def train_worker(rank, world_size, cfg):

    try:

        setup_distributed(rank, world_size, cfg.multi_gpu, cfg.local_host_id)

        gpu_id = cfg.multi_gpu[rank]

        workspace = WorkspaceIL(cfg, rank=gpu_id, world_size=world_size, process_rank=rank)
        workspace.train()

    finally:

        cleanup_distributed()

@hydra.main(config_path="../cfgs", config_name=USED_YAML)
def main(cfg):
    world_size = len(cfg.multi_gpu)
    print(f"@@@  using multi-GPU training with GPUs: {cfg.multi_gpu}")
    print(f"@@@  world_size: {world_size}")

    mp.spawn(
        train_worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
