import random
import numpy as np
import pickle as pkl
from pathlib import Path
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
import time

from tqdm import tqdm
import cv2

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        obs_mode,

        num_queries,
        debug,
        work_dir=None,
        rank=0,
        world_size=1,
        img_size=None,
        spilt_datasets_for_multi_gpu=True,
    ):
        self.rank = rank
        self.world_size = world_size
        self.obs_mode = obs_mode
        self.img_size = img_size
        self.num_queries = num_queries
        self.spilt_datasets_for_multi_gpu = spilt_datasets_for_multi_gpu

        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}
        task_names = []
        for scene in scenes:
            task_names.extend([task_name for task_name in tasks[scene]])
        self._paths = []
        subfolders = [f for f in Path(path).iterdir() if f.is_dir()]
        self.path = path
        self._paths.extend(subfolders)
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
        all_items = []
        for _path_idx in self._paths:
            data_dir = self._paths[_path_idx]
            if self.obs_mode == 'feature':
                data_dir = str(data_dir) + '_DinoFeatures'
            print('data_dir: ',data_dir)
            all_items.extend([str(p) for p in Path(data_dir).rglob("*.hdf5")])

        self.actions = []
        self.max_episode_len = 0
        self.num_frames = 0
        self.num_frames_for_each_rank = [0]*world_size
        for i, item in enumerate(all_items):
            with h5py.File(item, "r") as data:
                action=data["qpos"][()]

            episode_len = action.shape[0]
            self.max_episode_len = max(self.max_episode_len,episode_len)
            self.num_frames += episode_len
            self.num_frames_for_each_rank[i % world_size] += episode_len
            self.actions.append(action)

            if debug:
                break

        if self.spilt_datasets_for_multi_gpu:
            self.items = [item for i, item in enumerate(all_items) if i % world_size == rank]
        else:
            self.items = all_items
        self.stats = {"min": 0,"max": 1}

        self.actions = np.concatenate(self.actions, axis=0)
        self.stats["min"] = np.min(self.actions, axis=0)
        self.stats["max"] = np.max(self.actions, axis=0)
        stats_path = work_dir / "stats.hdf5"
        if rank == 0:
            print(f'actions min and max: {self.stats["min"]}, {self.stats["max"]}')
            with h5py.File(stats_path, 'w') as f:
                f.create_dataset('min', data=self.stats["min"])
                f.create_dataset('max', data=self.stats["max"])
            print(f'###### stats.hdf5 saved to {stats_path} ######')
        else:

            while not Path(stats_path).exists():
                time.sleep(0.1)

            with h5py.File(stats_path, 'r') as f:
                self.stats["min"] = f['min'][:]
                self.stats["max"] = f['max'][:]

        self.preprocess = self._preprocess_actions

        if self.obs_mode == 'pixel':
            self.aug = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),])
            self.resize_aug = transforms.Compose([transforms.ToPILImage(),transforms.Resize(self.img_size),transforms.ToTensor(),])
        else:
            assert self.obs_mode == 'feature'

        self._episodes = []

        for i,item in enumerate(self.items):
            print(f'rank {self.rank} loading {i+1} / {len(self.items)} files')
            with h5py.File(item, "r") as data:
                episode = {
                    "cam_high": data["cam_high"][:],
                    "cam_left_wrist": data["cam_left_wrist"][:],
                    "cam_right_wrist": data["cam_right_wrist"][:],
                    "actions": data["action"][:],
                    "qposes": data['qpos'][:],
                    "task_emb": data["task_emb"][:],
                }
                self._episodes.append(episode)
            if debug:
                print('debug')
                print('debug')
                print('debug')
                print('debug')
                print('debug')
                print('debug')
                print('debug')

                break

        self.len_episodes = len(self._episodes)

    def _preprocess_actions(self, x):
        return (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5)

    def _sample(self):
        episode = self._episodes[random.randint(0, self.len_episodes - 1)]
        cam_high = episode["cam_high"]
        cam_left_wrist = episode["cam_left_wrist"]
        cam_right_wrist = episode["cam_right_wrist"]
        qposes = episode["qposes"]

        actions = episode["actions"]
        task_emb = episode["task_emb"]
        num_frames = cam_high.shape[0]
        sample_idx = random.randint(0, num_frames - 1)
        if self.obs_mode == 'pixel':
            assert cam_high.shape[1] == self.img_size[0] and cam_high.shape[2] == self.img_size[1]

        cam_high = cam_high[sample_idx]
        cam_left_wrist = cam_left_wrist[sample_idx]
        cam_right_wrist = cam_right_wrist[sample_idx]

        sampled_qpos = qposes[sample_idx]
        actions = episode["actions"]

        if self.obs_mode == 'pixel':
            cam_high = self.aug(cam_high)
            cam_left_wrist = self.aug(cam_left_wrist)
            cam_right_wrist = self.aug(cam_right_wrist)

        num_future_actions = self.num_queries

        last_action = actions[-1:]

        last_action = np.tile(last_action, (2*num_future_actions, 1))
        extended_actions = np.concatenate((actions, last_action), axis=0)
        assert extended_actions.shape[0] == 2*num_future_actions + num_frames

        act_future = extended_actions[sample_idx: sample_idx+num_future_actions]

        return {
            "cam_high": cam_high,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist,
            "qpos": self.preprocess(sampled_qpos),
            "actions": self.preprocess(act_future),
            "task_emb": task_emb,
        }

    def get_norm_stats(self):
        return self.stats

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self.num_frames
