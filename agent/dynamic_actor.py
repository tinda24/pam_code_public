import os
import einops
import numpy as np
from collections import deque
import torch
from torch import nn
from torchvision import transforms as T
from pathlib import Path
import utils.agent_utils as utils
from utils.mlp import MLP

class DynamicActor(nn.Module):
    def __init__(
        self,
        stage,
        obs_mode,
        action_dim,
        proprio_dim,
        img_size,
        num_views,
        pixel_keys,
        proprio_key,
        lang_key,
        hidden_dim,
        num_queries,
        action_arrange_type,
        lr,
        action_loss_coef,
        lang_encoder,
        use_proprio,
        use_language,
        history_len,
        num_act_tokens,
        num_img_tokens,
        num_his_tokens,
        img_encoder,
        backbone,
        action_head,
        unified_optimizer=False,
        device = 'cuda',
    ):
        super(DynamicActor, self).__init__()
        self.stage = stage
        self.obs_mode = obs_mode
        self.img_encoder = img_encoder.to(device)
        self.backbone = backbone.to(device)
        self.action_head = action_head.to(device)
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.img_size = img_size
        self.num_views = num_views
        self.pixel_keys = pixel_keys
        self.proprio_key = proprio_key
        self.lang_key = lang_key
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.action_arrange_type = action_arrange_type
        self.lr = lr
        self.action_loss_coef = action_loss_coef
        self.lang_encoder_type = lang_encoder
        self.use_proprio = use_proprio
        self.use_language = use_language
        self.history_len = history_len
        self.num_act_tokens = num_act_tokens
        self.num_img_tokens = num_img_tokens
        self.num_his_tokens = num_his_tokens
        self.device = device
        self.unified_optimizer = unified_optimizer

        if self.lang_encoder_type == 't5':
            self.language_dim = 768
        elif self.lang_encoder_type == 'minilm':
            self.language_dim = 384
        self.repr_dim = self.img_encoder.output_dim
        self.img_feature_projector = MLP(self.repr_dim,hidden_channels=[self.hidden_dim, self.hidden_dim],).to(device)
        self.img_feature_projector.apply(utils.weight_init)

        if self.use_language:
            self.language_projector = MLP(self.language_dim,hidden_channels=[self.hidden_dim, self.hidden_dim],).to(device)
            self.language_projector.apply(utils.weight_init)
        if self.use_proprio:
            self.proprio_projector = MLP(self.proprio_dim,hidden_channels=[self.hidden_dim, self.hidden_dim],).to(device)
            self.proprio_projector.apply(utils.weight_init)

        self.all_time_actions = torch.zeros(
            [
                1000,
                1000 + self.num_queries,
                self.action_dim,
            ]
        ).to(self.device)

    def update(self, expert_replay_iter, train_step, update=True,use_cached_dino_feature=False):
        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        if self.use_language:
            lang_features = (data[self.lang_key].float()[:, None].repeat(1, self.history_len, 1))
            lang_features = self.language_projector(lang_features)
            lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")
        else:
            lang_features = None

        img_features = []
        for key in self.pixel_keys:
            if self.obs_mode=='pixel':
                raw_img = data[key]
                if len(raw_img.shape) == 5:
                    raw_img = einops.rearrange(raw_img, "b t c h w -> (b t) c h w")
                img_feature = self.img_encoder(raw_img,lang=lang_features)
                img_feature = einops.rearrange(img_feature, "(b t) d -> b t d",b=img_feature.shape[0])
                img_feature = self.img_feature_projector(img_feature)
                img_features.append(img_feature)
            else:
                img_feature = self.img_encoder(data[key],lang=lang_features,use_cached_dino_feature=True)
                img_feature = einops.rearrange(img_feature, "(b t) d -> b t d",b=img_feature.shape[0])
                img_feature = self.img_feature_projector(img_feature)
                img_features.append(img_feature)
        img_features = torch.stack(img_features, dim=2)

        if self.use_proprio:
            proprio = data[self.proprio_key].float()
            proprio = self.proprio_projector(proprio)

            if len(proprio.shape) == 2:
                proprio = proprio.unsqueeze(1).unsqueeze(1)
            elif len(proprio.shape) == 3:
                proprio = proprio.unsqueeze(1)
            img_features = torch.cat([img_features, proprio], dim=2)

        features = einops.rearrange(img_features, "b t v d -> b (t v) d")

        if self.use_language:
            lang_features = einops.rearrange(lang_features, "(b t) d -> b t d", b=features.shape[0])
            prompt_features = lang_features[:, -1:]

            features = torch.cat([prompt_features, features], dim=1)
        action = data["actions"].float()

        features_dict = self.backbone(features)
        action_features = features_dict['act_features']
        if len(action_features.shape) == 4:
            action_features = einops.rearrange(action_features, "b his_len num_act_tokens n_embd -> b (his_len num_act_tokens) n_embd")

        if hasattr(self.action_head, 'n_samples_per_condition'):
            action_loss = self.action_head(action_cond=action_features,
                                        actions=action,training_step=train_step)
        else:
            action_loss = self.action_head(action_cond=action_features,
                                        actions=action)
        action_loss = action_loss.mean()
        action_loss = action_loss * self.action_loss_coef

        metrics = {"actor_loss": action_loss}
        return metrics

    def act(self, obs, lang_emb, norm_stats, step, temporal_agg):

        if norm_stats is not None:

            pre_process = lambda s_qpos: (
                s_qpos - norm_stats["min"]
            ) / (
                norm_stats["max"]
                - norm_stats["min"]
                + 1e-5
            )
            post_process = (
                lambda a: a
                * (norm_stats["max"] - norm_stats["min"])
                + norm_stats["min"]
            )

        if self.use_language:
            repeat_len = 1
            lang_features = (
                torch.as_tensor(lang_emb, device=self.device)
                .float()[None].repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None

        test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])
        features = []
        for key in self.pixel_keys:
            obs_temp = test_aug(obs[key].transpose(1, 2, 0)).numpy()
            raw_img = torch.as_tensor(np.array(obs_temp), device=self.device).float()
            if len(raw_img.shape) == 3:
                raw_img = raw_img.unsqueeze(0)

            img_feature = self.img_encoder(raw_img,lang=lang_features)
            img_feature = self.img_feature_projector(img_feature)

            if len(img_feature.shape) == 2:
                img_feature = img_feature.unsqueeze(1)
            features.append(img_feature)

        if self.use_proprio:
            if norm_stats is not None:
                obs[self.proprio_key] = pre_process(obs[self.proprio_key])
            proprio = torch.as_tensor(
                np.array(obs[self.proprio_key]), device=self.device
            ).float()
            proprio = self.proprio_projector(proprio)
            while len(proprio.shape) < 3:
                proprio = proprio.unsqueeze(0)
            features.append(proprio)
        features = torch.cat(features, dim=1)

        if self.use_language:
            prompt_features = lang_features[-1:].view(1, 1, self.hidden_dim)
            features = torch.cat([prompt_features, features], dim=1)

        features_dict = self.backbone(features)
        action_features = features_dict['act_features']

        if len(action_features.shape) == 4:
            action_features = action_features.squeeze(1)

        action = self.action_head(action_features)
        action = action.squeeze(0)

        if temporal_agg:
            action = action.view(-1, self.num_queries, self.action_dim)
            self.all_time_actions[[step], step : step + self.num_queries] = action
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0])
            return action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process(action.cpu().numpy()[0, -1])
            return action.cpu().numpy()[0]

    def eval_act(self,obs,qpos,lang_emb,future_actions):
        if self.use_language:
            repeat_len = 1
            lang_features = (
                torch.as_tensor(lang_emb, device=self.device)
                .float()[None].repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None

        features = []
        for key in self.pixel_keys:
            obs_temp = obs[key].to(self.device)
            img_feature = self.img_encoder(obs_temp,lang=lang_features)
            img_feature = self.img_feature_projector(img_feature)

            if len(img_feature.shape) == 2:
                img_feature = img_feature.unsqueeze(1)
            features.append(img_feature)
        if self.use_proprio:
            proprio = qpos.to(self.device).float()
            proprio = self.proprio_projector(proprio)
            while len(proprio.shape) < 3:
                proprio = proprio.unsqueeze(1)
            features.append(proprio)
        features = torch.cat(features, dim=1)

        if self.use_language:
            prompt_features = lang_features[-1:].view(1, 1, self.hidden_dim)
            num_frames = features.shape[0]
            prompt_features = prompt_features.repeat(num_frames, 1, 1)
            features = torch.cat([prompt_features, features], dim=1)

        features_dict = self.backbone(features)
        action_features = features_dict['act_features']
        if len(action_features.shape) == 4:
            action_features = action_features.squeeze(1)

        action = self.action_head(action_features)
        future_actions = future_actions.to(self.device)
        gt_action = future_actions
        loss = (action - gt_action).pow(2).sum()
        return loss.item()

    def act_no_temporal_agg(self, obs, lang_emb, norm_stats, step, temporal_agg):

        if norm_stats is not None:

            pre_process = lambda s_qpos: (
                s_qpos - norm_stats["min"]
            ) / (
                norm_stats["max"]
                - norm_stats["min"]
                + 1e-5
            )
            post_process = (
                lambda a: a
                * (norm_stats["max"] - norm_stats["min"])
                + norm_stats["min"]
            )

        if self.use_language:
            repeat_len = 1
            lang_features = (
                torch.as_tensor(lang_emb, device=self.device)
                .float()[None].repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None
        features = []
        for key in self.pixel_keys:
            obs_temp = obs[key].numpy()
            raw_img = torch.as_tensor(np.array(obs_temp), device=self.device).float()

            img_feature = self.img_encoder(raw_img,lang=lang_features)
            img_feature = self.img_feature_projector(img_feature)
            features.append(img_feature)

        if self.use_proprio:
            obs[self.proprio_key] = pre_process(obs[self.proprio_key])
            proprio = torch.as_tensor(
                np.array(obs[self.proprio_key]), device=self.device
            ).float()
            proprio = self.proprio_projector(proprio)
            if proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)
            features.append(proprio)
        features = torch.cat(features, dim=-1).view(-1, self.hidden_dim)

        if self.use_language:
            prompt_features = lang_features[-1:].view(-1, self.hidden_dim)
            features = torch.cat([prompt_features, features], dim=0)

        features = features.unsqueeze(0)
        features_dict = self.backbone(features)
        action_features = features_dict['act_features']

        if len(action_features.shape) == 4:
            action_features = einops.rearrange(action_features, "b his_len num_act_tokens n_embd -> b (his_len num_act_tokens) n_embd")
        action = self.action_head(action_features)
        action = action.squeeze(0)

        assert norm_stats is not None, "norm_stats must not be None"
        return post_process(action.cpu().numpy())

    def save_basemodel_KVcache(self, obs, lang_emb, norm_stats, episode_id, task_name, base_save_dir):

        if norm_stats is not None:

            pre_process = lambda s_qpos: (
                s_qpos - torch.from_numpy(norm_stats["min"]).cuda()
            ) / (
                torch.from_numpy(norm_stats["max"]).cuda()
                - torch.from_numpy(norm_stats["min"]).cuda()
                + 1e-5
            )

        B = obs[self.pixel_keys[0]].shape[0]
        assert self.use_language
        lang_emb = torch.as_tensor(lang_emb, device=self.device)
        lang_emb = lang_emb.unsqueeze(0).unsqueeze(0)
        lang_features = lang_emb.repeat(B, self.history_len, 1)
        lang_features = self.language_projector(lang_features)
        lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")

        features = []
        for key in self.pixel_keys:

            raw_img = obs[key].to(self.device)

            img_feature = self.img_encoder(raw_img,lang=lang_features)
            img_feature = self.img_feature_projector(img_feature)
            features.append(img_feature)

        if self.use_proprio:
            if norm_stats is not None:
                obs[self.proprio_key] = pre_process(obs[self.proprio_key])
            else:
                obs[self.proprio_key] = obs[self.proprio_key]

            proprio = torch.as_tensor(
                obs[self.proprio_key], device=self.device
            ).float()
            proprio = self.proprio_projector(proprio)
            if proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)
            features.append(proprio)
        features = torch.cat(features, dim=-1).view(B, -1, self.hidden_dim)

        if self.use_language:
            prompt_features = lang_features.view(B, -1, self.hidden_dim)
            features = torch.cat([prompt_features, features], dim=-2)

        with torch.no_grad():
            action_features,Kcache,Vcache = self.backbone.forward_to_save_KVcache(features)

        import h5py
        Kcache = Kcache.detach().cpu().to(torch.float32)
        Vcache = Vcache.detach().cpu().to(torch.float32)
        action_features = action_features.detach().cpu().to(torch.float32)

        save_path = os.path.join(base_save_dir, task_name, f"cache_{episode_id}.hdf5")
        os.makedirs(os.path.join(base_save_dir, task_name), exist_ok=True)
        with h5py.File(save_path, "w") as f:
            f.create_dataset("Kcache", data=Kcache.numpy(), compression="gzip")
            f.create_dataset("Vcache", data=Vcache.numpy(), compression="gzip")
            f.create_dataset("action_features", data=action_features.numpy(), compression="gzip")
        print(f"KV cache saved to {save_path}")

    def save_snapshot(self):
        model_keys = ["img_encoder", "backbone", "action_head"]

        if True:
            model_keys += ["img_feature_projector"]

        if self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]

        payload = {
            k: self.__dict__['_modules'][k].state_dict() for k in model_keys
        }

        return payload

    def load_snapshot(self, pretrained_weight_dir=None):
        if pretrained_weight_dir is not None:
            bc_snapshot = Path(pretrained_weight_dir)
            if not bc_snapshot.exists():
                raise FileNotFoundError(f"Specified pretrained weights do not exist: {pretrained_weight_dir}")
            else:
                print('#'*30)
                print(f"Pretrained weights: {pretrained_weight_dir}")
                print('#'*30)

            with bc_snapshot.open("rb") as f:
                payload = torch.load(f, weights_only=False, map_location="cpu")

            model_keys = ["img_encoder", "img_feature_projector","backbone", "action_head"]
            if self.use_proprio:
                model_keys += ["proprio_projector"]
            if self.use_language:
                model_keys += ["language_projector"]

            for k in model_keys:
                self.__dict__['_modules'][k].load_state_dict(payload[k])
