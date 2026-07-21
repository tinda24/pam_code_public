import einops
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from pathlib import Path
import utils.agent_utils as utils
from utils.log_print import log_print_params as log_print
from agent.dynamic_actor import DynamicActor
from tqdm import tqdm

class PostDynamicActor(DynamicActor):
    def __init__(
        self,
        stage,
        base_weight_dir,
        lr,
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
        action_loss_coef,
        img_loss_coef,
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
        use_memory_router,
        use_img_head,
        memory_router,
        img_head,
        load_prior_action_head_weights,
        freeze_modal_encoder,
        backbone_frozen_when_posttrain,
        action_query_frozen_when_posttrain,
        sample_interval,
        max_his_cache,
        q_type,
        kv_type,
        use_adaptive_loss_coef = False,
        use_group_loss = False,

        device = 'cuda',
    ):
        obs_mode = 'pixel'
        super(PostDynamicActor, self).__init__(stage, obs_mode, action_dim, proprio_dim, img_size,
                                    num_views, pixel_keys, proprio_key, lang_key, hidden_dim,
                                    num_queries, action_arrange_type, lr,action_loss_coef, lang_encoder,
                                    use_proprio, use_language, history_len, num_act_tokens, num_img_tokens, num_his_tokens,
                                    img_encoder, backbone, action_head, False, device)
        self.use_memory_router = use_memory_router
        self.use_img_head = use_img_head

        self.base_weight_dir = base_weight_dir

        if self.use_memory_router:
            self.memory_router = memory_router.to(device)
        else:
            self.memory_router = None

        if self.use_img_head:
            self.img_head = img_head.to(device)
        else:
            self.img_head = None

        self.load_prior_action_head_weights = load_prior_action_head_weights
        self.freeze_modal_encoder = freeze_modal_encoder
        self.backbone_frozen_when_posttrain = backbone_frozen_when_posttrain
        self.action_query_frozen_when_posttrain = action_query_frozen_when_posttrain

        self.sample_interval = sample_interval
        self.max_his_cache = max_his_cache
        log_print(f"sample_interval: {self.sample_interval}")
        log_print(f"max_his_cache: {self.max_his_cache}")

        self.q_type = q_type
        self.kv_type = kv_type
        self.num_his_tokens = num_his_tokens

        self.device = device
        self.img_loss_coef = img_loss_coef

        self.latents_cache = []

        if self.freeze_modal_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            for param in self.proprio_projector.parameters():
                param.requires_grad = False
            for param in self.language_projector.parameters():
                param.requires_grad = False
            for param in self.img_feature_projector.parameters():
                param.requires_grad = False

        if self.backbone_frozen_when_posttrain:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        if self.action_query_frozen_when_posttrain:
            self.backbone._action_token.requires_grad = False
        else:
            self.backbone._action_token.requires_grad = True

        if self.backbone._his_token is not None:
            self.backbone._his_token.requires_grad = True

    def update_latents_cache(self, features,frame_step):

        if frame_step % self.sample_interval == 0 and len(self.latents_cache) < self.max_his_cache:
            self.latents_cache.insert(0,features)

        elif frame_step % self.sample_interval == 0 and len(self.latents_cache) == self.max_his_cache:
            print(f'step {frame_step}, delete last')
            self.latents_cache.pop()
            self.latents_cache.insert(0,features)

    def fetch_latents_cache(self):
        if len(self.latents_cache) == 0:
            return None
        else:
            return torch.cat(self.latents_cache, dim=1)

    def episode_update(self, expert_replay_iter, train_step, update=True):
        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        frame_is_valid = data["frame_is_valid"]
        B, episode_length = data["actions"].shape[0], data["actions"].shape[1]
        action_loss_total = 0
        img_loss_total = 0

        actions = data["actions"]
        Kcache = data["Kcache"]
        Vcache = data["Vcache"]
        action_features_input = data["action_features"]

        for frame_step in tqdm(range(episode_length)):
            if frame_step == 0:
                self.latents_cache = []
                if self.use_img_head:
                    self.img_head.reset_cache()

            num_actions = self.num_queries
            act_future = torch.zeros((B,self.num_future_queries, self.action_dim))
            act_future[:,
                : min(episode_length, frame_step + self.num_future_queries) - frame_step
            ] = actions[:,frame_step : frame_step + self.num_future_queries]

            last_action = actions[:,-1].unsqueeze(1)
            if frame_step+self.num_future_queries > episode_length:
                act_future[:,episode_length-frame_step : ] = last_action

            action_is_valid_future = torch.ones((self.num_future_queries))

            if self.num_past_queries > 0:
                if frame_step >= self.num_past_queries:
                    action_past = actions[:,frame_step - self.num_past_queries : frame_step]
                    action_is_valid_past = torch.ones((self.num_past_queries))
                else:
                    action_past = torch.zeros((self.num_past_queries,self.action_dim))

                    if frame_step == 0:
                        action_is_valid_past = torch.zeros((self.num_past_queries))
                    else:
                        action_past[-frame_step:] = actions[:,0:frame_step]
                        action_is_valid_past = torch.zeros((self.num_past_queries))
                        action_is_valid_past[-frame_step:] = 1

                action = torch.cat((action_past, act_future), axis=0)
                act_is_valid = torch.cat((action_is_valid_past, action_is_valid_future), axis=0)
            else:
                action = act_future
                act_is_valid = action_is_valid_future

            act_is_valid = act_is_valid.unsqueeze(0).unsqueeze(-1).repeat(B, 1, self.action_dim).to(device=self.device)

            if len(action.shape) == 3:
                action = action.unsqueeze(1)

            if self.action_arrange_type == 'joint_chunk':
                action = einops.rearrange(action, "b t chunk d -> b (t d) chunk")
            elif self.action_arrange_type == 'chunk_joint':
                action = einops.rearrange(action, "b t chunk d -> b (t chunk) d")
            elif self.action_arrange_type == 'chunkjoint':
                action = einops.rearrange(action, "b t chunk d -> b t (chunk d)")
            else:
                raise ValueError(f"Invalid action arrange type: {self.action_arrange_type}")

            his_features = self.backbone.incremental_forward(Kcache[:,frame_step],Vcache[:,frame_step])

            action_features = action_features_input[:,frame_step]

            q_input = None
            q_input = action_features if self.q_type == 'act_token' else q_input
            q_input = his_features if self.q_type == 'his_token' else q_input

            kv_input = self.fetch_latents_cache()
            kv_input = his_features if kv_input is None else kv_input

            if self.kv_type == 'his_token':
                self.update_latents_cache(his_features,frame_step)
                single_kv_period_len = self.num_his_tokens
                assert single_kv_period_len == his_features.shape[1]
            elif self.kv_type == 'act_token':
                self.update_latents_cache(action_features,frame_step)
                single_kv_period_len = action_features.shape[1]

            additional_cond = self.memory_router(query=q_input,kv=kv_input,max_kv_length=single_kv_period_len*self.max_his_cache,max_kv_period=self.max_his_cache)
            action_loss = self.action_head(action_cond=action_features,
                                            addition_cond=additional_cond,
                                            actions=action)
            action_loss = action_loss * act_is_valid
            action_loss = action_loss.mean(dim=(-1,-2))
            action_loss = action_loss * frame_is_valid[:,frame_step]
            action_loss = action_loss.mean()
            action_loss = action_loss * self.action_loss_coef

            img_loss = 0
            if self.use_img_head:
                global_img_now = data['high_camera'][:,frame_step]
                global_img_future = data['high_camera'][:,frame_step + self.num_future_queries]

                img_loss = self.img_head(global_img_now, global_img_future, additional_cond,frame_step)
                assert img_loss.shape == (B,)
                img_loss = img_loss * frame_is_valid[:,frame_step]
                img_loss = img_loss.mean()
                img_loss = img_loss * self.img_loss_coef

            action_loss_total += action_loss
            img_loss_total += img_loss

        action_loss_total = action_loss_total / 500
        img_loss_total = img_loss_total / 500

        metrics = {"actor_loss": action_loss_total, "img_loss": img_loss_total}
        return metrics

    def update(self, expert_replay_iter, train_step, update=True):
        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        actions = data["actions"]
        Kcache = data["Kcache"]
        Vcache = data["Vcache"]
        action_features = data["action_features"]
        action_is_valid = data["action_is_valid"]
        cache_is_valid = data["cache_is_valid"]
        if self.use_img_head:
            if hasattr(self.img_head,'past_img_pred_flag_for_mainagent'):
                past_img_is_pad = cache_is_valid.clone()

        cache_is_valid = torch.repeat_interleave(cache_is_valid, repeats=self.num_his_tokens, dim=1)

        all_his_features = []
        for memory_frame_idx in range(self.max_his_cache):
            his_features = self.backbone.incremental_forward(Kcache[:,memory_frame_idx],Vcache[:,memory_frame_idx])
            all_his_features.append(his_features)
        all_his_features = torch.cat(all_his_features, dim=1)

        q_input = None
        q_input = action_features if self.q_type == 'act_token' else q_input
        q_input = his_features if self.q_type == 'his_token' else q_input

        kv_input = all_his_features
        assert self.kv_type == 'his_token',"The current post stage only supports his_token; extend this if needed"

        if self.use_memory_router:
            additional_cond = self.memory_router(query=q_input,
                                                    kv=kv_input,
                                                    max_kv_length=self.num_his_tokens * self.max_his_cache,
                                                    max_kv_period=self.max_his_cache,
                                                    optional_input_pad_mask=cache_is_valid)
        else:
            kv_input = kv_input
            assert cache_is_valid is not None
            kv_masked = kv_input * cache_is_valid.float().unsqueeze(-1)
            sum_valid = kv_masked.sum(dim=1)
            count_valid = cache_is_valid.float().unsqueeze(-1).sum(dim=1)
            mean_kv = sum_valid / count_valid
            mean_kv = mean_kv.unsqueeze(1)

            additional_cond = mean_kv
        action_loss = self.action_head(action_cond=action_features,
                                            addition_cond=additional_cond,
                                            actions=actions)

        action_loss = action_loss * action_is_valid
        action_loss = action_loss.mean()
        action_loss = action_loss * self.action_loss_coef
        img_loss = 0
        if self.use_img_head:
            if hasattr(self.img_head, 'z_dim') and not hasattr(self.img_head,'past_img_pred_flag_for_mainagent'):
                correspond_img_latents = data["correspond_img_latents"]
                current_img_latents = data["current_img_latents"]
                pred_img_latents = self.img_head(curr_latent=current_img_latents, cond=additional_cond)

                img_is_valid = action_is_valid[:,:,0]
                img_loss = F.mse_loss(pred_img_latents, correspond_img_latents,reduction="none")
                img_loss = img_loss.mean(dim=(-1,-2))
                img_loss = img_loss * img_is_valid
                img_loss = img_loss.mean()
                img_loss = img_loss * self.img_loss_coef
            elif hasattr(self.img_head, 'sampling_steps') and not hasattr(self.img_head,'past_img_pred_flag_for_mainagent'):
                past_actions = data["past_actions"]
                img_loss = self.img_head(actions=past_actions, action_cond=additional_cond)
                img_loss = img_loss.mean()
                img_loss = img_loss * self.img_loss_coef
            elif hasattr(self.img_head,'past_img_pred_flag_for_mainagent'):
                past_img = data['sampled_z_cache']
                img_loss = self.img_head(gt_img=past_img, img_cond=additional_cond, valid_mask = past_img_is_pad)
                img_loss = img_loss.mean()
                img_loss = img_loss * self.img_loss_coef
            else:
                raise ValueError(f"Invalid img head")

        metrics = {"actor_loss": action_loss, "img_loss": img_loss}
        return metrics

    def act(self, obs, lang_emb, norm_stats, step, temporal_agg, action_head_sample_times=1):
        if step == 0:
            self.latents_cache = []
            if self.use_img_head:
                self.img_head.reset_cache()

        if norm_stats is not None:
            pre_process = lambda s_qpos: (s_qpos.cpu() - norm_stats["min"]) / (norm_stats["max"]- norm_stats["min"]+ 1e-5)
            post_process = (lambda a: a* (norm_stats["max"] - norm_stats["min"])+ norm_stats["min"])

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
            raw_img = test_aug(obs[key].transpose(1, 2, 0)).unsqueeze(0).cuda()

            img_feature = self.img_encoder(raw_img,lang=lang_features)
            img_feature = self.img_feature_projector(img_feature)
            features.append(img_feature)

        if self.use_proprio:
            if norm_stats is not None:
                obs[self.proprio_key] = pre_process(obs[self.proprio_key])

            proprio = obs[self.proprio_key].unsqueeze(0).cuda()
            proprio = proprio.float()
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
        his_features = features_dict['his_features']

        if len(action_features.shape) == 4:
            action_features = einops.rearrange(action_features, "b his_len num_act_tokens n_embd -> b (his_len num_act_tokens) n_embd")

        q_input = None
        q_input = action_features if self.q_type == 'act_token' else q_input
        q_input = his_features if self.q_type == 'his_token' else q_input

        kv_input = self.fetch_latents_cache()
        kv_input = his_features if kv_input is None else kv_input

        if self.kv_type == 'his_token':
            self.update_latents_cache(his_features,step)
            single_kv_period_len = self.num_his_tokens
            assert single_kv_period_len == his_features.shape[1]
        elif self.kv_type == 'act_token':
            self.update_latents_cache(action_features,step)
            single_kv_period_len = action_features.shape[1]

        if self.use_memory_router:
            additional_cond = self.memory_router(query=q_input,kv=kv_input,max_kv_length=single_kv_period_len*self.max_his_cache,max_kv_period=self.max_his_cache)
        else:
            mean_kv = kv_input.mean(dim=1)
            mean_kv = mean_kv.unsqueeze(1)
            additional_cond = mean_kv

        if action_head_sample_times == 1:
            pred_action = self.action_head(action_cond=action_features,
                                            addition_cond=additional_cond,)

            action = pred_action[:,self.num_past_queries:].squeeze(0)
            action = action[1:20]

            self.num_future_queries = action.shape[0]

            if temporal_agg:
                action = action.view(-1, self.num_future_queries, self.action_dim)
                self.all_time_actions[[step], step : step + self.num_future_queries] = action
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
                else:
                    return action.cpu().numpy()[0]
            else:
                if norm_stats is not None:
                    return post_process(action.cpu().numpy())
                return action.cpu().numpy()

        elif action_head_sample_times > 1:
            all_actions = []
            for i in range(action_head_sample_times):
                pred_action = self.action_head(action_cond=action_features,
                                                addition_cond=additional_cond,)

                action = pred_action[:,self.num_past_queries:].squeeze(0)

                if norm_stats is not None:
                    action = post_process(action)
                all_actions.append(action)

            all_actions = torch.stack(all_actions, dim=0)

            return all_actions.cpu().numpy()

    def save_snapshot(self):
        model_keys = ["img_encoder", "backbone", "action_head"]

        if True:
            model_keys += ["img_feature_projector"]

        if self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]
        if self.use_memory_router:
            model_keys += ["memory_router"]
        if self.use_img_head:
            model_keys += ["img_head"]

        payload = {
            k: self.__dict__['_modules'][k].state_dict() for k in model_keys
        }

        return payload

    def load_snapshot(self, pretrained_weight_dir=None):

        if pretrained_weight_dir is None:
            self.load_base_model()
        else:
            bc_snapshot = Path(pretrained_weight_dir)
            if not bc_snapshot.exists():
                raise FileNotFoundError(f"Specified pretrained weights do not exist: {pretrained_weight_dir}")
            else:
                print('#'*30)
                log_print(f"Pretrained weights: {pretrained_weight_dir}")
                print('#'*30)

            with bc_snapshot.open("rb") as f:
                payload = torch.load(f, weights_only=False, map_location="cpu")

            model_keys = ["img_encoder", "img_feature_projector","backbone", "action_head"]
            if self.use_proprio:
                model_keys += ["proprio_projector"]
            if self.use_language:
                model_keys += ["language_projector"]
            if self.use_memory_router:
                model_keys += ["memory_router"]
            if self.use_img_head:
                model_keys += ["img_head"]
            for k in model_keys:
                self.__dict__['_modules'][k].load_state_dict(payload[k])

    def load_base_model(self):
        bc_snapshot = Path(self.base_weight_dir)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"Specified base weights do not exist: {bc_snapshot}")
        else:
            print('#'*30)
            log_print(f"Base weights: {bc_snapshot}")
            print('#'*30)

        with bc_snapshot.open("rb") as f:
            payload = torch.load(f, weights_only=False, map_location="cpu")

        model_keys = ["img_encoder", "backbone"]
        if self.load_prior_action_head_weights:
            model_keys += ["action_head"]
            log_print(f"Loaded base-stage action head weights for post training")
        else:
            log_print(f"Did not load base-stage action head weights for post training")

        if True:
            model_keys += ["img_feature_projector"]
        if self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]
        for k in model_keys:
            self.__dict__['_modules'][k].load_state_dict(payload[k],strict=False)

    def act_analysis(self, obs, task_emb, priprio, step):
        if step == 0:
            self.latents_cache = []

        lang_emb = torch.from_numpy(task_emb).float().unsqueeze(0).cuda()
        lang_emb = self.language_projector(lang_emb)
        lang_emb = lang_emb.view(1,1,self.hidden_dim)

        priprio = self.proprio_projector(priprio.cuda())
        priprio = priprio.view(1,self.hidden_dim)

        obs_features = []
        obs_features_raw = {}
        for key in self.pixel_keys:
            raw_img = obs[key].unsqueeze(0).cuda()
            img_feature, img_feature_raw = self.img_encoder.act_analysis(raw_img,lang=lang_emb.view(1,self.hidden_dim))
            img_feature = self.img_feature_projector(img_feature)
            obs_features.append(img_feature)
            obs_features_raw[key] = img_feature_raw

        if self.use_proprio:
            obs_features.append(priprio)

        features = torch.cat(obs_features, dim=0)
        features = features.unsqueeze(0)

        if self.use_language:
            features = torch.cat([lang_emb, features], dim=1)

        with torch.no_grad():
            raw_input_features = features.detach().clone()

        features_dict, attn_group = self.backbone.act_analysis(features)

        action_features = features_dict['act_features']
        his_features = features_dict['his_features']

        with torch.no_grad():
            raw_action_features = action_features.detach().clone()
            raw_his_features = his_features.detach().clone()

        if len(action_features.shape) == 4:
            action_features = einops.rearrange(action_features, "b his_len num_act_tokens n_embd -> b (his_len num_act_tokens) n_embd")

        q_input = None
        q_input = action_features if self.q_type == 'act_token' else q_input
        q_input = his_features if self.q_type == 'his_token' else q_input

        kv_input = self.fetch_latents_cache()
        kv_input = his_features if kv_input is None else kv_input
        with torch.no_grad():
            raw_memory_router_input = kv_input.detach().clone()

        if self.kv_type == 'his_token':
            self.update_latents_cache(his_features,step)
            single_kv_period_len = self.num_his_tokens
            assert single_kv_period_len == his_features.shape[1]
        elif self.kv_type == 'act_token':
            self.update_latents_cache(action_features,step)
            single_kv_period_len = action_features.shape[1]

        additional_cond, memory_attn_map, integrated_mask = self.memory_router.act_analysis(
                                                query=q_input,
                                                kv=kv_input,
                                                max_kv_length=single_kv_period_len*self.max_his_cache,
                                                max_kv_period=self.max_his_cache)

        pred_action, _, _, _ = self.action_head.act_analysis(action_cond=action_features,
                                        addition_cond=additional_cond,)

        action = pred_action[:,self.num_past_queries:].squeeze(0)

        return {
            'img_head_patch_features': obs_features_raw,
            'backbone_input': raw_input_features,
            'action_features': raw_action_features,
            'his_features': raw_his_features,
            'backbone_attn_maps': attn_group,
            'memory_router_input': raw_memory_router_input,
            'memory_router_output': additional_cond,
            'memory_router_attn_maps': memory_attn_map,
            'memory_router_integrated_mask': integrated_mask,
            'action_head_pred_action': action,
        }

    def eval_act(self, Kcache, Vcache, action_features, future_actions_group):
        Kcache = torch.from_numpy(Kcache).to(self.device)
        Vcache = torch.from_numpy(Vcache).to(self.device)
        action_features = torch.from_numpy(action_features).to(self.device)

        sampled_Kcache = Kcache[0 :: self.sample_interval]
        sampled_Vcache = Vcache[0 :: self.sample_interval]

        num_frames = future_actions_group.shape[0]

        assert Kcache.shape[0] == num_frames
        assert (num_frames-1) // self.sample_interval + 1 == len(sampled_Kcache)

        all_his_features = self.backbone.incremental_forward(sampled_Kcache,sampled_Vcache)

        q_input = None

        kv_input = []
        cache_is_valid = []
        for i in range(num_frames):
            valid_his_frames = i // self.sample_interval + 1
            valid_his_cond = all_his_features[:valid_his_frames]
            valid_his_cond = valid_his_cond[-self.max_his_cache:]

            if len(valid_his_cond) == self.max_his_cache:
                cache_is_valid_this_step = torch.ones(self.max_his_cache)
                kv_input_this_step = valid_his_cond
            elif len(valid_his_cond) < self.max_his_cache:
                cache_is_valid_this_step = torch.cat([torch.ones(len(valid_his_cond)),torch.zeros(self.max_his_cache-len(valid_his_cond))])
                pad_kv_input = torch.zeros(self.max_his_cache-len(valid_his_cond),self.num_his_tokens,self.hidden_dim)
                pad_kv_input = pad_kv_input.to(self.device)
                kv_input_this_step = torch.cat([valid_his_cond,pad_kv_input], dim=0)
            else:
                raise ValueError(f"Invalid valid_his_cond length: {len(valid_his_cond)}")

            kv_input.append(kv_input_this_step)
            cache_is_valid.append(cache_is_valid_this_step.to(self.device))

        kv_input = torch.stack(kv_input, dim=0)
        kv_input = einops.rearrange(kv_input, "b cache_len num_his_tokens n_embd -> b (cache_len num_his_tokens) n_embd")
        assert kv_input.shape[1] == self.num_his_tokens*self.max_his_cache

        cache_is_valid = torch.stack(cache_is_valid, dim=0)
        cache_is_valid = torch.repeat_interleave(cache_is_valid, repeats=self.num_his_tokens, dim=1)

        if self.use_memory_router:
            additional_cond = self.memory_router(query=q_input,
                                                    kv=kv_input,
                                                    max_kv_length=self.num_his_tokens*self.max_his_cache,
                                                    max_kv_period=self.max_his_cache,
                                                    optional_input_pad_mask=cache_is_valid)
        else:
            kv_input = kv_input
            assert cache_is_valid is not None
            kv_masked = kv_input * cache_is_valid.float().unsqueeze(-1)
            sum_valid = kv_masked.sum(dim=1)
            count_valid = cache_is_valid.float().unsqueeze(-1).sum(dim=1)
            mean_kv = sum_valid / count_valid
            mean_kv = mean_kv.unsqueeze(1)

            additional_cond = mean_kv

        pred_action = self.action_head(action_cond=action_features,
                                        addition_cond=additional_cond,)

        action = pred_action[:,self.num_past_queries:]

        future_actions = future_actions_group.to(self.device)
        gt_action = future_actions

        loss = (action - gt_action).pow(2).sum()
        return loss.item()
