import sys
import os
sys.append(os.getpwd())
from repos.cleandiffuser.nn_diffusion import BaseNNDiffusion
from repos.cleandiffuser.utils import UntrainablePositionalEmbedding, set_seed
from typing import Callable, Optional, Union
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiTBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_cross_attn: bool = False,
        adaLN_on_cross_attn: bool = False,
    ):
        super().__init__()
        self._adaLN_on_cross_attn = adaLN_on_cross_attn
        self._use_cross_attn = use_cross_attn

        self.sa_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(hidden_size, n_heads, attn_dropout, batch_first=True)

        if use_cross_attn:
            self.ca_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=not adaLN_on_cross_attn, eps=1e-6
            )
            self.ca_attn = nn.MultiheadAttention(
                hidden_size, n_heads, attn_dropout, batch_first=True
            )
        else:
            self.ca_norm, self.ca_attn = None, None

        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(ffn_dropout),
        )

        n_coeff = 9 if adaLN_on_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * n_coeff)
        )

    def forward(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: Optional[torch.Tensor] = None,
        seq_condition_mask: Optional[torch.Tensor] = None,
        vaild_mask = None,
    ):
        adaLN_coeff = self.adaLN_modulation(vec_condition.unsqueeze(-2))
        if self._adaLN_on_cross_attn:
            (
                shift_sa,
                scale_sa,
                gate_sa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = adaLN_coeff.chunk(9, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = adaLN_coeff.chunk(
                6, dim=-1
            )

        h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.sa_attn(h, h, h)[0]

        if self._use_cross_attn:
            if self._adaLN_on_cross_attn:
                h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
            else:
                h = self.ca_norm(x)
                gate_ca = 1.0

            if vaild_mask is not None:
                seq_condition_mask = (1.0-vaild_mask.float()).bool()

            x = (
                x
                + gate_ca
                * self.ca_attn(
                    h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
                )[0]
            )

        h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.mlp(h)
        return x

    def act_analysis(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: Optional[torch.Tensor] = None,
        seq_condition_mask: Optional[torch.Tensor] = None,
    ):
        adaLN_coeff = self.adaLN_modulation(vec_condition.unsqueeze(-2))
        if self._adaLN_on_cross_attn:
            (
                shift_sa,
                scale_sa,
                gate_sa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = adaLN_coeff.chunk(9, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = adaLN_coeff.chunk(
                6, dim=-1
            )

        h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.sa_attn(h, h, h)[0]

        if self._use_cross_attn:
            if self._adaLN_on_cross_attn:
                h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
            else:
                h = self.ca_norm(x)
                gate_ca = 1.0

            y, attn_map = self.ca_attn(h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask)
            x = x + gate_ca* y

        h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.mlp(h)
        return x, attn_map

class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int, head_type: str = "linear"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, out_dim),
            )
            nn.init.constant_(self.head[-1].weight, 0)
            nn.init.constant_(self.head[-1].bias, 0)
        else:
            self.head = nn.Linear(hidden_size, out_dim)
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = self.modulate(self.norm(x), shift, scale)
        return self.head(x)

class DiT1dWithACICrossAttention(BaseNNDiffusion):

    def __init__(
        self,
        stage: str,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int ,
        n_heads: int ,
        depth: int ,
        num_new_layers: int,
        post_type: str,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        head_type: str = "mlp",
        use_trainable_pos_emb: bool = True,
        use_cross_attn: bool = True,
        adaLN_on_cross_attn: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.stage = stage
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        if use_cross_attn:
            self.seq_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        else:
            self.seq_cond_proj = None

        self.num_new_layers = num_new_layers
        self.post_type = post_type

        pos_emb = UntrainablePositionalEmbedding(d_model)(torch.arange(x_seq_len))[None]
        self.pos_emb = nn.Parameter(pos_emb, requires_grad=use_trainable_pos_emb)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model, n_heads, attn_dropout, ffn_dropout, use_cross_attn, adaLN_on_cross_attn
                )
                for _ in range(depth)
            ]
        )
        if self.stage == 'post':
            self.addition_blocks = nn.ModuleList(
                [
                    DiTBlock(d_model, n_heads, attn_dropout, ffn_dropout, use_cross_attn, adaLN_on_cross_attn)
                    for _ in range(num_new_layers)
                ]
            )
        self.final_layer = FinalLayer1d(d_model, x_dim, head_type)
        self.initialize_weights()

        self.seq_cond_proj = None
        self.vis_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.lang_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))

    def initialize_weights(self):

        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.stage == 'post':
            for block in self.addition_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action_cond: torch.Tensor,
        addition_cond: torch.Tensor = None,
        vaild_mask = None,
    ):

        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb
        cond_emb = t_emb

        if action_cond is not None and self.vis_cond_proj is not None:
            action_cond = self.vis_cond_proj(action_cond)
        if addition_cond is not None and self.lang_cond_proj is not None:
            addition_cond = self.lang_cond_proj(addition_cond)

        if self.stage == 'post' and self.post_type == 'first':
            for i, block in enumerate(self.addition_blocks):
                seq_condition = addition_cond
                seq_condition_mask = None
                x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask, vaild_mask = vaild_mask)

        for i, block in enumerate(self.blocks):
            seq_condition = action_cond

            seq_condition_mask = None
            x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask)

        if self.stage == 'post' and self.post_type == 'last':
            for i, block in enumerate(self.addition_blocks):
                seq_condition = addition_cond
                seq_condition_mask = None
                x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask, vaild_mask = vaild_mask)

        x_emb = self.final_layer(x_emb, cond_emb)

        return x_emb

    def cosine_similarity(self, a, b):

        a_flat = a.flatten(start_dim=1)
        b_flat = b.flatten(start_dim=1)
        a_norm = F.normalize(a_flat, p=2, dim=1)
        b_norm = F.normalize(b_flat, p=2, dim=1)
        cos = (a_norm * b_norm).sum(dim=1)
        return cos.item()

    def magnitude_ratio(self,a, b):

        a_flat = a.flatten(start_dim=1)
        b_flat = b.flatten(start_dim=1)
        norm_a = a_flat.norm(p=2, dim=1)
        norm_b = b_flat.norm(p=2, dim=1)
        ratio = norm_b / (norm_a + 1e-8)
        return ratio.item()

    def act_analysis(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action_cond: torch.Tensor,
        addition_cond: torch.Tensor = None,
     ):
        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb
        cond_emb = t_emb

        if action_cond is not None and self.vis_cond_proj is not None:
            action_cond = self.vis_cond_proj(action_cond)
        if addition_cond is not None and self.lang_cond_proj is not None:
            addition_cond = self.lang_cond_proj(addition_cond)

        cos_similas = []
        norm_ratios = []
        head_hiscond_attn_maps = {}

        if self.stage == 'post' and self.post_type == 'first':
            for i, block in enumerate(self.addition_blocks):
                seq_condition = addition_cond
                seq_condition_mask = None
                x_emb_new, head_hiscond_attn_map = block.act_analysis(x_emb, cond_emb, seq_condition, seq_condition_mask)
                norm_ratio = self.magnitude_ratio(x_emb, x_emb_new)
                cos_simila = self.cosine_similarity(x_emb, x_emb_new)
                norm_ratios.append(norm_ratio)
                cos_similas.append(cos_simila)
                head_hiscond_attn_maps[i] = head_hiscond_attn_map
                x_emb = x_emb_new

        for i, block in enumerate(self.blocks):
            seq_condition = action_cond

            seq_condition_mask = None
            x_emb_new = block(x_emb, cond_emb, seq_condition, seq_condition_mask)
            norm_ratio = self.magnitude_ratio(x_emb, x_emb_new)
            cos_simila = self.cosine_similarity(x_emb, x_emb_new)
            norm_ratios.append(norm_ratio)
            cos_similas.append(cos_simila)
            x_emb = x_emb_new

        if self.stage == 'post' and self.post_type == 'last':
            for i, block in enumerate(self.addition_blocks):
                seq_condition = addition_cond
                seq_condition_mask = None
                x_emb_new, head_hiscond_attn_map = block.act_analysis(x_emb, cond_emb, seq_condition, seq_condition_mask)
                norm_ratio = self.magnitude_ratio(x_emb, x_emb_new)
                cos_simila = self.cosine_similarity(x_emb, x_emb_new)
                norm_ratios.append(norm_ratio)
                cos_similas.append(cos_simila)
                head_hiscond_attn_maps[i] = head_hiscond_attn_map
                x_emb = x_emb_new

        x_emb = self.final_layer(x_emb, cond_emb)

        return x_emb, norm_ratios, cos_similas, head_hiscond_attn_maps

from repos.cleandiffuser.diffusion.basic import DiffusionModel
from repos.cleandiffuser.utils import (
    TensorDict,
    at_least_ndim,
    concat_zeros,
    dict_apply,
    get_sampling_scheduler,
)

class ContinuousRectifiedFlow(DiffusionModel):

    def __init__(
        self,

        nn_diffusion: BaseNNDiffusion,
        nn_condition = None,

        fix_mask: Optional[torch.Tensor] = None,

        loss_weight: Optional[torch.Tensor] = None,

        classifier = None,

        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,

        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
     ):
        super().__init__(
            nn_diffusion,
            nn_condition,
            fix_mask,
            loss_weight,
            classifier,
            ema_rate,
            optimizer_params,
        )

        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
     ):

        t = torch.rand((x0.shape[0],), device=self.device) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask

        return xt, t, eps

    def loss(
        self,
        x0: torch.Tensor,
        action_cond : torch.Tensor,
        addition_cond : torch.Tensor = None,
        x1: torch.Tensor = None,
        vaild_mask = None,
    ):

        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        x0 = x0.to(device=self.device)

        x1 = x1.to(device=self.device)

        xt, t, _ = self.add_noise(x0, eps=x1)

        loss = (self.model["diffusion"](xt, t, action_cond, addition_cond, vaild_mask = vaild_mask) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask))

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    def sample(
        self,

        prior: torch.Tensor,
        action_cond: torch.Tensor,
        addition_cond: torch.Tensor = None,
        x1: Optional[torch.Tensor] = None,

        solver: str = "euler",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,

        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
        w_cg: float = 0.0,

        diffusion_x_sampling_steps: int = 0,

        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,

        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):

        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, (
            "Rectified Flow does not support classifier-guidance."
        )

        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(
            sample_steps, device=self.device, **sampling_schedule_params
        )

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.float32, device=self.device)

            delta_t = t_schedule[i] - t_schedule[i - 1]

            with torch.set_grad_enabled(requires_grad):

                if w_cfg == 1.0:
                    assert condition_cfg is None
                    vel = model["diffusion"](xt, t, action_cond, addition_cond)

                elif w_cfg == 0.0:
                    vel = model["diffusion"](xt, t, None, None)

                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)

                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ..."), t.repeat(2), condition
                    )

                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

            xt = xt + delta_t * vel

            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule

        return xt, log

    def act_analysis(
        self,

        prior: torch.Tensor,
        action_cond: torch.Tensor,
        addition_cond: torch.Tensor = None,
        x1: Optional[torch.Tensor] = None,

        solver: str = "euler",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,

        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
        w_cg: float = 0.0,

        diffusion_x_sampling_steps: int = 0,

        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,

        requires_grad: bool = False,
        preserve_history: bool = False,
        **kwargs,
    ):
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, (
            "Rectified Flow does not support classifier-guidance."
        )

        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(
            sample_steps, device=self.device, **sampling_schedule_params
        )

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        relative_norm_group = {}
        relative_cos_group = {}
        head_hiscond_attn_maps_group = {}
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.float32, device=self.device)

            delta_t = t_schedule[i] - t_schedule[i - 1]

            with torch.set_grad_enabled(requires_grad):

                assert w_cfg == 1.0
                assert condition_cfg is None
                vel, relative_norm, relative_cos, head_hiscond_attn_maps = model["diffusion"].act_analysis(xt, t, action_cond, addition_cond)
                relative_norm_group[i] = relative_norm
                relative_cos_group[i] = relative_cos
                head_hiscond_attn_maps_group[i] = head_hiscond_attn_maps

            xt = xt + delta_t * vel

            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule

        return xt, relative_norm_group, relative_cos_group, head_hiscond_attn_maps_group

class ActionHead(nn.Module):
    def __init__(self,

                stage,
                action_dim,
                hidden_dim,

                num_past_queries,
                num_future_queries,

                num_heads,
                num_layers,

                num_new_layers,
                post_type,

                d_model,
                timestep_emb_type,
                timestep_emb_params,

                sampling_steps,

                device='cuda',
                ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_past_queries = num_past_queries
        self.num_future_queries = num_future_queries
        self.num_queries = num_past_queries + num_future_queries

        self.sampling_steps = sampling_steps
        self.device = device

        nn_diffusion = DiT1dWithACICrossAttention(
            stage=stage,
            x_dim=action_dim,
            x_seq_len=self.num_queries,
            emb_dim=hidden_dim,
            d_model=d_model,
            n_heads=num_heads,
            depth=num_layers,
            timestep_emb_type=timestep_emb_type,
            timestep_emb_params=timestep_emb_params,
            num_new_layers=num_new_layers,
            post_type=post_type,

        ).to(self.device)

        self.policy = ContinuousRectifiedFlow(
            nn_diffusion=nn_diffusion,
            nn_condition=None,
            x_max=torch.full((self.num_queries, self.action_dim), 1.0, device=self.device),
            x_min=torch.full((self.num_queries, self.action_dim), -1.0, device=self.device),
        ).to(self.device)

    def forward(self, action_cond, addition_cond=None, actions=None,init_noise=None, vaild_mask=None):

        if actions is None:
            B = action_cond.shape[0]
            act, log = self.policy.sample(
                prior=torch.zeros((B, self.num_queries, self.action_dim), device=self.device),
                action_cond=action_cond,
                addition_cond=addition_cond,
                x1=init_noise,
                solver="euler",
                sample_steps=self.sampling_steps,

                use_ema=False,
                w_cfg=1.0,
            )
            return act

        return self.policy.loss(
            x0=actions,
            action_cond=action_cond,
            addition_cond=addition_cond,
            x1=init_noise,
            vaild_mask=vaild_mask,
        )

    def act_analysis(self, action_cond, addition_cond=None, actions=None,init_noise=None):
        assert actions is None
        act, relative_norm, relative_cos, head_hiscond_attn_maps = self.policy.act_analysis(
                prior=torch.zeros((1, self.num_queries, self.action_dim), device=self.device),
                action_cond=action_cond,
                addition_cond=addition_cond,
                x1=init_noise,
                solver="euler",
                sample_steps=self.sampling_steps,

                use_ema=False,
                w_cfg=1.0,
            )
        return act,relative_norm, relative_cos, head_hiscond_attn_maps
