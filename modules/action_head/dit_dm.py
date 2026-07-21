import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class CrossAttention(nn.Module):

    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_dit_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_dit_heads == 0, 'dim should be divisible by num_dit_heads'
        self.num_dit_heads = num_dit_heads
        self.head_dim = dim // num_dit_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_dit_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_dit_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x

class _Dit_Attn_Decoder_singleblock(nn.Module):

    def __init__(self, hidden_size, num_dit_heads, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_dit_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,**block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_dit_heads=num_dit_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,**block_kwargs)

        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c=None, mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        if c is None:
            x = self.cross_attn(x, x, mask)
        else:
            x1 = self.cross_attn(x, c, mask)
            x = x1
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x

        return x

class DitHead(nn.Module):
    def __init__(self,
                 stage,
                 action_dim,
                 num_past_queries,
                 num_future_queries,
                 decoder_hidden_dim,

                 num_dit_heads,
                 num_dit_layers,
                 num_new_layers,
                 train_timesteps = 100,
                 val_timesteps = 10,

                 time_embed_dim = 256,
                 learnable_w = False,
                 freeze_prior_layer = False,
                 device = 'cuda',
                 ):
        super(DitHead, self).__init__()

        self.stage = stage
        self.device = device
        self.action_dim = action_dim

        self.num_past_queries = num_past_queries
        self.num_future_queries = num_future_queries
        self.num_queries = num_past_queries + num_future_queries

        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_dit_heads = num_dit_heads
        self.num_dit_layers = num_dit_layers
        self.num_new_layers = num_new_layers
        self.train_timesteps = train_timesteps
        self.val_timesteps = val_timesteps
        self.freeze_prior_layer = freeze_prior_layer
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        self.time_net = _TimeNetwork(time_embed_dim, self.decoder_hidden_dim, learnable_w)

        self.register_parameter(
            "dec_pos",
            nn.Parameter(
                torch.empty(1,self.num_queries,self.decoder_hidden_dim),
                requires_grad=True
            )
        )

        nn.init.xavier_uniform_(self.dec_pos.data)

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, decoder_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        ).to(device)

        self.action_decoder = nn.Sequential(
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(decoder_hidden_dim, action_dim)
        ).to(device)

        decoder_cross_module = _Dit_Attn_Decoder_singleblock(
            hidden_size=self.decoder_hidden_dim,
            num_dit_heads=self.num_dit_heads,
        ).to(device)

        self.decoder_blocks = nn.ModuleList([
            copy.deepcopy(decoder_cross_module) for _ in range(num_dit_layers)
        ]).to(device)

        if self.stage == 'post':
            self.new_decoder_blocks = nn.ModuleList([
                copy.deepcopy(decoder_cross_module) for _ in range(self.num_new_layers)
            ]).to(device)

    def denoise_net(self,noise_action_token,timesteps,encode_latent,his_latent=None):

        noise_action_token = self.action_encoder(noise_action_token)
        noise_action_token = noise_action_token + self.dec_pos

        time_enc = self.time_net(timesteps).to(noise_action_token.device)
        time_enc = time_enc.unsqueeze(1)

        x = torch.cat((time_enc, noise_action_token), dim=1)

        if self.stage == 'post':
            assert his_latent is not None
            for i in range(self.num_new_layers):
                x=self.new_decoder_blocks[i](x,his_latent)

        for i in range(self.num_dit_layers):
            x=self.decoder_blocks[i](x,encode_latent)

        action_noise_pred = self.action_decoder(x[:, 1:])
        return action_noise_pred

    def forward(self, action_cond, addition_cond=None, actions=None,init_noise=None):
        if actions is None:
            return self.get_prediction(action_cond = action_cond,
                                        addition_cond = addition_cond,
                                        init_noise = init_noise)

        B = action_cond.shape[0]

        timesteps = torch.randint(
            low=0,
            high=self.train_timesteps,
            size=(B,),
        ).long()

        if not init_noise:
            action_noise = torch.randn(actions.shape,device = action_cond.device)
        else:
            action_noise = init_noise

        noise_action_token = self.diffusion_schedule.add_noise(
            actions, action_noise, timesteps
        ).to(dtype=torch.float32)

        action_noise_pred = self.denoise_net(noise_action_token,timesteps,action_cond,addition_cond)

        actor_loss = F.mse_loss(action_noise_pred, action_noise,reduction="none")

        return actor_loss

    def get_prediction(self,action_cond,addition_cond=None,init_noise=None):
        B = action_cond.shape[0]

        if init_noise is None:
            action_sample = torch.randn(
                    B, self.num_queries, self.action_dim,
                    device=action_cond.device
                )
        else:
            action_sample = init_noise

        self.diffusion_schedule.set_timesteps(self.val_timesteps)

        for timestep in self.diffusion_schedule.timesteps:
            batched_timestep = timestep.unsqueeze(0).repeat(B)
            pred_action_noise = self.denoise_net(
                    action_sample,
                    batched_timestep,
                    action_cond,
                    addition_cond
                )

            action_sample = self.diffusion_schedule.step(
                model_output=pred_action_noise,
                timestep=timestep,
                sample=action_sample
            ).prev_sample

        return action_sample

class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, tim_out_dim, learnable_w=False):
        super(_TimeNetwork, self).__init__()
        assert time_dim % 2 == 0, "time_dim must be even"
        half_dim = int(time_dim // 2)
        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.out_net = nn.Sequential(
            nn.Linear(time_dim, tim_out_dim),
            nn.SiLU(),
            nn.Linear(tim_out_dim, tim_out_dim)
        )

    def forward(self, x):

        x = x[:, None]
        w = self.w.to(x.device)
        x = x * w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)

        self.out_net = self.out_net.to(x.device)

        return self.out_net(x)
