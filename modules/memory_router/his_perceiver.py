import numpy as np
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention

import math
import torch
import torch.nn as nn

def PositionalEncoding(d_model,max_len, refered_tensor):

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    B = refered_tensor.shape[0]
    pe = pe.repeat(B, 1, 1)

    needed_len = refered_tensor.shape[1]
    pe = pe[:, :needed_len, :]
    return pe

def make_mask_matrix_fast(N: int, n: int):

    points = torch.round(torch.arange(1, n + 1) * N / n).to(torch.int)

    for i in range(n):
        if points[i] == 0:
            points[i] = 1

    col_idx = torch.arange(N).unsqueeze(0)

    mask = (col_idx < points.unsqueeze(1)).to(torch.int)

    return mask.float()

class TransformerBlock(nn.Module):
    def __init__(self,
        hidden_dim=512,
        cond_dim=512,
        num_heads=8,
        dropout=0.,
        cross_attn_only=False,
        q_length=1,
        kv_length=1,
    ):
        super().__init__()

        self.cross_attn_only = cross_attn_only
        assert not cross_attn_only or cond_dim is not None, 'If only do cross attention, cond_dim must NOT be None!'

        if not cross_attn_only:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.attn1 = Attention(
                hidden_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
            )

        self.attn2 = None
        if cond_dim is not None:
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.attn2 = Attention(
                hidden_dim,
                cross_attention_dim=cond_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
            )

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x, cond=None, mask=None, cond_mask=None):

        if not self.cross_attn_only:
            norm_x = self.norm1(x)
            x = self.attn1(norm_x, attention_mask=mask) + x

        if self.attn2 is not None:
            norm_x = self.norm2(x)
            x = self.attn2(norm_x, cond, attention_mask=cond_mask) + x

        norm_x = self.norm3(x)
        x = self.ff(norm_x) + x

        return x

    def act_analysis(self, x, cond=None, mask=None, cond_mask=None):

        if not self.cross_attn_only:
            norm_x = self.norm1(x)
            y = self.attn1(norm_x, attention_mask=mask)
            attn_map = self.attn1.get_attention_scores(norm_x, norm_x, attention_mask=mask)
            x = y + x

        if self.attn2 is not None:
            norm_x = self.norm2(x)
            y = self.attn2(norm_x, cond, attention_mask=cond_mask)
            attn_map = self.attn2.get_attention_scores(norm_x, cond, attention_mask=cond_mask)
            x = y + x

        norm_x = self.norm3(x)
        x = self.ff(norm_x) + x

        return x, attn_map

class HisPerceiver(nn.Module):
    def __init__(self,
                hidden_dim=512,
                q_type = 'xxx',
                global_token_length=1,
                num_heads=8,
                layers=1,
                dropout=0.0,
                use_avg_patch_token=False,
            ):
        super().__init__()
        self.q_type = q_type
        self.global_token_length = global_token_length
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.x_attn = TransformerBlock(hidden_dim, cond_dim=hidden_dim,num_heads=num_heads, dropout=dropout, cross_attn_only=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(layers)
        ])
        self.use_avg_patch_token = use_avg_patch_token
        if self.q_type == 'global_token':
            self.global_token = nn.Parameter(torch.randn(1, self.global_token_length, hidden_dim))

    def act_analysis(self, query=None, kv=None, max_kv_length=0, max_kv_period=0,optional_input_pad_mask=None):
        assert kv is not None
        assert max_kv_length > 0

        B, kv_input_length, D = kv.shape

        q = query if query is not None else self.global_token
        q_length = q.shape[1]
        q = q.expand(B, q_length, D)

        kv = self.norm(kv)
        if kv_input_length <= max_kv_length:
            kv = F.pad(kv, (0, 0, 0, max_kv_length - kv_input_length))
        else:
            raise ValueError(f"Input KV length {kv_input_length} exceeds max length {max_kv_length}")

        if optional_input_pad_mask is None:
            idx = torch.arange(max_kv_length).unsqueeze(0).unsqueeze(0)
            pad_mask = (idx < kv_input_length).expand(B, q_length, max_kv_length).to(device=kv.device)

        else:
            pad_mask = optional_input_pad_mask.unsqueeze(1)
            pad_mask = pad_mask.repeat(1,q_length,1)

        kv_length = max_kv_length

        mask = make_mask_matrix_fast(max_kv_period, q_length)
        single = max_kv_length // max_kv_period
        mask = mask.repeat_interleave(single, dim=1)
        cond_mask = mask.unsqueeze(0).expand(B, q_length, kv_length).to(device=kv.device)

        integrated_mask = cond_mask * pad_mask

        integrated_mask = torch.where(integrated_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf')))

        positional_encoding = PositionalEncoding(self.hidden_dim, max_kv_period, kv).to(device=kv.device)
        positional_encoding = positional_encoding.repeat_interleave(single, dim=1)
        kv = kv + positional_encoding

        latents, memory_attn_map = self.x_attn.act_analysis(q, kv, cond_mask=integrated_mask)

        mask = torch.ones(q_length, q_length, dtype=torch.bool).tril().unsqueeze(0).expand(B, q_length, q_length).to(device=kv.device)
        for block in self.blocks:
            latents = block(latents, mask=mask)

        return latents, memory_attn_map, integrated_mask

    def forward(self, query=None, kv=None, max_kv_length=0, max_kv_period=0,optional_input_pad_mask=None):
        assert kv is not None
        assert max_kv_length > 0
        B, kv_input_length, D = kv.shape

        q = query if query is not None else self.global_token
        q_length = q.shape[1]
        q = q.expand(B, q_length, D)

        kv = self.norm(kv)
        if self.use_avg_patch_token:
            if optional_input_pad_mask is not None:
                kv_masked = kv * optional_input_pad_mask.float().unsqueeze(-1)
                sum_valid = kv_masked.sum(dim=1)
                count_valid = optional_input_pad_mask.float().unsqueeze(-1).sum(dim=1)
                mean_kv = sum_valid / count_valid
                mean_kv = mean_kv.unsqueeze(1)
            else:
                mean_kv = kv.mean(dim=1)
                mean_kv = mean_kv.unsqueeze(1)

        if kv_input_length <= max_kv_length:
            kv = F.pad(kv, (0, 0, 0, max_kv_length - kv_input_length))
        else:
            raise ValueError(f"Input KV length {kv_input_length} exceeds max length {max_kv_length}")

        if optional_input_pad_mask is None:
            idx = torch.arange(max_kv_length).unsqueeze(0).unsqueeze(0)
            pad_mask = (idx < kv_input_length).expand(B, q_length, max_kv_length).to(device=kv.device)

        else:
            pad_mask = optional_input_pad_mask.unsqueeze(1)
            pad_mask = pad_mask.repeat(1,q_length,1)

        kv_length = max_kv_length

        mask = make_mask_matrix_fast(max_kv_period, q_length)
        single = max_kv_length // max_kv_period
        mask = mask.repeat_interleave(single, dim=1)
        cond_mask = mask.unsqueeze(0).expand(B, q_length, kv_length).to(device=kv.device)

        integrated_mask = cond_mask * pad_mask

        integrated_mask = torch.where(integrated_mask == 1, torch.tensor(0.0), torch.tensor(float('-inf')))

        positional_encoding = PositionalEncoding(self.hidden_dim, max_kv_period, kv).to(device=kv.device)
        positional_encoding = positional_encoding.repeat_interleave(single, dim=1)
        kv = kv + positional_encoding

        latents = self.x_attn(q, kv, cond_mask=integrated_mask)

        mask = torch.ones(q_length, q_length, dtype=torch.bool).tril().unsqueeze(0).expand(B, q_length, q_length).to(device=kv.device)
        for block in self.blocks:
            latents = block(latents, mask=mask)

        if self.use_avg_patch_token:
            latents = torch.cat([mean_kv, latents], dim=1)

        return latents
