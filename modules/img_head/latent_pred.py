import numpy as np
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention

class TransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim=None, num_heads=8, dropout=0., cross_attn_only=False):
        super().__init__()

        self.cross_attn_only = cross_attn_only
        assert not cross_attn_only or cond_dim is not None, 'If only do cross attention, cond_dim must NOT be None!'

        if not cross_attn_only:
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = Attention(
                dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        self.attn2 = None
        if cond_dim is not None:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                dim,
                cross_attention_dim=cond_dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

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

class ImgHead(nn.Module):
    def __init__(self,
            cond_dim = 512,
            hidden_dim = 512,
            z_length = 18,
            z_dim = 14,
            num_future_queries = 16,
            num_past_queries = 0,
            num_iterations = 2,
            dropout = 0.1
            ):
        super().__init__()

        self.z_length = z_length
        self.z_dim = z_dim
        self.num_future_queries = num_future_queries
        self.num_past_queries = num_past_queries
        self.num_queries = num_future_queries + num_past_queries
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.dropout = dropout

        self.query_latent = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_dim))

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cond_norm = nn.LayerNorm(hidden_dim)

        self.curr_latent_proj = nn.Linear(z_length*z_dim, hidden_dim)

        self.final_proj = nn.Linear(hidden_dim, z_length*z_dim)

        self.curr_latent_attn = TransformerBlock(hidden_dim, cond_dim=hidden_dim, dropout=dropout, cross_attn_only=True)
        self.cond_attn = TransformerBlock(hidden_dim, cond_dim=hidden_dim, dropout=dropout, cross_attn_only=True)
        self.self_attn = TransformerBlock(hidden_dim, dropout=dropout)

        total = sum(p.numel() for p in self.parameters())
        print(f'Image prediction head parameters: {total/1e6:.2f}M, iterations: {num_iterations}, effective parameters: {total*num_iterations/1e6:.2f}M')

    def forward(self, curr_latent, cond):

        curr_latent = torch.flatten(curr_latent, start_dim=-2)
        if len(curr_latent.shape) == 2:
            curr_latent = curr_latent.unsqueeze(1)
        curr_latent = self.curr_latent_proj(curr_latent)

        cond = self.cond_proj(cond)
        cond = self.cond_norm(cond)

        query_latent = self.query_latent.repeat(curr_latent.shape[0], 1, 1)

        for i in range(self.num_iterations):
            query_latent = self.curr_latent_attn(query_latent, curr_latent)
            query_latent = self.cond_attn(query_latent, cond)
            query_latent = self.self_attn(query_latent)

        query_latent = self.final_proj(query_latent)
        query_latent = query_latent.reshape(curr_latent.shape[0], self.num_queries, self.z_length, self.z_dim)
        return query_latent
