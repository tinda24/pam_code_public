import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

def new_gelu(x):

    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",

            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.backbone_use_uncausal_mask = config.backbone_use_uncausal_mask
        self.config = config

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if not self.backbone_use_uncausal_mask:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        else:
            bias_copy = self.bias[:, :, :T, :T].clone()

            for i in range(self.config.his_len):
                num_features_every_time = (T-1)//self.config.his_len
                assert num_features_every_time * self.config.his_len == T-1
                m = i*num_features_every_time

                bias_copy[:, :, m+1:m+1+self.config.num_cameras, m+1:m+1+self.config.num_cameras] = 1
            att = att.masked_fill(bias_copy == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward_to_save_KVcache(self, x):

        (
            B,
            T,
            C,
        ) = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if not self.backbone_use_uncausal_mask:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        else:
            bias_copy = self.bias[:, :, :T, :T].clone()

            for i in range(self.config.his_len):
                num_features_every_time = (T-1)//self.config.his_len
                assert num_features_every_time * self.config.his_len == T-1
                m = i*num_features_every_time

                bias_copy[:, :, m+1:m+1+self.config.num_cameras, m+1:m+1+self.config.num_cameras] = 1
            att = att.masked_fill(bias_copy == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        y = self.resid_dropout(self.c_proj(y))
        return y,k,v

    def incremental_forward(self,x,Kcache,Vcache):
        (B,T,C,) = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )

        k = torch.cat([Kcache,k],dim=2)
        v = torch.cat([Vcache,v],dim=2)

        T_seq = Kcache.shape[2] + x.shape[1]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:, :, T_seq-T:T_seq, :T_seq] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        y = self.resid_dropout(self.c_proj(y))
        return y

    def act_analysis(self, x):
        (B,T,C,) = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if not self.backbone_use_uncausal_mask:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        else:
            bias_copy = self.bias[:, :, :T, :T].clone()

            for i in range(self.config.his_len):
                num_features_every_time = (T-1)//self.config.his_len
                assert num_features_every_time * self.config.his_len == T-1
                m = i*num_features_every_time

                bias_copy[:, :, m+1:m+1+self.config.num_cameras, m+1:m+1+self.config.num_cameras] = 1
            att = att.masked_fill(bias_copy == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        with torch.no_grad():
            raw_attn = att.detach().clone()

        att = self.attn_dropout(att)
        y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )

        y = self.resid_dropout(self.c_proj(y))
        return y,raw_attn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_to_save_KVcache(self, x):
        y,Kcache,Vcache = self.attn.forward_to_save_KVcache(self.ln_1(x))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        assert len(Kcache.shape) == len(Vcache.shape) == 4, f"Check the KV features returned by each layer, Kcache.shape: {Kcache.shape}, Vcache.shape: {Vcache.shape}"
        return x,Kcache,Vcache

    def incremental_forward(self, x,Kcache,Vcache):
        x = x + self.attn.incremental_forward(self.ln_1(x),Kcache,Vcache)
        x = x + self.mlp(self.ln_2(x))
        return x

    def act_analysis(self, x):
        y,attn_map = self.attn.act_analysis(self.ln_1(x))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x,attn_map

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    backbone_use_uncausal_mask: bool = False
    num_cameras: int = 2
    his_len: int = 1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None
        self.config = config
        self.backbone_use_uncausal_mask = config.backbone_use_uncausal_mask

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(config.input_dim, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        n_params = sum(p.numel() for p in self.parameters())
        print("number of backbone parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, input, targets=None):
        device = input.device
        b, t, d = input.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )

        tok_emb = self.transformer.wte(
            input
        )
        pos_emb = self.transformer.wpe(
            pos
        )
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward_to_save_KVcache(self, obs):

        device = obs.device
        b, t, d = obs.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )

        tok_emb = self.transformer.wte(
            obs
        )
        pos_emb = self.transformer.wpe(
            pos
        )
        x = self.transformer.drop(tok_emb + pos_emb)

        Kcache = []
        Vcache = []
        for block in self.transformer.h:
            x,K,V = block.forward_to_save_KVcache(x)
            Kcache.append(K)
            Vcache.append(V)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        Kcache = torch.stack(Kcache, dim=1)
        Vcache = torch.stack(Vcache, dim=1)
        return logits,Kcache,Vcache

    def incremental_forward(self, Kcache,Vcache,his_token):

        device = his_token.device
        b, t1, d = his_token.size()
        b, _, _, t2, _ = Kcache.size()
        assert (
            t1+t2 <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t1+t2, dtype=torch.long, device=device).unsqueeze(
            0
        )
        new_token_pos = pos[:,-t1:]

        tok_emb = self.transformer.wte(
            his_token
        )
        pos_emb = self.transformer.wpe(
            new_token_pos
        )
        x = self.transformer.drop(tok_emb + pos_emb)

        for i,block in enumerate(self.transformer.h):
            x = block.incremental_forward(x,Kcache[:,i],Vcache[:,i])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):

                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):

                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):

                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def act_analysis(self, input):
        device = input.device
        b, t, d = input.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )

        tok_emb = self.transformer.wte(
            input
        )
        pos_emb = self.transformer.wpe(
            pos
        )
        x = self.transformer.drop(tok_emb + pos_emb)

        attn_group = {}
        for i,block in enumerate(self.transformer.h):
            x,attn_map = block.act_analysis(x)
            attn_group[i] = attn_map
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits,attn_group

class GPT_Backbone(nn.Module):
    def __init__(self,
        block_size,
        input_dim,
        output_dim,
        n_layer,
        n_head,
        n_embd,
        dropout,
        backbone_use_uncausal_mask,
        num_cameras,
        use_proprio,
        his_len,

        num_act_tokens,
        num_img_tokens,
        num_his_tokens,

        device = 'cuda',
    ):
        super().__init__()
        print('Initializing backbone')
        self.config = GPTConfig(
            block_size=block_size,
            input_dim=input_dim,
            output_dim=output_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            backbone_use_uncausal_mask=backbone_use_uncausal_mask,
            num_cameras=num_cameras,
            his_len=his_len,
        )
        self.net = GPT(self.config).to(device)

        self.num_act_tokens = num_act_tokens
        self.num_img_tokens = num_img_tokens
        self.num_his_tokens = num_his_tokens
        print(f"num_act_tokens: {num_act_tokens}, num_img_tokens: {num_img_tokens}, num_his_tokens: {num_his_tokens}")

        self.his_len = his_len

        self._action_token = nn.Parameter(torch.randn(1, 1, num_act_tokens, input_dim).to(device))
        if num_img_tokens > 0:
            self._img_token = nn.Parameter(torch.randn(1, 1, num_img_tokens, input_dim).to(device))
        else:
            self._img_token = None
        if num_his_tokens > 0:
            self._his_token = nn.Parameter(torch.randn(1, num_his_tokens, input_dim).to(device))
        else:
            self._his_token = None

        self.num_cameras = num_cameras
        self.use_proprio = use_proprio
        self._num_feat_per_step = self.num_cameras + self.use_proprio

    def forward(self, features, targets=None):
        B = features.shape[0]

        prompt = features[:, :1]
        obs = features[:, 1:]
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])

        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        if self.num_img_tokens > 0:
            img_token = self._img_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, img_token], dim=-2)

        obs = torch.cat([obs, action_token], dim=-2)

        if self.num_his_tokens > 0:
            his_token = self._his_token.repeat(B, 1, 1, 1)
            obs = torch.cat([obs, his_token], dim=-2)

        obs = obs.view(B, -1, self.config.input_dim)

        obs = torch.cat([prompt, obs], dim=1)

        features = self.net(obs)

        feature_dict = {}
        features = features[:, 1:]

        action_features = []
        img_features = []

        for i in range(self.his_len):
            num_feat_per_step = self._num_feat_per_step + self.num_act_tokens + self.num_img_tokens
            a_start = i*num_feat_per_step+self._num_feat_per_step+self.num_img_tokens
            img_start = i*num_feat_per_step+self._num_feat_per_step
            action_features.append(features[:, a_start:a_start+self.num_act_tokens, :])
            if self.num_img_tokens > 0:
                img_features.append(features[:, img_start:img_start+self.num_img_tokens, :])

        action_features = torch.stack(action_features, dim=1)
        feature_dict['act_features'] = action_features

        if self.num_img_tokens > 0:
            img_features = torch.stack(img_features, dim=1)
            feature_dict['img_features'] = img_features

        if self.num_his_tokens > 0:
            his_features = features[:, -self.num_his_tokens:, :]
            feature_dict['his_features'] = his_features

        return feature_dict

    def forward_to_save_KVcache(self, features):

        B = features.shape[0]
        prompt = features[:, :1]
        obs = features[:, 1:]

        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        if self.num_img_tokens > 0:
            img_token = self._img_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, img_token], dim=-2)
        obs = torch.cat([obs, action_token], dim=-2)
        if self.num_his_tokens > 0:
            raise NotImplementedError("Base cache export should not include his_token here")
        obs = obs.view(B, -1, self.config.input_dim)
        obs = torch.cat([prompt, obs], dim=1)

        features,Kcache,Vcache = self.net.forward_to_save_KVcache(obs)
        action_features = features[:, -self.num_act_tokens:, :]

        return action_features,Kcache,Vcache

    def incremental_forward(self, Kcache,Vcache):

        assert self._his_token is not None, "Incremental inference requires his_token"

        B = Kcache.shape[0]

        his_token = self._his_token.repeat(B, 1, 1)

        his_features = self.net.incremental_forward(Kcache,Vcache,his_token)

        return his_features

    def act_analysis(self, features):
        B = features.shape[0]
        prompt = features[:, :1]
        obs = features[:, 1:]
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])

        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        if self.num_img_tokens > 0:
            img_token = self._img_token.repeat(B, obs.shape[1], 1, 1)
            obs = torch.cat([obs, img_token], dim=-2)

        obs = torch.cat([obs, action_token], dim=-2)

        if self.num_his_tokens > 0:
            his_token = self._his_token.repeat(B, 1, 1, 1)
            obs = torch.cat([obs, his_token], dim=-2)

        obs = obs.view(B, -1, self.config.input_dim)

        obs = torch.cat([prompt, obs], dim=1)

        features,attn_group = self.net.act_analysis(obs)

        feature_dict = {}
        features = features[:, 1:]

        action_features = []
        img_features = []

        for i in range(self.his_len):
            num_feat_per_step = self._num_feat_per_step + self.num_act_tokens + self.num_img_tokens
            a_start = i*num_feat_per_step+self._num_feat_per_step+self.num_img_tokens
            img_start = i*num_feat_per_step+self._num_feat_per_step
            action_features.append(features[:, a_start:a_start+self.num_act_tokens, :])
            if self.num_img_tokens > 0:
                img_features.append(features[:, img_start:img_start+self.num_img_tokens, :])

        action_features = torch.stack(action_features, dim=1)
        feature_dict['act_features'] = action_features

        if self.num_img_tokens > 0:
            img_features = torch.stack(img_features, dim=1)
            feature_dict['img_features'] = img_features

        if self.num_his_tokens > 0:
            his_features = features[:, -self.num_his_tokens:, :]
            feature_dict['his_features'] = his_features

        return feature_dict,attn_group
