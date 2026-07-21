from typing import Optional
import sys
import os
import sys
sys.path.append(os.getcwd())
import einops
import torch
import torch.nn as nn
import os
import utils.agent_utils as utils
from repos.cleandiffuser.utils import UntrainablePositionalEmbedding
from transformers import AutoModel
from utils.log_print import log_print_params as log_print

class RepresentationModel:
    def __init__(
        self,
        model_path: str = "facebook/dinov2-with-registers-base",
        device: str = "cpu",
    ):
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            self.model = self.model.to(x.device)
            self.device = x.device

        with torch.no_grad():
            return self.model(x).last_hidden_state

class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        nheads: int = 6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, key_padding_mask=mask)[0]
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        src_dim: int = 768,
        src_len: int = 261,
        tgt_dim: int = 7,
        tgt_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
        num_layers: int = 2,
        add_tanh: bool = True,
    ):
        super().__init__()
        self.tgt_len = tgt_len
        self.adapter = nn.Linear(src_dim, hidden_size)

        self.tgt_emb = nn.Parameter(torch.randn((1, 1, hidden_size)) * 0.02)

        pos_indices = torch.arange(tgt_len + src_len).unsqueeze(0)
        pos_emb = UntrainablePositionalEmbedding(hidden_size, max_positions=1000)(pos_indices)
        self.pos_emb = nn.Parameter(pos_emb * 0.2)
        self.num_layers = num_layers
        self.transformer = nn.ModuleList([TransformerLayer(hidden_size, nheads) for _ in range(num_layers)])
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, tgt_dim),
            nn.Tanh() if add_tanh else nn.Identity(),
        )

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = src.size(0)

        if mask is not None:
            tgt_mask = torch.zeros((b, self.tgt_len), device=src.device, dtype=torch.bool)
            mask = torch.cat([tgt_mask, mask], dim=1)

        tgt_emb = self.tgt_emb.expand(b, self.tgt_len, -1)
        src_emb = self.adapter(src)
        if len(src_emb.shape) == 2:
            src_emb = src_emb.unsqueeze(1)

        x = torch.cat([tgt_emb, src_emb], dim=1)
        x = x + self.pos_emb
        for layer in self.transformer:
            x = layer(x, mask)
        return self.out_layer(x[:, : self.tgt_len])

class Decoder(nn.Module):
    def __init__(
        self,
        src_dim: int = 768,
        src_len: int = 261,
        tgt_dim: int = 7,
        tgt_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
        num_layers: int = 2,
    ):
        super().__init__()
        self.tgt_len = tgt_len
        self.src_len = src_len
        self.adapter = nn.Linear(tgt_dim, hidden_size)

        self.src_emb = nn.Parameter(torch.randn((1, 1, hidden_size)) * 0.02)

        pos_indices = torch.arange(tgt_len + src_len).unsqueeze(0)
        pos_emb = UntrainablePositionalEmbedding(hidden_size, max_positions=1000)(pos_indices)
        self.pos_emb = nn.Parameter(pos_emb * 0.2)
        self.num_layers = num_layers
        self.transformer = nn.ModuleList([TransformerLayer(hidden_size, nheads) for _ in range(num_layers)])

        self.out_layer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, src_dim))

    def forward(self, tgt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = tgt.size(0)

        if mask is not None:
            tgt_mask = torch.zeros((b, self.tgt_len), device=tgt.device, dtype=torch.bool)
            mask = torch.cat([tgt_mask, mask], dim=1)

        src_emb = self.src_emb.expand(b, self.src_len, -1)
        tgt_emb = self.adapter(tgt)
        x = torch.cat([tgt_emb, src_emb], dim=1)
        x = x + self.pos_emb
        for layer in self.transformer:
            x = layer(x, mask)
        return self.out_layer(x[:, self.tgt_len :])

class AutoEncoder(nn.Module):

    def __init__(
        self,
        img_len: int = 374 * 3,
        lang_len: int = 1,
        z_dim: int = 14,
        z_img_len: int = 24,
        z_lang_len: int = 3,
        hidden_size: int = 384,
        nheads: int = 6,
        num_layers: int = 2,
        dino_type: str = 'vitb',
    ):
        super().__init__()
        if dino_type == 'vitb':
            img_dim = 768
        elif dino_type == 'vits':
            img_dim = 384

        lang_dim = 384
        self.lang_len = int(lang_len)
        self.img_len = int(img_len)
        self.tgt_img_len = int(z_img_len)
        self.tgt_lang_len = int(z_lang_len)

        self.img_encoder = Encoder(img_dim, img_len, z_dim, z_img_len, hidden_size, nheads, num_layers)
        self.img_decoder = Decoder(img_dim, img_len, z_dim, z_img_len, hidden_size, nheads, num_layers)
        self.lang_encoder = Encoder(lang_dim, lang_len, z_dim, z_lang_len, hidden_size, nheads, num_layers)
        self.lang_decoder = Decoder(lang_dim, lang_len, z_dim, z_lang_len, hidden_size, nheads, num_layers)

    def encode(self,img_emb,lang_emb):

        img_tgt = self.img_encoder(img_emb)
        lang_tgt = self.lang_encoder(lang_emb)
        return torch.cat([img_tgt, lang_tgt], dim=1)

    def decode(self, z):
        img_emb = self.img_decoder(z[:, : self.tgt_img_len])
        lang_emb = self.lang_decoder(z[:, -self.tgt_lang_len :])
        return img_emb, lang_emb

    def loss(self,img_emb,lang_emb):

        z = self.encode(img_emb, lang_emb)
        recon_img_emb, recon_lang_emb = self.decode(z)

        img_loss = nn.functional.cosine_similarity(recon_img_emb, img_emb, dim=-1)
        lang_loss = nn.functional.cosine_similarity(recon_lang_emb, lang_emb, dim=-1)
        lang_loss = -lang_loss.mean()
        img_loss = -img_loss.mean()

        img_loss_weight, lang_loss_weight = 1.0, 0.1

        return img_loss, lang_loss, img_loss_weight, lang_loss_weight

class AutoEncoder_Assumed(nn.Module):
    def __init__(self,
                model_type,
                pixel_keys,
                z_dim = 7,
                img_len = 375,
                lang_len = 1,
                z_img_len = 24,
                z_lang_len = 3,
                hidden_size = 384,
                nheads = 6,
                num_layers = 2,
                dino_type = 'vitb',
                device = "cuda",
                lr = 1e-4,
                 ):
        super().__init__()
        self.model_type = model_type
        self.pixel_keys = pixel_keys
        self.z_dim = z_dim
        self.img_len = img_len
        self.lang_len = lang_len
        self.z_img_len = z_img_len
        self.z_lang_len = z_lang_len
        self.num_layers = num_layers

        self.dino_type = dino_type
        if self.dino_type == 'vitb':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif self.dino_type == 'vits':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.dinov2.parameters():
            param.requires_grad = False

        if self.model_type == "autoencoder":
            self.autoencoder = AutoEncoder(
                img_len=self.img_len,
                lang_len=self.lang_len,
                z_dim=self.z_dim,
                z_img_len=self.z_img_len,
                z_lang_len=self.z_lang_len,
                num_layers=self.num_layers,
                dino_type=dino_type
            )
        else:
            raise ValueError(f"model_type: {self.model_type} not supported")
        self.device = device
        self.to(device)

    def encode(self,img_emb,lang_emb):
        return self.autoencoder.encode(img_emb, lang_emb)

    def decode(self,z):
        return self.autoencoder.decode(z)

    def crop_image(self, image):
        assert len(image.shape) == 4
        H, W = image.shape[-2:]
        new_H = H - H%14
        new_W = W - W%14
        det_H = H - new_H
        det_W = W - new_W
        high = det_H//2
        low = det_H - high
        left = det_W//2
        right = det_W - left

        if low == 0:
            low = -10000
        if right == 0:
            right = -10000

        return image[:, :, high:-low, left:-right]

    def update(self, expert_replay_iter):
        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        pixels = []
        for pixel_key in self.pixel_keys:
            pixels.append(data[pixel_key])
        pixels = torch.stack(pixels, dim=1)
        B = pixels.shape[0]

        pixels = einops.rearrange(pixels, "b v c h w -> (b v) c h w")
        pixels = self.crop_image(pixels)

        feature_assume = self.dinov2(pixels)
        cls_feature = feature_assume['x_norm_clstoken'].unsqueeze(1)
        patch_feature = feature_assume['x_norm_patchtokens']

        pixels = torch.cat((cls_feature, patch_feature), dim=1)
        pixels = einops.rearrange(pixels, "(b v) l d -> b (v l) d", b=B)

        lang_emb = data["task_emb"].unsqueeze(1)
        img_loss, lang_loss, img_loss_weight, lang_loss_weight = self.autoencoder.loss(pixels, lang_emb)

        metrics = {"img_loss": img_loss * img_loss_weight ,"lang_loss": lang_loss * lang_loss_weight}
        return metrics

    def save_cache(self, obs, prompt, episode_id, task_name, base_save_dir):
        pixels = []
        for pixel_key in self.pixel_keys:
            pixels.append(obs[pixel_key])
        pixels = torch.stack(pixels, dim=1)
        pixels = pixels.to(self.device)
        prompt = torch.from_numpy(prompt).to(self.device)
        num_frames = pixels.shape[0]

        pixels = einops.rearrange(pixels, "b v c h w -> (b v) c h w")

        pixels = self.crop_image(pixels)
        feature_assume = self.dinov2(pixels)

        cls_feature = feature_assume['x_norm_clstoken'].unsqueeze(1)
        patch_feature = feature_assume['x_norm_patchtokens']

        pixels = torch.cat((cls_feature, patch_feature), dim=1)

        pixels = einops.rearrange(pixels, "(b v) l d -> b (v l) d", b=num_frames)
        prompt = prompt.view(1,1,-1)
        prompt = prompt.repeat(num_frames, 1, 1)

        z = self.autoencoder.encode(pixels, prompt)

        import h5py
        z = z.detach().cpu().to(torch.float32)

        save_path = os.path.join(base_save_dir, task_name, f"cache_{episode_id}.hdf5")
        os.makedirs(os.path.join(base_save_dir, task_name), exist_ok=True)
        with h5py.File(save_path, "w") as f:
            f.create_dataset("z", data=z.numpy(), compression="gzip")

    def save_snapshot(self):
        model_keys = ["autoencoder", "dinov2"]
        payload = {
            k: self.__dict__['_modules'][k].state_dict() for k in model_keys
        }
        return payload

    def load_snapshot(self, payload):
        model_keys = ["autoencoder", "dinov2"]
        for k in model_keys:
            self.__dict__['_modules'][k].load_state_dict(payload[k])

    def eval_act(self,obs,task_emb):
        task_emb = task_emb.to(self.device)

        pixels = []
        for pixel_key in self.pixel_keys:
            pixels.append(obs[pixel_key].to(self.device))
        pixels = torch.stack(pixels, dim=1)
        B = pixels.shape[0]

        pixels = einops.rearrange(pixels, "b v c h w -> (b v) c h w")
        pixels = self.crop_image(pixels)

        feature_assume = self.dinov2(pixels)
        cls_feature = feature_assume['x_norm_clstoken'].unsqueeze(1)
        patch_feature = feature_assume['x_norm_patchtokens']

        pixels = torch.cat((cls_feature, patch_feature), dim=1)

        pixels = einops.rearrange(pixels, "(b v) l d -> b (v l) d", b=B)

        num_frames = pixels.shape[0]
        task_emb = task_emb.view(1, 1, self.hidden_dim)
        task_emb = task_emb.repeat(num_frames, 1, 1)
        lang_emb = task_emb.unsqueeze(1)

        img_loss, lang_loss, img_loss_weight, lang_loss_weight = self.autoencoder.loss(pixels, lang_emb)

        loss = img_loss * img_loss_weight + lang_loss * lang_loss_weight
        return loss
