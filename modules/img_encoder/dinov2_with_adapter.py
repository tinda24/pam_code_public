import torch
import torch.nn as nn
import os,sys
sys.path.append(os.getcwd())
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
import einops
from torchvision import transforms as T
import torchvision.transforms as transforms

class Dinov2ObservationEncoder(nn.Module):
    def __init__(self,
                dinov2_type,
                frozen,
                norm,
                device,
                separate_encoders,
                pixel_keys,
                output_dim,
                img_size,
                ):
        super().__init__()
        self.device = device

        if not separate_encoders:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        else:
            self.dinov2 = {key: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device) for key in pixel_keys}

        self.dinov2_type = dinov2_type
        self.norm = norm
        self.separate_encoders = separate_encoders
        self.pixel_keys = pixel_keys
        self.frozen = frozen
        self.output_dim = output_dim
        self.img_size = img_size
        if self.norm:
            MEAN = torch.tensor([0.485, 0.456, 0.406])
            STD = torch.tensor([0.229, 0.224, 0.225])
            self.normalize = T.Normalize(mean=MEAN, std=STD)

        if frozen:
            if not separate_encoders:
                for param in self.dinov2.parameters():
                    param.requires_grad = False
            else:
                for key,value in self.dinov2.items():
                    for param in value.parameters():
                        param.requires_grad = False

        if self.img_size == [480,640]:
            self.spatial_adapter = nn.Sequential(
                nn.Conv2d(768, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, stride = 2, padding=1),
                nn.Flatten(1),
            )
            self.adapter_2 = nn.Sequential(
                nn.Linear(64*12*9, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.10)
            )
        elif self.img_size == [128,128]:
            if self.dinov2_type=='vitb':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(768, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(128*5*5, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
            elif self.dinov2_type=='vits':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(384, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(128*5*5, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
            else:
                raise ValueError(f"dinov2_type: {self.dinov2_type} not supported")
        elif self.img_size == [240,320]:
            if self.dinov2_type=='vitb':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(768, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(128*9*11, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
            elif self.dinov2_type=='vits':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(384, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(128*9*11, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
        elif self.img_size == [336,336]:
            if self.dinov2_type == 'vitb':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(768, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, stride = 2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(64*6*6, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
            elif self.dinov2_type == 'vits':
                self.spatial_adapter = nn.Sequential(
                    nn.Conv2d(384, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, stride = 2, padding=1),
                    nn.Flatten(1),
                )
                self.adapter_2 = nn.Sequential(
                    nn.Linear(64*12*12, self.output_dim),
                    nn.LayerNorm(self.output_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10)
                )
        else:
            raise ValueError(f"img_size: {self.img_size} not supported")
        print(f'Trainable DINO parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, pixel, pixel_key=None,lang=None,use_cached_dino_feature=False):

        if not  use_cached_dino_feature:
            BT,H,W = pixel.shape[0], pixel.shape[-2], pixel.shape[-1]
            if H == 128 and W == 128:
                pixel = pixel[...,1:-1,1:-1]
            elif H == 480 and W == 640:
                pixel = pixel[...,2:-2,5:-5]
            elif H == 240 and W == 320:
                pixel = pixel[...,1:-1,6:-6]

            if self.norm:
                pixel = self.normalize(pixel)

            feature_assume = self.dinov2(pixel) if not self.separate_encoders else self.dinov2[pixel_key](pixel)

            patch_feature = feature_assume['x_norm_patchtokens']

            B = patch_feature.shape[0]
            H_patch = H // 14
            W_patch = W // 14
            patch_feature = patch_feature.permute(0, 2, 1).view(B, -1, H_patch, W_patch)
        else:
            patch_feature = pixel

        patch_feature = self.spatial_adapter(patch_feature)
        patch_feature = self.adapter_2(patch_feature)

        return patch_feature

    def act_analysis(self, pixel, pixel_key=None,lang=None):
        BT,H,W = pixel.shape[0], pixel.shape[-2], pixel.shape[-1]
        if H == 128 and W == 128:
            pixel = pixel[...,1:-1,1:-1]
        elif H == 480 and W == 640:
            pixel = pixel[...,2:-2,5:-5]
        elif H == 240 and W == 320:
            pixel = pixel[...,1:-1,6:-6]

        if self.norm:
            pixel = self.normalize(pixel)

        feature_assume = self.dinov2(pixel) if not self.separate_encoders else self.dinov2[pixel_key](pixel)

        patch_feature = feature_assume['x_norm_patchtokens']

        with torch.no_grad():
            patch_feature_raw = patch_feature.detach().clone()

        B = patch_feature.shape[0]
        H_patch = H // 14
        W_patch = W // 14
        patch_feature = patch_feature.permute(0, 2, 1).view(B, -1, H_patch, W_patch)

        patch_feature = self.spatial_adapter(patch_feature)
        patch_feature = self.adapter_2(patch_feature)

        return patch_feature , patch_feature_raw

if __name__ == "__main__":
    encoder = Dinov2ObservationEncoder(
        dinov2_type="vits",
        frozen=True,
        norm=True,
        device="cuda",
        separate_encoders=False,
        pixel_keys=None,
        output_dim=2048,
        img_size=[336,336],
    ).to("cuda")
    import time
    x = torch.randn(256,3,336,336)
    x = x.to("cuda")

    print(f'Parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)/1e6:.2f}M')
    import time
    import numpy as np
    from tqdm import tqdm
    time_taken = []
    for i in tqdm(range(100)):
        start_time = time.time()
        loss = encoder(x)
        end_time = time.time()
        time_taken.append(end_time - start_time)
    time_taken = time_taken[2:]
    time_taken = np.array(time_taken)
    print(f'100x inference time: {100*np.mean(time_taken)}s')
    print(f'max inference time: {np.max(time_taken)}s')
    print(f'min inference time: {np.min(time_taken)}s')
    print(f'std inference time: {np.std(time_taken)}s')
    print(f'median inference time: {np.median(time_taken)}s')
    print(f'90% inference time: {np.percentile(time_taken, 90)}s')
    print(f'95% inference time: {np.percentile(time_taken, 95)}s')
    print(f'99% inference time: {np.percentile(time_taken, 99)}s')
