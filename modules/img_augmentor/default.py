import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class Processer(nn.Module):
    def __init__(
        self,
        use_color_jitter=True, probability_color_jitter=0.2, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02,
        use_affine=True, probability_affine=0.2, rotation=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5,
    ):
        super().__init__()

        if use_color_jitter:
            self.color_tf = T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )

        if use_affine:
            self.affine_tf = T.RandomAffine(
                degrees=rotation,
                translate=translate,
                scale=scale,
                shear=shear
            )
        self.use_color_jitter = use_color_jitter
        self.use_affine = use_affine

        self.probability_color_jitter = probability_color_jitter
        self.probability_affine = probability_affine

    def forward(self, img):

        single = len(img.shape) == 3
        if single:
            img = img.unsqueeze(0)

        new_img = []
        for i in range(img.shape[0]):
            img_temp = img[i]
            if self.use_color_jitter and random.random() < self.probability_color_jitter:
                img_temp = self.color_tf(img_temp)
            if self.use_affine and random.random() < self.probability_affine:
                img_temp = self.affine_tf(img_temp)
            new_img.append(img_temp)
        new_img = torch.stack(new_img, dim=0)

        return new_img[0] if single else new_img
