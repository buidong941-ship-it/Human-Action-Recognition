import random

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class VideoTransform:
    def __init__(self, mode="train", img_size=224):
        self.is_train = (mode=='train')
        self.image_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        if self.is_train:
            # random resized crop
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h*scale), int(w*scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)

            # random crop
            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=transforms.InterpolationMode.BILINEAR)

            # horizontal flip (left <--> right)
            if random.random() < 0.5:
                frames = TF.hflip(frames)

        else: #val
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=transforms.InterpolationMode.BILINEAR)

        normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized)
