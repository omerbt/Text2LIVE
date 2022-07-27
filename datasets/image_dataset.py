import os.path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path
import torch

from util.aug_utils import RandomScale, RandomSizeCrop, DivisibleCrop


class SingleImageDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.base_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: transforms.ToTensor()(x).unsqueeze(0)),
                DivisibleCrop(cfg["d_divisible_crops"]),
            ]
        )

        # used to create the internal dataset
        self.input_transforms = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                    p=cfg["jitter_p"],
                ),
                transforms.RandomHorizontalFlip(p=cfg["flip_p"]),
                RandomScale((cfg["scale_min"], cfg["scale_max"])),
                RandomSizeCrop(cfg["crops_min_cover"]),
                self.base_transforms,
            ]
        )

        # open source image
        self.src_img = Image.open(cfg["image_path"]).convert("RGB")

        if cfg["resize_input"] > 0:
            self.src_img = transforms.Resize(cfg["resize_input"])(self.src_img)

        self.step = -1

    def get_img(self):
        return self.base_transforms(self.src_img)

    def __getitem__(self, index):
        self.step += 1
        sample = {"step": self.step}
        if self.step % self.cfg["source_image_every"] == 0:
            sample["input_image"] = self.get_img()

        sample["input_crop"] = self.input_transforms(self.src_img)

        return sample

    def __len__(self):
        return 1
