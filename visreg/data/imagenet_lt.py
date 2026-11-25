"""ImageNet-LT (Liu et al., 2019): long-tailed 115.8K-image subset of ImageNet-1k, 5-1280 images per class.

The train split is built once by datasets/build_imagenet_lt_hf_dataset.py from
datasets/ImageNet_LT_train.txt and loaded from disk; the test split is the full
ImageNet-1k validation set (50 images per class).
"""

from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .multicrop import DinoMultiCrop, PILGaussianBlur


class ImageNetLTDataset(Dataset):
    """ImageNet-LT with either V independently augmented views or DINO multi-crop (n_global + n_local)."""

    num_classes = 1000

    def __init__(
        self,
        split: str,
        *,
        data_dir: str = "./data/imagenets/imagenet_lt_hf",
        cache_dir: str | None = "./data/imagenets",
        V: int = 1,
        n_global: int | None = None,
        n_local: int | None = None,
        global_img_size: int = 128,
        local_img_size: int = 64,
    ):
        self.split = split
        self.V = V
        if split == "train":
            train_dir = Path(data_dir) / "train"
            if not train_dir.exists():
                raise FileNotFoundError(
                    f"ImageNet-LT train split not found: {train_dir}. "
                    "Build it with: uv run python datasets/build_imagenet_lt_hf_dataset.py"
                )
            self.ds = load_from_disk(str(train_dir))
        else:
            self.ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=cache_dir, split="validation")
        print(f"Loaded ImageNet-LT {split}: {len(self.ds)} samples")

        self.multi_crop = None
        if split == "train" and n_global is not None and n_local is not None:
            self.multi_crop = DinoMultiCrop(
                n_global=n_global,
                n_local=n_local,
                global_size=global_img_size,
                local_size=local_img_size,
                global_scale=(0.3, 1.0),
                local_scale=(0.05, 0.3),
            )

        self.aug = v2.Compose([
            v2.RandomResizedCrop(global_img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([PILGaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test = v2.Compose([
            v2.Resize(global_img_size),
            v2.CenterCrop(global_img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_class_split(self):
        """Standard ImageNet-LT groups by train count: many (>100), medium (20-100), few (<20)."""
        counts = Counter(self.ds["label"])
        many = [c for c, n in counts.items() if n > 100]
        medium = [c for c, n in counts.items() if 20 <= n <= 100]
        few = [c for c, n in counts.items() if n < 20]
        return many, medium, few

    def __getitem__(self, i):
        item = self.ds[i]
        img, label = item["image"].convert("RGB"), item["label"]
        if self.multi_crop is not None:
            return self.multi_crop(img), label
        transform = self.aug if self.split == "train" and self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), label

    def __len__(self):
        return len(self.ds)
