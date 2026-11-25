from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from .multicrop import DinoMultiCrop
from .multicrop_kornia import DinoMultiCropKornia


class ImagenetteDataset(Dataset):
    """Imagenette (10 ImageNet classes) read from the extracted fast.ai archive.

    Download https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz (or run
    datasets/download_datasets.sh) and point `root` at the extracted `imagenette2-160/`,
    which holds `train/<wnid>/*.JPEG` and `val/<wnid>/*.JPEG`.
    """

    def __init__(
        self,
        split,
        *,
        root: str = "./data/imagenette2-160",
        n_global: int | None = None,
        n_local: int | None = None,
        global_img_size: int = 128,
        local_img_size: int = 64,
        multicrop_backend: str = "torchvision",
    ):
        self.split = split
        split_dir = Path(root) / ("train" if split == "train" else "val")
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Imagenette split not found: {split_dir}. Download and extract "
                "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz "
                "(see datasets/download_datasets.sh) and set `dataset_args.root`."
            )
        # Classes are the sorted wnid folders, matching the label order of the 10 Imagenette classes.
        self.ds = ImageFolder(split_dir)

        self.multi_crop = None
        if n_global is not None and split == "train":
            if multicrop_backend == "kornia":
                self.multi_crop = DinoMultiCropKornia(
                    n_global=n_global,
                    n_local=n_local or 0,
                    global_size=global_img_size,
                    local_size=local_img_size,
                    global_scale=(0.08, 1.0),
                    local_scale=(0.05, 0.3),
                )
            else:
                self.multi_crop = DinoMultiCrop(
                    n_global=n_global,
                    n_local=n_local or 0,
                    global_size=global_img_size,
                    local_size=local_img_size,
                    global_scale=(0.08, 1.0),
                    local_scale=(0.05, 0.3),
                    cj_brightness=0.8,
                    cj_contrast=0.8,
                    cj_saturation=0.8,
                    cj_hue=0.2,
                )

        self.test = v2.Compose(
            [
                v2.Resize(global_img_size),
                v2.CenterCrop(global_img_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        img, label = self.ds[i]
        if self.multi_crop is not None:
            return self.multi_crop(img), label
        return self.test(img).unsqueeze(0), label

    def __len__(self):
        return len(self.ds)

    @property
    def num_classes(self):
        return len(self.ds.classes)

