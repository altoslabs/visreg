# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file of https://github.com/facebookresearch/dinov2.
#
# Adapted from dinov2/data/datasets/image_net_22k.py: tarball/mmap indexing kept, decoding moved
# to torchvision.io and DINO multi-crop applied in the dataset.

"""ImageNet-22k (fall11, 21,841 classes) read directly from tar files.

Layout expected under `root`:
    <name>.tar            either one tarball per class (<wnid>.tar, as distributed by image-net.org)
                          or the training shards of timm/imagenet-22k-wds (used for the released
                          checkpoint); every image member of every tar is a sample
    blocks/<name>.log     tar block index per tar             (tools/prepare_imagenet22k.py)
and under `extra`:
    entries.npy, class-ids.npy                                (tools/prepare_imagenet22k.py)

The target is the index of the tar a sample came from: the class for per-class tarballs, the shard
for webdataset shards. Pretraining does not use it (there is no validation split; train with
`run_validation: false`). Images are read with mmap at the offsets in entries.npy, no extraction.
"""

import io
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from gzip import GzipFile
from mmap import ACCESS_READ, mmap

import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .multicrop import DinoMultiCrop
from .multicrop_kornia import DinoMultiCropKornia

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"})
_SKIP_FILES = frozenset({"n06470073_47249.JPEG"})  # known-corrupt file (skipped as in DINOv2)
_MAX_DECODE_RETRIES = 10


def _load_entries_for_one_class(root: str, class_index: int, class_id: str):
    """Parse root/blocks/<class_id>.log into [(class_index, start_offset, end_offset, filename)]."""
    path = os.path.join(root, "blocks", f"{class_id}.log")
    blocks = []
    with open(path) as f:
        for line in f:
            block, filename = line.rstrip().split(":", 1)
            filename = filename[1:] if filename.startswith(" ") else filename
            blocks.append((int(block[6:]), None if filename == "** Block of NULs **" else filename))
    entries = []
    for (b1, name), (b2, _) in zip(blocks, blocks[1:]):
        if name is None or name in _SKIP_FILES:
            continue
        if os.path.splitext(name)[1].lower() not in _IMAGE_EXTENSIONS:
            continue
        entries.append((class_index, 512 * b1, 512 * b2, name))
    return entries


def dump_imagenet22k_indices(root: str, extra: str, workers: int = 0) -> None:
    """Build extra/entries.npy and extra/class-ids.npy from root/*.tar + root/blocks/*.log."""
    class_ids = sorted(os.path.splitext(e.name)[0] for e in os.scandir(root) if e.name.endswith(".tar"))
    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            per_class = list(ex.map(_load_entries_for_one_class, [root] * len(class_ids),
                                    range(len(class_ids)), class_ids, chunksize=64))
    else:
        per_class = [_load_entries_for_one_class(root, i, c) for i, c in enumerate(class_ids)]
    entries = [e for class_entries in per_class for e in class_entries]

    max_class_id_len = max(len(c) for c in class_ids)
    max_filename_len = max(len(e[3]) for e in entries)
    dtype = np.dtype([
        ("class_index", "<u4"),
        ("class_id", f"U{max_class_id_len}"),
        ("start_offset", "<u4"),
        ("end_offset", "<u4"),
        ("filename", f"U{max_filename_len}"),
    ])
    entries_array = np.array(
        [(ci, class_ids[ci], start, end, name) for ci, start, end, name in entries], dtype=dtype
    )
    os.makedirs(extra, exist_ok=True)
    np.save(os.path.join(extra, "entries.npy"), entries_array)
    np.save(os.path.join(extra, "class-ids.npy"), np.array(class_ids, dtype=f"U{max_class_id_len}"))


class ImageNet22kDataset(Dataset):
    """ImageNet-22k training set with DINO multi-crop augmentation."""

    def __init__(
        self,
        split: str = "train",
        *,
        root: str,
        extra: str,
        n_global: int | None = None,
        n_local: int | None = None,
        global_img_size: int = 224,
        local_img_size: int = 98,
        multicrop_backend: str = "torchvision",
        mmap_cache_size: int = 16,  # open tarballs kept per worker; large values exhaust file descriptors
    ):
        if split != "train":
            raise ValueError("ImageNet-22k has no validation split; set run_validation: false.")
        self.root = root
        self._entries = np.load(os.path.join(extra, "entries.npy"), mmap_mode="r")
        self._class_ids = np.load(os.path.join(extra, "class-ids.npy"), mmap_mode="r")

        @lru_cache(maxsize=mmap_cache_size)
        def _mmap_tarball(class_id: str) -> mmap:
            with open(os.path.join(root, f"{class_id}.tar")) as f:
                return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)

        self._mmap_tarball = _mmap_tarball

        self.multi_crop = None
        if n_global is not None and n_local is not None:
            crop_cls = DinoMultiCropKornia if multicrop_backend == "kornia" else DinoMultiCrop
            self.multi_crop = crop_cls(
                n_global=n_global,
                n_local=n_local,
                global_size=global_img_size,
                local_size=local_img_size,
                global_scale=(0.3, 1.0),
                local_scale=(0.05, 0.3),
            )
        print(f"Loaded ImageNet-22k: {self.num_classes} classes, {len(self)} samples")

    def _read_bytes(self, index: int) -> bytes:
        entry = self._entries[index]
        data = self._mmap_tarball(str(entry["class_id"]))[entry["start_offset"]:entry["end_offset"]]
        data = data[512:]  # skip the tar member header block
        if data[:2] == b"\x1f\x8b":
            with GzipFile(fileobj=io.BytesIO(data)) as g:
                data = g.read()
        return data

    def _decode(self, index: int):
        """Decode with libjpeg-turbo (PIL fallback); on an unreadable file retry a random index."""
        for attempt in range(_MAX_DECODE_RETRIES + 1):
            try:
                data = self._read_bytes(index)
                try:
                    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                    return torchvision.io.decode_image(buf, mode=torchvision.io.ImageReadMode.RGB), index
                except RuntimeError:
                    return TF.to_image(Image.open(io.BytesIO(data)).convert("RGB")), index
            except Exception as e:
                if attempt >= _MAX_DECODE_RETRIES:
                    raise
                logger.warning("Skipping unreadable image at index %d: %s", index, e)
                index = random.randint(0, len(self) - 1)

    def __getitem__(self, index: int):
        img, index = self._decode(index)
        target = int(self._entries[index]["class_index"])
        if self.multi_crop is not None:
            img = self.multi_crop(img)
        return img, target

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def num_classes(self) -> int:
        return len(self._class_ids)
