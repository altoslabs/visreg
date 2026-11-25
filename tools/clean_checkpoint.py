"""Clean VISReg checkpoints for HuggingFace upload.

Extracts backbone + projection head weights (no optimizer, scheduler, online probe
or training config), strips wrapper prefixes, and saves a flat state dict:

    cls_token, pos_embed, patch_embed.*, blocks.*, norm.*   # timm ViT backbone
    proj.0 ... proj.8                                       # projection head

The source checkpoint (--src) is resolved under $VISREG_CKPT_DIR (default ./checkpoints).

Usage:
    python tools/clean_checkpoint.py visreg-vit-l-inet22k --src <run-dir>/checkpoints/<file>.pt
"""

import argparse
import os
from pathlib import Path

import torch

from visreg.models import ViTEncoder

_CKPT_BASE = os.environ.get("VISREG_CKPT_DIR", "./checkpoints")

CHECKPOINTS = {
    "visreg-vit-b-inet1k": {
        "arch": "vit_base_patch16_224",
        "model": "vit_b",
        "proj_dim": 256,
        "proj_gelu": False,
    },
    "visreg-vit-l-inet1k": {
        "arch": "vit_large_patch14_224",
        "model": "vit_l",
        "proj_dim": 384,
        "proj_gelu": True,
    },
    "visreg-vit-l-inet22k": {
        "arch": "vit_large_patch14_224",
        "model": "vit_l",
        "proj_dim": 384,
        "proj_gelu": True,
    },
}


def clean_state_dict(raw_state_dict: dict) -> dict:
    """Strip prefixes, keeping backbone (unprefixed) and projection head keys."""
    cleaned = {}
    for k, v in raw_state_dict.items():
        # Strip torch.compile prefix
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        # Strip backbone wrapper prefix
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        elif k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def verify(cfg: dict, cleaned: dict) -> None:
    """Fail unless the export reconstructs a complete ViTEncoder and holds only tensors."""
    non_tensor = [k for k, v in cleaned.items() if not torch.is_tensor(v)]
    if non_tensor:
        raise ValueError(f"Non-tensor entries would be published: {non_tensor}")

    encoder = ViTEncoder(model_name=cfg["model"], proj_dim=cfg["proj_dim"], proj_gelu=cfg["proj_gelu"])
    remapped = {k if k.startswith("proj.") else f"backbone.{k}": v for k, v in cleaned.items()}
    encoder.load_state_dict(remapped, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=list(CHECKPOINTS), help="Release to export")
    parser.add_argument("--src", required=True, help="Source checkpoint, relative to $VISREG_CKPT_DIR")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name
    cfg = CHECKPOINTS[name]
    src = os.path.join(_CKPT_BASE, args.src)
    print(f"Processing: {name}")
    print(f"Source: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    raw = ckpt.get("net_state_dict", ckpt)
    print(f"  Raw keys: {len(raw)}, dropping {sorted(k for k in ckpt if k != 'net_state_dict')}")

    cleaned = clean_state_dict(raw)
    verify(cfg, cleaned)
    n_proj = sum(1 for k in cleaned if k.startswith("proj."))
    print(f"  Clean keys: {len(cleaned) - n_proj} backbone + {n_proj} projection head")

    out_path = args.out_dir / f"{name}.pth"
    torch.save(cleaned, out_path)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
