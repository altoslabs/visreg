"""Self-supervised pretraining on ImageNet-LT with an online linear probe.

Reports overall top-1 on the ImageNet-1k validation set plus the standard many-shot (>100 train
images), medium-shot (20-100) and few-shot (<20) class groups.

    uv run python datasets/build_imagenet_lt_hf_dataset.py   # once
    uv run accelerate launch --num_processes 8 train_imagenet_lt.py
    uv run accelerate launch --num_processes 8 train_imagenet_lt.py loss=vicreg
"""

import warnings
warnings.filterwarnings("ignore", message=".*Metadata Warning.*")
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

import torch, torch.nn as nn, torch.nn.functional as F
import timm
import hydra, tqdm, wandb
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from accelerate import Accelerator
from accelerate.utils import set_seed

from visreg.data import build_dataset
from visreg.utils import get_parameter_groups, fmt_lr


class ViTEncoderLT(nn.Module):
    """ViT-S/8 whose classifier head maps the CLS token to a 512-d embedding, then a 2048-2048 projector."""

    def __init__(self, proj_dim=256, img_size=128, dynamic_img_size=False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=img_size,
            dynamic_img_size=dynamic_img_size,
        )
        self.embed_dim = 512
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """x: [N, V, C, H, W] (same-size views) or a list of [N, C, H, W] crops.

        Returns emb [N*V, embed_dim] (sample-major) and proj [V, N, proj_dim].
        """
        if isinstance(x, (list, tuple)):
            B, V = x[0].shape[0], len(x)
            if all(c.shape[-2:] == x[0].shape[-2:] for c in x):
                x = torch.stack(x, dim=1)
            else:
                by_size = {}
                for i, crop in enumerate(x):
                    by_size.setdefault(tuple(crop.shape[-2:]), []).append(i)
                emb_per_crop, proj_per_crop = [None] * V, [None] * V
                for idxs in by_size.values():
                    e = self.backbone(torch.cat([x[i] for i in idxs], dim=0))
                    for j, (ej, pj) in enumerate(zip(e.split(B), self.proj(e).split(B))):
                        emb_per_crop[idxs[j]], proj_per_crop[idxs[j]] = ej, pj
                emb = torch.stack(emb_per_crop, dim=1).reshape(B * V, -1)
                return emb, torch.stack(proj_per_crop, dim=0)
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


@hydra.main(version_base=None, config_path="configs", config_name="imagenet_lt")
def main(cfg: DictConfig):
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(0 + accelerator.process_index)

    method = HydraConfig.get().runtime.choices["loss"]
    use_multicrop = cfg.get("n_global") is not None and cfg.get("n_local") is not None
    if use_multicrop:
        n_global, n_local = int(cfg.n_global), int(cfg.n_local)
        V_total = n_global + n_local
    else:
        n_global = V_total = int(cfg.V)
        n_local = 0

    if accelerator.is_main_process:
        wandb.init(
            project="SSL-ImageNet-LT",
            name=f"{method}_lr{fmt_lr(cfg.lr)}_lamb{cfg.lamb}_projdim{cfg.proj_dim}",
            config=dict(cfg),
        )

    data_kw = dict(data_dir=cfg.data_dir, cache_dir=cfg.cache_dir, global_img_size=cfg.global_img_size)
    train_ds = build_dataset(
        "imagenet_lt", split="train", V=V_total,
        n_global=n_global if use_multicrop else None,
        n_local=n_local if use_multicrop else None,
        local_img_size=cfg.local_img_size, **data_kw,
    )
    test_ds = build_dataset("imagenet_lt", split="validation", V=1, **data_kw)
    num_classes = train_ds.num_classes
    many_shot, medium_shot, few_shot = train_ds.get_class_split()
    if accelerator.is_main_process:
        print(f"Class split: many={len(many_shot)}, medium={len(medium_shot)}, few={len(few_shot)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=8, pin_memory=True)

    needs_dynamic = use_multicrop and n_local > 0 and cfg.global_img_size != cfg.local_img_size
    net = ViTEncoderLT(proj_dim=cfg.proj_dim, img_size=cfg.global_img_size, dynamic_img_size=needs_dynamic)
    probe = nn.Sequential(nn.LayerNorm(net.embed_dim), nn.Linear(net.embed_dim, num_classes))
    if accelerator.state.num_processes > 1:
        net = timm.layers.convert_sync_batchnorm(net)
    reg_loss_fn = instantiate(cfg.loss)

    net_param_groups = get_parameter_groups(net, weight_decay=5e-2, no_decay_head=True)
    for g in net_param_groups:
        g["lr"] = cfg.lr
    probe_param_groups = [{"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}]
    opt = torch.optim.AdamW(net_param_groups + probe_param_groups)

    net, probe, opt, train_loader, test_loader = accelerator.prepare(net, probe, opt, train_loader, test_loader)
    reg_loss_fn = reg_loss_fn.to(accelerator.device)

    warmup_steps = len(train_loader) * cfg.warmup_epochs
    total_steps = len(train_loader) * cfg.epochs
    scheduler = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-5),
        ],
        milestones=[warmup_steps],
    )

    if accelerator.is_main_process:
        print(f"{accelerator.num_processes} GPUs | V={V_total} | bs={cfg.bs}/GPU | lr={cfg.lr} | "
              f"lamb={cfg.lamb} | proj_dim={cfg.proj_dim} | loss={method}")

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        for vs, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", disable=not accelerator.is_main_process):
            emb, proj = net(vs)

            global_mean = proj[:n_global].mean(0, keepdim=True)
            inv_loss = (proj - global_mean).square().mean()
            if accelerator.use_distributed:
                proj_gathered = torch.cat(torch.distributed.nn.all_gather(proj.transpose(0, 1).contiguous()), dim=0)
                proj_gathered = proj_gathered.transpose(0, 1).contiguous()
            else:
                proj_gathered = proj
            reg_loss = reg_loss_fn(proj_gathered)
            jepa_loss = reg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

            # Online probe on global views only (multi-crop orders crops [global..., local...] per sample).
            emb_global = emb.detach().view(y.shape[0], V_total, -1)[:, :n_global].flatten(0, 1)
            probe_loss = F.cross_entropy(probe(emb_global), y.repeat_interleave(n_global))
            loss = jepa_loss + probe_loss

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            if accelerator.is_main_process:
                wandb.log({
                    "train/probe": probe_loss.item(),
                    "train/jepa": jepa_loss.item(),
                    "train/reg": reg_loss.item(),
                    "train/inv": inv_loss.item(),
                })

        if cfg.eval_last_only and epoch != cfg.epochs - 1:
            continue

        net.eval()
        probe.eval()
        class_correct = torch.zeros(num_classes, device=accelerator.device, dtype=torch.long)
        class_total = torch.zeros(num_classes, device=accelerator.device, dtype=torch.long)
        with torch.inference_mode():
            for vs, y in test_loader:
                emb, _ = net(vs)
                preds = probe(emb).argmax(1)
                class_total.scatter_add_(0, y, torch.ones_like(y))
                class_correct.scatter_add_(0, y, (preds == y).long())
        class_correct = accelerator.reduce(class_correct, reduction="sum")
        class_total = accelerator.reduce(class_total, reduction="sum")

        def group_acc(classes):
            idx = torch.tensor(classes, device=accelerator.device, dtype=torch.long)
            return class_correct[idx].sum().item() / max(class_total[idx].sum().item(), 1)

        acc = class_correct.sum().item() / max(class_total.sum().item(), 1)
        many_acc, medium_acc, few_acc = group_acc(many_shot), group_acc(medium_shot), group_acc(few_shot)
        if accelerator.is_main_process:
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1}: Overall={acc:.4f}, Many={many_acc:.4f}, Medium={medium_acc:.4f}, Few={few_acc:.4f}")
            wandb.log({
                "test/acc": acc,
                "test/many_shot": many_acc,
                "test/medium_shot": medium_acc,
                "test/few_shot": few_acc,
                "test/epoch": epoch,
            })

    if accelerator.is_main_process:
        print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
        wandb.finish()


if __name__ == "__main__":
    main()
