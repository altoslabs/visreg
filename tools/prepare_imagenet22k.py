"""One-time index build for ImageNet-22k (visreg/data/imagenet22k.py).

`--root` holds the ImageNet-22k tar files: per-class tarballs from image-net.org (<wnid>.tar)
or the training shards of timm/imagenet-22k-wds (imagenet22k-train-XXXX.tar).
Step 1 writes root/blocks/<name>.log (offset and name of every tar member);
step 2 writes entries.npy and class-ids.npy under `--extra` (default: root/indices).

Usage:
    python tools/prepare_imagenet22k.py --root <imagenet22k-dir> --workers 32
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from visreg.data.imagenet22k import dump_imagenet22k_indices

TAR_BLOCK = 512


def _parse_name(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip("\x00 ").strip()


def _parse_size(raw: bytes) -> int:
    s = raw.decode("ascii", errors="replace").strip("\x00 ")
    return int(s, 8) if s else 0


def write_blocks_log(tar_path: str, log_path: str) -> None:
    """Write `block <N>: <name>` for each member, ending with a `** Block of NULs **` sentinel."""
    lines = []
    with open(tar_path, "rb") as f:
        while True:
            pos = f.tell()
            header = f.read(TAR_BLOCK)
            if len(header) < TAR_BLOCK or header.strip(b"\x00") == b"":
                lines.append(f"block {pos // TAR_BLOCK}: ** Block of NULs **")
                break
            name = _parse_name(header[0:100])
            size = _parse_size(header[124:136])
            data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
            if header[156] == ord("L"):  # GNU long name: payload is the name, real header follows
                name = _parse_name(f.read(data_blocks * TAR_BLOCK))
                pos = f.tell()
                header = f.read(TAR_BLOCK)
                size = _parse_size(header[124:136])
                data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK
            if name and name != "." and not name.startswith("./"):
                lines.append(f"block {pos // TAR_BLOCK}: {name}")
            f.seek(data_blocks * TAR_BLOCK, os.SEEK_CUR)
    Path(log_path).write_text("\n".join(lines) + "\n")


def _write_blocks_log(job):
    write_blocks_log(*job)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory with per-class <wnid>.tar files")
    p.add_argument("--extra", default=None, help="Output directory for entries.npy / class-ids.npy (default: root/indices)")
    p.add_argument("--workers", type=int, default=0, help="Parallel worker processes (0 = sequential)")
    args = p.parse_args()

    root = Path(args.root)
    extra = args.extra or str(root / "indices")
    tars = sorted(root.glob("*.tar"))
    if not tars:
        raise FileNotFoundError(f"No .tar files found under {root}")

    blocks_dir = root / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    jobs = [(str(t), str(blocks_dir / f"{t.stem}.log")) for t in tars]
    print(f"[1/2] Indexing {len(jobs)} tarballs -> {blocks_dir}")
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_write_blocks_log, jobs, chunksize=64))
    else:
        for job in jobs:
            write_blocks_log(*job)

    print(f"[2/2] Building entries.npy and class-ids.npy -> {extra}")
    dump_imagenet22k_indices(str(root), extra, workers=args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
