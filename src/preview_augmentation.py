"""Look at what the training augmentation does to real characters. NO training, no dataset files are changed.

    python src\\preview_augmentation.py
    python src\\preview_augmentation.py --aug-rotate-deg 8 --aug-p-thin 0     # try other settings

Writes reports\\aug_preview_printed.png and reports\\aug_preview_handwritten.png
(each row: the original crop, then 6 random augmentations; dark ink on white), and prints numbers that help judge
whether characters stay recognizable:
  - how often the geometric step is undone because ink reached the image border (a stroke/dot would be cut off)
  - how often a character ends up with FEWER separate parts (e.g. a dot lost or merged) or MORE parts
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import preprocess as pp
from augment_cfg import AugConfig, add_aug_args, cfg_from_args, augment_configurable, cfg_to_dict


def to_ink(crops):
    """uint8 dark-on-white crops (N,H,W) -> float ink tensor (N,1,H,W)."""
    return torch.from_numpy(1.0 - crops.astype(np.float32) / 255.0).unsqueeze(1)


def n_parts(mask, min_area=6):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int((st[1:, cv2.CC_STAT_AREA] >= min_area).sum())


def load_sample(manifest, n, seed, printed):
    df = pd.read_csv(manifest)
    if printed and "glitch" in df.columns:
        df = df[~df["glitch"].astype(bool)]
    df = df[df["split"] == "train"]
    paths = df.sample(min(n, len(df)), random_state=seed)["image_path"].tolist()
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[0]}")
    return np.stack([pp.preprocess(p)["crop"] for p in paths])


def sheet(ink, cfg, rows, k=6, thumb=72):
    W, H = (k + 1) * (thumb + 4), rows * (thumb + 4)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    cols = [ink[:rows]] + [augment_configurable(ink[:rows], cfg) for _ in range(k)]
    for c, batch in enumerate(cols):
        for r in range(rows):
            tile = ((1 - batch[r, 0].numpy()) * 255).astype(np.uint8)
            img.paste(Image.fromarray(tile).resize((thumb, thumb), Image.LANCZOS).convert("RGB"),
                      (c * (thumb + 4), r * (thumb + 4)))
        dr.text((c * (thumb + 4) + 2, 0), "original" if c == 0 else f"aug {c}", fill="red")
    return img


def report(name, crops, cfg, passes=5):
    ink = to_ink(crops)
    base_parts = [n_parts(c < 128) for c in crops]
    stats, fewer, more, total = {}, 0, 0, 0
    for _ in range(passes):
        aug = augment_configurable(ink, cfg, stats)[:, 0].numpy() > 0.5
        for i in range(len(aug)):
            p = n_parts(aug[i]); total += 1
            fewer += p < base_parts[i]; more += p > base_parts[i]
    print(f"{name}: {total} augmented samples from {len(crops)} characters")
    print(f"   geometric step undone because ink reached the border: {100 * stats['reverted_for_clipping'] / stats['n']:.1f}%")
    print(f"   fewer separate parts than the original (possible lost/merged dot): {100 * fewer / total:.1f}%")
    print(f"   more separate parts than the original: {100 * more / total:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--printed-manifest", default="dataset/manifest.csv")
    ap.add_argument("--hw-manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--rows", type=int, default=14, help="rows in each preview image")
    ap.add_argument("--stat-n", type=int, default=300, help="characters used for the numbers")
    ap.add_argument("--seed", type=int, default=0)
    add_aug_args(ap)
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    torch.manual_seed(args.seed)
    print("Augmentation settings:", cfg_to_dict(cfg), "\n")
    Path("reports").mkdir(exist_ok=True)
    for name, manifest, printed, out in (("printed", args.printed_manifest, True, "reports/aug_preview_printed.png"),
                                         ("handwritten", args.hw_manifest, False, "reports/aug_preview_handwritten.png")):
        if not Path(manifest).exists():
            print(f"{name}: {manifest} not found, skipped"); continue
        crops = load_sample(manifest, args.stat_n, args.seed, printed)
        sheet(to_ink(crops), cfg, min(args.rows, len(crops))).save(out)
        report(name, crops, cfg)
        print(f"   preview image: {out}\n")


if __name__ == "__main__":
    main()
