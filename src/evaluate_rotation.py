"""How does a model behave on ROTATED images? (Measurement only: nothing is trained, no files in dataset/ are changed.)

    python src\\evaluate_rotation.py                       # combined model
    python src\\evaluate_rotation.py --model printed
    python src\\evaluate_rotation.py --model handwritten --angles -30 -15 0 15 30 90 180 270

Takes a stratified sample of the TEST images of both datasets (--n-per-class per class), rotates each original image by
every angle (counter-clockwise, canvas enlarged, filled with the image's own background colour), runs the normal
preprocessing + model, and prints accuracy per angle and per dataset. Images the pipeline rejects count as wrong.
Writes reports\\rotation_<model>.csv and reports\\rotation_<model>.png.

Use it BEFORE and AFTER any change meant to improve rotated-image accuracy, with the same --angles and --seed.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import preprocess as pp
from labels import verify_labels
from evaluate import load_checkpoint, predict_batches, resolve_path      # reused unchanged

CKPTS = {"printed": "models/mobilenetv3_small_best.pt",
         "handwritten": "models/handwritten_mobilenetv3_small_best.pt",
         "combined": "models/combined_mobilenetv3_small_best.pt"}
DEFAULT_ANGLES = [-45, -30, -20, -10, 0, 10, 20, 30, 45, 90, 180, 270]


def rotate(gray, deg):
    """Rotate counter-clockwise by deg, enlarge the canvas so nothing is cut off, fill with the border colour."""
    if deg % 360 == 0:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    bg = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=bg)


def load_test(manifest, printed, n_per_class, seed, data_root):
    df = pd.read_csv(manifest)
    if printed:
        df = df[~df["glitch"].astype(bool)]
    df = df[df["split"] == "test"]
    # shuffle once (fixed seed), then keep the first n per class; keeps every column in any pandas version
    df = df.sample(frac=1, random_state=seed).groupby("class_idx").head(n_per_class).reset_index(drop=True)
    df["image_path"] = [resolve_path(p, data_root if printed else None) for p in df["image_path"]]
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[0]}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CKPTS), default="combined")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (overrides --model's default)")
    ap.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--printed-manifest", default="dataset/manifest.csv")
    ap.add_argument("--hw-manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--data-root", default=None, help="printed dataset folder if the images were moved")
    ap.add_argument("--tag", default=None, help="name used in the output file names (default: the model name)")
    args = ap.parse_args()

    verify_labels()
    ckpt = args.ckpt or CKPTS[args.model]
    if not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    margin = ck["preprocess"]["margin_frac"]
    print(f"Model: {ckpt}   device: {dev}")

    sets = {"printed": load_test(args.printed_manifest, True, args.n_per_class, args.seed, args.data_root),
            "handwritten": load_test(args.hw_manifest, False, args.n_per_class, args.seed, None)}
    rows = []
    for name, df in sets.items():
        grays = [pp.load_image(p) for p in tqdm(df["image_path"], desc=f"loading {name}", unit="img")]
        y = df["class_idx"].to_numpy()
        for ang in args.angles:
            crops, ok = [], []
            for i, g in enumerate(grays):
                try:
                    crops.append(pp.preprocess(rotate(g, ang), out_size=224, margin_frac=margin)["crop"])
                    ok.append(i)
                except pp.InvalidImageError:
                    pass
            correct = 0
            if crops:
                pred = predict_batches(model, np.stack(crops), dev).argmax(1)
                correct = int((pred == y[ok]).sum())
            rows.append(dict(dataset=name, angle=ang, n=len(grays), rejected=len(grays) - len(ok),
                             accuracy=correct / len(grays)))
    res = pd.DataFrame(rows)

    table = res.pivot(index="angle", columns="dataset", values="accuracy").reindex(args.angles)
    print(f"\nAccuracy by rotation angle (counter-clockwise), {args.n_per_class} test images per class per dataset")
    print((table * 100).round(1).rename(columns=lambda c: c + " %").to_string())
    print(f"\nimages per dataset: {len(sets['printed'])} printed, {len(sets['handwritten'])} handwritten; "
          f"rejected by the pipeline: {int(res['rejected'].sum())} (counted as wrong)")

    tag = args.tag or args.model
    Path("reports").mkdir(exist_ok=True)
    res.to_csv(f"reports/rotation_{tag}.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in table.columns:
        ax.plot(table.index, table[name] * 100, marker="o", label=name)
    ax.set_xlabel("rotation (degrees, counter-clockwise)"); ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100); ax.set_title(f"Rotation robustness: {tag}"); ax.grid(alpha=.3); ax.legend()
    fig.savefig(f"reports/rotation_{tag}.png", dpi=110, bbox_inches="tight")
    print(f"Saved reports/rotation_{tag}.csv and reports/rotation_{tag}.png")


if __name__ == "__main__":
    main()