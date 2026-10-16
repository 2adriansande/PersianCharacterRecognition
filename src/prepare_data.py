"""Phase 2: build a clean, deduplicated manifest with leakage-aware splits.

Usage (from the project root, with .venv active):
    python src\\prepare_data.py --root "<path to final_merged_dataset>"

What it does
  1. Scans class folders 01_alef ... 32_ye (ignores __MACOSX and ._ files).
  2. Computes a "canonical signature" per image: binarize, crop to the ink,
     pad to square, shrink to 24x24. Identical-looking images share a signature,
     even when they differ in file bytes, size or position.
  3. Removes exact/near duplicates inside a class, drops signatures that appear
     under two different classes (label conflict), blank images, and the
     mislabeled wide "word" images in alpha 01_alef.
  4. Splits the unique images into train/val/test per class (stratified).
  5. Writes dataset/manifest.csv, dataset/processed/dropped.csv, labels.csv.

LIMITATION: the data has no writer or font IDs, so the split is by unique image,
not by writer. Printed test accuracy is therefore optimistic for unseen fonts.
"""
import argparse, hashlib, re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels, write_label_csv

IMG_EXT = {".png", ".jpg", ".jpeg"}
NAME_PAT = re.compile(r"^(ds\d+|alpha)_(\d+)_(\d+)$")
FOLDER_ALIASES = {"he_j": "he_jimi", "ain": "ayn", "ghain": "ghayn"}
SIG_SIZE = 24


def class_from_folder(folder_name):
    """'08_he_j' -> (7, 'he_jimi'). Raises if the number and name disagree with labels.py."""
    m = re.match(r"^(\d{2})_(.+)$", folder_name)
    if not m:
        raise ValueError(f"Unexpected class folder name: {folder_name}")
    idx = int(m.group(1)) - 1
    name = FOLDER_ALIASES.get(m.group(2), m.group(2))
    if not (0 <= idx < NUM_CLASSES) or IDX_TO_NAME[idx] != name:
        raise ValueError(f"Folder '{folder_name}' does not match label map "
                         f"(expected '{IDX_TO_NAME.get(idx)}').")
    return idx, name


def signature(path):
    """Return dict(sig, touches_border, aspect, ink_frac) or dict(error=...)."""
    try:
        gray = pp.load_image(path)
    except pp.InvalidImageError as e:
        return {"error": f"unreadable: {e}"}
    mask = pp.binarize(gray)
    ink_frac = float((mask > 0).mean())
    bbox = pp.find_character_bbox(mask)
    if bbox is None or ink_frac < 0.002:
        return {"error": "blank"}
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    glyph = mask[y0:y1, x0:x1]
    gh, gw = glyph.shape
    side = max(gh, gw)
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - gh) // 2:(side - gh) // 2 + gh, (side - gw) // 2:(side - gw) // 2 + gw] = glyph
    small = cv2.resize(canvas, (SIG_SIZE, SIG_SIZE), interpolation=cv2.INTER_AREA) > 127
    return {
        "sig": hashlib.md5(np.packbits(small).tobytes()).hexdigest(),
        "touches_border": bool(x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1),
        "aspect": float((x1 - x0) / max(1, (y1 - y0))),
        "ink_frac": ink_frac,
    }


def scan(root):
    root = Path(root)
    rows, dropped = [], []
    folders = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "__MACOSX")
    if len(folders) != NUM_CLASSES:
        raise SystemExit(f"Expected {NUM_CLASSES} class folders in {root}, found {len(folders)}.")
    for folder in folders:
        idx, name = class_from_folder(folder.name)
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith("._"))
        for p in tqdm(files, desc=folder.name, unit="img", leave=False):
            m = NAME_PAT.match(p.stem)
            source, group = (m.group(1), int(m.group(2))) if m else ("unknown", -1)
            info = signature(p)
            base = dict(image_path=str(p), class_idx=idx, class_name=name, char=IDX_TO_CHAR[idx],
                        source=source, form_group=group)
            if "error" in info:
                dropped.append({**base, "reason": info["error"]})
            else:
                rows.append({**base, "canonical_hash": info["sig"], "touches_border": info["touches_border"],
                             "aspect": info["aspect"], "ink_frac": info["ink_frac"]})
    return pd.DataFrame(rows), dropped


def clean(df, dropped, alef_max_aspect=1.0):
    # a) wide 'word' images inside alpha alef (the word 'alef' written out, not the letter)
    wide = (df["source"] == "alpha") & (df["class_idx"] == 0) & (df["aspect"] > alef_max_aspect)
    dropped += [{**r, "reason": "alef_word_image"} for r in df[wide].to_dict("records")]
    df = df[~wide]

    # b) same signature under two different classes = label conflict -> drop all of them
    nclass = df.groupby("canonical_hash")["class_idx"].nunique()
    bad = set(nclass[nclass > 1].index)
    conflict = df["canonical_hash"].isin(bad)
    dropped += [{**r, "reason": "label_conflict"} for r in df[conflict].to_dict("records")]
    df = df[~conflict]

    # c) keep one file per (class, signature); count how many copies existed
    df = df.sort_values("image_path")
    copies = df.groupby(["class_idx", "canonical_hash"])["image_path"].transform("size")
    df = df.assign(n_copies=copies)
    dup = df.duplicated(["class_idx", "canonical_hash"], keep="first")
    dropped += [{**r, "reason": "duplicate"} for r in df[dup].to_dict("records")]
    return df[~dup].reset_index(drop=True), dropped


def split(df, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = "train"
    for c, g in df.groupby("class_idx"):
        order = rng.permutation(g.index.to_numpy())
        n_test = max(1, int(round(len(order) * test_frac)))
        n_val = max(1, int(round(len(order) * val_frac)))
        df.loc[order[:n_test], "split"] = "test"
        df.loc[order[n_test:n_test + n_val], "split"] = "val"
    return df


def check_manifest(df):
    assert df["image_path"].is_unique, "image listed twice"
    per_hash = df.groupby("canonical_hash")["split"].nunique()
    assert (per_hash == 1).all(), "identical-looking image appears in more than one split"
    assert set(df["split"]) <= {"train", "val", "test"}
    assert df["class_idx"].between(0, NUM_CLASSES - 1).all()
    missing = set(range(NUM_CLASSES)) - set(df["class_idx"])
    assert not missing, f"classes with no images: {sorted(missing)}"
    for s in ("train", "val", "test"):
        assert (df["split"] == s).any(), f"empty split: {s}"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing 01_alef ... 32_ye")
    ap.add_argument("--out", default="dataset/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alef-max-aspect", type=float, default=1.0)
    a = ap.parse_args()

    verify_labels()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    proc = out.parent / "processed"; proc.mkdir(exist_ok=True)
    write_label_csv(proc / "labels.csv")

    df, dropped = scan(a.root)
    print(f"\nReadable, non-blank images: {len(df)}   (dropped while reading: {len(dropped)})")
    print("Touching the image border (possible clipped strokes):")
    print(df.groupby("source")["touches_border"].agg(["sum", "size"]).rename(columns={"sum": "touching", "size": "total"}))

    df, dropped = clean(df, dropped, a.alef_max_aspect)
    df["style"] = "printed"
    df["writer_id"] = ""          # not available in this dataset
    df = split(df, seed=a.seed)
    check_manifest(df)

    cols = ["image_path", "class_idx", "class_name", "char", "style", "source", "form_group",
            "writer_id", "canonical_hash", "touches_border", "aspect", "n_copies", "split"]
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame(dropped).to_csv(proc / "dropped.csv", index=False, encoding="utf-8-sig")

    print("\nDropped, by reason:")
    print(pd.DataFrame(dropped)["reason"].value_counts().to_string())
    print(f"\nKept {len(df)} unique images")
    print("\nSplit x source:")
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0))
    print("\nUnique images per class: min / median / max =",
          *(int(x) for x in df.groupby('class_idx').size().agg(['min', 'median', 'max'])))
    print(df.groupby("class_name").size().sort_values().head(6).to_string())
    print(f"\nWrote {out}  and  {proc / 'dropped.csv'}")
    print("NOTE: no writer/font IDs exist, so this split is by image, not by writer or font.")


if __name__ == "__main__":
    main()
