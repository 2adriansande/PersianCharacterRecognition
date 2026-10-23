"""Handwritten dataset preparation (separate from the printed pipeline; prepare_data.py is not used or changed).

    python src\\prepare_data_hw.py --root "C:\\Users\\adria\\Downloads\\dataset_cropped\\dataset_cropped"

What it does
  1. Scans class_01 ... class_32 (class number -> labels.py index, see labels_hw.py).
  2. Runs the SAME image checks as the app (preprocess.load_image / binarize / find_character_bbox);
     unreadable or blank images are dropped and listed.
  3. Drops exact duplicates (identical pixels): a pixel-identical image under two classes is a label
     conflict (all copies dropped); inside one class only the first copy is kept.
     (The 24x24 look-alike signature of the printed pipeline is NOT used: handwritten strokes are thin
     and simple letters such as alef would collide too often and good samples would be thrown away.)
  4. Splits BY FORM (the fXX part of the file name). Every form goes entirely to train, val or test, so the
     test set contains handwriting sessions the model never saw in training. Which forms go where is
     chosen with a fixed seed (or give --val-forms / --test-forms) and is printed and stored in the manifest.
  5. Writes dataset/manifest_hw.csv, dataset/processed/dropped_hw.csv and two preview images
     (reports/hw_preview_crops.png, reports/hw_preview_extremes.png) so the processed crops can be checked by eye.

LIMITATION: a form is assumed to be one writing session / writer. The dataset does not say whether two forms
were written by the same person; if they were, the test score is somewhat optimistic.
"""
import argparse, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_CHAR, NUM_CLASSES, verify_labels
from labels_hw import class_from_folder, parse_filename

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(p):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def inspect_image(path):
    """dict(pixel_hash, ink_frac, touches_border, aspect) or dict(error=...). Same checks as preprocess.preprocess."""
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
    return dict(pixel_hash=hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest(),
                ink_frac=ink_frac,
                touches_border=bool(x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1),
                aspect=float((x1 - x0) / max(1, (y1 - y0))))


def find_class_root(root):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=natural_key)
    if len(dirs) == 1 and sum(1 for d in dirs[0].iterdir() if d.is_dir()) == NUM_CLASSES:
        raise SystemExit(f"'{root}' only contains one folder. The class folders are one level deeper; use:\n  --root \"{dirs[0]}\"")
    if len(dirs) != NUM_CLASSES:
        raise SystemExit(f"Expected {NUM_CLASSES} class folders (class_01 ... class_{NUM_CLASSES}) in {root}, found {len(dirs)}.")
    return dirs


def scan(root):
    rows, dropped = [], []
    for folder in find_class_root(root):
        try:
            idx, name = class_from_folder(folder.name)
        except ValueError as e:
            raise SystemExit(str(e))
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
        for p in tqdm(files, desc=folder.name, unit="img", leave=False):
            parsed = parse_filename(p.stem)
            if parsed is None or parsed[0] != idx:
                raise SystemExit(f"File name does not match class_NN_fXX_sYYYY in folder {folder.name}: {p.name}")
            _, form, sample_no = parsed
            base = dict(image_path=str(p), class_idx=idx, class_name=name, char=IDX_TO_CHAR[idx],
                        form=form, sample_no=sample_no)
            info = inspect_image(p)
            if "error" in info:
                dropped.append({**base, "reason": info["error"]})
            else:
                rows.append({**base, **info})
    return pd.DataFrame(rows), dropped


def clean(df, dropped):
    # a) identical pixels under two different classes = label conflict -> drop all copies
    nclass = df.groupby("pixel_hash")["class_idx"].nunique()
    conflict = df["pixel_hash"].isin(set(nclass[nclass > 1].index))
    dropped += [{**r, "reason": "label_conflict"} for r in df[conflict].to_dict("records")]
    df = df[~conflict]
    # b) identical pixels inside one class: keep the first file only
    df = df.sort_values("image_path")
    dup = df.duplicated(["class_idx", "pixel_hash"], keep="first")
    dropped += [{**r, "reason": "duplicate"} for r in df[dup].to_dict("records")]
    return df[~dup].reset_index(drop=True), dropped


def assign_forms(forms, val_forms, test_forms, seed):
    forms = sorted(forms)
    if (val_forms is None) != (test_forms is None):
        raise SystemExit("Give both --val-forms and --test-forms, or neither.")
    if val_forms is None:
        order = np.random.RandomState(seed).permutation(forms).tolist()
        n_test, n_val = max(1, round(0.2 * len(forms))), max(1, round(0.1 * len(forms)))
        test_forms, val_forms = sorted(order[:n_test]), sorted(order[n_test:n_test + n_val])
    if set(val_forms) & set(test_forms):
        raise SystemExit("A form cannot be in both --val-forms and --test-forms.")
    unknown = (set(val_forms) | set(test_forms)) - set(forms)
    if unknown:
        raise SystemExit(f"Unknown form number(s): {sorted(unknown)}. The dataset has forms {forms}.")
    train_forms = [f for f in forms if f not in val_forms and f not in test_forms]
    if not train_forms:
        raise SystemExit("No forms left for training.")
    return train_forms, sorted(val_forms), sorted(test_forms)


def check_manifest(df):
    assert df["image_path"].is_unique, "image listed twice"
    assert set(df["split"]) <= {"train", "val", "test"}
    per_form = df.groupby("form")["split"].nunique()
    assert (per_form == 1).all(), "a form appears in more than one split"
    per_hash = df.groupby("pixel_hash")["split"].nunique()
    assert (per_hash == 1).all(), "identical image appears in more than one split"
    missing = set(range(NUM_CLASSES)) - set(df["class_idx"])
    assert not missing, f"classes with no images: {sorted(missing)}"
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        assert len(sub), f"empty split: {s}"
        assert sub["class_idx"].nunique() == NUM_CLASSES, f"split '{s}' misses some classes"
    return True


def _sheet(items, cols, thumb=64, label_h=11):
    """items: list of (crop uint8 HxW, caption)."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb + 4), rows * (thumb + label_h + 4)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (crop, cap) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * (thumb + 4), r * (thumb + label_h + 4)
        sheet.paste(Image.fromarray(crop).resize((thumb, thumb), Image.LANCZOS).convert("RGB"), (x, y + label_h))
        dr.text((x + 1, y), cap, fill="black")
    return sheet


def previews(df, seed):
    Path("reports").mkdir(exist_ok=True)
    rng = np.random.RandomState(seed)
    items = []
    for c, g in df[df["split"] == "train"].groupby("class_idx"):
        for p in g.sample(min(8, len(g)), random_state=int(rng.randint(1 << 30)))["image_path"]:
            items.append((pp.preprocess(p)["crop"], f"c{c + 1:02d}"))
    _sheet(items, 8).save("reports/hw_preview_crops.png")
    low, high = df.nsmallest(24, "ink_frac"), df.nlargest(24, "ink_frac")
    ext = [(pp.preprocess(r.image_path)["crop"], f"{r.ink_frac:.3f} c{r.class_idx + 1:02d}") for r in low.itertuples()] + \
          [(pp.preprocess(r.image_path)["crop"], f"{r.ink_frac:.3f} c{r.class_idx + 1:02d}") for r in high.itertuples()]
    _sheet(ext, 8).save("reports/hw_preview_extremes.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that directly contains class_01 ... class_32")
    ap.add_argument("--out", default="dataset/manifest_hw.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-forms", type=int, nargs="+", default=None, help="e.g. --val-forms 8")
    ap.add_argument("--test-forms", type=int, nargs="+", default=None, help="e.g. --test-forms 9 10")
    args = ap.parse_args()

    verify_labels()
    df, dropped = scan(args.root)
    n_scanned = len(df) + len(dropped)
    df, dropped = clean(df, dropped)

    train_f, val_f, test_f = assign_forms(df["form"].unique().tolist(), args.val_forms, args.test_forms, args.seed)
    df["split"] = np.where(df["form"].isin(test_f), "test", np.where(df["form"].isin(val_f), "val", "train"))
    check_manifest(df)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    drop_path = Path("dataset/processed/dropped_hw.csv"); drop_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dropped).to_csv(drop_path, index=False, encoding="utf-8-sig")

    print(f"\nScanned {n_scanned} images, kept {len(df)}, dropped {len(dropped)}")
    if dropped:
        print("Dropped by reason:", pd.DataFrame(dropped)["reason"].value_counts().to_dict(), f"(list: {drop_path})")
    print(f"Forms -> train {train_f} | val {val_f} | test {test_f}")
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        per = sub.groupby("class_idx").size()
        print(f"  {s:5s} {len(sub):6d} images   per class min/max {per.min()}/{per.max()}")
    print(f"Characters touching the image border: {df['touches_border'].mean() * 100:.1f}% "
          "(expected: the dataset is tightly cropped; it is only a warning in the app)")
    print(f"Ink fraction (after Otsu) min/median/max: {df['ink_frac'].min():.4f} / {df['ink_frac'].median():.4f} / {df['ink_frac'].max():.4f}")
    previews(df, args.seed)
    print(f"\nWrote {out}, {drop_path}, reports/hw_preview_crops.png, reports/hw_preview_extremes.png")


if __name__ == "__main__":
    main()
