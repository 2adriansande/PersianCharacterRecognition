"""READ-ONLY inspection of a folder-per-class handwritten dataset.

    python src\\inspect_handwritten.py "C:\\Users\\adria\\Downloads\\dataset_cropped\\dataset_cropped"

Nothing in the dataset is modified. Outputs go to reports\\handwritten_inspection\\ :
    summary.csv        one row per class folder (counts, formats, sizes, brightness, ink, duplicates)
    contact_sheet.png  8 random samples per class folder, labelled with the FOLDER NAME,
                       so the folder -> letter mapping can be checked by eye
"""
import csv, hashlib, random, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SAMPLES, THUMB = 8, 64


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main(root):
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    out = Path("reports") / "handwritten_inspection"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Root: {root}")
    top = sorted(root.iterdir(), key=lambda p: natural(p.name))
    dirs = [p for p in top if p.is_dir()]
    loose = [p for p in top if p.is_file()]
    print(f"Top level: {len(dirs)} folders, {len(loose)} loose files")
    print("Folder names:", ", ".join(d.name for d in dirs))
    if loose:
        print("Loose files (first 10):", ", ".join(p.name for p in loose[:10]))
    nested = [d.name for d in dirs if any(c.is_dir() for c in d.iterdir())]
    if nested:
        print("WARNING: these folders contain sub-folders (train/test split or extra nesting?):", nested)

    rows, sheet_rows, all_hashes = [], [], {}
    rng = random.Random(0)
    for d in dirs:
        files = sorted(p for p in d.rglob("*") if p.is_file())
        imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
        other = Counter(p.suffix.lower() for p in files if p.suffix.lower() not in IMG_EXT)
        ext = Counter(p.suffix.lower() for p in imgs)
        ws, hs, bright, ink, modes, bad, hashes = [], [], [], [], Counter(), 0, Counter()
        for p in imgs:
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()
                hashes[h] += 1
                all_hashes.setdefault(h, set()).add(d.name)
                with Image.open(p) as im:
                    modes[im.mode] += 1
                    ws.append(im.width); hs.append(im.height)
                    g = np.asarray(im.convert("L"), dtype=np.float32)
                    bright.append(float(g.mean()))
                    ink.append(float((g < 128).mean()))
            except Exception:
                bad += 1
        dup = sum(c - 1 for c in hashes.values() if c > 1)
        med = lambda v: float(np.median(v)) if v else 0
        rows.append(dict(folder=d.name, images=len(imgs), unreadable=bad, exact_duplicates=dup,
                         formats=dict(ext), modes=dict(modes), non_image_files=dict(other),
                         w_min=min(ws, default=0), w_med=med(ws), w_max=max(ws, default=0),
                         h_min=min(hs, default=0), h_med=med(hs), h_max=max(hs, default=0),
                         mean_brightness=round(med(bright), 1), ink_frac_dark_pixels=round(med(ink), 3)))
        sheet_rows.append((d.name, rng.sample(imgs, min(SAMPLES, len(imgs)))))

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader(); w.writerows(rows)

    counts = [r["images"] for r in rows]
    print("\nfolder            images  unread  dups   median WxH     brightness  dark-px")
    for r in rows:
        print(f"{r['folder']:<16} {r['images']:>6} {r['unreadable']:>7} {r['exact_duplicates']:>5}   "
              f"{r['w_med']:>5.0f}x{r['h_med']:<5.0f}  {r['mean_brightness']:>9}  {r['ink_frac_dark_pixels']:>7}")
    if counts:
        print(f"\nClasses: {len(rows)} | images: {sum(counts)} | per class min/median/max: "
              f"{min(counts)}/{int(np.median(counts))}/{max(counts)}")
    print("Formats overall:", dict(sum((Counter(r['formats']) for r in rows), Counter())))
    print("Modes overall:", dict(sum((Counter(r['modes']) for r in rows), Counter())))
    cross = {h: s for h, s in all_hashes.items() if len(s) > 1}
    print(f"Identical files that appear in more than one class folder: {len(cross)}")
    print("(brightness near 255 = light background, near 0 = dark background; "
          "dark-px = fraction of pixels darker than 128)")

    # contact sheet
    label_w, pad = 110, 4
    W = label_w + SAMPLES * (THUMB + pad)
    H = max(1, len(sheet_rows)) * (THUMB + pad)
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (name, paths) in enumerate(sheet_rows):
        y = r * (THUMB + pad)
        dr.text((4, y + THUMB // 2 - 5), name, fill="black")
        for c, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    g = im.convert("RGBA")
                    bg = Image.new("RGBA", g.size, (255, 255, 255, 255))
                    g = Image.alpha_composite(bg, g).convert("RGB")
                    g.thumbnail((THUMB, THUMB))
                    tile = Image.new("RGB", (THUMB, THUMB), (200, 200, 200))
                    tile.paste(g, ((THUMB - g.width) // 2, (THUMB - g.height) // 2))
                    sheet.paste(tile, (label_w + c * (THUMB + pad), y))
            except Exception:
                pass
    sheet.save(out / "contact_sheet.png")
    print(f"\nWrote {out / 'summary.csv'} and {out / 'contact_sheet.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])