"""READ-ONLY: check what the 'fXX' part of the file names (class_NN_fXX_sYYYY.png) means.

    python src\\inspect_forms_hw.py "C:\\Users\\adria\\Downloads\\dataset_cropped\\dataset_cropped"

Prints how many images each form (fXX) has overall and per class, and writes
reports\\handwritten_inspection\\forms_sheet.png : one row per form, the SAME three letters
(class_02, class_15, class_26), 4 samples each. If each form is a different writer, the
handwriting style should visibly differ from row to row.
Nothing in the dataset is modified.
"""
import re, sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PAT = re.compile(r"^class_(\d+)_f(\d+)_s(\d+)\.(png|jpg|jpeg|bmp)$", re.I)
SHEET_CLASSES, PER = ("class_02", "class_15", "class_26"), 4
THUMB, PAD = 64, 4


def main(root):
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    by_f, by_cf, s_range, bad = Counter(), defaultdict(Counter), {}, []
    files = defaultdict(list)                      # (class, form) -> paths
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            m = PAT.match(p.name)
            if not m:
                bad.append(p.name); continue
            c, f, s = f"class_{m.group(1)}", int(m.group(2)), int(m.group(3))
            if c != d.name:
                bad.append(f"{p.name} (inside {d.name})"); continue
            by_f[f] += 1
            by_cf[d.name][f] += 1
            lo, hi = s_range.get(d.name, (10**9, -1))
            s_range[d.name] = (min(lo, s), max(hi, s))
            files[(d.name, f)].append(p)

    print(f"Distinct forms (fXX): {len(by_f)}  ->  {sorted(by_f)}")
    print("Images per form:", {f: by_f[f] for f in sorted(by_f)})
    per = [sorted(c.values()) for c in by_cf.values()]
    if per:
        print(f"Images per class per form: min {min(v[0] for v in per)}, max {max(v[-1] for v in per)}")
    n_forms_per_class = Counter(len(c) for c in by_cf.values())
    print("Number of forms present in each class folder:", dict(n_forms_per_class))
    print("Sample-number range (sYYYY) per class (first 6):",
          {c: s_range[c] for c in list(s_range)[:6]})
    if bad:
        print(f"WARNING: {len(bad)} files did not match the pattern, e.g. {bad[:5]}")

    forms = sorted(by_f)
    out = Path("reports") / "handwritten_inspection"
    out.mkdir(parents=True, exist_ok=True)
    label_w = 60
    W = label_w + len(SHEET_CLASSES) * PER * (THUMB + PAD) + 2 * PAD
    sheet = Image.new("RGB", (W, max(1, len(forms)) * (THUMB + PAD)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, f in enumerate(forms):
        y = r * (THUMB + PAD)
        dr.text((4, y + THUMB // 2 - 5), f"f{f:02d}", fill="black")
        x = label_w
        for c in SHEET_CLASSES:
            for p in files.get((c, f), [])[:PER]:
                try:
                    with Image.open(p) as im:
                        g = im.convert("L").convert("RGB")
                        g.thumbnail((THUMB, THUMB))
                        sheet.paste(g, (x, y))
                except Exception:
                    pass
                x += THUMB + PAD
            x += max(0, PER - len(files.get((c, f), [])[:PER])) * (THUMB + PAD)
            dr.line([(x - PAD // 2, y), (x - PAD // 2, y + THUMB)], fill="red")
    sheet.save(out / "forms_sheet.png")
    print(f"\nWrote {out / 'forms_sheet.png'}  (rows = forms; columns = {', '.join(SHEET_CLASSES)}, red line between letters)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])