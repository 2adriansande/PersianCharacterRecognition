"""Mark broken source images (a solid rectangle with stray frame fragments) in dataset/manifest.csv.

Rule A: ds101 files whose running index is 00269 (all 23 label-conflict files had it).
Rule B: the ink is dominated by one solid, wide rectangle (fill > 0.9, width/height 2-5,
        at least half of all ink) - not a letter.
Adds columns `glitch` (bool) and `glitch_reason`. Writes reports/glitch_review.png.
"""
import re
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp

IDX_PAT = re.compile(r"_(\d+)\.\w+$")


def rect_dominates(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return False
    total = float((mask > 0).sum())
    for x, y, w, h, area in st[1:]:
        if area < 60 or h == 0:
            continue
        fill, ratio = area / float(w * h), w / float(h)
        if fill > 0.9 and 2.0 <= ratio <= 5.0 and area >= 0.5 * total:
            return True
    return False


def main(manifest="dataset/manifest.csv"):
    df = pd.read_csv(manifest)
    a, b = [], []
    for p, src in tqdm(list(zip(df["image_path"], df["source"])), unit="img"):
        m = IDX_PAT.search(Path(p).name)
        a.append(bool(src == "ds101" and m and int(m.group(1)) == 269))
        try:
            b.append(rect_dominates(pp.binarize(pp.load_image(p))))
        except Exception:
            b.append(False)
    df["glitch_a"], df["glitch_b"] = a, b
    df["glitch"] = df["glitch_a"] | df["glitch_b"]
    df["glitch_reason"] = np.where(df.glitch_a & df.glitch_b, "A+B", np.where(df.glitch_a, "A_idx269",
                                   np.where(df.glitch_b, "B_rectangle", "")))
    df = df.drop(columns=["glitch_a", "glitch_b"])
    df.to_csv(manifest, index=False, encoding="utf-8-sig")
    g = df[df["glitch"]]
    print(f"\nGlitch images: {len(g)} of {len(df)}")
    print(g["glitch_reason"].value_counts().to_string())
    print("By split:", g["split"].value_counts().to_dict())
    print("Classes that lose images:", g["class_name"].value_counts().head(6).to_dict())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    pick = g.sample(min(48, len(g)), random_state=0) if len(g) else g
    if len(pick):
        fig, axes = plt.subplots(6, 8, figsize=(14, 10))
        for ax in axes.ravel(): ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
            im = Image.open(r["image_path"]).convert("RGBA")
            ax.imshow(Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im).convert("L"), cmap="gray")
            ax.set_title(f"{r['class_name']} {r['glitch_reason']}\n{Path(r['image_path']).name}", fontsize=6)
        Path("reports").mkdir(exist_ok=True)
        fig.savefig("reports/glitch_review.png", dpi=100, bbox_inches="tight")
        print("Saved reports/glitch_review.png")

if __name__ == "__main__":
    main()
