"""Show original / ink mask / final crop for the suspicious images."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp

df = pd.read_csv("dataset/manifest.csv")
pick = pd.concat([
    df[(df["class_name"] == "alef") & (df["form_group"] == 3)].head(4),
    df[(df["class_name"] == "za") & (df["form_group"] == 54)].head(4),
])
print("Rows picked:", len(pick))

fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
for row, (_, r) in enumerate(pick.iterrows()):
    res = pp.preprocess(r["image_path"])
    gray, mask = res["original"], res["mask"]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = sorted(stats[1:].tolist(), key=lambda s: -s[4])[:3]
    faint = gray < 235
    ys, xs = np.where(faint)
    fb = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    print(f"\n{r['class_name']} g{r['form_group']}  {Path(r['image_path']).name}")
    print("  image size (w,h):", gray.shape[::-1], " ink bbox:", res["bbox"], " faint(<235) bbox:", fb)
    print("  components:", n - 1, " largest 3 (x,y,w,h,area):", comps)
    print("  warnings:", res["warnings"])
    axes[row, 0].imshow(gray, cmap="gray"); axes[row, 0].set_title(f"{r['class_name']} g{r['form_group']} original", fontsize=7)
    axes[row, 1].imshow(mask, cmap="gray"); axes[row, 1].set_title("ink mask (what the crop uses)", fontsize=7)
    axes[row, 2].imshow(res["crop"], cmap="gray"); axes[row, 2].set_title("final crop", fontsize=7)
    for a in axes[row]: a.axis("off")
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/debug_frames.png", dpi=90, bbox_inches="tight")
print("\nSaved reports/debug_frames.png")