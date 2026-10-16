"""Make review sheets: label-conflict images and border-touching images."""
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def white_gray(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("L")

def sheet(rows, title_fn, out, cols=6):
    n = len(rows)
    if n == 0:
        print("nothing to show for", out); return
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(2.2 * cols, 2.4 * nrows))
    for ax in (axes.ravel() if n > 1 else [axes]):
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(white_gray(r["image_path"]), cmap="gray")
        ax.set_title(title_fn(r), fontsize=7)
    Path("reports").mkdir(exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)

dropped = pd.read_csv("dataset/processed/dropped.csv")
conf = dropped[dropped["reason"] == "label_conflict"].sort_values(["canonical_hash", "class_name"])
print("Label-conflict files:", len(conf))
print(conf.groupby("canonical_hash")["class_name"].apply(lambda s: " vs ".join(sorted(set(s)))).value_counts().to_string())
sheet(conf.head(24), lambda r: f"{r['class_name']}\n{Path(r['image_path']).stem}", "reports/review_conflicts.png")

man = pd.read_csv("dataset/manifest.csv")
b = man[(man["touches_border"]) & (man["source"] == "ds101")]
print("\nBorder-touching ds101 kept images:", len(b))
print(b.groupby("class_name").size().sort_values(ascending=False).head(8).to_string())
random.seed(0)
sample = b.sample(min(30, len(b)), random_state=0)
sheet(sample, lambda r: f"{r['class_name']} g{r['form_group']}", "reports/review_border.png", cols=6)