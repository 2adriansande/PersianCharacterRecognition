"""Contact sheets of the EXACT crops the model will receive (preprocess.preprocess output)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp


def sheet(rows, title, out, cols=8):
    n = len(rows)
    nr = (n + cols - 1) // cols
    fig, axes = plt.subplots(nr, cols, figsize=(1.7 * cols, 1.9 * nr))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        ax.imshow(r["crop"], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{r['name']} g{r['group']}", fontsize=6)
    fig.suptitle(title, fontsize=10)
    Path("reports").mkdir(exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main(manifest="dataset/manifest.csv"):
    df = pd.read_csv(manifest)
    recs = []
    for _, r in tqdm(df.iterrows(), total=len(df), unit="img"):
        try:
            res = pp.preprocess(r["image_path"])
        except pp.InvalidImageError:
            continue
        ink = float((res["crop"] < 128).mean())
        recs.append(dict(name=r["class_name"], group=r["form_group"], crop=res["crop"], ink=ink,
                         border=bool(r.get("touches_border", False)), src=r.get("source", "")))
    print("Processed:", len(recs))
    rng = np.random.RandomState(0)
    sheet([recs[i] for i in rng.choice(len(recs), min(48, len(recs)), replace=False)],
          "48 random model inputs", "reports/inputs_random.png")
    by_ink = sorted(recs, key=lambda r: r["ink"])
    sheet(by_ink[:32], "32 crops with the LEAST ink (suspect: tiny marks or faint lines)", "reports/inputs_least_ink.png")
    sheet(by_ink[-32:], "32 crops with the MOST ink", "reports/inputs_most_ink.png")
    print("ink fraction percentiles (0,1,50,99,100):",
          [round(float(np.percentile([r['ink'] for r in recs], q)), 3) for q in (0, 1, 50, 99, 100)])

if __name__ == "__main__":
    main()