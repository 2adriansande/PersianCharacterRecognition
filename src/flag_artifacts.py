"""Flag images with a thin page-frame artifact. Adds `suspect_frame` to dataset/manifest.csv.
Review reports/review_frames.png before excluding anything."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp

def is_frame(mask, span=0.7, hollow=0.2):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return False
    H, W = mask.shape
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1      # largest first
    for i in order[1:]:                                        # skip the largest component
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        if (w / W > span or h / H > span) and area / float(w * h) < hollow:
            return True
    return False

def main(manifest="dataset/manifest.csv"):
    df = pd.read_csv(manifest)
    flags = []
    for p in tqdm(df["image_path"], unit="img"):
        try:
            flags.append(is_frame(pp.binarize(pp.load_image(p))))
        except Exception:
            flags.append(False)
    df["suspect_frame"] = flags
    df.to_csv(manifest, index=False, encoding="utf-8-sig")
    s = df[df["suspect_frame"]]
    print(f"\nFlagged {len(s)} of {len(df)} images ({100*len(s)/len(df):.1f}%)")
    print(s.groupby("source").size().to_string())
    print("\nBy class (top 8):")
    print(s.groupby("class_name").size().sort_values(ascending=False).head(8).to_string())
    print("\nBy split:", s.groupby("split").size().to_dict())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    def wg(p):
        im = Image.open(p).convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", im.size, (255,)*4), im).convert("L")
    pick = s.sample(min(36, len(s)), random_state=0) if len(s) else s
    if len(pick):
        fig, axes = plt.subplots(6, 6, figsize=(12, 12))
        for ax in axes.ravel(): ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
            ax.imshow(wg(r["image_path"]), cmap="gray")
            ax.set_title(f"{r['class_name']} g{r['form_group']}", fontsize=7)
        Path("reports").mkdir(exist_ok=True)
        fig.savefig("reports/review_frames.png", dpi=100, bbox_inches="tight")
        print("Saved reports/review_frames.png")

if __name__ == "__main__":
    main()