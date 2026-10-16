"""Find images whose ink spans almost the whole canvas but is mostly empty (thin frame lines).
Writes reports/review_frames.png with FILE NAMES in the titles and dataset/processed/frame_candidates.csv."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp


def stats(path):
    gray = pp.load_image(path)
    mask = pp.binarize(gray)
    H, W = mask.shape
    bbox = pp.find_character_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    area = int((mask > 0).sum())
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return dict(span_w=(x1 - x0) / W, span_h=(y1 - y0) / H, fill=area / max(1, (x1 - x0) * (y1 - y0)),
                comps=n - 1, largest_comp_frac=big / max(1, area))


def main(manifest="dataset/manifest.csv"):
    df = pd.read_csv(manifest)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), unit="img"):
        try:
            s = stats(r["image_path"])
        except Exception:
            s = None
        if s:
            rows.append({"image_path": r["image_path"], "class_name": r["class_name"],
                         "form_group": r["form_group"], "split": r["split"], **s})
    st = pd.DataFrame(rows)
    # frame-like: spans most of the canvas in BOTH directions but very little ink inside that box
    cand = st[(st.span_w > 0.8) & (st.span_h > 0.8) & (st.fill < 0.12)].copy()
    cand["file"] = cand["image_path"].map(lambda p: Path(p).name)
    out = Path("dataset/processed"); out.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out / "frame_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"\nCandidates: {len(cand)} of {len(st)} ({100*len(cand)/len(st):.2f}%)")
    print("Distribution of span_w / span_h / fill over ALL images (percentiles 50, 90, 99, 100):")
    for c in ("span_w", "span_h", "fill"):
        print(f"  {c}: ", [round(float(np.percentile(st[c], q)), 3) for q in (50, 90, 99, 100)])
    if len(cand):
        print("\nBy class:", cand.groupby("class_name").size().sort_values(ascending=False).head(8).to_dict())
        print("File-name index endings:", cand["file"].str.extract(r"_(\d+)\.")[0].value_counts().head(5).to_dict())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        pick = cand.sample(min(24, len(cand)), random_state=0)
        fig, axes = plt.subplots(4, 6, figsize=(13, 9))
        for ax in axes.ravel(): ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
            im = Image.open(r["image_path"]).convert("RGBA")
            ax.imshow(Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im).convert("L"), cmap="gray")
            ax.set_title(f"{r['class_name']} g{r['form_group']}\n{r['file']}", fontsize=6)
        Path("reports").mkdir(exist_ok=True)
        fig.savefig("reports/review_frames.png", dpi=100, bbox_inches="tight")
        print("Saved reports/review_frames.png")

if __name__ == "__main__":
    main()