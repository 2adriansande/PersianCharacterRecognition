"""Duplicate/group structure analysis. Saves dataset/processed/file_hashes.csv and sample sheets."""
import sys, re, hashlib
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
PAT = re.compile(r"^(ds\d+|alpha)_(\d+)_(\d+)$")

def white_gray(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("L")

def main(root):
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        if (p.is_file() and p.suffix.lower() in IMG_EXT
                and "__MACOSX" not in p.parts and not p.name.startswith("._")):
            m = PAT.match(p.stem)
            if m:
                rows.append(dict(path=str(p), cls=p.parent.name, prefix=m.group(1),
                                 group=int(m.group(2)), idx=int(m.group(3))))
    df = pd.DataFrame(rows)
    print("Images:", len(df))
    df["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in tqdm(df["path"], unit="img")]
    out = Path("dataset/processed"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "file_hashes.csv", index=False)

    print("\nUnique images overall:", df["md5"].nunique())
    for pre, g in df.groupby("prefix"):
        print(f"  {pre}: {len(g)} files, {g['md5'].nunique()} unique")

    print("\nUnique images per group (prefix, group):")
    pg = df.groupby(["prefix", "group"])["md5"].nunique()
    for pre in pg.index.get_level_values(0).unique():
        s = pg[pre]
        print(f"  {pre}: groups {len(s)}, unique per group min {s.min()}, median {int(s.median())}, max {s.max()}")

    print("\nUnique images per class (min / median / max):")
    uc = df.groupby("cls")["md5"].nunique()
    print(f"  {uc.min()} / {int(uc.median())} / {uc.max()}")
    print(uc.sort_values().head(8).to_string())

    # Does the same image appear in different groups of the same class?
    gmd5 = df.groupby("md5").apply(lambda x: x[["prefix", "group"]].drop_duplicates().shape[0])
    print("\nDuplicate images shared across different groups:", int((gmd5 > 1).sum()))

    # Pattern inside one group: which file indices are identical?
    g1 = df[(df.prefix == "ds101") & (df.group == 1)].sort_values("idx")
    ids = {}
    labels = [ids.setdefault(h, len(ids)) for h in g1["md5"]]
    print(f"\nds101 group 1 (class {g1['cls'].iloc[0]}): first 60 files -> unique-image id")
    print(" ", " ".join(f"{i}:{l}" for i, l in zip(g1["idx"].tolist()[:60], labels[:60])))
    diffs = Counter()
    for h, grp in g1.groupby("md5"):
        v = sorted(grp["idx"])
        diffs.update(np.diff(v).tolist())
    print("  most common index gaps between identical copies:", diffs.most_common(6))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = Path("reports"); rep.mkdir(exist_ok=True)

    # Sheet 1: first 48 files of ds101 group 1
    fig, axes = plt.subplots(6, 8, figsize=(14, 10))
    for ax, (_, r), l in zip(axes.ravel(), g1.head(48).iterrows(), labels[:48]):
        ax.imshow(white_gray(r["path"]), cmap="gray"); ax.axis("off")
        ax.set_title(f"idx {r['idx']} uid {l}", fontsize=6)
    fig.savefig(rep / "group_ds101_1.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    # Sheet 2: one image from each ds101 group of 01_alef and 02_be (what each group is)
    gs = df[(df.prefix == "ds101") & (df["cls"].isin(["01_alef", "02_be", "10_dal"]))]
    gs = gs.sort_values("idx").groupby(["cls", "group"]).head(3)
    fig, axes = plt.subplots(4, 9, figsize=(14, 7))
    for ax in axes.ravel(): ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), gs.iterrows()):
        ax.imshow(white_gray(r["path"]), cmap="gray")
        ax.set_title(f"{r['cls']} g{r['group']}", fontsize=6)
    fig.savefig(rep / "groups_overview.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    # Sheet 3: widest 12 images in alpha alef (label sanity check)
    ae = df[(df.prefix == "alpha") & (df["cls"] == "01_alef")].drop_duplicates("md5")
    asp = []
    for p in ae["path"]:
        m = np.array(white_gray(p)) < 128
        ys, xs = np.where(m)
        asp.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) if len(xs) else 0)
    ae = ae.assign(aspect=asp).sort_values("aspect", ascending=False).head(12)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    for ax, (_, r) in zip(axes.ravel(), ae.iterrows()):
        ax.imshow(white_gray(r["path"]), cmap="gray"); ax.axis("off")
        ax.set_title(f"{Path(r['path']).stem}\nw/h {r['aspect']:.2f}", fontsize=6)
    fig.savefig(rep / "alef_widest.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("\nSaved reports\\group_ds101_1.png, groups_overview.png, alef_widest.png")

if __name__ == "__main__":
    main(sys.argv[1])