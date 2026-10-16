"""Read-only inspection of final_merged_dataset. Ignores __MACOSX and ._ stubs."""
import sys, re, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
PAT = re.compile(r"^(ds\d+|alpha)_(\d+)_(\d+)$")

def white_gray(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("L")

def main(root):
    root = Path(root)
    imgs = [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXT
            and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    print("Real images:", len(imgs))

    classes = Counter(p.parent.name for p in imgs)
    print("\nImages per class:")
    for k in sorted(classes):
        print(f"  {k:12s} {classes[k]}")
    print("Number of class folders:", len(classes))

    rows = []
    unmatched = []
    for p in imgs:
        m = PAT.match(p.stem)
        if m:
            rows.append((p, p.parent.name, m.group(1), m.group(2), m.group(3)))
        else:
            unmatched.append(p.name)
    print("\nFile names not matching ds101_n_idx / alpha_n_idx:", len(unmatched), unmatched[:5])

    prefix_ext = Counter((r[2], r[0].suffix.lower()) for r in rows)
    print("Prefix x extension:", dict(prefix_ext))

    print("\nPrefix per class (first 8 classes):")
    pc = defaultdict(Counter)
    for p, c, pre, mid, idx in rows:
        pc[c][pre] += 1
    for c in sorted(pc)[:8]:
        print(f"  {c:12s} {dict(pc[c])}")

    # Middle number: could be writer / font / batch id?
    for pre in sorted({r[2] for r in rows}):
        mids = defaultdict(lambda: [0, set()])
        for p, c, pr, mid, idx in rows:
            if pr == pre:
                mids[mid][0] += 1
                mids[mid][1].add(c)
        counts = [v[0] for v in mids.values()]
        spread = [len(v[1]) for v in mids.values()]
        print(f"\n[{pre}] distinct middle numbers: {len(mids)}")
        print(f"  images per middle number: min {min(counts)}, median {int(np.median(counts))}, max {max(counts)}")
        print(f"  classes per middle number: min {min(spread)}, median {int(np.median(spread))}, max {max(spread)}")
        print("  sample middle numbers:", sorted(mids, key=int)[:12])

    # Image properties by prefix
    random.seed(0)
    print("\nProperties (sample of up to 150 per prefix):")
    for pre in sorted({r[2] for r in rows}):
        pr = [r[0] for r in rows if r[2] == pre]
        smp = random.sample(pr, min(150, len(pr)))
        sizes, modes, alpha_used, ink = Counter(), Counter(), 0, []
        for p in smp:
            with Image.open(p) as im:
                sizes[im.size] += 1; modes[im.mode] += 1
                if im.mode == "RGBA" and np.array(im)[..., 3].min() < 250:
                    alpha_used += 1
            g = np.array(white_gray(p))
            ink.append((g < 128).mean())
        print(f"  {pre}: sizes {sizes.most_common(4)}, modes {dict(modes)}, "
              f"images with real transparency {alpha_used}, mean ink fraction {np.mean(ink):.3f}, "
              f"blank-looking (<0.2% ink) {sum(i < 0.002 for i in ink)}")

    # Exact duplicates
    print("\nHashing for exact duplicates ...")
    seen, dups = {}, []
    for p in imgs:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            dups.append((p, seen[h]))
        else:
            seen[h] = p
    print("Exact duplicate files:", len(dups))
    cross = sum(1 for a, b in dups if a.parent.name != b.parent.name)
    print("  of which appear under two DIFFERENT classes (label conflict):", cross)

    # Sample sheets, one per prefix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("reports"); out.mkdir(exist_ok=True)
    for pre in sorted({r[2] for r in rows}):
        pr = [r for r in rows if r[2] == pre]
        pick = random.sample(pr, min(24, len(pr)))
        fig, axes = plt.subplots(4, 6, figsize=(12, 8))
        for ax, r in zip(axes.ravel(), pick):
            ax.imshow(white_gray(r[0]), cmap="gray")
            ax.set_title(f"{r[1]}\n{r[0].stem}", fontsize=6)
            ax.axis("off")
        f = out / f"samples_{pre}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("Saved", f)

if __name__ == "__main__":
    main(sys.argv[1])