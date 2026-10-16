"""Read-only inspection of the raw dataset. Changes nothing."""
import sys, zipfile, random, re
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def main(root):
    root = Path(root)
    if not root.exists():
        sys.exit(f"Path not found: {root}")

    zips = list(root.rglob("*.zip")) if root.is_dir() else ([root] if root.suffix == ".zip" else [])
    if zips:
        print("ZIP files found (extract them first):")
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                print(" ", z, "->", len(zf.namelist()), "entries")
        if root.suffix == ".zip":
            return

    files = [p for p in root.rglob("*") if p.is_file()]
    ext = Counter(p.suffix.lower() for p in files)
    print(f"\nTotal files: {len(files)}")
    print("By extension:", dict(ext))

    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    print("Image files:", len(imgs))
    if not imgs:
        return

    # Folder structure: images per immediate parent folder
    per_parent = Counter(p.parent.relative_to(root) for p in imgs)
    depths = Counter(len(p.relative_to(root).parts) - 1 for p in imgs)
    print("\nFolder depth of images (0 = directly in root):", dict(depths))
    print(f"Number of folders containing images: {len(per_parent)}")
    print("\nFirst 40 folders (images per folder):")
    for k, v in sorted(per_parent.items(), key=lambda x: str(x[0]))[:40]:
        print(f"  {str(k):50s} {v}")

    # Image properties
    sizes, modes, bad = Counter(), Counter(), []
    sample = random.Random(0).sample(imgs, min(300, len(imgs)))
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            bad.append((str(p), str(e)))
    print(f"\nProperties of {len(sample)} random images:")
    print("  Sizes (top 8):", sizes.most_common(8))
    print("  Modes:", dict(modes))
    print("  Unreadable:", len(bad), bad[:3])

    # File name patterns (writer IDs, labels?)
    print("\nFirst 15 file names:")
    for p in imgs[:15]:
        print("  ", p.relative_to(root))
    nums = Counter(re.sub(r"\d+", "#", p.stem) for p in imgs)
    print("\nFile-name patterns (digits -> #):", nums.most_common(8))

    # Save a sample sheet for visual check
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = random.Random(1).sample(imgs, min(24, len(imgs)))
        fig, axes = plt.subplots(4, 6, figsize=(12, 8))
        for ax, p in zip(axes.ravel(), pick):
            ax.imshow(Image.open(p).convert("L"), cmap="gray")
            ax.set_title(f"{p.parent.name}"[:14], fontsize=7)
            ax.axis("off")
        out = Path("reports"); out.mkdir(exist_ok=True)
        fig.savefig(out / "dataset_samples.png", dpi=110, bbox_inches="tight")
        print("\nSaved sample sheet: reports\\dataset_samples.png")
    except Exception as e:
        print("Sample sheet skipped:", e)

if __name__ == "__main__":
    main(sys.argv[1])