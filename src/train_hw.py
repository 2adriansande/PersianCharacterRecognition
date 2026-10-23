"""Train the HANDWRITTEN model (separate checkpoint; the printed model and its files are never touched).

    python src\\train_hw.py --smoke      # 1-2 minute pipeline check on a tiny subset (no checkpoint is saved)
    python src\\train_hw.py              # full training

Same recipe as train.py (MobileNetV3 Small, ImageNet start, head stage then fine-tuning, AdamW + OneCycle,
label smoothing, GPU augmentation) - the training loop (train.run_stage) is imported, not copied.
Trains on the 'train' split, picks the best epoch on 'val'. The 'test' split is never loaded here.
Output: models/handwritten_mobilenetv3_small_best.pt  (same checkpoint format as the printed model)
"""
import argparse, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels
from model import build_model
from train import set_seed, run_stage          # reused unchanged

OUT_SIZE = 224
DEFAULT_OUT = "models/handwritten_mobilenetv3_small_best.pt"
PRINTED_CKPT = "models/mobilenetv3_small_best.pt"


def load_crops_hw(df, cache_path, desc="preprocessing"):
    """uint8 array (N,224,224), dark ink on white = exactly what preprocess.preprocess returns. Cached (compressed)."""
    key = hashlib.md5(("|".join(df["image_path"]) + f"|{OUT_SIZE}").encode()).hexdigest()
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["x"]
    X = np.zeros((len(df), OUT_SIZE, OUT_SIZE), np.uint8)
    for i, p in enumerate(tqdm(df["image_path"], desc=desc, unit="img")):
        X[i] = pp.preprocess(p, out_size=OUT_SIZE)["crop"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=X, key=np.array(key))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing handwritten checkpoint")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=5e-4)
    ap.add_argument("--unfreeze-from", type=int, default=8)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.resolve() == Path(PRINTED_CKPT).resolve():
        raise SystemExit(f"Refusing to write to the printed model checkpoint ({PRINTED_CKPT}).")
    if out.exists() and not args.overwrite and not args.smoke:
        raise SystemExit(f"{out} already exists. Use --overwrite to replace it, or --out <other path>.")

    verify_labels(); set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    if not Path(args.manifest).exists():
        raise SystemExit(f"{args.manifest} not found. Run  python src\\prepare_data_hw.py --root <dataset folder>  first.")
    df = pd.read_csv(args.manifest)
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    if args.smoke:
        tr = tr.groupby("class_idx", group_keys=False).head(12)
        va = va.groupby("class_idx", group_keys=False).head(4)
        args.epochs_head, args.epochs_ft, args.batch = 1, 1, 32
    print(f"train {len(tr)} (forms {sorted(tr['form'].unique())})  val {len(va)} (forms {sorted(va['form'].unique())})"
          "   (test split is not loaded)")

    cache = "dataset/processed/crops_hw_smoke.npz" if args.smoke else "dataset/processed/crops_hw_trainval.npz"
    Xall = load_crops_hw(pd.concat([tr, va]), cache)
    Xtr, Xva = Xall[:len(tr)], Xall[len(tr):]
    ytr, yva = tr["class_idx"].to_numpy(np.int64), va["class_idx"].to_numpy(np.int64)

    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), device=dev)   # ~1.0 when balanced

    model = build_model(pretrained=not args.no_pretrained).to(dev)
    history, state = [], dict(best_acc=-1.0, best_loss=1e9, best_epoch=0, bad=0, weights=None)
    n_blocks = len(model.features)
    stopped = run_stage("head", model, Xtr, ytr, Xva, yva, dev, args.epochs_head, args.lr_head,
                        n_blocks, args, history, state, class_w)
    if not stopped:
        run_stage("finetune", model, Xtr, ytr, Xva, yva, dev, args.epochs_ft, args.lr_ft,
                  args.unfreeze_from, args, history, state, class_w)

    ckpt = dict(
        state_dict=state["weights"],
        arch="mobilenet_v3_small", num_classes=NUM_CLASSES, style="handwritten",
        idx_to_char={str(k): v for k, v in IDX_TO_CHAR.items()},
        idx_to_name={str(k): v for k, v in IDX_TO_NAME.items()},
        preprocess=dict(out_size=OUT_SIZE, margin_frac=0.12, mean=[float(v) for v in pp.IMAGENET_MEAN],
                        std=[float(v) for v in pp.IMAGENET_STD], ink="dark on white crop"),
        training=dict(seed=args.seed, best_epoch=state["best_epoch"], best_val_acc=state["best_acc"],
                      best_val_loss=state["best_loss"], n_train=len(tr), n_val=len(va), smoke=args.smoke,
                      train_forms=[int(f) for f in sorted(tr["form"].unique())],
                      val_forms=[int(f) for f in sorted(va["form"].unique())],
                      split="by form"),
    )
    if not args.smoke:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        print(f"\nSaved best checkpoint (epoch {state['best_epoch']}, val_acc {state['best_acc']:.4f}) -> {out}")
    else:
        print("\nSMOKE RUN: checkpoint not saved (pipeline check only; numbers are meaningless).")

    hist = pd.DataFrame(history)
    Path("reports").mkdir(exist_ok=True)
    hist.to_csv("reports/hw_train_history_smoke.csv" if args.smoke else "reports/hw_train_history.csv", index=False)
    if not args.smoke and len(hist):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train (augmented, label-smoothed)")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="validation"); ax[0].set_title("Loss (handwritten)"); ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_acc"]); ax[1].set_title("Validation accuracy (handwritten)")
        for a in ax: a.set_xlabel("epoch")
        fig.savefig("reports/hw_training_curves.png", dpi=110, bbox_inches="tight")
        print("Saved reports/hw_training_curves.png")


if __name__ == "__main__":
    main()
