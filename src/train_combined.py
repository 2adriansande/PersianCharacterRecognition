"""Train ONE model on printed + handwritten characters (new checkpoint; the printed and handwritten models are never touched).

    python src\\train_combined.py --smoke      # 1-2 minute pipeline check (no checkpoint saved)
    python src\\train_combined.py              # full training

Data: the TRAIN splits of both existing manifests (dataset/manifest.csv without glitch images, and
dataset/manifest_hw.csv) are combined; validation = both VAL splits. The splits are NOT redone, so the two test sets
stay exactly the ones the separate models were tested on (see evaluate_combined.py). Cached crops from the earlier
training runs are reused when they exist (no reprocessing).
Training loop = train.run_stage (imported, unchanged); only its augmentation function is replaced by the configurable
one in augment_cfg.py (all settings: --aug-*, see --help).
Output: models/combined_mobilenetv3_small_best.pt  (same checkpoint format as the other two models)
"""
import argparse, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import preprocess as pp
import train as T                                   # T.augment is replaced below; the rest is reused unchanged
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels
from model import build_model
from train import set_seed, run_stage, load_crops
from train_hw import load_crops_hw
from augment_cfg import add_aug_args, cfg_from_args, cfg_to_dict, augment_configurable

OUT_SIZE = 224
DEFAULT_OUT = "models/combined_mobilenetv3_small_best.pt"
PRINTED_CKPT = "models/mobilenetv3_small_best.pt"
HW_CKPT = "models/handwritten_mobilenetv3_small_best.pt"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--printed-manifest", default="dataset/manifest.csv")
    ap.add_argument("--hw-manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing combined checkpoint")
    ap.add_argument("--printed-repeat", type=int, default=1,
                    help="repeat each printed training image this many times per epoch (1 = natural mix, about 25%% printed)")
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
    add_aug_args(ap)
    args = ap.parse_args()
    cfg = cfg_from_args(args)

    out = Path(args.out)
    for protected in (PRINTED_CKPT, HW_CKPT):
        if out.resolve() == Path(protected).resolve():
            raise SystemExit(f"Refusing to write to an existing model's checkpoint ({protected}).")
    if out.exists() and not args.overwrite and not args.smoke:
        raise SystemExit(f"{out} already exists. Use --overwrite to replace it, or --out <other path>.")
    hashes_before = {p: sha256(p) for p in (PRINTED_CKPT, HW_CKPT) if Path(p).exists()}

    verify_labels(); set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    print("Augmentation:", cfg_to_dict(cfg))
    T.augment = lambda ink: augment_configurable(ink, cfg)       # run_stage looks this name up at call time

    # ---- printed data (same selection as train.py: glitch images excluded)
    dp = pd.read_csv(args.printed_manifest)
    if "glitch" not in dp.columns:
        raise SystemExit("Printed manifest has no 'glitch' column. Run  python src\\mark_glitch.py  first.")
    dp = dp[~dp["glitch"].astype(bool)]
    ptr, pva = dp[dp["split"] == "train"], dp[dp["split"] == "val"]
    # ---- handwritten data
    if not Path(args.hw_manifest).exists():
        raise SystemExit(f"{args.hw_manifest} not found. Run  python src\\prepare_data_hw.py --root <dataset folder>  first.")
    dh = pd.read_csv(args.hw_manifest)
    htr, hva = dh[dh["split"] == "train"], dh[dh["split"] == "val"]
    if args.smoke:
        ptr, pva = ptr.groupby("class_idx", group_keys=False).head(8), pva.groupby("class_idx", group_keys=False).head(2)
        htr, hva = htr.groupby("class_idx", group_keys=False).head(8), hva.groupby("class_idx", group_keys=False).head(2)
        args.epochs_head, args.epochs_ft, args.batch = 1, 1, 32
        cache_p, cache_h = "dataset/processed/crops_comb_smoke_p.npz", "dataset/processed/crops_comb_smoke_h.npz"
    else:
        cache_p, cache_h = "dataset/processed/crops_trainval.npz", "dataset/processed/crops_hw_trainval.npz"

    Xp = load_crops(pd.concat([ptr, pva]), cache_p)
    Xh = load_crops_hw(pd.concat([htr, hva]), cache_h)
    rep = max(1, args.printed_repeat)
    Xtr = np.concatenate([Xp[:len(ptr)]] * rep + [Xh[:len(htr)]])
    ytr = np.concatenate([ptr["class_idx"].to_numpy(np.int64)] * rep + [htr["class_idx"].to_numpy(np.int64)])
    Xva = np.concatenate([Xp[len(ptr):], Xh[len(htr):]])
    yva = np.concatenate([pva["class_idx"].to_numpy(np.int64), hva["class_idx"].to_numpy(np.int64)])
    n_pva = len(pva)
    del Xp, Xh
    print(f"train {len(Xtr)} (printed {len(ptr)} x{rep}, handwritten {len(htr)})   "
          f"val {len(Xva)} (printed {len(pva)}, handwritten {len(hva)})   (test splits are not loaded)")

    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), device=dev)

    model = build_model(pretrained=not args.no_pretrained).to(dev)
    history, state = [], dict(best_acc=-1.0, best_loss=1e9, best_epoch=0, bad=0, weights=None)
    n_blocks = len(model.features)
    stopped = run_stage("head", model, Xtr, ytr, Xva, yva, dev, args.epochs_head, args.lr_head,
                        n_blocks, args, history, state, class_w)
    if not stopped:
        run_stage("finetune", model, Xtr, ytr, Xva, yva, dev, args.epochs_ft, args.lr_ft,
                  args.unfreeze_from, args, history, state, class_w)

    model.load_state_dict(state["weights"])                      # best epoch, to report validation per domain
    _, acc_p = T.evaluate(model, Xva[:n_pva], yva[:n_pva], dev)
    _, acc_h = T.evaluate(model, Xva[n_pva:], yva[n_pva:], dev)
    print(f"\nBest epoch {state['best_epoch']}: validation accuracy overall {state['best_acc']:.4f} | "
          f"printed {acc_p:.4f} | handwritten {acc_h:.4f}")

    ckpt = dict(
        state_dict=state["weights"],
        arch="mobilenet_v3_small", num_classes=NUM_CLASSES, style="combined",
        idx_to_char={str(k): v for k, v in IDX_TO_CHAR.items()},
        idx_to_name={str(k): v for k, v in IDX_TO_NAME.items()},
        preprocess=dict(out_size=OUT_SIZE, margin_frac=0.12, mean=[float(v) for v in pp.IMAGENET_MEAN],
                        std=[float(v) for v in pp.IMAGENET_STD], ink="dark on white crop"),
        training=dict(seed=args.seed, best_epoch=state["best_epoch"], best_val_acc=state["best_acc"],
                      best_val_loss=state["best_loss"], val_acc_printed=acc_p, val_acc_handwritten=acc_h,
                      n_train=len(Xtr), n_train_printed=len(ptr), n_train_handwritten=len(htr),
                      printed_repeat=rep, n_val=len(Xva), smoke=args.smoke, augmentation=cfg_to_dict(cfg),
                      split="existing splits: printed by image, handwritten by form"),
    )
    if not args.smoke:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        print(f"Saved best checkpoint (epoch {state['best_epoch']}) -> {out}")
    else:
        print("SMOKE RUN: checkpoint not saved (pipeline check only; numbers are meaningless).")

    for p, h in hashes_before.items():                           # the other models must be byte-for-byte unchanged
        print(f"{p}: {'unchanged' if sha256(p) == h else 'CHANGED (unexpected!)'}")

    hist = pd.DataFrame(history)
    Path("reports").mkdir(exist_ok=True)
    hist.to_csv("reports/comb_train_history_smoke.csv" if args.smoke else "reports/comb_train_history.csv", index=False)
    if not args.smoke and len(hist):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train (augmented, label-smoothed)")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="validation"); ax[0].set_title("Loss (combined)"); ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_acc"]); ax[1].set_title("Validation accuracy (combined)")
        for a in ax: a.set_xlabel("epoch")
        fig.savefig("reports/comb_training_curves.png", dpi=110, bbox_inches="tight")
        print("Saved reports/comb_training_curves.png")


if __name__ == "__main__":
    main()
