"""Phase 3: train MobileNetV3 Small on the TRAIN split, choose the best epoch on VAL.
The TEST split is never used here (evaluate.py uses it once, at the end).

    python src\\train.py --smoke        # 1-2 minute pipeline check on a tiny subset
    python src\\train.py                # full training
"""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels
from model import build_model, set_trainable, set_train_mode
from augment import augment

MEAN = torch.tensor(pp.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(pp.IMAGENET_STD).view(1, 3, 1, 1)
OUT_SIZE = 224


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_crops(df, cache_path):
    """uint8 array (N,224,224), dark ink on white = exactly what preprocess.preprocess returns."""
    key = hashlib.md5(("|".join(df["image_path"]) + f"|{OUT_SIZE}").encode()).hexdigest()
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["x"]
    X = np.zeros((len(df), OUT_SIZE, OUT_SIZE), np.uint8)
    for i, p in enumerate(tqdm(df["image_path"], desc="preprocessing", unit="img")):
        X[i] = pp.preprocess(p, out_size=OUT_SIZE)["crop"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, x=X, key=np.array(key))
    return X


def to_input(batch_u8, dev):
    """uint8 crops -> (ink float for augmentation)"""
    x = torch.from_numpy(batch_u8).to(dev).float().div_(255.0)
    return (1.0 - x).unsqueeze(1)                       # ink = 1


def to_model(ink):
    img = (1.0 - ink).expand(-1, 3, -1, -1)             # back to dark-on-white, 3 channels
    return (img - MEAN.to(ink.device)) / STD.to(ink.device)   # same math as preprocess.to_model_tensor_array


@torch.no_grad()
def evaluate(model, X, y, dev, bs=256):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss, correct = 0.0, 0
    for s in range(0, len(X), bs):
        xb = to_model(to_input(X[s:s + bs], dev))
        yb = torch.from_numpy(y[s:s + bs]).to(dev)
        out = model(xb)
        tot_loss += crit(out, yb).item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
    return tot_loss / len(X), correct / len(X)


def run_stage(name, model, Xtr, ytr, Xva, yva, dev, epochs, lr, unfreeze_from, args, history, state, class_w):
    set_trainable(model, unfreeze_from)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * int(np.ceil(len(Xtr) / args.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for ep in range(1, epochs + 1):
        set_train_mode(model, unfreeze_from)
        perm = np.random.permutation(len(Xtr))
        run_loss, t0 = 0.0, time.time()
        for s in range(0, len(perm), args.batch):
            idx = np.sort(perm[s:s + args.batch])
            ink = augment(to_input(Xtr[idx], dev))
            yb = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = crit(model(to_model(ink)), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item() * len(idx)
        va_loss, va_acc = evaluate(model, Xva, yva, dev)
        history.append(dict(stage=name, epoch=len(history) + 1, train_loss=run_loss / len(Xtr),
                            val_loss=va_loss, val_acc=va_acc))
        better = (va_acc, -va_loss) > (state["best_acc"], -state["best_loss"])
        if better:
            state.update(best_acc=va_acc, best_loss=va_loss, best_epoch=len(history), bad=0,
                         weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            state["bad"] += 1
        print(f"[{name}] epoch {ep}/{epochs}  train_loss {run_loss/len(Xtr):.4f}  val_loss {va_loss:.4f}  "
              f"val_acc {va_acc:.4f}  {'*best*' if better else ''}  ({time.time()-t0:.0f}s)")
        if state["bad"] >= args.patience:
            print("Early stopping: no validation improvement."); return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/manifest.csv")
    ap.add_argument("--out", default="models/mobilenetv3_small_best.pt")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=5e-4)
    ap.add_argument("--unfreeze-from", type=int, default=8)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-border", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    verify_labels(); set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    df = pd.read_csv(args.manifest)
    if "glitch" not in df.columns:
        raise SystemExit("Run  python src\\mark_glitch.py  first (adds the 'glitch' column).")
    df = df[~df["glitch"].astype(bool)]
    if args.exclude_border:
        df = df[~df["touches_border"].astype(bool)]
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    if args.smoke:
        tr = tr.groupby("class_idx", group_keys=False).head(12)
        va = va.groupby("class_idx", group_keys=False).head(4)
        args.epochs_head, args.epochs_ft, args.batch = 1, 1, 32
    print(f"train {len(tr)}  val {len(va)}   (test split is not loaded)")

    cache = "dataset/processed/crops_smoke.npz" if args.smoke else "dataset/processed/crops_trainval.npz"
    tv = pd.concat([tr, va])
    Xall = load_crops(tv, cache)
    Xtr, Xva = Xall[:len(tr)], Xall[len(tr):]
    ytr, yva = tr["class_idx"].to_numpy(np.int64), va["class_idx"].to_numpy(np.int64)

    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
    w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    class_w = torch.tensor(w, device=dev)

    model = build_model(pretrained=not args.no_pretrained).to(dev)
    history, state = [], dict(best_acc=-1.0, best_loss=1e9, best_epoch=0, bad=0, weights=None)
    n_blocks = len(model.features)
    stopped = run_stage("head", model, Xtr, ytr, Xva, yva, dev, args.epochs_head, args.lr_head,
                        n_blocks, args, history, state, class_w)
    if not stopped:
        run_stage("finetune", model, Xtr, ytr, Xva, yva, dev, args.epochs_ft, args.lr_ft,
                  args.unfreeze_from, args, history, state, class_w)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = dict(
        state_dict=state["weights"],
        arch="mobilenet_v3_small", num_classes=NUM_CLASSES,
        idx_to_char={str(k): v for k, v in IDX_TO_CHAR.items()},
        idx_to_name={str(k): v for k, v in IDX_TO_NAME.items()},
        preprocess=dict(out_size=OUT_SIZE, margin_frac=0.12, mean=[float(v) for v in pp.IMAGENET_MEAN],
                        std=[float(v) for v in pp.IMAGENET_STD], ink="dark on white crop"),
        training=dict(seed=args.seed, best_epoch=state["best_epoch"], best_val_acc=state["best_acc"],
                      best_val_loss=state["best_loss"], excluded_border=args.exclude_border,
                      n_train=len(tr), n_val=len(va), smoke=args.smoke),
    )
    if not args.smoke:
        torch.save(ckpt, out)
        print(f"\nSaved best checkpoint (epoch {state['best_epoch']}, val_acc {state['best_acc']:.4f}) -> {out}")
    else:
        print("\nSMOKE RUN: checkpoint not saved (pipeline check only; numbers are meaningless).")

    hist = pd.DataFrame(history)
    Path("reports").mkdir(exist_ok=True)
    hist.to_csv("reports/train_history_smoke.csv" if args.smoke else "reports/train_history.csv", index=False)
    if not args.smoke and len(hist):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train (augmented, label-smoothed)")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="validation"); ax[0].set_title("Loss"); ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_acc"]); ax[1].set_title("Validation accuracy")
        for a in ax: a.set_xlabel("epoch")
        fig.savefig("reports/training_curves.png", dpi=110, bbox_inches="tight")
        print("Saved reports/training_curves.png")


if __name__ == "__main__":
    main()
