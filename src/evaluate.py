"""Phase 4: final evaluation on the held-out TEST split (run once, after training is finished).

    python src\\evaluate.py
    python src\\evaluate.py --data-root "D:\\data\\final_merged_dataset"   # if the images moved

Uses the saved checkpoint and the SAME preprocessing as training (preprocess.preprocess ->
preprocess.to_model_tensor_array). Nothing is trained or tuned here.

Outputs (reports/):
    eval_summary.json            overall + per-style + border-subset results, timing, checkpoint info
    eval_per_class.csv           precision / recall / F1 / support per class
    eval_confusion_matrix.csv    32x32 counts (rows = true, columns = predicted)
    eval_confusion_matrix.png    heatmap (row-normalised)
    eval_top_confusions.csv      most frequent (true -> predicted) mistakes
    eval_predictions.csv         one row per test image (for error analysis)
    eval_misclassified.png       a grid of wrongly classified test images
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels
from model import build_model


def load_checkpoint(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert ck["arch"] == "mobilenet_v3_small", f"unexpected arch {ck['arch']}"
    assert ck["num_classes"] == NUM_CLASSES == 32
    # the class order stored in the checkpoint must equal labels.py
    for i in range(NUM_CLASSES):
        assert ck["idx_to_char"][str(i)] == IDX_TO_CHAR[i], f"class {i} differs from labels.py"
        assert ck["idx_to_name"][str(i)] == IDX_TO_NAME[i], f"class {i} name differs from labels.py"
    model = build_model(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(dev).eval(), ck


def resolve_path(p, data_root):
    """Manifest stores absolute Windows paths; remap if the folder was moved."""
    if Path(p).exists() or data_root is None:
        return p
    tail = str(p).replace("\\", "/").split("final_merged_dataset/")[-1]
    return str(Path(data_root) / tail)


def to_tensor(crop_u8):
    return torch.from_numpy(pp.to_model_tensor_array(crop_u8))


@torch.no_grad()
def predict_batches(model, crops, dev, bs=128):
    probs = []
    for s in range(0, len(crops), bs):
        xb = torch.stack([to_tensor(c) for c in crops[s:s + bs]]).to(dev)
        probs.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(probs)


def metrics_block(y, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/manifest.csv")
    ap.add_argument("--ckpt", default="models/mobilenetv3_small_best.pt")
    ap.add_argument("--data-root", default=None, help="folder 'final_merged_dataset' if images were moved")
    ap.add_argument("--timing-n", type=int, default=300, help="images used for the timing measurement")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    verify_labels()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt, dev)
    print(f"Loaded {args.ckpt}  (best epoch {ck['training']['best_epoch']}, "
          f"val_acc {ck['training']['best_val_acc']:.4f})  device={dev}")

    df = pd.read_csv(args.manifest)
    n_all_test = int((df["split"] == "test").sum())
    # same cleaning rule as training: glitch images are excluded
    df = df[(df["split"] == "test") & (~df["glitch"].astype(bool))].reset_index(drop=True)
    df["image_path"] = [resolve_path(p, args.data_root) for p in df["image_path"]]
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images not found, e.g. {missing[0]}\n"
                         f"Use  --data-root <path to final_merged_dataset>")
    print(f"Test images: {len(df)} (of {n_all_test} in the test split; {n_all_test - len(df)} glitch files excluded)")

    crops = np.zeros((len(df), 224, 224), np.uint8)
    for i, p in enumerate(tqdm(df["image_path"], desc="preprocessing", unit="img")):
        crops[i] = pp.preprocess(p, out_size=224, margin_frac=ck["preprocess"]["margin_frac"])["crop"]

    probs = predict_batches(model, crops, dev)
    y = df["class_idx"].to_numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    df["pred_idx"], df["confidence"], df["correct"] = pred, conf, pred == y
    df["pred_name"] = [IDX_TO_NAME[int(i)] for i in pred]

    # ---- overall + per style + border subset
    overall = metrics_block(y, pred)
    per_style = {s: metrics_block(g["class_idx"].to_numpy(), g["pred_idx"].to_numpy())
                 for s, g in df.groupby("style")}
    border = df["touches_border"].astype(bool)
    subsets = {"touches_border": metrics_block(y[border], pred[border]) if border.any() else None,
               "not_touching_border": metrics_block(y[~border], pred[~border])}
    print(f"\nTEST accuracy: {overall['accuracy']:.4f}  macro-F1: {overall['macro_f1']:.4f}  (n={overall['n']})")
    for s, m in per_style.items():
        print(f"  style={s:10s} n={m['n']}  accuracy={m['accuracy']:.4f}")

    # ---- per-class table
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=range(NUM_CLASSES), zero_division=0)
    per_class = pd.DataFrame(dict(index=range(NUM_CLASSES), char=[IDX_TO_CHAR[i] for i in range(NUM_CLASSES)],
                                  name=[IDX_TO_NAME[i] for i in range(NUM_CLASSES)],
                                  precision=p, recall=r, f1=f, support=sup))
    Path("reports").mkdir(exist_ok=True)
    per_class.to_csv("reports/eval_per_class.csv", index=False, encoding="utf-8-sig")
    print("\nWeakest classes by F1:\n", per_class.sort_values("f1").head(5)[["name", "precision", "recall", "f1", "support"]].to_string(index=False))

    # ---- confusion matrix
    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    names = [IDX_TO_NAME[i] for i in range(NUM_CLASSES)]
    pd.DataFrame(cm, index=names, columns=names).to_csv("reports/eval_confusion_matrix.csv")
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmn, xticklabels=names, yticklabels=names, cmap="Blues", vmin=0, vmax=1, ax=ax,
                cbar_kws=dict(label="fraction of true class"))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Test confusion matrix (accuracy {overall['accuracy']:.4f})")
    fig.savefig("reports/eval_confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- most common confusions
    off = [(names[i], names[j], int(cm[i, j])) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
           if i != j and cm[i, j] > 0]
    top = pd.DataFrame(off, columns=["true", "predicted", "count"]).sort_values("count", ascending=False)
    top.to_csv("reports/eval_top_confusions.csv", index=False)
    print("\nMost common mistakes (true -> predicted):\n", top.head(10).to_string(index=False) if len(top) else "none")

    # ---- misclassified examples
    wrong = df[~df["correct"]]
    if len(wrong):
        pick = wrong.sample(min(40, len(wrong)), random_state=0)
        fig, axes = plt.subplots(5, 8, figsize=(14, 9)); [a.axis("off") for a in axes.ravel()]
        for ax, (i, row) in zip(axes.ravel(), pick.iterrows()):
            ax.imshow(crops[i], cmap="gray")
            ax.set_title(f"{row['class_name']} -> {row['pred_name']}\n{row['confidence']:.2f}", fontsize=7)
        fig.savefig("reports/eval_misclassified.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    df.drop(columns=["image_path"]).assign(file=df["image_path"].map(lambda s: Path(s).name)) \
      .to_csv("reports/eval_predictions.csv", index=False, encoding="utf-8-sig")

    # ---- timing: one image at a time, median over images (after warm-up)
    idx = np.random.RandomState(0).choice(len(df), min(args.timing_n, len(df)), replace=False)
    for _ in range(10):
        with torch.no_grad(): model(to_tensor(crops[idx[0]])[None].to(dev))
    t_model, t_full = [], []
    for i in idx:
        t0 = time.perf_counter()
        res = pp.preprocess(df["image_path"][i], out_size=224)
        x = to_tensor(res["crop"])[None].to(dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
            if dev.type == "cuda": torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_model.append((t2 - t1) * 1000); t_full.append((t2 - t0) * 1000)
    timing = dict(n_images=len(idx), device=str(dev), median_model_ms=float(np.median(t_model)),
                  median_end_to_end_ms=float(np.median(t_full)))
    print(f"\nMedian inference time per image: model {timing['median_model_ms']:.2f} ms, "
          f"preprocessing+model {timing['median_end_to_end_ms']:.2f} ms ({dev})")

    summary = dict(checkpoint=args.ckpt, checkpoint_best_epoch=ck["training"]["best_epoch"],
                   checkpoint_val_acc=ck["training"]["best_val_acc"], overall=overall, per_style=per_style,
                   subsets=subsets, timing=timing, n_test_in_manifest=n_all_test, n_glitch_excluded=n_all_test - len(df),
                   mean_confidence_correct=float(conf[df["correct"]].mean()),
                   mean_confidence_wrong=float(conf[~df["correct"]].mean()) if len(wrong) else None,
                   note="Split is by unique image, not by writer/font; see README limitations.")
    Path("reports/eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved results to reports/eval_*.csv|png|json")


if __name__ == "__main__":
    main()