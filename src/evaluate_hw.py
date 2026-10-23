"""Final evaluation of the HANDWRITTEN model on the held-out TEST forms (run once, after training).

    python src\\evaluate_hw.py

Uses the saved checkpoint and the same preprocessing as training. Nothing is trained or tuned here.
Reuses load_checkpoint / predict_batches / metrics_block from evaluate.py (unchanged).

Outputs (reports/, all prefixed hw_ so the printed-model reports are not overwritten):
    hw_eval_summary.json   overall + per-form results, confidence statistics, checkpoint info
    hw_eval_per_class.csv, hw_eval_confusion_matrix.csv/.png, hw_eval_top_confusions.csv,
    hw_eval_predictions.csv, hw_eval_misclassified.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES, verify_labels
from evaluate import load_checkpoint, predict_batches, metrics_block       # reused unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--ckpt", default="models/handwritten_mobilenetv3_small_best.pt")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    verify_labels()
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}\nTrain it first:  python src\\train_hw.py")
    if not Path(args.manifest).exists():
        raise SystemExit(f"{args.manifest} not found. Run  python src\\prepare_data_hw.py --root <dataset folder>")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt, dev)
    print(f"Loaded {args.ckpt}  (best epoch {ck['training']['best_epoch']}, "
          f"val_acc {ck['training']['best_val_acc']:.4f})  device={dev}")

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == "test"].reset_index(drop=True)
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images not found, e.g. {missing[0]}\n"
                         "The manifest stores absolute paths; re-run prepare_data_hw.py on this computer.")
    print(f"Test images: {len(df)} (forms {sorted(df['form'].unique())})")

    crops = np.zeros((len(df), 224, 224), np.uint8)
    for i, p in enumerate(tqdm(df["image_path"], desc="preprocessing", unit="img")):
        crops[i] = pp.preprocess(p, out_size=224, margin_frac=ck["preprocess"]["margin_frac"])["crop"]

    probs = predict_batches(model, crops, dev)
    y = df["class_idx"].to_numpy()
    pred, conf = probs.argmax(1), probs.max(1)
    df["pred_idx"], df["confidence"], df["correct"] = pred, conf, pred == y
    df["pred_name"] = [IDX_TO_NAME[int(i)] for i in pred]

    overall = metrics_block(y, pred)
    per_form = {int(f): metrics_block(g["class_idx"].to_numpy(), g["pred_idx"].to_numpy()) for f, g in df.groupby("form")}
    print(f"\nTEST accuracy: {overall['accuracy']:.4f}  macro-F1: {overall['macro_f1']:.4f}  (n={overall['n']})")
    for f, m in per_form.items():
        print(f"  form f{f:02d}: n={m['n']}  accuracy={m['accuracy']:.4f}")

    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=range(NUM_CLASSES), zero_division=0)
    per_class = pd.DataFrame(dict(index=range(NUM_CLASSES), char=[IDX_TO_CHAR[i] for i in range(NUM_CLASSES)],
                                  name=[IDX_TO_NAME[i] for i in range(NUM_CLASSES)],
                                  precision=p, recall=r, f1=f1, support=sup))
    Path("reports").mkdir(exist_ok=True)
    per_class.to_csv("reports/hw_eval_per_class.csv", index=False, encoding="utf-8-sig")
    print("\nWeakest classes by F1:\n", per_class.sort_values("f1").head(8)[["name", "precision", "recall", "f1", "support"]].to_string(index=False))

    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    names = [IDX_TO_NAME[i] for i in range(NUM_CLASSES)]
    pd.DataFrame(cm, index=names, columns=names).to_csv("reports/hw_eval_confusion_matrix.csv")
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmn, xticklabels=names, yticklabels=names, cmap="Blues", vmin=0, vmax=1, ax=ax,
                cbar_kws=dict(label="fraction of true class"))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Handwritten test confusion matrix (accuracy {overall['accuracy']:.4f})")
    fig.savefig("reports/hw_eval_confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    off = [(names[i], names[j], int(cm[i, j])) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
           if i != j and cm[i, j] > 0]
    top = pd.DataFrame(off, columns=["true", "predicted", "count"]).sort_values("count", ascending=False)
    top.to_csv("reports/hw_eval_top_confusions.csv", index=False)
    print("\nMost common mistakes (true -> predicted):\n", top.head(12).to_string(index=False) if len(top) else "none")

    wrong = df[~df["correct"]]
    if len(wrong):
        pick = wrong.sample(min(40, len(wrong)), random_state=0)
        fig, axes = plt.subplots(5, 8, figsize=(14, 9)); [a.axis("off") for a in axes.ravel()]
        for ax, (i, row) in zip(axes.ravel(), pick.iterrows()):
            ax.imshow(crops[i], cmap="gray")
            ax.set_title(f"{row['class_name']} -> {row['pred_name']}\n{row['confidence']:.2f}", fontsize=7)
        fig.savefig("reports/hw_eval_misclassified.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    df.assign(file=df["image_path"].map(lambda s: Path(s).name)).drop(columns=["image_path"]) \
      .to_csv("reports/hw_eval_predictions.csv", index=False, encoding="utf-8-sig")

    summary = dict(checkpoint=args.ckpt, checkpoint_best_epoch=ck["training"]["best_epoch"],
                   checkpoint_val_acc=ck["training"]["best_val_acc"], overall=overall, per_form=per_form,
                   test_forms=[int(f) for f in sorted(df["form"].unique())],
                   mean_confidence_correct=float(conf[df["correct"]].mean()),
                   mean_confidence_wrong=float(conf[~df["correct"]].mean()) if len(wrong) else None,
                   note="Split is by form (writing session). Whether two forms share a writer is unknown.")
    Path("reports/hw_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved results to reports/hw_eval_*.csv|png|json")


if __name__ == "__main__":
    main()
