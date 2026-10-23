"""Evaluate the COMBINED model on BOTH held-out test sets, reported separately (run once, after training).

    python src\\evaluate_combined.py

Test sets = exactly the ones the separate models were tested on: the printed test split (glitch images excluded)
and the handwritten test forms. Nothing is trained or tuned here. If reports\\eval_summary.json and
reports\\hw_eval_summary.json exist, their stored accuracies are printed next to the combined model's, so the three
models can be compared on identical images.

Outputs (reports/, prefixed comb_): comb_eval_summary.json, comb_eval_per_class.csv (recall per class and domain),
comb_eval_top_confusions_printed.csv, comb_eval_top_confusions_handwritten.csv, comb_eval_confusion_matrix.png,
comb_eval_predictions.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import preprocess as pp
from labels import IDX_TO_NAME, NUM_CLASSES, verify_labels
from evaluate import load_checkpoint, predict_batches, metrics_block, resolve_path     # reused unchanged


def reference(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return dict(accuracy=d["overall"]["accuracy"], macro_f1=d["overall"]["macro_f1"], n=d["overall"]["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--printed-manifest", default="dataset/manifest.csv")
    ap.add_argument("--hw-manifest", default="dataset/manifest_hw.csv")
    ap.add_argument("--ckpt", default="models/combined_mobilenetv3_small_best.pt")
    ap.add_argument("--data-root", default=None, help="printed dataset folder 'final_merged_dataset' if the images were moved")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    verify_labels()
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}\nTrain it first:  python src\\train_combined.py")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt, dev)
    print(f"Loaded {args.ckpt}  (best epoch {ck['training']['best_epoch']}, "
          f"val_acc {ck['training']['best_val_acc']:.4f})  device={dev}")

    dp = pd.read_csv(args.printed_manifest)
    dp = dp[(dp["split"] == "test") & (~dp["glitch"].astype(bool))].copy()
    dp["image_path"] = [resolve_path(p, args.data_root) for p in dp["image_path"]]
    dh = pd.read_csv(args.hw_manifest)
    dh = dh[dh["split"] == "test"].copy()
    dp["domain"], dh["domain"] = "printed", "handwritten"
    cols = ["image_path", "class_idx", "class_name", "domain"]
    df = pd.concat([dp[cols], dh[cols]]).reset_index(drop=True)
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images not found, e.g. {missing[0]}\n"
                         "The manifests store absolute paths. Re-run prepare_data_hw.py, or use --data-root for the printed set.")
    print(f"Test images: printed {len(dp)}, handwritten {len(dh)}")

    crops = np.zeros((len(df), 224, 224), np.uint8)
    for i, p in enumerate(tqdm(df["image_path"], desc="preprocessing", unit="img")):
        crops[i] = pp.preprocess(p, out_size=224, margin_frac=ck["preprocess"]["margin_frac"])["crop"]
    probs = predict_batches(model, crops, dev)
    y, pred, conf = df["class_idx"].to_numpy(), probs.argmax(1), probs.max(1)
    df["pred_idx"], df["confidence"], df["correct"] = pred, conf, pred == y
    df["pred_name"] = [IDX_TO_NAME[int(i)] for i in pred]

    res = {"overall": metrics_block(y, pred)}
    for dom, g in df.groupby("domain"):
        res[dom] = metrics_block(g["class_idx"].to_numpy(), g["pred_idx"].to_numpy())
    refs = {"printed": reference("reports/eval_summary.json"), "handwritten": reference("reports/hw_eval_summary.json")}

    print(f"\nCOMBINED model, TEST accuracy: overall {res['overall']['accuracy']:.4f}  macro-F1 {res['overall']['macro_f1']:.4f}  (n={res['overall']['n']})")
    for dom in ("printed", "handwritten"):
        line = f"  {dom:12s} n={res[dom]['n']:5d}  accuracy {res[dom]['accuracy']:.4f}  macro-F1 {res[dom]['macro_f1']:.4f}"
        if refs[dom]:
            line += f"   | separate {dom} model: accuracy {refs[dom]['accuracy']:.4f}  macro-F1 {refs[dom]['macro_f1']:.4f}  (n={refs[dom]['n']})"
        print(line)

    names = [IDX_TO_NAME[i] for i in range(NUM_CLASSES)]
    per = pd.DataFrame(dict(index=range(NUM_CLASSES), name=names))
    for dom in ("printed", "handwritten"):
        g = df[df["domain"] == dom]
        per[f"{dom}_n"] = g.groupby("class_idx").size().reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
        per[f"{dom}_recall"] = g.groupby("class_idx")["correct"].mean().reindex(range(NUM_CLASSES)).to_numpy()
    Path("reports").mkdir(exist_ok=True)
    per.to_csv("reports/comb_eval_per_class.csv", index=False, encoding="utf-8-sig")

    for dom in ("printed", "handwritten"):
        g = df[(df["domain"] == dom) & (~df["correct"])]
        top = g.groupby(["class_name", "pred_name"]).size().reset_index(name="count").sort_values("count", ascending=False)
        top.columns = ["true", "predicted", "count"]
        top.to_csv(f"reports/comb_eval_top_confusions_{dom}.csv", index=False)
        print(f"\nMost common {dom} mistakes (true -> predicted):\n",
              top.head(8).to_string(index=False) if len(top) else "none")

    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmn, xticklabels=names, yticklabels=names, cmap="Blues", vmin=0, vmax=1, ax=ax,
                cbar_kws=dict(label="fraction of true class"))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Combined model, both test sets (accuracy {res['overall']['accuracy']:.4f})")
    fig.savefig("reports/comb_eval_confusion_matrix.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    df.assign(file=df["image_path"].map(lambda s: Path(s).name)).drop(columns=["image_path"]) \
      .to_csv("reports/comb_eval_predictions.csv", index=False, encoding="utf-8-sig")

    summary = dict(checkpoint=args.ckpt, checkpoint_best_epoch=ck["training"]["best_epoch"],
                   checkpoint_val_acc=ck["training"]["best_val_acc"], results=res,
                   separate_models_for_reference=refs, augmentation=ck["training"].get("augmentation"),
                   mean_confidence_correct=float(conf[df["correct"]].mean()),
                   mean_confidence_wrong=float(conf[~df["correct"]].mean()) if (~df["correct"]).any() else None,
                   note="Same test images as the printed and handwritten models; handwritten test = forms held out by form.")
    Path("reports/comb_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved results to reports/comb_eval_*.csv|png|json")


if __name__ == "__main__":
    main()
