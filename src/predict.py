"""Phase 5: load the saved model on its own (no training code) and predict one character image.

    python src\\predict.py --export-config        # writes models/model_config.json
    python src\\predict.py --selftest             # checks the loaded model on 64 TEST images (2 per class)
    python src\\predict.py path\\to\\image.png ... # predict your own images

From other code (e.g. app.py):
    from predict import load_predictor
    predictor = load_predictor()
    result = predictor.predict(image)   # path, PIL.Image or numpy array
    result -> dict(label="alef", char="ا", codepoint="U+0627", confidence=0.98,
                   top3=[...], low_confidence=False, warnings=[...])

Confidence is the model's softmax probability for its top class. It is NOT the accuracy.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp
from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES
from model import build_model

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "models" / "mobilenetv3_small_best.pt"
LOW_CONFIDENCE = 0.70   # warning threshold only; not tuned or validated (see Phase 7)


class Predictor:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location=self.dev, weights_only=False)
        if ck.get("arch") != "mobilenet_v3_small" or ck.get("num_classes") != NUM_CLASSES:
            raise ValueError("Checkpoint is not a 32-class MobileNetV3 Small model.")
        # class order comes from the checkpoint itself and must agree with labels.py
        self.idx_to_char = {int(k): v for k, v in ck["idx_to_char"].items()}
        self.idx_to_name = {int(k): v for k, v in ck["idx_to_name"].items()}
        if self.idx_to_char != IDX_TO_CHAR or self.idx_to_name != IDX_TO_NAME:
            raise ValueError("Class mapping in checkpoint differs from src/labels.py.")
        self.prep = ck["preprocess"]
        self.training = ck["training"]
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.dev).eval()

    @staticmethod
    def _to_gray_array(image):
        if isinstance(image, Image.Image):          # same white-background handling as pp.load_image
            im = image.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            return np.array(Image.alpha_composite(bg, im).convert("L"))
        return image                                # path or ndarray: pp.load_image handles it

    @torch.no_grad()
    def predict(self, image, low_confidence=LOW_CONFIDENCE):
        """Raises preprocess.InvalidImageError for unreadable, unsupported or blank images."""
        res = pp.preprocess(self._to_gray_array(image), out_size=self.prep["out_size"],
                            margin_frac=self.prep["margin_frac"])
        x = torch.from_numpy(pp.to_model_tensor_array(res["crop"]))[None].to(self.dev)
        p = torch.softmax(self.model(x), 1)[0].cpu().numpy()
        order = p.argsort()[::-1]
        i = int(order[0])
        return dict(label=self.idx_to_name[i], char=self.idx_to_char[i], index=i,
                    codepoint=f"U+{ord(self.idx_to_char[i]):04X}", confidence=float(p[i]),
                    top3=[dict(label=self.idx_to_name[int(j)], char=self.idx_to_char[int(j)],
                               confidence=float(p[j])) for j in order[:3]],
                    low_confidence=bool(p[i] < low_confidence), warnings=res["warnings"],
                    crop=res["crop"])


_CACHE = {}
def load_predictor(ckpt_path=DEFAULT_CKPT):
    """Loads the model once per process and reuses it."""
    key = str(ckpt_path)
    if key not in _CACHE:
        _CACHE[key] = Predictor(ckpt_path)
    return _CACHE[key]


def export_config(ckpt_path=DEFAULT_CKPT, out=None):
    """Write the readable configuration that lives inside the checkpoint to models/model_config.json."""
    pr = load_predictor(ckpt_path)
    cfg = dict(
        architecture=dict(name="mobilenet_v3_small", num_classes=NUM_CLASSES,
                          head="classifier[3] = Linear(1024, 32)", checkpoint=Path(ckpt_path).name),
        class_to_index={pr.idx_to_name[i]: i for i in range(NUM_CLASSES)},
        label_to_unicode={pr.idx_to_name[i]: dict(char=pr.idx_to_char[i], codepoint=f"U+{ord(pr.idx_to_char[i]):04X}")
                          for i in range(NUM_CLASSES)},
        preprocessing=dict(pr.prep, polarity="auto-detected, crop is dark ink on white, square padding, no stretching"),
        training=pr.training)
    out = Path(out or Path(ckpt_path).with_name("model_config.json"))
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def selftest(per_class=2):
    import pandas as pd
    pr = load_predictor()
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "manifest.csv")
    df = df[(df["split"] == "test") & (~df["glitch"].astype(bool))]
    df = df.groupby("class_idx", group_keys=False).head(per_class)
    ok = 0
    for _, r in df.iterrows():
        out = pr.predict(r["image_path"])
        assert out["char"] in IDX_TO_CHAR.values() and 0.0 <= out["confidence"] <= 1.0
        ok += out["index"] == r["class_idx"]
    print(f"Loaded checkpoint OK. {ok}/{len(df)} sample images correct; all outputs valid "
          f"(class, Persian character, confidence in [0,1]).")
    print("This is a loading/sanity check on 64 images, NOT the evaluation result (see reports/eval_summary.json).")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--export-config" in args:
        print("Wrote", export_config())
    elif "--selftest" in args:
        selftest()
    elif args:
        pr = load_predictor()
        for a in args:
            try:
                r = pr.predict(a)
                print(f"{a}: {r['label']} (U+{ord(r['char']):04X}) confidence {r['confidence']*100:.1f}%"
                      f"{'  [LOW CONFIDENCE]' if r['low_confidence'] else ''}")
            except pp.InvalidImageError as e:
                print(f"{a}: rejected - {e}")
    else:
        print(__doc__)
