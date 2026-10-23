"""Predict one HANDWRITTEN character image. Thin wrapper around predict.Predictor (which is unchanged).

    python src\\predict_hw.py path\\to\\image.png ...

From other code (e.g. app.py):
    from predict_hw import load_handwritten_predictor
    predictor = load_handwritten_predictor()      # raises FileNotFoundError with a clear message if not trained yet
    result = predictor.predict(image)             # same result dict as the printed model

Predictor loads whatever checkpoint it is given and checks that its class order equals labels.py,
so the handwritten and printed models share one label mapping. The printed model is not touched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp
from predict import load_predictor

HW_CKPT = Path(__file__).resolve().parent.parent / "models" / "handwritten_mobilenetv3_small_best.pt"


def load_handwritten_predictor(ckpt_path=HW_CKPT):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Handwritten model not found: {ckpt_path}\n"
            "Train it with  python src\\train_hw.py  (after python src\\prepare_data_hw.py), "
            "or copy the trained file to that path.")
    return load_predictor(ckpt_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    try:
        pr = load_handwritten_predictor()
    except FileNotFoundError as e:
        sys.exit(str(e))
    for a in args:
        try:
            r = pr.predict(a)
            print(f"{a}: {r['label']} (U+{ord(r['char']):04X}) confidence {r['confidence'] * 100:.1f}%"
                  f"{'  [LOW CONFIDENCE]' if r['low_confidence'] else ''}")
        except pp.InvalidImageError as e:
            print(f"{a}: rejected - {e}")
