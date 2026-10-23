"""Predict one character image with the COMBINED (printed + handwritten) model. Thin wrapper around predict.Predictor.

    python src\\predict_combined.py path\\to\\image.png ...

From other code:
    from predict_combined import load_combined_predictor
    predictor = load_combined_predictor()          # FileNotFoundError with a clear message if not trained yet
    result = predictor.predict(image)              # same result dict as the other models
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp
from predict import load_predictor

COMBINED_CKPT = Path(__file__).resolve().parent.parent / "models" / "combined_mobilenetv3_small_best.pt"


def load_combined_predictor(ckpt_path=COMBINED_CKPT):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Combined model not found: {ckpt_path}\n"
            "Train it with  python src\\train_combined.py, or copy the trained file to that path.")
    return load_predictor(ckpt_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    try:
        pr = load_combined_predictor()
    except FileNotFoundError as e:
        sys.exit(str(e))
    for a in args:
        try:
            r = pr.predict(a)
            print(f"{a}: {r['label']} (U+{ord(r['char']):04X}) confidence {r['confidence'] * 100:.1f}%"
                  f"{'  [LOW CONFIDENCE]' if r['low_confidence'] else ''}")
        except pp.InvalidImageError as e:
            print(f"{a}: rejected - {e}")
