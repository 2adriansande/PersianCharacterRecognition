"""Image validation, character detection/cropping, resize and normalisation.

- Crop uses the union bounding box of ALL ink components above a tiny area, so dots
  (nuqta) and detached strokes stay inside the crop.
- Square padding keeps aspect ratio (no stretching).
- Polarity is auto-detected. Output crop is DARK ink on WHITE, out_size x out_size.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
MIN_SIDE, MAX_SIDE = 16, 8000
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InvalidImageError(ValueError):
    pass


def load_image(path_or_array):
    """Return a grayscale uint8 array. Validates format and size."""
    if isinstance(path_or_array, (str, Path)):
        p = Path(path_or_array)
        if p.suffix.lower() not in ALLOWED_EXT:
            raise InvalidImageError(f"Unsupported format '{p.suffix}'. Use PNG, JPG, JPEG or BMP.")
        if not p.exists():
            raise InvalidImageError(f"File not found: {p}")
        try:
            with Image.open(p) as im:
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                arr = np.array(Image.alpha_composite(bg, im).convert("L"))  # transparent -> white
        except Exception as e:
            raise InvalidImageError(f"Cannot read image: {e}")
    else:
        arr = np.asarray(path_or_array)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise InvalidImageError(f"Image size {w}x{h} outside [{MIN_SIDE}, {MAX_SIDE}] px.")
    return arr


def binarize(gray):
    """Otsu threshold -> ink mask (255 = ink). Polarity auto-detected from border pixels."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:  # border is "ink" -> background was dark, flip
        mask = 255 - mask
    return mask


def find_character_bbox(mask, min_area_frac=1e-4):
    """Union bbox of all ink components (keeps dots/strokes). (x0,y0,x1,y1) or None."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, int(min_area_frac * mask.size))
    boxes = [(s[0], s[1], s[0] + s[2], s[1] + s[3]) for s in stats[1:] if s[4] >= min_area]
    if not boxes:
        return None
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def preprocess(path_or_array, out_size=224, margin_frac=0.12, min_ink_frac=0.002):
    """Full pipeline -> dict(original, mask, crop, bbox, warnings). Raises InvalidImageError."""
    gray = load_image(path_or_array)
    mask = binarize(gray)
    if (mask > 0).mean() < min_ink_frac:
        raise InvalidImageError("Image appears blank (no character found).")
    bbox = find_character_bbox(mask)
    if bbox is None:
        raise InvalidImageError("No character found in image.")
    x0, y0, x1, y1 = bbox
    warnings = []
    if (mask > 0).mean() > 0.6:
        warnings.append("Very high ink coverage; image may be noisy.")
    if x0 <= 1 or y0 <= 1 or x1 >= mask.shape[1] - 1 or y1 >= mask.shape[0] - 1:
        warnings.append("Character touches image border; strokes may be clipped.")
    glyph = mask[y0:y1, x0:x1]
    h, w = glyph.shape
    side = int(round(max(h, w) * (1 + 2 * margin_frac)))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = glyph
    resized = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return dict(original=gray, mask=mask, crop=255 - resized, bbox=bbox, warnings=warnings)


def to_model_tensor_array(crop):
    """uint8 HxW -> float32 3xHxW normalised with ImageNet stats (for MobileNetV3)."""
    rgb = np.stack([crop] * 3, axis=-1).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def show_original_and_crop(result, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(result["original"], cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(result["crop"], cmap="gray"); ax[1].set_title("Processed crop"); ax[1].axis("off")
    if title:
        fig.suptitle(title)
    plt.show()
