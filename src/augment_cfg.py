"""Configurable training-time augmentation (GPU, per batch, nothing is saved to disk).

Same idea as augment.py (which is unchanged and still used by train.py / train_hw.py), but every probability
and range is a field of AugConfig and can be set from the command line (--aug-rotate-deg 8, --aug-p-blur 0.1, ...).

Input x: float tensor (B,1,H,W), ink = 1, background = 0 (the binary crops the model is trained on).
Never flips horizontally (it would change letter identity).

Brightness / contrast / background variation are deliberately NOT included: the pipeline binarizes every image
(Otsu) before the model sees it, so they would have almost no effect on the model input.

Safety check: after the geometric step, a sample whose ink reaches the outer edge of the image (a sign that a stroke or
dot was cut off) is put back to its un-augmented version. The preprocessing leaves a ~10% margin around the
character, so with the default ranges this should be rare; preview_augmentation.py reports how often it happens.
"""
import math
from dataclasses import dataclass, fields, asdict

import torch
import torch.nn.functional as F


@dataclass
class AugConfig:
    strength: float = 1.0        # multiplies the geometric ranges below (0 = off, 1 = as written)
    p_geometric: float = 1.0     # chance that a sample gets the rotation / zoom / shift / shear step
    rotate_deg: float = 10.0     # rotation in [-rotate_deg, +rotate_deg] degrees
    scale_min: float = 0.90      # zoom factor range (>1 = zoom in)
    scale_max: float = 1.10
    shift_frac: float = 0.05     # shift up to this fraction of the image width / height
    shear: float = 0.08          # small slant (0 = off)
    p_crop: float = 0.15         # chance of an extra conservative zoom-in ("crop")
    crop_max: float = 0.06       # extra zoom-in of up to 6%
    edge_px: int = 3             # width of the border ring used by the ink-clipping safety check (0 = off)
    p_thicken: float = 0.20      # chance of thicker strokes (3x3 dilation)
    p_thin: float = 0.15         # chance of thinner strokes (3x3 erosion); may weaken small dots
    p_blur: float = 0.30         # chance of a mild blur
    p_noise: float = 0.30        # chance of light pixel noise
    noise_std: float = 0.04


def add_aug_args(ap):
    """Adds --aug-<field> options for every AugConfig field to an argparse parser."""
    g = ap.add_argument_group("augmentation (see src/augment_cfg.py)")
    for f in fields(AugConfig):
        g.add_argument(f"--aug-{f.name.replace('_', '-')}", dest=f"aug_{f.name}",
                       type=type(f.default), default=f.default,
                       help=f"default {f.default}")


def cfg_from_args(args):
    return AugConfig(**{f.name: getattr(args, f"aug_{f.name}") for f in fields(AugConfig)})


def cfg_to_dict(cfg):
    return asdict(cfg)


def augment_configurable(x, cfg=None, stats=None):
    """Returns an augmented copy of x. `stats` (optional dict) collects counts for the preview script."""
    cfg = cfg or AugConfig()
    B, dev = x.size(0), x.device
    s = cfg.strength
    rnd = lambda: torch.rand(B, device=dev)
    u = lambda: rnd() * 2 - 1                                   # uniform in [-1, 1]

    # ---- geometry: rotation, zoom (+ optional extra crop zoom), shift, shear
    ang = u() * math.radians(cfg.rotate_deg) * s
    lo, hi = 1 - (1 - cfg.scale_min) * s, 1 + (cfg.scale_max - 1) * s
    sc = lo + (hi - lo) * rnd()
    crop = (rnd() < cfg.p_crop).float()
    sc = sc * (1 + crop * rnd() * cfg.crop_max * s)
    shear = u() * cfg.shear * s
    tx, ty = u() * 2 * cfg.shift_frac * s, u() * 2 * cfg.shift_frac * s   # grid coordinates span [-1, 1] = 2 units
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([cos / sc, -sin / sc + shear, tx], 1),
        torch.stack([sin / sc, cos / sc, ty], 1)], 1)           # (B,2,3)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    geo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    want = rnd() < cfg.p_geometric
    if cfg.edge_px > 0:
        k = cfg.edge_px
        ring = torch.zeros(x.shape[-2:], dtype=torch.bool, device=dev)
        ring[:k] = True; ring[-k:] = True; ring[:, :k] = True; ring[:, -k:] = True
        clipped = (geo * ring).amax(dim=(1, 2, 3)) > 0.5         # ink reached the border -> probably cut off
    else:
        clipped = torch.zeros(B, dtype=torch.bool, device=dev)
    if stats is not None:
        stats["n"] = stats.get("n", 0) + B
        stats["reverted_for_clipping"] = stats.get("reverted_for_clipping", 0) + int((want & clipped).sum().item())
    x = torch.where((want & ~clipped).view(B, 1, 1, 1), geo, x)

    # ---- stroke thickness (mutually exclusive), blur, noise
    r = rnd().view(B, 1, 1, 1)
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < cfg.p_thicken, dil, torch.where(r < cfg.p_thicken + cfg.p_thin, ero, x))
    blur = (rnd() < cfg.p_blur).view(B, 1, 1, 1)
    x = torch.where(blur, F.avg_pool2d(x, 3, 1, 1), x)
    noise = (rnd() < cfg.p_noise).view(B, 1, 1, 1)
    x = x + noise * cfg.noise_std * torch.randn_like(x)
    return x.clamp(0, 1)
