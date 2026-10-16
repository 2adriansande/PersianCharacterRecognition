"""Batch augmentation on the GPU. Input x: float tensor (B,1,H,W), ink=1, background=0.
Only used on the training split."""
import math
import torch
import torch.nn.functional as F


def augment(x, strength=1.0):
    B, dev = x.size(0), x.device
    u = lambda: (torch.rand(B, device=dev) * 2 - 1)           # uniform in [-1, 1]
    ang = u() * math.radians(10) * strength
    sc = 1.0 + u() * 0.15 * strength                           # zoom 0.85 .. 1.15
    shear = u() * 0.15 * strength
    tx, ty = u() * 0.08 * strength, u() * 0.08 * strength
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([cos / sc, -sin / sc + shear, tx], 1),
        torch.stack([sin / sc, cos / sc, ty], 1)], 1)           # (B,2,3)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    r = torch.rand(B, 1, 1, 1, device=dev)                      # stroke thickness change
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < 0.2, dil, torch.where(r < 0.4, ero, x))

    blur = torch.rand(B, 1, 1, 1, device=dev) < 0.3             # mild blur
    x = torch.where(blur, F.avg_pool2d(x, 3, 1, 1), x)

    noise = torch.rand(B, 1, 1, 1, device=dev) < 0.3            # light noise
    x = x + noise * 0.04 * torch.randn_like(x)
    return x.clamp_(0, 1)
