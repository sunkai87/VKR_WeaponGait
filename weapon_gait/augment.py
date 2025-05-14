"""
weapon_gait/augment.py
=====================
Simple, deterministic‑enough pose‑level augmentations to up‑sample scarce
classes (e.g., *weapon*). Designed to work offline **before** feature
extraction.  Works only with 2‑D (x, y, z) landmark arrays produced by our
pose extractors.

Augmentations implemented
-------------------------
1. **jitter_xy**      – small Gaussian noise ±σ pixels to every (x, y)
2. **time_warp**      – random stretch/compress (resample to 90‑110 % len).
3. **horizontal_flip**– mirror along x‑axis (if camera frontal).

`augment_sequence` applies a random subset (configurable) and returns the
new sequence; original left intact.
"""
from __future__ import annotations

import random
from typing import Sequence, Callable, List

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def jitter_xy(seq: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    jitter = np.random.normal(scale=sigma, size=seq[..., :2].shape)
    seq_aug = seq.copy()
    seq_aug[..., :2] += jitter
    return seq_aug


def horizontal_flip(seq: np.ndarray, img_width: int | None = None) -> np.ndarray:
    """Flip x‑coords around centre or given width."""
    seq_aug = seq.copy()
    if img_width is None:
        # assume coords normalised 0‑1
        seq_aug[..., 0] = 1.0 - seq_aug[..., 0]
    else:
        seq_aug[..., 0] = img_width - seq_aug[..., 0]
    return seq_aug


def time_warp(seq: np.ndarray, factor_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    factor = random.uniform(*factor_range)
    T, J, C = seq.shape
    t_in = np.linspace(0, 1, T)
    t_out = np.linspace(0, 1, int(T * factor))
    f = interp1d(t_in, seq, axis=0, kind="linear")
    return f(t_out)

# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def augment_sequence(
    seq: np.ndarray,
    transforms: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
) -> np.ndarray:
    if transforms is None:
        transforms = [jitter_xy, time_warp]
    seq_aug = seq
    for fn in transforms:
        seq_aug = fn(seq_aug)
    return seq_aug

# ---------------------------------------------------------------------------
# Generator that yields N augmented versions per original
# ---------------------------------------------------------------------------

def generate_augmented(seq: np.ndarray, n: int = 3) -> List[np.ndarray]:
    out = []
    for _ in range(n):
        out.append(augment_sequence(seq))
    return out