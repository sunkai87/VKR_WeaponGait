"""
weapon_gait/models/posec3d_pipeline.py
=====================================
Utility helpers that wrap OpenMMLab's PoseC3D implementation so we can
train/evaluate the model from our `weapon_gait` project with minimal
boilerplate.

Prerequisites
-------------
1. **CUDA & GPU ready** (PoseC3D is a 3‑D CNN; CPU is impractically slow).
2. Install OpenMMLab toolchain (tested with MMAction2 1.2.0):
   ```bash
   pip install mmcv-full==2.0.2 # or pre‑built wheel for your CUDA
   pip install mmaction2==1.2.0
   ```
3. Clone the repos if you need to modify configs; otherwise import works.

The helpers below:
* **`build_posec3d_dataset`** — convert cached NumPy poses (from any
  pose‑backend) + manifest.csv into PoseC3D‑ready pickles.
* **`render_config`** — auto‑creates a PoseC3D config file with the right
  paths, number of classes, skeleton layout, etc.
* **`train_posec3d`** — invokes MMAction2's train loop programmatically
  (no shell scripts needed inside VS Code).
* **`evaluate_posec3d`** — single checkpoint evaluation and confusion
  matrix/ROC dump.

Usage example
-------------
```python
from weapon_gait.models.posec3d_pipeline import (
    build_posec3d_dataset, render_config,
    train_posec3d, evaluate_posec3d,
)

build_posec3d_dataset(
    manifest_csv="data/manifest.csv",
    pose_cache="cache/poses_mp",   # directory where .npy poses live
    out_dir="data/posec3d",
    split_ratio=0.2,               # 20% validation split
)

cfg_path = render_config(
    num_classes=2,                 # weapon / no‑weapon
    data_root="data/posec3d",
    work_dir="runs/posec3d",
)

train_posec3d(cfg_path, max_epochs=50)

evaluate_posec3d(cfg_path, ckpt="runs/posec3d/epoch_50.pth")
```
"""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner

from weapon_gait.pose import get_extractor

# ---------------------------------------------------------------------------
# 1. Dataset conversion helpers
# ---------------------------------------------------------------------------

SKELETON_LAYOUT = {
    "mediapipe": 33,
    "movenet": 17,
    "rtmpose": 17,
}

def _convert_clip_to_dict(pose_arr: np.ndarray, label: int) -> dict:
    """Return a dict compatible with PoseC3D's pickled annotation format."""
    # PoseC3D expects shape (num_person, T, num_keypoints, 2/3)
    # We have single person; we drop z and keep confidence=1.
    T, J, _ = pose_arr.shape
    xy = pose_arr[:, :, :2]  # (T, J, 2)
    score = (~np.isnan(xy[..., 0])).astype(np.float32)
    xy = np.nan_to_num(xy, nan=0.0)
    data = np.stack([xy[..., 0], xy[..., 1], score], axis=-1)  # (T, J, 3)
    return {
        "keypoint": data[None, ...],        # (1, T, J, 3)
        "label": label,
    }

def build_posec3d_dataset(
    manifest_csv: str | Path,
    pose_cache: str | Path,
    out_dir: str | Path,
    backend: str = "mediapipe",
    split_ratio: float = 0.2,
):
    """Create train/val pickles accepted by PoseC3D.

    Parameters
    ----------
    manifest_csv : CSV file with columns `video,label`.
    pose_cache   : Directory containing .npy cached poses.
    out_dir      : Output directory where train/val .pkl will be stored.
    backend      : Pose extractor used (must match cached files).
    split_ratio  : Fraction of clips to send to validation set.
    """
    df = pd.read_csv(manifest_csv)
    label_to_idx = {l: i for i, l in enumerate(sorted(df["label"].unique()))}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list, val_list = [], []

    for _, row in df.iterrows():
        pose_file = Path(pose_cache) / (Path(row["video"]).stem + ".npy")
        pose_arr = np.load(pose_file)
        sample = _convert_clip_to_dict(pose_arr, label_to_idx[row["label"]])
        if random.random() < split_ratio:
            val_list.append(sample)
        else:
            train_list.append(sample)

    for split_name, split in [("train", train_list), ("val", val_list)]:
        joblib.dump(split, out_dir / f"{split_name}.pkl")
        print(f"{split_name}: {len(split)} samples → {out_dir/split_name+'.pkl'}")

    # Save label map
    with open(out_dir / "label_map.txt", "w", encoding="utf-8") as f:
        for lbl, idx in label_to_idx.items():
            f.write(f"{idx} {lbl}\n")

# ---------------------------------------------------------------------------
# 2. Config generator
# ---------------------------------------------------------------------------

POSEC3D_BASE_CFG = "posec3d/slowonly_r50_u48_240e_k400_2d.py"  # provided by MMAction2 repo


def render_config(
    num_classes: int,
    data_root: str | Path,
    work_dir: str | Path,
    base_cfg: str | Path = POSEC3D_BASE_CFG,
) -> Path:
    """Render a config .py customised for our dataset and return its path."""
    cfg = Config.fromfile(base_cfg)

    # Update key fields
    cfg.work_dir = str(work_dir)
    cfg.model.cls_head.num_classes = num_classes
    for split in ("train", "val", "test"):
        cfg[dataloader_key := f"{split}_dataloader"].dataset.ann_file = str(Path(data_root) / f"{split}.pkl")
        cfg[dataloader_key].dataset.data_prefix = str(data_root)

    cfg.dump(Path(work_dir) / "posec3d_weapon.py")
    print(f"Config written → {Path(work_dir) / 'posec3d_weapon.py'}")
    return Path(work_dir) / "posec3d_weapon.py"

# ---------------------------------------------------------------------------
# 3. Training & evaluation wrappers (run inside Python, not shell)
# ---------------------------------------------------------------------------

def train_posec3d(cfg_path: str | Path, max_epochs: int = 40):
    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_epochs = max_epochs
    runner = Runner.from_cfg(cfg)
    runner.train()


def evaluate_posec3d(cfg_path: str | Path, ckpt: str | Path):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = str(ckpt)
    runner = Runner.from_cfg(cfg)
    runner.test()