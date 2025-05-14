"""
weapon_gait/models/stgcn_pipeline.py
===================================
Wrapper utilities for **ST‑GCN (Spatial‑Temporal Graph CNN)** via
MMAction2.  API и структура полностью совпадают с PoseC3D‑pipeline, так
что переключение backend'а требует лишь замену модуля.

Prerequisites
-------------
* Установлен MMAction2 ≥ 1.2.0 (тот же, что и для PoseC3D).
* CUDA‑GPU доступен (ST‑GCN тоже хотелось бы обучать на GPU; CPU будет
  крайне медленным).

Использование
-------------
```python
from weapon_gait.models.stgcn_pipeline import (
    build_stgcn_dataset, render_config,
    train_stgcn, evaluate_stgcn,
)

build_stgcn_dataset(
    manifest_csv="data/manifest.csv",
    pose_cache="cache/poses_mp",
    out_dir="data/stgcn",
)

cfg = render_config(
    num_classes=2,
    data_root="data/stgcn",
    work_dir="runs/stgcn",
)

train_stgcn(cfg, max_epochs=80)

evaluate_stgcn(cfg, ckpt="runs/stgcn/epoch_80.pth")
```
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import random

import joblib
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner

# ---------------------------------------------------------------------------
# Dataset conversion (same format as PoseC3D) — reuse logic
# ---------------------------------------------------------------------------

def _convert_clip(pose_arr: np.ndarray, label: int) -> Dict:
    T, J, _ = pose_arr.shape
    xy = pose_arr[:, :, :2]
    conf = (~np.isnan(xy[..., 0])).astype(np.float32)
    xy = np.nan_to_num(xy, nan=0.0)
    data = np.stack([xy[..., 0], xy[..., 1], conf], axis=-1)  # (T, J, 3)
    return {"keypoint": data[None, ...], "label": label}  # (1, T, J, 3)


def build_stgcn_dataset(
    manifest_csv: str | Path,
    pose_cache: str | Path,
    out_dir: str | Path,
    split_ratio: float = 0.2,
):
    df = pd.read_csv(manifest_csv)
    label_map = {l: i for i, l in enumerate(sorted(df["label"].unique()))}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for _, row in df.iterrows():
        pose_file = Path(pose_cache) / (Path(row["video"]).stem + ".npy")
        arr = np.load(pose_file)
        sample = _convert_clip(arr, label_map[row["label"]])
        (val if random.random() < split_ratio else train).append(sample)

    joblib.dump(train, out_dir / "train.pkl")
    joblib.dump(val, out_dir / "val.pkl")

    with open(out_dir / "label_map.txt", "w", encoding="utf-8") as f:
        for lbl, idx in label_map.items():
            f.write(f"{idx} {lbl}\n")

    print(f"Train {len(train)} | Val {len(val)} samples saved in {out_dir}")

# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------

STGCN_BASE_CFG = "stgcn/stgcn_80e_ntu120_xsub_keypoint.py"  # path inside MMAction2 configs


def render_config(
    num_classes: int,
    data_root: str | Path,
    work_dir: str | Path,
    base_cfg: str | Path = STGCN_BASE_CFG,
) -> Path:
    cfg = Config.fromfile(base_cfg)
    cfg.work_dir = str(work_dir)
    cfg.model.cls_head.num_classes = num_classes
    for split in ("train", "val", "test"):
        cfg[f"{split}_dataloader"].dataset.ann_file = str(Path(data_root) / f"{split}.pkl")
        cfg[f"{split}_dataloader"].dataset.data_prefix = str(data_root)

    out_path = Path(work_dir) / "stgcn_weapon.py"
    cfg.dump(out_path)
    print(f"Config dumped → {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# Train & Eval wrappers
# ---------------------------------------------------------------------------

def train_stgcn(cfg_path: str | Path, max_epochs: int = 80):
    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_epochs = max_epochs
    runner = Runner.from_cfg(cfg)
    runner.train()


def evaluate_stgcn(cfg_path: str | Path, ckpt: str | Path):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = str(ckpt)
    runner = Runner.from_cfg(cfg)
    runner.test()
