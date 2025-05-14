"""
weapon_gait/models/agcn_pipeline.py
==================================
Wrapper utilities for **2S‑AGCN (Two‑Stream Adaptive Graph Convolution
Network)**, one of the top graph‑based skeleton action recognisers. 2S
интегрирует spatial‑temporal поток поз + поток bone‑vectors, что часто
даёт +2‑3 %mAP vs. ST‑GCN.

Используем готовые конфиги MMAction2 → `agcn_2s_*_keypoint.py`.

API ровно такой же, как у posec3d/stgcn wrappers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import random

import joblib
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner

# ---------------------------------------------------------------------------
# Dataset conversion (тот же формат)
# ---------------------------------------------------------------------------

def _convert_clip(pose_arr: np.ndarray, label: int) -> Dict:
    T, J, _ = pose_arr.shape
    xy = pose_arr[:, :, :2]
    conf = (~np.isnan(xy[..., 0])).astype(np.float32)
    xy = np.nan_to_num(xy, nan=0.0)
    data = np.stack([xy[..., 0], xy[..., 1], conf], axis=-1)  # (T, J, 3)
    return {"keypoint": data[None, ...], "label": label}


def build_agcn_dataset(
    manifest_csv: str | Path,
    pose_cache: str | Path,
    out_dir: str | Path,
    split_ratio: float = 0.2,
):
    df = pd.read_csv(manifest_csv)
    lbl_map = {l: i for i, l in enumerate(sorted(df["label"].unique()))}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, val = [], []
    for _, row in df.iterrows():
        p = Path(pose_cache) / (Path(row["video"]).stem + ".npy")
        arr = np.load(p)
        sample = _convert_clip(arr, lbl_map[row["label"]])
        (val if random.random() < split_ratio else tr).append(sample)
    joblib.dump(tr, out_dir / "train.pkl")
    joblib.dump(val, out_dir / "val.pkl")
    with open(out_dir / "label_map.txt", "w", encoding="utf-8") as f:
        for k, v in lbl_map.items():
            f.write(f"{v} {k}\n")
    print(f"AGCN dataset saved in {out_dir}: train {len(tr)} | val {len(val)}")

# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------

AGCN_BASE_CFG = "agcn/agcn_2s_100e_ntu120_xsub_keypoint.py"


def render_config(
    num_classes: int,
    data_root: str | Path,
    work_dir: str | Path,
    base_cfg: str | Path = AGCN_BASE_CFG,
) -> Path:
    cfg = Config.fromfile(base_cfg)
    cfg.work_dir = str(work_dir)
    cfg.model.cls_head.num_classes = num_classes
    for split in ("train", "val", "test"):
        cfg[f"{split}_dataloader"].dataset.ann_file = str(Path(data_root) / f"{split}.pkl")
        cfg[f"{split}_dataloader"].dataset.data_prefix = str(data_root)
    out_cfg = Path(work_dir) / "agcn_weapon.py"
    cfg.dump(out_cfg)
    print("Config written →", out_cfg)
    return out_cfg

# ---------------------------------------------------------------------------
# Train / Eval wrappers
# ---------------------------------------------------------------------------

def train_agcn(cfg_path: str | Path, max_epochs: int = 100):
    cfg = Config.fromfile(cfg_path)
    cfg.train_cfg.max_epochs = max_epochs
    runner = Runner.from_cfg(cfg)
    runner.train()


def evaluate_agcn(cfg_path: str | Path, ckpt: str | Path):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = str(ckpt)
    runner = Runner.from_cfg(cfg)
    runner.test()