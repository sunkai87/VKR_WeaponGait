"""
weapon_gait/pose/pifpaf_pose.py
===============================
CPU‑ориентированный экстрактор поз на базе **OpenPifPaf**.

* 17 keypoints (COCO format)
* Без TensorFlow/protobuf, только PyTorch.
* Стабильно ставится:    pip install openpifpaf==0.13.10  (PyTorch ≥ 1.12)

Usage in CLI
------------
```
python -m weapon_gait.cli extract --video videos/w_022.mp4 --backend pifpaf
```
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import openpifpaf

from .base import PoseExtractor


class PifPafPose(PoseExtractor):
    """OpenPifPaf ResNet50 + decoding on CPU."""

    def __init__(self, cache_dir: str | Path = "cache/poses_pifpaf"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Predictor автоматически скачает веса resnet50.
        self.predictor = openpifpaf.Predictor(checkpoint="resnet50")

    # friendly name
    @property
    def name(self) -> str:
        return "pifpaf"

    # ------------------------------------------------------------------
    def extract(self, video_path: Path) -> np.ndarray:
        out = self.cache_dir / f"{video_path.stem}.npy"
        if out.exists():
            return np.load(out)

        cap = cv2.VideoCapture(str(video_path))
        keypoints_seq: List[np.ndarray] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds, _, _ = self.predictor.numpy_image(rgb)
            if preds:
                # берём первое обнаруженное тело
                kps = preds[0].data  # (17,3) — x, y, confidence
                # PifPaf уже возвращает COCO‑порядок
                kp = np.stack([
                    kps[:, 0],  # x
                    kps[:, 1],  # y
                    kps[:, 2],  # conf
                ], axis=-1)
                kp[kp[:, 2] < 0.3] = np.nan
            else:
                kp = np.full((17, 3), np.nan)
            keypoints_seq.append(kp)

        cap.release()
        arr = np.stack(keypoints_seq)
        np.save(out, arr)
        return arr