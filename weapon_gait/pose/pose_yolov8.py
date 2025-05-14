"""
weapon_gait/pose/pose_yolov8.py
==============================
YOLOv8‑Pose wrapper (Ultralytics) that follows PoseExtractor API.
Runs on CPU; expect ~4–6 fps @ 480p.
Install: `pip install ultralytics>=8.1.0`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO

from .base import PoseExtractor

class YOLOv8Pose(PoseExtractor):
    """Ultralytics YOLOv8‑Pose single‑person extractor."""

    def __init__(self, model_name: str = "yolov8n-pose.pt", cache_dir: str | Path = "cache/poses_yolo"):
        self.model = YOLO(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return "yolov8-pose"

    def extract(self, video_path: Path) -> np.ndarray:
        cache_file = self.cache_dir / f"{video_path.stem}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        # batch inference
        results = list(self.model.predict(frames, device="cpu", verbose=False))
        pose_seq = []
        for res in results:
            if len(res.keypoints) == 0:
                pose_seq.append(np.full((17, 3), np.nan))
            else:
                k = res.keypoints[0].cpu().numpy()  # (17,3): x,y,conf
                pose_seq.append(k)
        arr = np.stack(pose_seq)  # (T,17,3)
        np.save(cache_file, arr)
        return arr