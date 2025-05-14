from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from .base import PoseExtractor

mp_pose = mp.solutions.pose

class MediaPipePose(PoseExtractor):
    """MediaPipe BlazePose CPU implementation."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir or "cache/poses_mp")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "mediapipe"

    def extract(self, video_path: Path) -> np.ndarray:
        cache_file = self.cache_dir / (video_path.stem + ".npy")
        if cache_file.exists():
            return np.load(cache_file)

        cap = cv2.VideoCapture(str(video_path))
        lms = []
        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          enable_segmentation=False, smooth_landmarks=True) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                if res.pose_landmarks:
                    lms.append(np.array([[p.x, p.y, p.z] for p in res.pose_landmarks.landmark]))
                else:
                    lms.append(np.full((33, 3), np.nan))
        cap.release()
        # arr = np.stack(lms)
        # np.save(cache_file, arr)
        # return arr
        if not lms:                       # детектор не нашёл ни одного кадра
            print("MediaPipe: no landmarks →", video_path.name)
            arr = np.full((1, 33, 3), np.nan)        # 1×33×3 NaN-заглушка
        else:
            arr = np.stack(lms)                      # (T,33,3)

        np.save(cache_file, arr)
        return arr