"""
weapon_gait/pose/movenet_pose.py  (working CPU implementation)
-------------------------------------------------------------
Uses TensorFlow (>=2.10) to load Google MoveNet SinglePose *Lightning*
(TFLite) model.  Works on Windows 11 + CPU; no CUDA требуется.

Установка зависимостей (Python 3.10):
------------------------------------
```
pip install tensorflow==2.13.0        # ~300 MB, CPU‑only wheel
pip install tensorflow-hub==0.15.0     # auto‑download модели
```
> *Если вы не хотите ставить полный TensorFlow*, можно собрать "легкий"
> tflite-runtime, но для Windows проще поставить полный TF‑wheel.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .base import PoseExtractor

# ---------------------------------------------------------------------------
# Helper to build interpreter once and reuse
# ---------------------------------------------------------------------------

_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"


class _InterpreterSingleton:
    _instance: tf.lite.Interpreter | None = None

    @classmethod
    def get(cls) -> tf.lite.Interpreter:
        if cls._instance is None:
            model = hub.load(_MODEL_URL)
            concrete_func = model.signatures["serving_default"]
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            cls._instance = tf.lite.Interpreter(model_content=tflite_model)
            cls._instance.allocate_tensors()
        return cls._instance


# MoveNet expects input 192×192 RGB
_IMG_SIZE = 192


class MoveNetPose(PoseExtractor):
    """MoveNet Lightning CPU implementation (17 keypoints)."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir or "cache/poses_movenet")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # friendly name used in factory
    @property
    def name(self) -> str:
        return "movenet"

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def extract(self, video_path: Path) -> np.ndarray:
        cache = self.cache_dir / (video_path.stem + ".npy")
        if cache.exists():
            return np.load(cache)

        interpreter = _InterpreterSingleton.get()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        cap = cv2.VideoCapture(str(video_path))
        keypoints_seq: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img_resized = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE))
            # input_tensor = tf.convert_to_tensor(img_resized, dtype=tf.uint8)[tf.newaxis, ...]

            img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE))

            # → float32
            input_tensor = tf.cast(img_resize, tf.int32)[tf.newaxis, ...]
            # если нужно нормализовать 0-255 → 0-1:
            # input_tensor = input_tensor / 255.0

            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])[0, 0]
            # keypoints_with_scores: (17, 3) → y, x, score
            # convert to x, y relative (0‑1), z=0 placeholder
            xy = keypoints_with_scores[:, [1, 0]]
            kp = np.concatenate([xy, np.zeros((17, 1))], axis=-1)
            thresh = 0.2
            kp[keypoints_with_scores[:, 2] < thresh] = np.nan
            keypoints_seq.append(kp)

        cap.release()
        arr = np.stack(keypoints_seq)  # (T, 17, 3)
        np.save(cache, arr)
        return arr