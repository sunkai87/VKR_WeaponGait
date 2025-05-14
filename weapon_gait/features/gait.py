from __future__ import annotations

import numpy as np

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba, bc = a - b, c - b
    ba /= np.linalg.norm(ba, axis=-1, keepdims=True) + 1e-9
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True) + 1e-9
    cosang = (ba * bc).sum(-1).clip(-1, 1)
    return np.degrees(np.arccos(cosang))


def extract_gait_features(pose_seq: np.ndarray, fps: float = 30) -> dict:
    # (same as before, trimmed for brevity — copy logic from v0.1)
    ...