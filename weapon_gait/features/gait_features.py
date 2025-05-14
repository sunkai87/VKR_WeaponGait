"""
weapon_gait/features/gait_features.py
====================================
Multiple strategies to transform a raw pose sequence (T, J, 3) into a
feature vector or a spatio‑temporal tensor suitable for a downstream
classifier.

Feature‑extractor classes implement:
    extract(pose: np.ndarray, fps: float) -> Any

Two categories:
* **Vector extractors** → return 1‑D feature vector (hand‑crafted stats).
* **Sequence extractors** → return (T, F) or graph tensors for DL models.

Add new extractors by inheriting `BaseFeatureExtractor` and registering
in `get_extractor` factory.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from scipy.fft import dct


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseFeatureExtractor(abc.ABC):
    name: str  # identifier used in CLI/model meta

    @abc.abstractmethod
    def extract(self, pose_seq: np.ndarray, fps: float = 30) -> Any:
        """Return features (vector or tensor) for one clip."""

# ---------------------------------------------------------------------------
# 1) Simple statistical features (v0.1 moved here)
# ---------------------------------------------------------------------------

# class StatsFeatureExtractor(BaseFeatureExtractor):
#     """Re‑implements v0.1: mean/std of velocities, stride, angles, arm swing."""
#     name = "stats"

#     def joint_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
#         ba, bc = a - b, c - b
#         ba /= np.linalg.norm(ba, axis=-1, keepdims=True) + 1e-9
#         bc /= np.linalg.norm(bc, axis=-1, keepdims=True) + 1e-9
#         cosang = (ba * bc).sum(-1).clip(-1, 1)
#         return np.degrees(np.arccos(cosang))

#     def extract(self, pose_seq: np.ndarray, fps: float = 30) -> Dict[str, float]:
#         HIP_L, HIP_R = 23, 24
#         KNEE_L, KNEE_R = 25, 26
#         ANKLE_L, ANKLE_R = 27, 28
#         SHOULDER_L, SHOULDER_R = 11, 12
#         wrist_L, wrist_R = 15, 16
#         valid = ~np.isnan(pose_seq[:, HIP_L, 0]) & ~np.isnan(pose_seq[:, HIP_R, 0])
#         if valid.sum() < 10:
#             return {"valid": False}
#         hips = (pose_seq[:, HIP_L, :2] + pose_seq[:, HIP_R, :2]) / 2
#         vel = np.linalg.norm(np.diff(hips, axis=0), axis=-1) * fps
#         ankle_dist = np.linalg.norm(pose_seq[:, ANKLE_L, :2] - pose_seq[:, ANKLE_R, :2], axis=-1)
#         angle_left_knee = self.joint_angle(pose_seq[:, HIP_L], pose_seq[:, KNEE_L], pose_seq[:, ANKLE_L])
#         angle_right_knee = self.joint_angle(pose_seq[:, HIP_R], pose_seq[:, KNEE_R], pose_seq[:, ANKLE_R])
#         arm_L = np.linalg.norm(pose_seq[:, SHOULDER_L, :2] - pose_seq[:, wrist_L, :2], axis=-1)
#         arm_R = np.linalg.norm(pose_seq[:, SHOULDER_R, :2] - pose_seq[:, wrist_R, :2], axis=-1)
#         return {
#             "valid": True,
#             "mean_velocity": np.nanmean(vel),
#             "std_velocity": np.nanstd(vel),
#             "mean_stride": np.nanmean(ankle_dist),
#             "std_stride": np.nanstd(ankle_dist),
#             "mean_knee_angle_left": np.nanmean(angle_left_knee),
#             "mean_knee_angle_right": np.nanmean(angle_right_knee),
#             "std_knee_angle_left": np.nanstd(angle_left_knee),
#             "std_knee_angle_right": np.nanstd(angle_right_knee),
#             "mean_arm_amp_left": np.nanmean(arm_L),
#             "mean_arm_amp_right": np.nanmean(arm_R),
#         }



class StatsFeatureExtractor(BaseFeatureExtractor):
    name = "stats"

    def joint_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        ba, bc = a - b, c - b
        ba /= np.linalg.norm(ba, axis=-1, keepdims=True) + 1e-9
        bc /= np.linalg.norm(bc, axis=-1, keepdims=True) + 1e-9
        cosang = (ba * bc).sum(-1).clip(-1, 1)
        return np.degrees(np.arccos(cosang))

    def _extract_with_indices(
        self,
        pose_seq: np.ndarray,
        fps: float,
        SHOULDER_L, SHOULDER_R,
        WRIST_L,    WRIST_R,
        HIP_L,      HIP_R,
        KNEE_L,     KNEE_R,
        ANKLE_L,    ANKLE_R,
    ) -> Dict[str, float]:
        valid = ~np.isnan(pose_seq[:, HIP_L, 0]) & ~np.isnan(pose_seq[:, HIP_R, 0])
        if valid.sum() < 10:
            return {"valid": False}
        hips  = (pose_seq[:, HIP_L, :2] + pose_seq[:, HIP_R, :2]) / 2
        vel   = np.linalg.norm(np.diff(hips, axis=0), axis=-1) * fps
        stride= np.linalg.norm(pose_seq[:, ANKLE_L, :2] - pose_seq[:, ANKLE_R, :2], axis=-1)
        ang_L = self.joint_angle(pose_seq[:, HIP_L],  pose_seq[:, KNEE_L], pose_seq[:, ANKLE_L])
        ang_R = self.joint_angle(pose_seq[:, HIP_R],  pose_seq[:, KNEE_R], pose_seq[:, ANKLE_R])
        arm_L = np.linalg.norm(pose_seq[:, SHOULDER_L, :2] - pose_seq[:, WRIST_L, :2], axis=-1)
        arm_R = np.linalg.norm(pose_seq[:, SHOULDER_R, :2] - pose_seq[:, WRIST_R, :2], axis=-1)
        return {
            "valid": True,
            "mean_velocity": np.nanmean(vel),   "std_velocity": np.nanstd(vel),
            "mean_stride":   np.nanmean(stride),"std_stride":   np.nanstd(stride),
            "mean_knee_angle_left":  np.nanmean(ang_L),
            "mean_knee_angle_right": np.nanmean(ang_R),
            "std_knee_angle_left":   np.nanstd(ang_L),
            "std_knee_angle_right":  np.nanstd(ang_R),
            "mean_arm_amp_left":  np.nanmean(arm_L),
            "mean_arm_amp_right": np.nanmean(arm_R),
        }

    # старая extract() просто вызывает ядро с MediaPipe-индексами
    def extract(self, pose_seq: np.ndarray, fps: float = 30):
        return self._extract_with_indices(
            pose_seq, fps,
            11,12,   # SHOULDER
            15,16,   # WRIST
            23,24,   # HIP
            25,26,   # KNEE
            27,28,   # ANKLE
        )

    
    
    

# ---------------------------------------------------------------------------
# 2) Cycle‑aligned DCT features (vector)
# ---------------------------------------------------------------------------

class CycleDCTFeatureExtractor(BaseFeatureExtractor):
    """Detect gait cycles → sample one cycle → apply 1‑D DCT to joint y‑trajectories."""
    name = "cycle_dct"

    def _find_cycle_length(self, ankle_dist: np.ndarray, fps: float) -> int | None:
        # naive autocorrelation peak
        ac = np.correlate(ankle_dist - ankle_dist.mean(), ankle_dist, mode="full")
        ac = ac[ac.size // 2:]
        peak = ac[1:].argmax() + 1
        if 10 < peak < fps * 2:  # plausible 0.3‑2 s gait
            return peak
        return None

    def extract(self, pose_seq: np.ndarray, fps: float = 30) -> Dict[str, Any]:
        ANKLE_L, ANKLE_R = 27, 28
        ankle_dist = np.linalg.norm(pose_seq[:, ANKLE_L, 1] - pose_seq[:, ANKLE_R, 1])
        L = self._find_cycle_length(ankle_dist, fps)
        if L is None:
            return {"valid": False}
        idx = slice(0, L)
        sub = pose_seq[idx, :, 1]  # use y‑coord only for invariance to camera offset
        coeffs = dct(sub, axis=0, norm="ortho")[:10]  # keep first 10 coeffs per joint
        feat_vec = coeffs.flatten()
        return {"valid": True, "dct": feat_vec}

# ---------------------------------------------------------------------------
# 3) Frame‑level tensor output for deep networks (sequence)
# ---------------------------------------------------------------------------

class SeqTensorExtractor(BaseFeatureExtractor):
    """Simply returns (T, J*3) normalised coordinates for RNN/TCN/Transformer."""
    name = "seq_tensor"

    def extract(self, pose_seq: np.ndarray, fps: float = 30) -> np.ndarray:
        # replace NaNs with 0 and append a mask channel
        mask = (~np.isnan(pose_seq[..., 0])).astype(np.float32)
        pose_seq = np.nan_to_num(pose_seq, nan=0.0)
        flat = pose_seq.reshape(pose_seq.shape[0], -1)
        return np.concatenate([flat, mask], axis=1)  # (T, J*3 + J)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_extractor(name: str) -> BaseFeatureExtractor:
    name = name.lower()
    match name:
        case "stats":
            return StatsFeatureExtractor()
        case "cycle_dct":
            return CycleDCTFeatureExtractor()
        case "seq_tensor":
            return SeqTensorExtractor()
        case "stats17":
            return Stats17FeatureExtractor()
        case "statsparts":
            return StatsPartsFeatureExtractor()
        case "statsmix":
            return StatsMixFeatureExtractor()
        case "statsplus":
            return StatsPlusFeatureExtractor()
        case _:
            raise ValueError(
                f"Unknown gait feature extractor '{name}'. Available: stats, cycle_dct, seq_tensor"
            )
        
# weapon_gait/features/gait_features.py
class Stats17FeatureExtractor(StatsFeatureExtractor):
    """Версия для MoveNet (17 keypoints)."""
    name = "stats17"

    def extract(self, pose_seq: np.ndarray, fps: float = 30):
        return self._extract_with_indices(
            pose_seq, fps,
            5, 6,    # SHOULDER
            9,10,    # WRIST
            11,12,   # HIP
            13,14,   # KNEE
            15,16,   # ANKLE
        )


class StatsPartsFeatureExtractor(StatsFeatureExtractor):
    """Считает torso / arms / legs отдельно + индикаторы видимости."""
    name = "statsparts"

    _IDX = {
        "SHO_L":11, "SHO_R":12,
        "HIP_L":23, "HIP_R":24,
        "ELB_L":13, "ELB_R":14, "WRI_L":15, "WRI_R":16,
        "KNE_L":25, "KNE_R":26, "ANK_L":27, "ANK_R":28,
    }

    def extract(self, pose_seq, fps: float = 30):
        f = {}                  # итоговый словарь
        idx = self._IDX

        # ---------- torso ----------
        torso_vis = ~np.isnan(pose_seq[:, idx["HIP_L"], 0])
        f["torso_visible"] = torso_vis.mean()
        if torso_vis.sum() > 5:
            shoulders = pose_seq[:, [idx["SHO_L"], idx["SHO_R"]], :2]
            width = np.linalg.norm(
                shoulders[:, 0] - shoulders[:, 1], axis=-1)
            f.update({"torso_width_mean": width.mean(),
                      "torso_width_std": width.std()})

        # ---------- arms ----------
        armL_vis = ~np.isnan(pose_seq[:, idx["WRI_L"], 0])
        armR_vis = ~np.isnan(pose_seq[:, idx["WRI_R"], 0])
        f["armL_visible"] = armL_vis.mean()
        f["armR_visible"] = armR_vis.mean()
        if armL_vis.sum() > 5:
            ampL = np.linalg.norm(
                pose_seq[:, idx["SHO_L"], :2] -
                pose_seq[:, idx["WRI_L"], :2], axis=-1)
            f["arm_amp_L"] = ampL.mean()
        if armR_vis.sum() > 5:
            ampR = np.linalg.norm(
                pose_seq[:, idx["SHO_R"], :2] -
                pose_seq[:, idx["WRI_R"], :2], axis=-1)
            f["arm_amp_R"] = ampR.mean()
            if "arm_amp_L" in f:
                f["arm_asym"] = abs(f["arm_amp_L"] - f["arm_amp_R"])

        # ---------- legs ----------
        leg_vis = ~np.isnan(pose_seq[:, idx["ANK_L"], 0]) & \
                  ~np.isnan(pose_seq[:, idx["ANK_R"], 0])
        f["legs_visible"] = leg_vis.mean()
        if leg_vis.sum() > 5:
            stride = np.linalg.norm(
                pose_seq[:, idx["ANK_L"], :2] -
                pose_seq[:, idx["ANK_R"], :2], axis=-1)
            f["stride_mean"] = stride.mean()
            f["stride_std"]  = stride.std()

        f["valid"] = any(k.endswith("_visible") and f[k] > 0.1 for k in f)
        return f

class StatsMixFeatureExtractor(BaseFeatureExtractor):
    """Объединяем classic-stats + parts-stats."""
    name = "statsmix"

    def __init__(self):
        self._base  = StatsFeatureExtractor()
        self._parts = StatsPartsFeatureExtractor()

    def extract(self, pose_seq, fps: float = 30):
        f = {}
        f.update(self._base.extract(pose_seq, fps))
        f.update(self._parts.extract(pose_seq, fps))
        # индикатор «нет ног»
        f["legs_missing"] = 1.0 if f.get("legs_visible", 0.0) < 0.2 else 0.0
        return f



STRIDE_IDXS = (27, 28)  # MediaPipe ankles L,R (x,y,z)

class StatsPlusFeatureExtractor(BaseFeatureExtractor):
    name = "statsplus"

    def __init__(self):
        self._base  = StatsFeatureExtractor()
        self._parts = StatsPartsFeatureExtractor()

    # ------------------------------------------------------------------
    def extract(self, pose_seq: np.ndarray, fps: float = 30):
        out = {}
        out.update(self._base.extract(pose_seq, fps))
        out.update(self._parts.extract(pose_seq, fps))

        # --------------------------------------------------------------
        #  step frequency via ankle‑ankle distance
        # --------------------------------------------------------------
        L, R = STRIDE_IDXS
        stride = np.linalg.norm(pose_seq[:, L, :2] - pose_seq[:, R, :2], axis=-1)
        stride = stride[~np.isnan(stride)]
        if len(stride) > 20:
            # ДБП: частота дом. пика в спектре
            z = stride - stride.mean()
            fft = np.fft.rfft(z)
            freqs = np.fft.rfftfreq(len(z), d=1 / fps)
            peak = freqs[np.argmax(np.abs(fft[1:])) + 1]  # пропустим DC
            out["step_freq"] = peak

            # ----------------------------------------------------------
            #   DCT‑коэффициенты (0..2) норм. сигнала
            # ----------------------------------------------------------
            d = dct(z, norm="ortho")[:3]
            for k in range(3):
                out[f"dct_{k+1}"] = float(d[k])

        # indicator legs missing
        out["legs_missing"] = 1.0 if out.get("legs_visible", 0.0) < 0.2 else 0.0
        return out

__all__: List[str] = [
    "BaseFeatureExtractor",
    "StatsFeatureExtractor",
    "Stats17FeatureExtractor"
    "CycleDCTFeatureExtractor",
    "StatsPartsFeatureExtractor",
    "StatsMixFeatureExtractor",
    "SeqTensorExtractor",
    "StatsPlusFeatureExtractor",
    "get_extractor",
]
