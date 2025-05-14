# """
# weapon_gait/features/stats_plus.py
# =================================
# Расширенный набор признаков «StatsPlus»
# --------------------------------------
# * базовые gait‐статистики (StatsFeatureExtractor)
# * visibility‑flags и амплитуды (StatsPartsFeatureExtractor)
# * **freq** — частота шага (Гц)
# * **dct _k1, dct _k2, dct _k3** — первые 3 коэффициента DCT траектории
#   горизонтальной дистанции между лодыжками (stride‑signal)

# Работает с 33‑точечными координатами MediaPipe.  Для 17‑точечных (YOLO,
# MoveNet) нужно адаптировать индексы лодыжек в STRIDE_IDXS.
# """
# from __future__ import annotations
# from pathlib import Path
# import numpy as np
# from scipy.fftpack import dct

# from .gait_features import (
#     BaseFeatureExtractor,
#     StatsFeatureExtractor,
#     StatsPartsFeatureExtractor,
# )

# STRIDE_IDXS = (27, 28)  # MediaPipe ankles L,R (x,y,z)

# class StatsPlusFeatureExtractor(BaseFeatureExtractor):
#     name = "statsplus"

#     def __init__(self):
#         self._base  = StatsFeatureExtractor()
#         self._parts = StatsPartsFeatureExtractor()

#     # ------------------------------------------------------------------
#     def extract(self, pose_seq: np.ndarray, fps: float = 30):
#         out = {}
#         out.update(self._base.extract(pose_seq, fps))
#         out.update(self._parts.extract(pose_seq, fps))

#         # --------------------------------------------------------------
#         #  step frequency via ankle‑ankle distance
#         # --------------------------------------------------------------
#         L, R = STRIDE_IDXS
#         stride = np.linalg.norm(pose_seq[:, L, :2] - pose_seq[:, R, :2], axis=-1)
#         stride = stride[~np.isnan(stride)]
#         if len(stride) > 20:
#             # ДБП: частота дом. пика в спектре
#             z = stride - stride.mean()
#             fft = np.fft.rfft(z)
#             freqs = np.fft.rfftfreq(len(z), d=1 / fps)
#             peak = freqs[np.argmax(np.abs(fft[1:])) + 1]  # пропустим DC
#             out["step_freq"] = peak

#             # ----------------------------------------------------------
#             #   DCT‑коэффициенты (0..2) норм. сигнала
#             # ----------------------------------------------------------
#             d = dct(z, norm="ortho")[:3]
#             for k in range(3):
#                 out[f"dct_{k+1}"] = float(d[k])

#         # indicator legs missing
#         out["legs_missing"] = 1.0 if out.get("legs_visible", 0.0) < 0.2 else 0.0
#         return out