"""
weapon_gait/visualize/draw_bbox_vis.py
--------------------------------------
Overlay bounding‑boxes (already tracked) **only** for people that
have predictions, and write `ID: label prob` next to the box.

Inputs
~~~~~~
video_path   : Path to original .mp4
frame_boxes  : list[dict[int, np.ndarray]]  – for each frame a mapping
               pid -> [x1,y1,x2,y2] (px, already absolute)
pid_pred     : dict[int, tuple[str,float]] – pid -> (label, prob)
out_mp4      : Path to resulting visualized video
"""
from __future__ import annotations

from pathlib import Path
import cv2, numpy as np

COLORS = {"weapon": (0, 0, 255),   # red
          "no_weapon": (0, 200, 0)} # green
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------------------------------------

def draw_bbox_vis(video_path: Path,
                  frame_boxes: list[dict[int, np.ndarray]],
                  pid_pred: dict[int, tuple[str, float]],
                  out_mp4: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vw = cv2.VideoWriter(str(out_mp4),
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (w, h))
    for boxes in frame_boxes:
        ok, frame = cap.read()
        if not ok:
            break
        for pid, bb in boxes.items():
            if pid not in pid_pred:              # ← фильтр по результатам
                continue
            label, prob = pid_pred[pid]
            color = COLORS.get(label, (255,255,0))
            x1, y1, x2, y2 = map(int, bb)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"ID{pid}:{label} {prob:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1-6)),
                        FONT, 0.5, color, 2)
        vw.write(frame)

    cap.release()
    vw.release()
    return out_mp4