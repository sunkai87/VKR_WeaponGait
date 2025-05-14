"""
pose_benchmark.py
=================
Minimal script to measure runtime and robustness of several pose
extractors on a folder of short videos.

Extractors supported out‑of‑the‑box:
    * mediapipe  (CPU, built‑in)
    * yolopose   (PyTorch, GPU)          ← stub
    * alphapose  (PyTorch, GPU)          ← stub
    * hrnet      (MMPose, GPU)           ← stub

For heavy models only a dummy timer is implemented for now; replace
`extract_stub()` with real inference when deps are installed.

Outputs
-------
pose_benchmark.csv with columns:
    extractor, video, frames, fps, drop_rate
Overlay videos saved in bench_vis/<extractor>/<orig‑name>.mp4
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import numpy as np
import pandas as pd

from weapon_gait.pose.mediapipe_pose import MediaPipePose

# ----------------------------------------------------------------------------
# Stub extractors for yet‑to‑be‑installed libs
# ----------------------------------------------------------------------------

def extract_stub(video_p: Path, name: str) -> np.ndarray:
    """Return dummy (T, J, 3) filled with NaNs; measure runtime only."""
    cap = cv2.VideoCapture(str(video_p))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return np.full((frames, 17, 3), np.nan)

# registry
EXTRACTORS: Dict[str, Callable[[Path], np.ndarray]] = {
    "mediapipe": lambda p: MediaPipePose().extract(p),
    "yolopose": lambda p: extract_stub(p, "yolopose"),
    "alphapose": lambda p: extract_stub(p, "alphapose"),
    "hrnet": lambda p: extract_stub(p, "hrnet"),
}

# ----------------------------------------------------------------------------

def benchmark_folder(folder: Path, out_csv: Path, overlay_dir: Path, extractors: List[str]):
    records = []
    vids = list(folder.glob("*.mp4"))
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for name in extractors:
        fn = EXTRACTORS[name]
        out_vis = overlay_dir / name
        out_vis.mkdir(exist_ok=True, parents=True)
        for vid in vids:
            t0 = time.time()
            pose_seq = fn(vid)
            dt = time.time() - t0
            fps = pose_seq.shape[0] / dt if dt > 0 else 0.0
            drop = np.isnan(pose_seq[..., 0]).mean()
            records.append({
                "extractor": name,
                "video": vid.name,
                "frames": int(pose_seq.shape[0]),
                "runtime_sec": round(dt, 3),
                "fps": round(fps, 2),
                "drop_rate": round(drop, 3),
            })
            # save simple overlay first 100 frames
            save_overlay(vid, pose_seq, out_vis / vid.name)
    pd.DataFrame(records).to_csv(out_csv, index=False)


def save_overlay(video_p: Path, pose_seq: np.ndarray, out_p: Path):
    cap = cv2.VideoCapture(str(video_p))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out_p), fourcc, 30, (w, h))
    frame_idx = 0
    while frame_idx < pose_seq.shape[0] and frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break
        draw_skeleton(frame, pose_seq[frame_idx])
        vw.write(frame)
        frame_idx += 1
    cap.release(); vw.release()


def draw_skeleton(img: np.ndarray, kpts: np.ndarray):
    for x, y, _ in kpts:
        if not np.isnan(x):
            cv2.circle(img, (int(x * img.shape[1]), int(y * img.shape[0])), 2, (0, 255, 0), -1)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark multiple pose extractors")
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--out", type=Path, default=Path("pose_benchmark.csv"))
    parser.add_argument("--overlay", type=Path, default=Path("bench_vis"))
    parser.add_argument("--extractors", nargs="*", default=list(EXTRACTORS.keys()))
    args = parser.parse_args()

    benchmark_folder(args.videos, args.out, args.overlay, args.extractors)
    print(f"Done → {args.out}")