"""
weapon_gait/pose/crop_mp_multi.py
=================================
CropMultiPoseExtractor
----------------------
* Детектируем людей YOLOv8s‑pose (**CPU**).
* Для каждого bbox делаем квадратный **кроп** (+10 % отступ),
  ресайз до 256×256.
* Запускаем MediaPipe Pose → получаем 33 кпт × (x,y,z,conf).
* Треки объединяются простым BYTETrack‑подобным ассоциатором.
* На выходе для **каждого человека** сохраняем
  ``cache/poses_crop_mp/<stem>_pid<ID>.npy``
  (shape = T × 33 × 4, где xyz — нормированные 0‥1 в кропе).

Зависимости
~~~~~~~~~~~
``ultralytics>=8.3``  (YOLOv8),  ``mediapipe>=0.10``  (Pose),  OpenCV.
Работает на CPU, но можно задать `device='cuda'` в YOLO.
"""
from __future__ import annotations

from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
import mediapipe as mp

EXPAND = 1.25
# ------------------------------------------------------------------
#  Mini BYTE‑style tracker (IoU‑greedy) -----------------------------
# ------------------------------------------------------------------
class _Track:
    __slots__ = ("tid", "bbox", "kps", "miss")
    def __init__(self, tid, bbox, kp):
        self.tid  = tid
        self.bbox = bbox        # last bbox  [x1,y1,x2,y2]
        self.kps  = [kp]        # list of (33,4)
        self.miss = 0

class _GreedyTracker:
    def __init__(self, iou_th=0.3, max_miss=15):
        self.iou_th, self.max_miss = iou_th, max_miss
        self.tracks: list[_Track] = []
        self.next_id = 0

    @staticmethod
    def _iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0:
            return 0.0
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA + areaB - inter)

    def update(self, bboxes: np.ndarray, kps: list[np.ndarray]):
        assigned = set()
        # simple greedy assignment
        for tr in self.tracks:
            best_iou, best_j = 0.0, -1
            for j, bb in enumerate(bboxes):
                if j in assigned:
                    continue
                i = self._iou(tr.bbox, bb)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou > self.iou_th:
                j = best_j; assigned.add(j)
                tr.bbox = bboxes[j]
                tr.kps.append(kps[j])
                tr.miss = 0
            else:
                tr.miss += 1
        # new tracks
        for j, bb in enumerate(bboxes):
            if j not in assigned:
                self.tracks.append(_Track(self.next_id, bb, kps[j]))
                self.next_id += 1
        # remove lost
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

# ------------------------------------------------------------------
class CropMultiPoseExtractor:
    """YOLO detect → crop → MediaPipe pose → track by IoU."""
    name = "crop_mp_multi"

    def __init__(self, device: str = "cpu"):
        self.det = YOLO("yolo11s-pose.pt", task="pose", verbose=False)
        if device == "cuda":
            self.det.to(device)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False,
                                              model_complexity=1,
                                              enable_segmentation=False)
        self.cache = Path("cache/poses_crop_mp"); self.cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _crop_resize(self, frame: np.ndarray, bbox: np.ndarray, out_sz: int = 256): 
        x1, y1, x2, y2 = bbox.astype(int)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        # square +10 %
        side = int(1.2 * max(w, h))
        cx, cy = x1 + w//2, y1 + h//2
        sx, sy = max(cx - side//2, 0), max(cy - side//2, 0)
        ex, ey = sx + side, sy + side
        crop = frame[sy:ey, sx:ex]
        if crop.size == 0:
            return None, None  # empty
        crop_rs = cv2.resize(crop, (out_sz, out_sz))
        return crop_rs, (sx, sy, side)   # affine for back‑transform if надо

    # ------------------------------------------------------------------
    def extract(self, video: Path, return_boxes:bool=False):
        cap = cv2.VideoCapture(str(video))
        tracker = _GreedyTracker()
        frame_boxes = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = self.det.predict(frame, verbose=False)[0]
            if len(res) == 0:
                tracker.update(np.empty((0,4)), [])
                continue
            bboxes = res.boxes.xyxy.cpu().numpy()
            kps_frame = []
            for bb in bboxes:
                crop, info = self._crop_resize(frame, bb)
                if crop is None:
                    kps_frame.append(np.full((33,4), np.nan)); continue
                res_mp = self.mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if not res_mp.pose_landmarks:
                    kps_frame.append(np.full((33,4), np.nan)); continue
                lm = res_mp.pose_landmarks.landmark
                arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])  # (33,4) normalized in crop
                kps_frame.append(arr)
            tracker.update(bboxes, kps_frame)
            frame_boxes.append({tr.tid: tr.bbox.copy() for tr in tracker.tracks})
        cap.release()

        out_files = []
        stem = video.stem
        for tr in tracker.tracks:
            arr = np.stack(tr.kps)  # T×33×4
            if arr.shape[0] < 20:
                continue
            out = self.cache / f"{stem}_pid{tr.tid}.npy"
            np.save(out, arr)
            out_files.append(out)

        # return out_files
        if return_boxes:
            return out_files, frame_boxes      # type: ignore # ← НОВОЕ
        return out_files  
    

# def _crop_resize(self, frame, bbox, out_sz=256):
#     h_img, w_img = frame.shape[:2]
#     x1, y1, x2, y2 = bbox.astype(int)
#     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#     side   = int(1.2 * max(x2 - x1, y2 - y1))

#     # сдвигаем окно, **чтобы оно целиком поместилось в кадр**
#     sx = np.clip(cx - side // 2, 0, w_img - side)
#     sy = np.clip(cy - side // 2, 0, h_img - side)
#     ex, ey = sx + side, sy + side          # теперь гарантированно ≤ границы

#     crop = frame[sy:ey, sx:ex]              # всегда квадрат side×side

#     # аккуратный letterbox-ресайз без искажения
#     crop_rs = cv2.resize(crop, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)