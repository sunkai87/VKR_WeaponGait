# """
# weapon_gait/render_predict.py
# =======================================
# Функция `render_prediction()`
# ----------------------------
# Создаёт ролик с подписанными bbox‑ами, person‑ID и предсказанным классом.
# * На вход: путь к видеофайлу, YOLO‑detector, модель (.joblib).
# * Снова прогоняет YOLO (быстро на CPU ~30–40 FPS для 720p).
# * Использует тот же `_GreedyTracker` (IoU‑greedy) для согласования ID c обучением.
# * Как только кадр принадлежит треку с уже рассчитанным классом, пишет «weapon»/«no_weapon» и вероятность.

# Записывает `out_path` (mp4, H.264).  Управление цветами и шрифтом через OpenCV.
# """
# from __future__ import annotations

# import cv2, numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# from weapon_gait.pose.crop_mp_multi import _GreedyTracker
# import joblib, pandas as pd

# # --------------------------------------------------------------
# COLORS = {
#     "weapon": (0, 0, 255),      # red
#     "no_weapon": (0, 200, 0),   # green
# }
# FONT = cv2.FONT_HERSHEY_SIMPLEX

# # --------------------------------------------------------------
# def _load_model(model_pth: Path):
#     payload = joblib.load(model_pth)
#     model = payload["model"]
#     gait_be = payload["gait_backend"]
#     cols = payload["feature_cols"]
#     return model, gait_be, cols

# # --------------------------------------------------------------
# def render_prediction(video: Path, model_pth: Path, out_mp4: Path):
#     # 1) detect pose per person once to know predictions
#     det = YOLO("yolov8s-pose.pt", task="pose", verbose=False)
#     tracker = _GreedyTracker()
#     mp_cache: dict[int, np.ndarray] = {}          # pid -> pose seq list

#     # load model + feature extractor
#     model, gait_be, cols = _load_model(model_pth)
#     from weapon_gait.features.gait_features import get_extractor as gaitx
#     gait_ex = gaitx(gait_be)

#     cap = cv2.VideoCapture(str(video))
#     w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps   = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     vw   = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

#     # pass 1 + render inline (saves memory)
#     frame_idx = 0
#     pid_class = {}
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         res = det.predict(frame, verbose=False)[0]
#         bboxes = res.boxes.xyxy.cpu().numpy() if len(res) else np.empty((0,4))
#         # tracker update
#         tracker.update(bboxes, [None]*len(bboxes))
#         # collect boxes per pid for drawing
#         pid2bbox = {tr.tid: tr.bbox for tr in tracker.tracks}

#         # predict per new pid when hits == 1
#         for tr in tracker.tracks:
#             if tr.tid in pid_class:
#                 continue
#             # crop & mediapipe once per pid (first frame)
#             x1, y1, x2, y2 = map(int, tr.bbox)
#             crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
#             if crop.size == 0:
#                 continue
#             from weapon_gait.pose.crop_mp_multi import CropMultiPoseExtractor
#             mp_pose = CropMultiPoseExtractor().mp_pose  # reuse object
#             res_mp = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             if not res_mp.pose_landmarks:
#                 continue
#             lm = res_mp.pose_landmarks.landmark
#             pose = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])[None, ...]  # 1×33×4
#             feats = gait_ex.extract(pose)
#             X = pd.DataFrame([{k: feats.get(k, np.nan) for k in cols}])
#             prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.0
#             pred = model.predict(X)[0]
#             pid_class[tr.tid] = (pred, prob)

#         # --------- draw -------------------------------------------
#         for tid, bb in pid2bbox.items():
#             x1,y1,x2,y2 = map(int, bb)
#             label, prob = pid_class.get(tid, ("det",0.0))
#             color = COLORS.get(label, (255,255,0))
#             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#             text = f"ID{tid}:{label} {prob:.2f}" if label!="det" else f"ID{tid}.."
#             cv2.putText(frame, text, (x1, max(0,y1-6)), FONT, 0.5, color, 2)

#         vw.write(frame)
#         frame_idx += 1

#     cap.release(); vw.release()
#     return out_mp4


# """
# weapon_gait/visualize/render_predict.py
# двух-проходный рендер: bbox + ID + предсказанный класс
# """
# from __future__ import annotations
# import cv2, numpy as np, pandas as pd, joblib
# from pathlib import Path
# from ultralytics import YOLO
# from weapon_gait.pose.crop_mp_multi import _GreedyTracker
# from weapon_gait.features.gait_features import get_extractor as gaitx

# COLORS = {"weapon": (0, 0, 255), "no_weapon": (0, 200, 0)}
# FONT   = cv2.FONT_HERSHEY_SIMPLEX

# # ──────────────────────────────────────────────────────────────
# def _collect_tracks(video: Path) -> dict[int, list[np.ndarray]]:
#     det      = YOLO("yolo11s-pose.pt", task="pose", verbose=False)
#     mp_pose  = __import__("mediapipe").solutions.pose.Pose(static_image_mode=False)
#     tracker  = _GreedyTracker()
#     pid2seq  : dict[int, list[np.ndarray]] = {}

#     cap = cv2.VideoCapture(str(video))
#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         res = det.predict(frame, verbose=False)[0]
#         bboxes = res.boxes.xyxy.cpu().numpy() if len(res) else np.empty((0,4))

#         kps_frame = []
#         for bb in bboxes:
#             x1,y1,x2,y2 = bb.astype(int)
#             crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
#             if crop.size == 0:
#                 kps_frame.append(np.full((33,4), np.nan)); continue
#             mp = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             if not mp.pose_landmarks:
#                 kps_frame.append(np.full((33,4), np.nan)); continue
#             lm = mp.pose_landmarks.landmark
#             kps_frame.append(np.array([[p.x, p.y, p.z, p.visibility] for p in lm]))
#         tracker.update(bboxes, kps_frame)

#         # append to dict
#         for tr in tracker.tracks:
#             pid2seq.setdefault(tr.tid, []).append(tr.kps[-1])
#     cap.release()
#     return pid2seq

# # ──────────────────────────────────────────────────────────────
# def _predict_classes(pid2seq: dict[int, list[np.ndarray]],
#                      model_pth: Path) -> dict[int, tuple[str,float]]:
#     payload = joblib.load(model_pth)
#     model, cols, gait_be = payload["model"], payload["feature_cols"], payload["gait_backend"]
#     gait_ex = gaitx(gait_be)

#     pid_class = {}
#     for pid, seq in pid2seq.items():
#         pose = np.stack(seq)                         # T×33×4
#         if pose.shape[0] < 20:
#             continue
#         feats = gait_ex.extract(pose)
#         if not feats.get("valid", True):
#             continue
#         X = pd.DataFrame([{k: feats.get(k, np.nan) for k in cols}])
#         prob = model.predict_proba(X)[0][1] if hasattr(model,"predict_proba") else 0.0
#         label = model.predict(X)[0]
#         pid_class[pid] = (label, prob)
#     return pid_class

# # # ──────────────────────────────────────────────────────────────
# # def render_prediction(video: Path, model_pth: Path, out_mp4: Path):
    
# #     # pass-1: треки и классы
# #     pid2seq  = _collect_tracks(video)
# #     pid_pred = _predict_classes(pid2seq, model_pth)

# #     # pass-2: заново детект + рисуем
# #     det     = YOLO("yolo11s-pose.pt", task="pose", verbose=False)
# #     tracker = _GreedyTracker()
# #     cap     = cv2.VideoCapture(str(video))
# #     w,h  = int(cap.get(3)), int(cap.get(4))
# #     fps  = cap.get(cv2.CAP_PROP_FPS)
# #     vw   = cv2.VideoWriter(str(out_mp4),
# #                            cv2.VideoWriter_fourcc(*"mp4v"),
# #                            fps, (w,h))

# #     while True:
# #         ok, frame = cap.read()
# #         if not ok: break
# #         res = det.predict(frame, verbose=False)[0]
# #         bboxes = res.boxes.xyxy.cpu().numpy() if len(res) else np.empty((0,4))
# #         tracker.update(bboxes, [None]*len(bboxes))

# #         for tr in tracker.tracks:
# #             x1,y1,x2,y2 = map(int, tr.bbox)
# #             label, prob = pid_pred.get(tr.tid, ("det",0.0))
# #             color = COLORS.get(label, (255,255,0))
# #             cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
# #             text = f"ID{tr.tid}:{label} {prob:.2f}" if label!="det" else f"ID{tr.tid}"     
# #             cv2.putText(frame,text,(x1,max(0,y1-6)),FONT,0.5,color,2)     
# #         vw.write(frame)
        
# #     cap.release()
# #     vw.release()
# #     return out_mp4

# # ──────────────────────────────────────────────────────────────
# def render_prediction(video: Path, model_pth: Path, out_mp4: Path):
#     frame_boxes: list[dict[int, np.ndarray]] = []
#     # pass-1: треки и классы
#     pid2seq  = _collect_tracks(video)
#     pid_pred = _predict_classes(pid2seq, model_pth)
#     tracker.update(bboxes, frame)
#     frame_boxes.append({tr.tid: tr.bbox.copy() for tr in tracker.tracks})

#     # pass-2: заново детект + рисуем
#     det     = YOLO("yolo11s-pose.pt", task="pose", verbose=False)
#     tracker = _GreedyTracker()
#     cap     = cv2.VideoCapture(str(video))
#     w,h  = int(cap.get(3)), int(cap.get(4))
#     fps  = cap.get(cv2.CAP_PROP_FPS)
#     vw   = cv2.VideoWriter(str(out_mp4),
#                            cv2.VideoWriter_fourcc(*"mp4v"),
#                            fps, (w,h))

#     while True:
#         ok, frame = cap.read()
#         if not ok: break
#         res = det.predict(frame, verbose=False)[0]
#         bboxes = res.boxes.xyxy.cpu().numpy() if len(res) else np.empty((0,4))
#         tracker.update(bboxes, [None]*len(bboxes))

#         for tr in tracker.tracks:
#             x1,y1,x2,y2 = map(int, tr.bbox)
#             label, prob = pid_pred.get(tr.tid, ("det",0.0))
#             color = COLORS.get(label, (255,255,0))
#             cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
#             text = f"ID{tr.tid}:{label} {prob:.2f}" if label!="det" else f"ID{tr.tid}"     
#             cv2.putText(frame,text,(x1,max(0,y1-6)),FONT,0.5,color,2)     
#         vw.write(frame)
        
#     cap.release()
#     vw.release()
#     return out_mp4

"""
weapon_gait/visualize/render_predict.py (v2)
===========================================
* двух‑проходный рендер без повторного YOLO ⇒ ID‑ы совпадают
  (bbox сохраняются после 1‑го прохода).
* выводит bbox + ID + predicted class + prob.
"""
from __future__ import annotations

from pathlib import Path
import cv2, numpy as np, pandas as pd, joblib
from ultralytics import YOLO
from weapon_gait.pose.crop_mp_multi import _GreedyTracker
from weapon_gait.features.gait_features import get_extractor as gaitx

COLORS = {"weapon": (0, 0, 255), "no_weapon": (0, 200, 0)}
FONT   = cv2.FONT_HERSHEY_SIMPLEX

# ───────────────────────── helpers ───────────────────────────

def _collect_tracks(video: Path):
    """Pass‑1: детект → трек → позы.  Возвращает:
        pid2seq  – pid → [T×33×4]
        frame_boxes – список словарей {pid: bbox} для каждого кадра
    """
    det      = YOLO("yolo11s-pose.pt", task="pose", verbose=False)
    mp_pose  = __import__("mediapipe").solutions.pose.Pose(static_image_mode=False)
    tracker  = _GreedyTracker()

    pid2seq:  dict[int, list[np.ndarray]] = {}
    frame_boxes: list[dict[int, np.ndarray]] = []

    cap = cv2.VideoCapture(str(video))
    while True:
        ok, frame = cap.read();
        if not ok: break
        res = det.predict(frame, verbose=False)[0]
        bboxes = res.boxes.xyxy.cpu().numpy() if len(res) else np.empty((0,4))

        kps_frame = []
        for bb in bboxes:
            x1,y1,x2,y2 = bb.astype(int)
            crop = frame[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            if crop.size == 0:
                kps_frame.append(np.full((33,4), np.nan)); continue
            mp = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not mp.pose_landmarks:
                kps_frame.append(np.full((33,4), np.nan)); continue
            lm = mp.pose_landmarks.landmark
            kps_frame.append(np.array([[p.x, p.y, p.z, p.visibility] for p in lm]))
        tracker.update(bboxes, kps_frame)
        frame_boxes.append({tr.tid: tr.bbox.copy() for tr in tracker.tracks})
        for tr in tracker.tracks:
            pid2seq.setdefault(tr.tid, []).append(tr.kps[-1])
    cap.release()
    return pid2seq, frame_boxes

# ------------------------------------------------------------

def _predict_classes(pid2seq: dict[int,list[np.ndarray]], model_pth: Path):
    pay = joblib.load(model_pth)
    model, gait_be, cols = pay["model"], pay["gait_backend"], pay["feature_cols"]
    gait_ex = gaitx(gait_be)
    pid_pred = {}
    for pid, seq in pid2seq.items():
        pose = np.stack(seq)
        if pose.shape[0] < 20:
            continue
        feats = gait_ex.extract(pose)
        if not feats.get("valid", True):
            continue
        X = pd.DataFrame([{k: feats.get(k, np.nan) for k in cols}])
        prob = model.predict_proba(X)[0][1] if hasattr(model,"predict_proba") else 0.0
        label = model.predict(X)[0]
        pid_pred[pid] = (label, prob)
    return pid_pred

# ------------------------------------------------------------

def render_prediction(video: Path, model_pth: Path, out_mp4: Path):
    pid2seq, frame_boxes = _collect_tracks(video)
    pid_pred             = _predict_classes(pid2seq, model_pth)

    cap = cv2.VideoCapture(str(video))
    w,h = int(cap.get(3)), int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS)
    vw  = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    for f_idx, boxes in enumerate(frame_boxes):
        ok, frame = cap.read(); assert ok
        for pid, bb in boxes.items():
            x1,y1,x2,y2 = map(int, bb)
            label, prob = pid_pred.get(pid, ("det",0.0))
            color = COLORS.get(label, (255,255,0))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            if label != "det":
                txt = f"ID{pid}:{label} {prob:.2f}"
                cv2.putText(frame, txt, (x1, max(0, y1-6)), FONT, 0.5, color, 2)
        vw.write(frame)

    cap.release(); vw.release()
    return out_mp4