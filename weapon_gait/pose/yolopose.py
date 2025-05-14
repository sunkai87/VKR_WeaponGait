# weapon_gait/pose/yolopose.py
from pathlib import Path
from typing import List
import numpy as np
import cv2
from ultralytics import YOLO
from .base import PoseExtractor

class YOLOPoseExtractor(PoseExtractor):
    def __init__(self, cache_dir: str | Path = "cache/poses_yolo"):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(exist_ok=True, parents=True)
        # yolov8n-pose.pt ~ 6MB → качается автоматически
        self.model = YOLO("yolo11s-pose.pt")  # 17 keypoints

    @property
    def name(self) -> str: return "yolopose"

    # def extract(self, video_path: Path) -> np.ndarray:
    #     out = self.cache_dir / f"{video_path.stem}.npy"
    #     if out.exists(): return np.load(out)

    #     cap = cv2.VideoCapture(str(video_path))
    #     kps: List[np.ndarray] = []
    #     while True:
    #         ok, frame = cap.read();  # покадрово
    #         if not ok: break
    #         res = self.model(frame, verbose=False)[0]
    #         if len(res.keypoints):
    #             xy = res.keypoints.xy[0].cpu().numpy()     # (17,2)
    #             conf = res.keypoints.conf[0].cpu().numpy() # (17,)
    #             kp = np.concatenate([xy, conf[:,None]], axis=1)
    #             kp[conf < 0.3] = np.nan
    #         else:
    #             kp = np.full((17,3), np.nan)
    #         kps.append(kp)
    #     cap.release()
    #     arr = np.stack(kps)        # (T,17,3)
    #     np.save(out, arr)
    #     return arr
    
    # weapon_gait/pose/yolopose.py  (замените extract)
    def extract(self, video_path: Path) -> np.ndarray:
        out = self.cache_dir / f"{video_path.stem}.npy"
        if out.exists():
            return np.load(out)

        cap = cv2.VideoCapture(str(video_path))
        kps = []
        # while True:
        #     ok, frame = cap.read()
        #     if not ok:
        #         break
        #     res = self.model(frame, verbose=False)[0]
        #     if res.keypoints and res.keypoints.xy is not None:
        #         xy = res.keypoints.xy[0].cpu().numpy()          # (17,2)
        #         conf = (
        #             res.keypoints.conf[0].cpu().numpy()
        #             if res.keypoints.conf is not None
        #             else np.ones(17)
        #         )
        #         kp = np.concatenate([xy, conf[:, None]], axis=1)
        #         kp[conf < 0.3] = np.nan
        #     else:
        #         kp = np.full((17, 3), np.nan)
        #     kps.append(kp)
        # weapon_gait/pose/yolopose.py  – замените весь while-цикл на этот блок
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = self.model(frame, verbose=False)[0]    # → ultralytics.engine.results.Results
            # if res.keypoints and res.keypoints.xy is not None:
            #     xy   = res.keypoints.xy[0].cpu().numpy()      # (N,2)
            #     conf = res.keypoints.conf[0].cpu().numpy()    # (N,)
            #     if xy.shape[0] == 17:                         # ожидаем 17 keypoints
            #         kp = np.concatenate([xy, conf[:, None]], axis=1)  # (17,3)
            #         kp[conf < 0.3] = np.nan
            #     else:
            #         kp = np.full((17, 3), np.nan)             # недобрал – считаем пропуск
            # else:
            #     kp = np.full((17, 3), np.nan)
            # weapon_gait/pose/yolopose.py  – замените кусок внутри цикла
            if res.keypoints and res.keypoints.xy is not None:
                xy = res.keypoints.xy[0].cpu().numpy()          # (N,2) or empty
                conf_arr = (
                    res.keypoints.conf[0].cpu().numpy()
                    if res.keypoints.conf is not None
                    else np.ones(len(xy))
                )
                if xy.shape[0] == 17:
                    kp = np.concatenate([xy, conf_arr[:, None]], axis=1)
                    kp[conf_arr < 0.3] = np.nan
                else:
                    kp = np.full((17, 3), np.nan)
            else:
                kp = np.full((17, 3), np.nan)




            kps.append(kp)



        cap.release()
        arr = np.stack(kps)        # (T,17,3)
        np.save(out, arr)
        return arr


