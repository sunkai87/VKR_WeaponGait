# from __future__ import annotations

# from pathlib import Path
# import numpy as np
# from .base import PoseExtractor

# class RTMPoseExtractor(PoseExtractor):
#     def __init__(self, cache_dir="cache/poses_rtmpose"):
#         from mmpose.apis import MMPoseInferencer
#         self.infer = MMPoseInferencer(
#             pose2d='rtmpose-s',      # скачивает чекпойнт
#             device='cpu')
#         self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)

#     @property
#     def name(self):
#         return "rtmpose"

#     def extract(self, video_path: Path) -> np.ndarray:  # pragma: no cover
#         raise NotImplementedError("RTMPose backend requires CUDA; will add once GPU is on")


# # weapon_gait/pose/rtmpose_cpu.py
# from pathlib import Path
# import numpy as np
# from mmpose.apis import MMPoseInferencer
# from .base import PoseExtractor

# class RTMPoseExtractor(PoseExtractor):
#     def __init__(self, cache_dir="cache/poses_rtmpose"):
#         self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
#         # скачает rtmpose-s по умолчанию
#         self.infer = MMPoseInferencer(
#             pose2d='rtmpose-s', device='cpu', det_model='yolox-tiny')

#     @property
#     def name(self): return "rtmpose"

#     def extract(self, video_path: Path) -> np.ndarray:
#         out = self.cache_dir / f"{video_path.stem}.npy"
#         if out.exists(): return np.load(out)

#         keypoints = []
#         for frame in self.infer(video_path, show_progress=False):
#             if frame['predictions']:
#                 pts = frame['predictions'][0]['keypoints']  # (17,3)
#                 keypoints.append(pts)
#             else:
#                 keypoints.append(np.full((17,3), np.nan))
#         arr = np.stack(keypoints)
#         np.save(out, arr)
#         return arr


# from pathlib import Path
# import numpy as np
# from mmpose.apis import MMPoseInferencer
# from .base import PoseExtractor

# # CFG  = ("https://github.com/open-mmlab/mmpose/tree/main/mmpose"
# #         "configs/body_2d_keypoint/rtmpose/coco/"
# #         "rtmpose-s_8xb256-420e_coco-256x192.py")


# CFG  = ("https://github.com/open-mmlab/mmpose/blob/main/mmpose/"
#         "configs/body_2d_keypoint/rtmpose/coco/"
#         "rtmpose_s_8xb256_420e_aic_coco_256x192.py")
# # CKPT = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/rtmpose/"
# #         "coco/rtmpose-s_8xb256-420e_coco-256x192-a9e118e4_20230315.pth")

# #CFG = "https://github.com/open-mmlab/mmpose/blob/main/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py"
# #CFG = ("https://github.com/open-mmlab/mmpose/blob/main/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose_s_8xb256_420e_aic_coco_256x192.py")
# #CKPT = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192-a9e118e4_20230315.pth")

# # <a href="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192-9049ed09_20221013.pth">ckpt</a>

# # https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth



# CKPT = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth") #rel="nofollow"
# class RTMPoseExtractor(PoseExtractor):
#     def __init__(self, cache_dir="cache/poses_rtmpose"):
#         self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.infer = MMPoseInferencer(
#             pose2d=CFG, pose2d_weights=CKPT,  # <- раздельные аргументы
#             det_model="yolox-tiny",                  # целый кадр bbox
#             device='cpu')

#     @property
#     def name(self): return "rtmpose"

#     def extract(self, video_path: Path) -> np.ndarray:
#         out = self.cache_dir / f"{video_path.stem}.npy"
#         if out.exists():
#             return np.load(out)

#         keypoints = []
#         for res in self.infer(video_path, show_progress=False):
#             if res['predictions']:
#                 keypoints.append(res['predictions'][0]['keypoints'])  # (17,3)
#             else:
#                 keypoints.append(np.full((17,3), np.nan))
#         arr = np.stack(keypoints)
#         np.save(out, arr)
#         return arr


from pathlib import Path
import numpy as np
from mmpose.apis import MMPoseInferencer
from .base import PoseExtractor


# configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-coco.yml

inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w32_8xb64-210e_ubody-256x192',
)

# CFG  = "https://download.openmmlab.com/mmpose/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.py"
# CKPT = "https://download.openmmlab.com/mmpose/rtmpose/rtmpose-s_8xb256-420e_coco-256x192.pth"


CKPT = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth")
class RTMPoseExtractor(PoseExtractor):
    def __init__(self, cache_dir: str | Path = "cache/poses_rtmpose"):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        # det_model=None →  весь кадр bbox, чтобы не тянуть mmdet
        self.infer = MMPoseInferencer(
            pose2d="configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-coco.yml", pose2d_weights=CKPT,
            det_model=None,
            device='cpu')

    @property
    def name(self): return "rtmpose"

    def extract(self, video_path: Path) -> np.ndarray:
        out = self.cache_dir / f"{video_path.stem}.npy"
        if out.exists():
            return np.load(out)

        keypoints = []
        for res in self.infer(video_path, show_progress=False):
            if res['predictions']:
                keypoints.append(res['predictions'][0]['keypoints'])  # (17,3)
            else:
                keypoints.append(np.full((17,3), np.nan))
        arr = np.stack(keypoints)
        np.save(out, arr)
        return arr
