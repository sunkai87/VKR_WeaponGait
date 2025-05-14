# # -------------------------------------------------
# # weapon_gait/pose/__init__.py — factory helper
# # -------------------------------------------------
# from .mediapipe_pose import MediaPipePose
# from .movenet_pose import MoveNetPose
# from .rtmpose import RTMPoseExtractor
# from .base import PoseExtractor

# def get_extractor(name: str) -> PoseExtractor:
#     match name.lower():
#         case "mediapipe":
#             return MediaPipePose()
#         case "movenet":
#             return MoveNetPose()
#         case "rtmpose":
#             return RTMPoseExtractor(checkpoint="checkpoints/rtmpose.pth")
#         case _:
#             raise ValueError(f"Unknown extractor {name}")

# __all__ = ["get_extractor", "MediaPipePose", "MoveNetPose", "RTMPoseExtractor", "PoseExtractor"]

# weapon_gait/pose/__init__.py
from importlib import import_module
from pathlib import Path
from typing import Protocol

class PoseExtractor(Protocol):
    def extract(self, video_path: Path): ...
    @property
    def name(self) -> str: ...

def _lazy(name: str, module: str, cls: str) -> PoseExtractor:
    return getattr(import_module(module, package=__name__), cls)()

def get_extractor(name: str) -> PoseExtractor:
    name = name.lower()
    match name:
        case "mediapipe":
            return _lazy(name, ".mediapipe_pose", "MediaPipePose")
        case "movenet":
            return _lazy(name, ".movenet_pose", "MoveNetPose")
        case "rtmpose":
            return _lazy(name, ".rtmpose", "RTMPoseExtractor")
        case "yolopose":
            return _lazy(name, ".yolopose","YOLOPoseExtractor")
        case "rtmpose":
            return _lazy(name, ".rtmpose", "RTMPoseExtractor")
        case "pifpaf":
            return _lazy(name, ".pifpaf_pose", "PifPafPose")
        case _:
            raise ValueError(f"Unknown pose backend: {name}")

__all__ = ["get_extractor", "PoseExtractor"]


