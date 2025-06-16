
"""
weapon_gait/models/baseline.py  (complete implementation)
---------------------------------------------------------
Lightweight RandomForest baseline that learns on
hand‑crafted gait features.

Highlights
~~~~~~~~~~
* **Caching** feature tables to `cache/features_<pose>_<gait>.csv` so that
  repeated training is instant.
* **Class‑imbalance aware** — uses `class_weight="balanced"` and prints
  per‑class metrics.
* Works fully on CPU; zero external deps beyond scikit‑learn.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from weapon_gait.pose import get_extractor as get_pose_extractor
from weapon_gait.features.gait_features import get_extractor as get_gait_extractor
from weapon_gait.augment import generate_augmented
from weapon_gait.render_predict import render_prediction


logger = logging.getLogger(__name__)       # ← берём по имени модуля
logger.setLevel(logging.INFO)              # уровень по-умолчанию


# чтобы сообщения сразу появлялись в консоли
if not logging.getLogger().handlers:       # настроено ли уже?
    logging.basicConfig(
        format="%(levelname)s | %(name)s | %(message)s",
        level=logging.INFO)
# ---------------------------------------------------------------------------
# Feature extraction & caching
# ---------------------------------------------------------------------------

def _cache_file(manifest: Path, pose_name: str, gait_name: str) -> Path:
    out = Path("cache") / f"features_{pose_name}_{gait_name}" / (manifest.stem + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def _flatten_feats(feats: dict) -> dict:
    """Разворачиваем векторные фичи в scaler-friendly словарь"""
    flat = {}
    for k, v in feats.items():
        if np.isscalar(v) or v is None:
            flat[k] = v
        elif isinstance(v, (list, tuple, np.ndarray)):
            v = np.asarray(v)
            if v.size == 1:              # случай [[val]]
                flat[k] = float(v.ravel()[0])
            else:                        # многомерная фича → k_i
                for i, val in enumerate(v.ravel()):
                    flat[f"{k}_{i}"] = float(val)
        else:
            raise TypeError(f"Unsupported type for feature {k}: {type(v)}")
    return flat


def build_dataset(
    manifest_csv: Path,
    pose_backend: str = "mediapipe",
    gait_backend: str = "stats",
    augment_factor: int | None = None,
    augment_if_label: str = "weapon"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    augment_factors > 0  →  для каждого примера
    сгенерировать N синтетических последовательностей.
    """
    manifest_csv = Path(manifest_csv)
    # tag = "_aug" if augment_factor else ""
    cache_csv = _cache_file(manifest_csv, pose_backend, gait_backend)
    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        y = df.pop("label")
        return df, y

    pose_extractor = get_pose_extractor(pose_backend)
    gait_extractor = get_gait_extractor(gait_backend)

    rows: List[dict] = []
    for _, row in pd.read_csv(manifest_csv).iterrows():
        vid = Path(row["video"])
        label = row["label"]

        try:
            poses = pose_extractor.extract(vid)
        except Exception as e:
            logger.warning(f"skip {vid.name}: {e}")
            continue

        poses = pose_extractor.extract(vid)
        if not isinstance(poses, list):
            poses = [poses]


        for pose in poses:
            feats = gait_extractor.extract(pose)
            if not feats.get("valid", True):
                continue

            feats = _flatten_feats(feats)
            feats["label"] = label
            # feats["pose_seq"] = pose #А это надо? Это надо включать при аугментации на 2 класса и выключать на 1 класс
            rows.append(feats)

            # аугментация на 1 клсс weapon
            if augment_factor and label == augment_if_label:
                for aug_seq in generate_augmented(pose, n=augment_factor):
                    aug_feats = gait_extractor.extract(aug_seq)
                    if aug_feats.get("valid", True):
                        aug_feats["label"] = label
                        rows.append(aug_feats)
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Dataset is empty: no valid pose sequences extracted.")


    # Аугментация на 2 класса:    
    # if augment_factor > 0:
    #     aug_rows = []
    #     for _, r in df.iterrows():
    #         for a in generate_augmented(r["pose_seq"], n=augment_factor):
    #             f = gait_extractor.extract(a)
    #             f["label"] = r["label"]
    #             aug_rows.append(f)
    #     if aug_rows:
    #         df = pd.concat([df.drop(columns=["pose_seq"]),
    #                         pd.DataFrame(aug_rows)], ignore_index=True)
    # else:
    #     df = df.drop(columns=["pose_seq"])


    # df.to_csv(cache_csv, index=False)
    # y = df.pop("label")
    # return df, y

    if "label" not in df.columns:
        raise RuntimeError("Column 'label' missing after feature extraction.")

    X = df.drop(columns=["label"])
    y = df["label"]

    cfile = _cache_file(manifest_csv, pose_backend, gait_backend)
    cfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfile, index=False)
    return X, y


# ---------------------------------------------------------------------------
# Training & predict wrappers
# ---------------------------------------------------------------------------

def train(
    manifest_csv: Path,
    model_out: Path = Path("runs/rf.joblib"),
    pose_backend: str = "mediapipe",
    gait_backend: str = "stats",
    test_size: float = 0.2,
    augment_factor: int | None = None
):
    X, y = build_dataset(manifest_csv, pose_backend, gait_backend, augment_factor=augment_factor)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    print("Validation metrics:\n", classification_report(y_val, clf.predict(X_val)))
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": clf,
        "pose_backend": pose_backend,
        "gait_backend": gait_backend,
        "feature_cols": list(X.columns),
    }, model_out)
    print("Saved →", model_out)


# def predict(video: Path, model_pth: Path):
#     payload = joblib.load(model_pth)
#     clf = payload["model"]
#     pose_backend = payload["pose_backend"]
#     gait_backend = payload["gait_backend"]
#     cols: List[str] = payload["feature_cols"]

#     pose = get_pose_extractor(pose_backend).extract(video)
#     feats = get_gait_extractor(gait_backend).extract(pose)
#     if not feats.get("valid", True):
#         print("Pose extraction failed for", video)
#         return None
#     X = pd.DataFrame([{k: feats[k] for k in cols}])
#     probs = clf.predict_proba(X)[0]
#     pred = clf.classes_[probs.argmax()]
#     print("Prediction:", pred, "|", dict(zip(clf.classes_, probs)))
#     return pred

def predict(video: Path, model_pth: Path, save_vis: Path | None = None):
    payload = joblib.load(model_pth)
    model   = payload["model"]
    pose_be = payload["pose_backend"]
    gait_be = payload["gait_backend"]
    cols    = payload["feature_cols"]

    from weapon_gait.features.gait_features import get_extractor as gaitx
    gait_extractor = gaitx(gait_be)

    # --- STEP A: get pose sequences ---------------------------------
    if pose_be == "npy":          # our multi-track case
        from weapon_gait.pose import get_extractor
        mp_extractor = get_extractor("crop_mp")
        npy_paths, frame_boxes = mp_extractor.extract(video, return_boxes=True)
        # npy_paths = mp_extractor.extract(video)     # list[Path]

        pid2seq = {}
        for p in npy_paths:
            pid = int(p.stem.split('_pid')[1])          # извлекаем ID
            pid2seq[pid] = np.load(p, allow_pickle=False) 

        pose_list = [(int(p.stem.split('_pid')[1]), np.load(p)) for p in npy_paths]
    else:
        from weapon_gait.pose import get_extractor
        pose = get_extractor(pose_be).extract(video)
        pose_list = [(0, pose)]                    # single person

    # --- STEP B: per-person features & predict ----------------------
    results = []
    for pid, pose_seq in pose_list:
        feats = gait_extractor.extract(pose_seq)
        if not feats.get("valid", True):
            continue
        X = pd.DataFrame([{k: feats.get(k, np.nan) for k in cols}])
        prob = (model.predict_proba(X)[0][1]
                if hasattr(model, 'predict_proba') else None)
        pred = model.predict(X)[0]
        results.append({"pid": pid, "pred": pred, "prob": prob})

    # --- STEP C: optional visualisation -----------------------------
    # if save_vis and results:
    #     from weapon_gait.draw_text_vis import draw_text_vis
    #     draw_text_vis(
    #         video_path = video,
    #         pid2seq    = pid2seq,
    #         pid_pred   = {r['pid']: (r['pred'], r['prob']) for r in results},
    #         out_mp4    = save_vis
    #     )
        if save_vis and results and pose_be == "npy":
            from weapon_gait.draw_bbox_vis import draw_bbox_vis
            draw_bbox_vis(
                video_path = video,
                frame_boxes = frame_boxes,          # ← получены с return_boxes=True
                pid_pred = {r["pid"]: (r["pred"], r["prob"]) for r in results},
                out_mp4 = save_vis)
    return results
    
    # print(results)
    # return results