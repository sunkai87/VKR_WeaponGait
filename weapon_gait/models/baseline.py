
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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from weapon_gait.pose import get_extractor as get_pose_extractor
from weapon_gait.features.gait_features import get_extractor as get_gait_extractor
from weapon_gait.augment import generate_augmented
# ---------------------------------------------------------------------------
# Feature extraction & caching
# ---------------------------------------------------------------------------

def _cache_file(manifest: Path, pose_name: str, gait_name: str) -> Path:
    out = Path("cache") / f"features_{pose_name}_{gait_name}" / (manifest.stem + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def build_dataset(
    manifest_csv: Path,
    pose_backend: str = "mediapipe",
    gait_backend: str = "stats",
    augment_factor: int | None = None,  # e.g. {"weapon":3,"no_weapon":3}
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
        pose = pose_extractor.extract(vid)
        feats = gait_extractor.extract(pose)
        if not feats.get("valid", True):
            continue
        feats["label"] = label
        feats["pose_seq"] = pose
        rows.append(feats)
    
    df = pd.DataFrame(rows)

    if augment_factor > 0:
        aug_rows = []
        for _, r in df.iterrows():
            for a in generate_augmented(r["pose_seq"], n=augment_factor):
                f = gait_extractor.extract(a)
                f["label"] = r["label"]
                aug_rows.append(f)
        if aug_rows:
            df = pd.concat([df.drop(columns=["pose_seq"]),
                            pd.DataFrame(aug_rows)], ignore_index=True)
    else:
        df = df.drop(columns=["pose_seq"])

    # df.to_csv(cache_csv, index=False)
    # y = df.pop("label")
    # return df, y

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


def predict(video: Path, model_pth: Path):
    payload = joblib.load(model_pth)
    clf = payload["model"]
    pose_backend = payload["pose_backend"]
    gait_backend = payload["gait_backend"]
    cols: List[str] = payload["feature_cols"]

    pose = get_pose_extractor(pose_backend).extract(video)
    feats = get_gait_extractor(gait_backend).extract(pose)
    if not feats.get("valid", True):
        print("Pose extraction failed for", video)
        return None
    X = pd.DataFrame([{k: feats[k] for k in cols}])
    probs = clf.predict_proba(X)[0]
    pred = clf.classes_[probs.argmax()]
    print("Prediction:", pred, "|", dict(zip(clf.classes_, probs)))
    return pred

