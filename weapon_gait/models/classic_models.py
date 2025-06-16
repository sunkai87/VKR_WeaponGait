"""
weapon_gait/models/classic_models.py
===================================
Classic scikit‑learn models + optional GridSearch and plots
----------------------------------------------------------
* Random Forest  (rf)
* Histogram Gradient Boosting (gb)
* Linear SVM  calibrated  (svm)
* Gaussian Naïve Bayes (nb)
* Logistic Regression  (logreg)
* **Easy to extend** — add key in `_make_model()` + grid in `GRID_PARAMS`.

Extras
~~~~~~
* `--grid-search`  → GridSearchCV with cv=3, F1‑weighted.
* `--rebuild-features`  → удаляем кэш CSV, считаем заново.
* Авто‑имя файла:  `runs/<model_key>_<gait>.joblib` если `--model` не задан.
* PNG‑графики рядом с моделью:  `<root>.cm.png` (confusion‑matrix) и
  `<root>.roc.png` (ROC‑curve с AUC) — выключается флагом `--no-plots`.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, roc_auc_score)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weapon_gait.models.baseline import _cache_file, build_dataset

from shutil import rmtree

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Model factory and Grid params
# ---------------------------------------------------------------------------
# Можно добавить параметр:  class_weight="balanced"
def _make_model(model_key: str):
    if model_key == "rf":
        # clf = RandomForestClassifier(n_estimators=400, max_depth=None,n_jobs=-1)
        clf = RandomForestClassifier()
    elif model_key == "svm":
        base = LinearSVC()
        # clf = CalibratedClassifierCV(base, cv=3)
        clf = CalibratedClassifierCV(base)
    elif model_key == "nb":
        clf = GaussianNB()
        
    elif model_key == "logreg":
        # clf = LogisticRegression(max_iter=300,solver="liblinear")
        clf = LogisticRegression()
    elif model_key == "gb":
        # clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        clf = HistGradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model_key '{model_key}'")

    # ————————————————————————————————————————————————
    # Составляем шаги пайплайна
    steps = []

    # Модели, которым нужен импути-Nan → среднее по колонке
    if model_key in {"rf", "svm", "nb", "logreg"}:
        steps.append(("imputer", SimpleImputer(strategy="mean")))
        steps.append(("scaler", StandardScaler()))     # линейным моделям полезен масштаб

    # Очень странный параметр, он то позитивно влияет, то негативно влияет
    elif model_key == "gb":
        # HistGradientBoosting сам умеет NaN, масштаб не обязателен,
        # но можно оставить scaler — на результат почти не влияет.
        steps.append(("imputer", SimpleImputer(strategy="mean")))
        steps.append(("scaler", StandardScaler()))


    steps.append(("clf", clf))
    return Pipeline(steps)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

# search‑space per model
GRID_PARAMS: Dict[str, Dict[str, list]] = {
    "rf":  {"clf__n_estimators": [100, 200, 400],
             "clf__max_depth": [None, 10, 20],
             "clf__min_samples_leaf": [1, 2, 4]},
    # "svm": {"base__C": [1, 0.1, 10],
    #         "base__loss":["hinge","squared_hinge"],
    #         "base__penalty":["l1","l2"]},
    "svm":{},
    "nb":  {"clf__var_smoothing":[1e-9, 1e-10, 1e-8]},
    "logreg": {"clf__C": [1, 0.1, 10],
               "clf__penalty":["l1","l2"],
               "clf__solver":["liblinear"],
               "clf__max_iter":[100,200,400]},
    "gb":  {"clf__learning_rate": [0.1, 0.05, 0.2],
             "clf__max_depth": [None, 2, 4],
             "clf__max_iter":[100,200,400]},
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def train_model(
    manifest_csv: Path,
    pose_backend: str,
    gait_backend: str,
    model_key: str = "rf",
    out_pth: Path | None = None,
    test_size: float = 0.2,
    augment_factor: int | None = None,
    grid_search: bool = False,
    rebuild_features: bool = True,
    save_plots: bool = True,
    kfold: int = 1
):
    
    """Train model, print metrics, save .joblib (+ plots)."""

    # before loading dataset -------------------------------------------------
    if rebuild_features:
        feat_csv = _cache_file(manifest_csv, pose_backend, gait_backend)
        # 1) удаляем CSV, если лежит один файл
        feat_csv.unlink(missing_ok=True)
        # 2) на всякий случай – удаляем папку целиком
        feat_dir = feat_csv.parent
        if feat_dir.exists():
            rmtree(feat_dir, ignore_errors=True)

    X, y = build_dataset(manifest_csv, pose_backend, gait_backend,
                         augment_factor=augment_factor)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=0)

    model = _make_model(model_key)

    if grid_search and GRID_PARAMS.get(model_key):
        model = GridSearchCV(model, GRID_PARAMS[model_key], cv=3,
                             scoring="f1_weighted", n_jobs=-1, verbose=1)


    if kfold > 1:
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0)
        y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)

        if hasattr(model, "predict_proba"):
            proba_oof = cross_val_predict(model, X, y, cv=skf,method="predict_proba", n_jobs=-1)[:, 1]
        else:
            proba_oof = None

        print(f"{kfold}-fold CV metrics ({model_key}):\n",classification_report(y, y_pred))
        model.fit(X, y)          # финальное дообучение на всём
        cm_y_true, cm_y_pred = y, y_pred
        roc_scores = proba_oof

        # roc_scores = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        print(f"Validation metrics ({model_key}):\n",classification_report(y_val, y_pred))
        cm_y_true, cm_y_pred = y_val, y_pred
        roc_scores = (model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None)


    # ------------------------------------------------------------------
    if save_plots:
        root = (out_pth if out_pth else Path(f"runs/{model_key}_{gait_backend}.joblib")).with_suffix("")

        # Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(cm_y_true, cm_y_pred, ax=ax,
                                                cmap="Blues", colorbar=False)
        ax.set_title(f"{model_key.upper()} – Confusion Matrix")
        fig.tight_layout(); fig.savefig(root.with_suffix('.cm.png'))
        plt.close(fig)

        # ROC
        
        if roc_scores is not None:
            auc = roc_auc_score(cm_y_true.map({"no_weapon":0,"weapon":1}), roc_scores)
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(
                cm_y_true.map({"no_weapon":0,"weapon":1}), roc_scores,
                ax=ax, name=f"AUC={auc:.3f}")
            ax.set_title(f"ROC curve - {model_key.upper()}")
            fig.tight_layout(); fig.savefig(root.with_suffix(".roc.png")); plt.close(fig)

    # ------------------------------------------------------------------
    # choose output path
    if out_pth is None:
        out_pth = Path(f"runs/{model_key}_{gait_backend}.joblib")
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_key": model_key,
        "model": model,
        "pose_backend": pose_backend,
        "gait_backend": gait_backend,
        "feature_cols": list(X.columns),
    }, out_pth)
    print("Saved in", out_pth)

def predict_video(video: Path, model_pth: Path):
    payload = joblib.load(model_pth)
    model = payload["model"]
    pose_backend = payload["pose_backend"]
    gait_backend = payload["gait_backend"]
    cols: List[str] = payload["feature_cols"]

    from weapon_gait.pose import get_extractor
    from weapon_gait.features.gait_features import get_extractor as get_feat

    pose = get_extractor(pose_backend).extract(video)
    feats = get_feat(gait_backend).extract(pose)
    if not feats.get("valid", True):
        return None
    X = pd.DataFrame([{k: feats.get(k, np.nan) for k in cols}])
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    pred = model.predict(X)[0]
    print("Prediction:", pred, "probas", probs)
    return pred
