"""
weapon_gait/cli.py  (synchronised with current package)
-------------------------------------------------------
Click-based command-line interface.

Usage (PowerShell):
  # извлечь позы
  python -m weapon_gait.cli extract --video videos\n_001.mp4 --backend mediapipe

  # обучить RandomForest
  python -m weapon_gait.cli train   --manifest data\manifest.csv \
                                    --pose-backend mediapipe \
                                    --gait-backend stats \
                                    --model runs\rf.joblib

  # инференс на одном клипе
  python -m weapon_gait.cli predict --video videos\n_001.mp4 --model runs\rf.joblib
"""
from __future__ import annotations

from pathlib import Path

import click

from weapon_gait.pose import get_extractor as get_pose_extractor
from weapon_gait.features.gait_features import get_extractor as get_gait_extractor
from weapon_gait.models import baseline

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Weapon-Gait toolbox."""

# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@cli.command(help="Extract poses and cache as .npy")
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", default="mediapipe", show_default=True, help="Pose backend")
def extract(video: str, backend: str):
    extractor = get_pose_extractor(backend)
    arr = extractor.extract(Path(video))
    click.echo(f"{extractor.name}: extracted pose seq of shape {arr.shape}")

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command(help="Train RandomForest baseline on hand-crafted gait features")
@click.option("--manifest", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--pose-backend", default="mediapipe", show_default=True)
@click.option("--gait-backend", default="stats", show_default=True)
@click.option("--model", default="runs/rf.joblib", show_default=True, type=click.Path(dir_okay=False))
@click.option("--test-size", default=0.2, show_default=True, help="Validation split ratio")
@click.option("--augment-factor", default=None, show_default = True, type = int) # I edit there

def train(manifest: str, pose_backend: str, gait_backend: str, model: str, test_size: float, augment_factor: int | None):
    baseline.train(Path(manifest), Path(model), pose_backend, gait_backend, test_size, augment_factor)

# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@cli.command(help="Predict weapon/no_weapon on a single video clip")
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
def predict(video: str, model: str):
    baseline.predict(Path(video), Path(model))

# ---------------------------------------------------------------------------

# ------------------------------------------------------------
#  train-ml   (классические модели: rf / svm / nb / logreg / gb)
# ------------------------------------------------------------
@cli.command(name="train-ml",
             help="Train a classic scikit-learn model "
                  "(rf | svm | nb | logreg | gb)")
@click.option("--manifest",      required=True,type=click.Path(exists=True, dir_okay=False))
@click.option("--pose-backend",  default="mediapipe", show_default=True)
@click.option("--gait-backend",  default="stats",     show_default=True)
@click.option("--model-key",     default="gb",       show_default=True,
              type=click.Choice(["rf", "svm", "nb", "logreg", "gb"]))
@click.option("--model",         default=None, type=click.Path(dir_okay=False), help="Output .joblib path (auto if omitted)")
@click.option("--test-size",     default=0.2, show_default=True)
@click.option("--augment-factor", default=None, show_default=True, type=int)
@click.option("--grid-search/--no-grid-search", default=False, help="Run GridSearchCV before final fit")
@click.option("--rebuild-features/--no-rebuild-features",default=True, help="Force re-compute feature CSV")
@click.option("--plots/--no-plots", default=True,help="Save confusion-matrix and ROC plots")
@click.option("--kfold", default=1, type = int, show_default=True,help="Stratified k-fold CV (k>1)")

def train_ml(manifest, pose_backend, gait_backend,
             model_key, model, test_size,grid_search,rebuild_features,plots,augment_factor,kfold): 
    """Thin wrapper around classic_models.train_model()."""
    from weapon_gait.models.classic_models import train_model
    
    train_model(Path(manifest),
                pose_backend=pose_backend,
                gait_backend=gait_backend,
                model_key=model_key,
                out_pth=model,
                test_size=test_size,
                grid_search = grid_search,
                rebuild_features=rebuild_features,
                save_plots=plots,
                augment_factor=augment_factor,
                kfold=kfold
                )

if __name__ == "__main__":
    cli()
