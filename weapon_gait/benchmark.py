"""
weapon_gait/benchmark.py
=======================
End‑to‑end benchmarking script that unifies dataset preparation, training
(optional) and evaluation for multiple skeleton‑action back‑ends
(PoseC3D, ST‑GCN, 2S‑AGCN).

Designed to be launched from command line:

```bash
python -m weapon_gait.benchmark \
    --manifest data/manifest.csv \
    --pose-cache cache/poses_mp \
    --out results.csv \
    --train   # include full training run (GPU required)
```

If `--train` не указан, ожидаются готовые чек‑пойнты в `runs/<model>/`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import click

from weapon_gait.models.posec3d_pipeline import (
    build_posec3d_dataset, render_config as cfg_posec3d,
    train_posec3d, evaluate_posec3d,
)
from weapon_gait.models.stgcn_pipeline import (
    build_stgcn_dataset, render_config as cfg_stgcn,
    train_stgcn, evaluate_stgcn,
)
from weapon_gait.models.agcn_pipeline import (
    build_agcn_dataset, render_config as cfg_agcn,
    train_agcn, evaluate_agcn,
)

BACKENDS = {
    "posec3d": {
        "build_dataset": build_posec3d_dataset,
        "render_cfg": cfg_posec3d,
        "train": train_posec3d,
        "eval": evaluate_posec3d,
        "work_dir": Path("runs/posec3d"),
        "epochs": 50,
    },
    "stgcn": {
        "build_dataset": build_stgcn_dataset,
        "render_cfg": cfg_stgcn,
        "train": train_stgcn,
        "eval": evaluate_stgcn,
        "work_dir": Path("runs/stgcn"),
        "epochs": 80,
    },
    "agcn": {
        "build_dataset": build_agcn_dataset,
        "render_cfg": cfg_agcn,
        "train": train_agcn,
        "eval": evaluate_agcn,
        "work_dir": Path("runs/agcn"),
        "epochs": 100,
    },
}

RESULT_KEYS = ["top1_acc", "top5_acc", "mean_class_accuracy", "mAP"]


def parse_eval_log(work_dir: Path) -> Dict[str, float]:
    """Reads MMEngine json log file to pull last eval metrics."""
    log_file = next(work_dir.glob("*.log.json"))
    metrics = {}
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "mode" in j and j["mode"] == "test" and all(k in j for k in RESULT_KEYS):
                metrics = {k: j.get(k) for k in RESULT_KEYS}
    return metrics

@click.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--pose-cache", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out", type=click.Path(), default="benchmark_results.csv")
@click.option("--train/--no-train", default=False, help="Whether to run full training")
@click.option("--backend", multiple=True, default=["posec3d", "stgcn", "agcn"], help="Subset backends")
def main(manifest, pose_cache, out, train, backend):
    manifest = Path(manifest)
    pose_cache = Path(pose_cache)
    results = []

    for name in backend:
        cfgs = BACKENDS[name]
        work_dir = cfgs["work_dir"]
        data_dir = Path("data") / name
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1) dataset
        cfgs["build_dataset"](manifest, pose_cache, data_dir)

        # 2) config
        cfg_path = cfgs["render_cfg"](num_classes=2, data_root=data_dir, work_dir=work_dir)

        ckpt = work_dir / f"epoch_{cfgs['epochs']}.pth"
        if train or not ckpt.exists():
            click.echo(f"[+] Training {name} for {cfgs['epochs']} epochs…")
            cfgs["train"](cfg_path, max_epochs=cfgs["epochs"])
        else:
            click.echo(f"[=] Using existing checkpoint {ckpt}")

        # 3) evaluation
        click.echo(f"[+] Evaluating {name}")
        cfgs["eval"](cfg_path, ckpt)

        metrics = parse_eval_log(work_dir)
        metrics["backend"] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    click.echo(f"Saved benchmark results → {out}")

if __name__ == "__main__":
    main()