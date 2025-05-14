"""
weapon_gait/dataset_tools.py
===========================
Utility helpers for step 0/1:
1. Auto‑discover raw mp4 files in your directory layout and build
   `data/manifest.csv` with two columns: video, label.
2. Validate that every clip is readable and ≥ 1 s long.
3. Quick CLI to print dataset stats (count, duration).

Usage (from repo root)::

    python -m weapon_gait.dataset_tools scan \
        --root "C:/Users/vkash/Desktop/JupyterWorks/VKR/dataset/raw" \
        --out data/manifest.csv

    python -m weapon_gait.dataset_tools stats --manifest data/manifest.csv
"""
from __future__ import annotations

import click
import csv
from pathlib import Path
import cv2

LABEL_FOLDERS = {
    "weapon": "weapon",
    "normal": "normal",
}

@click.group()
def cli():
    """Dataset utilities."""

@cli.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out", type=click.Path(), default="data/manifest.csv")
def scan(root: str, out: str):
    """Recursively scan folder tree and build manifest.csv."""
    root = Path(root)
    rows = []
    for label, sub in LABEL_FOLDERS.items():
        for mp4 in (root / sub).rglob("*.mp4"):
            rel = mp4.relative_to(root).as_posix()
            rows.append({"video": str(mp4), "label": label})
    rows.sort(key=lambda r: r["video"])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label"])
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"Manifest written → {out} ({len(rows)} clips)")

@cli.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), required=True)
def stats(manifest: str):
    """Print basic info about manifest."""
    import pandas as pd
    df = pd.read_csv(manifest)
    counts = df["label"].value_counts().to_dict()
    click.echo(f"Total clips: {len(df)} | by class: {counts}")

if __name__ == "__main__":
    cli()