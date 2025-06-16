"""
weapon_gait/rnn_cli.py
======================
Adds CLI commands to train / predict sequence models (LSTM, GRU, TCN,
BiLSTMPool) over per‑frame gait features.

Usage examples
--------------
# train LSTM on inst_stats
python -m weapon_gait.rnn_cli train --arch lstm \
            --manifest data/manifest_crop_npy.csv \
            --feat-backend inst_stats --epochs 20 --batch 64

# predict video using trained gru model
python -m weapon_gait.rnn_cli predict \
            --video videos/n_130.mp4 \
            --model runs/gru_statsplus.pt
"""
import click
from pathlib import Path
import pandas as pd
import torch, numpy as np

# import training helpers
from weapon_gait.rnn.lstm_ftr      import train_lstm, LSTMClassifier
from weapon_gait.rnn.gru_ftr       import train_gru, GRUClassifier
from weapon_gait.rnn.tcn_ftr       import train_tcn, TCNClassifier
from weapon_gait.rnn.bilstm_ftr import train_bilstm_pool, BiLSTMPool
# dataset
from weapon_gait.rnn.lstm_ftr import GaitSeq, collate_padded
from torch.utils.data import DataLoader

ARCH_TABLE = {
    "lstm":  (train_lstm,      LSTMClassifier),
    "gru":   (train_gru,       GRUClassifier),
    "tcn":   (train_tcn,       TCNClassifier),
    "bilstm":(train_bilstm_pool,BiLSTMPool),
}

def _load_model(model_pth: Path):
    payload = torch.load(model_pth, map_location="cpu")
    feat_dim = payload["feat_dim"]
    arch = {
        "lstm":     LSTMClassifier,
        "gru":      GRUClassifier,
        "tcn":      TCNClassifier,
        "bilstm":   BiLSTMPool,
    }
    # detect arch by filename prefix or stored flag
    key = model_pth.stem.split("_")[0].replace("bilstm","bilstm")
    cls = arch.get(key)
    model = cls(feat_dim)
    model.load_state_dict(payload["model_state"]) 
    model.eval()
    return model, payload

@click.group()
def cli():
    """RNN/TCN training & inference commands"""

# ───────────────────────── train ──────────────────────────────
@cli.command("train")
@click.option("--arch", type=click.Choice(list(ARCH_TABLE)), required=True)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--feat-backend", default="inst_stats", show_default=True)
@click.option("--epochs", default=20, show_default=True)
@click.option("--batch",  default=32, show_default=True)
@click.option("--window", default=150, show_default=True)
@click.option("--lr",     default=3e-4, show_default=True)
@click.option("--out",    type=click.Path(dir_okay=False), default=None,
              help="Path to save .pt file (optional)")
@click.option("--eval", is_flag=True, help="Run AUROC & CM on hold-out 20% after training")
def train_cli(arch, manifest, feat_backend, epochs, batch, window, lr, out,eval):
    train_fn,_ = ARCH_TABLE[arch]
    out_pth = Path(out) if out else Path(f"runs/{arch}_{feat_backend}.pt")

    if eval:
    # simple 80/20 split on the same manifest
        df = pd.read_csv(manifest)
        val_df = df.sample(frac=0.2, random_state=42)
        val_df.to_csv("tmp_val.csv", index=False)
        ds_val = GaitSeq(Path("tmp_val.csv"), window=window, feat_backend=feat_backend)
        dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False,collate_fn=collate_padded, num_workers=0)

        model, _ = _load_model(out_pth)
        evaluate_model(model.to(device), dl_val, device=device)

    train_fn(Path(manifest), feat_backend=feat_backend, epochs=epochs,
             batch=batch, window=window, lr=lr, out_path=out_pth)

# ───────────────────────── predict ────────────────────────────
@cli.command("predict")
@click.option("--video", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--model", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--pid",   type=int, default=None, help="Optional single pid npy file instead of full video")
def predict_cli(video, model, pid):
    """Predict weapon / no_weapon for each person in video (multi‑people).
    Currently supports npy backend only (pose already extracted)."""
    model, meta = _load_model(Path(model))
    from weapon_gait.pose import get_extractor

    with torch.no_grad():
        if pid is not None:
            seq = np.load(Path(pid), allow_pickle=False)
            ds  = [ (torch.tensor(meta_feat(seq),dtype=torch.float32), len(seq)) ]
        else:
            mp = get_extractor("crop_mp_multi")
            npys,_ = mp.extract(Path(video), return_boxes=False)
            ds = []
            for p in npys:
                seq = np.load(p, allow_pickle=False)
                ds.append((torch.tensor(meta_feat(seq),dtype=torch.float32), len(seq), p))
        model.eval()
        for seq,l in ds:
            seq = seq.unsqueeze(0)
            prob = model(seq, torch.tensor([l]))
            print(f"{seq}: weapon_prob={prob.item():.2f}")

# helper to compute per‑frame inst_stats (lazy import)

def meta_feat(pose_seq: np.ndarray):
    from weapon_gait.features.gait_features import InstantStatsExtractor
    inst = InstantStatsExtractor()
    return inst.extract_seq(pose_seq)

import sklearn.metrics as skm

def evaluate_model(model, dl, device="cpu"):
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,lens,y in dl:
            x,lens = x.to(device), lens.to(device)
            p = model(x,lens).cpu().numpy()
            y_pred.extend(p)
            y_true.extend(y.numpy())
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    au = skm.roc_auc_score(y_true, y_pred)
    cm = skm.confusion_matrix(y_true, y_pred_bin)
    print(f"AUROC: {au:.3f}\nConfusion Matrix:\n{cm}")

if __name__ == "__main__":
    cli()