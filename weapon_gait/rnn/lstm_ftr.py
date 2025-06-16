"""
weapon_gait/rnn/lstm_ftr.py
===========================
Self‑contained implementation of
* GaitSeq    – torch Dataset that streams **statsplus** feature vectors of
               fixed‑length windows.
* LSTMClassifier – bidirectional LSTM + global pooling.
* train_lstm()  – convenience function; can be called from CLI.

Assumptions
-----------
* manifest CSV has columns   video,label   where *video* points to
  npy‑file (one person) OR mp4 (multi‑people) – we rely on existing
  pose / feature extractors.
* statsplus.extract_seq(arr)  already exists and returns  T×F  matrix
  (per‑frame feature vector, F≈80).
* PyTorch 1.13+ is available.
"""
from __future__ import annotations

from pathlib import Path
import logging, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torchmetrics.classification import BinaryAUROC

# import gait feature extractor
from weapon_gait.features.gait_features import get_extractor as gaitx
# pose backend for npy files only; other backends can be added later

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────
class GaitSeq(Dataset):
    """Stream per‑frame statsplus features.
    Parameters
    ----------
    manifest : Path
        CSV with columns video,label (weapon/no_weapon)
    window : int
        Optional fixed max length; sequences > window are center‑cropped;
        shorter ones are kept as‑is (pad in collate).
    feat_backend : str
        e.g. "statsplus" or "stats" – passed to gait feature extractor.
    """
    def __init__(self, manifest: Path, window: int = 150, feat_backend: str = "inst_stats"):
        self.rows = pd.read_csv(manifest)
        self.win  = window
        self.gait_ex = gaitx(feat_backend)

    def __len__(self):
        return len(self.rows)

    def _load_pose(self, p: Path) -> np.ndarray:
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)          # T×33×4
        raise ValueError("Only .npy supported in LSTM stage")

    def __getitem__(self, idx):
        row  = self.rows.iloc[idx]
        pose = self._load_pose(Path(row["video"]))
        feats = self.gait_ex.extract_seq(pose)             # T×F
        if np.isnan(feats).all():
            raise ValueError("all-NaN sequence")  # Dataset выпадет, DataLoader поймает
        
        if feats.ndim == 1:
            feats = feats[None, :]                         # 1×F edge‑case
        # center crop / trim to self.win
        if feats.shape[0] > self.win:
            start = (feats.shape[0] - self.win) // 2
            feats = feats[start:start+self.win]
        lens  = feats.shape[0]
        feats = torch.from_numpy(feats).float()            # L×F
        label = torch.tensor(1 if row.label == "weapon" else 0, dtype=torch.long)
        return feats, lens, label


def collate_padded(batch):
    seqs, lens, labels = zip(*batch)
    lens   = torch.tensor(lens, dtype=torch.long)
    pad_seqs = pad_sequence(seqs, batch_first=True)        # B×L×F
    labels = torch.stack(labels)
    return pad_seqs, lens, labels

# ────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(hidden*2, 1)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        # x B×L×F, lens B
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out,_  = self.lstm(packed)
        out,_  = pad_packed_sequence(out, batch_first=True)   # B×L×H*2
        # mask invalid steps
        mask = torch.arange(out.size(1), device=out.device)[None,:] < lens[:,None]
        out  = out * mask.unsqueeze(2)                       # zero‑out padded positions
        # max‑pool over time
        vec  = self.pool(out.transpose(1,2)).squeeze(2)      # B×H*2
        return torch.sigmoid(self.fc(vec)).squeeze(1)        # B

# ────────────────────────────────────────────────────────────────────────
# Training loop helper
# ────────────────────────────────────────────────────────────────────────

def train_lstm(manifest: Path, feat_backend: str = "inst_stats",
               epochs: int = 20, batch: int = 32, lr: float = 1e-3,
               window: int = 150, device: str | None = None,
               out_path: Path = Path("runs/lstm_statsplus.pt")):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds  = GaitSeq(manifest, window=window, feat_backend=feat_backend)
    dl  = DataLoader(ds, batch_size=batch, shuffle=True,
                     collate_fn=collate_padded, num_workers=0)
    feat_dim = ds[0][0].shape[1]

    model = LSTMClassifier(feat_dim).to(device)
    opt   = AdamW(model.parameters(), lr=lr)
    bce   = nn.BCELoss()
    auroc = BinaryAUROC().to(device)

    for ep in range(1, epochs+1):
        model.train(); running = []
        for x,lens,y in dl:
            x,lens,y = x.to(device), lens.to(device), y.float().to(device)
            p = model(x,lens)
            loss = bce(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())
        logger.info(f"E{ep:02d} loss={np.mean(running):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "feat_backend": feat_backend,
        "feat_dim": feat_dim,
        "window": window
    }, out_path)
    logger.info(f"Saved → {out_path}")

    return out_path