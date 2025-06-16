"""
weapon_gait/rnn/gru_ftr.py
==========================
GRU-based sequence classifier for gait features (statsplus).

* Uses the same `GaitSeq` Dataset and `collate_padded` from lstm_ftr.py
  (imported to avoid duplication).
* Architecture: 2‑layer bidirectional GRU → Multihead Self‑Attention
  (optional) → global average + max pooling → FC → sigmoid.
* `train_gru()` – helper analogous to `train_lstm()`.

Note: torchmetrics 0.11.4 works with Torch 1.13.1.
"""
from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import BinaryAUROC

from weapon_gait.rnn.lstm_ftr import GaitSeq, collate_padded  # re‑use

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128, layers: int = 2,
                 attn_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden*2, num_heads=attn_heads,
                                          batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden*4, 1)  # avg+max concat

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        # x: B×L×F
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(),
                                                   batch_first=True,
                                                   enforce_sorted=False)
        out,_  = self.gru(packed)
        out,_  = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # B×L×H*2

        # self‑attention (mask padded positions)
        mask = torch.arange(out.size(1), device=out.device)[None,:] >= lens[:,None]
        attn_out,_ = self.attn(out, out, out, key_padding_mask=mask)

        # global pooling
        attn_out = attn_out.masked_fill(mask.unsqueeze(2), 0.0)
        avg = attn_out.sum(1) / lens.unsqueeze(1)              # B×H*2
        mx  = attn_out.max(1).values                           # B×H*2
        vec = torch.cat([avg, mx], dim=1)                      # B×H*4
        return torch.sigmoid(self.fc(vec)).squeeze(1)

# ────────────────────────────────────────────────────────────────────────
# Training helper
# ────────────────────────────────────────────────────────────────────────

def train_gru(manifest: Path, feat_backend: str = "inst_stats",
              epochs: int = 20, batch: int = 32, lr: float = 3e-4,
              window: int = 150, device: str | None = None,
              out_path: Path = Path("runs/gru_statsplus.pt")):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = GaitSeq(manifest, window=window, feat_backend=feat_backend)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    collate_fn=collate_padded, num_workers=0)
    feat_dim = ds[0][0].shape[1]

    model = GRUClassifier(feat_dim).to(device)
    opt   = AdamW(model.parameters(), lr=lr)
    bce   = nn.BCELoss()
    auroc = BinaryAUROC().to(device)

    for ep in range(1, epochs+1):
        model.train(); loss_run=[]; au_run=[]
        for x,lens,y in dl:
            x,lens,y = x.to(device), lens.to(device), y.float().to(device)
            p = model(x,lens)
            loss = bce(p, y)
            loss.backward(); opt.step(); opt.zero_grad()
            loss_run.append(loss.item())
            au_run.append(auroc(p.detach(), y).item())
        logger.info(f"GRU E{ep:02d} loss={np.mean(loss_run):.4f} auroc={np.mean(au_run):.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "feat_backend": feat_backend,
        "feat_dim": feat_dim,
        "window": window
    }, out_path)
    logger.info(f"Saved → {out_path}")
    return out_path