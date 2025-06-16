"""
weapon_gait/rnn/bilstm_pool_ftr.py
==================================
Bidirectional‑LSTM with Global **Average + Max** Pooling.
(похоже на GaitRNN‑V1, но упрощён для CPU)

Differences versus lstm_ftr.LSTMClassifier
-----------------------------------------
* три Bi‑LSTM слоя (128 hidden) → чуть более длинный контекст;
* **нет** pack_padded – работаем на zero‑padded последовательностях
  (это быстрее на GPU, чуть медленнее на CPU, но код проще);
* pooling = concat(AVG, MAX) ⇒ FC ⇒ sigmoid.

Reuse GaitSeq + collate_padded.
"""
from __future__ import annotations

from pathlib import Path
import logging, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import BinaryAUROC

from weapon_gait.rnn.lstm_ftr import GaitSeq, collate_padded

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
class BiLSTMPool(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128, layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.fc   = nn.Linear(hidden*4, 1)   # avg + max concat

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        # x: B×L×F (already padded), lens: B
        out,_ = self.lstm(x)                 # B×L×H*2
        mask = torch.arange(out.size(1), device=x.device)[None,:] < lens[:,None]
        out = out * mask.unsqueeze(2)
        avg = out.sum(1) / lens.unsqueeze(1)  # B×H*2
        mx  = out.max(1).values              # B×H*2
        vec = torch.cat([avg, mx], dim=1)    # B×H*4
        return torch.sigmoid(self.fc(vec)).squeeze(1)

# ─────────────────────────────────────────────────────────────

def train_bilstm_pool(manifest: Path, feat_backend: str = "statsplus",
                      epochs: int = 20, batch: int = 32, lr: float = 3e-4,
                      window: int = 150, device: str | None = None,
                      out_path: Path = Path("runs/bilstm_pool.pt")):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = GaitSeq(manifest, window=window, feat_backend=feat_backend)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    collate_fn=collate_padded, num_workers=0)
    feat_dim = ds[0][0].shape[1]

    model = BiLSTMPool(feat_dim).to(device)
    opt   = AdamW(model.parameters(), lr=lr)
    bce   = nn.BCELoss()
    auroc = BinaryAUROC().to(device)

    for ep in range(1, epochs+1):
        model.train(); losses=[]; aus=[]
        for x,lens,y in dl:
            x,lens,y = x.to(device), lens.to(device), y.float().to(device)
            p = model(x,lens)
            loss = bce(p, y)
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(loss.item()); aus.append(auroc(p.detach(), y).item())
        logger.info(f"BiLSTM E{ep:02d} loss={np.mean(losses):.4f} auroc={np.mean(aus):.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "feat_backend": feat_backend,
        "feat_dim": feat_dim,
        "window": window,
        "hidden": 128,
        "layers": 3
    }, out_path)
    logger.info(f"Saved → {out_path}")
    return out_path