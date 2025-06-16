"""
weapon_gait/rnn/tcn_ftr.py
==========================
Temporal Convolutional Network (TCN) classifier over per–frame
`statsplus` features.

Architecture
~~~~~~~~~~~~
1. **TemporalConvNet** – 4 residual blocks with exponentially growing
   dilation (1, 2, 4, 8).  Kernel size 3, causal padding.
2. Global average‑and‑max pooling along time → concat.
3. Linear → sigmoid.

Parameters chosen для CPU‑дружественной скорости (≈90 k weights).
Reuse the same `GaitSeq` Dataset and `collate_padded` from lstm_ftr.py.
"""
from __future__ import annotations

from pathlib import Path
import logging, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import BinaryAUROC

from weapon_gait.rnn.lstm_ftr import GaitSeq, collate_padded  # reuse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# TCN building blocks
# ──────────────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__(); self.chomp = chomp
    def forward(self, x):
        return x[..., :-self.chomp] if self.chomp > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksz, stride, dil, pad, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, ksz,
                                                    stride=stride, padding=pad,
                                                    dilation=dil))
        self.chomp1 = Chomp1d(pad)
        self.relu1  = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, ksz,
                                                    stride=1, padding=pad,
                                                    dilation=dil))
        self.chomp2 = Chomp1d(pad)
        self.relu2  = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):          # B×C×L
        out = self.conv1(x)
        out = self.relu1(self.chomp1(out))
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(self.chomp2(out))
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channels, ksz=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dil = 2 ** i
            in_ch = num_inputs if i == 0 else channels[i-1]
            out_ch = channels[i]
            pad = (ksz - 1) * dil
            layers += [TemporalBlock(in_ch, out_ch, ksz, stride=1, dil=dil,
                                      pad=pad, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):          # B×C×L
        return self.network(x)

# ──────────────────────────────────────────────────────────────────────
class TCNClassifier(nn.Module):
    def __init__(self, feat_dim: int, channels: list[int] = [64,64,64,64]):
        super().__init__()
        self.tcn = TemporalConvNet(feat_dim, channels)
        self.fc  = nn.Linear(channels[-1]*2, 1)   # avg + max pooling

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        # x B×L×F → B×F×L
        x = x.transpose(1,2)
        out = self.tcn(x)               # B×C×L
        mask = torch.arange(out.size(2), device=out.device)[None,:] < lens[:,None]
        out = out * mask.unsqueeze(1)
        avg = out.sum(2) / lens.unsqueeze(1)   # B×C
        mx  = out.max(2).values                # B×C
        vec = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.fc(vec)).squeeze(1)

# ──────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────

def train_tcn(manifest: Path, feat_backend: str = "inst_stats",
              epochs: int = 20, batch: int = 32, lr: float = 1e-3,
              window: int = 150, device: str | None = None,
              out_path: Path = Path("runs/tcn_statsplus.pt")):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = GaitSeq(manifest, window=window, feat_backend=feat_backend)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    collate_fn=collate_padded, num_workers=0)
    feat_dim = ds[0][0].shape[1]

    model = TCNClassifier(feat_dim).to(device)
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
        logger.info(f"TCN E{ep:02d} loss={np.mean(losses):.4f} auroc={np.mean(aus):.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "feat_backend": feat_backend,
        "feat_dim": feat_dim,
        "window": window,
        "channels": [64,64,64,64]
    }, out_path)
    logger.info(f"Saved → {out_path}")
    return out_path