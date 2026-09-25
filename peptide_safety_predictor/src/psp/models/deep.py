"""Deep sequence models for HC50 (CNN / multiscale / BiLSTM) + fine-tune head."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psp.paths import AA20

AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA20)}  # 0 = pad


def encode_sequences(sequences: Sequence[str], max_len: int = 50) -> torch.Tensor:
    arr = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, s in enumerate(sequences):
        for j, a in enumerate(s[:max_len]):
            arr[i, j] = AA_TO_IDX.get(a, 0)
    return torch.from_numpy(arr)


class CNNRegressor(nn.Module):
    def __init__(self, embed_dim: int = 32, n_filters: int = 64, max_len: int = 50):
        super().__init__()
        self.embed = nn.Embedding(21, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, n_filters, kernel_size=5, padding=2)
        self.fc = nn.Sequential(
            nn.Linear(n_filters, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # mu, log_sigma
        )

    def forward(self, x):
        # x: (B, L)
        h = self.embed(x).transpose(1, 2)  # (B, E, L)
        h = F.relu(self.conv(h))
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        out = self.fc(h)
        return out[:, 0], out[:, 1]


class MultiscaleCNN(nn.Module):
    """LysePred-inspired multiscale CNN with kernels 2,4,8,16,32."""

    def __init__(self, embed_dim: int = 32, n_filters: int = 32):
        super().__init__()
        self.embed = nn.Embedding(21, embed_dim, padding_idx=0)
        self.kernels = [2, 4, 8, 16, 32]
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in self.kernels]
        )
        self.fc = nn.Sequential(
            nn.Linear(n_filters * len(self.kernels), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        h = self.embed(x).transpose(1, 2)
        parts = []
        for conv in self.convs:
            y = F.relu(conv(h))
            parts.append(F.adaptive_max_pool1d(y, 1).squeeze(-1))
        h = torch.cat(parts, dim=-1)
        out = self.fc(h)
        return out[:, 0], out[:, 1]


class BiLSTMRegressor(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(21, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden, batch_first=True, bidirectional=True, num_layers=1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        h = self.embed(x)
        lengths = (x != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, (hn, _) = self.lstm(packed)
        # concat last forward/backward
        h = torch.cat([hn[-2], hn[-1]], dim=-1)
        out = self.fc(h)
        return out[:, 0], out[:, 1]


def censored_nll_torch(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y_log: torch.Tensor,
    censor_code: torch.Tensor,
    censor_lower_log: torch.Tensor,
    censor_upper_log: torch.Tensor,
) -> torch.Tensor:
    """censor_code: 0=exact, 1=right, 2=left, 3=interval."""
    sigma = torch.exp(torch.clamp(log_sigma, -5, 5))
    normal = torch.distributions.Normal(mu, sigma)
    loss = torch.zeros_like(mu)
    exact = censor_code == 0
    if exact.any():
        loss = loss + torch.where(
            exact, -normal.log_prob(y_log), torch.zeros_like(mu)
        )
    right = censor_code == 1
    if right.any():
        # -log P(Y > c) = -log(1 - cdf(c))
        surv = 1.0 - normal.cdf(censor_lower_log)
        loss = loss + torch.where(
            right, -torch.log(surv.clamp(min=1e-12)), torch.zeros_like(mu)
        )
    left = censor_code == 2
    if left.any():
        cdf = normal.cdf(censor_upper_log)
        loss = loss + torch.where(
            left, -torch.log(cdf.clamp(min=1e-12)), torch.zeros_like(mu)
        )
    interval = censor_code == 3
    if interval.any():
        p = (normal.cdf(censor_upper_log) - normal.cdf(censor_lower_log)).clamp(min=1e-12)
        loss = loss + torch.where(interval, -torch.log(p), torch.zeros_like(mu))
    return loss.mean()


def censor_type_to_code(types) -> np.ndarray:
    m = {"exact": 0, "right": 1, "left": 2, "interval": 3}
    return np.asarray([m.get(t, 0) for t in types], dtype=np.int64)
