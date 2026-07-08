"""
Two-tower retrieval model.

The user tower and ad tower never see each other's raw inputs -- only their
dot product is trained to predict click. That separation is the entire
point: once trained, ad embeddings can be precomputed and indexed once
offline, so scoring "this user against 1M ads" at serve time collapses to
one user-tower forward pass plus an approximate nearest-neighbor lookup,
instead of 1M GBDT evaluations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    """Trained with in-batch sampled softmax: for a batch of B (user, ad)
    positive pairs, each user's positive ad is contrasted against the other
    B-1 ads in the batch as implicit negatives -- no explicit negative
    mining needed."""

    def __init__(self, user_dim: int, ad_dim: int, embed_dim: int = 32):
        super().__init__()
        self.user_tower = Tower(user_dim, out_dim=embed_dim)
        self.ad_tower = Tower(ad_dim, out_dim=embed_dim)

    def forward(self, user_x: torch.Tensor, ad_x: torch.Tensor):
        return self.user_tower(user_x), self.ad_tower(ad_x)

    def in_batch_loss(self, user_x: torch.Tensor, ad_x: torch.Tensor) -> torch.Tensor:
        u, a = self(user_x, ad_x)
        logits = u @ a.T / 0.07  # temperature-scaled cosine similarity
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def train_two_tower(
    user_x: np.ndarray, ad_x: np.ndarray, epochs: int = 5, batch_size: int = 512, lr: float = 1e-3
) -> TwoTowerModel:
    model = TwoTowerModel(user_dim=user_x.shape[1], ad_dim=ad_x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ux = torch.tensor(user_x, dtype=torch.float32)
    ax = torch.tensor(ad_x, dtype=torch.float32)
    n = ux.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - batch_size, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = model.in_batch_loss(ux[idx], ax[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def embed_ads(model: TwoTowerModel, ad_x: np.ndarray) -> np.ndarray:
    model.eval()
    return model.ad_tower(torch.tensor(ad_x, dtype=torch.float32)).numpy()


@torch.no_grad()
def embed_users(model: TwoTowerModel, user_x: np.ndarray) -> np.ndarray:
    model.eval()
    return model.user_tower(torch.tensor(user_x, dtype=torch.float32)).numpy()
