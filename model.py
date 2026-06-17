"""Point cloud models for protein-ligand binding affinity regression.

Two backbones are provided:

* ``PointNet``           - shared per-point MLPs + global max-pool (Qi et al. 2017).
* ``PointTransformer``   - local vector self-attention over kNN neighborhoods
                           (Zhao et al. 2021), simplified for set regression.

Both expose the same forward signature expected by the API:

    pred = model(coords, features)
        coords:   (B, N, 3)   xyz positions
        features: (B, N, F)   per-point features (atom type, charge, ...)
        returns:  (B,)        scalar affinity per complex

Use ``build_model(name, **kwargs)`` to construct by string name.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointNet", "PointTransformer", "build_model"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _mlp(sizes: list[int], last_act: bool = True) -> nn.Sequential:
    """Pointwise MLP (1x1 convs) with BatchNorm + ReLU between layers."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=1, bias=False))
        is_last = i == len(sizes) - 2
        if not is_last or last_act:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _knn(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the k nearest neighbors for each point.

    coords: (B, N, 3) -> idx: (B, N, k)
    """
    # Pairwise squared distances via (a-b)^2 = a^2 - 2ab + b^2.
    inner = torch.bmm(coords, coords.transpose(1, 2))           # (B, N, N)
    sq = (coords ** 2).sum(dim=-1, keepdim=True)                # (B, N, 1)
    dist = sq - 2 * inner + sq.transpose(1, 2)                  # (B, N, N)
    # k+1 because the nearest neighbor is the point itself; drop it.
    idx = dist.topk(k + 1, dim=-1, largest=False).indices       # (B, N, k+1)
    return idx[..., 1:]


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features.

    x:   (B, N, C)
    idx: (B, N, k)
    out: (B, N, k, C)
    """
    b, n, c = x.shape
    k = idx.shape[-1]
    batch_idx = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, n, k)
    return x[batch_idx, idx]


# --------------------------------------------------------------------------- #
# PointNet
# --------------------------------------------------------------------------- #
class PointNet(nn.Module):
    """Vanilla PointNet regressor (no T-Net; absolute coords are meaningful
    for binding pockets, so we keep them rather than learn a canonical pose)."""

    def __init__(
        self,
        in_features: int = 0,
        hidden: tuple[int, ...] = (64, 64, 128, 1024),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = _mlp([3 + in_features, *hidden])
        feat_dim = hidden[-1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, coords: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        x = coords if features is None else torch.cat([coords, features], dim=-1)
        x = x.transpose(1, 2)                  # (B, C, N) for Conv1d
        x = self.encoder(x)                    # (B, feat_dim, N)
        x = torch.max(x, dim=2).values         # global max-pool -> (B, feat_dim)
        return self.head(x).squeeze(-1)        # (B,)


# --------------------------------------------------------------------------- #
# Point Transformer
# --------------------------------------------------------------------------- #
class PointTransformerLayer(nn.Module):
    """Vector self-attention within local kNN neighborhoods.

    Follows Zhao et al. (2021): attention weights are produced by an MLP over
    (q_i - k_j + position_encoding), and aggregated values are also offset by a
    learned positional encoding.
    """

    def __init__(self, dim: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        # Position encoding from relative xyz.
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        # Attention weight MLP (gamma).
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), coords: (B, N, 3), idx: (B, N, k)
        q = self.to_q(x)                                   # (B, N, C)
        k = _gather_neighbors(self.to_k(x), idx)           # (B, N, k, C)
        v = _gather_neighbors(self.to_v(x), idx)           # (B, N, k, C)

        rel_pos = coords.unsqueeze(2) - _gather_neighbors(coords, idx)  # (B, N, k, 3)
        pos_enc = self.pos_mlp(rel_pos)                    # (B, N, k, C)

        attn = self.attn_mlp(q.unsqueeze(2) - k + pos_enc)  # (B, N, k, C)
        attn = F.softmax(attn, dim=2)                       # over k neighbors

        out = (attn * (v + pos_enc)).sum(dim=2)             # (B, N, C)
        return out


class PointTransformerBlock(nn.Module):
    """Residual block: linear -> attention -> linear, with a skip connection."""

    def __init__(self, dim: int, k: int = 16) -> None:
        super().__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.attn = PointTransformerLayer(dim, k=k)
        self.linear_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear_in(x))
        x = self.attn(x, coords, idx)
        x = self.linear_out(x)
        return self.norm(x + residual)


class PointTransformer(nn.Module):
    """Stacked Point Transformer blocks + global pooling for set regression.

    A single fixed kNN graph is built from the input coordinates and reused
    across blocks (no learned downsampling). This keeps the model simple and
    is adequate for pocket-sized point clouds; swap in transition-down layers
    if you need hierarchy on larger inputs.
    """

    def __init__(
        self,
        in_features: int = 0,
        dim: int = 128,
        depth: int = 4,
        k: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.k = k
        self.embed = nn.Linear(3 + in_features, dim)
        self.blocks = nn.ModuleList(
            [PointTransformerBlock(dim, k=k) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, coords: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        n = coords.shape[1]
        k = min(self.k, n - 1)                  # guard against tiny point clouds
        idx = _knn(coords, k)
        x = coords if features is None else torch.cat([coords, features], dim=-1)
        x = self.embed(x)                       # (B, N, dim)
        for block in self.blocks:
            x = block(x, coords, idx)
        x = torch.max(x, dim=1).values          # global max-pool -> (B, dim)
        return self.head(x).squeeze(-1)         # (B,)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "pointnet": PointNet,
    "point_transformer": PointTransformer,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct a model by name.

    Args:
        name: one of ``pointnet`` or ``point_transformer``.
        **kwargs: forwarded to the model constructor (e.g. ``in_features``).
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)