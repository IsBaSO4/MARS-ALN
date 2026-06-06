"""Self-Attention fusion module for combining decomposed feature tokens.

Takes the 6 tokens from the decomposition module and learns their
relative importance through self-attention. Uses attention-weighted pooling
(instead of naive mean pooling) to adaptively weight each component.

v2 Changes:
  - Replaced mean pooling with learned attention pooling via a [CLS]-like
    query token that attends to all decomposed tokens.
  - Token importance weights are directly interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionConfig:
    token_dim: int = 128
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1


class AttentionPool(nn.Module):
    """Learned attention pooling over a set of tokens.

    Uses a learnable query vector to attend to all tokens, producing
    a single aggregated output. The attention weights are interpretable
    as component importance scores.
    """

    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool tokens via learned attention.

        Parameters
        ----------
        tokens : (B, N, D) — N tokens of dimension D

        Returns
        -------
        pooled : (B, D) — attention-weighted aggregate
        weights : (B, N) — per-token attention weights (interpretable)
        """
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)

        pooled, attn_w = self.attn(
            q, tokens, tokens,
            need_weights=True, average_attn_weights=True,
        )  # pooled: (B, 1, D), attn_w: (B, 1, N)

        pooled = self.norm(pooled.squeeze(1))  # (B, D)
        weights = attn_w.squeeze(1)  # (B, N)

        return pooled, weights


class TokenFusion(nn.Module):
    """Self-attention fusion of decomposed feature tokens.

    Uses a standard Transformer encoder to learn interactions between
    shared, longitudinal, and region-specific tokens, followed by
    attention-weighted pooling for adaptive aggregation.
    """

    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.token_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers,
        )
        self.attn_pool = AttentionPool(cfg.token_dim, n_heads=cfg.n_heads)
        self.norm = nn.LayerNorm(cfg.token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Fuse decomposed tokens via self-attention + attention pooling.

        Parameters
        ----------
        tokens : (B, N, token_dim) — N decomposed tokens

        Returns
        -------
        (B, token_dim) — attention-pooled fused representation
        """
        out = self.transformer(tokens)
        out = self.norm(out)
        pooled, _ = self.attn_pool(out)
        return pooled

    def forward_with_attention(
        self, tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Forward pass that also returns attention weights for interpretability.

        Returns
        -------
        fused : (B, token_dim)
        attn_weights : list of (B, n_heads, N, N) per transformer layer
        pool_weights : (B, N) — per-token importance from attention pooling
        """
        attn_weights = []
        x = tokens
        for layer in self.transformer.layers:
            # Manually call self-attention to capture weights
            x_norm = layer.norm1(x)
            attn_out, w = layer.self_attn(
                x_norm, x_norm, x_norm, need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)
            # Feed-forward
            x = x + layer._ff_block(layer.norm2(x))
            attn_weights.append(w)

        x = self.norm(x)
        fused, pool_weights = self.attn_pool(x)
        return fused, attn_weights, pool_weights
