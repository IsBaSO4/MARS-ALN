"""Cross-Attention based shared/specific feature decomposition.

Core innovation: uses cross-attention between Breast and ALN features
to extract shared representations. Region-specific features are obtained
via gated soft decomposition (not hard subtraction), providing structural
decomposition without destroying useful signals.

v4 Changes:
  - ResidualProjection: all projection heads use skip connections to
    preserve information through the 256→128 bottleneck.
  - Gated soft decomposition: replaces hard subtraction (original - shared)
    with a learned sigmoid gate that adaptively controls how much of the
    original signal to retain. When shared estimation is poor (noisy ALN),
    the gate stays near 0, preserving original features.

Decomposition produces 6 tokens:
  - shared_T0, shared_T1: cross-region shared features at each timepoint
  - delta_B, delta_A: longitudinal treatment response features
  - residual_B, residual_A: region-specific features (gated)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DecompositionConfig:
    feature_dim: int = 256
    n_heads: int = 2
    dropout: float = 0.1
    token_dim: int = 128
    n_sub_tokens: int = 4  # split each feature into sub-tokens for cross-attention


class ResidualProjection(nn.Module):
    """Projection with skip connection to prevent information loss.

    out = MLP(x) + Linear(x)

    The skip path ensures raw information passes through even if
    the MLP path saturates or learns a poor transformation.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.skip(x)


class CrossRegionAttention(nn.Module):
    """Bidirectional cross-attention between Breast and ALN features.

    Features are reshaped from (B, D) to (B, n_sub_tokens, D//n_sub_tokens)
    so cross-attention operates on meaningful sub-feature sequences, avoiding
    the degenerate single-token case where attention weight is always 1.0.
    """

    def __init__(
        self, d_model: int, n_heads: int, n_sub_tokens: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_sub_tokens = n_sub_tokens
        self.sub_dim = d_model // n_sub_tokens
        assert d_model % n_sub_tokens == 0, (
            f"feature_dim ({d_model}) must be divisible by n_sub_tokens ({n_sub_tokens})"
        )

        self.attn_b2a = nn.MultiheadAttention(
            self.sub_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_a2b = nn.MultiheadAttention(
            self.sub_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_b = nn.LayerNorm(self.sub_dim)
        self.norm_a = nn.LayerNorm(self.sub_dim)

    def forward(
        self,
        h_breast: torch.Tensor,
        h_aln: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bidirectional cross-attention shared features.

        Parameters
        ----------
        h_breast : (B, D) — breast region features
        h_aln    : (B, D) — ALN region features

        Returns
        -------
        shared     : (B, D) — symmetrized shared representation
        shared_b   : (B, D) — breast-side shared (what breast found in ALN)
        shared_a   : (B, D) — ALN-side shared (what ALN found in breast)
        attn_b2a   : (B, n_heads, n_sub, n_sub) — attention weights
        """
        B = h_breast.shape[0]

        # Reshape to multi-token sequences: (B, n_sub, sub_dim)
        q_b = h_breast.view(B, self.n_sub_tokens, self.sub_dim)
        q_a = h_aln.view(B, self.n_sub_tokens, self.sub_dim)

        # Breast queries ALN: which breast sub-features are shared with ALN?
        shared_b, w_b2a = self.attn_b2a(q_b, q_a, q_a, need_weights=True,
                                         average_attn_weights=False)
        shared_b = self.norm_b(shared_b)  # (B, n_sub, sub_dim)

        # ALN queries Breast: which ALN sub-features are shared with breast?
        shared_a, w_a2b = self.attn_a2b(q_a, q_b, q_b, need_weights=True,
                                         average_attn_weights=False)
        shared_a = self.norm_a(shared_a)  # (B, n_sub, sub_dim)

        # Flatten back to (B, D)
        shared_b_flat = shared_b.reshape(B, -1)
        shared_a_flat = shared_a.reshape(B, -1)

        # Symmetrize: average both directions
        shared = (shared_b_flat + shared_a_flat) / 2

        return shared, shared_b_flat, shared_a_flat, w_b2a


class FeatureDecomposer(nn.Module):
    """Decompose multi-region multi-timepoint features into shared,
    longitudinal, and region-specific components.

    Architecture:
        1. Cross-attention at T0 and T1 → shared_T0, shared_T1
        2. Difference vectors → delta_B, delta_A (longitudinal)
        3. Gated soft decomposition → r_B, r_A (region-specific)
    """

    def __init__(self, cfg: DecompositionConfig) -> None:
        super().__init__()
        self._cfg = cfg

        # Cross-attention module (shared weights for T0 and T1)
        self.cross_attn = CrossRegionAttention(
            cfg.feature_dim, cfg.n_heads,
            n_sub_tokens=cfg.n_sub_tokens, dropout=cfg.dropout,
        )

        # Residual projections (with skip connections)
        self.shared_proj = ResidualProjection(cfg.feature_dim, cfg.token_dim)
        self.delta_proj = ResidualProjection(cfg.feature_dim, cfg.token_dim)
        self.residual_proj_B = ResidualProjection(cfg.feature_dim, cfg.token_dim)
        self.residual_proj_A = ResidualProjection(cfg.feature_dim, cfg.token_dim)

        # Gated soft decomposition for region-specific features
        # Input: concat(original, shared) → sigmoid gate
        # When gate → 0: preserve original (shared estimation is poor)
        # When gate → 1: remove shared (good decomposition)
        self.gate_B = nn.Sequential(
            nn.Linear(cfg.feature_dim * 2, cfg.feature_dim),
            nn.Sigmoid(),
        )
        self.gate_A = nn.Sequential(
            nn.Linear(cfg.feature_dim * 2, cfg.feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_B0: torch.Tensor,
        h_B1: torch.Tensor,
        h_A0: torch.Tensor,
        h_A1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decompose features into shared, longitudinal, and specific tokens.

        Parameters
        ----------
        h_B0 : (B, d) — Breast at T0
        h_B1 : (B, d) — Breast at T1
        h_A0 : (B, d) — ALN at T0
        h_A1 : (B, d) — ALN at T1

        Returns
        -------
        dict with keys:
            tokens  : (B, 6, token_dim) — all decomposed tokens
            shared_T0, shared_T1 : (B, token_dim)
            delta_B, delta_A     : (B, token_dim)
            residual_B, residual_A : (B, token_dim)
            delta_B_raw, delta_A_raw : (B, d) — raw difference vectors for L_delta
            gate_B, gate_A : (B, d) — gating values for interpretability
            attn_weights_T0, attn_weights_T1 : attention maps
        """
        # 1. Cross-region shared features
        shared_T0, shared_B_T0, shared_A_T0, attn_T0 = self.cross_attn(h_B0, h_A0)
        shared_T1, shared_B_T1, shared_A_T1, attn_T1 = self.cross_attn(h_B1, h_A1)

        shared_T0_tok = self.shared_proj(shared_T0)
        shared_T1_tok = self.shared_proj(shared_T1)

        # 2. Longitudinal difference (treatment response)
        delta_B_raw = h_B1 - h_B0
        delta_A_raw = h_A1 - h_A0
        delta_B_tok = self.delta_proj(delta_B_raw)
        delta_A_tok = self.delta_proj(delta_A_raw)

        # 3. Region-specific via gated soft decomposition
        # Use mean of T0 and T1 for stable region representation
        mean_B = (h_B0 + h_B1) / 2
        mean_A = (h_A0 + h_A1) / 2
        mean_shared_B = (shared_B_T0 + shared_B_T1) / 2
        mean_shared_A = (shared_A_T0 + shared_A_T1) / 2

        # Gated decomposition: gate controls how much shared to remove
        # gate → 0: keep original (shared estimation poor, e.g. noisy ALN)
        # gate → 1: remove shared (good decomposition)
        g_B = self.gate_B(torch.cat([mean_B, mean_shared_B], dim=-1))
        specific_B = mean_B * (1 - g_B)  # soft extraction of region-specific
        residual_B = self.residual_proj_B(specific_B)

        g_A = self.gate_A(torch.cat([mean_A, mean_shared_A], dim=-1))
        specific_A = mean_A * (1 - g_A)
        residual_A = self.residual_proj_A(specific_A)

        # 4. Stack all tokens: (B, 6, token_dim)
        tokens = torch.stack(
            [shared_T0_tok, shared_T1_tok, delta_B_tok, delta_A_tok,
             residual_B, residual_A],
            dim=1,
        )

        return {
            "tokens": tokens,
            "shared_T0": shared_T0_tok,
            "shared_T1": shared_T1_tok,
            "delta_B": delta_B_tok,
            "delta_A": delta_A_tok,
            "residual_B": residual_B,
            "residual_A": residual_A,
            "delta_B_raw": delta_B_raw,
            "delta_A_raw": delta_A_raw,
            "gate_B": g_B,
            "gate_A": g_A,
            "attn_weights_T0": attn_T0,
            "attn_weights_T1": attn_T1,
        }
