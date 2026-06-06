"""Classification head for ALN pCR prediction.

Combines fused imaging features with clinical variables through
a lightweight MLP classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ClassifierConfig:
    token_dim: int = 128
    clinical_dim: int = 5
    hidden_dim: int = 64
    dropout: float = 0.3


class PCRClassifier(nn.Module):
    """Binary classifier for ALN pCR prediction.

    Concatenates fused imaging features with clinical features
    and predicts pCR probability.
    """

    def __init__(self, cfg: ClassifierConfig) -> None:
        super().__init__()
        in_dim = cfg.token_dim + cfg.clinical_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ALN pCR logit.

        Parameters
        ----------
        fused_features : (B, token_dim) — from TokenFusion
        clinical : (B, clinical_dim) — clinical feature vector

        Returns
        -------
        (B, 1) — pre-sigmoid logit
        """
        x = torch.cat([fused_features, clinical], dim=-1)
        return self.net(x)
