"""MARS: Multi-region Attentive Representation with Shared-Specific decomposition.

Shared 3D ResNet-50 encoder -> cross-attention shared/specific decomposition
(6 tokens) -> self-attention fusion + attention pooling -> MLP classifier with
clinical features -> ALN pCR prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.classifier import ClassifierConfig, PCRClassifier
from src.models.decomposition import DecompositionConfig, FeatureDecomposer
from src.models.encoder import EncoderConfig, SharedEncoder
from src.models.fusion import FusionConfig, TokenFusion

# Token names (for interpretability / bookkeeping)
TOKEN_NAMES = [
    "shared_T0", "shared_T1",     # cross-region shared features
    "delta_B", "delta_A",          # longitudinal treatment response
    "residual_B", "residual_A",    # region-specific features
]


@dataclass(frozen=True)
class MARSConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


def build_mars_config(cfg: dict) -> MARSConfig:
    """Build a ``MARSConfig`` from a parsed YAML config dict."""
    enc, dec = cfg["encoder"], cfg["decomposition"]
    fus, cls = cfg["fusion"], cfg["classifier"]
    return MARSConfig(
        encoder=EncoderConfig(
            in_channels=enc.get("in_channels", 4),
            out_dim=enc["out_dim"],
            dropout=enc["dropout"],
            pretrained_path=enc.get("pretrained_path"),
        ),
        decomposition=DecompositionConfig(
            feature_dim=enc["out_dim"],
            n_heads=dec["n_heads"],
            dropout=dec["dropout"],
            token_dim=dec["token_dim"],
            n_sub_tokens=dec.get("n_sub_tokens", 4),
        ),
        fusion=FusionConfig(
            token_dim=dec["token_dim"],
            n_heads=fus["n_heads"],
            n_layers=fus["n_layers"],
            dropout=fus["dropout"],
        ),
        classifier=ClassifierConfig(
            token_dim=dec["token_dim"],
            clinical_dim=cls["clinical_dim"],
            hidden_dim=cls["hidden_dim"],
            dropout=cls["dropout"],
        ),
    )


class MARS(nn.Module):
    """MARS with a shared 3D ResNet-50 encoder."""

    def __init__(self, cfg: MARSConfig) -> None:
        super().__init__()
        self.encoder = SharedEncoder(cfg.encoder)
        self.decomposer = FeatureDecomposer(cfg.decomposition)
        self.fusion = TokenFusion(cfg.fusion)
        self.classifier = PCRClassifier(cfg.classifier)

    def forward(
        self,
        breast_T0: torch.Tensor,
        breast_T1: torch.Tensor,
        aln_T0: torch.Tensor,
        aln_T1: torch.Tensor,
        clinical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_B0 = self.encoder(breast_T0)
        h_B1 = self.encoder(breast_T1)
        h_A0 = self.encoder(aln_T0)
        h_A1 = self.encoder(aln_T1)

        decomp = self.decomposer(h_B0, h_B1, h_A0, h_A1)
        fused = self.fusion(decomp["tokens"])
        pcr_logit = self.classifier(fused, clinical)

        return {
            "pcr_logit": pcr_logit,
            "decomposition": decomp,
            "fused": fused,
            "h_B0": h_B0, "h_B1": h_B1, "h_A0": h_A0, "h_A1": h_A1,
        }


class MARSWarmup(nn.Module):
    """Phase-1 encoder warmup head: concat the 4 encoded ROIs + clinical -> logit."""

    def __init__(self, encoder: SharedEncoder, cfg: MARSConfig) -> None:
        super().__init__()
        self.encoder = encoder
        in_dim = cfg.encoder.out_dim * 4 + cfg.classifier.clinical_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        breast_T0: torch.Tensor,
        breast_T1: torch.Tensor,
        aln_T0: torch.Tensor,
        aln_T1: torch.Tensor,
        clinical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_B0 = self.encoder(breast_T0)
        h_B1 = self.encoder(breast_T1)
        h_A0 = self.encoder(aln_T0)
        h_A1 = self.encoder(aln_T1)
        concat = torch.cat([h_B0, h_B1, h_A0, h_A1, clinical], dim=-1)
        return {"pcr_logit": self.head(concat)}
