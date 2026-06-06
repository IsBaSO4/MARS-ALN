"""Loss functions for MARS ALN pCR prediction.

Total loss: L = L_pCR + lambda_delta * L_delta

- L_pCR: Binary cross-entropy with label smoothing
- L_delta: Treatment response consistency loss — encourages aligned
  longitudinal change vectors when ALN achieves pCR
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    label_smoothing: float = 0.05
    lambda_delta: float = 0.1
    delta_margin: float = 0.2


def pcr_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Binary cross-entropy with label smoothing for pCR prediction.

    Parameters
    ----------
    logits : (B, 1) — pre-sigmoid
    labels : (B,) — {0, 1}
    """
    targets = labels.float().unsqueeze(-1)
    targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets)


def delta_consistency_loss(
    delta_B: torch.Tensor,
    delta_A: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Treatment response consistency loss.

    Clinical intuition: when ALN achieves pCR, breast and ALN treatment
    responses should be similar (both effectively shrink). Residual
    disease means the two regions' responses diverge.

    L = -y * cos(delta_B, delta_A) + (1-y) * max(0, cos(delta_B, delta_A) - margin)

    Parameters
    ----------
    delta_B : (B, d) — breast longitudinal change vector (h_B1 - h_B0)
    delta_A : (B, d) — ALN longitudinal change vector (h_A1 - h_A0)
    labels  : (B,) — pCR labels {0, 1}
    margin  : float — margin for non-pCR cases
    """
    cos_sim = F.cosine_similarity(delta_B, delta_A, dim=-1)  # (B,)
    y = labels.float()

    # pCR: maximize cosine similarity (minimize -cos)
    loss_pcr = -y * cos_sim

    # non-pCR: penalize high similarity (push below margin)
    loss_non_pcr = (1.0 - y) * torch.clamp(cos_sim - margin, min=0.0)

    return (loss_pcr + loss_non_pcr).mean()


class MARSLoss(nn.Module):
    """Combined loss for MARS model."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self._cfg = cfg

    def forward(
        self,
        model_out: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Parameters
        ----------
        model_out : output dict from MARS.forward()
        labels    : (B,) — ALN pCR labels {0, 1}

        Returns
        -------
        dict with keys: total, pcr, delta
        """
        l_pcr = pcr_loss(
            model_out["pcr_logit"], labels, self._cfg.label_smoothing,
        )

        decomp = model_out.get("decomposition")
        if decomp is not None and "delta_B_raw" in decomp:
            l_delta = delta_consistency_loss(
                decomp["delta_B_raw"],
                decomp["delta_A_raw"],
                labels,
                self._cfg.delta_margin,
            )
        else:
            l_delta = torch.tensor(0.0, device=l_pcr.device)

        total = l_pcr + self._cfg.lambda_delta * l_delta

        return {
            "total": total,
            "pcr": l_pcr,
            "delta": l_delta,
        }
