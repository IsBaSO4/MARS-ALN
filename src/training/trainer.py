"""Two-stage trainer for MARS model.

Phase 1 (Encoder Warmup): Train shared encoder + simple concat classifier.
Phase 2 (Full Training): Train full MARS model with decomposition + fusion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.models.mars import MARS, MARSConfig, MARSWarmup
from src.training.losses import LossConfig, MARSLoss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    epochs_phase1: int = 10
    epochs_phase2: int = 50
    lr_phase1: float = 1e-3
    lr_phase2: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 15
    batch_size: int = 16
    output_dir: str = "outputs"


class MARSTrainer:
    """Two-stage trainer for MARS."""

    def __init__(
        self,
        model_cfg: MARSConfig,
        loss_cfg: LossConfig,
        trainer_cfg: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model_cfg = model_cfg
        self.loss_fn = MARSLoss(loss_cfg)
        self.cfg = trainer_cfg
        self.device = device
        self.output_dir = Path(trainer_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> MARS:
        """Run two-stage training."""
        # Phase 1: Encoder warmup
        logger.info("=== Phase 1: Encoder Warmup ===")
        mars = MARS(self.model_cfg).to(self.device)
        warmup_model = MARSWarmup(mars.encoder, self.model_cfg).to(self.device)

        self._train_phase(
            warmup_model, train_loader, val_loader,
            epochs=self.cfg.epochs_phase1,
            lr=self.cfg.lr_phase1,
            phase_name="phase1",
        )

        # Phase 2: Full model (encoder weights carry over)
        logger.info("=== Phase 2: Full MARS Training ===")
        best_state = self._train_phase(
            mars, train_loader, val_loader,
            epochs=self.cfg.epochs_phase2,
            lr=self.cfg.lr_phase2,
            phase_name="phase2",
        )

        # Load best checkpoint
        mars.load_state_dict(best_state)
        return mars

    def _train_phase(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        phase_name: str,
    ) -> dict:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)

        # Cosine annealing with linear warmup
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, total_iters=self.cfg.warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=epochs - self.cfg.warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched],
            milestones=[self.cfg.warmup_epochs],
        )

        best_auc = 0.0
        best_state = model.state_dict()
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                out = model(
                    batch["breast_T0"], batch["breast_T1"],
                    batch["aln_T0"], batch["aln_T1"],
                    batch["clinical"],
                )
                losses = self.loss_fn(out, batch["label"])

                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += losses["total"].item()

            scheduler.step()

            # Validate
            val_auc = self._validate(model, val_loader)
            avg_train_loss = train_loss / max(len(train_loader), 1)

            logger.info(
                f"[{phase_name}] Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                torch.save(
                    best_state,
                    self.output_dir / f"best_{phase_name}.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"[{phase_name}] Best Val AUC: {best_auc:.4f}")
        return best_state

    @torch.no_grad()
    def _validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Compute validation AUC-ROC."""
        from sklearn.metrics import roc_auc_score

        model.eval()
        all_probs = []
        all_labels = []

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                batch["breast_T0"], batch["breast_T1"],
                batch["aln_T0"], batch["aln_T1"],
                batch["clinical"],
            )
            probs = torch.sigmoid(out["pcr_logit"]).squeeze(-1)
            all_probs.append(probs.cpu())
            all_labels.append(batch["label"].cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if len(set(all_labels)) < 2:
            return 0.5

        return roc_auc_score(all_labels, all_probs)
