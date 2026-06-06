#!/usr/bin/env python3
"""Train MARS (3D ResNet-50) with joint multi-cohort 5-fold cross-validation.

Edit the cohort paths in the config, then:
    python scripts/train.py --config configs/train.yaml --output_dir outputs/mars_run --gpu 0

Per-fold checkpoints are written to:
    <output_dir>/MARS/fold{0..N-1}/best_phase2.pt
(use this <output_dir> as --checkpoints_dir for scripts/cross_cohort_eval.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_source import (
    build_fold_datasets,
    build_patient_index,
    joint_stratified_kfold,
    load_sources,
    parse_sources,
)
from src.data.transforms import PairedTransform3D
from src.models.mars import build_mars_config
from src.training.losses import LossConfig
from src.training.trainer import MARSTrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("train")


def main() -> None:
    p = argparse.ArgumentParser(description="Train MARS (ResNet-50), joint multi-cohort 5-fold CV")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--output_dir", default=None, help="overrides training.output_dir in the config")
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    seed = args.seed if args.seed is not None else cfg["data"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mars_cfg = build_mars_config(cfg)
    loss_cfg = LossConfig(
        label_smoothing=cfg["loss"]["label_smoothing"],
        lambda_delta=cfg["loss"]["lambda_delta"],
        delta_margin=cfg["loss"]["delta_margin"],
    )
    volume_size = tuple(cfg["data"]["volume_size"])

    # ---- Data: one or more cohorts, joint stratified by (label x cohort) ----
    specs = parse_sources(cfg["data"], project_root=PROJECT_ROOT)
    sources = load_sources(specs, volume_size)
    entries = build_patient_index(sources)
    n_pcr = sum(e.label for e in entries)
    logger.info("Cohorts: %s | patients=%d (pCR=%d, non-pCR=%d)",
                [s.spec.name for s in sources], len(entries), n_pcr, len(entries) - n_pcr)

    n_folds = cfg["data"]["n_folds"]
    splits = joint_stratified_kfold(entries, n_folds=n_folds, seed=seed)

    aug = cfg["augmentation"]
    transform = PairedTransform3D(
        rotation_range=aug["rotation_range"],
        scale_range=tuple(aug["scale_range"]),
        intensity_shift=aug["intensity_shift"],
        noise_std=aug["noise_std"],
        flip_prob=aug["flip_prob"],
    )

    tr = cfg["training"]
    out_dir = Path(args.output_dir or tr["output_dir"])
    pin = device.type == "cuda"
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info("\n%s\n  Fold %d/%d  train=%d  val=%d\n%s",
                    "=" * 60, fold + 1, n_folds, len(train_idx), len(val_idx), "=" * 60)
        train_ds, val_ds = build_fold_datasets(
            sources, train_idx, val_idx, entries, volume_size=volume_size, train_transform=transform,
        )
        train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)

        trainer_cfg = TrainerConfig(
            epochs_phase1=tr["epochs_phase1"], epochs_phase2=tr["epochs_phase2"],
            lr_phase1=tr["lr_phase1"], lr_phase2=tr["lr_phase2"],
            weight_decay=tr["weight_decay"], warmup_epochs=tr["warmup_epochs"],
            patience=tr["patience"], batch_size=tr["batch_size"],
            output_dir=str(out_dir / "MARS" / f"fold{fold}"),
        )
        trainer = MARSTrainer(mars_cfg, loss_cfg, trainer_cfg, device)
        model = trainer.train(train_loader, val_loader)
        auc = trainer._validate(model, val_loader)
        fold_aucs.append(auc)
        logger.info("Fold %d val AUC: %.4f", fold + 1, auc)

    logger.info("\n%s\nCross-validation AUC: %.4f +/- %.4f\n%s",
                "=" * 60, float(np.mean(fold_aucs)), float(np.std(fold_aucs)), "=" * 60)
    for i, a in enumerate(fold_aucs):
        logger.info("  Fold %d: %.4f", i + 1, a)
    logger.info("Checkpoints saved under %s/MARS/fold*/best_phase2.pt", out_dir)


if __name__ == "__main__":
    main()
