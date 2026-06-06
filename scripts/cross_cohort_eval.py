#!/usr/bin/env python3
"""External validation: score a target cohort with the per-fold MARS checkpoints.

Loads every fold checkpoint trained by ``scripts/train.py`` and runs it over the
ENTIRE external cohort (no split — the cohort was never seen in training), then
reports per-fold metrics, mean +/- std AUC, and the 5-fold ensemble.

Usage:
    python scripts/cross_cohort_eval.py \
        --source_config configs/external.yaml \
        --target_config configs/external.yaml \
        --checkpoints_dir outputs/mars_run \
        --checkpoint_name best_phase2.pt \
        --output_dir outputs/external_validation --gpu 0

--source_config rebuilds the architecture; --target_config defines the external
cohort. The provided configs/external.yaml serves as both. --checkpoints_dir must
contain MARS/fold*/<checkpoint_name>.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ALNpCRDataset
from src.data.multi_source import load_sources, parse_sources
from src.evaluation.metrics import compute_classification_metrics
from src.models.mars import MARS, build_mars_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("cross_cohort_eval")


def build_target_loader(target_cfg: dict, batch_size: int, num_workers: int, device: torch.device):
    """One ALNpCRDataset over the entire single-source target cohort."""
    specs = parse_sources(target_cfg["data"], project_root=PROJECT_ROOT)
    if len(specs) != 1:
        raise ValueError(f"target_config must define exactly ONE source; got {len(specs)}")
    src = load_sources(specs, tuple(target_cfg["data"]["volume_size"]))[0]
    ds = ALNpCRDataset(str(src.spec.data_dir), src.clinical_df, src.patient_ids,
                       volume_size=tuple(target_cfg["data"]["volume_size"]), source=src.spec.name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    return loader, src.spec.name, len(ds.patient_ids)


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.train(False)   # inference mode
    probs, labels, pids = [], [], []
    for b in loader:
        out = model(b["breast_T0"].to(device), b["breast_T1"].to(device),
                    b["aln_T0"].to(device), b["aln_T1"].to(device), b["clinical"].to(device))
        probs.append(torch.sigmoid(out["pcr_logit"]).squeeze(-1).cpu())
        labels.append(b["label"])
        pids.extend(list(b.get("patient_id", [])))
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), pids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_config", required=True, help="config defining the architecture")
    p.add_argument("--target_config", required=True, help="config defining the external cohort (one source)")
    p.add_argument("--checkpoints_dir", required=True, help="dir containing MARS/fold*/<checkpoint_name>")
    p.add_argument("--checkpoint_name", default="best_phase2.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    src_cfg = yaml.safe_load(open(args.source_config))
    tgt_cfg = yaml.safe_load(open(args.target_config))
    mars_cfg = build_mars_config(src_cfg)
    batch_size = args.batch_size or tgt_cfg["training"]["batch_size"]

    ckpt_dir = Path(args.checkpoints_dir)
    fold_ckpts = sorted(ckpt_dir.glob(f"MARS/fold*/{args.checkpoint_name}"))
    if not fold_ckpts:
        raise FileNotFoundError(
            f"No checkpoints matching MARS/fold*/{args.checkpoint_name} under {ckpt_dir}")
    logger.info("Found %d fold checkpoints", len(fold_ckpts))

    loader, target_name, n_target = build_target_loader(tgt_cfg, batch_size, args.num_workers, device)
    logger.info("Target cohort: %s (n=%d)", target_name, n_target)

    all_probs: list[np.ndarray] = []
    labels = pids = None
    per_fold: list[dict] = []
    for ckpt in fold_ckpts:
        model = MARS(mars_cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        probs, lbl, pid = collect_predictions(model, loader, device)
        if labels is None:
            labels, pids = lbl, pid
        m = compute_classification_metrics(labels, probs) if len(set(labels)) >= 2 else {}
        per_fold.append({
            "checkpoint": str(ckpt.relative_to(ckpt_dir)),
            "auc_roc": float(m.get("auc_roc", 0.5)),
            "sensitivity": float(m.get("sensitivity", 0.0)),
            "specificity": float(m.get("specificity", 0.0)),
            "accuracy": float(m.get("accuracy", 0.0)),
            "f1": float(m.get("f1", 0.0)),
            "brier": float(m.get("brier_score", 0.0)),
        })
        all_probs.append(probs)
        logger.info("  %s  AUC=%.4f sens=%.4f spec=%.4f", ckpt.parent.name,
                    per_fold[-1]["auc_roc"], per_fold[-1]["sensitivity"], per_fold[-1]["specificity"])

    fold_aucs = np.array([f["auc_roc"] for f in per_fold])
    ensemble_probs = np.mean(np.stack(all_probs), axis=0)
    ensemble = compute_classification_metrics(labels, ensemble_probs) if len(set(labels)) >= 2 else {}

    summary = {
        "target_cohort": target_name,
        "n_patients": int(n_target),
        "n_fold_models": len(fold_ckpts),
        "per_fold": per_fold,
        "fold_auc_mean": float(fold_aucs.mean()),
        "fold_auc_std": float(fold_aucs.std()),
        "ensemble": {k: float(v) for k, v in ensemble.items() if not isinstance(v, np.ndarray)},
    }
    predictions = {
        "target_cohort": target_name,
        "patient_ids": list(pids),
        "labels": labels.tolist(),
        "fold_probs": [pr.tolist() for pr in all_probs],
        "ensemble_probs": ensemble_probs.tolist(),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_cohort_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "cross_cohort_predictions.json").write_text(json.dumps(predictions, indent=2))

    logger.info("\n%s\nEXTERNAL VALIDATION  (cohort = %s, n = %d)\n%s",
                "=" * 60, target_name, n_target, "=" * 60)
    logger.info("  fold AUCs           : %s", [round(a, 4) for a in fold_aucs])
    logger.info("  fold AUC mean +/- std: %.4f +/- %.4f", fold_aucs.mean(), fold_aucs.std())
    if ensemble:
        logger.info("  ensemble AUC        : %.4f", ensemble["auc_roc"])
        logger.info("  ensemble sens/spec  : %.4f / %.4f", ensemble["sensitivity"], ensemble["specificity"])
    logger.info("  saved -> %s", out)


if __name__ == "__main__":
    main()
