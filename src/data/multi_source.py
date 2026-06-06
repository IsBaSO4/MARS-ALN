"""Multi-source data loading and joint cross-validation utilities.

Supports two cohorts (e.g. ``internal_A`` and ``internal_B``) with:
  * one ``ALNpCRDataset`` per source, each tagged with a ``source`` name
    that propagates to every batch via the ``dataset_source`` field;
  * a globally indexed patient table with composite stratification keys
    (``label`` × ``dataset_source``) so 5-fold splits keep both label
    balance and source balance per fold;
  * per-fold ``ConcatDataset`` factories preserving train-vs-val transforms.

Backward compatibility: if the YAML config has no ``data.sources`` entry,
the legacy ``data.data_dir`` + ``data.clinical_xlsx`` keys are interpreted
as a single source (``internal_A`` by default), so old configs and saved
outputs keep working.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, Dataset

from src.data.dataset import ALNpCRDataset

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("breast_T0.npy", "breast_T1.npy", "aln_T0.npy", "aln_T1.npy")
DEFAULT_SOURCE_NAME = "internal_A"


@dataclass(frozen=True)
class SourceSpec:
    """Path + label for one cohort."""
    name: str
    data_dir: Path
    clinical_xlsx: Path
    sheet_name: str = "已完成"
    header: int = 1


@dataclass
class SourceData:
    """Resolved patient list + clinical data for a single source."""
    spec: SourceSpec
    patient_ids: list[str]
    labels: np.ndarray
    clinical_df: pd.DataFrame


def parse_sources(data_cfg: dict, project_root: Path | None = None) -> list[SourceSpec]:
    """Parse ``data.sources`` (list) or fall back to ``data.data_dir`` (single).

    Resolved paths are absolute. If the legacy single-path keys are used,
    the single source is named ``internal_A``.
    """
    project_root = project_root or Path.cwd()
    sources: list[SourceSpec] = []

    raw = data_cfg.get("sources")
    if raw:
        for entry in raw:
            data_dir = Path(entry["data_dir"])
            if not data_dir.is_absolute():
                data_dir = project_root / data_dir
            clinical = Path(entry["clinical_xlsx"])
            if not clinical.is_absolute():
                clinical = project_root / clinical
            sources.append(SourceSpec(
                name=entry["name"],
                data_dir=data_dir,
                clinical_xlsx=clinical,
                sheet_name=entry.get("sheet_name", "已完成"),
                header=entry.get("header", 1),
            ))
    else:
        # Legacy single-source mode
        data_dir = Path(data_cfg["data_dir"])
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        clinical = Path(data_cfg["clinical_xlsx"])
        if not clinical.is_absolute():
            clinical = project_root / clinical
        sources.append(SourceSpec(
            name=data_cfg.get("source_name", DEFAULT_SOURCE_NAME),
            data_dir=data_dir,
            clinical_xlsx=clinical,
        ))

    if not sources:
        raise ValueError("No data sources resolved from config.")

    seen_names: set[str] = set()
    for s in sources:
        if s.name in seen_names:
            raise ValueError(f"Duplicate source name in config: {s.name}")
        seen_names.add(s.name)
    return sources


def _list_complete_patients(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and all((p / fname).exists() for fname in REQUIRED_FILES)
    )


def load_sources(
    specs: list[SourceSpec],
    volume_size: tuple[int, int, int],
) -> list[SourceData]:
    """For each source, find patients with complete imaging + parse labels."""
    out: list[SourceData] = []
    for spec in specs:
        patient_ids = _list_complete_patients(spec.data_dir)
        if not patient_ids:
            raise RuntimeError(f"Source '{spec.name}' has no complete patients in {spec.data_dir}")
        clinical_df = pd.read_excel(
            spec.clinical_xlsx, sheet_name=spec.sheet_name, header=spec.header,
        )
        full_ds = ALNpCRDataset(
            str(spec.data_dir), clinical_df, patient_ids,
            volume_size=volume_size, source=spec.name,
        )
        labels = np.asarray(full_ds.labels, dtype=np.float32)
        logger.info(
            "Source %s: %d patients (pCR=%d, non-pCR=%d) — %s",
            spec.name, len(patient_ids),
            int(labels.sum()), int(len(labels) - labels.sum()), spec.data_dir,
        )
        out.append(SourceData(
            spec=spec, patient_ids=patient_ids, labels=labels, clinical_df=clinical_df,
        ))
    return out


# ------------------------------------------------------------------
# Joint patient index (across sources)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class PatientEntry:
    """A globally indexed patient: (source, position-in-source) + bookkeeping."""
    source_idx: int       # index into the SourceData list
    local_idx: int        # index into source.patient_ids
    source_name: str
    patient_id: str
    label: int


def build_patient_index(sources_data: list[SourceData]) -> list[PatientEntry]:
    entries: list[PatientEntry] = []
    for s_idx, sd in enumerate(sources_data):
        for l_idx, (pid, lbl) in enumerate(zip(sd.patient_ids, sd.labels)):
            entries.append(PatientEntry(
                source_idx=s_idx, local_idx=l_idx,
                source_name=sd.spec.name, patient_id=pid, label=int(lbl),
            ))
    return entries


def make_stratify_keys(entries: list[PatientEntry]) -> np.ndarray:
    """Composite key ``f"{source}__{label}"`` → enables joint stratification."""
    return np.array([f"{e.source_name}__{e.label}" for e in entries])


def joint_stratified_kfold(
    entries: list[PatientEntry],
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified folds over the composite (source, label) key.

    Each fold is balanced both for class label and for source identity,
    so every test fold contains samples from every source whenever the
    source is large enough to allow it.
    """
    keys = make_stratify_keys(entries)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(entries)), keys))


# ------------------------------------------------------------------
# Per-fold dataset construction
# ------------------------------------------------------------------

def _entries_grouped_by_source(
    entries: list[PatientEntry], indices: np.ndarray,
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for i in indices:
        e = entries[int(i)]
        grouped[e.source_idx].append(e.patient_id)
    return grouped


def _build_concat(
    sources_data: list[SourceData],
    grouped: dict[int, list[str]],
    volume_size: tuple[int, int, int],
    transform: object | None,
) -> Dataset:
    parts: list[Dataset] = []
    for s_idx, pids in grouped.items():
        sd = sources_data[s_idx]
        parts.append(ALNpCRDataset(
            str(sd.spec.data_dir),
            sd.clinical_df,
            pids,
            transform=transform,
            volume_size=volume_size,
            source=sd.spec.name,
        ))
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def build_fold_datasets(
    sources_data: list[SourceData],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    entries: list[PatientEntry],
    volume_size: tuple[int, int, int],
    train_transform: object | None,
) -> tuple[Dataset, Dataset]:
    """Build train/val ``ConcatDataset`` for one fold.

    Train datasets receive the augmentation transform; val datasets do not
    (mirrors the existing single-source behavior).
    """
    train_grouped = _entries_grouped_by_source(entries, train_idx)
    val_grouped = _entries_grouped_by_source(entries, val_idx)

    train_ds = _build_concat(sources_data, train_grouped, volume_size, train_transform)
    val_ds = _build_concat(sources_data, val_grouped, volume_size, None)
    return train_ds, val_ds


__all__ = [
    "DEFAULT_SOURCE_NAME",
    "REQUIRED_FILES",
    "PatientEntry",
    "SourceData",
    "SourceSpec",
    "build_fold_datasets",
    "build_patient_index",
    "joint_stratified_kfold",
    "load_sources",
    "make_stratify_keys",
    "parse_sources",
]
