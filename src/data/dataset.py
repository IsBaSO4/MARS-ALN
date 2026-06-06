"""Multi-region multi-timepoint DCE-MRI dataset for ALN pCR prediction.

Each patient provides 4 ROI volumes (each 4-channel from DCE kinetic maps):
  - Breast tumour at T0 (pre-NAT) and T1 (post-NAT)
  - Axillary lymph node region at T0 and T1

Preprocessed by ``preprocessing.py`` into .npy files of shape (4, D, H, W).
Clinical features are loaded from the xlsx file and encoded as a vector.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Column name for patient ID — pandas renames the second duplicate "ID" to "ID.1"
# Column [3] "ID.1" has all 192 patient IDs; Column [1] "ID" is mostly NaN.
ID_COLUMN = "ID.1"

# Clinical feature columns from xlsx (biopsy/穿刺 data)
CLINICAL_COLUMNS = {
    "HER2": ("HER2(1:positive;0:negative)", "binary"),
    "ER": ("ER(数值)", "continuous"),
    "PR": ("PR(数值)", "continuous"),
    "HR": ("HR(1:positive;0:negative)", "binary"),
    "Ki67": ("Ki67(数值)", "continuous"),
}


def _safe_float(val: Any, col_name: str = "") -> float:
    """Convert a potentially dirty clinical value to float.

    Handles:
      - NaN, None → 0.0
      - "/" , "l", "//" , empty string → 0.0
      - Range "15-20" → midpoint 17.5
      - "＜1" / "<1" → 0.5 (half of threshold)
      - Comma-separated cycle "2,4,6" → max value
      - Chinese text / unparseable → 0.0
      - "1（10点）" → leading digit 1.0
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0

    s = str(val).strip()

    # Obvious non-numeric placeholders
    if s in ("", "/", "//", "l"):
        return 0.0

    # Try direct conversion first (fast path)
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Range values like "15-20" or "5-10" → midpoint
    if "-" in s and s[0].isdigit():
        parts = s.split("-")
        try:
            nums = [float(p) for p in parts if p]
            return sum(nums) / len(nums)
        except ValueError:
            pass

    # "＜1" or "<1" → half the threshold
    for prefix in ("＜", "<", "≤", "<="):
        if s.startswith(prefix):
            try:
                return float(s[len(prefix):]) * 0.5
            except ValueError:
                pass

    # Extract all numbers and return the max (cycle column: "2,4" → 4, "3，5" → 5)
    all_nums = re.findall(r"\d+\.?\d*", s)
    if all_nums:
        return max(float(x) for x in all_nums)

    # Completely unparseable (Chinese text etc.)
    return 0.0


class ALNpCRDataset(Dataset):
    """Dataset for 4-ROI ALN pCR prediction.

    Directory structure (output of preprocessing.py):
        data_dir/
            <patient_id>/
                breast_T0.npy   # (4, D, H, W) — 4-channel DCE kinetic maps
                breast_T1.npy
                aln_T0.npy
                aln_T1.npy

    Each .npy file contains a 4-channel 3D volume:
        Ch0 = P0 normalised (baseline morphology)
        Ch1 = early enhancement ratio
        Ch2 = peak enhancement ratio
        Ch3 = delayed enhancement ratio
    """

    def __init__(
        self,
        data_dir: str,
        clinical_df: pd.DataFrame,
        patient_ids: list[str],
        transform: Any | None = None,
        volume_size: tuple[int, int, int] = (32, 64, 64),
        source: str = "internal_A",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.transform = transform
        self.volume_size = volume_size
        self.source = source

        # Parse clinical features and labels
        self.clinical_features, self.labels = self._parse_clinical(
            clinical_df, patient_ids,
        )

    def _parse_clinical(
        self,
        df: pd.DataFrame,
        patient_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract clinical feature vectors and ALN pCR labels."""
        features = []
        labels = []

        # Build lookup: str(ID) → row index for fast matching
        # IDs can be int, float, or str (e.g. "80171546L")
        id_to_idx: dict[str, int] = {}
        for i, val in enumerate(df[ID_COLUMN]):
            key = str(val).strip()
            # Also strip ".0" for float IDs so "80159446.0" matches "80159446"
            if key.endswith(".0"):
                key = key[:-2]
            id_to_idx[key] = i

        for pid in patient_ids:
            pid_str = str(pid).strip()
            if pid_str not in id_to_idx:
                logger.warning(f"Patient {pid} not found in clinical data — skipping")
                features.append([0.0] * len(CLINICAL_COLUMNS))
                labels.append(0.0)
                continue
            row = df.iloc[id_to_idx[pid_str]]

            feat = []
            for name, (col, dtype) in CLINICAL_COLUMNS.items():
                val = row.get(col, 0)
                feat.append(_safe_float(val, name))

            features.append(feat)

            # Label: post-LN (0=negative=pCR achieved, 1=positive=residual)
            post_ln = row.get("post-LN（1:positive；0:negative）", 0)
            # Invert: pCR=1 means post-LN=0
            label = 1.0 - float(post_ln) if not pd.isna(post_ln) else 0.0
            labels.append(label)

        features_arr = np.array(features, dtype=np.float32)
        # Normalize continuous features (per-dataset, not per-fold to avoid leakage later)
        self._cont_stats: dict[int, tuple[float, float]] = {}
        for i, (name, (_, dtype)) in enumerate(CLINICAL_COLUMNS.items()):
            if dtype == "continuous":
                col_vals = features_arr[:, i]
                mean = col_vals.mean()
                std = col_vals.std() + 1e-8
                features_arr[:, i] = (col_vals - mean) / std
                self._cont_stats[i] = (mean, std)

        return features_arr, np.array(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        patient_dir = self.data_dir / pid

        # Load 4 ROI volumes — each (4, D, H, W) — 4 DCE channels
        breast_T0 = self._load_volume(patient_dir / "breast_T0.npy")
        breast_T1 = self._load_volume(patient_dir / "breast_T1.npy")
        aln_T0 = self._load_volume(patient_dir / "aln_T0.npy")
        aln_T1 = self._load_volume(patient_dir / "aln_T1.npy")

        # Apply transforms (same spatial transform for same-timepoint pairs)
        if self.transform is not None:
            breast_T0, aln_T0 = self.transform(breast_T0, aln_T0)
            breast_T1, aln_T1 = self.transform(breast_T1, aln_T1)

        return {
            "breast_T0": torch.from_numpy(breast_T0).float(),   # (4, D, H, W)
            "breast_T1": torch.from_numpy(breast_T1).float(),
            "aln_T0": torch.from_numpy(aln_T0).float(),
            "aln_T1": torch.from_numpy(aln_T1).float(),
            "clinical": torch.from_numpy(self.clinical_features[idx]),
            "label": torch.tensor(self.labels[idx]),
            "patient_id": pid,
            "dataset_source": self.source,
        }

    # def _load_volume(self, path: Path) -> np.ndarray:
    #     """Load a preprocessed 4-channel 3D volume.

    #     Expected shape: (4, D, H, W).
    #     If shape is (D, H, W) for backward compat, unsqueezes channel dim.
    #     """
    #     vol = np.load(path).astype(np.float32)
    #     if vol.ndim == 3:
    #         # Legacy single-channel: replicate to 4 channels
    #         vol = np.stack([vol] * 4, axis=0)
    #     if vol.shape[1:] != self.volume_size:
    #         from scipy.ndimage import zoom
    #         factors = [1.0] + [t / s for t, s in zip(self.volume_size, vol.shape[1:])]
    #         vol = zoom(vol, factors, order=1)
    #     return vol
    def _load_volume(self, path: Path) -> np.ndarray:
        vol = np.load(path).astype(np.float32)

        if vol.ndim == 3:
            vol = np.stack([vol] * 4, axis=0)

        if vol.shape[1:] != self.volume_size:
            from scipy.ndimage import zoom
            factors = [1.0] + [t / s for t, s in zip(self.volume_size, vol.shape[1:])]
            vol = zoom(vol, factors, order=1)

        # Per-channel z-score within volume:
        # 去除 cohort 间 intensity gain / bias，
        # 同时保留空间结构和通道间信息。
        for c in range(vol.shape[0]):
            vol[c] = (vol[c] - vol[c].mean()) / (vol[c].std() + 1e-6)

        return vol
