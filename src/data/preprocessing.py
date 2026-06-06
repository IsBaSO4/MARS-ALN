"""DCE-MRI preprocessing pipeline for ALN pCR prediction.

Adaptive variant (v2.0): supports variable-length DCE series. Each patient
timepoint may contain N >= 4 phases named T0..T(N-1).nii.gz (or legacy
P0..P(N-1).nii.gz). The pipeline:

  1. Discovers phases (`discover_phases`), accepting either naming convention.
  2. Builds a TIC region mask (`_build_region_mask`): union of tumor & ALN
     masks if available, else foreground heuristic.
  3. Picks (early, peak, delayed) indices from the TIC (`select_pk_indices`):
       early   = T1                           (clinical convention)
       peak    = argmax over interior phases  (data-driven, restricted to T2..T(N-2))
       delayed = T(N-1)                       (clinical convention)
  4. Computes the 4-channel pharmacokinetic volume (`compute_enhancement_maps`,
     unchanged): [P0_norm, early_ratio, peak_ratio, delayed_ratio].
  5. Crops per-region ROIs (`extract_roi`, unchanged).

These pharmacokinetic maps are clinically interpretable: radiologists routinely
inspect wash-in, plateau and washout patterns on DCE-MRI.

Expected raw data layout
------------------------
raw_dir/
    pre/                       # Pre-NAT (T0)
        <patient_id>/
            T0.nii.gz          # baseline (or P0.nii.gz for legacy data)
            T1.nii.gz          # post-contrast phase 1
            ...
            T{N-1}.nii.gz      # last post-contrast phase (N varies per patient)
    post/                      # Post-NAT (T1)
        <patient_id>/
            T0.nii.gz, T1.nii.gz, ...
    mask/
        pre/<patient_id>/  tumor.nii.gz, aln.nii.gz
        post/<patient_id>/ tumor.nii.gz, aln.nii.gz

Output layout (UNCHANGED contract — saved as .npy for fast loading)
--------------------------------------------------------------------
output_dir/
    <patient_id>/
        breast_T0.npy          # (4, D, H, W)  4-channel
        breast_T1.npy
        aln_T0.npy
        aln_T1.npy
        phase_selection.json   # (optional sidecar; cfg.write_sidecar)

Note: "T0"/"T1" in the *output* filenames denote NAT timepoint (pre-/post-NAT),
not phase index. Input filenames "T0".."TN" denote phase index. This naming
collision is unavoidable given existing downstream conventions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, binary_dilation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing hyper-parameters."""

    # Target volume size (D, H, W) after ROI cropping + resize
    volume_size: tuple[int, int, int] = (32, 64, 64)
    # Margin (voxels) around mask bounding box for context
    crop_margin: int = 10
    # Small epsilon to avoid division by zero in enhancement ratios
    eps: float = 1e-6
    # Whether to apply N4 bias field correction (requires SimpleITK)
    apply_n4: bool = False
    # Mask dilation iterations (expand ROI slightly for context)
    mask_dilation: int = 2
    # Phase-file discovery: prefer "T*.nii.gz", fall back to legacy "P*.nii.gz"
    phase_pattern_primary: str = "T*.nii.gz"
    phase_pattern_legacy: str = "P*.nii.gz"
    # Minimum number of phases required to process a patient timepoint
    min_phases: int = 4
    # Whether to write the phase_selection.json sidecar
    write_sidecar: bool = True
    # Mask file names
    tumor_mask_name: str = "tumor.nii.gz"
    aln_mask_name: str = "aln.nii.gz"


# ======================================================================
# Core functions
# ======================================================================


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI file and return the image array as float32."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def _parse_phase_index(stem: str, prefix: str) -> int | None:
    """Extract trailing integer from a stem like 'T12' (after stripping prefix).

    Returns None if the remainder after the prefix is not a non-negative integer.
    """
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix):]
    if rest == "" or not rest.isdigit():
        return None
    return int(rest)


def discover_phases(
    phase_dir: str | Path,
    cfg: "PreprocessConfig",
) -> list[Path] | None:
    """Discover and sort DCE phase files in a single timepoint directory.

    Tries the primary glob pattern (e.g. ``T*.nii.gz``) first; falls back to the
    legacy pattern (``P*.nii.gz``) only if the primary yields nothing. Mixed
    naming inside one directory is treated as an error.

    Returns
    -------
    list[Path] | None
        Sorted phase file paths (ascending phase index), or ``None`` if any
        validation fails (logs a warning/error explaining why).

    Validation rules
    ----------------
    1. At least one of the two glob patterns must match; mixing both is invalid.
    2. Each filename's stem (with ``.nii.gz`` stripped) must be ``<prefix><int>``,
       where prefix is ``T`` or ``P``.
    3. The smallest phase index must be 0 (baseline present).
    4. The number of discovered phases must be >= ``cfg.min_phases``.
    """
    phase_dir = Path(phase_dir)

    primary = sorted(phase_dir.glob(cfg.phase_pattern_primary))
    legacy = sorted(phase_dir.glob(cfg.phase_pattern_legacy))

    if primary and legacy:
        logger.error(
            f"Phase dir {phase_dir} contains both primary "
            f"('{cfg.phase_pattern_primary}') and legacy "
            f"('{cfg.phase_pattern_legacy}') files; refusing to guess."
        )
        return None

    if primary:
        candidates = primary
        prefix = cfg.phase_pattern_primary[0]  # 'T'
    elif legacy:
        candidates = legacy
        prefix = cfg.phase_pattern_legacy[0]   # 'P'
    else:
        logger.warning(f"No phase files found in {phase_dir}")
        return None

    indexed: list[tuple[int, Path]] = []
    for p in candidates:
        stem = p.name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        idx = _parse_phase_index(stem, prefix)
        if idx is None:
            logger.error(
                f"Cannot parse phase index from '{p.name}' in {phase_dir} "
                f"(expected '<{prefix}><int>.nii.gz')"
            )
            return None
        indexed.append((idx, p))

    indexed.sort(key=lambda t: t[0])
    indices = [i for i, _ in indexed]

    if indices[0] != 0:
        logger.error(
            f"Phase dir {phase_dir} missing baseline (smallest index = "
            f"{indices[0]}, expected 0)"
        )
        return None

    if len(indices) < cfg.min_phases:
        logger.warning(
            f"Phase dir {phase_dir} has {len(indices)} phases "
            f"(< min_phases={cfg.min_phases}); skipping."
        )
        return None

    return [p for _, p in indexed]


def n4_bias_correction(volume: np.ndarray) -> np.ndarray:
    """Apply N4 bias field correction using SimpleITK.

    Falls back to returning the original volume if SimpleITK is unavailable.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.warning("SimpleITK not installed — skipping N4 correction")
        return volume

    sitk_img = sitk.GetImageFromArray(volume)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    # Create a mask (non-zero voxels)
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(sitk_img, mask)

    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def compute_enhancement_maps(
    phases: list[np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute 4-channel pharmacokinetic feature maps from DCE phases.

    Parameters
    ----------
    phases : list of 4 arrays, each (D, H, W)
        P0 (pre-contrast), P1, P2, P3 (post-contrast)
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray : (4, D, H, W)
        Channel 0: P0 (Z-score normalised)
        Channel 1: Early enhancement ratio (P1-P0)/(P0+eps)
        Channel 2: Peak enhancement ratio (P2-P0)/(P0+eps)
        Channel 3: Delayed enhancement ratio (P3-P0)/(P0+eps)
    """
    p0, p1, p2, p3 = phases

    # Z-score normalise P0 (within the volume, excluding background)
    fg_mask = p0 > p0.mean() * 0.1  # rough foreground
    if fg_mask.sum() > 0:
        mu = p0[fg_mask].mean()
        sigma = p0[fg_mask].std() + eps
        p0_norm = (p0 - mu) / sigma
    else:
        p0_norm = p0.copy()

    # Enhancement ratios relative to P0 (baseline)
    denominator = np.abs(p0) + eps
    early_enh = (p1 - p0) / denominator
    peak_enh = (p2 - p0) / denominator
    delayed_enh = (p3 - p0) / denominator

    # Clip extreme values (robustness)
    for arr in [early_enh, peak_enh, delayed_enh]:
        np.clip(arr, -5.0, 10.0, out=arr)

    return np.stack([p0_norm, early_enh, peak_enh, delayed_enh], axis=0)


def extract_roi(
    volume_4ch: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int, int],
    margin: int = 10,
    dilation: int = 2,
) -> np.ndarray:
    """Crop ROI using mask bounding box + margin, then resize.

    Parameters
    ----------
    volume_4ch : (4, D, H, W) — multi-channel volume
    mask : (D, H, W) — binary mask
    target_size : (D, H, W) — output spatial dimensions
    margin : int — voxels of context around the mask bbox
    dilation : int — expand mask slightly before computing bbox

    Returns
    -------
    np.ndarray : (4, *target_size)
    """
    # Binarise and dilate mask for bbox computation
    bin_mask = (mask > 0.5).astype(np.uint8)
    if dilation > 0:
        struct = np.ones((3, 3, 3), dtype=np.uint8)
        bin_mask = binary_dilation(bin_mask, structure=struct, iterations=dilation).astype(np.uint8)

    # Find bounding box
    coords = np.argwhere(bin_mask > 0)
    if coords.size == 0:
        logger.warning("Empty mask — returning zero-filled volume")
        return np.zeros((4, *target_size), dtype=np.float32)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # exclusive upper bound

    # Add margin (clamped to volume bounds)
    spatial_shape = volume_4ch.shape[1:]  # (D, H, W)
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(spatial_shape))

    # Crop
    d0, h0, w0 = mins
    d1, h1, w1 = maxs
    cropped = volume_4ch[:, d0:d1, h0:h1, w0:w1]

    # Apply mask within the cropped region (zero out non-ROI background)
    cropped_mask = bin_mask[d0:d1, h0:h1, w0:w1]
    cropped = cropped * cropped_mask[np.newaxis]

    # Resize to target size
    if cropped.shape[1:] != target_size:
        factors = [1.0] + [t / s for t, s in zip(target_size, cropped.shape[1:])]
        cropped = zoom(cropped, factors, order=1)

    return cropped.astype(np.float32)


def _build_region_mask(
    tumor_mask: np.ndarray | None,
    aln_mask: np.ndarray | None,
    p0: np.ndarray,
    cfg: "PreprocessConfig",
) -> tuple[np.ndarray, str]:
    """Build the boolean region mask used for time-intensity-curve averaging.

    Selection priority:
        - tumor & aln present  -> union of (tumor > 0.5) | (aln > 0.5)
        - only one present     -> that one > 0.5
        - neither present      -> foreground heuristic ``p0 > p0.mean() * 0.1``

    The result is dilated by ``cfg.mask_dilation`` purely for TIC averaging
    (so peri-tumoral voxels contribute). This is decoupled from ``extract_roi``,
    which keeps performing its own dilation for bbox computation on the
    per-region mask.

    Returns
    -------
    (mask: np.ndarray of bool with shape p0.shape, source_label: str)
    """
    if tumor_mask is not None and aln_mask is not None:
        mask = (tumor_mask > 0.5) | (aln_mask > 0.5)
        source = "union(tumor,aln)"
    elif tumor_mask is not None:
        mask = tumor_mask > 0.5
        source = "tumor"
    elif aln_mask is not None:
        mask = aln_mask > 0.5
        source = "aln"
    else:
        mask = p0 > p0.mean() * 0.1
        source = "foreground"

    if cfg.mask_dilation > 0:
        struct = np.ones((3, 3, 3), dtype=np.uint8)
        mask = binary_dilation(mask, structure=struct, iterations=cfg.mask_dilation)

    return mask.astype(bool), source


def select_pk_indices(
    phases: list[np.ndarray],
    region_mask: np.ndarray,
) -> dict:
    """Pick (baseline, early, peak, delayed) phase indices via the TIC.

    Conventions
    -----------
    - ``baseline_idx = 0``
    - ``early_idx = 1``  (first post-contrast; clinical convention)
    - ``delayed_idx = N - 1``  (last phase; clinical convention)
    - ``peak_idx = 2 + argmax(tic[2 : N-1])``  — restricted to **interior**
      phases so it cannot collide with ``early`` or ``delayed``.

    Parameters
    ----------
    phases : list of (D, H, W) ndarrays, length N >= 4
    region_mask : (D, H, W) bool ndarray. If empty (no foreground voxels),
        TIC falls back to whole-volume mean and the returned dict contains
        ``"region_fallback": True``.

    Returns
    -------
    dict with keys: baseline_idx, early_idx, peak_idx, delayed_idx, tic.
        tic[t] = mean(phases[t] - phases[0]) over the region (length N,
        tic[0] == 0.0 by construction). May also include
        ``region_fallback: True`` if the supplied mask was empty.
    """
    if len(phases) < 4:
        raise ValueError(
            f"select_pk_indices requires N>=4 phases (got {len(phases)})"
        )

    p0 = phases[0]
    n = len(phases)

    region_fallback = False
    if region_mask.dtype != bool:
        region_mask = region_mask.astype(bool)
    if not region_mask.any():
        logger.warning(
            "Empty region mask passed to select_pk_indices; "
            "falling back to whole-volume mean for TIC."
        )
        region_fallback = True

    tic = [0.0]
    for t in range(1, n):
        diff = phases[t] - p0
        if region_fallback:
            tic.append(float(diff.mean()))
        else:
            tic.append(float(diff[region_mask].mean()))

    interior = tic[2 : n - 1]  # phases 2..N-2 inclusive
    if not interior:
        raise RuntimeError("Empty interior — internal invariant violated")
    peak_idx = 2 + int(np.argmax(interior))

    if all(v <= 0.0 for v in tic[1:]):
        logger.warning(
            "TIC has no positive enhancement; peak selection may be unreliable."
        )

    out: dict = {
        "baseline_idx": 0,
        "early_idx": 1,
        "peak_idx": peak_idx,
        "delayed_idx": n - 1,
        "tic": tic,
    }
    if region_fallback:
        out["region_fallback"] = True
    return out


# ======================================================================
# Patient-level processing
# ======================================================================


def process_single_patient(
    patient_id: str,
    raw_dir: Path,
    output_dir: Path,
    cfg: PreprocessConfig,
) -> bool:
    """Process one patient: discover phases, pick PK indices, crop ROIs, save.

    Output (unchanged contract):
        <output_dir>/<patient_id>/breast_T0.npy   (4, *cfg.volume_size) float32
        <output_dir>/<patient_id>/breast_T1.npy
        <output_dir>/<patient_id>/aln_T0.npy
        <output_dir>/<patient_id>/aln_T1.npy
        <output_dir>/<patient_id>/phase_selection.json   (if cfg.write_sidecar)
    """
    out_dir = output_dir / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar: dict = {
        "patient_id": patient_id,
        "preprocessing_version": "2.0-adaptive",
    }

    for timepoint, tp_label in [("pre", "T0"), ("post", "T1")]:
        phase_dir = raw_dir / timepoint / patient_id
        mask_dir = raw_dir / "mask" / timepoint / patient_id

        if not phase_dir.exists():
            logger.warning(f"Phase dir missing: {phase_dir}")
            return False

        # ---- Discover & load phases ----
        phase_paths = discover_phases(phase_dir, cfg)
        if phase_paths is None:
            return False  # discover_phases already logged the reason

        phases = []
        for ppath in phase_paths:
            vol = load_nifti(ppath)
            if cfg.apply_n4:
                vol = n4_bias_correction(vol)
            phases.append(vol)

        # ---- Load masks (may be missing) ----
        tumor_mask_path = mask_dir / cfg.tumor_mask_name
        aln_mask_path = mask_dir / cfg.aln_mask_name
        tumor_mask = load_nifti(tumor_mask_path) if tumor_mask_path.exists() else None
        aln_mask = load_nifti(aln_mask_path) if aln_mask_path.exists() else None

        if tumor_mask is None:
            logger.warning(f"Tumor mask missing: {tumor_mask_path}")
        if aln_mask is None:
            logger.warning(f"ALN mask missing: {aln_mask_path}")

        # ---- Build region mask & pick PK indices ----
        region_mask, region_source = _build_region_mask(
            tumor_mask, aln_mask, phases[0], cfg,
        )
        idx_info = select_pk_indices(phases, region_mask)

        # ---- Compute 4-channel enhancement maps (UNCHANGED math) ----
        four_phases = [
            phases[idx_info["baseline_idx"]],
            phases[idx_info["early_idx"]],
            phases[idx_info["peak_idx"]],
            phases[idx_info["delayed_idx"]],
        ]
        volume_4ch = compute_enhancement_maps(four_phases, eps=cfg.eps)

        # ---- Crop ROIs (per-region masks unchanged) ----
        if tumor_mask is not None:
            breast_roi = extract_roi(
                volume_4ch, tumor_mask, cfg.volume_size,
                margin=cfg.crop_margin, dilation=cfg.mask_dilation,
            )
        else:
            breast_roi = np.zeros((4, *cfg.volume_size), dtype=np.float32)

        if aln_mask is not None:
            aln_roi = extract_roi(
                volume_4ch, aln_mask, cfg.volume_size,
                margin=cfg.crop_margin, dilation=cfg.mask_dilation,
            )
        else:
            aln_roi = np.zeros((4, *cfg.volume_size), dtype=np.float32)

        # ---- Save ROIs ----
        np.save(out_dir / f"breast_{tp_label}.npy", breast_roi)
        np.save(out_dir / f"aln_{tp_label}.npy", aln_roi)
        logger.info(
            f"[{patient_id}] {tp_label}: breast {breast_roi.shape}, aln {aln_roi.shape}"
            f" | n_phases={len(phases)} peak_idx={idx_info['peak_idx']}"
        )

        # ---- Stash sidecar payload for this timepoint ----
        sidecar[tp_label] = {
            "n_phases": len(phases),
            "input_naming": "T" if phase_paths[0].name.startswith("T") else "P",
            "phase_files": [p.name for p in phase_paths],
            "selected": {
                "baseline_idx": idx_info["baseline_idx"],
                "early_idx": idx_info["early_idx"],
                "peak_idx": idx_info["peak_idx"],
                "delayed_idx": idx_info["delayed_idx"],
            },
            "tic": idx_info["tic"],
            "region_mask_source": (
                "whole_volume_fallback" if idx_info.get("region_fallback")
                else region_source
            ),
        }

    # ---- After both timepoints: write sidecar ----
    if cfg.write_sidecar:
        sidecar["config"] = {
            "volume_size": list(cfg.volume_size),
            "crop_margin": cfg.crop_margin,
            "apply_n4": cfg.apply_n4,
            "mask_dilation": cfg.mask_dilation,
            "min_phases": cfg.min_phases,
        }
        try:
            sidecar_path = out_dir / "phase_selection.json"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(
                f"Could not write sidecar for {patient_id}: {e}; "
                f".npy outputs are unaffected."
            )

    return True


# ======================================================================
# Batch processing
# ======================================================================


def preprocess_all(
    raw_dir: str | Path,
    output_dir: str | Path,
    cfg: PreprocessConfig | None = None,
    patient_ids: list[str] | None = None,
) -> list[str]:
    """Process all patients (or a specified subset).

    Parameters
    ----------
    raw_dir : root containing pre/, post/, mask/
    output_dir : where to save processed volumes
    cfg : preprocessing config (uses defaults if None)
    patient_ids : if None, auto-discover from pre/ folder

    Returns
    -------
    list[str] : successfully processed patient IDs
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = PreprocessConfig()

    # Auto-discover patients from pre/ folder
    if patient_ids is None:
        pre_dir = raw_dir / "pre"
        if not pre_dir.exists():
            raise FileNotFoundError(f"Pre-NAT directory not found: {pre_dir}")
        patient_ids = sorted([
            d.name for d in pre_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        logger.info(f"Found {len(patient_ids)} patients in {pre_dir}")

    successful = []
    failed = []

    for pid in patient_ids:
        try:
            ok = process_single_patient(pid, raw_dir, output_dir, cfg)
            if ok:
                successful.append(pid)
            else:
                failed.append(pid)
        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")
            failed.append(pid)

    logger.info(
        f"Preprocessing complete: {len(successful)} success, {len(failed)} failed"
    )
    if failed:
        logger.warning(f"Failed patients: {failed}")

    return successful


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess DCE-MRI for ALN pCR")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Root of raw data (contains pre/, post/, mask/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed .npy files")
    parser.add_argument("--volume_size", type=int, nargs=3, default=[32, 64, 64],
                        help="Target volume size (D H W)")
    parser.add_argument("--crop_margin", type=int, default=10,
                        help="Margin around mask bounding box")
    parser.add_argument("--n4", action="store_true",
                        help="Apply N4 bias field correction")
    parser.add_argument("--tumor_mask", type=str, default="tumor.nii.gz",
                        help="Tumor mask filename")
    parser.add_argument("--aln_mask", type=str, default="aln.nii.gz",
                        help="ALN mask filename")
    args = parser.parse_args()

    cfg = PreprocessConfig(
        volume_size=tuple(args.volume_size),
        crop_margin=args.crop_margin,
        apply_n4=args.n4,
        tumor_mask_name=args.tumor_mask,
        aln_mask_name=args.aln_mask,
    )

    preprocess_all(args.raw_dir, args.output_dir, cfg)
