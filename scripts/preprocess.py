#!/usr/bin/env python3
"""Batch preprocess DCE-MRI data for ALN pCR project.

Usage:
    python scripts/preprocess.py --raw_dir /path/to/raw --output_dir data/processed

The raw directory should contain:
    raw_dir/
        pre/<patient_id>/P0.nii.gz, P1.nii.gz, P2.nii.gz, P3.nii.gz
        post/<patient_id>/P0.nii.gz, P1.nii.gz, P2.nii.gz, P3.nii.gz
        mask/pre/<patient_id>/tumor.nii.gz, aln.nii.gz
        mask/post/<patient_id>/tumor.nii.gz, aln.nii.gz

Output:
    output_dir/<patient_id>/breast_T0.npy, breast_T1.npy, aln_T0.npy, aln_T1.npy
    Each .npy is (4, D, H, W) with 4 DCE pharmacokinetic channels.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import PreprocessConfig, preprocess_all


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Preprocess DCE-MRI for ALN pCR")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Root of raw data (contains pre/, post/, mask/)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed .npy files")
    parser.add_argument("--volume_size", type=int, nargs=3, default=[32, 64, 64],
                        help="Target volume size (D H W)")
    parser.add_argument("--crop_margin", type=int, default=10,
                        help="Margin around mask bounding box (voxels)")
    parser.add_argument("--n4", action="store_true",
                        help="Apply N4 bias field correction (requires SimpleITK)")
    parser.add_argument("--tumor_mask", type=str, default="tumor.nii.gz",
                        help="Tumor mask filename in mask folders")
    parser.add_argument("--aln_mask", type=str, default="aln.nii.gz",
                        help="ALN region mask filename in mask folders")
    parser.add_argument("--min_phases", type=int, default=4,
                        help="Skip patient timepoints with fewer than this many phases (default: 4)")
    parser.add_argument("--no_sidecar", action="store_true",
                        help="Do not write phase_selection.json sidecars")
    args = parser.parse_args()

    cfg = PreprocessConfig(
        volume_size=tuple(args.volume_size),
        crop_margin=args.crop_margin,
        apply_n4=args.n4,
        tumor_mask_name=args.tumor_mask,
        aln_mask_name=args.aln_mask,
        min_phases=args.min_phases,
        write_sidecar=not args.no_sidecar,
    )

    successful = preprocess_all(args.raw_dir, args.output_dir, cfg)
    print(f"\nDone! Successfully processed {len(successful)} patients.")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
