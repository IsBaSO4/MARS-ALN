"""Dataset INPUT INTERFACE for MARS external validation.

This is the single, documented entry point between *preprocessing output* and
*model input*. Preprocessing (``prepare_external_data.py``) writes per-patient
``.npy`` files; this module turns a directory of them + an aligned clinical xlsx
into the exact ``torch`` batches the MARS model consumes — using the SAME
``ALNpCRDataset`` the model was trained and internally validated with, so the
external interface is identical to the internal one.

==========================  THE INPUT CONTRACT  ==========================
For each patient, preprocessing produces a folder ``<data_dir>/<patient_id>/``:

    breast_T0.npy   (4, 32, 64, 64) float32   pre-NAT  breast-tumour ROI
    breast_T1.npy   (4, 32, 64, 64) float32   post-NAT breast-tumour ROI
    aln_T0.npy      (4, 32, 64, 64) float32   pre-NAT  axillary-LN ROI
    aln_T1.npy      (4, 32, 64, 64) float32   post-NAT axillary-LN ROI

    4 channels  = [P0_norm, early_enh, peak_enh, delayed_enh]  (DCE kinetics)
    32x64x64    = (D, H, W) after ROI crop + resize (volume_size)

Clinical vector (length 5), read from the aligned xlsx, z-scored per cohort
at load time: [HER2, ER, PR, HR, Ki67].

Label (ALN pCR) = 1 - post-LN   (post-LN column in the aligned xlsx).

Each ``__getitem__`` returns a dict of tensors:
    breast_T0, breast_T1, aln_T0, aln_T1 : (4, 32, 64, 64) float32
    clinical                              : (5,) float32
    label                                 : scalar float32
    patient_id, dataset_source            : str
=========================================================================

Produce the .npy:   scripts/preprocess.py + scripts/align_clinical.py
Consume them here:  build_external_dataloader(data_dir, clinical_xlsx)
Score them:         scripts/cross_cohort_eval.py
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import ALNpCRDataset
from src.data.multi_source import REQUIRED_FILES, SourceSpec, load_sources

# Public, machine-readable description of the input contract.
INPUT_VOLUME_SHAPE = (4, 32, 64, 64)          # (channels, D, H, W) per ROI
ROI_FILES = REQUIRED_FILES                    # ("breast_T0.npy", ..., "aln_T1.npy")
DCE_CHANNELS = ("P0_norm", "early_enh", "peak_enh", "delayed_enh")
CLINICAL_FEATURES = ("HER2", "ER", "PR", "HR", "Ki67")
LABEL_DEFINITION = "ALN pCR = 1 - post-LN"


def build_external_dataset(
    data_dir: str | Path,
    clinical_xlsx: str | Path,
    sheet_name: str = "已完成",
    header: int = 1,
    volume_size: tuple[int, int, int] = (32, 64, 64),
    name: str = "external",
) -> ALNpCRDataset:
    """Build the evaluation ``ALNpCRDataset`` for one external cohort.

    Lists patients that have all four ``REQUIRED_FILES`` under ``data_dir``,
    reads + parses the aligned clinical xlsx (same path the trained model used),
    and returns a ready-to-iterate dataset. No augmentation (eval transform=None).

    Parameters mirror ``configs/external.yaml`` (sheet_name / header / volume_size).
    """
    spec = SourceSpec(
        name=name,
        data_dir=Path(data_dir),
        clinical_xlsx=Path(clinical_xlsx),
        sheet_name=sheet_name,
        header=header,
    )
    sd = load_sources([spec], tuple(volume_size))[0]
    return ALNpCRDataset(
        str(sd.spec.data_dir),
        sd.clinical_df,
        sd.patient_ids,
        volume_size=tuple(volume_size),
        source=name,
    )


def build_external_dataloader(
    data_dir: str | Path,
    clinical_xlsx: str | Path,
    batch_size: int = 12,
    num_workers: int = 4,
    sheet_name: str = "已完成",
    header: int = 1,
    volume_size: tuple[int, int, int] = (32, 64, 64),
    name: str = "external",
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap :func:`build_external_dataset` in a non-shuffling ``DataLoader``.

    This is exactly the loader ``cross_cohort_eval.py`` scores each fold
    checkpoint over.
    """
    ds = build_external_dataset(
        data_dir, clinical_xlsx, sheet_name=sheet_name, header=header,
        volume_size=volume_size, name=name,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


if __name__ == "__main__":
    # Quick self-check / interface demo:
    #   python -m src.data.dataset_interface <data_dir> <clinical_xlsx>
    import sys
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: python -m src.data.dataset_interface <data_dir> <clinical_xlsx>")
        raise SystemExit(0)
    ds = build_external_dataset(sys.argv[1], sys.argv[2])
    print(f"patients: {len(ds)}  | pCR(label=1): {int(ds.labels.sum())} "
          f"non-pCR: {int(len(ds.labels) - ds.labels.sum())}")
    sample = ds[0]
    for k in ("breast_T0", "breast_T1", "aln_T0", "aln_T1", "clinical", "label"):
        v = sample[k]
        print(f"  {k:10s}: shape={tuple(v.shape)} dtype={v.dtype}")
