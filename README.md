# MARS: Multi-region Attentive Representation with Shared–Specific decomposition

Code accompanying the manuscript under review. MARS predicts axillary lymph node
pathologic complete response (ALN pCR) to neoadjuvant therapy from multi-region,
multi-timepoint breast DCE-MRI.

MARS takes four region-of-interest (ROI) volumes per patient — the **breast
tumour** and the **axillary lymph node** region, each at **pre-** and
**post-**neoadjuvant therapy (NAT) — encodes them with a shared 3D **ResNet-50**,
decomposes the features into *cross-region shared*, *longitudinal
treatment-response*, and *region-specific* tokens via cross-attention, fuses them
with self-attention, and predicts ALN pCR together with biopsy clinical features.


Patient imaging/clinical data are not included; trained weights can be downloaded from https://drive.google.com/drive/folders/1_cZoTh8bS6MeqahbDicK8CcHGlu-5T6m?usp=sharing.

## Installation

```bash
conda env create -f environment.yml && conda activate mars-aln   # or: pip install -r requirements.txt
pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Data format

### Raw layout (input to preprocessing)
```
raw_dir/
├── pre/<patient_id>/   T0.nii.gz T1.nii.gz ... T{N-1}.nii.gz   # pre-NAT  DCE phases (N≥4)
├── post/<patient_id>/  T0.nii.gz T1.nii.gz ...                 # post-NAT DCE phases
└── mask/
    ├── pre/<patient_id>/   tumor.nii.gz  aln.nii.gz
    └── post/<patient_id>/  tumor.nii.gz  aln.nii.gz
```

### Preprocessed layout = the model's dataset input interface
```
data_dir/<patient_id>/
├── breast_T0.npy   (4, 32, 64, 64) float32   # pre-NAT  breast-tumour ROI
├── breast_T1.npy   (4, 32, 64, 64) float32   # post-NAT breast-tumour ROI
├── aln_T0.npy      (4, 32, 64, 64) float32   # pre-NAT  axillary-LN ROI
└── aln_T1.npy      (4, 32, 64, 64) float32   # post-NAT axillary-LN ROI
```
The 4 channels are the DCE pharmacokinetic maps `[P0_norm, early_enh, peak_enh,
delayed_enh]`. The contract is exposed in `src/data/dataset_interface.py`:
```bash
python -m src.data.dataset_interface <data_dir> <clinical_aligned.xlsx>
```

### Clinical schema (produced by `align_clinical.py`)
`ID.1`, `HER2(1:positive;0:negative)`, `ER(数值)`, `PR(数值)`,
`HR(1:positive;0:negative)`, `Ki67(数值)`, `post-LN（1:positive；0:negative）`.
Label = **ALN pCR = 1 − post-LN**; the 5 clinical features `[HER2, ER, PR, HR,
Ki67]` are z-scored per cohort at load time.

## Usage

```bash
# 1) Preprocess imaging + clinical
python scripts/preprocess.py     --raw_dir path/to/raw --output_dir path/to/ALN_preprocess
python scripts/align_clinical.py --input  path/to/clinical.xlsx --output path/to/clinical_aligned.xlsx

# 2) Train MARS (joint multi-cohort 5-fold CV) — edit cohort paths in configs/train.yaml
python scripts/train.py --config configs/train.yaml --output_dir outputs/mars_run --gpu 0

# 3) External validation — edit the cohort paths in configs/external.yaml
python scripts/cross_cohort_eval.py \
    --source_config configs/external.yaml --target_config configs/external.yaml \
    --checkpoints_dir outputs/mars_run --checkpoint_name best_phase2.pt \
    --output_dir outputs/external_validation --gpu 0
```
Use `--gpu -1` for CPU. `--checkpoints_dir` must contain `MARS/fold{0..4}/best_phase2.pt`.

## Output

```
outputs/mars_run/MARS/fold{0..4}/best_phase2.pt          # per-fold MARS checkpoints
outputs/external_validation/
├── cross_cohort_summary.json     # per-fold + mean±std AUC + ensemble metrics
└── cross_cohort_predictions.json # per-patient probabilities (per fold + ensemble)
```
