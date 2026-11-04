# antigen-ml

A machine-learning pipeline for predicting protein antigenicity from sequence and structural features, with optional ESM-2 protein language model embeddings.

**Author:** Amit Sah — SignorLab, North Dakota State University  
**Language:** Python 3.10+  
**License:** MIT

---

## Overview

antigen-ml trains classifiers to discriminate antigenic proteins (recognized by the immune system) from non-antigenic proteins. It supports two complementary feature tracks:

| Track | Features | Dimensionality | Notes |
|---|---|---|---|
| Classical | AAC, DPC, CTD, physicochemical, structural | 503-dim | No GPU needed |
| ESM-2 | Pretrained protein language model embeddings | 480–1280-dim | Requires PyTorch + internet |

Both tracks can be combined. The pipeline fetches labeled training data from [IEDB](https://www.iedb.org/) and [UniProt](https://www.uniprot.org/), performs redundancy-aware cross-validation (GroupKFold with MinHash clustering), and outputs a calibrated XGBoost or Random Forest classifier.

### Results on the full IEDB + UniProt dataset (n = 3,763)

| Model | Features | CV AUC | Test AUC | Test MCC | Best threshold |
|---|---|---|---|---|---|
| XGBoost | Classical (503-dim) | 0.858 | 0.858 | 0.571 | 0.40 |
| XGBoost | ESM-2 35M + Classical | ≥0.95 | — | — | — |

---

## Repository layout

```
antigen-ml/
├── train.py                  # Main training script
├── predict.py                # Inference on new FASTA sequences
├── requirements.txt          # Python dependencies
├── LICENSE
├── README.md
│
├── src/
│   ├── download_data.py      # Fetch labeled data from IEDB + UniProt
│   ├── data.py               # FASTA/CSV I/O, cleaning, train/test split
│   ├── features.py           # AAC, DPC, CTD, physicochemical features
│   ├── struct_features.py    # Structural propensity features (28-dim)
│   ├── models_classical.py   # Logistic regression, SVM, RF, XGBoost
│   ├── models_deep.py        # ESM-2 embedding extractor + ProteinCNN
│   └── evaluate.py           # AUC, MCC, PR, threshold search, plots
│
├── data/
│   ├── raw/                  # Downloaded IEDB CSVs and UniProt FASTAs (not tracked)
│   └── processed/            # dataset.csv built by download_data.py (not tracked)
│
└── results/                  # Model weights, plots, summary (best_model.pkl not tracked)
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/asah15/antigen-ml.git
cd antigen-ml
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. Install core dependencies**

```bash
pip install -r requirements.txt
```

**4. (Optional) Install ESM-2 dependencies**

Required only if you use `--features esm` during training.

```bash
# CPU-only PyTorch (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.40"

# GPU (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40"
```

---

## Quickstart

### Step 1 — Download labeled training data

The pipeline supports three data modes. **Mode A** (IEDB bulk export) gives the most sequences and is recommended.

#### Mode A: IEDB bulk export (recommended, ~1,900 antigens)

1. Go to [https://www.iedb.org/database_export_v3.php](https://www.iedb.org/database_export_v3.php)
2. Download **B Cell Assays** → `bcell_full_v3.zip`
3. Extract so that `bcell_full_v3_single_file/bcell_full_v3.csv` is accessible

```bash
python src/download_data.py \
    --iedb_csv path/to/bcell_full_v3_single_file/bcell_full_v3.csv \
    --out data/processed/dataset.csv
```

#### Mode B: UniProt KW-0045 keyword query (~600 antigens, internet only)

```bash
python src/download_data.py \
    --out data/processed/dataset.csv
```

#### Mode C: Synthetic demo (no data needed, for testing the pipeline)

```bash
python src/download_data.py \
    --demo \
    --out data/processed/dataset.csv
```

---

### Step 2 — Train a classifier

#### Classical features (fast, no GPU required)

```bash
python train.py \
    --csv data/processed/dataset.csv \
    --features aac,dpc,ctd,phys,struct \
    --models rf,xgb \
    --cv 5
```

#### ESM-2 embeddings only

```bash
python train.py \
    --csv data/processed/dataset.csv \
    --features esm \
    --esm_model facebook/esm2_t12_35M_UR50D \
    --models xgb
```

Embeddings are automatically cached to `results/esm_cache.npy` after the first run. Subsequent training runs load from cache and skip the slow embedding step.

#### Combined: ESM-2 + classical features (highest accuracy)

```bash
python train.py \
    --csv data/processed/dataset.csv \
    --features aac,dpc,ctd,phys,struct,esm \
    --esm_model facebook/esm2_t12_35M_UR50D \
    --models xgb
```

#### All options

```
--csv PATH          Labeled CSV with 'sequence' and 'label' columns
--pos PATH          FASTA of antigen proteins (use with --neg)
--neg PATH          FASTA of non-antigen proteins (use with --pos)
--demo              Use 300 synthetic sequences per class (no files needed)

--features STR      Comma-separated: aac,dpc,ctd,phys,struct,esm
                    Default: aac,dpc,ctd,phys,struct
--esm_model STR     ESM-2 HuggingFace model ID (default: facebook/esm2_t12_35M_UR50D)
                    Options:
                      facebook/esm2_t6_8M_UR50D    320-dim  (fastest)
                      facebook/esm2_t12_35M_UR50D  480-dim  (default)
                      facebook/esm2_t30_150M_UR50D 640-dim
                      facebook/esm2_t33_650M_UR50D 1280-dim (best quality)
--esm_cache PATH    Path to save/load ESM embeddings (default: results/esm_cache.npy)
--models STR        Comma-separated: logreg,svm,rf,xgb  (default: rf,xgb)
--cv INT            Cross-validation folds (default: 5)
--out DIR           Output directory (default: results/)
--seed INT          Random seed (default: 42)
```

**Outputs** written to `results/`:

| File | Description |
|---|---|
| `best_model.pkl` | Trained classifier (best CV AUC) |
| `summary.csv` | Per-model CV and test metrics |
| `roc_pr.png` | ROC and Precision-Recall curves |
| `metric_bars.png` | AUC bar chart across models |
| `feature_importance.png` | Top-30 feature importances (RF/XGB) |
| `esm_cache.npy` | Cached ESM-2 embeddings (if applicable) |

---

### Step 3 — Predict antigenicity for new proteins

```bash
python predict.py \
    --fasta my_proteins.fasta \
    --model results/best_model.pkl \
    --threshold 0.40 \
    --out predictions.tsv
```

#### Threshold guidance

The default threshold is 0.50. For the XGBoost + classical model, threshold **0.40** maximizes MCC on the held-out test set. Use the value printed during training (`best-thr=X.XX`) for the model you saved.

#### Output columns

| Column | Description |
|---|---|
| `protein_id` | Sequence ID from FASTA header (first word) |
| `length` | Amino-acid length after cleaning |
| `antigenicity_score` | Model probability (0–1); higher = more likely antigen |
| `prediction` | `antigen` or `non-antigen` at the given threshold |
| `flag` | `low_confidence` if score is between 0.40 and 0.60 |

#### All options

```
--fasta PATH        Input FASTA file (required)
--model PATH        Trained model pickle (default: results/best_model.pkl)
--threshold FLOAT   Decision threshold (default: 0.5)
--out PATH          Output TSV path (default: stdout)
--min_len INT       Skip sequences shorter than N aa (default: 30)
--max_len INT       Truncate sequences longer than N aa (default: 6000)
```

---

## Feature sets

| Set | Flag | Dim | Description |
|---|---|---|---|
| Amino acid composition | `aac` | 20 | Fraction of each of the 20 standard amino acids |
| Dipeptide composition | `dpc` | 400 | Fraction of each of the 400 dipeptide pairs |
| CTD | `ctd` | 42 | Composition, transition, distribution descriptors (Dubchak 1995) |
| Physicochemical | `phys` | 13 | MW, charge, pI, GRAVY, aromaticity, instability index, etc. |
| Structural propensity | `struct` | 28 | Chou-Fasman, Eisenberg hydrophobicity, Vihinen flexibility, Janin surface exposure |
| ESM-2 embeddings | `esm` | 320–1280 | Mean-pooled last-hidden-state from ESM-2 (Meta AI) |

---

## Data sources

- **IEDB** (Immune Epitope Database): B-cell assay bulk export v3. Positive labels are proteins with experimentally confirmed B-cell epitopes. [https://www.iedb.org](https://www.iedb.org)
- **UniProt/Swiss-Prot**: Reviewed entries. Positives are annotated with keyword KW-0045 (Antigen) in viral and bacterial organisms. Negatives are human, yeast, and plant entries without that keyword. [https://www.uniprot.org](https://www.uniprot.org)

---

## Reproducing the benchmark

```bash
# 1. Download IEDB bulk export manually (see Step 1 above), then:
python src/download_data.py \
    --iedb_csv path/to/bcell_full_v3.csv \
    --out data/processed/dataset.csv

# 2. Train
python train.py \
    --csv data/processed/dataset.csv \
    --features aac,dpc,ctd,phys,struct \
    --models rf,xgb \
    --seed 42

# 3. Inspect results
cat results/summary.csv
```

---

## Citation

If you use this tool in published work, please cite:

```
Sah, A. (2026). antigen-ml: sequence- and structure-based antigenicity prediction
using classical and protein language model features. GitHub.
https://github.com/asah15/antigen-ml
```

---

## References

- Dubchak, I. et al. (1995). Prediction of protein folding class using global description of amino acid sequence. *PNAS* 92(19), 8700–8704.
- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379(6637), 1123–1130. (ESM-2)
- Vita, R. et al. (2019). The Immune Epitope Database (IEDB): 2018 update. *Nucleic Acids Research* 47(D1), D339–D343.
