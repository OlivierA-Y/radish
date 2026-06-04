# Zero-Shot IDH Mutation Prediction in Brain Gliomas

A Python implementation of the pipeline described in:

> **"Computational Imaging Meets LLMs: Zero-Shot IDH Mutation Prediction in Brain Gliomas"**  
> Syed Muqeem Mahmood and Hassan Mohy-ud-Din (2025)  
> [arXiv:2511.03376](https://arxiv.org/abs/2511.03376)

The pipeline extracts interpretable imaging features from multi-parametric MRI, serializes them into structured text, and queries a large language model to predict IDH mutation status — without any task-specific fine-tuning. Supports **OpenAI** (GPT-4o, GPT-5) and **OpenRouter** (Claude, Gemini, Llama, Mistral, and 300+ other models via a single API key).

---

## Overview

IDH (isocitrate dehydrogenase) mutation status is a key molecular marker that defines glioma subtype and prognosis. Definitive diagnosis requires tissue biopsy, but imaging-derived features correlate strongly with IDH status. This project operationalizes that correlation through a zero-shot LLM reasoning approach.

**Pipeline stages:**

```
Multi-parametric MRI (T1, T1-CE, T2, FLAIR)
        │
        ▼
  Preprocessing          intensity normalization, skull stripping
        │
        ▼
  Segmentation           BrainSegFounder (SwinUNETR) or threshold fallback
        │
        ▼
  Registration           ANTs SyN → MNI152 space (or affine fallback)
        │
        ▼
  Atlas Mapping          Harvard-Oxford, Juelich, Hammers
        │
        ▼
  Feature Extraction     per-region signal stats, enhancement ratios
        │
        ▼
  Serialization          JSON + radiology-style text narrative
        │
        ▼
  LLM Query              OpenAI or OpenRouter (zero-shot, no fine-tuning)
        │
        ▼
  IDH Prediction         IDH-mutant | IDH-wildtype + confidence + reasoning
```

---

## Requirements

- Python 3.10+
- `uv` (recommended) or any virtual environment manager

**Core dependencies:**

| Package | Purpose |
|---|---|
| `nibabel` | NIfTI MRI file I/O |
| `nilearn` | Atlas fetching (Harvard-Oxford, Juelich) |
| `antspyx` | ANTs nonlinear registration *(optional)* |
| `monai` + `torch` | BrainSegFounder segmentation *(optional)* |
| `openai` | GPT-4o / GPT-5 API, and OpenRouter (same SDK) |
| `scikit-learn` | AUC-ROC and metric utilities |
| `scipy` / `numpy` | Image processing |

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd radish

# 2. Create virtual environment
uv venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=sk-...     # Linux/macOS
$env:OPENAI_API_KEY="sk-..."     # Windows PowerShell
# or add OPENAI_API_KEY=sk-... to a .env file in the project root
```

For **segmentation only** (no deep learning):
```bash
uv pip install nibabel numpy scipy scikit-learn nilearn pandas tqdm
```

For **full pipeline** with BrainSegFounder segmentation:
```bash
uv pip install torch monai    # add --index-url for CUDA builds as needed
```

---

## Quick Start

### Synthetic data (no MRI files or API key needed)

```bash
# Feature extraction only — inspect the JSON and narrative outputs
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --no_llm

# Dry run with mock LLM responses (no API calls)
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --dry_run
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...

python main.py --mode synthetic --model gpt-4o
python main.py --mode synthetic --model gpt-5-chat-latest
```

### OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-...

# Provider is auto-detected from the slash in the model name
python main.py --mode synthetic --model anthropic/claude-opus-4
python main.py --mode synthetic --model google/gemini-pro-1.5
python main.py --mode synthetic --model meta-llama/llama-3.3-70b-instruct

# Or use short aliases
python main.py --mode synthetic --model claude-opus-4 --provider openrouter
python main.py --mode synthetic --model gemini-flash  --provider openrouter
python main.py --mode synthetic --model llama-70b     --provider openrouter
```

### Real data

```bash
# TCGA-LGG / TCGA-GBM directory layout
python main.py \
  --mode tcga \
  --data_root data/TCGA-LGG/ \
  --model gpt-4o \
  --output_dir results/tcga-lgg/

# Same dataset, via OpenRouter with Claude
python main.py \
  --mode tcga \
  --data_root data/TCGA-LGG/ \
  --model anthropic/claude-opus-4 \
  --output_dir results/tcga-lgg-claude/

# Custom cohort via manifest CSV
python main.py \
  --mode manifest \
  --manifest data/my_cohort.csv \
  --data_root data/ \
  --model gpt-4o
```

### Cross-dataset evaluation (paper's 6-cohort protocol)

```bash
python evaluate_datasets.py \
  --config configs/datasets.json \
  --model gpt-4o \
  --output_dir results/cross_dataset/

# OpenRouter variant
python evaluate_datasets.py \
  --config configs/datasets.json \
  --model anthropic/claude-opus-4 \
  --output_dir results/cross_dataset_claude/
```

---

## CLI Reference

### `main.py`

| Flag | Default | Description |
|---|---|---|
| `--mode` | `synthetic` | Data source: `synthetic`, `tcga`, `brats`, `manifest` |
| `--data_root` | `data/` | Root directory for real MRI data |
| `--manifest` | — | CSV manifest path (required for `--mode manifest`) |
| `--model` | `gpt-4o` | Model ID — OpenAI or OpenRouter (see table below) |
| `--provider` | auto | `openai` or `openrouter`. Auto-detected: a `/` in the model name → OpenRouter |
| `--n_mutant` | `10` | Synthetic mutant subjects to generate |
| `--n_wildtype` | `10` | Synthetic wildtype subjects to generate |
| `--output_dir` | `results/` | Output directory |
| `--seg_method` | `threshold` | `threshold` or `brainsegfounder` |
| `--seg_checkpoint` | — | Path to BrainSegFounder checkpoint |
| `--reg_method` | `affine` | `auto`, `ants`, or `affine` |
| `--no_llm` | off | Extract features only, skip LLM calls |
| `--dry_run` | off | Use mock LLM responses (no API calls) |

---

## Supported Models

### OpenAI (key: `OPENAI_API_KEY`)

| Short alias | Full model ID |
|---|---|
| `gpt-4o` | `gpt-4o` |
| `gpt-4o-mini` | `gpt-4o-mini` |
| `gpt-5` | `gpt-5-chat-latest` |

### OpenRouter (key: `OPENROUTER_API_KEY`)

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys). The `openai` Python SDK is reused with `base_url="https://openrouter.ai/api/v1"`.

| Short alias | Full model ID | Notes |
|---|---|---|
| `claude-opus-4` | `anthropic/claude-opus-4` | Strongest reasoning |
| `claude-sonnet-4-5` | `anthropic/claude-sonnet-4-5` | Balanced |
| `claude-haiku-4-5` | `anthropic/claude-haiku-4-5-20251001` | Fast/cheap |
| `gemini-pro` | `google/gemini-pro-1.5` | Long context |
| `gemini-flash` | `google/gemini-flash-1.5` | Fast/cheap |
| `llama-70b` | `meta-llama/llama-3.3-70b-instruct` | Open-weight |
| `llama-8b` | `meta-llama/llama-3.1-8b-instruct` | Fastest/free tier |
| `mistral-large` | `mistralai/mistral-large` | |
| `qwen-72b` | `qwen/qwen-2.5-72b-instruct` | |
| *(any)* | Pass any `provider/model` slug | Browse all at [openrouter.ai/models](https://openrouter.ai/models) |

Provider is auto-detected from the model string — if it contains `/`, OpenRouter is used automatically.

---

## Data Formats

### TCGA-style directory layout

```
data/TCGA-LGG/
├── clinical.csv                 # subject_id, IDH, age, sex, grade
├── TCGA-CS-4941/
│   ├── TCGA-CS-4941_T1.nii.gz
│   ├── TCGA-CS-4941_T1ce.nii.gz
│   ├── TCGA-CS-4941_T2.nii.gz
│   └── TCGA-CS-4941_FLAIR.nii.gz
└── ...
```

`clinical.csv` must have at minimum `subject_id` and `IDH` (values: `Mutant`/`WT` or `1`/`0`).

### BraTS layout

```
data/BraTS2021/
├── survival_info.csv
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ...
```

### Manifest CSV (UPENN-GBM, Erasmus, DFCI, custom)

```csv
subject_id,t1,t1ce,t2,flair,seg,idh_label,age,sex
PAT001,PAT001_t1.nii.gz,PAT001_t1ce.nii.gz,PAT001_t2.nii.gz,PAT001_flair.nii.gz,,1,45,M
PAT002,PAT002_t1.nii.gz,PAT002_t1ce.nii.gz,PAT002_t2.nii.gz,PAT002_flair.nii.gz,,0,62,F
```

Paths in the CSV are relative to `--data_root`.

---

## Outputs

```
results/
├── features/
│   ├── SUBJECT_001.json     ← structured imaging features (per-region, per-modality)
│   └── ...
├── predictions.csv          ← subject_id, model, idh_status, confidence, correct, reasoning
└── metrics.json             ← accuracy, sensitivity, specificity, PPV, NPV, F1, AUC-ROC
```

### Example `metrics.json`

```json
{
  "n_total": 1427,
  "n_mutant": 743,
  "n_wildtype": 684,
  "accuracy": 0.834,
  "sensitivity": 0.871,
  "specificity": 0.793,
  "ppv": 0.812,
  "npv": 0.857,
  "f1": 0.840,
  "balanced_accuracy": 0.832,
  "auc_roc": 0.891,
  "model": "gpt-4o",
  "dataset": "TCGA-LGG"
}
```

---

## Project Structure

```
radish/
├── src/
│   ├── pipeline/
│   │   ├── preprocessing.py      # MRI I/O, normalization, skull stripping
│   │   ├── segmentation.py       # BrainSegFounder + threshold fallback
│   │   ├── registration.py       # ANTs SyN + affine fallback → MNI152
│   │   ├── atlas_mapping.py      # Harvard-Oxford, Juelich, Hammers
│   │   ├── feature_extraction.py # Per-region signal statistics
│   │   ├── serialization.py      # JSON + text narrative generation
│   │   ├── llm_predictor.py      # OpenAI API wrapper + response parser
│   │   └── evaluation.py         # Classification metrics
│   ├── data/
│   │   ├── data_loader.py        # TCGA, BraTS, manifest loaders
│   │   └── synthetic.py          # Synthetic MRI generator
│   └── prompts/
│       └── templates.py          # LLM system + user prompt templates
├── tests/
│   ├── test_milestone1.py        # 18 unit tests
│   └── test_milestone2.py        # 12 integration tests
├── main.py                       # Main CLI
├── evaluate_datasets.py          # Cross-dataset evaluation
└── requirements.txt
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/test_milestone1.py -v

# Integration tests only
python -m pytest tests/test_milestone2.py -v
```

All tests run without MRI data or an API key (uses synthetic data and mock LLM responses). Expected output: **55 passed**.

---

## Segmentation Options

| Method | Requirement | Notes |
|---|---|---|
| `threshold` *(default)* | None | Intensity-based heuristic; suitable for testing |
| `brainsegfounder` | MONAI, PyTorch, checkpoint file | SwinUNETR trained on BraTS; use for real data |

Download the BrainSegFounder checkpoint from the [MONAI Model Zoo](https://monai.io/model-zoo.html) and pass it via `--seg_checkpoint path/to/checkpoint.pt`.

---

## Registration Options

| Method | Requirement | Notes |
|---|---|---|
| `affine` *(default)* | None | Resampling only; fast but imprecise |
| `ants` | `antspyx` | Full SyN nonlinear warp to MNI152 |
| `auto` | — | Uses ANTs if installed, falls back to affine |

For best results on real data, install `antspyx` and use `--reg_method ants`.

---

## Supported Datasets

| Dataset | Loader | IDH labels |
|---|---|---|
| TCGA-LGG | `load_tcga()` | Yes (via clinical CSV) |
| TCGA-GBM | `load_tcga()` | Yes (via clinical CSV) |
| BraTS 2021 | `load_brats()` | Not released per-subject |
| UPENN-GBM | `load_from_manifest()` | Yes |
| Erasmus-GBM | `load_from_manifest()` | Yes |
| DFCI | `load_from_manifest()` | Yes |

---

## License

This implementation is provided for research and educational use. The paper's findings, model weights (BrainSegFounder), and public datasets are subject to their own respective licenses.
