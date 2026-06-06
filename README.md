# Zero-Shot IDH Mutation Prediction in Brain Gliomas

A Python implementation of the pipeline described in:

> **"Computational Imaging Meets LLMs: Zero-Shot IDH Mutation Prediction in Brain Gliomas"**  
> Syed Muqeem Mahmood and Hassan Mohy-ud-Din (2025)  
> [arXiv:2511.03376](https://arxiv.org/abs/2511.03376)

Extended with a faithfulness evaluation layer described in:

> **"Faithful Reasoning in LLM-Based Radiological AI: Explainability Through Structured Rationales and Intervention Tests"** *(in preparation)*

The pipeline extracts interpretable imaging features from multi-parametric MRI, serializes them into a structured narrative, queries a large language model for a **structured rationale** (predicted label, decisive feature, supporting/contradicting features, draft radiology report), then evaluates whether the reasoning is faithful to the prediction through five automated tests. Supports **OpenAI** (GPT-4o, GPT-5) and **OpenRouter** (Claude, Gemini, Llama, Mistral, and 300+ other models via a single API key).

---

## Overview

IDH (isocitrate dehydrogenase) mutation status is a key molecular marker that defines glioma subtype and prognosis. Definitive diagnosis requires tissue biopsy, but imaging-derived features correlate strongly with IDH status. This project operationalizes that correlation through a zero-shot LLM reasoning approach and adds a faithfulness layer to verify that the model's stated reasoning actually drives its predictions.

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
  Serialization          JSON + feature vocabulary + radiology narrative
        │
        ▼
  LLM Query              structured rationale: label, decisive feature,
                         supporting/contradicting features, findings, impression
        │
        ▼
  Faithfulness Tests     sufficiency · comprehensiveness · citation–importance
                         alignment · counterfactual · rationale corruption
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

# 4. Set your API key
export OPENAI_API_KEY=sk-...           # OpenAI
export OPENROUTER_API_KEY=sk-or-...   # OpenRouter (300+ models)
# or add either to a .env file in the project root
```

---

## Quick Start

### Synthetic data (no MRI files or API key needed)

```bash
# Feature extraction only
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --no_llm

# Dry run with mock LLM responses
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --dry_run
```

### OpenAI

```bash
python main.py --mode synthetic --model gpt-4o
python main.py --mode synthetic --model gpt-5-chat-latest
```

### OpenRouter

```bash
# Provider is auto-detected from the slash in the model name
python main.py --mode synthetic --model anthropic/claude-opus-4
python main.py --mode synthetic --model google/gemini-flash-1.5

# Short aliases with explicit provider
python main.py --mode synthetic --model claude-opus-4   --provider openrouter
python main.py --mode synthetic --model claude-haiku-4-5 --provider openrouter
python main.py --mode synthetic --model llama-70b       --provider openrouter
```

### Real data

```bash
# TCGA-LGG
python main.py --mode tcga --data_root data/TCGA-LGG/ --model gpt-4o

# UCSF-PDGM v3  (T1c only — non-contrast T1 absent from this dataset)
python main.py --mode ucsf --data_root data/UCSF-PDGM/ \
  --provider openrouter --model claude-haiku-4-5 \
  --output_dir results/ucsf-pdgm/

# Custom cohort via manifest CSV
python main.py --mode manifest --manifest data/my_cohort.csv \
  --data_root data/ --model gpt-4o
```

### Live monitoring

While a run is in progress, start the monitor in a second terminal:

```bash
python monitor.py            # opens http://localhost:8765
python monitor.py --port 9000
```

The dashboard auto-refreshes every 3 seconds and shows progress, running accuracy, confidence distribution, and a scrollable predictions table. Click any row to inspect the full model input and structured output in a modal.

---

## CLI Reference

### `main.py`

| Flag | Default | Description |
|---|---|---|
| `--mode` | `synthetic` | Data source: `synthetic`, `tcga`, `brats`, `ucsf`, `manifest` |
| `--data_root` | `data/` | Root directory for real MRI data |
| `--manifest` | — | CSV manifest path (required for `--mode manifest`) |
| `--model` | `gpt-4o` | Model ID — OpenAI or OpenRouter (see table below) |
| `--provider` | auto | `openai` or `openrouter`; auto-detected from `/` in model name |
| `--n_mutant` | `10` | Synthetic mutant subjects to generate |
| `--n_wildtype` | `10` | Synthetic wildtype subjects to generate |
| `--output_dir` | `results/` | Output directory |
| `--seg_method` | `threshold` | `threshold` or `brainsegfounder` |
| `--seg_checkpoint` | — | Path to BrainSegFounder checkpoint |
| `--reg_method` | `affine` | `auto`, `ants`, or `affine` |
| `--seed` | `42` | Random seed; also passed to the LLM API for deterministic output |
| `--no_llm` | off | Extract features only, skip LLM calls |
| `--dry_run` | off | Use mock LLM responses (no API calls) |

### `monitor.py`

```bash
python monitor.py [--port PORT]
```

Serves a live HTML dashboard at `http://localhost:<port>` (default 8765). Reads `results/ucsf-pdgm/llm_log.jsonl` in real time. Click any prediction row to view the full model input/output in a tabbed modal.

---

## Supported Models

### OpenAI (key: `OPENAI_API_KEY`)

| Alias | Full model ID |
|---|---|
| `gpt-4o` | `gpt-4o` |
| `gpt-4o-mini` | `gpt-4o-mini` |
| `gpt-5` | `gpt-5-chat-latest` |

### OpenRouter (key: `OPENROUTER_API_KEY`)

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

| Alias | Full OpenRouter ID | Notes |
|---|---|---|
| `claude-opus-4` | `anthropic/claude-opus-4` | Strongest reasoning |
| `claude-sonnet-4-5` | `anthropic/claude-sonnet-4.5` | Balanced |
| `claude-haiku-4-5` | `anthropic/claude-haiku-4.5` | Fast, low cost |
| `gemini-pro` | `google/gemini-pro-1.5` | Long context |
| `gemini-flash` | `google/gemini-flash-1.5` | Fast, low cost |
| `llama-70b` | `meta-llama/llama-3.3-70b-instruct` | Open-weight |
| `llama-8b` | `meta-llama/llama-3.1-8b-instruct` | Fastest / free tier |
| `mistral-large` | `mistralai/mistral-large` | |
| `qwen-72b` | `qwen/qwen-2.5-72b-instruct` | |
| *(any)* | Pass any `provider/model` slug directly | Browse [openrouter.ai/models](https://openrouter.ai/models) |

Provider is auto-detected: a `/` in the model name → OpenRouter.

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

### UCSF-PDGM v3 (TCIA)

Download from [TCIA](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/).  
Non-contrast T1 is absent from this dataset; the pipeline uses T1c, T2, and FLAIR.

```
data/UCSF-PDGM/
├── UCSF-PDGM-metadata_v2.csv   # ID, Sex, Age, WHO Grade, IDH, ...
└── UCSF-PDGM-v3/
    ├── UCSF-PDGM-0004_nifti/
    │   ├── UCSF-PDGM-0004_T1c.nii.gz
    │   ├── UCSF-PDGM-0004_T2.nii.gz
    │   ├── UCSF-PDGM-0004_FLAIR.nii.gz
    │   └── UCSF-PDGM-0004_tumor_segmentation.nii.gz
    └── ...
```

IDH labels parsed from the `IDH` column: `wildtype` → 0; `mutated (NOS)`, `IDH1 p.R132H`, and all other specific mutation variants → 1.

### BraTS layout

```
data/BraTS2021/
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
```

Paths are relative to `--data_root`.

---

## Outputs

```
results/
├── features/
│   └── SUBJECT.json        ← imaging features (label stats + atlas regions)
├── llm_log.jsonl           ← one JSON entry per prediction (written live)
├── predictions.csv         ← per-subject predictions + structured rationale fields
├── reasoning_traces.txt    ← full untruncated reasoning strings
└── metrics.json            ← accuracy, sensitivity, specificity, PPV, NPV, F1, AUC-ROC
```

### Structured rationale fields in `predictions.csv`

Beyond the original `idh_status`, `confidence`, and `reasoning`, each prediction now includes:

| Field | Description |
|---|---|
| `decisive_feature` | Single most decisive feature identifier cited by the model |
| `supporting_features` | Comma-separated list of supporting feature identifiers |
| `contradicting_features` | Comma-separated list of contradicting feature identifiers |
| `findings` | Draft Findings paragraph for a radiology report |
| `impression` | Draft Impression sentence |
| `phantom_citations` | Feature names cited by the model that are not in the input JSON |
| `model_version` | Exact model version returned by the API |

Feature identifiers use dot notation tied to the input JSON vocabulary, e.g. `wt.t1ce.mean`, `et.volume_mm3`, `atlas.harvard_oxford.Frontal_Pole.t1.mean`.

---

## Faithfulness Evaluation

The faithfulness module (`src/pipeline/faithfulness.py`) verifies that the model's stated reasoning actually drives its predictions, using five automated tests:

| Test | Method | Faithful if… |
|---|---|---|
| **Sufficiency** | Re-query with only cited features kept | Prediction and confidence preserved |
| **Comprehensiveness** | Re-query with cited features removed | Prediction degrades (label changes or confidence drops) |
| **Citation–importance alignment** | Ablate each feature; group by citation strength (none/low/high) | High-cited features cause larger performance drop when ablated |
| **Counterfactual consistency** | Ask model for minimal flip; apply it; re-query | Prediction actually flips |
| **Rationale corruption** | Generate opposite-class rationale; query for label only | Label follows the supplied rationale |

Each test returns per-case results and cohort-level scores with 95% Wilson confidence intervals.

### Running faithfulness evaluation

```python
from src.pipeline.faithfulness import FaithfulnessCase, run_all_faithfulness_tests, save_faithfulness_results

cases = [
    FaithfulnessCase(
        subject_id=pred.subject_id,
        features=features_dict,        # from feature extraction
        original_prediction=pred,      # IDHPrediction with structured rationale
        ground_truth=pred.ground_truth,
    )
    for pred, features_dict in zip(predictions, all_features)
]

results = run_all_faithfulness_tests(predictor, cases)
save_faithfulness_results(results, "results/faithfulness.json")
```

The `run_citation_importance=False` flag skips Test 3 (which makes O(n_cases × n_features) LLM calls).

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
│   │   ├── serialization.py      # JSON + narrative generation
│   │   ├── llm_predictor.py      # LLM API wrapper + structured rationale parser
│   │   ├── evaluation.py         # Classification metrics
│   │   ├── interventions.py      # Feature modification for faithfulness tests
│   │   └── faithfulness.py       # Five faithfulness tests + cohort aggregation
│   ├── data/
│   │   ├── data_loader.py        # TCGA, BraTS, UCSF-PDGM, manifest loaders
│   │   └── synthetic.py          # Synthetic MRI generator
│   └── prompts/
│       └── templates.py          # LLM prompt templates (structured output schema)
├── tests/
│   ├── test_milestone1.py        # 18 unit tests (preprocessing, features, metrics)
│   ├── test_milestone2.py        # 12 integration tests (full pipeline, CLI)
│   ├── test_openrouter.py        # 29 provider / alias tests
│   ├── test_faithfulness.py      # 54 tests (structured rationale, interventions,
│   │                             #           faithfulness tests, determinism)
│   └── test_ucsf_pdgm.py         # 23 tests (UCSF-PDGM loader, live smoke tests)
├── main.py                       # Main CLI
├── monitor.py                    # Live web dashboard for monitoring runs
├── evaluate_datasets.py          # Cross-dataset evaluation
└── requirements.txt
```

---

## Running Tests

```bash
# All 136 tests
python -m pytest tests/ -v

# By module
python -m pytest tests/test_milestone1.py -v        # 18 unit tests
python -m pytest tests/test_milestone2.py -v        # 12 integration tests
python -m pytest tests/test_openrouter.py -v        # 29 provider/alias tests
python -m pytest tests/test_faithfulness.py -v      # 54 faithfulness tests
python -m pytest tests/test_ucsf_pdgm.py -v         # 23 UCSF-PDGM tests
                                                    #    (live tests skipped if data absent)
```

All tests run without MRI data or an API key (uses synthetic data and mock LLM responses).

---

## Supported Datasets

| Dataset | Mode | Loader | IDH labels |
|---|---|---|---|
| TCGA-LGG | `tcga` | `load_tcga()` | Yes (clinical CSV) |
| TCGA-GBM | `tcga` | `load_tcga()` | Yes (clinical CSV) |
| BraTS 2021 | `brats` | `load_brats()` | Not released per-subject |
| UCSF-PDGM v3 | `ucsf` | `load_ucsf_pdgm()` | Yes (metadata CSV) |
| UPENN-GBM | `manifest` | `load_from_manifest()` | Yes |
| Erasmus-GBM | `manifest` | `load_from_manifest()` | Yes |
| DFCI | `manifest` | `load_from_manifest()` | Yes |

---

## Segmentation Options

| Method | Requirement | Notes |
|---|---|---|
| `threshold` *(default)* | None | Intensity-based heuristic; suitable for testing |
| `brainsegfounder` | MONAI, PyTorch, checkpoint | SwinUNETR trained on BraTS; recommended for real data |

Download the BrainSegFounder checkpoint from the [MONAI Model Zoo](https://monai.io/model-zoo.html) and pass it via `--seg_checkpoint`.

---

## Registration Options

| Method | Requirement | Notes |
|---|---|---|
| `affine` *(default)* | None | Resampling only; fast |
| `ants` | `antspyx` | Full SyN nonlinear warp to MNI152 |
| `auto` | — | Uses ANTs if installed, falls back to affine |

For best accuracy on real data, install `antspyx` and use `--reg_method ants`.

---

## License

This implementation is provided for research and educational use. The paper's findings, model weights (BrainSegFounder), and public datasets are subject to their own respective licenses.
