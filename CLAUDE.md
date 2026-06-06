# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Radish** is a zero-shot medical imaging AI pipeline that predicts IDH (isocitrate dehydrogenase) mutation status in brain gliomas using MRI-derived features and LLMs. The implementation is described in:

> "Computational Imaging Meets LLMs: Zero-Shot IDH Mutation Prediction in Brain Gliomas"  
> Syed Muqeem Mahmood and Hassan Mohy-ud-Din (2025)  
> [arXiv:2511.03376](https://arxiv.org/abs/2511.03376)

It supports **OpenAI** (GPT-4o, GPT-5) and **OpenRouter** (Claude, Gemini, Llama, Mistral, and 300+ other models) via a single API key.

## Development Setup

### Installation

```bash
# Create and activate virtual environment
uv venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
uv pip install -r requirements.txt
```

### API Keys

Set up environment variables for LLM access:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...    # Linux/macOS
$env:OPENAI_API_KEY="sk-..."    # Windows PowerShell

# Or OpenRouter (supports 300+ models)
export OPENROUTER_API_KEY=sk-or-...

# Alternatively, create a .env file in the project root:
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=sk-or-...
```

## Running Tests

All tests run without MRI data or API keys (using synthetic data and mock LLM responses).

```bash
# Run all tests (59 total: 18 unit + 12 integration + 29 provider/alias tests)
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/test_milestone1.py -v

# Run integration tests only
python -m pytest tests/test_milestone2.py -v

# Run OpenRouter/provider tests
python -m pytest tests/test_openrouter.py -v

# Run a single test
python -m pytest tests/test_milestone1.py::test_normalize_zscore -v
```

Expected result: **59 passed** in ~28-30 seconds.

## Common CLI Commands

### Quick Start (No Data Required)

```bash
# Synthetic data, feature extraction only (no LLM calls)
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --no_llm

# Dry run with mock LLM responses (no API key needed, shows prediction format)
python main.py --mode synthetic --n_mutant 10 --n_wildtype 10 --dry_run
```

### With Real Data

```bash
# TCGA-LGG dataset with OpenAI
python main.py --mode tcga --data_root data/TCGA-LGG/ --model gpt-4o

# OpenRouter with alias (auto-detects provider from "/" in model name)
python main.py --mode synthetic --model anthropic/claude-opus-4 --n_mutant 5 --n_wildtype 5

# Custom manifest CSV
python main.py --mode manifest --manifest data/my_cohort.csv --data_root data/ --model gpt-4o
```

### Cross-Dataset Evaluation

```bash
python evaluate_datasets.py --config configs/datasets.json --model gpt-4o --output_dir results/cross_dataset/
```

### Segmentation & Registration Options

```bash
# Use threshold segmentation (default, fast, suitable for testing)
python main.py --mode synthetic --seg_method threshold

# Use BrainSegFounder (SwinUNETR, requires checkpoint)
python main.py --mode tcga --data_root data/TCGA-LGG/ --seg_method brainsegfounder --seg_checkpoint path/to/checkpoint.pt

# Use ANTs registration to MNI152 (requires antspyx, better accuracy)
python main.py --mode tcga --data_root data/TCGA-LGG/ --reg_method ants

# Auto-detect registration (ANTs if installed, falls back to affine)
python main.py --mode tcga --data_root data/TCGA-LGG/ --reg_method auto
```

## Architecture & Data Flow

The pipeline follows this sequence:

```
MRI Modalities (T1, T1-CE, T2, FLAIR)
    ↓
Preprocessing (normalization, skull stripping)
    ↓
Segmentation (BrainSegFounder or threshold heuristic)
    ↓
Registration (ANTs SyN or affine resampling)
    ↓
Atlas Mapping (Harvard-Oxford, Juelich, Hammers)
    ↓
Feature Extraction (per-region signal stats, enhancement ratios)
    ↓
Serialization (JSON + radiology-style text narrative)
    ↓
LLM Query (OpenAI or OpenRouter, zero-shot, no fine-tuning)
    ↓
IDH Prediction (mutant | wildtype + confidence + reasoning)
    ↓
Evaluation (accuracy, sensitivity, specificity, F1, AUC-ROC)
```

### Key Modules

**src/pipeline/**
- `preprocessing.py`: MRI I/O (nibabel), intensity normalization (z-score/minmax/percentile), skull stripping
  - Class: `MRISubject` – holds 4-modality images, affine matrix, voxel size, brain mask
  - Functions: `load_subject()`, `normalize_intensity()`, `simple_skull_strip()`, `preprocess_subject()`
  
- `segmentation.py`: Tumor segmentation with BraTS label convention
  - `TumorSegmenter` interface (abstract base)
  - `BrainSegFounderSegmenter`: SwinUNETR via MONAI (requires checkpoint)
  - `ThresholdSegmenter`: intensity-based fallback (suitable for testing)
  - `get_segmenter()`: factory function that handles both methods
  - Label mapping: 1=necrotic core, 2=peritumoral edema, 4=enhancing tumor
  
- `registration.py`: Brain alignment to MNI152 standard space
  - `register_to_mni_ants()`: Full nonlinear warp via ANTs SyN
  - `register_subject()`: Wrapper with auto-detection (ANTs if available, else affine)
  - Falls back to resampling-only (affine) when ANTs unavailable
  
- `atlas_mapping.py`: Anatomical region mapping from nilearn
  - `AtlasRegion`: dataclass with index, name, binary voxel mask
  - `AtlasMap`: container for multiple regions (Harvard-Oxford, Juelich, Hammers)
  - `load_all_atlases()`: fetches all three via nilearn (may download on first run)
  
- `feature_extraction.py`: Per-region signal statistics and derived metrics
  - `extract_region_features()`: compute mean/std/median/p5/p95 within atlas regions
  - `extract_label_features()`: global stats for tumor compartments (WT, ET, ED, NCR)
  - Computes enhancement ratio (T1-CE / T1) and FLAIR/T2 ratio
  - `extract_all_features()`: orchestrates all extraction, returns nested dict
  
- `serialization.py`: Feature dict → structured JSON and human-readable narrative
  - `to_json()`: serialize features to JSON (for storage/debugging)
  - `to_narrative()`: radiology-style plain-text report (inserted into LLM prompt)
  - `save_json()`: write JSON to disk
  
- `llm_predictor.py`: LLM API wrapper with multi-provider support
  - `LLMPredictor`: main class handling OpenAI and OpenRouter
  - `IDHPrediction`: dataclass with subject_id, status, confidence, reasoning, raw_response, latency
  - `_infer_provider()`: auto-detect OpenRouter from "/" in model name
  - `_resolve_model()`: expand aliases (e.g., "claude-opus-4" → "anthropic/claude-opus-4")
  - `_parse_response()`: extract JSON from LLM output (handles markdown fences + fallback regex)
  - Supports model aliases: gpt-4o, gpt-5, claude-opus-4, gemini-flash, llama-70b, etc.
  
- `evaluation.py`: Classification metrics (accuracy, sensitivity, specificity, F1, AUC-ROC)
  - `EvalMetrics`: dataclass with all standard metrics
  - `compute_metrics()`: compute from y_true/y_pred arrays
  - `evaluate_predictions()`: wrapper for `IDHPrediction` lists
  - `save_predictions_csv()`: write predictions to CSV with reasoning
  - `save_metrics()`: write metrics to JSON

**src/data/**
- `synthetic.py`: Synthetic MRI generator for testing
  - `generate_subject()`: create 4-modality synthetic data with IDH-specific signal profiles
    - IDH-mutant: frontal location, minimal enhancement, higher FLAIR/T2
    - IDH-wildtype (GBM-like): temporal location, strong ring enhancement, necrotic core
  - `generate_dataset()`: generate balanced mutant/wildtype cohort with controlled randomness
  
- `data_loader.py`: Multi-dataset support (TCGA, BraTS, manifest CSV)
  - `SubjectRecord`: standardized container (subject_id, modality_paths, idh_label, seg_path, clinical metadata)
  - `load_tcga()`: expects directory structure with clinical.csv
  - `load_brats()`: BraTS 2021 layout parser
  - `load_from_manifest()`: generic CSV loader (supports UPENN-GBM, Erasmus, DFCI, custom cohorts)
  - Automatic modality pattern matching (e.g., "*T1.nii.gz", "*t1ce*.nii.gz")

**src/prompts/**
- `templates.py`: LLM system and user prompt templates
  - `SYSTEM_PROMPT`: neuroradiologist persona with IDH imaging correlates
  - `USER_PROMPT_TEMPLATE`: feature report insertion point, JSON response format

### Output Structure

```
results/
├── features/
│   ├── SUBJECT_001.json     ← structured imaging features (per-region, per-modality)
│   └── ...
├── predictions.csv          ← subject_id, model, idh_status, confidence, correct, reasoning
├── metrics.json             ← accuracy, sensitivity, specificity, PPV, NPV, F1, AUC-ROC, balanced_acc
└── reasoning_traces.txt     ← full LLM responses for transparency
```

## Model Provider Selection

**OpenAI (default):**
- Models: gpt-4o, gpt-4o-mini, gpt-5
- Key: `OPENAI_API_KEY`
- Auto-used if no "/" in model name

**OpenRouter (unified API for 300+ models):**
- Models: `anthropic/claude-opus-4`, `google/gemini-flash-1.5`, `meta-llama/llama-3.3-70b-instruct`, etc.
- Key: `OPENROUTER_API_KEY`
- Auto-detected if model contains "/" (e.g., `python main.py --model anthropic/claude-opus-4`)
- Or explicitly: `python main.py --model claude-opus-4 --provider openrouter`

## Testing Strategy

The test suite is organized in three modules:

1. **test_milestone1.py** (18 tests): Unit tests for individual components
   - Preprocessing: normalization, skull stripping
   - Segmentation: label correctness, volume computation
   - Feature extraction: statistics, JSON structure
   - Serialization: narrative generation, JSON validity
   - Metrics: perfect predictions, AUC computation

2. **test_milestone2.py** (12 tests): Integration tests for full pipeline
   - End-to-end feature extraction (synthetic data)
   - Narrative generation across subjects
   - Mock LLM prediction and evaluation
   - CSV/JSON output file writing
   - Main CLI invocation (--no_llm, --dry_run)

3. **test_openrouter.py** (29 tests): Provider and alias resolution
   - Auto-detection of OpenAI vs OpenRouter from model name
   - Model alias expansion (e.g., "claude-opus-4" → "anthropic/claude-opus-4")
   - API key loading (environment or .env file)
   - Client initialization with correct base URLs and headers

Tests use `pytest` fixtures with synthetic data and mock LLM responses—no external API calls or real MRI files required.

## Key Conventions & Patterns

**IDH Labels:**
- `1` = IDH-mutant (typically lower-grade astrocytoma)
- `0` = IDH-wildtype (typically GBM)
- `None` = unlabeled/unknown

**MRI Modalities:**
- `t1`: T1-weighted (gray/white matter contrast)
- `t1ce`: T1-weighted with gadolinium contrast (enhancement = blood-brain barrier breakdown)
- `t2`: T2-weighted (fluid-sensitive)
- `flair`: FLAIR (fluid-attenuated inversion recovery; white matter hyperintensity)

**Segmentation Labels (BraTS convention):**
- `1` = NCR (necrotic core, hypointense on T1-CE)
- `2` = ED (peritumoral edema, hyperintense on T2/FLAIR)
- `4` = ET (enhancing tumor, bright on T1-CE)
- WT (whole tumor) = union of {1, 2, 4}

**Confidence Levels:**
- `high` (score: 0.9): Strong evidence from imaging
- `medium` (score: 0.65): Ambiguous or mixed signals
- `low` (score: 0.4): Uncertain or contradictory features

**Atlas Regions:**
Coordinates and region names follow MNI152 standard space (182×218×182 voxels at 1mm isotropic).

## Important Notes

- **Optional Deep Learning:** BrainSegFounder (SwinUNETR) requires MONAI, PyTorch, and a checkpoint file. Without these, the pipeline falls back to threshold-based segmentation (suitable for testing/prototyping).
- **Optional Registration:** ANTs nonlinear registration (antspyx) greatly improves accuracy on real data. Without it, only affine resampling (fast, less accurate) is available.
- **API Keys in .env:** The .env file is in .gitignore and safe to use for development. Never commit API keys.
- **Synthetic Data:** Useful for testing without real patient data. Generated subjects follow realistic IDH-specific imaging patterns.
- **Cross-Dataset Evaluation:** The evaluate_datasets.py script runs the pipeline on multiple cohorts and combines results into a single metrics table (useful for paper reproducibility).
