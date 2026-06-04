"""
Milestone 1: preprocessing + segmentation + serialization smoke tests.
No real MRI data or API keys required.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Synthetic data ────────────────────────────────────────────────────────────

def make_fake_images(shape=(32, 32, 32)):
    rng = np.random.default_rng(0)
    return {
        "t1":    rng.normal(0.5, 0.1, shape).astype(np.float32),
        "t1ce":  rng.normal(0.55, 0.12, shape).astype(np.float32),
        "t2":    rng.normal(0.65, 0.15, shape).astype(np.float32),
        "flair": rng.normal(0.70, 0.13, shape).astype(np.float32),
    }


def make_fake_seg(shape=(32, 32, 32)):
    seg = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = [s // 2 for s in shape]
    seg[cx-3:cx+3, cy-3:cy+3, cz-3:cz+3] = 4   # ET core
    seg[cx-5:cx+5, cy-5:cy+5, cz-5:cz+5] = 2   # ED rim (overwritten by ET below)
    seg[cx-3:cx+3, cy-3:cy+3, cz-3:cz+3] = 4   # restore ET
    return seg


# ── Preprocessing ────────────────────────────────────────────────────────────

def test_normalize_zscore():
    from src.pipeline.preprocessing import normalize_intensity
    vol = np.random.default_rng(1).normal(100, 20, (32, 32, 32)).astype(np.float32)
    out = normalize_intensity(vol, method="zscore")
    brain = out[vol > 0]
    assert abs(brain.mean()) < 0.1, "z-score mean should be near 0"
    assert abs(brain.std() - 1.0) < 0.1, "z-score std should be near 1"


def test_normalize_minmax():
    from src.pipeline.preprocessing import normalize_intensity
    vol = np.random.default_rng(2).uniform(10, 200, (32, 32, 32)).astype(np.float32)
    out = normalize_intensity(vol, method="minmax")
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


def test_skull_strip_returns_binary():
    from src.pipeline.preprocessing import simple_skull_strip
    vol = np.random.default_rng(3).normal(0.5, 0.1, (32, 32, 32)).astype(np.float32)
    mask = simple_skull_strip(vol)
    assert set(np.unique(mask)).issubset({0, 1}), "Mask should be binary"
    assert mask.sum() > 0, "Mask should not be empty"


def test_preprocess_subject():
    from src.pipeline.preprocessing import MRISubject, preprocess_subject
    import nibabel as nib
    images = make_fake_images()
    subject = MRISubject(
        subject_id="test_001",
        images=images,
        affine=np.eye(4),
        header=nib.Nifti1Header(),
        voxel_size_mm=(3.0, 3.0, 3.0),
    )
    subject = preprocess_subject(subject)
    assert subject.brain_mask is not None
    assert subject.brain_mask.shape == (32, 32, 32)
    for mod, vol in subject.images.items():
        assert vol.dtype == np.float32, f"{mod} should be float32"


# ── Segmentation ─────────────────────────────────────────────────────────────

def test_threshold_segmenter_labels():
    from src.pipeline.segmentation import ThresholdSegmenter
    images = make_fake_images()
    seg = ThresholdSegmenter().segment(images, np.eye(4))
    unique = set(np.unique(seg))
    assert unique.issubset({0, 1, 2, 4}), f"Unexpected labels: {unique}"


def test_compute_tumor_volumes():
    from src.pipeline.segmentation import compute_tumor_volumes
    seg = make_fake_seg()
    vols = compute_tumor_volumes(seg, (1.0, 1.0, 1.0))
    assert "wt_mm3" in vols
    assert vols["wt_mm3"] >= vols["et_mm3"]
    assert all(v >= 0 for v in vols.values())


# ── Feature extraction ───────────────────────────────────────────────────────

def test_label_features_keys():
    from src.pipeline.feature_extraction import extract_label_features
    images = make_fake_images()
    seg    = make_fake_seg()
    feats  = extract_label_features(images, seg, (1.0, 1.0, 1.0))
    assert "wt" in feats
    assert "et" in feats
    assert "volume_mm3" in feats["wt"]
    assert "t1" in feats["wt"]
    assert "mean" in feats["wt"]["t1"]


def test_extract_all_features_no_atlases():
    from src.pipeline.feature_extraction import extract_all_features
    images = make_fake_images()
    seg    = make_fake_seg()
    feats  = extract_all_features(images, seg, atlases={}, voxel_size_mm=(1.0, 1.0, 1.0))
    assert "label_stats" in feats
    assert "atlas_regions" in feats
    assert feats["atlas_regions"] == {}


# ── Serialization ────────────────────────────────────────────────────────────

def test_to_narrative_contains_keywords():
    from src.pipeline.feature_extraction import extract_all_features
    from src.pipeline.serialization import to_narrative
    images   = make_fake_images()
    seg      = make_fake_seg()
    features = extract_all_features(images, seg, atlases={}, voxel_size_mm=(1.0, 1.0, 1.0))
    narr     = to_narrative("sub001", features)
    assert "sub001" in narr
    assert "Volume" in narr
    assert "T1" in narr or "t1" in narr.lower()


def test_to_json_valid():
    import json
    from src.pipeline.feature_extraction import extract_all_features
    from src.pipeline.serialization import to_json
    images   = make_fake_images()
    seg      = make_fake_seg()
    features = extract_all_features(images, seg, atlases={}, voxel_size_mm=(1.0, 1.0, 1.0))
    j        = to_json("sub001", features)
    parsed   = json.loads(j)
    assert parsed["subject_id"] == "sub001"
    assert "imaging_features" in parsed


# ── Synthetic generator ──────────────────────────────────────────────────────

def test_synthetic_generate_shapes():
    from src.data.synthetic import generate_dataset
    subjects = generate_dataset(n_mutant=2, n_wildtype=2, seed=0)
    assert len(subjects) == 4
    for s in subjects:
        assert "images" in s
        assert "seg" in s
        for mod in ("t1", "t1ce", "t2", "flair"):
            assert mod in s["images"]
            assert s["images"][mod].ndim == 3


def test_synthetic_label_balance():
    from src.data.synthetic import generate_dataset
    subjects = generate_dataset(n_mutant=5, n_wildtype=5, seed=1)
    mutants   = sum(1 for s in subjects if s["idh_label"] == 1)
    wildtypes = sum(1 for s in subjects if s["idh_label"] == 0)
    assert mutants == 5
    assert wildtypes == 5


# ── Evaluation ───────────────────────────────────────────────────────────────

def test_compute_metrics_perfect():
    from src.pipeline.evaluation import compute_metrics
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]
    m = compute_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.sensitivity == 1.0
    assert m.specificity == 1.0


def test_compute_metrics_with_auc():
    from src.pipeline.evaluation import compute_metrics
    y_true  = [1, 1, 0, 0]
    y_pred  = [1, 0, 0, 1]
    y_score = [0.9, 0.4, 0.2, 0.7]
    m = compute_metrics(y_true, y_pred, y_score)
    assert 0.0 <= m.auc_roc <= 1.0
    assert m.accuracy == 0.5


def test_compute_metrics_balanced_acc():
    from src.pipeline.evaluation import compute_metrics
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    m = compute_metrics(y_true, y_pred)
    expected_bal = (m.sensitivity + m.specificity) / 2
    assert abs(m.balanced_accuracy - expected_bal) < 1e-9


# ── LLM response parsing ─────────────────────────────────────────────────────

def test_parse_llm_response_clean_json():
    from src.pipeline.llm_predictor import _parse_response
    text = '{"idh_status": "IDH-mutant", "confidence": "high", "reasoning": "Frontal location."}'
    r = _parse_response(text)
    assert r["idh_status"] == "IDH-mutant"
    assert r["confidence"] == "high"


def test_parse_llm_response_fenced():
    from src.pipeline.llm_predictor import _parse_response
    text = '```json\n{"idh_status": "IDH-wildtype", "confidence": "medium", "reasoning": "Ring enhancement."}\n```'
    r = _parse_response(text)
    assert r["idh_status"] == "IDH-wildtype"


def test_parse_llm_response_malformed_fallback():
    from src.pipeline.llm_predictor import _parse_response
    text = "The tumour shows IDH-mutant features based on frontal location."
    r = _parse_response(text)
    assert "idh_status" in r
    assert "mutant" in r["idh_status"].lower()
