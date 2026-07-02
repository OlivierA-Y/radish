"""
Milestone 2: full pipeline integration tests — synthetic data, no API key needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Full feature pipeline (synthetic images → JSON) ───────────────────────────

def _run_feature_pipeline(n_subjects: int = 4):
    from src.data.synthetic import generate_dataset
    from src.pipeline.feature_extraction import extract_all_features
    from src.pipeline.serialization import to_json

    subjects = generate_dataset(n_mutant=n_subjects // 2, n_wildtype=n_subjects // 2, seed=7)
    results = []
    for s in subjects:
        features = extract_all_features(
            images=s["images"],
            seg=s["seg"],
            atlases={},
            voxel_size_mm=s["voxel_size"],
        )
        json_str = to_json(s["subject_id"], features)
        results.append({"subject": s, "features": features, "json": json_str})
    return results


def test_full_feature_pipeline_runs():
    results = _run_feature_pipeline(4)
    assert len(results) == 4


def test_feature_pipeline_volumes_positive():
    results = _run_feature_pipeline(4)
    for r in results:
        wt_vol = r["features"]["quantitative"]["volumes (ml)"]["WT"]
        assert wt_vol > 0, "Whole-tumour volume must be positive"


def test_feature_pipeline_json_parses():
    results = _run_feature_pipeline(2)
    for r in results:
        parsed = json.loads(r["json"])
        assert "subject_id"       in parsed
        assert "semantic_visual"  in parsed
        assert "quantitative"     in parsed



# ── Mock LLM prediction ───────────────────────────────────────────────────────

def _mock_predict(subject_id, ground_truth=None):
    import hashlib
    from src.pipeline.llm_predictor import IDHPrediction
    h = int(hashlib.md5(subject_id.encode()).hexdigest(), 16)
    status = "IDH-mutant" if h % 2 == 0 else "IDH-wildtype"
    return IDHPrediction(
        subject_id=subject_id,
        model="mock",
        idh_status=status,
        reasoning="Mock",
        raw_response='{"idh_status":"' + status + '","reasoning":"Mock."}',
        latency_s=0.001,
        ground_truth=ground_truth,
    )


def test_mock_predictions_have_correct_fields():
    from src.data.synthetic import generate_dataset

    subjects = generate_dataset(n_mutant=3, n_wildtype=3, seed=99)
    for s in subjects:
        pred = _mock_predict(s["subject_id"], ground_truth=s["idh_label"])
        assert pred.idh_status in ("IDH-mutant", "IDH-wildtype")
        assert pred.ground_truth in (0, 1)
        assert pred.correct is not None
        assert pred.label in (0, 1)


def test_prediction_label_matches_status():
    from src.pipeline.llm_predictor import IDHPrediction
    p_mut = IDHPrediction("s1", "mock", "IDH-mutant",  "r", "{}", 0.0, ground_truth=1)
    p_wt  = IDHPrediction("s2", "mock", "IDH-wildtype", "r", "{}", 0.0, ground_truth=0)
    assert p_mut.label == 1
    assert p_wt.label  == 0


# ── Evaluation on mock predictions ───────────────────────────────────────────

def test_evaluate_predictions_metrics():
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import evaluate_predictions

    preds = [
        IDHPrediction("s1", "mock", "IDH-mutant",   "r", "{}", 0.0, ground_truth=1),
        IDHPrediction("s2", "mock", "IDH-mutant",   "r", "{}", 0.0, ground_truth=1),
        IDHPrediction("s3", "mock", "IDH-wildtype", "r", "{}", 0.0, ground_truth=0),
        IDHPrediction("s4", "mock", "IDH-wildtype", "r", "{}", 0.0, ground_truth=0),
    ]
    m = evaluate_predictions(preds)
    assert m.accuracy     == 1.0
    assert m.sensitivity  == 1.0
    assert m.specificity  == 1.0
    assert m.n_total      == 4
    assert m.n_mutant     == 2
    assert m.n_wildtype   == 2


def test_evaluate_saves_csv(tmp_path):
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import save_predictions_csv

    preds = [
        IDHPrediction("s1", "mock", "IDH-mutant", "reasoning", "{}", 0.5, ground_truth=1),
    ]
    csv_path = tmp_path / "preds.csv"
    save_predictions_csv(preds, csv_path)
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "s1" in content
    assert "IDH-mutant" in content


def test_evaluate_saves_metrics_json(tmp_path):
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import evaluate_predictions, save_metrics

    preds = [
        IDHPrediction("s1", "mock", "IDH-mutant",   "r", "{}", 0.0, ground_truth=1),
        IDHPrediction("s2", "mock", "IDH-wildtype", "r", "{}", 0.0, ground_truth=0),
    ]
    m = evaluate_predictions(preds)
    path = tmp_path / "metrics.json"
    save_metrics(m, path)
    parsed = json.loads(path.read_text())
    assert parsed["accuracy"] == 1.0
    assert "auc_roc" in parsed


# ── Reasoning traces ─────────────────────────────────────────────────────────

def test_save_reasoning_traces_format(tmp_path):
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import save_reasoning_traces

    preds = [
        IDHPrediction("SYN_MUTANT_001",   "mock", "IDH-mutant",
                      "Frontal location with low enhancement.", "{}", 0.0),
        IDHPrediction("SYN_WILDTYPE_001", "mock", "IDH-wildtype",
                      "Ring-enhancing lesion with necrotic core.", "{}", 0.0),
    ]
    path = tmp_path / "reasoning_traces.txt"
    save_reasoning_traces(preds, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == 'SYN_MUTANT_001: "Frontal location with low enhancement."'
    assert lines[1] == 'SYN_WILDTYPE_001: "Ring-enhancing lesion with necrotic core."'


def test_save_reasoning_traces_not_truncated(tmp_path):
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import save_reasoning_traces

    long_reasoning = "A" * 2000  # well beyond the 200-char CSV truncation
    preds = [IDHPrediction("S1", "mock", "IDH-mutant", long_reasoning, "{}", 0.0)]
    path = tmp_path / "traces.txt"
    save_reasoning_traces(preds, path)

    content = path.read_text(encoding="utf-8")
    assert long_reasoning in content, "Full reasoning must not be truncated"


def test_save_reasoning_traces_escapes_quotes(tmp_path):
    from src.pipeline.llm_predictor import IDHPrediction
    from src.pipeline.evaluation import save_reasoning_traces

    reasoning_with_quotes = 'He said "IDH-mutant" based on the T2 signal.'
    preds = [IDHPrediction("S1", "mock", "IDH-mutant", reasoning_with_quotes, "{}", 0.0)]
    path = tmp_path / "traces.txt"
    save_reasoning_traces(preds, path)

    line = path.read_text(encoding="utf-8").strip()
    # The line must be parseable: outer quotes wrap an escaped interior
    assert line.startswith('S1: "')
    assert line.endswith('"')
    assert '\\"' in line, "Embedded double-quotes must be escaped"


def test_save_reasoning_traces_empty(tmp_path):
    from src.pipeline.evaluation import save_reasoning_traces

    path = tmp_path / "traces.txt"
    save_reasoning_traces([], path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


# ── main.py --no_llm flag (feature extraction only, no API call) ──────────────

def test_main_no_llm_creates_feature_files(tmp_path):
    import subprocess, sys
    result = subprocess.run(
        [
            sys.executable, "main.py",
            "--mode", "synthetic",
            "--n_mutant", "3",
            "--n_wildtype", "3",
            "--no_llm",
            "--output_dir", str(tmp_path),
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    feat_dir = tmp_path / "features"
    json_files = list(feat_dir.glob("*.json"))
    assert len(json_files) == 6, f"Expected 6 feature JSON files, got {len(json_files)}"
    for jf in json_files:
        parsed = json.loads(jf.read_text())
        assert "subject_id"      in parsed
        assert "semantic_visual" in parsed


def test_main_dry_run_creates_predictions_csv(tmp_path):
    import subprocess, sys
    result = subprocess.run(
        [
            sys.executable, "main.py",
            "--mode", "synthetic",
            "--n_mutant", "3",
            "--n_wildtype", "3",
            "--dry_run",
            "--output_dir", str(tmp_path),
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    run_dirs = list((tmp_path / "runs").glob("*/"))
    assert run_dirs, "runs/ subdirectory should be created"
    run_dir = run_dirs[0]
    preds_csv = run_dir / "predictions.csv"
    assert preds_csv.exists(), "predictions.csv should be created"
    traces_txt = run_dir / "reasoning_traces.txt"
    assert traces_txt.exists(), "reasoning_traces.txt should be created"
    trace_lines = traces_txt.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 6, f"Expected 6 trace lines, got {len(trace_lines)}"
    for line in trace_lines:
        assert ': "' in line, f"Line missing expected format: {line!r}"
    lines = preds_csv.read_text().splitlines()
    assert len(lines) == 7, f"Expected header + 6 rows, got {len(lines)}"

    metrics_json = run_dir / "metrics.json"
    assert metrics_json.exists(), "metrics.json should be created"
    m = json.loads(metrics_json.read_text())
    assert m["n_total"] == 6
