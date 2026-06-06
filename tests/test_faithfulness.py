"""
Tests for Changes 1-4:
  Change 1 — Structured rationale output (IDHPrediction extended fields,
              feature vocabulary, phantom citation detection)
  Change 2 — Intervention layer (vocab map, null/keep/remove/substitute)
  Change 3 — Five faithfulness tests (using mock predictor, no API calls)
  Change 4 — Determinism (seed stored on LLMPredictor, model_version field)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_simple_features() -> Dict:
    """Minimal features dict matching the schema expected by interventions."""
    return {
        "label_stats": {
            "wt": {
                "volume_mm3": 1500.0,
                "t1":   {"mean": 0.45, "std": 0.10, "median": 0.44, "p5": 0.30, "p95": 0.60, "skewness": 0.1},
                "t1ce": {"mean": 0.80, "std": 0.15, "median": 0.78, "p5": 0.55, "p95": 1.05, "skewness": 0.2},
                "t2":   {"mean": 0.65, "std": 0.12, "median": 0.64, "p5": 0.45, "p95": 0.85, "skewness": 0.0},
                "flair":{"mean": 0.70, "std": 0.13, "median": 0.69, "p5": 0.50, "p95": 0.90, "skewness": 0.1},
                "enhancement_ratio": 1.78,
            },
            "et": {
                "volume_mm3": 500.0,
                "t1":   {"mean": 0.50, "std": 0.08, "median": 0.49, "p5": 0.38, "p95": 0.63, "skewness": 0.0},
                "t1ce": {"mean": 1.20, "std": 0.20, "median": 1.18, "p5": 0.85, "p95": 1.55, "skewness": 0.3},
                "t2":   {"mean": 0.60, "std": 0.10, "median": 0.59, "p5": 0.43, "p95": 0.77, "skewness": 0.1},
                "flair":{"mean": 0.65, "std": 0.11, "median": 0.64, "p5": 0.47, "p95": 0.84, "skewness": 0.1},
                "enhancement_ratio": 2.40,
            },
        },
        "atlas_regions": {
            "harvard_oxford": {
                "Frontal Pole": {
                    "t1":   {"mean": 0.42, "std": 0.09, "median": 0.41, "p5": 0.28, "p95": 0.57, "skewness": 0.0},
                    "t1ce": {"mean": 0.55, "std": 0.11, "median": 0.54, "p5": 0.37, "p95": 0.73, "skewness": 0.1},
                    "t2":   {"mean": 0.60, "std": 0.10, "median": 0.59, "p5": 0.44, "p95": 0.77, "skewness": 0.0},
                    "flair":{"mean": 0.65, "std": 0.12, "median": 0.64, "p5": 0.46, "p95": 0.85, "skewness": 0.1},
                    "enhancement_ratio": 1.31,
                },
            },
        },
    }


def make_mutant_prediction(subject_id="S1", decisive="wt.enhancement_ratio",
                           supporting=None, contradicting=None):
    from src.pipeline.llm_predictor import IDHPrediction
    return IDHPrediction(
        subject_id=subject_id,
        model="mock",
        idh_status="IDH-mutant",
        confidence="high",
        reasoning="Low enhancement ratio typical of IDH-mutant.",
        raw_response="{}",
        latency_s=0.0,
        ground_truth=1,
        decisive_feature=decisive,
        supporting_features=supporting or ["wt.flair.mean"],
        contradicting_features=contradicting or ["et.t1ce.mean"],
        findings="Frontal glioma with low enhancement.",
        impression="IDH-mutant glioma.",
        phantom_citations=[],
        model_version="mock-v1",
    )


# ---------------------------------------------------------------------------
# Change 1: Structured rationale / IDHPrediction extended fields
# ---------------------------------------------------------------------------

class TestStructuredRationale:

    def test_idh_prediction_new_fields_default_none(self):
        from src.pipeline.llm_predictor import IDHPrediction
        p = IDHPrediction("s1", "mock", "IDH-mutant", "high", "r", "{}", 0.0, ground_truth=1)
        assert p.decisive_feature is None
        assert p.supporting_features is None
        assert p.contradicting_features is None
        assert p.findings is None
        assert p.impression is None
        assert p.phantom_citations is None
        assert p.model_version is None
        assert p.correct is True

    def test_idh_prediction_stores_structured_fields(self):
        from src.pipeline.llm_predictor import IDHPrediction
        p = IDHPrediction(
            subject_id="s1", model="mock",
            idh_status="IDH-mutant", confidence="high",
            reasoning="Frontal.", raw_response="{}", latency_s=0.0,
            ground_truth=1,
            decisive_feature="wt.enhancement_ratio",
            supporting_features=["wt.flair.mean"],
            contradicting_features=["et.t1ce.mean"],
            findings="Frontal glioma.",
            impression="IDH-mutant.",
            phantom_citations=[],
            model_version="gpt-4o-2024-11-20",
        )
        assert p.decisive_feature == "wt.enhancement_ratio"
        assert p.supporting_features == ["wt.flair.mean"]
        assert p.contradicting_features == ["et.t1ce.mean"]
        assert p.findings == "Frontal glioma."
        assert p.impression == "IDH-mutant."
        assert p.phantom_citations == []
        assert p.model_version == "gpt-4o-2024-11-20"

    def test_cited_identifiers_method(self):
        p = make_mutant_prediction(
            decisive="wt.enhancement_ratio",
            supporting=["wt.flair.mean", "wt.volume_mm3"],
            contradicting=["et.t1ce.mean"],
        )
        cited = p.cited_identifiers()
        assert "wt.enhancement_ratio" in cited
        assert "wt.flair.mean" in cited
        assert "wt.volume_mm3" in cited
        assert "et.t1ce.mean" in cited

    def test_cited_identifiers_empty_when_no_rationale(self):
        from src.pipeline.llm_predictor import IDHPrediction
        p = IDHPrediction("s1", "mock", "IDH-mutant", "high", "r", "{}", 0.0)
        assert p.cited_identifiers() == []

    def test_parse_response_extracts_new_fields(self):
        from src.pipeline.llm_predictor import _parse_response
        text = json.dumps({
            "idh_status": "IDH-mutant",
            "confidence": "high",
            "reasoning": "Frontal location.",
            "decisive_feature": "wt.enhancement_ratio",
            "supporting_features": ["wt.flair.mean"],
            "contradicting_features": ["et.t1ce.mean"],
            "findings": "Frontal glioma with low enhancement.",
            "impression": "Consistent with IDH-mutant glioma.",
        })
        r = _parse_response(text)
        assert r["decisive_feature"] == "wt.enhancement_ratio"
        assert r["supporting_features"] == ["wt.flair.mean"]
        assert r["contradicting_features"] == ["et.t1ce.mean"]
        assert "Frontal" in r["findings"]
        assert "impression" in r

    def test_parse_response_old_format_backward_compatible(self):
        from src.pipeline.llm_predictor import _parse_response
        text = '{"idh_status": "IDH-wildtype", "confidence": "medium", "reasoning": "Ring enhancement."}'
        r = _parse_response(text)
        assert r["idh_status"] == "IDH-wildtype"
        assert r["confidence"] == "medium"
        # New fields should be empty/None when absent from old-format JSON
        assert r.get("supporting_features", []) == [] or r.get("supporting_features") is None

    def test_phantom_citations_detected(self):
        from src.pipeline.llm_predictor import _compute_phantom_citations
        vocabulary = ["wt.t1.mean", "wt.enhancement_ratio", "et.volume_mm3"]
        phantoms = _compute_phantom_citations(
            decisive="nonexistent.feature",
            supporting=["wt.t1.mean", "also.fake"],
            contradicting=["wt.enhancement_ratio"],
            vocabulary=vocabulary,
        )
        assert "nonexistent.feature" in phantoms
        assert "also.fake" in phantoms
        assert "wt.t1.mean" not in phantoms
        assert "wt.enhancement_ratio" not in phantoms

    def test_phantom_citations_empty_when_no_vocab(self):
        from src.pipeline.llm_predictor import _compute_phantom_citations
        phantoms = _compute_phantom_citations("wt.t1.mean", ["wt.enhancement_ratio"], [], None)
        assert phantoms == []

    def test_phantom_citations_none_when_all_valid(self):
        from src.pipeline.llm_predictor import _compute_phantom_citations
        vocab = ["wt.t1.mean", "wt.enhancement_ratio"]
        phantoms = _compute_phantom_citations("wt.t1.mean", ["wt.enhancement_ratio"], [], vocab)
        assert phantoms == []

    def test_build_prompt_includes_feature_vocabulary(self):
        from src.prompts.templates import build_prompt
        vocab = ["wt.t1.mean", "wt.enhancement_ratio", "et.volume_mm3"]
        sys_p, usr_p = build_prompt("Some narrative.", feature_vocabulary=vocab)
        assert "wt.t1.mean" in usr_p
        assert "wt.enhancement_ratio" in usr_p
        assert "decisive_feature" in usr_p
        assert "supporting_features" in usr_p
        assert "findings" in usr_p
        assert "impression" in usr_p

    def test_build_prompt_no_vocab_uses_placeholder(self):
        from src.prompts.templates import build_prompt
        sys_p, usr_p = build_prompt("Some narrative.")
        assert "not provided" in usr_p.lower() or "feature list" in usr_p.lower()


# ---------------------------------------------------------------------------
# Change 2: Intervention layer
# ---------------------------------------------------------------------------

class TestInterventions:

    def test_build_vocab_map_keys(self):
        from src.pipeline.interventions import build_vocab_map
        features = make_simple_features()
        vmap = build_vocab_map(features)
        assert "wt.volume_mm3" in vmap
        assert "wt.t1.mean" in vmap
        assert "wt.enhancement_ratio" in vmap
        assert "et.t1ce.mean" in vmap
        assert "atlas.harvard_oxford.Frontal_Pole.t1.mean" in vmap

    def test_build_vocab_map_paths_correct(self):
        from src.pipeline.interventions import build_vocab_map
        features = make_simple_features()
        vmap = build_vocab_map(features)
        assert vmap["wt.volume_mm3"] == ["label_stats", "wt", "volume_mm3"]
        assert vmap["wt.t1.mean"] == ["label_stats", "wt", "t1", "mean"]
        assert vmap["atlas.harvard_oxford.Frontal_Pole.t1.mean"] == [
            "atlas_regions", "harvard_oxford", "Frontal Pole", "t1", "mean"
        ]

    def test_extract_feature_vocabulary_sorted(self):
        from src.pipeline.interventions import extract_feature_vocabulary
        features = make_simple_features()
        vocab = extract_feature_vocabulary(features)
        assert vocab == sorted(vocab)
        assert len(vocab) > 0
        assert "wt.volume_mm3" in vocab

    def test_null_by_path_returns_copy(self):
        from src.pipeline.interventions import null_by_path
        features = make_simple_features()
        modified = null_by_path(features, ["label_stats", "wt", "volume_mm3"])
        assert modified["label_stats"]["wt"]["volume_mm3"] is None
        # Original unchanged
        assert features["label_stats"]["wt"]["volume_mm3"] == 1500.0

    def test_null_by_path_leaf_stat(self):
        from src.pipeline.interventions import null_by_path
        features = make_simple_features()
        modified = null_by_path(features, ["label_stats", "wt", "t1", "mean"])
        assert modified["label_stats"]["wt"]["t1"]["mean"] is None
        assert modified["label_stats"]["wt"]["t1"]["std"] is not None

    def test_null_group_by_path_nulls_all_children(self):
        from src.pipeline.interventions import null_group_by_path
        features = make_simple_features()
        modified = null_group_by_path(features, ["label_stats", "wt", "t1"])
        for stat in ("mean", "std", "median", "p5", "p95", "skewness"):
            assert modified["label_stats"]["wt"]["t1"][stat] is None
        # Other modalities untouched
        assert modified["label_stats"]["wt"]["t1ce"]["mean"] is not None

    def test_keep_only_by_identifiers(self):
        from src.pipeline.interventions import build_vocab_map, keep_only_by_identifiers
        features = make_simple_features()
        vmap = build_vocab_map(features)
        keep = ["wt.volume_mm3", "wt.enhancement_ratio"]
        modified = keep_only_by_identifiers(features, keep, vmap)
        assert modified["label_stats"]["wt"]["volume_mm3"] == 1500.0
        assert modified["label_stats"]["wt"]["enhancement_ratio"] == 1.78
        # All other leaves should be None
        assert modified["label_stats"]["wt"]["t1"]["mean"] is None
        assert modified["label_stats"]["et"]["volume_mm3"] is None

    def test_remove_by_identifiers(self):
        from src.pipeline.interventions import build_vocab_map, remove_by_identifiers
        features = make_simple_features()
        vmap = build_vocab_map(features)
        modified = remove_by_identifiers(features, ["wt.volume_mm3", "et.t1ce.mean"], vmap)
        assert modified["label_stats"]["wt"]["volume_mm3"] is None
        assert modified["label_stats"]["et"]["t1ce"]["mean"] is None
        # Other values preserved
        assert modified["label_stats"]["wt"]["enhancement_ratio"] == 1.78
        assert modified["label_stats"]["et"]["volume_mm3"] == 500.0

    def test_substitute_by_identifier(self):
        from src.pipeline.interventions import build_vocab_map, substitute_by_identifier
        features = make_simple_features()
        vmap = build_vocab_map(features)
        modified = substitute_by_identifier(features, "wt.enhancement_ratio", 5.0, vmap)
        assert modified["label_stats"]["wt"]["enhancement_ratio"] == 5.0
        # Original unchanged
        assert features["label_stats"]["wt"]["enhancement_ratio"] == 1.78

    def test_substitute_unknown_identifier_returns_copy(self):
        from src.pipeline.interventions import build_vocab_map, substitute_by_identifier
        features = make_simple_features()
        vmap = build_vocab_map(features)
        modified = substitute_by_identifier(features, "nonexistent.feature", 99.9, vmap)
        # No modification, but returns a copy
        assert modified["label_stats"]["wt"]["volume_mm3"] == 1500.0

    def test_null_by_path_invalid_path_no_exception(self):
        from src.pipeline.interventions import null_by_path
        features = make_simple_features()
        # Should not raise
        modified = null_by_path(features, ["label_stats", "wt", "nonexistent_key"])
        assert modified is not features


# ---------------------------------------------------------------------------
# Mock predictor for faithfulness tests
# ---------------------------------------------------------------------------

class _MockPredictor:
    """
    Configurable mock predictor for faithfulness tests.

    predict_fn: callable(subject_id, narrative, features_dict) -> IDHPrediction
    raw_fn:     callable(messages) -> str
    """

    def __init__(self, predict_fn=None, raw_fn=None):
        self._predict_fn = predict_fn
        self._raw_fn = raw_fn or (lambda msgs: "{}")
        self.predict_calls: List = []
        self.raw_calls: List = []

    def predict(self, subject_id, narrative, ground_truth=None, features_dict=None,
                system_prompt=None, user_prompt_template=None):
        self.predict_calls.append(subject_id)
        if self._predict_fn:
            return self._predict_fn(subject_id, narrative, features_dict)
        return make_mutant_prediction(subject_id)

    def raw_query(self, messages):
        self.raw_calls.append(messages)
        return self._raw_fn(messages)


# ---------------------------------------------------------------------------
# Change 3: Faithfulness tests
# ---------------------------------------------------------------------------

class TestWilsonCI:

    def test_wilson_ci_perfect_score(self):
        from src.pipeline.faithfulness import _wilson_ci
        lo, hi = _wilson_ci(10, 10)
        assert lo > 0.6
        assert hi <= 1.0

    def test_wilson_ci_zero_score(self):
        from src.pipeline.faithfulness import _wilson_ci
        lo, hi = _wilson_ci(0, 10)
        assert lo >= 0.0
        assert hi < 0.4

    def test_wilson_ci_empty(self):
        from src.pipeline.faithfulness import _wilson_ci
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_wilson_ci_bounds(self):
        from src.pipeline.faithfulness import _wilson_ci
        for successes, n in [(3, 10), (7, 10), (1, 2), (5, 5)]:
            lo, hi = _wilson_ci(successes, n)
            assert 0.0 <= lo <= hi <= 1.0


class TestSufficiencyTest:

    def _make_cases_and_predictor(self, faithful_outcome: bool):
        """faithful_outcome=True → mock returns same status+conf as original."""
        features = make_simple_features()
        orig = make_mutant_prediction()

        def predict_fn(sid, narrative, fdict):
            if faithful_outcome:
                return make_mutant_prediction(sid)  # same status+conf
            else:
                from src.pipeline.llm_predictor import IDHPrediction
                return IDHPrediction(
                    subject_id=sid, model="mock", idh_status="IDH-wildtype",
                    confidence="medium", reasoning="R", raw_response="{}", latency_s=0.0,
                )

        from src.pipeline.faithfulness import FaithfulnessCase
        case = FaithfulnessCase(subject_id="S1", features=features,
                                original_prediction=orig, ground_truth=1)
        predictor = _MockPredictor(predict_fn=predict_fn)
        return [case], predictor

    def test_sufficiency_faithful_when_same_prediction(self):
        from src.pipeline.faithfulness import run_sufficiency_test
        cases, predictor = self._make_cases_and_predictor(faithful_outcome=True)

        def narrative_fn(sid, feat):
            return "narrative"

        result = run_sufficiency_test(predictor, cases, narrative_fn)
        assert result.cohort_score == 1.0
        assert result.per_case[0].faithful is True
        assert result.test_name == "sufficiency"

    def test_sufficiency_unfaithful_when_prediction_changes(self):
        from src.pipeline.faithfulness import run_sufficiency_test
        cases, predictor = self._make_cases_and_predictor(faithful_outcome=False)

        def narrative_fn(sid, feat):
            return "narrative"

        result = run_sufficiency_test(predictor, cases, narrative_fn)
        assert result.cohort_score == 0.0
        assert result.per_case[0].faithful is False

    def test_sufficiency_skips_case_with_no_cited_features(self):
        from src.pipeline.faithfulness import run_sufficiency_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction
        pred = IDHPrediction("S2", "mock", "IDH-mutant", "high", "r", "{}", 0.0)
        # No decisive/supporting/contradicting features
        case = FaithfulnessCase("S2", make_simple_features(), pred)
        result = run_sufficiency_test(_MockPredictor(), [case], lambda s, f: "narrative")
        assert result.per_case[0].details.get("skipped") == "no_cited_features"

    def test_sufficiency_ci_within_bounds(self):
        from src.pipeline.faithfulness import run_sufficiency_test
        cases, predictor = self._make_cases_and_predictor(faithful_outcome=True)
        result = run_sufficiency_test(predictor, cases, lambda s, f: "n")
        assert 0.0 <= result.ci_lower <= result.cohort_score <= result.ci_upper <= 1.0


class TestComprehensivenessTest:

    def test_comprehensive_faithful_when_prediction_degrades(self):
        from src.pipeline.faithfulness import run_comprehensiveness_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig, ground_truth=1)

        def predict_fn(sid, narrative, fdict):
            # Return opposite → prediction degraded → faithful
            return IDHPrediction(sid, "mock", "IDH-wildtype", "medium", "r", "{}", 0.0)

        result = run_comprehensiveness_test(
            _MockPredictor(predict_fn=predict_fn), [case], lambda s, f: "n"
        )
        assert result.cohort_score == 1.0
        assert result.per_case[0].faithful is True

    def test_comprehensive_unfaithful_when_no_change(self):
        from src.pipeline.faithfulness import run_comprehensiveness_test, FaithfulnessCase
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig, ground_truth=1)

        def predict_fn(sid, narrative, fdict):
            return make_mutant_prediction(sid)  # same → not degraded → unfaithful

        result = run_comprehensiveness_test(
            _MockPredictor(predict_fn=predict_fn), [case], lambda s, f: "n"
        )
        assert result.cohort_score == 0.0

    def test_comprehensive_confidence_drop_counts_as_degraded(self):
        from src.pipeline.faithfulness import run_comprehensiveness_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction
        orig = make_mutant_prediction()  # confidence="high"
        case = FaithfulnessCase("S1", make_simple_features(), orig)

        def predict_fn(sid, narrative, fdict):
            # Same label but lower confidence
            return IDHPrediction(sid, "mock", "IDH-mutant", "low", "r", "{}", 0.0)

        result = run_comprehensiveness_test(
            _MockPredictor(predict_fn=predict_fn), [case], lambda s, f: "n"
        )
        assert result.per_case[0].faithful is True


class TestCitationImportanceTest:

    def test_citation_importance_all_buckets_present(self):
        from src.pipeline.faithfulness import run_citation_importance_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction

        features = make_simple_features()
        # decisive = wt.enhancement_ratio  (high)
        # supporting = wt.flair.mean      (low)
        # everything else = none
        orig = make_mutant_prediction(
            decisive="wt.enhancement_ratio",
            supporting=["wt.flair.mean"],
            contradicting=[],
        )
        case = FaithfulnessCase("S1", features, orig, ground_truth=1)

        ablation_count = [0]

        def predict_fn(sid, narrative, fdict):
            ablation_count[0] += 1
            # For "wt.enhancement_ratio" ablation: flip prediction
            # For all others: keep same
            wt_enh = None
            try:
                wt_enh = fdict["label_stats"]["wt"]["enhancement_ratio"]
            except (KeyError, TypeError):
                pass
            if wt_enh is None:
                return IDHPrediction(sid, "mock", "IDH-wildtype", "medium", "r", "{}", 0.0)
            return make_mutant_prediction(sid)

        result = run_citation_importance_test(
            _MockPredictor(predict_fn=predict_fn), [case], lambda s, f: "n"
        )
        assert result.high_n >= 1
        assert result.none_n > 0
        assert ablation_count[0] > 0

    def test_citation_importance_high_cited_drops_more(self):
        """High-cited features should have equal or higher drop than none-cited."""
        from src.pipeline.faithfulness import run_citation_importance_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction

        features = make_simple_features()
        orig = make_mutant_prediction(decisive="wt.enhancement_ratio", supporting=[], contradicting=[])
        case = FaithfulnessCase("S1", features, orig, ground_truth=1)

        def predict_fn(sid, narrative, fdict):
            # Only flip on wt.enhancement_ratio ablation
            wt_enh = (fdict or {}).get("label_stats", {}).get("wt", {}).get("enhancement_ratio")
            if wt_enh is None:
                return IDHPrediction(sid, "mock", "IDH-wildtype", "low", "r", "{}", 0.0)
            return make_mutant_prediction(sid)

        result = run_citation_importance_test(
            _MockPredictor(predict_fn=predict_fn), [case], lambda s, f: "n"
        )
        assert result.high_avg_drop >= result.none_avg_drop

    def test_citation_importance_result_to_dict(self):
        from src.pipeline.faithfulness import run_citation_importance_test, FaithfulnessCase
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig)
        result = run_citation_importance_test(
            _MockPredictor(predict_fn=lambda s, n, f: make_mutant_prediction(s)),
            [case], lambda s, f: "n"
        )
        d = result.to_dict()
        assert d["test_name"] == "citation_importance_alignment"
        assert "none" in d and "low" in d and "high" in d
        for bucket in ("none", "low", "high"):
            assert "avg_drop" in d[bucket]
            assert "ci_95" in d[bucket]
            assert "n" in d[bucket]


class TestCounterfactualTest:

    def test_counterfactual_faithful_when_prediction_flips(self):
        from src.pipeline.faithfulness import run_counterfactual_test, FaithfulnessCase
        from src.pipeline.llm_predictor import IDHPrediction

        features = make_simple_features()
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", features, orig, ground_truth=1)

        def raw_fn(messages):
            return json.dumps({
                "feature_to_change": "wt.enhancement_ratio",
                "current_value": 1.78,
                "new_value": 5.0,
                "reasoning": "High enhancement suggests GBM.",
            })

        call_count = [0]

        def predict_fn(sid, narrative, fdict):
            call_count[0] += 1
            # After substituting wt.enhancement_ratio, predict wildtype
            wt_enh = (fdict or {}).get("label_stats", {}).get("wt", {}).get("enhancement_ratio")
            if wt_enh is not None and wt_enh > 3.0:
                return IDHPrediction(sid, "mock", "IDH-wildtype", "high", "r", "{}", 0.0)
            return make_mutant_prediction(sid)

        result = run_counterfactual_test(
            _MockPredictor(predict_fn=predict_fn, raw_fn=raw_fn),
            [case], lambda s, f: "n"
        )
        assert result.per_case[0].faithful is True
        assert result.cohort_score == 1.0

    def test_counterfactual_unfaithful_when_flip_does_not_occur(self):
        from src.pipeline.faithfulness import run_counterfactual_test, FaithfulnessCase

        features = make_simple_features()
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", features, orig)

        def raw_fn(messages):
            return json.dumps({
                "feature_to_change": "wt.enhancement_ratio",
                "current_value": 1.78,
                "new_value": 5.0,
                "reasoning": "High enhancement.",
            })

        result = run_counterfactual_test(
            _MockPredictor(
                predict_fn=lambda s, n, f: make_mutant_prediction(s),  # never flips
                raw_fn=raw_fn,
            ),
            [case], lambda s, f: "n"
        )
        assert result.per_case[0].faithful is False

    def test_counterfactual_skips_unparseable_response(self):
        from src.pipeline.faithfulness import run_counterfactual_test, FaithfulnessCase

        features = make_simple_features()
        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", features, orig)

        result = run_counterfactual_test(
            _MockPredictor(raw_fn=lambda msgs: "this is not json"),
            [case], lambda s, f: "n"
        )
        assert result.per_case[0].details.get("skipped") == "unparseable_counterfactual"


class TestRationaleCorruptionTest:

    def test_rationale_corruption_faithful_when_label_follows_corrupt_rationale(self):
        from src.pipeline.faithfulness import run_rationale_corruption_test, FaithfulnessCase

        features = make_simple_features()
        orig = make_mutant_prediction()  # IDH-mutant
        case = FaithfulnessCase("S1", features, orig, ground_truth=1)

        raw_call = [0]

        def raw_fn(messages):
            raw_call[0] += 1
            if raw_call[0] == 1:
                # First call: generate opposite (IDH-wildtype) rationale
                return json.dumps({
                    "findings": "Ring-enhancing lesion with necrosis.",
                    "impression": "IDH-wildtype GBM.",
                })
            else:
                # Second call: label from rationale → follows opposite class
                return json.dumps({
                    "idh_status": "IDH-wildtype",
                    "confidence": "high",
                })

        result = run_rationale_corruption_test(
            _MockPredictor(raw_fn=raw_fn), [case], lambda s, f: "n"
        )
        assert result.per_case[0].faithful is True
        assert result.cohort_score == 1.0

    def test_rationale_corruption_unfaithful_when_label_ignores_rationale(self):
        from src.pipeline.faithfulness import run_rationale_corruption_test, FaithfulnessCase

        features = make_simple_features()
        orig = make_mutant_prediction()  # IDH-mutant
        case = FaithfulnessCase("S1", features, orig)

        raw_call = [0]

        def raw_fn(messages):
            raw_call[0] += 1
            if raw_call[0] == 1:
                return json.dumps({
                    "findings": "Ring-enhancing lesion.",
                    "impression": "IDH-wildtype GBM.",
                })
            else:
                # Label ignores opposite rationale, still says mutant
                return json.dumps({
                    "idh_status": "IDH-mutant",
                    "confidence": "medium",
                })

        result = run_rationale_corruption_test(
            _MockPredictor(raw_fn=raw_fn), [case], lambda s, f: "n"
        )
        assert result.per_case[0].faithful is False

    def test_rationale_corruption_skips_unparseable_rationale(self):
        from src.pipeline.faithfulness import run_rationale_corruption_test, FaithfulnessCase

        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig)

        result = run_rationale_corruption_test(
            _MockPredictor(raw_fn=lambda msgs: "not json at all"),
            [case], lambda s, f: "n"
        )
        assert result.per_case[0].details.get("skipped") == "unparseable_opposite_rationale"

    def test_rationale_corruption_result_structure(self):
        from src.pipeline.faithfulness import run_rationale_corruption_test, FaithfulnessCase

        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig)

        call_n = [0]

        def raw_fn(msgs):
            call_n[0] += 1
            if call_n[0] == 1:
                return json.dumps({"findings": "findings", "impression": "impression"})
            return json.dumps({"idh_status": "IDH-wildtype", "confidence": "high"})

        result = run_rationale_corruption_test(
            _MockPredictor(raw_fn=raw_fn), [case], lambda s, f: "n"
        )
        assert result.test_name == "rationale_corruption"
        assert len(result.per_case) == 1
        d = result.to_dict()
        assert "cohort_score" in d
        assert "ci_95" in d
        assert "n_cases" in d


class TestFaithfulnessOrchestrator:

    def test_run_all_faithfulness_tests_returns_all_keys(self):
        from src.pipeline.faithfulness import run_all_faithfulness_tests, FaithfulnessCase

        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig, ground_truth=1)

        call_n = [0]

        def raw_fn(msgs):
            call_n[0] += 1
            if call_n[0] % 2 == 1:
                return json.dumps({
                    "findings": "Frontal location.", "impression": "IDH-mutant.",
                    "feature_to_change": "wt.enhancement_ratio",
                    "current_value": 1.78, "new_value": 5.0, "reasoning": "More enhancement.",
                })
            return json.dumps({"idh_status": "IDH-wildtype", "confidence": "medium"})

        results = run_all_faithfulness_tests(
            _MockPredictor(
                predict_fn=lambda s, n, f: make_mutant_prediction(s),
                raw_fn=raw_fn,
            ),
            [case],
            narrative_fn=lambda s, f: "n",
            run_citation_importance=False,  # Skip for speed in this test
        )
        assert "sufficiency" in results
        assert "comprehensiveness" in results
        assert "counterfactual" in results
        assert "rationale_corruption" in results

    def test_save_faithfulness_results(self, tmp_path):
        from src.pipeline.faithfulness import (
            run_all_faithfulness_tests, FaithfulnessCase, save_faithfulness_results
        )

        orig = make_mutant_prediction()
        case = FaithfulnessCase("S1", make_simple_features(), orig, ground_truth=1)

        call_n = [0]

        def raw_fn(msgs):
            call_n[0] += 1
            if call_n[0] % 2 == 1:
                return json.dumps({"findings": "F", "impression": "I",
                                   "feature_to_change": "wt.enhancement_ratio",
                                   "current_value": 1.0, "new_value": 5.0, "reasoning": "r"})
            return json.dumps({"idh_status": "IDH-wildtype", "confidence": "medium"})

        results = run_all_faithfulness_tests(
            _MockPredictor(
                predict_fn=lambda s, n, f: make_mutant_prediction(s),
                raw_fn=raw_fn,
            ),
            [case],
            narrative_fn=lambda s, f: "n",
            run_citation_importance=False,
        )
        out_path = tmp_path / "faithfulness.json"
        save_faithfulness_results(results, out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "sufficiency" in data
        assert "cohort_score" in data["sufficiency"]


# ---------------------------------------------------------------------------
# Change 4: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_llm_predictor_has_seed(self):
        from src.pipeline.llm_predictor import LLMPredictor
        p = LLMPredictor(model="gpt-4o", seed=42)
        assert p.seed == 42

    def test_llm_predictor_seed_default_is_42(self):
        from src.pipeline.llm_predictor import LLMPredictor
        p = LLMPredictor(model="gpt-4o")
        assert p.seed == 42

    def test_llm_predictor_seed_none_allowed(self):
        from src.pipeline.llm_predictor import LLMPredictor
        p = LLMPredictor(model="gpt-4o", seed=None)
        assert p.seed is None

    def test_idh_prediction_stores_model_version(self):
        from src.pipeline.llm_predictor import IDHPrediction
        p = IDHPrediction(
            "s1", "gpt-4o", "IDH-mutant", "high", "r", "{}", 0.0,
            model_version="gpt-4o-2024-11-20",
        )
        assert p.model_version == "gpt-4o-2024-11-20"

    def test_parse_response_deterministic_same_input(self):
        from src.pipeline.llm_predictor import _parse_response
        text = json.dumps({
            "idh_status": "IDH-mutant", "confidence": "high",
            "reasoning": "Frontal location.", "decisive_feature": "wt.t1.mean",
            "supporting_features": [], "contradicting_features": [],
            "findings": "F", "impression": "I",
        })
        r1 = _parse_response(text)
        r2 = _parse_response(text)
        assert r1 == r2

    def test_feature_vocabulary_deterministic(self):
        from src.pipeline.interventions import extract_feature_vocabulary
        features = make_simple_features()
        v1 = extract_feature_vocabulary(features)
        v2 = extract_feature_vocabulary(features)
        assert v1 == v2


# ---------------------------------------------------------------------------
# Integration: full feature pipeline with vocabulary extraction
# ---------------------------------------------------------------------------

class TestVocabularyFromRealFeatures:

    def test_vocabulary_from_synthetic_features(self):
        from src.data.synthetic import generate_dataset
        from src.pipeline.feature_extraction import extract_all_features
        from src.pipeline.interventions import extract_feature_vocabulary

        subjects = generate_dataset(n_mutant=1, n_wildtype=1, seed=0)
        for s in subjects:
            features = extract_all_features(
                s["images"], s["seg"], atlases={}, voxel_size_mm=s["voxel_size"]
            )
            vocab = extract_feature_vocabulary(features)
            assert len(vocab) > 0
            assert vocab == sorted(vocab)
            # Check expected identifiers
            assert "wt.volume_mm3" in vocab
            assert "wt.t1.mean" in vocab
            # enhancement_ratio is stored globally in actual features
            has_enh = any("enhancement" in v for v in vocab)
            assert has_enh, f"No enhancement feature found in vocab: {vocab[:10]}"

    def test_vocab_map_paths_are_valid(self):
        from src.data.synthetic import generate_dataset
        from src.pipeline.feature_extraction import extract_all_features
        from src.pipeline.interventions import build_vocab_map, _get_by_path

        subjects = generate_dataset(n_mutant=1, n_wildtype=0, seed=42)
        s = subjects[0]
        features = extract_all_features(
            s["images"], s["seg"], atlases={}, voxel_size_mm=s["voxel_size"]
        )
        vmap = build_vocab_map(features)
        for ident, path in vmap.items():
            # Every path should resolve to a non-dict (leaf) value
            val = _get_by_path(features, path)
            assert not isinstance(val, dict), \
                f"{ident} path resolves to a dict, not a leaf"

    def test_keep_only_restores_correct_values(self):
        from src.data.synthetic import generate_dataset
        from src.pipeline.feature_extraction import extract_all_features
        from src.pipeline.interventions import build_vocab_map, keep_only_by_identifiers, _get_by_path

        subjects = generate_dataset(n_mutant=1, n_wildtype=0, seed=0)
        s = subjects[0]
        features = extract_all_features(
            s["images"], s["seg"], atlases={}, voxel_size_mm=s["voxel_size"]
        )
        vmap = build_vocab_map(features)
        keep_ids = ["wt.volume_mm3", "wt.t1.mean"]
        modified = keep_only_by_identifiers(features, keep_ids, vmap)

        for ident in keep_ids:
            assert ident in vmap, f"{ident} not in vocab map"
            orig_val = _get_by_path(features, vmap[ident])
            mod_val  = _get_by_path(modified, vmap[ident])
            assert orig_val == mod_val, f"Value for {ident} should be preserved"

        # Other values should be None
        for ident, path in vmap.items():
            if ident not in keep_ids:
                mod_val = _get_by_path(modified, path)
                assert mod_val is None, f"Value for {ident} should be None"
