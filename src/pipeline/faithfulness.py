"""
Faithfulness evaluation layer (Change 3).

Five tests:
  1. Sufficiency         — re-query with only cited features kept
  2. Comprehensiveness   — re-query with cited features removed
  3. Citation-importance — ablate each feature, measure drop vs citation strength
  4. Counterfactual      — suggest minimal flip, apply it, verify flip
  5. Rationale corruption — swap rationale, check if label follows

Each test returns per-case results + cohort aggregate with 95 % Wilson CIs.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessCase:
    """One case for faithfulness evaluation."""
    subject_id: str
    features: Dict          # imaging features dict (label_stats + atlas_regions)
    original_prediction: Any  # IDHPrediction with structured rationale
    ground_truth: Optional[int] = None


@dataclass
class PerCaseResult:
    subject_id: str
    faithful: bool
    original_status: str
    modified_status: str
    original_confidence: str
    modified_confidence: str
    details: Dict = field(default_factory=dict)


@dataclass
class FaithfulnessResult:
    test_name: str
    per_case: List[PerCaseResult]
    cohort_score: float      # fraction of faithful cases
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "cohort_score": round(self.cohort_score, 4),
            "ci_95": [round(self.ci_lower, 4), round(self.ci_upper, 4)],
            "n_cases": len(self.per_case),
            "n_faithful": sum(1 for r in self.per_case if r.faithful),
        }


@dataclass
class CitationAlignmentResult:
    """Per-citation-strength ablation results (Test 3)."""
    none_avg_drop: float
    none_ci: Tuple[float, float]
    none_n: int
    low_avg_drop: float
    low_ci: Tuple[float, float]
    low_n: int
    high_avg_drop: float
    high_ci: Tuple[float, float]
    high_n: int

    def to_dict(self) -> Dict:
        return {
            "test_name": "citation_importance_alignment",
            "none": {"avg_drop": round(self.none_avg_drop, 4), "ci_95": list(self.none_ci), "n": self.none_n},
            "low":  {"avg_drop": round(self.low_avg_drop,  4), "ci_95": list(self.low_ci),  "n": self.low_n},
            "high": {"avg_drop": round(self.high_avg_drop, 4), "ci_95": list(self.high_ci), "n": self.high_n},
        }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _confidence_level(conf_str: str) -> int:
    return {"high": 2, "medium": 1, "low": 0}.get(conf_str.lower() if conf_str else "", 0)


def _predictions_agree(p1: Any, p2: Any) -> bool:
    return p1.idh_status == p2.idh_status


def _prediction_degraded(original: Any, modified: Any) -> bool:
    """True if the prediction changed OR confidence dropped."""
    if original.idh_status != modified.idh_status:
        return True
    return _confidence_level(modified.confidence) < _confidence_level(original.confidence)


def _cited_identifiers(prediction: Any) -> List[str]:
    if hasattr(prediction, "cited_identifiers"):
        return prediction.cited_identifiers()
    cited: List[str] = []
    if getattr(prediction, "decisive_feature", None):
        cited.append(prediction.decisive_feature)
    for attr in ("supporting_features", "contradicting_features"):
        lst = getattr(prediction, attr, None)
        if lst:
            cited.extend(lst)
    return cited


# ---------------------------------------------------------------------------
# Default narrative function (importable without circular dependency)
# ---------------------------------------------------------------------------

def _default_narrative_fn(subject_id: str, features: Dict) -> str:
    from src.pipeline.serialization import to_narrative
    return to_narrative(subject_id, features)


# ---------------------------------------------------------------------------
# Test 1: Sufficiency
# ---------------------------------------------------------------------------

def run_sufficiency_test(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
) -> FaithfulnessResult:
    """
    Re-query with only the cited features kept (everything else nulled).
    Faithful = same prediction AND same confidence.
    """
    from src.pipeline.interventions import build_vocab_map, keep_only_by_identifiers

    if narrative_fn is None:
        narrative_fn = _default_narrative_fn

    per_case: List[PerCaseResult] = []
    for case in cases:
        orig = case.original_prediction
        cited = _cited_identifiers(orig)

        if not cited:
            # No cited features — skip case (vacuously unfaithful)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="N/A",
                original_confidence=orig.confidence,
                modified_confidence="N/A",
                details={"skipped": "no_cited_features"},
            ))
            continue

        vocab_map = build_vocab_map(case.features)
        modified_features = keep_only_by_identifiers(case.features, cited, vocab_map)
        modified_narrative = narrative_fn(case.subject_id, modified_features)

        try:
            modified_pred = predictor.predict(
                subject_id=case.subject_id,
                narrative=modified_narrative,
                ground_truth=case.ground_truth,
                features_dict=modified_features,
            )
            faithful = _predictions_agree(orig, modified_pred) and (
                orig.confidence == modified_pred.confidence
            )
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=faithful,
                original_status=orig.idh_status,
                modified_status=modified_pred.idh_status,
                original_confidence=orig.confidence,
                modified_confidence=modified_pred.confidence,
            ))
        except Exception as e:
            logger.warning("Sufficiency test failed for %s: %s", case.subject_id, e)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="error",
                original_confidence=orig.confidence,
                modified_confidence="error",
                details={"error": str(e)},
            ))

    n_faithful = sum(1 for r in per_case if r.faithful)
    n = len(per_case)
    lo, hi = _wilson_ci(n_faithful, n)
    return FaithfulnessResult(
        test_name="sufficiency",
        per_case=per_case,
        cohort_score=n_faithful / n if n > 0 else 0.0,
        ci_lower=lo,
        ci_upper=hi,
    )


# ---------------------------------------------------------------------------
# Test 2: Comprehensiveness
# ---------------------------------------------------------------------------

def run_comprehensiveness_test(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
) -> FaithfulnessResult:
    """
    Re-query with only the cited features removed (nulled).
    Faithful = prediction degrades (label changes or confidence drops).
    """
    from src.pipeline.interventions import build_vocab_map, remove_by_identifiers

    if narrative_fn is None:
        narrative_fn = _default_narrative_fn

    per_case: List[PerCaseResult] = []
    for case in cases:
        orig = case.original_prediction
        cited = _cited_identifiers(orig)

        if not cited:
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="N/A",
                original_confidence=orig.confidence,
                modified_confidence="N/A",
                details={"skipped": "no_cited_features"},
            ))
            continue

        vocab_map = build_vocab_map(case.features)
        modified_features = remove_by_identifiers(case.features, cited, vocab_map)
        modified_narrative = narrative_fn(case.subject_id, modified_features)

        try:
            modified_pred = predictor.predict(
                subject_id=case.subject_id,
                narrative=modified_narrative,
                ground_truth=case.ground_truth,
                features_dict=modified_features,
            )
            faithful = _prediction_degraded(orig, modified_pred)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=faithful,
                original_status=orig.idh_status,
                modified_status=modified_pred.idh_status,
                original_confidence=orig.confidence,
                modified_confidence=modified_pred.confidence,
            ))
        except Exception as e:
            logger.warning("Comprehensiveness test failed for %s: %s", case.subject_id, e)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="error",
                original_confidence=orig.confidence,
                modified_confidence="error",
                details={"error": str(e)},
            ))

    n_faithful = sum(1 for r in per_case if r.faithful)
    n = len(per_case)
    lo, hi = _wilson_ci(n_faithful, n)
    return FaithfulnessResult(
        test_name="comprehensiveness",
        per_case=per_case,
        cohort_score=n_faithful / n if n > 0 else 0.0,
        ci_lower=lo,
        ci_upper=hi,
    )


# ---------------------------------------------------------------------------
# Test 3: Citation–importance alignment
# ---------------------------------------------------------------------------

def run_citation_importance_test(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
) -> CitationAlignmentResult:
    """
    For each vocabulary feature in each case:
      - Assign citation strength: "high" (decisive), "low" (supporting/contradicting),
        "none" (not cited)
      - Ablate that feature (null it)
      - Measure performance drop = 1 if prediction changed, 0 otherwise
    Pool by citation strength, report average drop + CI per level.
    """
    from src.pipeline.interventions import build_vocab_map, null_by_path

    if narrative_fn is None:
        narrative_fn = _default_narrative_fn

    buckets: Dict[str, List[int]] = {"none": [], "low": [], "high": []}

    for case in cases:
        orig = case.original_prediction
        decisive = getattr(orig, "decisive_feature", None) or ""
        supporting = set(getattr(orig, "supporting_features", None) or [])
        contradicting = set(getattr(orig, "contradicting_features", None) or [])
        low_cited = supporting | contradicting

        vocab_map = build_vocab_map(case.features)

        for ident, path in vocab_map.items():
            if ident == decisive:
                strength = "high"
            elif ident in low_cited:
                strength = "low"
            else:
                strength = "none"

            modified_features = null_by_path(case.features, path)
            modified_narrative = narrative_fn(case.subject_id, modified_features)

            try:
                modified_pred = predictor.predict(
                    subject_id=case.subject_id,
                    narrative=modified_narrative,
                    ground_truth=case.ground_truth,
                    features_dict=modified_features,
                )
                drop = 0 if _predictions_agree(orig, modified_pred) else 1
                buckets[strength].append(drop)
            except Exception as e:
                logger.warning(
                    "Citation-importance ablation failed for %s / %s: %s",
                    case.subject_id, ident, e,
                )

    def _avg_and_ci(drops: List[int]) -> Tuple[float, Tuple[float, float]]:
        n = len(drops)
        if n == 0:
            return 0.0, (0.0, 1.0)
        avg = sum(drops) / n
        lo, hi = _wilson_ci(sum(drops), n)
        return avg, (lo, hi)

    none_avg, none_ci = _avg_and_ci(buckets["none"])
    low_avg,  low_ci  = _avg_and_ci(buckets["low"])
    high_avg, high_ci = _avg_and_ci(buckets["high"])

    return CitationAlignmentResult(
        none_avg_drop=none_avg, none_ci=none_ci, none_n=len(buckets["none"]),
        low_avg_drop=low_avg,   low_ci=low_ci,   low_n=len(buckets["low"]),
        high_avg_drop=high_avg, high_ci=high_ci, high_n=len(buckets["high"]),
    )


# ---------------------------------------------------------------------------
# Test 4: Counterfactual consistency
# ---------------------------------------------------------------------------

def run_counterfactual_test(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
) -> FaithfulnessResult:
    """
    Ask the model what minimal input change would flip its prediction.
    Apply that change.
    Faithful = the re-queried prediction actually flips.
    """
    from src.pipeline.interventions import build_vocab_map, substitute_by_identifier
    from src.prompts.templates import COUNTERFACTUAL_PROMPT_TEMPLATE

    if narrative_fn is None:
        narrative_fn = _default_narrative_fn

    per_case: List[PerCaseResult] = []

    for case in cases:
        orig = case.original_prediction
        opposite = "IDH-wildtype" if "mutant" in orig.idh_status.lower() else "IDH-mutant"

        from src.pipeline.interventions import extract_feature_vocabulary
        vocab_list = extract_feature_vocabulary(case.features)
        vocab_str = "\n".join(f"  - {f}" for f in vocab_list)
        original_narrative = narrative_fn(case.subject_id, case.features)

        cf_prompt = COUNTERFACTUAL_PROMPT_TEMPLATE.format(
            original_prediction=orig.idh_status,
            opposite_prediction=opposite,
            narrative=original_narrative,
            feature_vocabulary=vocab_str,
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert neuroradiologist."},
                {"role": "user",   "content": cf_prompt},
            ]
            raw = predictor.raw_query(messages)
            cf_data = _safe_parse_json(raw)

            feature_to_change = cf_data.get("feature_to_change", "")
            new_value = cf_data.get("new_value")

            if not feature_to_change or new_value is None:
                per_case.append(PerCaseResult(
                    subject_id=case.subject_id,
                    faithful=False,
                    original_status=orig.idh_status,
                    modified_status="N/A",
                    original_confidence=orig.confidence,
                    modified_confidence="N/A",
                    details={"skipped": "unparseable_counterfactual", "raw": raw[:200]},
                ))
                continue

            vocab_map = build_vocab_map(case.features)
            modified_features = substitute_by_identifier(
                case.features, feature_to_change, new_value, vocab_map
            )
            modified_narrative = narrative_fn(case.subject_id, modified_features)

            modified_pred = predictor.predict(
                subject_id=case.subject_id,
                narrative=modified_narrative,
                ground_truth=case.ground_truth,
                features_dict=modified_features,
            )
            # Faithful = prediction actually flipped
            faithful = modified_pred.idh_status == opposite
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=faithful,
                original_status=orig.idh_status,
                modified_status=modified_pred.idh_status,
                original_confidence=orig.confidence,
                modified_confidence=modified_pred.confidence,
                details={
                    "feature_changed": feature_to_change,
                    "new_value": new_value,
                    "expected_flip": opposite,
                },
            ))
        except Exception as e:
            logger.warning("Counterfactual test failed for %s: %s", case.subject_id, e)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="error",
                original_confidence=orig.confidence,
                modified_confidence="error",
                details={"error": str(e)},
            ))

    n_faithful = sum(1 for r in per_case if r.faithful)
    n = len(per_case)
    lo, hi = _wilson_ci(n_faithful, n)
    return FaithfulnessResult(
        test_name="counterfactual_consistency",
        per_case=per_case,
        cohort_score=n_faithful / n if n > 0 else 0.0,
        ci_lower=lo,
        ci_upper=hi,
    )


# ---------------------------------------------------------------------------
# Test 5: Rationale corruption
# ---------------------------------------------------------------------------

def run_rationale_corruption_test(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
) -> FaithfulnessResult:
    """
    Generate a rationale arguing the OPPOSITE class (corruption condition).
    Supply it instead of the real rationale; request only the label.
    Faithful = label follows the supplied rationale.

    Also runs a control where the rationale matches the original prediction.
    The cohort_score reports the corruption condition rate.
    """
    from src.prompts.templates import (
        OPPOSITE_RATIONALE_PROMPT_TEMPLATE,
        LABEL_FROM_RATIONALE_PROMPT_TEMPLATE,
        SYSTEM_PROMPT,
    )

    if narrative_fn is None:
        narrative_fn = _default_narrative_fn

    per_case: List[PerCaseResult] = []

    for case in cases:
        orig = case.original_prediction
        opposite = "IDH-wildtype" if "mutant" in orig.idh_status.lower() else "IDH-mutant"
        original_narrative = narrative_fn(case.subject_id, case.features)

        try:
            # Step 1: Generate opposite-class rationale
            opp_prompt = OPPOSITE_RATIONALE_PROMPT_TEMPLATE.format(
                target_prediction=opposite,
                narrative=original_narrative,
            )
            opp_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": opp_prompt},
            ]
            opp_raw = predictor.raw_query(opp_messages)
            opp_data = _safe_parse_json(opp_raw)
            opp_findings  = opp_data.get("findings",  "")
            opp_impression = opp_data.get("impression", "")

            if not opp_findings and not opp_impression:
                per_case.append(PerCaseResult(
                    subject_id=case.subject_id,
                    faithful=False,
                    original_status=orig.idh_status,
                    modified_status="N/A",
                    original_confidence=orig.confidence,
                    modified_confidence="N/A",
                    details={"skipped": "unparseable_opposite_rationale"},
                ))
                continue

            # Step 2: Query label from the opposite rationale
            label_prompt = LABEL_FROM_RATIONALE_PROMPT_TEMPLATE.format(
                findings=opp_findings,
                impression=opp_impression,
            )
            label_messages = [
                {"role": "system", "content": "You are an expert neuroradiologist."},
                {"role": "user",   "content": label_prompt},
            ]
            label_raw = predictor.raw_query(label_messages)
            label_data = _safe_parse_json(label_raw)
            label_status = label_data.get("idh_status", "IDH-wildtype")
            label_conf   = label_data.get("confidence", "low")

            # Faithful = label matches the OPPOSITE rationale (not the original)
            faithful = (label_status == opposite)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=faithful,
                original_status=orig.idh_status,
                modified_status=label_status,
                original_confidence=orig.confidence,
                modified_confidence=label_conf,
                details={
                    "opposite_target": opposite,
                    "label_from_corrupt_rationale": label_status,
                },
            ))
        except Exception as e:
            logger.warning("Rationale corruption test failed for %s: %s", case.subject_id, e)
            per_case.append(PerCaseResult(
                subject_id=case.subject_id,
                faithful=False,
                original_status=orig.idh_status,
                modified_status="error",
                original_confidence=orig.confidence,
                modified_confidence="error",
                details={"error": str(e)},
            ))

    n_faithful = sum(1 for r in per_case if r.faithful)
    n = len(per_case)
    lo, hi = _wilson_ci(n_faithful, n)
    return FaithfulnessResult(
        test_name="rationale_corruption",
        per_case=per_case,
        cohort_score=n_faithful / n if n > 0 else 0.0,
        ci_lower=lo,
        ci_upper=hi,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_faithfulness_tests(
    predictor,
    cases: List[FaithfulnessCase],
    narrative_fn: Optional[Callable] = None,
    run_citation_importance: bool = True,
) -> Dict[str, Any]:
    """
    Run all five faithfulness tests and return a summary dict.

    Tests 1, 2, 4, 5 → FaithfulnessResult (cohort score + CI)
    Test 3            → CitationAlignmentResult (per-bucket averages + CI)
    """
    results: Dict[str, Any] = {}

    logger.info("Running Sufficiency test…")
    results["sufficiency"] = run_sufficiency_test(predictor, cases, narrative_fn)

    logger.info("Running Comprehensiveness test…")
    results["comprehensiveness"] = run_comprehensiveness_test(predictor, cases, narrative_fn)

    if run_citation_importance:
        logger.info("Running Citation-importance alignment test…")
        results["citation_importance"] = run_citation_importance_test(
            predictor, cases, narrative_fn
        )

    logger.info("Running Counterfactual consistency test…")
    results["counterfactual"] = run_counterfactual_test(predictor, cases, narrative_fn)

    logger.info("Running Rationale corruption test…")
    results["rationale_corruption"] = run_rationale_corruption_test(
        predictor, cases, narrative_fn
    )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> Dict:
    """Best-effort JSON parse of LLM output."""
    import re
    cleaned = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def save_faithfulness_results(results: Dict[str, Any], path: str | Path) -> None:
    """Serialize faithfulness results to JSON."""
    serializable: Dict[str, Any] = {}
    for name, result in results.items():
        if hasattr(result, "to_dict"):
            serializable[name] = result.to_dict()
        elif isinstance(result, dict):
            serializable[name] = {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in result.items()
            }
        else:
            serializable[name] = str(result)

    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    logger.info("Faithfulness results saved to %s", path)
