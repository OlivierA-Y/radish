"""
Evaluation metrics: accuracy, sensitivity, specificity, F1, AUC-ROC.
Matches the paper's reported evaluation protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    n_total: int
    n_mutant: int
    n_wildtype: int
    accuracy: float
    sensitivity: float      # recall for IDH-mutant (positive class)
    specificity: float      # recall for IDH-wildtype
    ppv: float              # precision for IDH-mutant
    npv: float              # precision for IDH-wildtype
    f1: float
    balanced_accuracy: float
    auc_roc: Optional[float]
    model: str = ""
    dataset: str = ""

    def summary(self) -> str:
        lines = [
            f"Model: {self.model}  |  Dataset: {self.dataset}",
            f"  N={self.n_total} (mutant={self.n_mutant}, wildtype={self.n_wildtype})",
            f"  Accuracy:          {self.accuracy:.3f}",
            f"  Sensitivity (TPR): {self.sensitivity:.3f}",
            f"  Specificity (TNR): {self.specificity:.3f}",
            f"  PPV:               {self.ppv:.3f}",
            f"  NPV:               {self.npv:.3f}",
            f"  F1-score:          {self.f1:.3f}",
            f"  Balanced accuracy: {self.balanced_accuracy:.3f}",
        ]
        if self.auc_roc is not None:
            lines.append(f"  AUC-ROC:           {self.auc_roc:.3f}")
        return "\n".join(lines)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
    model: str = "",
    dataset: str = "",
) -> EvalMetrics:
    """
    y_true, y_pred: 0 = IDH-wildtype, 1 = IDH-mutant
    y_score: continuous confidence score for AUC computation (optional)
    """
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)

    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    accuracy    = (tp + tn) / max(len(yt), 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)
    f1          = 2 * tp / max(2 * tp + fp + fn, 1)
    bal_acc     = (sensitivity + specificity) / 2

    auc = None
    if y_score is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(yt, y_score))
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)

    return EvalMetrics(
        n_total=len(yt),
        n_mutant=int((yt == 1).sum()),
        n_wildtype=int((yt == 0).sum()),
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        ppv=float(ppv),
        npv=float(npv),
        f1=float(f1),
        balanced_accuracy=float(bal_acc),
        auc_roc=auc,
        model=model,
        dataset=dataset,
    )


def evaluate_predictions(predictions, model: str = "", dataset: str = "") -> EvalMetrics:
    """Compute metrics from a list of IDHPrediction objects."""
    labelled = [p for p in predictions if p.ground_truth is not None]
    if not labelled:
        raise ValueError("No predictions have ground-truth labels")

    y_true  = [p.ground_truth for p in labelled]
    y_pred  = [p.label for p in labelled]
    y_score = [p.confidence_score for p in labelled]

    return compute_metrics(y_true, y_pred, y_score, model=model, dataset=dataset)


def save_metrics(metrics: EvalMetrics, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def save_predictions_csv(predictions, path: str | Path) -> None:
    import csv
    fieldnames = [
        "subject_id", "model", "idh_status", "confidence",
        "label", "ground_truth", "correct", "latency_s", "reasoning",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "subject_id":   p.subject_id,
                "model":        p.model,
                "idh_status":   p.idh_status,
                "confidence":   p.confidence,
                "label":        p.label,
                "ground_truth": p.ground_truth,
                "correct":      p.correct,
                "latency_s":    round(p.latency_s, 3),
                "reasoning":    p.reasoning[:200],
            })


def save_reasoning_traces(predictions, path: str | Path) -> None:
    """
    Write full, untruncated reasoning strings to a plain-text file.

    Format (one entry per subject):
        SUBJECT_ID: "full reasoning text…"
    """
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            # Escape any embedded double-quotes so the format stays parseable
            escaped = p.reasoning.replace('"', '\\"')
            f.write(f'{p.subject_id}: "{escaped}"\n')
