"""
evaluation.py
-------------
Model evaluation utilities.

When labelled data (is_fraud column) is available:
    - Precision, Recall, F1-score
    - Confusion matrix
    - False positive / false negative analysis
    - Comparison between IsolationForest and statistical baseline

When labels are unavailable:
    - Anomaly score distribution summary
    - Contamination rate report
    - Clear explanation of limitations

Public API
----------
evaluate_with_labels(df)     -> EvaluationResult
evaluate_without_labels(df)  -> EvaluationResult
auto_evaluate(df)            -> EvaluationResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Structured evaluation output."""
    has_labels: bool = False

    # Supervised metrics (IF model)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion: Optional[np.ndarray] = None
    classification_report_str: Optional[str] = None
    false_positives: int = 0
    false_negatives: int = 0

    # Supervised metrics (statistical baseline)
    stat_precision: Optional[float] = None
    stat_recall: Optional[float] = None
    stat_f1: Optional[float] = None

    # Unsupervised summary
    score_mean: Optional[float] = None
    score_std: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    flagged_count: int = 0
    total_count: int = 0
    flagged_pct: float = 0.0

    # Human-readable summary
    summary_lines: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(self.summary_lines)


# ---------------------------------------------------------------------------
# Supervised evaluation
# ---------------------------------------------------------------------------

def evaluate_with_labels(df: pd.DataFrame) -> EvaluationResult:
    """
    Compute precision, recall, F1, and confusion matrix using
    the ground-truth 'is_fraud' column.

    The DataFrame must contain:
        is_fraud      : int  0/1
        is_anomaly    : bool  (Isolation Forest prediction)
        stat_anomaly  : bool  (statistical baseline prediction)
    """
    result = EvaluationResult(has_labels=True)

    if "is_fraud" not in df.columns:
        result.summary_lines.append(
            "❌ Cannot evaluate: 'is_fraud' column not found in data."
        )
        return result

    required = {"is_anomaly", "is_fraud"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        result.summary_lines.append(f"❌ Missing columns for evaluation: {missing}")
        return result

    y_true = df["is_fraud"].astype(int)
    y_pred_if = df["is_anomaly"].astype(int)

    # Guard against degenerate predictions
    if y_pred_if.nunique() < 2 or y_true.nunique() < 2:
        logger.warning("Evaluation skipped: predictions or labels have only one class.")
        result.summary_lines.append(
            "⚠️ Evaluation skipped: only one class present in predictions or labels."
        )
        return result

    # Isolation Forest metrics
    result.precision = float(precision_score(y_true, y_pred_if, zero_division=0))
    result.recall = float(recall_score(y_true, y_pred_if, zero_division=0))
    result.f1 = float(f1_score(y_true, y_pred_if, zero_division=0))
    result.confusion = confusion_matrix(y_true, y_pred_if)
    result.classification_report_str = classification_report(
        y_true, y_pred_if, target_names=["Normal", "Anomaly"], zero_division=0
    )

    tn, fp, fn, tp = result.confusion.ravel()
    result.false_positives = int(fp)
    result.false_negatives = int(fn)

    # Statistical baseline metrics
    if "stat_anomaly" in df.columns:
        y_pred_stat = df["stat_anomaly"].astype(int)
        result.stat_precision = float(precision_score(y_true, y_pred_stat, zero_division=0))
        result.stat_recall = float(recall_score(y_true, y_pred_stat, zero_division=0))
        result.stat_f1 = float(f1_score(y_true, y_pred_stat, zero_division=0))

    # Summary
    result.summary_lines = [
        "=== Supervised Evaluation (Ground-Truth Labels Available) ===",
        "",
        "Isolation Forest:",
        f"  Precision : {result.precision:.3f}",
        f"  Recall    : {result.recall:.3f}",
        f"  F1-score  : {result.f1:.3f}",
        f"  False Positives : {result.false_positives}",
        f"  False Negatives : {result.false_negatives}",
    ]
    if result.stat_f1 is not None:
        result.summary_lines += [
            "",
            "Statistical Baseline (Z-score on amount):",
            f"  Precision : {result.stat_precision:.3f}",
            f"  Recall    : {result.stat_recall:.3f}",
            f"  F1-score  : {result.stat_f1:.3f}",
        ]
    result.summary_lines += [
        "",
        "Confusion Matrix (Isolation Forest):",
        f"  True Negatives  : {tn}",
        f"  False Positives : {fp}",
        f"  False Negatives : {fn}",
        f"  True Positives  : {tp}",
    ]

    return result


# ---------------------------------------------------------------------------
# Unsupervised evaluation
# ---------------------------------------------------------------------------

def evaluate_without_labels(df: pd.DataFrame) -> EvaluationResult:
    """
    When no ground-truth labels are available, report anomaly score
    distribution and contamination statistics.
    """
    result = EvaluationResult(has_labels=False)

    if "anomaly_score" not in df.columns:
        result.summary_lines.append(
            "❌ 'anomaly_score' column not found. Run detection first."
        )
        return result

    scores = df["anomaly_score"].dropna()
    result.score_mean = float(scores.mean())
    result.score_std = float(scores.std())
    result.score_min = float(scores.min())
    result.score_max = float(scores.max())
    result.total_count = len(df)
    result.flagged_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    result.flagged_pct = 100.0 * result.flagged_count / max(result.total_count, 1)

    result.summary_lines = [
        "=== Unsupervised Evaluation (No Ground-Truth Labels) ===",
        "",
        "Anomaly Score Distribution (Isolation Forest):",
        f"  Mean  : {result.score_mean:.4f}",
        f"  Std   : {result.score_std:.4f}",
        f"  Min   : {result.score_min:.4f}",
        f"  Max   : {result.score_max:.4f}",
        "",
        f"Flagged transactions : {result.flagged_count} / {result.total_count} "
        f"({result.flagged_pct:.1f}%)",
        "",
        "⚠️  Limitations of unsupervised evaluation:",
        "  - Without ground-truth labels, precision/recall/F1 cannot be computed.",
        "  - The contamination parameter directly influences the flag rate.",
        "  - Anomaly scores indicate relative deviation, not confirmed fraud.",
        "  - Manual review of flagged transactions is strongly recommended.",
    ]

    return result


# ---------------------------------------------------------------------------
# Auto-selector
# ---------------------------------------------------------------------------

def auto_evaluate(df: pd.DataFrame) -> EvaluationResult:
    """
    Automatically choose supervised or unsupervised evaluation
    based on whether 'is_fraud' is present in df.
    """
    if "is_fraud" in df.columns and df["is_fraud"].notna().any():
        logger.info("Labels detected – running supervised evaluation.")
        return evaluate_with_labels(df)
    else:
        logger.info("No labels – running unsupervised evaluation.")
        return evaluate_without_labels(df)
