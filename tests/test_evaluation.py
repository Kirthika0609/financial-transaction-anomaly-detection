"""
test_evaluation.py
------------------
Tests for src/evaluation.py

Covers
------
1.  Supervised evaluation returns all expected metric fields
2.  Perfect predictions → F1 = 1.0
3.  Imperfect predictions → F1 < 1.0
4.  Missing 'is_fraud' column handled gracefully
5.  Unsupervised evaluation reports score statistics
6.  auto_evaluate selects supervised when labels are present
7.  auto_evaluate selects unsupervised when labels are absent
8.  Confusion matrix has correct 2×2 shape
9.  FP / FN counts are non-negative integers
10. Baseline (stat_anomaly) comparison included when column is present
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.evaluation import (
    EvaluationResult,
    auto_evaluate,
    evaluate_with_labels,
    evaluate_without_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_df(
    n: int = 100,
    flagged_indices: list | None = None,
    include_labels: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a fake detection-results DataFrame.

    flagged_indices : rows that are flagged AND labelled fraud (perfect predictions).
                      Defaults to the first 5 rows.
    """
    rng = np.random.default_rng(seed)
    flagged = set(flagged_indices if flagged_indices is not None else list(range(5)))

    is_anomaly = [i in flagged for i in range(n)]
    is_fraud   = [1 if i in flagged else 0 for i in range(n)]
    scores     = [-0.5 if i in flagged else 0.05 for i in range(n)]
    stat_anom  = is_anomaly[:]          # perfect baseline for test purposes

    data = {
        "transaction_id": [f"TXN{i:06d}" for i in range(n)],
        "amount":         rng.uniform(10.0, 500.0, size=n).round(2),
        "is_anomaly":     is_anomaly,
        "anomaly_score":  scores,
        "stat_anomaly":   stat_anom,
        "alert_status":   ["⚠️ Review Required" if a else "Normal" for a in is_anomaly],
    }
    if include_labels:
        data["is_fraud"] = is_fraud

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test 1 – Supervised evaluation returns all expected metric fields
# ---------------------------------------------------------------------------

def test_supervised_evaluation_returns_metrics():
    df = _make_result_df()
    result = evaluate_with_labels(df)
    assert result.has_labels is True
    assert result.precision is not None, "precision should not be None"
    assert result.recall    is not None, "recall should not be None"
    assert result.f1        is not None, "f1 should not be None"
    assert result.confusion is not None, "confusion matrix should not be None"


# ---------------------------------------------------------------------------
# Test 2 – Perfect predictions give F1 = 1.0
# ---------------------------------------------------------------------------

def test_perfect_predictions_f1_is_one():
    df = _make_result_df()          # is_anomaly == is_fraud by construction
    result = evaluate_with_labels(df)
    assert result.f1 is not None
    assert abs(result.f1 - 1.0) < 1e-9, f"Expected F1=1.0, got {result.f1}"


# ---------------------------------------------------------------------------
# Test 3 – Imperfect predictions give F1 < 1.0
# ---------------------------------------------------------------------------

def test_imperfect_predictions_f1_less_than_one():
    df = _make_result_df()
    # Introduce a false negative (row 3 labelled fraud but not flagged)
    df.loc[3, "is_anomaly"] = False
    # Introduce a false positive (row 50 flagged but not fraud)
    df.loc[50, "is_anomaly"] = True
    result = evaluate_with_labels(df)
    assert result.f1 is not None
    assert result.f1 < 1.0, f"Expected F1 < 1.0 with errors, got {result.f1}"


# ---------------------------------------------------------------------------
# Test 4 – Missing 'is_fraud' column handled gracefully
# ---------------------------------------------------------------------------

def test_evaluate_with_labels_missing_column():
    df = _make_result_df(include_labels=False)
    result = evaluate_with_labels(df)
    # Should not raise; should return an EvaluationResult with an explanatory message
    assert isinstance(result, EvaluationResult)
    assert len(result.summary_lines) > 0
    combined = " ".join(result.summary_lines)
    assert "is_fraud" in combined or "Cannot" in combined or "Missing" in combined


# ---------------------------------------------------------------------------
# Test 5 – Unsupervised evaluation reports score statistics
# ---------------------------------------------------------------------------

def test_unsupervised_evaluation_reports_stats():
    df = _make_result_df(include_labels=False)
    result = evaluate_without_labels(df)
    assert result.has_labels is False
    assert result.score_mean is not None
    assert result.score_std  is not None
    assert result.score_min  is not None
    assert result.score_max  is not None
    assert result.total_count > 0
    assert result.flagged_count >= 0


# ---------------------------------------------------------------------------
# Test 6 – auto_evaluate selects supervised when labels are present
# ---------------------------------------------------------------------------

def test_auto_evaluate_chooses_supervised_with_labels():
    df = _make_result_df(include_labels=True)
    result = auto_evaluate(df)
    assert result.has_labels is True
    assert result.f1 is not None


# ---------------------------------------------------------------------------
# Test 7 – auto_evaluate selects unsupervised when labels are absent
# ---------------------------------------------------------------------------

def test_auto_evaluate_chooses_unsupervised_without_labels():
    df = _make_result_df(include_labels=False)
    result = auto_evaluate(df)
    assert result.has_labels is False
    assert result.score_mean is not None


# ---------------------------------------------------------------------------
# Test 8 – Confusion matrix has correct 2×2 shape
# ---------------------------------------------------------------------------

def test_confusion_matrix_shape():
    df = _make_result_df()
    result = evaluate_with_labels(df)
    assert result.confusion is not None
    assert result.confusion.shape == (2, 2), (
        f"Expected (2,2) confusion matrix, got {result.confusion.shape}"
    )


# ---------------------------------------------------------------------------
# Test 9 – FP / FN counts are non-negative
# ---------------------------------------------------------------------------

def test_fp_fn_non_negative():
    df = _make_result_df()
    result = evaluate_with_labels(df)
    assert result.false_positives >= 0
    assert result.false_negatives >= 0


# ---------------------------------------------------------------------------
# Test 10 – Baseline comparison included when stat_anomaly is present
# ---------------------------------------------------------------------------

def test_baseline_comparison_included():
    df = _make_result_df()          # stat_anomaly is included in _make_result_df
    result = evaluate_with_labels(df)
    assert result.stat_f1 is not None, (
        "stat_f1 should be computed when stat_anomaly column is present"
    )
    assert result.stat_precision is not None
    assert result.stat_recall    is not None


# ---------------------------------------------------------------------------
# Test 11 – EvaluationResult.summary() returns a non-empty string
# ---------------------------------------------------------------------------

def test_evaluation_result_summary_string():
    df = _make_result_df()
    result = evaluate_with_labels(df)
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# Test 12 – Unsupervised flagged_pct is in [0, 100]
# ---------------------------------------------------------------------------

def test_unsupervised_flagged_pct_range():
    df = _make_result_df(include_labels=False)
    result = evaluate_without_labels(df)
    assert 0.0 <= result.flagged_pct <= 100.0
