"""
test_anomaly_detection.py
-------------------------
Tests for src/anomaly_detection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.anomaly_detection import (
    IsolationForestDetector,
    StatisticalBaselineDetector,
    run_detection,
)
from src.preprocessing import preprocess_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, n + 1)],
            "timestamp": pd.date_range("2026-08-01", periods=n, freq="30min"),
            "amount": rng.lognormal(4.0, 1.0, size=n).round(2),
            "merchant_category": rng.choice(["grocery", "electronics", "travel"], size=n),
            "transaction_type": rng.choice(["purchase", "refund", "transfer"], size=n),
            "customer_id": [f"CUST{i:04d}" for i in range(n)],
            "card_present": rng.choice([True, False], size=n),
            "country_code": rng.choice(["US", "GB", "CA"], size=n),
        }
    )


def _get_processed(n: int = 100):
    df = _make_df(n)
    processed, X, _ = preprocess_dataframe(df)
    return processed, X


# ---------------------------------------------------------------------------
# Test 1 – IsolationForest produces output columns
# ---------------------------------------------------------------------------

def test_isolation_forest_output_columns():
    processed, X = _get_processed()
    feature_names = ["amount", "hour_of_day", "day_of_week"]
    ifd = IsolationForestDetector(contamination=0.05)
    ifd.fit(X)
    result = ifd.predict(X, processed, feature_names)
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert "alert_status" in result.columns
    assert "explanation" in result.columns


# ---------------------------------------------------------------------------
# Test 2 – Anomaly scores are in expected range
# ---------------------------------------------------------------------------

def test_anomaly_scores_are_numeric():
    processed, X = _get_processed()
    ifd = IsolationForestDetector(contamination=0.05)
    ifd.fit(X)
    result = ifd.predict(X, processed, [])
    scores = result["anomaly_score"]
    assert scores.dtype in [np.float64, np.float32, float]
    assert not scores.isna().any()


# ---------------------------------------------------------------------------
# Test 3 – Higher contamination produces more (or equal) flags
# ---------------------------------------------------------------------------

def test_contamination_controls_flag_rate():
    """
    Higher contamination should produce more flagged transactions.

    Note: IsolationForest's contamination parameter sets the anomaly-score
    decision boundary during fit.  Our predict() method then applies a
    *separate* explicit score threshold (ANOMALY_SCORE_THRESHOLD).  The two
    thresholds interact, so the raw flagged fraction does not equal
    contamination exactly.  We verify the monotonic ordering instead:
    more contamination → more flags.
    """
    processed, X = _get_processed(200)

    flag_counts = []
    for cont in [0.02, 0.05, 0.10, 0.20]:
        ifd = IsolationForestDetector(contamination=cont)
        ifd.fit(X)
        result = ifd.predict(X, processed, [])
        flag_counts.append(int(result["is_anomaly"].sum()))

    # Monotonically non-decreasing (each higher contamination >= previous)
    for i in range(1, len(flag_counts)):
        assert flag_counts[i] >= flag_counts[i - 1], (
            f"Expected flag counts to be non-decreasing with contamination, "
            f"got {flag_counts}"
        )


# ---------------------------------------------------------------------------
# Test 4 – Statistical baseline produces expected columns
# ---------------------------------------------------------------------------

def test_statistical_baseline_columns():
    processed, _ = _get_processed()
    sbd = StatisticalBaselineDetector(z_threshold=3.0)
    sbd.fit(processed)
    result = sbd.predict(processed)
    assert "stat_z_score" in result.columns
    assert "stat_anomaly" in result.columns
    assert "stat_alert" in result.columns


# ---------------------------------------------------------------------------
# Test 5 – Statistical baseline flags extreme amounts
# ---------------------------------------------------------------------------

def test_statistical_baseline_flags_extreme_amounts():
    df = _make_df(50)
    # Inject an obvious outlier
    df.loc[0, "amount"] = 500_000.0
    processed, _, _ = preprocess_dataframe(df)
    sbd = StatisticalBaselineDetector(z_threshold=2.0)
    sbd.fit(processed)
    result = sbd.predict(processed)
    assert result.loc[0, "stat_anomaly"] is True or result.loc[0, "stat_anomaly"] == True


# ---------------------------------------------------------------------------
# Test 6 – run_detection returns annotated DataFrame
# ---------------------------------------------------------------------------

def test_run_detection_returns_full_result():
    processed, X = _get_processed()
    from src.preprocessing import get_feature_names
    feature_names = get_feature_names(processed)
    result, ifd, sbd = run_detection(processed, X, feature_names)
    assert "anomaly_score" in result.columns
    assert "stat_z_score" in result.columns
    assert len(result) == len(processed)


# ---------------------------------------------------------------------------
# Test 7 – Explain column is non-empty for flagged transactions
# ---------------------------------------------------------------------------

def test_explanations_non_empty_for_flagged():
    processed, X = _get_processed(100)
    ifd = IsolationForestDetector(contamination=0.1)
    ifd.fit(X)
    result = ifd.predict(X, processed, [])
    flagged = result[result["is_anomaly"] == True]
    if len(flagged) > 0:
        assert (flagged["explanation"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# Test 8 – Model raises error if not fitted
# ---------------------------------------------------------------------------

def test_unfitted_model_raises():
    processed, X = _get_processed()
    ifd = IsolationForestDetector()
    with pytest.raises(RuntimeError, match="not been fitted"):
        ifd.predict(X, processed, [])


# ---------------------------------------------------------------------------
# Test 9 – Unfitted statistical detector raises
# ---------------------------------------------------------------------------

def test_unfitted_statistical_baseline_raises():
    processed, _ = _get_processed()
    sbd = StatisticalBaselineDetector()
    with pytest.raises(RuntimeError):
        sbd.predict(processed)
