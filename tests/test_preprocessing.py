"""
test_preprocessing.py
---------------------
Tests for src/preprocessing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.preprocessing import preprocess_dataframe, get_feature_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, n + 1)],
            "timestamp": pd.date_range("2026-08-01", periods=n, freq="h"),
            "amount": rng.uniform(10, 500, size=n).round(2),
            "merchant_category": rng.choice(["grocery", "electronics", "travel"], size=n),
            "transaction_type": rng.choice(["purchase", "refund", "transfer"], size=n),
            "customer_id": [f"CUST{i:04d}" for i in range(n)],
            "card_present": rng.choice([True, False], size=n),
            "country_code": rng.choice(["US", "GB", "CA"], size=n),
            "is_fraud": rng.integers(0, 2, size=n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 – Output shape and types
# ---------------------------------------------------------------------------

def test_preprocess_returns_correct_shapes():
    df = _make_df(20)
    processed, X, encoders = preprocess_dataframe(df)
    assert isinstance(processed, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(processed)
    assert X.shape[1] > 0


# ---------------------------------------------------------------------------
# Test 2 – Duplicates are removed
# ---------------------------------------------------------------------------

def test_duplicates_removed():
    df = _make_df(10)
    df.loc[9, "transaction_id"] = df.loc[0, "transaction_id"]  # introduce dup
    processed, X, _ = preprocess_dataframe(df)
    assert len(processed) == 9  # one duplicate removed


# ---------------------------------------------------------------------------
# Test 3 – Engineered features present
# ---------------------------------------------------------------------------

def test_engineered_features_present():
    df = _make_df(10)
    processed, _, _ = preprocess_dataframe(df)
    assert "hour_of_day" in processed.columns
    assert "day_of_week" in processed.columns
    assert "amount_log" in processed.columns


# ---------------------------------------------------------------------------
# Test 4 – Missing categorical values filled
# ---------------------------------------------------------------------------

def test_missing_categoricals_filled():
    df = _make_df(10)
    df.loc[0, "merchant_category"] = None
    df.loc[1, "country_code"] = None
    processed, X, _ = preprocess_dataframe(df)
    assert processed["merchant_category"].isna().sum() == 0
    assert processed["country_code"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test 5 – Scaler produces standardised output
# ---------------------------------------------------------------------------

def test_feature_matrix_is_scaled():
    df = _make_df(50)
    _, X, _ = preprocess_dataframe(df)
    # Column means should be close to 0 and stds close to 1
    col_means = X.mean(axis=0)
    col_stds = X.std(axis=0)
    assert np.allclose(col_means, 0, atol=1.0), f"Means not near zero: {col_means}"
    assert np.all(col_stds > 0), "Some features have zero variance after scaling"


# ---------------------------------------------------------------------------
# Test 6 – Reproducibility (same input → same output)
# ---------------------------------------------------------------------------

def test_preprocessing_is_reproducible():
    df = _make_df(20)
    _, X1, _ = preprocess_dataframe(df.copy())
    _, X2, _ = preprocess_dataframe(df.copy())
    np.testing.assert_array_almost_equal(X1, X2)


# ---------------------------------------------------------------------------
# Test 7 – Inference mode (fit_encoders=False)
# ---------------------------------------------------------------------------

def test_inference_mode_uses_provided_encoders():
    df_train = _make_df(20)
    df_test = _make_df(5)
    _, _, encoders = preprocess_dataframe(df_train)
    processed_test, X_test, _ = preprocess_dataframe(
        df_test, fit_encoders=False, encoders=encoders
    )
    assert X_test.shape[0] == len(df_test)


# ---------------------------------------------------------------------------
# Test 8 – get_feature_names
# ---------------------------------------------------------------------------

def test_get_feature_names_returns_list():
    df = _make_df(10)
    processed, _, _ = preprocess_dataframe(df)
    names = get_feature_names(processed)
    assert isinstance(names, list)
    assert len(names) > 0
