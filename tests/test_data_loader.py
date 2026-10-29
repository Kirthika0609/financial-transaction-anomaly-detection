"""
test_data_loader.py
-------------------
Tests for src/data_loader.py

Covers
------
1. Valid dataset loads successfully
2. Missing required columns triggers error
3. Missing values trigger a warning (not an error)
4. Duplicate transaction IDs trigger a warning
5. Invalid transaction amounts trigger an error
6. Empty CSV raises ValueError
7. Unknown transaction types trigger an error
8. Optional label column detection
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root and src/ are importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.data_loader import load_csv, validate_dataframe, load_and_validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid transaction DataFrame."""
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, n + 1)],
            "timestamp": ["2026-08-01 12:00:00"] * n,
            "amount": [100.0 + i * 10 for i in range(n)],
            "merchant_category": ["grocery"] * n,
            "transaction_type": ["purchase"] * n,
            "customer_id": [f"CUST{i:04d}" for i in range(n)],
            "card_present": [True] * n,
            "country_code": ["US"] * n,
            "is_fraud": [0] * n,
        }
    )


def _df_to_buffer(df: pd.DataFrame) -> io.StringIO:
    """Serialise a DataFrame to a StringIO CSV buffer."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Test 1 – Valid dataset
# ---------------------------------------------------------------------------

def test_valid_dataset_loads():
    """A well-formed CSV should load without errors or validation failures."""
    df = _make_valid_df()
    buf = _df_to_buffer(df)
    loaded = load_csv(buf)
    assert len(loaded) == 5
    result = validate_dataframe(loaded)
    assert result.is_valid, f"Unexpected errors: {result.errors}"
    assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Test 2 – Missing required columns
# ---------------------------------------------------------------------------

def test_missing_required_columns():
    """DataFrame missing required columns should fail validation."""
    df = _make_valid_df().drop(columns=["amount", "transaction_type"])
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("Missing required column" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test 3 – Missing values produce a warning
# ---------------------------------------------------------------------------

def test_missing_values_produce_warning():
    """NaN values in required columns should produce a warning, not a hard error."""
    df = _make_valid_df()
    df.loc[0, "merchant_category"] = None
    df.loc[1, "country_code"] = None
    result = validate_dataframe(df)
    # Missing values are handled in preprocessing, so dataset is still valid
    assert result.is_valid
    assert any("Missing values" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 4 – Duplicate transaction IDs
# ---------------------------------------------------------------------------

def test_duplicate_transaction_ids_produce_warning():
    """Duplicate IDs should produce a warning but not fail validation."""
    df = _make_valid_df()
    df.loc[2, "transaction_id"] = df.loc[0, "transaction_id"]  # duplicate
    result = validate_dataframe(df)
    assert result.is_valid  # still valid; duplicates are removed in preprocessing
    assert any("duplicate" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 5 – Invalid transaction amounts
# ---------------------------------------------------------------------------

def test_invalid_transaction_amounts():
    """Amounts of 0 or negative values should fail validation."""
    df = _make_valid_df()
    df.loc[0, "amount"] = -50.0   # negative
    df.loc[1, "amount"] = 0.0     # zero
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("amount" in e.lower() for e in result.errors)


def test_amount_above_maximum():
    """Amounts above the allowed maximum should fail validation."""
    df = _make_valid_df()
    df.loc[0, "amount"] = 2_000_000.0
    result = validate_dataframe(df)
    assert not result.is_valid


# ---------------------------------------------------------------------------
# Test 6 – Empty CSV
# ---------------------------------------------------------------------------

def test_empty_csv_raises_error():
    """An empty CSV (header only) should raise ValueError."""
    buf = io.StringIO("transaction_id,timestamp,amount\n")
    with pytest.raises(ValueError):
        load_csv(buf)


def test_completely_empty_csv_raises_error():
    """A CSV with no content at all should raise ValueError."""
    buf = io.StringIO("")
    with pytest.raises(ValueError):
        load_csv(buf)


# ---------------------------------------------------------------------------
# Test 7 – Invalid transaction types
# ---------------------------------------------------------------------------

def test_invalid_transaction_type():
    """Unrecognised transaction_type values should fail validation."""
    df = _make_valid_df()
    df.loc[0, "transaction_type"] = "gift_card"  # not in VALID_TRANSACTION_TYPES
    result = validate_dataframe(df)
    assert not result.is_valid
    assert any("transaction_type" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 8 – Optional label column detection
# ---------------------------------------------------------------------------

def test_label_column_detected():
    """Presence of 'is_fraud' should be noted in validation info."""
    df = _make_valid_df()
    result = validate_dataframe(df)
    assert result.info.get("has_labels") is True


def test_missing_label_column_warns():
    """Absence of 'is_fraud' should produce a warning, not an error."""
    df = _make_valid_df().drop(columns=["is_fraud"])
    result = validate_dataframe(df)
    assert result.is_valid
    assert result.info.get("has_labels") is False
    assert any("is_fraud" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 9 – load_and_validate convenience wrapper
# ---------------------------------------------------------------------------

def test_load_and_validate_convenience():
    """load_and_validate should return a tuple of (df, ValidationResult)."""
    df = _make_valid_df()
    buf = _df_to_buffer(df)
    loaded_df, result = load_and_validate(buf)
    assert isinstance(loaded_df, pd.DataFrame)
    assert result.is_valid
