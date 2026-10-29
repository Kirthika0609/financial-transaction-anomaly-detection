"""
data_loader.py
--------------
Responsible for loading, validating, and reporting on raw transaction data.
No transformations are applied here – that is preprocessing.py's job.

Public API
----------
load_csv(path_or_buffer) -> pd.DataFrame
validate_dataframe(df)   -> ValidationResult
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

from config import (
    AMOUNT_MAX,
    AMOUNT_MIN,
    OPTIONAL_LABEL_COLUMN,
    REQUIRED_COLUMNS,
    VALID_TRANSACTION_TYPES,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Holds the outcome of dataset validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✅ Dataset passed all required checks.")
        else:
            lines.append("❌ Dataset failed validation.")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for k, v in self.info.items():
            lines.append(f"  INFO  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path_or_buffer: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """
    Load a CSV file and return a raw DataFrame.

    Parameters
    ----------
    path_or_buffer : file path, Path object, or file-like object (e.g. Streamlit upload)

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError  – if the file is empty or cannot be parsed as CSV
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or contains no data.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the file as CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file contains no rows.")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Run all validation checks on a raw DataFrame.

    Checks performed
    ----------------
    1. Required columns present
    2. No completely empty DataFrame
    3. Transaction amount range
    4. Duplicate transaction IDs
    5. Missing values per column
    6. Valid transaction types
    7. Timestamp parsability
    8. Optional column warnings

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    result.info["total_rows"] = len(df)
    result.info["total_columns"] = len(df.columns)

    # --- 1. Required columns -------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.add_error(
            f"Missing required column(s): {missing_cols}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )
        # Cannot continue meaningfully without required columns
        return result

    # --- 2. Not empty --------------------------------------------------------
    if len(df) == 0:
        result.add_error("Dataset contains zero rows.")
        return result

    # --- 3. Amount range -----------------------------------------------------
    if "amount" in df.columns:
        df_amounts = pd.to_numeric(df["amount"], errors="coerce")
        non_numeric = df_amounts.isna().sum()
        if non_numeric > 0:
            result.add_error(
                f"{non_numeric} row(s) have non-numeric values in 'amount'."
            )
        else:
            out_of_range = df[
                (df_amounts < AMOUNT_MIN) | (df_amounts > AMOUNT_MAX)
            ]
            if len(out_of_range) > 0:
                result.add_error(
                    f"{len(out_of_range)} row(s) have 'amount' outside the valid "
                    f"range [{AMOUNT_MIN}, {AMOUNT_MAX}]. "
                    f"Example IDs: {out_of_range['transaction_id'].head(3).tolist()}"
                )

    # --- 4. Duplicate transaction IDs ----------------------------------------
    dup_count = df["transaction_id"].duplicated().sum()
    if dup_count > 0:
        result.add_warning(
            f"{dup_count} duplicate transaction_id value(s) detected. "
            "Duplicates will be removed during preprocessing."
        )
        result.info["duplicate_ids"] = dup_count

    # --- 5. Missing values ---------------------------------------------------
    missing = df[REQUIRED_COLUMNS].isnull().sum()
    missing_cols_with_nulls = missing[missing > 0]
    if not missing_cols_with_nulls.empty:
        detail = ", ".join(
            f"{col}: {cnt}" for col, cnt in missing_cols_with_nulls.items()
        )
        result.add_warning(
            f"Missing values found – {detail}. "
            "These will be handled during preprocessing."
        )
        result.info["missing_values"] = missing_cols_with_nulls.to_dict()

    # --- 6. Valid transaction types ------------------------------------------
    if "transaction_type" in df.columns:
        invalid_types = df[
            ~df["transaction_type"].isin(VALID_TRANSACTION_TYPES)
        ]["transaction_type"].unique()
        if len(invalid_types) > 0:
            result.add_error(
                f"Invalid transaction_type value(s): {list(invalid_types)}. "
                f"Allowed values: {sorted(VALID_TRANSACTION_TYPES)}"
            )

    # --- 7. Timestamp parsability --------------------------------------------
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        bad_ts = parsed.isna().sum()
        if bad_ts > 0:
            result.add_error(
                f"{bad_ts} row(s) have unparsable 'timestamp' values."
            )

    # --- 8. Optional label column -------------------------------------------
    if OPTIONAL_LABEL_COLUMN in df.columns:
        result.info["has_labels"] = True
        label_counts = df[OPTIONAL_LABEL_COLUMN].value_counts().to_dict()
        result.info["label_distribution"] = label_counts
    else:
        result.info["has_labels"] = False
        result.add_warning(
            f"Optional column '{OPTIONAL_LABEL_COLUMN}' not found. "
            "Model evaluation with ground-truth labels will be unavailable."
        )

    return result


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def load_and_validate(
    path_or_buffer: Union[str, Path, io.BytesIO, io.StringIO],
) -> tuple[pd.DataFrame, ValidationResult]:
    """Load a CSV and immediately validate it. Returns (df, result)."""
    df = load_csv(path_or_buffer)
    result = validate_dataframe(df)
    return df, result
