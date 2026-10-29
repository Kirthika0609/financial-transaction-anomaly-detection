"""
preprocessing.py
----------------
Cleans and transforms raw transaction data into a model-ready feature matrix.

Every step is documented and the pipeline is stateless / reproducible:
calling preprocess_dataframe() on the same input always yields the same output.

Public API
----------
preprocess_dataframe(df, fit_encoders=True, encoders=None)
    -> (processed_df, feature_matrix, encoders_dict)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_FEATURES,
    OPTIONAL_LABEL_COLUMN,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Copy raw data (never mutate the original)
    2. Drop duplicate transaction IDs
    3. Parse and validate timestamps  → extract hour_of_day, day_of_week
    4. Coerce amount to float; clip to valid range
    5. Fill / drop missing values
    6. Encode categoricals with LabelEncoder
    7. Add engineered features (log-amount)
    8. Scale numeric features with StandardScaler
    9. Return processed DataFrame, feature matrix (numpy), and encoder dict

    Parameters
    ----------
    df          : Raw DataFrame (output of data_loader.load_csv)
    fit_encoders: If True, fit new encoders/scaler on this data.
                  If False, use the provided `encoders` dict (inference mode).
    encoders    : Dict with keys 'label_encoders' (dict) and 'scaler'.
                  Required when fit_encoders=False.

    Returns
    -------
    processed_df : pd.DataFrame  – cleaned, with all original + new columns
    X            : np.ndarray    – scaled feature matrix ready for the model
    encoders     : dict          – fitted encoders/scaler for later reuse
    """
    data = df.copy()

    # ------------------------------------------------------------------
    # 1. Drop duplicate transaction IDs
    # ------------------------------------------------------------------
    before = len(data)
    data = data.drop_duplicates(subset=["transaction_id"], keep="first")
    dropped = before - len(data)
    if dropped:
        logger.info("Dropped %d duplicate transaction IDs.", dropped)

    # ------------------------------------------------------------------
    # 2. Parse timestamps
    # ------------------------------------------------------------------
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    bad_ts = data["timestamp"].isna().sum()
    if bad_ts:
        logger.warning("Dropping %d rows with unparsable timestamps.", bad_ts)
        data = data.dropna(subset=["timestamp"])

    data["hour_of_day"] = data["timestamp"].dt.hour.astype(int)
    data["day_of_week"] = data["timestamp"].dt.dayofweek.astype(int)  # 0=Mon

    # ------------------------------------------------------------------
    # 3. Coerce and clip amount
    # ------------------------------------------------------------------
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data = data.dropna(subset=["amount"])
    data["amount"] = data["amount"].clip(lower=0.01)

    # ------------------------------------------------------------------
    # 4. Fill missing categoricals with mode; drop if mode unavailable
    # ------------------------------------------------------------------
    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            mode = data[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            missing_count = data[col].isna().sum()
            if missing_count:
                logger.info("Filling %d missing values in '%s' with '%s'.", missing_count, col, fill_value)
            data[col] = data[col].fillna(fill_value).astype(str)

    # ------------------------------------------------------------------
    # 5. Encode card_present as integer (True→1, False→0)
    # ------------------------------------------------------------------
    if "card_present" in data.columns:
        data["card_present_enc"] = (
            data["card_present"]
            .map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
            .fillna(0)
            .astype(int)
        )
    else:
        data["card_present_enc"] = 0

    # ------------------------------------------------------------------
    # 6. Label-encode categorical columns
    # ------------------------------------------------------------------
    if fit_encoders:
        label_encoders: dict[str, LabelEncoder] = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            data[f"{col}_enc"] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    else:
        if encoders is None:
            raise ValueError("encoders dict must be provided when fit_encoders=False")
        label_encoders = encoders.get("label_encoders", {})
        for col in CATEGORICAL_COLUMNS:
            le = label_encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            data[col] = data[col].apply(
                lambda x: x if x in known else le.classes_[0]
            )
            data[f"{col}_enc"] = le.transform(data[col].astype(str))

    # ------------------------------------------------------------------
    # 7. Engineered feature: log(amount)
    # ------------------------------------------------------------------
    data["amount_log"] = np.log1p(data["amount"])

    # ------------------------------------------------------------------
    # 8. Build feature matrix and scale
    # ------------------------------------------------------------------
    # Only keep features that actually exist in the DataFrame
    available_features = [f for f in NUMERIC_FEATURES if f in data.columns]
    X_raw = data[available_features].values.astype(float)

    if fit_encoders:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        scaler = encoders["scaler"]
        X_scaled = scaler.transform(X_raw)

    encoders_out = {"label_encoders": label_encoders, "scaler": scaler}

    logger.info(
        "Preprocessing complete: %d rows, %d features.",
        len(data),
        X_scaled.shape[1],
    )

    return data, X_scaled, encoders_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the list of feature names actually available after preprocessing."""
    return [f for f in NUMERIC_FEATURES if f in df.columns]
