"""
anomaly_detection.py
--------------------
Implements two anomaly detection approaches:

1. Isolation Forest (ML baseline – unsupervised)
2. Z-score on transaction amount (statistical baseline)

Each approach produces:
  - anomaly_score  : continuous risk score (lower = more suspicious for IF)
  - is_anomaly     : boolean flag
  - alert_status   : human-readable string
  - explanation    : plain-English reasons why the transaction was flagged

The model is intentionally kept separate from the Streamlit interface.

Public API
----------
IsolationForestDetector
    .fit(X)
    .predict(X, df) -> pd.DataFrame
    .save(path) / .load(path)

StatisticalBaselineDetector
    .fit(df)
    .predict(df)    -> pd.DataFrame

run_detection(df, X, contamination, threshold) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    ANOMALY_SCORE_THRESHOLD,
    ISOLATION_FOREST_PARAMS,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    Z_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert labels
# ---------------------------------------------------------------------------
ALERT_NORMAL = "Normal"
ALERT_REVIEW = "⚠️ Review Required"


# ---------------------------------------------------------------------------
# Isolation Forest detector
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Thin wrapper around sklearn's IsolationForest that also
    generates per-transaction explanations.
    """

    def __init__(
        self,
        contamination: float = ISOLATION_FOREST_PARAMS["contamination"],
        n_estimators: int = ISOLATION_FOREST_PARAMS["n_estimators"],
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit the Isolation Forest on feature matrix X."""
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        logger.info(
            "IsolationForest fitted on %d samples with contamination=%.3f",
            X.shape[0],
            self.contamination,
        )
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        feature_names: list[str],
        score_threshold: float = ANOMALY_SCORE_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Run inference and attach results to df.

        Adds columns
        ------------
        anomaly_score : float  (raw decision function output; lower = worse)
        is_anomaly    : bool
        alert_status  : str
        explanation   : str    (feature-based plain-English reason)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        scores = self._model.decision_function(X)  # shape (n,)
        flags = scores < score_threshold

        result = df.copy()
        result["anomaly_score"] = np.round(scores, 4)
        result["is_anomaly"] = flags
        result["alert_status"] = np.where(flags, ALERT_REVIEW, ALERT_NORMAL)
        result["explanation"] = _build_explanations(result, scores, flags, feature_names)

        n_flagged = flags.sum()
        logger.info(
            "IsolationForest flagged %d / %d transactions (%.1f%%).",
            n_flagged,
            len(flags),
            100 * n_flagged / max(len(flags), 1),
        )
        return result

    # ------------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        """Persist the fitted model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IsolationForestDetector":
        """Load a previously saved model."""
        detector = cls()
        detector._model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return detector


# ---------------------------------------------------------------------------
# Statistical baseline
# ---------------------------------------------------------------------------

class StatisticalBaselineDetector:
    """
    Flags transactions whose log-amount deviates by more than
    Z_SCORE_THRESHOLD standard deviations from the mean.

    This is intentionally simple – it serves as a comparison
    baseline for the Isolation Forest.
    """

    def __init__(self, z_threshold: float = Z_SCORE_THRESHOLD) -> None:
        self.z_threshold = z_threshold
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "StatisticalBaselineDetector":
        log_amounts = np.log1p(df["amount"].clip(lower=0.01))
        self._mean = float(log_amounts.mean())
        self._std = float(log_amounts.std())
        logger.info(
            "StatisticalBaseline fitted: mean_log_amount=%.4f, std=%.4f",
            self._mean,
            self._std,
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns
        ------------
        stat_z_score  : float  (z-score of log-amount)
        stat_anomaly  : bool
        stat_alert    : str
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Detector not fitted. Call .fit() first.")

        log_amounts = np.log1p(df["amount"].clip(lower=0.01))
        z_scores = (log_amounts - self._mean) / max(self._std, 1e-9)

        result = df.copy()
        result["stat_z_score"] = np.round(z_scores, 4)
        result["stat_anomaly"] = z_scores.abs() > self.z_threshold
        result["stat_alert"] = np.where(
            result["stat_anomaly"], ALERT_REVIEW, ALERT_NORMAL
        )
        return result


# ---------------------------------------------------------------------------
# Unified run_detection helper
# ---------------------------------------------------------------------------

def run_detection(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    contamination: float = ISOLATION_FOREST_PARAMS["contamination"],
    score_threshold: float = ANOMALY_SCORE_THRESHOLD,
    z_threshold: float = Z_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Convenience function: fits both detectors and returns an annotated DataFrame.

    Parameters
    ----------
    df            : Preprocessed transaction DataFrame
    X             : Scaled feature matrix (from preprocessing.py)
    feature_names : List of feature column names corresponding to X columns
    contamination : Fraction of expected anomalies for IsolationForest
    score_threshold : Anomaly score cut-off (lower = more anomalous)
    z_threshold   : Z-score cut-off for the statistical baseline

    Returns
    -------
    pd.DataFrame with all original columns plus:
      anomaly_score, is_anomaly, alert_status, explanation,
      stat_z_score, stat_anomaly, stat_alert
    """
    # -- Isolation Forest
    ifd = IsolationForestDetector(contamination=contamination)
    ifd.fit(X)
    result = ifd.predict(X, df, feature_names, score_threshold=score_threshold)

    # -- Statistical baseline
    sbd = StatisticalBaselineDetector(z_threshold=z_threshold)
    sbd.fit(df)
    result = sbd.predict(result)

    return result, ifd, sbd


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def _build_explanations(
    df: pd.DataFrame,
    scores: np.ndarray,
    flags: np.ndarray,
    feature_names: list[str],
) -> pd.Series:
    """
    Build a plain-English explanation for each transaction.
    Only flagged transactions receive a detailed reason string;
    normal transactions get a short confirmation.

    This is a rule-assisted approach using domain knowledge and
    the raw feature values.  It does NOT claim that a transaction
    is definitely fraudulent.
    """
    explanations = []

    # Compute per-feature statistics once for context
    amount_mean = df["amount"].mean() if "amount" in df.columns else 0
    amount_std = df["amount"].std() if "amount" in df.columns else 1

    for idx in range(len(df)):
        row = df.iloc[idx]
        score = scores[idx]
        flagged = bool(flags[idx])

        if not flagged:
            explanations.append("No unusual patterns detected.")
            continue

        reasons = []

        # Amount outlier
        if "amount" in df.columns and amount_std > 0:
            z = (row["amount"] - amount_mean) / amount_std
            if abs(z) > 2.5:
                direction = "high" if z > 0 else "low"
                reasons.append(
                    f"Transaction amount (${row['amount']:,.2f}) is unusually {direction} "
                    f"({z:.1f}σ from the dataset mean of ${amount_mean:,.2f})."
                )

        # Card not present (CNP fraud risk)
        if "card_present" in df.columns:
            cp = row.get("card_present", True)
            if str(cp).lower() in ("false", "0"):
                reasons.append("Card was not physically present (card-not-present transaction).")

        # Unusual hour
        if "hour_of_day" in df.columns:
            hour = int(row.get("hour_of_day", 12))
            if hour < 5 or hour >= 22:
                reasons.append(f"Transaction occurred at {hour:02d}:00, an unusual time of day.")

        # High anomaly score
        reasons.append(
            f"Isolation Forest anomaly score: {score:.4f} "
            f"(threshold: {ANOMALY_SCORE_THRESHOLD}). "
            "Lower scores indicate greater deviation from typical transaction patterns."
        )

        # Combine
        explanation = " | ".join(reasons) if reasons else (
            "Flagged by the anomaly detection model based on the overall feature pattern."
        )
        # Append disclaimer
        explanation += (
            " ⚠️ Note: This alert does not confirm fraud. "
            "Human review is required."
        )
        explanations.append(explanation)

    return pd.Series(explanations, index=df.index)
