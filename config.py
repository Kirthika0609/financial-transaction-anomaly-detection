"""
config.py
---------
Central configuration for the Financial Anomaly Detection project.
All tuneable parameters live here so modules stay consistent and
experiments are reproducible.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"
MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "merchant_category",
    "transaction_type",
    "customer_id",
    "card_present",
    "country_code",
]

OPTIONAL_LABEL_COLUMN = "is_fraud"

COLUMN_DTYPES = {
    "transaction_id": "str",
    "amount": "float",
    "merchant_category": "str",
    "transaction_type": "str",
    "customer_id": "str",
    "card_present": "bool",
    "country_code": "str",
}

VALID_TRANSACTION_TYPES = {"purchase", "refund", "transfer", "withdrawal"}
AMOUNT_MIN = 0.01          # dollars
AMOUNT_MAX = 1_000_000.0   # dollars

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS = ["merchant_category", "transaction_type", "country_code"]
NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "card_present_enc",
    "merchant_category_enc",
    "transaction_type_enc",
    "country_code_enc",
    "amount_log",
]

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.05,   # expected fraction of anomalies
    "random_state": 42,
    "n_jobs": -1,
}

# Scores below this threshold are flagged as anomalies.
# Isolation Forest returns scores in (-1, 0]; lower = more anomalous.
ANOMALY_SCORE_THRESHOLD = -0.1

# Statistical baseline: flag transactions whose amount deviates from the
# mean by more than this many standard deviations.
Z_SCORE_THRESHOLD = 3.0

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
REPORT_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "merchant_category",
    "transaction_type",
    "customer_id",
    "country_code",
    "anomaly_score",
    "is_anomaly",
    "alert_status",
    "explanation",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
