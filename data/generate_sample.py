"""
generate_sample.py
------------------
Generates a reproducible synthetic financial transaction dataset.
Run once to create data/sample_transactions.csv.

Usage:
    python data/generate_sample.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_NORMAL = 950
N_ANOMALY = 50
OUTPUT_PATH = Path(__file__).parent / "sample_transactions.csv"

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel",
    "fuel", "clothing", "healthcare", "entertainment",
]
TRANSACTION_TYPES = ["purchase", "refund", "transfer", "withdrawal"]
COUNTRY_CODES = ["US", "GB", "CA", "AU", "DE", "FR", "JP", "IN", "BR", "MX"]


def _random_timestamp(n: int) -> pd.Series:
    """Generate n random timestamps within the last 90 days."""
    start = pd.Timestamp("2026-08-01 00:00:00")
    end = pd.Timestamp("2026-10-29 23:59:59")
    delta_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, delta_seconds, size=n)
    timestamps = [start + pd.Timedelta(seconds=int(s)) for s in offsets]
    return pd.Series(timestamps)


def _generate_normal(n: int) -> pd.DataFrame:
    """Generate n normal transactions."""
    amounts = rng.lognormal(mean=4.0, sigma=1.2, size=n).round(2)
    amounts = np.clip(amounts, 0.01, 9_999.99)
    return pd.DataFrame(
        {
            "amount": amounts,
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n),
            "transaction_type": rng.choice(
                TRANSACTION_TYPES, size=n, p=[0.70, 0.10, 0.12, 0.08]
            ),
            "customer_id": [
                f"CUST{rng.integers(1000, 9999):04d}" for _ in range(n)
            ],
            "card_present": rng.choice([True, False], size=n, p=[0.75, 0.25]),
            "country_code": rng.choice(
                COUNTRY_CODES, size=n, p=[0.45, 0.10, 0.10, 0.08, 0.07, 0.07, 0.05, 0.04, 0.02, 0.02]
            ),
            "is_fraud": 0,
        }
    )


def _generate_anomalies(n: int) -> pd.DataFrame:
    """Generate n synthetic anomalous transactions."""
    # Mix of different anomaly patterns
    n_high_amount = n // 3
    n_unusual_hour = n // 3
    n_foreign = n - n_high_amount - n_unusual_hour

    # Pattern 1: extremely high amounts
    high_amounts = rng.uniform(50_000, 250_000, size=n_high_amount).round(2)

    # Pattern 2: unusual hour + not card-present
    normal_amounts2 = rng.lognormal(mean=5.5, sigma=1.5, size=n_unusual_hour).round(2)

    # Pattern 3: foreign + high amount
    foreign_amounts = rng.uniform(10_000, 80_000, size=n_foreign).round(2)

    amounts = np.concatenate([high_amounts, normal_amounts2, foreign_amounts])
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n)
    transaction_type = rng.choice(TRANSACTION_TYPES, size=n, p=[0.50, 0.05, 0.30, 0.15])
    customer_id = [f"CUST{rng.integers(1000, 9999):04d}" for _ in range(n)]
    card_present = np.array([False] * n_high_amount + [False] * n_unusual_hour + [False] * n_foreign)
    country_code = (
        rng.choice(["RU", "NG", "CN", "VN"], size=n_high_amount).tolist()
        + rng.choice(COUNTRY_CODES, size=n_unusual_hour).tolist()
        + rng.choice(["RU", "NG", "CN", "VN"], size=n_foreign).tolist()
    )
    return pd.DataFrame(
        {
            "amount": amounts.round(2),
            "merchant_category": merchant_category,
            "transaction_type": transaction_type,
            "customer_id": customer_id,
            "card_present": card_present,
            "country_code": country_code,
            "is_fraud": 1,
        }
    )


def generate_dataset() -> pd.DataFrame:
    normal = _generate_normal(N_NORMAL)
    anomalies = _generate_anomalies(N_ANOMALY)
    df = pd.concat([normal, anomalies], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add IDs and timestamps
    df.insert(0, "transaction_id", [f"TXN{i+1:06d}" for i in range(len(df))])
    df.insert(1, "timestamp", _random_timestamp(len(df)))

    # Sort by timestamp for realism
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"TXN{i+1:06d}" for i in range(len(df))]

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} transactions to {OUTPUT_PATH}")
    print(f"  Normal: {(df['is_fraud'] == 0).sum()}")
    print(f"  Anomaly: {(df['is_fraud'] == 1).sum()}")
    print(df.head())
