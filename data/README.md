# Data Directory

## Files

| File | Description |
|---|---|
| `sample_transactions.csv` | 1 000-row synthetic dataset (950 normal, 50 anomalous) |
| `generate_sample.py` | Script that reproduces the sample dataset from scratch |

---

## Data Dictionary — `sample_transactions.csv`

| Column | Type | Example | Description |
|---|---|---|---|
| `transaction_id` | string | TXN000001 | Unique identifier for every transaction. Format: `TXN` followed by a six-digit zero-padded integer. |
| `timestamp` | datetime (UTC) | 2026-08-01 04:04:52 | Date and time the transaction was recorded. ISO 8601 format. |
| `amount` | float (USD) | 124.50 | Monetary value of the transaction in US dollars. Must be ≥ 0.01 and ≤ 1 000 000. |
| `merchant_category` | string | grocery | Merchant category code label. One of: `grocery`, `electronics`, `restaurant`, `travel`, `fuel`, `clothing`, `healthcare`, `entertainment`. |
| `transaction_type` | string | purchase | Type of financial operation. One of: `purchase`, `refund`, `transfer`, `withdrawal`. |
| `customer_id` | string | CUST4271 | Anonymised customer identifier. Format: `CUST` followed by a four-digit number. No personally identifiable information. |
| `card_present` | boolean | True | Whether the physical card was present at the point of sale. `True` for in-person; `False` for card-not-present (online, phone). |
| `country_code` | string | US | ISO 3166-1 alpha-2 country code of the merchant or receiving account. |
| `is_fraud` | integer (0/1) | 0 | **Optional label** used only for model evaluation. `1` = known anomalous / fraudulent; `0` = normal. This column may be absent in unlabelled uploads. |

---

## Important Notes

* **This dataset is entirely synthetic.** It was generated programmatically using `generate_sample.py` and contains no real financial data.
* The `is_fraud` column is provided for evaluation purposes only. Its presence does not mean the system should rely on it at inference time.
* Do not commit real financial data to version control. The `.gitignore` file is configured to block `data/raw/` and `data/private/` directories.

---

## Regenerating the Dataset

```powershell
python data/generate_sample.py
```

This will overwrite `sample_transactions.csv` with a fresh reproducible copy (random seed is fixed at 42).
