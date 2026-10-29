# Financial Anomaly Detection

A complete, modular machine learning project that identifies unusual financial
transactions using Isolation Forest and a statistical Z-score baseline.
Flagged transactions are surfaced in a Streamlit dashboard for human analyst
review.

> ⚠️ **Important:** An anomaly alert does **not** prove fraud or any illegal
> activity. Every flagged transaction requires independent human investigation.

---

## Features

| Feature | Details |
|---|---|
| Dataset management | Upload CSV or use built-in 1 000-row synthetic sample |
| Data validation | Checks required columns, types, amount range, duplicates, malformed records |
| Preprocessing | Deduplication, timestamp parsing, categorical encoding, feature engineering, StandardScaler |
| ML model | Isolation Forest (unsupervised, scikit-learn) |
| Statistical baseline | Z-score on log-transformed transaction amount |
| Explainable alerts | Per-transaction plain-English reasons (amount outlier, card-not-present, unusual hour, anomaly score) |
| Model evaluation | Precision / Recall / F1 / Confusion matrix when labels available; score-distribution report otherwise |
| Dashboard | 8-section Streamlit app with search, sort, filter, KPI cards, Plotly charts |
| Reports | Full CSV, flagged-only CSV, and plain-text summary download |
| Tests | 41 automated pytest tests — 41/41 passing |

---

## Technology Stack

| Component | Library / Version |
|---|---|
| Language | Python ≥ 3.11 |
| Data handling | pandas ≥ 2.0, numpy ≥ 1.26 |
| Machine learning | scikit-learn ≥ 1.4 |
| Visualisations | Plotly ≥ 5.22, Matplotlib ≥ 3.8, Seaborn ≥ 0.13 |
| Dashboard | Streamlit ≥ 1.35 |
| Model persistence | joblib ≥ 1.4 |
| Testing | pytest ≥ 8.0, pytest-cov ≥ 5.0 |
| Notebook | Jupyter / nbformat |

---

## Project Structure

```
Financial_Anomaly_Detection/
├── app.py                         # Streamlit dashboard (8 sections)
├── config.py                      # Centralised configuration & constants
├── requirements.txt               # Pinned dependencies
├── README.md
├── .gitignore
│
├── data/
│   ├── sample_transactions.csv    # 1 000-row synthetic dataset
│   ├── generate_sample.py         # Regenerate the dataset from scratch
│   └── README.md                  # Data dictionary
│
├── models/                        # Saved model artefacts (gitignored)
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive walkthrough
│
├── outputs/                       # Generated reports (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # CSV loading & validation
│   ├── preprocessing.py           # Cleaning, encoding, feature engineering
│   ├── anomaly_detection.py       # IsolationForest + Z-score detectors
│   ├── evaluation.py              # Supervised & unsupervised metrics
│   ├── visualization.py           # Plotly chart functions
│   └── reporting.py               # CSV / text report generation
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py        # 11 tests
    ├── test_preprocessing.py      #  8 tests
    ├── test_anomaly_detection.py  #  9 tests
    └── test_evaluation.py         # 12 tests  (41 total, all passing)
```

---

## Installation

### Prerequisites

- Python 3.11 or newer
- pip

### Steps (Windows PowerShell)

```powershell
# 1. Navigate to the project directory
cd "C:\Users\ADMIN\Downloads\Financial_Anomaly_Detection"

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Regenerate the sample dataset (already included, but run if needed)
python data/generate_sample.py
```

---

## Running the Application

```powershell
cd "C:\Users\ADMIN\Downloads\Financial_Anomaly_Detection"
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).
Open it in your browser.

### Dashboard walkthrough

| Section | What to do |
|---|---|
| 🏠 Home | Read the overview and column dictionary |
| 📤 Upload | Click **Load sample dataset** (or upload your own CSV) |
| 🔎 Validation & Preprocessing | Click **Run preprocessing** |
| 🤖 Anomaly Detection | Adjust sliders if desired, click **Run anomaly detection** |
| 🚨 Flagged Transactions | Search, filter, and inspect individual flagged records |
| 📊 Transaction Analysis | Explore the five visualisation tabs |
| 📈 Model Evaluation | View precision / recall / F1 and confusion matrix |
| ⬇️ Download Reports | Download full CSV, flagged-only CSV, or text summary |

---

## Running the Tests

```powershell
# Run all tests with summary
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single test file
python -m pytest tests/test_data_loader.py -v
```

### Actual test results (verified)

```
41 passed, 1 warning in ~10s
```

| File | Tests | Result |
|---|---|---|
| test_data_loader.py | 11 | ✅ All pass |
| test_preprocessing.py | 8 | ✅ All pass |
| test_anomaly_detection.py | 9 | ✅ All pass |
| test_evaluation.py | 12 | ✅ All pass |
| **Total** | **41** | **✅ 41/41** |

---

## Dataset Format & Data Dictionary

The CSV must contain these columns (column order does not matter):

| Column | Type | Required | Description |
|---|---|---|---|
| `transaction_id` | string | ✅ | Unique identifier. Format: `TXN000001` |
| `timestamp` | datetime (ISO 8601) | ✅ | UTC transaction time, e.g. `2026-08-01 12:00:00` |
| `amount` | float (USD) | ✅ | Transaction value. Must be in range [0.01, 1 000 000] |
| `merchant_category` | string | ✅ | Category: grocery, electronics, restaurant, travel, fuel, clothing, healthcare, entertainment |
| `transaction_type` | string | ✅ | One of: `purchase`, `refund`, `transfer`, `withdrawal` |
| `customer_id` | string | ✅ | Anonymised customer ID, e.g. `CUST4271` |
| `card_present` | bool | ✅ | `True` if physical card used; `False` for online/phone |
| `country_code` | string | ✅ | ISO 3166-1 alpha-2 code, e.g. `US`, `GB` |
| `is_fraud` | int (0/1) | ⬜ optional | Ground-truth label for model evaluation only |

---

## Model Explanation

### Isolation Forest

Isolation Forest is an ensemble of random binary trees. Each tree isolates
observations by randomly selecting a feature and a split value. Anomalies,
being unusual, require fewer splits to isolate and receive a **lower anomaly
score** (more negative). The model is *unsupervised* — it requires no fraud
labels to train.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `contamination` | 0.05 | Expected fraction of anomalies. Adjustable via the dashboard slider. |
| `n_estimators` | 100 | Number of trees in the forest. |
| `score_threshold` | -0.1 | Transactions with scores below this value are flagged. |

### Z-score Statistical Baseline

Flags transactions whose **log(1 + amount)** deviates by more than
`z_threshold` (default 3.0) standard deviations from the dataset mean.
This simple rule is easy to interpret and serves as a comparison baseline.

### Per-transaction Explanations

For each flagged transaction, the system reports:
- Whether the amount is statistically extreme (z-score > 2.5σ)
- Whether the card was not physically present
- Whether the transaction occurred at an unusual hour (before 05:00 or after 22:00)
- The raw Isolation Forest anomaly score and threshold

Explanations are based on actual feature values — no explanations are invented.

---

## Evaluation Methodology

### When labels are available (`is_fraud` column present)

| Metric | Description |
|---|---|
| Precision | Of all flagged transactions, what fraction were actually fraud? |
| Recall | Of all actual fraud transactions, what fraction were flagged? |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | TN / FP / FN / TP breakdown |

Both the Isolation Forest and the Z-score baseline are evaluated side-by-side.

### When labels are absent

The system reports:
- Anomaly score distribution (mean, std, min, max)
- Number and percentage of flagged transactions
- Clear explanation of why precision/recall cannot be computed
- Limitations of unsupervised evaluation

Evaluation results are never fabricated. The system only reports what it can
actually compute from the provided data.

---

## Known Limitations

1. **Unsupervised nature:** Isolation Forest does not learn from labelled fraud
   examples. It can flag legitimate unusual transactions (high false-positive
   rate) and miss fraud that resembles normal behaviour.

2. **Synthetic data:** The sample dataset is entirely generated; results do not
   reflect real-world fraud detection performance.

3. **Single-session state:** The Streamlit app holds all state in memory. Closing
   the browser tab resets all results.

4. **No time-series modelling:** Sequential patterns (velocity checks, behavioural
   change over time) are not modelled.

5. **Threshold sensitivity:** Results depend heavily on the contamination and
   score-threshold parameters. Tune these using domain expertise and labelled data.

6. **Label quality:** Evaluation metrics are only as good as the `is_fraud` labels.

---

## Future Improvements

- Add SHAP values for more rigorous feature attribution
- Implement Local Outlier Factor (LOF) as a second ML baseline
- Add velocity features (transactions per customer per hour/day)
- Support real-time streaming data via Kafka or Kinesis
- Add a feedback loop to incorporate analyst verdicts into model retraining
- Role-based access control for the analyst dashboard
- Export to PDF report format
- Deploy with Docker + authentication for team use

---

## Reproducing the Dataset

```powershell
python data/generate_sample.py
```

Fixed random seed (42) guarantees identical output every run.
