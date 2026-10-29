"""
app.py
------
Streamlit dashboard for the Financial Anomaly Detection project.

Sections
--------
1. Home / Project Overview
2. Upload Transactions
3. Data Validation & Preprocessing
4. Anomaly Detection
5. Flagged Transactions
6. Transaction Analysis & Visualisations
7. Model Evaluation
8. Download Reports
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import io
import logging

import pandas as pd
import streamlit as st

from config import (
    ANOMALY_SCORE_THRESHOLD,
    ISOLATION_FOREST_PARAMS,
    SAMPLE_DATA_PATH,
    Z_SCORE_THRESHOLD,
)
from src.anomaly_detection import run_detection
from src.data_loader import load_csv, validate_dataframe
from src.evaluation import auto_evaluate
from src.preprocessing import get_feature_names, preprocess_dataframe
from src.reporting import generate_csv_report, generate_flagged_csv, generate_summary_text
from src.visualization import (
    plot_amount_distribution,
    plot_anomaly_scores,
    plot_confusion_matrix,
    plot_flagged_by_category,
    plot_kpi_cards,
    plot_normal_vs_flagged,
    plot_score_vs_amount,
    plot_time_series,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        border-left: 4px solid #2196F3;
    }
    .alert-card {
        background: #2A0A0A;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        border-left: 4px solid #F44336;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebar"] { background-color: #12121F; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "result_df": None,
        "encoders": None,
        "eval_result": None,
        "validation_result": None,
        "detection_run": False,
        "data_source": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
SECTIONS = [
    "🏠  Home",
    "📤  Upload Transactions",
    "🔎  Data Validation & Preprocessing",
    "🤖  Anomaly Detection",
    "🚨  Flagged Transactions",
    "📊  Transaction Analysis",
    "📈  Model Evaluation",
    "⬇️  Download Reports",
]

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/fraud--v1.png",
        width=72,
    )
    st.title("Anomaly Detector")
    st.markdown("---")
    section = st.radio("Navigate", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    st.caption("⚠️ Anomaly alerts do **not** confirm fraud. Human review is required.")


# ===========================================================================
# SECTION 1 – HOME
# ===========================================================================
if section == SECTIONS[0]:
    st.title("🔍 Financial Anomaly Detection")
    st.subheader("Identifying unusual transactions for analyst review")

    st.info(
        "**Important disclaimer:** This tool flags transactions that deviate from "
        "typical patterns using machine learning. An anomaly alert does **not** "
        "prove fraud or any illegal activity. All flagged transactions require "
        "independent human investigation.",
        icon="⚠️",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### What this tool does
            - Accepts a CSV of financial transactions
            - Validates and cleans the data automatically
            - Runs **Isolation Forest** (unsupervised ML) to detect anomalies
            - Compares results with a **Z-score statistical baseline**
            - Provides per-transaction explanations
            - Generates downloadable reports
            """
        )
    with col2:
        st.markdown(
            """
            ### Technology stack
            | Component | Library |
            |---|---|
            | Data handling | Pandas, NumPy |
            | ML model | Scikit-learn |
            | Visualisations | Plotly |
            | Dashboard | Streamlit |
            | Testing | Pytest |

            ### Quick start
            1. Go to **Upload Transactions**
            2. Use the sample dataset or upload your own CSV
            3. Run **Anomaly Detection**
            4. Review flagged transactions
            5. Download reports
            """
        )

    st.markdown("---")
    st.markdown("### Required CSV columns")
    cols_info = {
        "transaction_id": "Unique identifier for each transaction",
        "timestamp": "Date and time (ISO 8601 format)",
        "amount": "Transaction value in USD (0.01 – 1,000,000)",
        "merchant_category": "Category label (grocery, electronics, etc.)",
        "transaction_type": "purchase | refund | transfer | withdrawal",
        "customer_id": "Anonymised customer identifier",
        "card_present": "True / False – was the card physically present",
        "country_code": "ISO 3166-1 alpha-2 country code",
        "is_fraud *(optional)*": "0/1 label for supervised evaluation",
    }
    st.table(pd.DataFrame({"Column": list(cols_info.keys()), "Description": list(cols_info.values())}))


# ===========================================================================
# SECTION 2 – UPLOAD
# ===========================================================================
elif section == SECTIONS[1]:
    st.title("📤 Upload Transactions")

    tab_upload, tab_sample = st.tabs(["Upload your CSV", "Use sample dataset"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Select a CSV file containing transaction data",
            type=["csv"],
            help="Must include all required columns listed on the Home page.",
        )
        if uploaded is not None:
            try:
                raw_df = load_csv(uploaded)
                st.session_state["raw_df"] = raw_df
                st.session_state["data_source"] = f"Uploaded: {uploaded.name}"
                st.session_state["detection_run"] = False
                st.success(f"✅ Loaded {len(raw_df):,} rows from **{uploaded.name}**")
                st.dataframe(raw_df.head(10), use_container_width=True)
            except ValueError as exc:
                st.error(f"❌ Could not load file: {exc}")

    with tab_sample:
        st.markdown(
            "The sample dataset contains **1 000 synthetic transactions** (950 normal, "
            "50 anomalous) generated with a fixed random seed for reproducibility. "
            "It includes the optional `is_fraud` label column so supervised evaluation is available."
        )
        if st.button("Load sample dataset", type="primary"):
            raw_df = load_csv(SAMPLE_DATA_PATH)
            st.session_state["raw_df"] = raw_df
            st.session_state["data_source"] = "Sample dataset"
            st.session_state["detection_run"] = False
            st.success(f"✅ Loaded {len(raw_df):,} transactions from the sample dataset.")
            st.dataframe(raw_df.head(10), use_container_width=True)

    if st.session_state["raw_df"] is not None:
        st.info(f"Active dataset: **{st.session_state['data_source']}** | {len(st.session_state['raw_df']):,} rows")


# ===========================================================================
# SECTION 3 – VALIDATION & PREPROCESSING
# ===========================================================================
elif section == SECTIONS[2]:
    st.title("🔎 Data Validation & Preprocessing")

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Please upload a dataset first (section: Upload Transactions).")
        st.stop()

    st.subheader("Validation report")
    val_result = validate_dataframe(raw_df)
    st.session_state["validation_result"] = val_result

    if val_result.is_valid:
        st.success("✅ Dataset passed all required validation checks.")
    else:
        st.error("❌ Dataset failed validation. See errors below.")

    for err in val_result.errors:
        st.error(f"**ERROR:** {err}")
    for warn in val_result.warnings:
        st.warning(f"**WARNING:** {warn}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{val_result.info.get('total_rows', 0):,}")
    col2.metric("Total Columns", val_result.info.get("total_columns", 0))
    col3.metric("Duplicate IDs", val_result.info.get("duplicate_ids", 0))
    col4.metric("Has Labels", "Yes" if val_result.info.get("has_labels") else "No")

    st.markdown("---")

    if not val_result.is_valid:
        st.error("Preprocessing cannot continue until validation errors are fixed.")
        st.stop()

    st.subheader("Preprocessing")
    if st.button("▶ Run preprocessing", type="primary"):
        with st.spinner("Preprocessing data…"):
            processed_df, X, encoders = preprocess_dataframe(raw_df)
            st.session_state["processed_df"] = processed_df
            st.session_state["X"] = X
            st.session_state["encoders"] = encoders
            st.session_state["detection_run"] = False
        st.success(f"✅ Preprocessing complete. {len(processed_df):,} rows, {X.shape[1]} features.")

    processed_df = st.session_state.get("processed_df")
    if processed_df is not None:
        st.markdown("#### Preview of processed data")
        display_cols = [c for c in processed_df.columns if not c.endswith("_enc")]
        st.dataframe(processed_df[display_cols].head(20), use_container_width=True)

        with st.expander("Feature engineering details"):
            st.markdown(
                """
                The following transformations are applied:

                | Feature | Source | Description |
                |---|---|---|
                | `hour_of_day` | timestamp | Hour (0–23) extracted from UTC timestamp |
                | `day_of_week` | timestamp | Day of week (0=Monday, 6=Sunday) |
                | `card_present_enc` | card_present | Boolean encoded as 0/1 |
                | `merchant_category_enc` | merchant_category | Integer-encoded label |
                | `transaction_type_enc` | transaction_type | Integer-encoded label |
                | `country_code_enc` | country_code | Integer-encoded label |
                | `amount_log` | amount | log(1 + amount) – reduces right-skew |

                All numeric features are then standardised (zero mean, unit variance).
                """
            )

        with st.expander("Missing value handling"):
            missing_info = val_result.info.get("missing_values", {})
            if missing_info:
                st.table(
                    pd.DataFrame(
                        {"Column": list(missing_info.keys()), "Missing Count": list(missing_info.values())}
                    )
                )
                st.caption("Categorical missing values were filled with the column mode.")
            else:
                st.success("No missing values detected.")


# ===========================================================================
# SECTION 4 – ANOMALY DETECTION
# ===========================================================================
elif section == SECTIONS[3]:
    st.title("🤖 Anomaly Detection")

    processed_df = st.session_state.get("processed_df")
    X = st.session_state.get("X")

    if processed_df is None or X is None:
        st.warning("Please complete Data Validation & Preprocessing first.")
        st.stop()

    st.markdown("### Model configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        contamination = st.slider(
            "Contamination (expected anomaly fraction)",
            min_value=0.01,
            max_value=0.20,
            value=float(ISOLATION_FOREST_PARAMS["contamination"]),
            step=0.01,
            help="Sets the proportion of transactions IsolationForest will flag as anomalies. "
                 "Adjust based on your expected fraud rate.",
        )
    with col2:
        score_threshold = st.slider(
            "Anomaly score threshold",
            min_value=-0.5,
            max_value=0.0,
            value=float(ANOMALY_SCORE_THRESHOLD),
            step=0.01,
            help="Transactions with scores below this value are flagged. "
                 "Lower = stricter (fewer flags). Scores come from IsolationForest.decision_function().",
        )
    with col3:
        z_threshold = st.slider(
            "Z-score threshold (statistical baseline)",
            min_value=1.5,
            max_value=5.0,
            value=float(Z_SCORE_THRESHOLD),
            step=0.1,
            help="Flags transactions whose log-amount deviates by more than this many "
                 "standard deviations from the mean.",
        )

    with st.expander("ℹ️ About the models"):
        st.markdown(
            """
            **Isolation Forest** partitions the feature space by randomly selecting features
            and split values. Anomalies require fewer splits to isolate, yielding lower scores.
            It is an unsupervised algorithm – it does not require labelled fraud data.

            **Z-score baseline** flags transactions whose log-transformed amount is statistically
            far from the dataset mean. This simple baseline helps compare against the ML approach.

            Neither model can definitively identify fraud. All flagged transactions should be
            reviewed by a qualified analyst.
            """
        )

    if st.button("▶ Run anomaly detection", type="primary"):
        with st.spinner("Running Isolation Forest and statistical baseline…"):
            feature_names = get_feature_names(processed_df)
            result_df, ifd, sbd = run_detection(
                processed_df,
                X,
                feature_names=feature_names,
                contamination=contamination,
                score_threshold=score_threshold,
                z_threshold=z_threshold,
            )
            st.session_state["result_df"] = result_df
            st.session_state["ifd"] = ifd
            st.session_state["detection_run"] = True

        n_flagged = int(result_df["is_anomaly"].sum())
        pct = 100 * n_flagged / max(len(result_df), 1)
        st.success(
            f"✅ Detection complete. **{n_flagged:,}** transactions flagged for review "
            f"({pct:.1f}% of {len(result_df):,} total)."
        )

        # KPI row
        avg_score = float(result_df["anomaly_score"].mean())
        fig_kpi = plot_kpi_cards(len(result_df), n_flagged, pct, avg_score)
        st.plotly_chart(fig_kpi, use_container_width=True)

    result_df = st.session_state.get("result_df")
    if result_df is not None and st.session_state["detection_run"]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_normal_vs_flagged(result_df), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_anomaly_scores(result_df), use_container_width=True)


# ===========================================================================
# SECTION 5 – FLAGGED TRANSACTIONS
# ===========================================================================
elif section == SECTIONS[4]:
    st.title("🚨 Flagged Transactions")

    result_df = st.session_state.get("result_df")
    if result_df is None or not st.session_state.get("detection_run"):
        st.warning("Please run anomaly detection first.")
        st.stop()

    flagged = result_df[result_df["is_anomaly"] == True].copy()

    if len(flagged) == 0:
        st.success("No transactions were flagged with the current settings.")
        st.stop()

    st.info(
        f"**{len(flagged):,}** transaction(s) flagged for review "
        f"({100*len(flagged)/max(len(result_df),1):.1f}% of total). "
        "Anomaly alerts do **not** confirm fraud — human review is required.",
        icon="⚠️",
    )

    # --- Filter controls ---------------------------------------------------
    st.markdown("#### Filter & search")
    col1, col2, col3 = st.columns(3)

    with col1:
        search_id = st.text_input("Search by Transaction ID", placeholder="TXN000001")
    with col2:
        if "merchant_category" in flagged.columns:
            cats = ["All"] + sorted(flagged["merchant_category"].unique().tolist())
            sel_cat = st.selectbox("Merchant category", cats)
        else:
            sel_cat = "All"
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["anomaly_score (lowest first)", "amount (highest first)", "timestamp (newest first)"],
        )

    filtered = flagged.copy()
    if search_id:
        filtered = filtered[filtered["transaction_id"].str.contains(search_id, case=False, na=False)]
    if sel_cat != "All":
        filtered = filtered[filtered["merchant_category"] == sel_cat]
    if "lowest" in sort_by:
        filtered = filtered.sort_values("anomaly_score", ascending=True)
    elif "highest" in sort_by:
        filtered = filtered.sort_values("amount", ascending=False)
    else:
        filtered = filtered.sort_values("timestamp", ascending=False)

    # --- Display columns ---------------------------------------------------
    display_cols = [
        c for c in [
            "transaction_id", "timestamp", "amount", "merchant_category",
            "transaction_type", "country_code", "card_present",
            "anomaly_score", "alert_status", "explanation",
        ]
        if c in filtered.columns
    ]

    st.markdown(f"Showing **{len(filtered):,}** result(s)")
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # --- Expandable detail cards -------------------------------------------
    st.markdown("---")
    st.markdown("#### Transaction detail")
    if len(filtered) > 0:
        selected_id = st.selectbox(
            "Select a transaction to inspect",
            options=filtered["transaction_id"].tolist(),
        )
        row = filtered[filtered["transaction_id"] == selected_id].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Amount", f"${row['amount']:,.2f}")
        c2.metric("Anomaly Score", f"{row['anomaly_score']:.4f}")
        c3.metric("Alert", row.get("alert_status", "—"))

        with st.container():
            st.markdown("**Why was this flagged?**")
            st.markdown(
                f"""
                <div class="alert-card">
                {row.get("explanation", "No explanation available.")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.json(
            {k: str(v) for k, v in row[display_cols].to_dict().items() if k != "explanation"},
            expanded=False,
        )


# ===========================================================================
# SECTION 6 – TRANSACTION ANALYSIS & VISUALISATIONS
# ===========================================================================
elif section == SECTIONS[5]:
    st.title("📊 Transaction Analysis & Visualisations")

    result_df = st.session_state.get("result_df")
    raw_df = st.session_state.get("raw_df")

    if result_df is None:
        st.warning("Please run anomaly detection first.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Amount Distribution",
        "Time Series",
        "Score vs Amount",
        "By Category",
        "Raw Statistics",
    ])

    with tab1:
        st.plotly_chart(plot_amount_distribution(result_df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_time_series(result_df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_score_vs_amount(result_df), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_flagged_by_category(result_df), use_container_width=True)

    with tab5:
        st.markdown("#### Summary statistics – all transactions")
        st.dataframe(
            result_df[["amount", "anomaly_score"]].describe().round(4),
            use_container_width=True,
        )

        if "is_anomaly" in result_df.columns:
            st.markdown("#### Summary statistics – flagged transactions")
            flagged_stats = result_df[result_df["is_anomaly"]][["amount", "anomaly_score"]]
            if len(flagged_stats) > 0:
                st.dataframe(flagged_stats.describe().round(4), use_container_width=True)

        st.markdown("#### Transaction type breakdown")
        if "transaction_type" in result_df.columns:
            type_counts = result_df.groupby(["transaction_type", "is_anomaly"]).size().unstack(fill_value=0)
            type_counts.columns = ["Normal", "Flagged"]
            st.dataframe(type_counts, use_container_width=True)


# ===========================================================================
# SECTION 7 – MODEL EVALUATION
# ===========================================================================
elif section == SECTIONS[6]:
    st.title("📈 Model Evaluation")

    result_df = st.session_state.get("result_df")
    if result_df is None or not st.session_state.get("detection_run"):
        st.warning("Please run anomaly detection first.")
        st.stop()

    with st.spinner("Computing evaluation metrics…"):
        eval_result = auto_evaluate(result_df)
        st.session_state["eval_result"] = eval_result

    if eval_result.has_labels:
        st.success("✅ Ground-truth labels found – running supervised evaluation.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (IF)", f"{eval_result.precision:.3f}" if eval_result.precision is not None else "—")
        col2.metric("Recall (IF)", f"{eval_result.recall:.3f}" if eval_result.recall is not None else "—")
        col3.metric("F1 (IF)", f"{eval_result.f1:.3f}" if eval_result.f1 is not None else "—")

        if eval_result.stat_f1 is not None:
            st.markdown("#### Baseline comparison")
            comp = pd.DataFrame(
                {
                    "Model": ["Isolation Forest", "Z-score Statistical"],
                    "Precision": [eval_result.precision, eval_result.stat_precision],
                    "Recall": [eval_result.recall, eval_result.stat_recall],
                    "F1-score": [eval_result.f1, eval_result.stat_f1],
                }
            )
            st.dataframe(comp.round(3), use_container_width=True, hide_index=True)

        if eval_result.confusion is not None:
            st.markdown("#### Confusion matrix")
            st.plotly_chart(
                plot_confusion_matrix(eval_result.confusion),
                use_container_width=True,
            )
            tn, fp, fn, tp = eval_result.confusion.ravel()
            cm_cols = st.columns(4)
            cm_cols[0].metric("True Negatives", tn)
            cm_cols[1].metric("False Positives (false alarms)", fp)
            cm_cols[2].metric("False Negatives (missed)", fn)
            cm_cols[3].metric("True Positives", tp)

        with st.expander("Full classification report"):
            if eval_result.classification_report_str:
                st.code(eval_result.classification_report_str, language="text")

    else:
        st.warning(
            "No ground-truth labels available. Showing unsupervised analysis only."
        )
        if eval_result.score_mean is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Score Mean", f"{eval_result.score_mean:.4f}")
            col2.metric("Score Std", f"{eval_result.score_std:.4f}")
            col3.metric("Flagged", f"{eval_result.flagged_count:,}")
            col4.metric("Flagged %", f"{eval_result.flagged_pct:.1f}%")

    with st.expander("Evaluation limitations"):
        st.markdown(
            """
            * **Unsupervised models** (like Isolation Forest) cannot be evaluated with standard
              classification metrics unless ground-truth labels are available.
            * The **contamination** parameter directly controls how many transactions are flagged,
              which inflates or deflates the apparent anomaly rate.
            * Precision and recall are computed only when `is_fraud` labels are present.
            * Results on the synthetic sample dataset are illustrative only and do not reflect
              real-world fraud detection performance.
            * Always validate model performance on labelled data representative of your
              production environment before deployment.
            """
        )


# ===========================================================================
# SECTION 8 – DOWNLOAD REPORTS
# ===========================================================================
elif section == SECTIONS[7]:
    st.title("⬇️ Download Reports")

    result_df = st.session_state.get("result_df")
    if result_df is None or not st.session_state.get("detection_run"):
        st.warning("Please run anomaly detection first.")
        st.stop()

    eval_result = st.session_state.get("eval_result")

    st.markdown("### Available exports")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Full analysis report")
        st.caption("All transactions with anomaly scores, flags, and explanations.")
        full_csv = generate_csv_report(result_df)
        st.download_button(
            "⬇️ Download full report (CSV)",
            data=full_csv,
            file_name="anomaly_detection_full_report.csv",
            mime="text/csv",
            type="primary",
        )

    with col2:
        st.markdown("#### Flagged transactions only")
        st.caption("Only transactions that were flagged for review.")
        try:
            flagged_csv = generate_flagged_csv(result_df)
            n_flagged = int(result_df["is_anomaly"].sum())
            st.download_button(
                f"⬇️ Download flagged transactions ({n_flagged:,} rows) (CSV)",
                data=flagged_csv,
                file_name="flagged_transactions.csv",
                mime="text/csv",
            )
        except ValueError as exc:
            st.error(str(exc))

    with col3:
        st.markdown("#### Text summary")
        st.caption("Human-readable analysis summary for reporting.")
        summary_txt = generate_summary_text(result_df, eval_result)
        st.download_button(
            "⬇️ Download summary (.txt)",
            data=summary_txt.encode("utf-8"),
            file_name="anomaly_detection_summary.txt",
            mime="text/plain",
        )

    st.markdown("---")
    st.markdown("#### Preview: text summary")
    st.code(generate_summary_text(result_df, eval_result), language="text")
