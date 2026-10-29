"""
reporting.py
------------
Generates downloadable reports from detection results.

Public API
----------
generate_csv_report(df)         -> bytes  (UTF-8 CSV)
generate_flagged_csv(df)        -> bytes  (flagged transactions only)
generate_summary_text(df, eval_result) -> str
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPORT_COLUMNS
try:
    from src.evaluation import EvaluationResult
except ImportError:
    from evaluation import EvaluationResult


# ---------------------------------------------------------------------------
# CSV reports
# ---------------------------------------------------------------------------

def generate_csv_report(df: pd.DataFrame) -> bytes:
    """
    Full analysis report as CSV bytes.
    Includes all REPORT_COLUMNS that are present in df, then any extras.
    """
    cols_present = [c for c in REPORT_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in REPORT_COLUMNS]
    ordered = cols_present + extra_cols

    buf = io.StringIO()
    df[ordered].to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def generate_flagged_csv(df: pd.DataFrame) -> bytes:
    """Flagged transactions only."""
    if "is_anomaly" not in df.columns:
        raise ValueError("Run anomaly detection before exporting flagged transactions.")

    flagged = df[df["is_anomaly"] == True].copy()
    return generate_csv_report(flagged)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def generate_summary_text(
    df: pd.DataFrame,
    eval_result: EvaluationResult | None = None,
) -> str:
    """
    Human-readable text summary of the analysis run.
    Suitable for copying into an incident report.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    total = len(df)
    flagged = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    flagged_pct = 100.0 * flagged / max(total, 1)

    lines = [
        "=" * 60,
        "  FINANCIAL ANOMALY DETECTION — ANALYSIS SUMMARY",
        "=" * 60,
        f"  Generated : {now}",
        f"  Total transactions analysed : {total:,}",
        f"  Transactions flagged for review : {flagged:,} ({flagged_pct:.1f}%)",
        "",
        "DISCLAIMER",
        "-" * 40,
        "Anomaly alerts indicate unusual transaction patterns.",
        "They do NOT confirm fraud or any illegal activity.",
        "All flagged transactions require human analyst review.",
        "",
    ]

    if eval_result is not None:
        lines += ["MODEL EVALUATION", "-" * 40]
        lines += eval_result.summary_lines
        lines.append("")

    if "amount" in df.columns and flagged > 0:
        flagged_df = df[df["is_anomaly"] == True]
        lines += [
            "FLAGGED TRANSACTION STATISTICS",
            "-" * 40,
            f"  Amount range : ${flagged_df['amount'].min():,.2f} – ${flagged_df['amount'].max():,.2f}",
            f"  Average amount : ${flagged_df['amount'].mean():,.2f}",
            "",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)
