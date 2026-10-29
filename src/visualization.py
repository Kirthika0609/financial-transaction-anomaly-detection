"""
visualization.py
----------------
Creates all charts used in the Streamlit dashboard.
Each function returns a Plotly Figure object so the app can call
st.plotly_chart() directly.

Public API
----------
plot_amount_distribution(df)
plot_normal_vs_flagged(df)
plot_anomaly_scores(df)
plot_confusion_matrix(cm, labels)
plot_flagged_by_category(df)
plot_time_series(df)
plot_score_vs_amount(df)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colour palette
NORMAL_COLOR = "#2196F3"
ANOMALY_COLOR = "#F44336"
BACKGROUND = "#0E1117"
GRID_COLOR = "#333333"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    xaxis=dict(gridcolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ---------------------------------------------------------------------------
# Amount distribution
# ---------------------------------------------------------------------------

def plot_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of transaction amounts, coloured by anomaly status."""
    color_col = "is_anomaly" if "is_anomaly" in df.columns else None
    color_map = {True: ANOMALY_COLOR, False: NORMAL_COLOR}

    fig = px.histogram(
        df,
        x="amount",
        color=color_col,
        color_discrete_map={True: ANOMALY_COLOR, False: NORMAL_COLOR},
        nbins=60,
        title="Transaction Amount Distribution",
        labels={"amount": "Amount (USD)", "is_anomaly": "Anomaly"},
        opacity=0.85,
    )
    fig.update_layout(**_LAYOUT, bargap=0.05)
    fig.update_xaxes(title_text="Amount (USD)")
    fig.update_yaxes(title_text="Count")
    return fig


# ---------------------------------------------------------------------------
# Normal vs flagged pie / bar
# ---------------------------------------------------------------------------

def plot_normal_vs_flagged(df: pd.DataFrame) -> go.Figure:
    """Donut chart comparing normal and flagged transaction counts."""
    if "is_anomaly" not in df.columns:
        return _empty_figure("Run detection first")

    counts = df["is_anomaly"].value_counts()
    labels = ["Normal", "Flagged (Review Required)"]
    values = [counts.get(False, 0), counts.get(True, 0)]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=[NORMAL_COLOR, ANOMALY_COLOR],
            textinfo="percent+value",
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title="Normal vs Flagged Transactions",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# Anomaly score strip plot
# ---------------------------------------------------------------------------

def plot_anomaly_scores(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of anomaly scores with flag threshold line."""
    if "anomaly_score" not in df.columns:
        return _empty_figure("Run detection first")

    fig = px.scatter(
        df,
        x=df.index,
        y="anomaly_score",
        color="is_anomaly",
        color_discrete_map={True: ANOMALY_COLOR, False: NORMAL_COLOR},
        title="Anomaly Score per Transaction",
        labels={
            "x": "Transaction Index",
            "anomaly_score": "Anomaly Score",
            "is_anomaly": "Flagged",
        },
        hover_data=["transaction_id", "amount"] if "transaction_id" in df.columns else None,
        opacity=0.7,
    )

    # Threshold line
    from config import ANOMALY_SCORE_THRESHOLD
    fig.add_hline(
        y=ANOMALY_SCORE_THRESHOLD,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({ANOMALY_SCORE_THRESHOLD})",
        annotation_position="top right",
    )
    fig.update_layout(**_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[list[str]] = None,
) -> go.Figure:
    """Heatmap of a 2×2 confusion matrix."""
    labels = labels or ["Normal", "Anomaly"]
    z = cm.tolist()
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(val),
                    font=dict(color="white", size=18),
                    showarrow=False,
                )
            )
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#1A237E"], [0.5, "#1976D2"], [1, ANOMALY_COLOR]],
            showscale=False,
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Flagged transactions by merchant category
# ---------------------------------------------------------------------------

def plot_flagged_by_category(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: number of flagged transactions per category."""
    if "is_anomaly" not in df.columns or "merchant_category" not in df.columns:
        return _empty_figure("Requires anomaly detection results")

    flagged = df[df["is_anomaly"] == True]
    counts = (
        flagged.groupby("merchant_category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )

    fig = px.bar(
        counts,
        x="count",
        y="merchant_category",
        orientation="h",
        title="Flagged Transactions by Merchant Category",
        labels={"count": "Number of Flagged Transactions", "merchant_category": "Category"},
        color_discrete_sequence=[ANOMALY_COLOR],
    )
    fig.update_layout(**_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Time-series of flagged transactions
# ---------------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame) -> go.Figure:
    """Daily transaction volume with flagged overlay."""
    if "timestamp" not in df.columns:
        return _empty_figure("timestamp column required")

    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date

    daily_all = tmp.groupby("date").size().reset_index(name="total")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_all["date"],
            y=daily_all["total"],
            name="All Transactions",
            line=dict(color=NORMAL_COLOR),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.15)",
        )
    )

    if "is_anomaly" in df.columns:
        daily_flagged = (
            tmp[tmp["is_anomaly"] == True]
            .groupby("date")
            .size()
            .reset_index(name="flagged")
        )
        fig.add_trace(
            go.Scatter(
                x=daily_flagged["date"],
                y=daily_flagged["flagged"],
                name="Flagged",
                mode="markers+lines",
                marker=dict(color=ANOMALY_COLOR, size=6),
                line=dict(color=ANOMALY_COLOR, dash="dot"),
            )
        )

    fig.update_layout(**_LAYOUT, title="Daily Transaction Volume", xaxis_title="Date", yaxis_title="Count")
    return fig


# ---------------------------------------------------------------------------
# Score vs amount scatter
# ---------------------------------------------------------------------------

def plot_score_vs_amount(df: pd.DataFrame) -> go.Figure:
    """Scatter: anomaly score vs transaction amount."""
    if "anomaly_score" not in df.columns or "amount" not in df.columns:
        return _empty_figure("Run detection first")

    fig = px.scatter(
        df,
        x="amount",
        y="anomaly_score",
        color="is_anomaly",
        color_discrete_map={True: ANOMALY_COLOR, False: NORMAL_COLOR},
        title="Anomaly Score vs Transaction Amount",
        labels={"amount": "Amount (USD)", "anomaly_score": "Anomaly Score", "is_anomaly": "Flagged"},
        opacity=0.65,
        hover_data=["transaction_id"] if "transaction_id" in df.columns else None,
        log_x=True,
    )
    fig.update_layout(**_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Summary metrics figure (4 KPI cards)
# ---------------------------------------------------------------------------

def plot_kpi_cards(
    total: int,
    flagged: int,
    flagged_pct: float,
    avg_score: float,
) -> go.Figure:
    """Simple indicator panel for the dashboard header."""
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total,
            title={"text": "Total Transactions"},
            domain={"x": [0, 0.25], "y": [0, 1]},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=flagged,
            title={"text": "Flagged"},
            number={"font": {"color": ANOMALY_COLOR}},
            domain={"x": [0.25, 0.5], "y": [0, 1]},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=round(flagged_pct, 1),
            number={"suffix": "%"},
            title={"text": "Flagged %"},
            domain={"x": [0.5, 0.75], "y": [0, 1]},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=round(avg_score, 4),
            title={"text": "Avg Anomaly Score"},
            domain={"x": [0.75, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(**_LAYOUT, height=150)
    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT,
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#AAAAAA"),
            )
        ],
    )
    return fig
