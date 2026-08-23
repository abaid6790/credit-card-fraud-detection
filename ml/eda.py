"""
Exploratory Data Analysis for the fraud dataset.

Every number here is computed from the actual DataFrame passed in — nothing
is hardcoded. Chart-building functions return Plotly figures (JSON-serializable)
so they can be reused by both the offline training report and the live Flask app.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def summary_stats(df: pd.DataFrame) -> dict:
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    total = len(df)

    return {
        "total_transactions": total,
        "fraud_transactions": len(fraud),
        "legitimate_transactions": len(legit),
        "fraud_percentage": round(len(fraud) / total * 100, 4) if total else 0.0,
        "amount_stats": {
            "overall": df["Amount"].describe().to_dict(),
            "fraud": fraud["Amount"].describe().to_dict(),
            "legitimate": legit["Amount"].describe().to_dict(),
        },
    }


def class_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Class"].value_counts().sort_index()
    labels = ["Legitimate" if c == 0 else "Fraud" for c in counts.index]
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=counts.values, hole=0.45,
                      marker=dict(colors=["#16A34A", "#DC2626"]))]
    )
    fig.update_layout(title="Class Distribution", template="plotly_white")
    return fig


def amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit, name="Legitimate", opacity=0.6,
                                marker_color="#16A34A", nbinsx=50))
    fig.add_trace(go.Histogram(x=fraud, name="Fraud", opacity=0.6,
                                marker_color="#DC2626", nbinsx=50))
    fig.update_layout(barmode="overlay", title="Transaction Amount Distribution",
                       xaxis_title="Amount", yaxis_title="Count", template="plotly_white")
    return fig


def fraud_over_time_chart(df: pd.DataFrame, bins: int = 48) -> go.Figure:
    fraud = df[df["Class"] == 1]
    fig = go.Figure(
        data=[go.Histogram(x=fraud["Time"], nbinsx=bins, marker_color="#DC2626")]
    )
    fig.update_layout(title="Fraud Frequency Over Time", xaxis_title="Time (seconds)",
                       yaxis_title="Fraud Count", template="plotly_white")
    return fig


def amount_by_class_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[df["Class"] == 0]["Amount"], name="Legitimate",
                          marker_color="#16A34A"))
    fig.add_trace(go.Box(y=df[df["Class"] == 1]["Amount"], name="Fraud",
                          marker_color="#DC2626"))
    fig.update_layout(title="Transaction Amount by Class", yaxis_title="Amount",
                       template="plotly_white")
    return fig
