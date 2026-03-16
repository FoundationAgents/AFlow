"""Summary metric cards for the AFlow dashboard."""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st


def _derive_run_id(df: pd.DataFrame) -> str:
    """Derive a yyyymmdd-hhmm run ID from the first round's timestamp."""
    if "time" not in df.columns:
        return "N/A"
    first_time = df.iloc[0]["time"]
    try:
        if isinstance(first_time, pd.Timestamp):
            return first_time.strftime("%Y%m%d-%H%M")
        dt = datetime.fromisoformat(str(first_time))
        return dt.strftime("%Y%m%d-%H%M")
    except (ValueError, TypeError):
        return "N/A"


def _compute_duration(run_config: Dict) -> str:
    """Compute human-readable duration from run config timestamps."""
    started = run_config.get("started_at")
    completed = run_config.get("completed_at")
    if not started or not completed:
        return "N/A"
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(completed)
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {seconds}s"
    except (ValueError, TypeError):
        return "N/A"


def display_metrics_cards(
    df: pd.DataFrame,
    run_config: Optional[Dict] = None,
    baseline_score_override: Optional[float] = None,
) -> None:
    """Display summary metrics in card format."""
    if df.empty:
        st.warning("No results available.")
        return

    best_score = df["score"].max()
    baseline_score = (
        baseline_score_override
        if baseline_score_override is not None
        else df.iloc[0]["score"]
    )
    baseline_label = "R0" if baseline_score_override is not None else "R1"
    best_round = int(df.loc[df["score"].idxmax(), "round"])
    improvement = best_score - baseline_score
    run_id = _derive_run_id(df)
    source = df["source"].iloc[0] if "source" in df.columns else "val"

    # Row 1: Core metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", run_id)
    c2.metric("Best Score", f"{best_score:.1%}")
    c3.metric(f"Baseline ({baseline_label})", f"{baseline_score:.1%}")
    c4.metric(
        "Improvement",
        f"+{improvement:.1%}",
        delta=(
            f"{improvement / baseline_score:.0%} relative"
            if baseline_score > 0
            else None
        ),
    )

    # Row 2: Additional metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Rounds", len(df))
    c6.metric("Best Round", f"Round {best_round}")

    # Total cost
    total_cost = df["total_cost"].sum() if "total_cost" in df.columns else 0
    c7.metric("Total Cost", f"${total_cost:.4f}")

    c8.metric("Split", source.upper())

    # Row 3: Run config info (if available)
    if run_config:
        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Opt Model", run_config.get("opt_model") or "N/A")
        c10.metric("Exec Model", run_config.get("exec_model", "N/A"))
        converged = run_config.get("converged")
        c11.metric(
            "Converged",
            "Yes" if converged else ("No" if converged is not None else "N/A"),
        )
        c12.metric("Duration", _compute_duration(run_config))
