"""Plotly visualization functions for AFlow optimization results."""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from utils.config import (
    COLORS,
    EVE_DIMENSION_COLORS,
    EVE_DIMENSIONS,
    LEGEND_BELOW,
    PLOT_HEIGHT,
)


def create_score_progression_plot(
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    source: str = "val",
) -> go.Figure:
    """Line chart of scores across MCTS rounds with running maximum overlay."""
    fig = go.Figure()

    if not val_df.empty:
        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=val_df["score"] * 100,
                mode="lines+markers",
                name="Round Score",
                line=dict(color=COLORS["validation"], width=2),
                marker=dict(size=8),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

        # Running maximum overlay
        running_max = val_df["score"].cummax() * 100
        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=running_max,
                mode="lines",
                name="Running Max",
                line=dict(color=COLORS["success"], width=2, dash="dash"),
                hovertemplate=(
                    "<b>Round %{x}</b><br>Best so far: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    if test_df is not None and not test_df.empty:
        fig.add_trace(
            go.Scatter(
                x=test_df["round"],
                y=test_df["score"] * 100,
                mode="lines+markers",
                name="Test",
                line=dict(color=COLORS["test"], width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

    split_label = source.upper()
    fig.update_layout(
        title="Score Progression",
        xaxis_title="MCTS Round",
        yaxis_title=f"Score — {split_label} (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=LEGEND_BELOW,
        xaxis=dict(dtick=1),
        margin=dict(b=80),
    )
    return fig


def create_running_max_plot(val_df: pd.DataFrame, source: str = "val") -> go.Figure:
    """Running maximum score tracker."""
    fig = go.Figure()

    if not val_df.empty:
        running_max = val_df["score"].cummax() * 100

        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=running_max,
                mode="lines",
                name="Best So Far",
                fill="tozeroy",
                line=dict(color=COLORS["success"], width=2),
                hovertemplate=("<b>Round %{x}</b><br>Best: %{y:.1f}%<extra></extra>"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=val_df["round"],
                y=val_df["score"] * 100,
                mode="markers",
                name="Round Score",
                marker=dict(color=COLORS["secondary"], size=8),
                hovertemplate=("<b>Round %{x}</b><br>Score: %{y:.1f}%<extra></extra>"),
            )
        )

    split_label = source.upper()
    fig.update_layout(
        title="Running Maximum",
        xaxis_title="MCTS Round",
        yaxis_title=f"Score — {split_label} (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=LEGEND_BELOW,
        xaxis=dict(dtick=1),
        margin=dict(b=80),
    )
    return fig


def create_cost_progression_plot(val_df: pd.DataFrame) -> go.Figure:
    """Bar chart of total cost per round (single split)."""
    fig = go.Figure()

    if not val_df.empty and "total_cost" in val_df.columns:
        fig.add_trace(
            go.Bar(
                x=val_df["round"],
                y=val_df["total_cost"],
                name="Round Cost",
                marker_color=COLORS["secondary"],
                hovertemplate="<b>Round %{x}</b><br>Cost: $%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Cost per Round",
        xaxis_title="MCTS Round",
        yaxis_title="Cost (USD)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        xaxis=dict(dtick=1),
    )
    return fig


def create_cost_all_splits_plot(
    train_df: pd.DataFrame,
    dev_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Grouped bar chart of total cost per round across all partitions."""
    fig = go.Figure()

    for label, df, color in [
        ("Train", train_df, COLORS["train"]),
        ("Dev", dev_df, COLORS["dev"]),
        ("Test", test_df, COLORS["test"]),
    ]:
        if df is not None and not df.empty and "total_cost" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["round"],
                    y=df["total_cost"],
                    name=label,
                    marker_color=color,
                    hovertemplate=f"<b>Round %{{x}}</b><br>{label}: $%{{y:.4f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Cost per Round (All Partitions)",
        xaxis_title="MCTS Round",
        yaxis_title="Cost (USD)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        barmode="stack",
        xaxis=dict(dtick=1),
        legend=LEGEND_BELOW,
        margin=dict(b=80),
    )
    return fig


def create_eve_dimension_progression(
    df: pd.DataFrame,
    source: str = "val",
    baseline: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Multi-line chart of per-dimension scores across MCTS rounds."""
    fig = go.Figure()

    if df.empty:
        return fig

    # Prepend Round 0 (Eve app baseline) if provided
    if baseline:
        r0: Dict[str, object] = {"round": 0}
        for dim in EVE_DIMENSIONS:
            if dim in baseline:
                r0[dim] = baseline[dim]
        if "score" in baseline:
            r0["score"] = baseline["score"]
        df = pd.concat([pd.DataFrame([r0]), df], ignore_index=True)

    # Plot each dimension
    for dim in EVE_DIMENSIONS:
        if dim not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["round"],
                y=df[dim] * 100,
                mode="lines+markers",
                name=dim.capitalize(),
                line=dict(color=EVE_DIMENSION_COLORS[dim], width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>Round %{{x}}</b><br>{dim.capitalize()}: %{{y:.1f}}%<extra></extra>",
            )
        )

    # Overall score — thicker dashed line
    if "score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["round"],
                y=df["score"] * 100,
                mode="lines+markers",
                name="Overall",
                line=dict(color=EVE_DIMENSION_COLORS["score"], width=3, dash="dash"),
                marker=dict(size=8),
                hovertemplate="<b>Round %{x}</b><br>Overall: %{y:.1f}%<extra></extra>",
            )
        )

    split_label = source.upper()
    fig.update_layout(
        title="Dimension Progression",
        xaxis_title="MCTS Round",
        yaxis_title=f"Score — {split_label} (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=LEGEND_BELOW,
        xaxis=dict(dtick=1),
        margin=dict(b=80),
    )
    return fig


def create_eve_radar_chart(
    baseline_scores: Dict[str, float],
    best_scores: Dict[str, float],
) -> go.Figure:
    """Radar chart comparing Eve baseline vs AFlow best round on 4 dimensions."""
    dimensions = EVE_DIMENSIONS
    labels = [d.capitalize() for d in dimensions]

    baseline_vals = [baseline_scores.get(d, 0) * 100 for d in dimensions]
    best_vals = [best_scores.get(d, 0) * 100 for d in dimensions]

    # Close the polygon
    baseline_vals.append(baseline_vals[0])
    best_vals.append(best_vals[0])
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=baseline_vals,
            theta=labels_closed,
            fill="toself",
            name="Eve Baseline",
            line=dict(color=COLORS["secondary"], dash="dash"),
            opacity=0.5,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=best_vals,
            theta=labels_closed,
            fill="toself",
            name="AFlow Best",
            line=dict(color=COLORS["primary"]),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="Baseline vs Best Round",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_white",
        height=PLOT_HEIGHT,
        legend=LEGEND_BELOW,
    )
    return fig


def create_eve_dimension_comparison_bar(
    baseline_scores: Dict[str, float],
    best_scores: Dict[str, float],
) -> go.Figure:
    """Grouped bar chart comparing baseline vs best across 4 dimensions."""
    dimensions = EVE_DIMENSIONS
    labels = [d.capitalize() for d in dimensions]

    baseline_vals = [baseline_scores.get(d, 0) * 100 for d in dimensions]
    best_vals = [best_scores.get(d, 0) * 100 for d in dimensions]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=baseline_vals,
            name="Eve App (GPT-4.1)",
            marker_color=COLORS["secondary"],
            hovertemplate="<b>%{x}</b><br>Eve App: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=best_vals,
            name="AFlow Best (GPT-4.1)",
            marker_color=COLORS["primary"],
            hovertemplate="<b>%{x}</b><br>AFlow Best: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Eve App (GPT-4.1) vs AFlow Best (GPT-4.1)",
        yaxis_title="Score (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        barmode="group",
        legend=LEGEND_BELOW,
        margin=dict(b=80),
    )
    return fig


def create_val_vs_test_comparison(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> go.Figure:
    """Grouped bar chart comparing validation vs test scores (legacy 2-split)."""
    return create_split_comparison(val_df, None, test_df)


def create_split_comparison(
    train_df: pd.DataFrame,
    dev_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    baseline_score: Optional[float] = None,
) -> go.Figure:
    """Grouped bar chart comparing train/dev/test scores across rounds."""
    # Prepend Round 0 (Eve app baseline) to each split
    if baseline_score is not None:
        r0_row = pd.DataFrame([{"round": 0, "score": baseline_score}])
        if train_df is not None and not train_df.empty:
            train_df = pd.concat([r0_row, train_df], ignore_index=True)
        if dev_df is not None and not dev_df.empty:
            dev_df = pd.concat([r0_row, dev_df], ignore_index=True)
        if test_df is not None and not test_df.empty:
            test_df = pd.concat([r0_row, test_df], ignore_index=True)

    # Find rounds common to all available splits
    common_rounds = set(train_df["round"].tolist())
    if dev_df is not None and not dev_df.empty:
        common_rounds &= set(dev_df["round"].tolist())
    if test_df is not None and not test_df.empty:
        common_rounds &= set(test_df["round"].tolist())
    common_rounds = sorted(common_rounds)

    round_labels = [f"Round {r}" for r in common_rounds]

    fig = go.Figure()

    # Train
    train_scores = [
        train_df.loc[train_df["round"] == r, "score"].iloc[0] * 100
        for r in common_rounds
    ]
    fig.add_trace(
        go.Bar(
            x=round_labels,
            y=train_scores,
            name="Train",
            marker_color=COLORS["train"],
            hovertemplate="<b>%{x}</b><br>Train: %{y:.1f}%<extra></extra>",
        )
    )

    # Dev
    if dev_df is not None and not dev_df.empty:
        dev_scores = [
            dev_df.loc[dev_df["round"] == r, "score"].iloc[0] * 100
            for r in common_rounds
        ]
        fig.add_trace(
            go.Bar(
                x=round_labels,
                y=dev_scores,
                name="Dev",
                marker_color=COLORS["dev"],
                hovertemplate="<b>%{x}</b><br>Dev: %{y:.1f}%<extra></extra>",
            )
        )

    # Test
    if test_df is not None and not test_df.empty:
        test_scores = [
            test_df.loc[test_df["round"] == r, "score"].iloc[0] * 100
            for r in common_rounds
        ]
        fig.add_trace(
            go.Bar(
                x=round_labels,
                y=test_scores,
                name="Test",
                marker_color=COLORS["test"],
                hovertemplate="<b>%{x}</b><br>Test: %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Train / Dev / Test Scores",
        yaxis_title="Score (%)",
        template="plotly_white",
        height=PLOT_HEIGHT,
        barmode="group",
        legend=LEGEND_BELOW,
        margin=dict(b=80),
    )
    return fig
