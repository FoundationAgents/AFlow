"""Eve AICW optimization dashboard page."""

import pandas as pd
import streamlit as st

from components.mcts_tree import display_mcts_tree
from components.sidebar import render_sidebar
from components.workflow_diagram import display_workflow_diagram_single
from components.workflow_viewer import display_workflow_code_single
from data_loader import AFlowDataLoader
from plots import (
    create_cost_progression_plot,
    create_eve_dimension_comparison_bar,
    create_eve_dimension_progression,
    create_eve_radar_chart,
)
from utils.config import EVE_BASELINE, EVE_DIMENSIONS, WORKSPACE_DIR

st.set_page_config(
    page_title="Eve Optimization - AFlow",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_data_loader():
    return AFlowDataLoader(WORKSPACE_DIR)


loader = get_data_loader()
sidebar_state = render_sidebar(loader)
dataset = sidebar_state["dataset"]
run_df = sidebar_state["selected_run_df"]

# Guard: only show Eve content
if not dataset.startswith("Eve"):
    st.info("Select an **Eve** dataset from the sidebar to view this page.")
    st.stop()

st.title("Eve AICW Optimization")
st.markdown("*LLM-as-Judge evaluation across 4 behavioral dimensions*")

# Load data
if run_df is not None and not run_df.empty:
    display_df = run_df
else:
    display_df = loader.load_validation_results(dataset)

if display_df.empty:
    st.warning(f"No results found for {dataset}.")
    st.stop()

source = display_df["source"].iloc[0] if "source" in display_df.columns else "val"
run_config = loader.load_run_config(dataset, split=source)

# Load per-dimension data from CSVs
dim_df = loader.load_eve_dimension_summary(dataset)
has_dimensions = not dim_df.empty and any(d in dim_df.columns for d in EVE_DIMENSIONS)

# --- Section 1: Metrics Cards ---
best_score = display_df["score"].max()
baseline_score = display_df.iloc[0]["score"]
improvement = best_score - baseline_score
best_round = int(display_df.loc[display_df["score"].idxmax(), "round"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Score", f"{best_score:.1%}")
c2.metric("Baseline (R1)", f"{baseline_score:.1%}")
c3.metric(
    "Improvement",
    f"+{improvement:.1%}",
    delta=(
        f"{improvement / baseline_score:.0%} relative" if baseline_score > 0 else None
    ),
)
c4.metric("Rounds", len(display_df))

# Per-dimension metrics (from CSVs)
if has_dimensions:
    best_round_dims = dim_df.loc[dim_df["round"] == best_round]
    if best_round_dims.empty and not dim_df.empty:
        # Fall back to the round with the highest overall score in dim_df
        best_dim_idx = dim_df["score"].idxmax() if "score" in dim_df.columns else 0
        best_round_dims = dim_df.iloc[[best_dim_idx]]

    if not best_round_dims.empty:
        cols = st.columns(4)
        for col, dim in zip(cols, EVE_DIMENSIONS):
            if dim in best_round_dims.columns:
                val = best_round_dims.iloc[0][dim]
                baseline_val = EVE_BASELINE.get(dim, 0)
                delta = val - baseline_val
                col.metric(
                    dim.capitalize(),
                    f"{val:.1%}",
                    delta=f"{delta:+.1%} vs baseline",
                    delta_color="normal",
                )

st.divider()

# --- Section 2: Charts ---
if has_dimensions:
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_eve_dimension_progression(dim_df, source=source)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Build best_scores dict from the best round's dimension data
        best_scores = {}
        best_round_dims = dim_df.loc[dim_df["round"] == best_round]
        if best_round_dims.empty and not dim_df.empty:
            best_dim_idx = dim_df["score"].idxmax() if "score" in dim_df.columns else 0
            best_round_dims = dim_df.iloc[[best_dim_idx]]
        if not best_round_dims.empty:
            for dim in EVE_DIMENSIONS:
                if dim in best_round_dims.columns:
                    best_scores[dim] = best_round_dims.iloc[0][dim]
        fig = create_eve_radar_chart(EVE_BASELINE, best_scores)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = create_cost_progression_plot(display_df)
        st.plotly_chart(fig, use_container_width=True)
else:
    # Fallback: show overall score progression + cost only
    from plots import create_score_progression_plot

    col1, col2 = st.columns(2)
    with col1:
        fig = create_score_progression_plot(display_df, source=source)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_cost_progression_plot(display_df)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Section 3: MCTS Tree ---
tree_data = loader.load_mcts_tree(dataset)
if tree_data:
    st.subheader("MCTS Search Tree")
    experiences = {}
    for r in loader.get_available_rounds(dataset):
        exp = loader.load_round_experience(dataset, r)
        if exp:
            experiences[r] = exp
    display_mcts_tree(tree_data, experiences)
    st.divider()

# --- Section 4: Mutation Log ---
available_rounds = loader.get_available_rounds(dataset)
mutation_rows = []
for r in available_rounds:
    exp = loader.load_round_experience(dataset, r)
    if exp:
        before = exp.get("before", 0)
        after = exp.get("after", 0)
        mutation_rows.append(
            {
                "Round": r,
                "Parent": exp.get("father node", exp.get("father_node", "?")),
                "Label": exp.get("short_label", ""),
                "Modification": exp.get("modification", ""),
                "Before": f"{before:.1%}",
                "After": f"{after:.1%}",
                "Delta": f"{after - before:+.1%}",
                "Status": "Improved" if exp.get("succeed") else "Regressed",
            }
        )

if mutation_rows:
    st.subheader("Mutation Log")
    mutation_df = pd.DataFrame(mutation_rows)
    st.dataframe(
        mutation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Modification": st.column_config.TextColumn(width="large"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    st.divider()

# --- Section 5: Comparison View ---
if has_dimensions:
    st.subheader("Baseline vs Best Comparison")

    # Dimension comparison bar chart
    best_scores = {}
    best_round_dims = dim_df.loc[dim_df["round"] == best_round]
    if best_round_dims.empty and not dim_df.empty:
        best_dim_idx = dim_df["score"].idxmax() if "score" in dim_df.columns else 0
        best_round_dims = dim_df.iloc[[best_dim_idx]]
    if not best_round_dims.empty:
        for dim in EVE_DIMENSIONS:
            if dim in best_round_dims.columns:
                best_scores[dim] = best_round_dims.iloc[0][dim]

    fig = create_eve_dimension_comparison_bar(EVE_BASELINE, best_scores)
    st.plotly_chart(fig, use_container_width=True)

    # Sample-level comparison table (if CSV data available for best round)
    best_csv = loader.load_round_csv(dataset, best_round)
    if best_csv is not None and "question" in best_csv.columns:
        st.markdown(f"**Per-sample results (Round {best_round})**")
        display_cols = ["question"]
        if "prediction" in best_csv.columns:
            display_cols.append("prediction")
        for dim in EVE_DIMENSIONS:
            if dim in best_csv.columns:
                display_cols.append(dim)
        if "score" in best_csv.columns:
            display_cols.append("score")

        # Truncate long text for display
        display_csv = best_csv[display_cols].copy()
        for col in ["question", "prediction"]:
            if col in display_csv.columns:
                display_csv[col] = display_csv[col].astype(str).str[:200]

        st.dataframe(display_csv, use_container_width=True, hide_index=True)

    st.divider()

# --- Section 6: Workflow Evolution ---
run_rounds = sorted(display_df["round"].unique().astype(int).tolist())
wf_rounds = [r for r in run_rounds if r in available_rounds]

if wf_rounds:
    st.subheader("Workflow Evolution")
    wf_tabs = st.tabs([f"Round {r}" for r in wf_rounds])
    for wf_tab, r in zip(wf_tabs, wf_rounds):
        with wf_tab:
            exp = loader.load_round_experience(dataset, r)
            if exp:
                status = "Improved" if exp.get("succeed") else "Regressed"
                delta = exp.get("after", 0) - exp.get("before", 0)
                sign = "+" if delta >= 0 else ""
                st.info(
                    f"**From Round {exp.get('father node', exp.get('father_node', '?'))}** | "
                    f"{status} ({sign}{delta:.1%}) | "
                    f"{exp.get('modification', '')}"
                )

            col_diagram, col_code = st.columns([1, 2])
            with col_diagram:
                st.markdown("**Structure**")
                display_workflow_diagram_single(loader, dataset, r)
            with col_code:
                st.markdown("**Code**")
                display_workflow_code_single(loader, dataset, r)
