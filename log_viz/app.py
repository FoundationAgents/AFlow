"""Main Streamlit dashboard for AFlow optimization visualization."""

import time

import streamlit as st

from components.mcts_tree import display_mcts_tree
from components.metrics_cards import display_metrics_cards
from components.predictions_table import display_predictions

from components.sidebar import render_sidebar
from components.workflow_diagram import display_workflow_diagram_single
from components.workflow_viewer import display_workflow_code_single
from data_loader import AFlowDataLoader
from plots import (
    create_cost_progression_plot,
    create_running_max_plot,
    create_score_progression_plot,
)
from utils.config import WORKSPACE_DIR

# Page config
st.set_page_config(
    page_title="AFlow Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_data_loader():
    return AFlowDataLoader(WORKSPACE_DIR)


loader = get_data_loader()
sidebar_state = render_sidebar(loader)
dataset = sidebar_state["dataset"]
selected_rounds = sidebar_state["selected_rounds"]
run_df = sidebar_state["selected_run_df"]
auto_refresh = sidebar_state["auto_refresh"]
refresh_interval = sidebar_state["refresh_interval"]

# Title
st.title("AFlow Optimization Dashboard")
st.markdown("*MCTS-based agentic workflow optimization*")
st.caption(
    "AFlow: Automating Agentic Workflow Generation "
    "([Zhang et al., ICLR 2025](https://arxiv.org/abs/2410.10762))"
)

# Use selected run's data for metrics/charts, fall back to val_df
if run_df is not None and not run_df.empty:
    display_df = run_df
else:
    display_df = loader.load_validation_results(dataset)

if display_df.empty:
    st.warning(f"No results found for {dataset}.")
    st.stop()

# Load supplementary data
tree_data = loader.load_mcts_tree(dataset)
source = display_df["source"].iloc[0] if "source" in display_df.columns else "val"
run_config = loader.load_run_config(dataset, split=source)

# --- Metrics Cards ---
display_metrics_cards(display_df, run_config=run_config)

st.divider()

# --- Charts ---
col1, col2, col3 = st.columns(3)
with col1:
    fig = create_score_progression_plot(display_df, source=source)
    st.plotly_chart(fig, width="stretch")
with col2:
    fig = create_running_max_plot(display_df, source=source)
    st.plotly_chart(fig, width="stretch")
with col3:
    fig = create_cost_progression_plot(display_df)
    st.plotly_chart(fig, width="stretch")

st.divider()

# --- MCTS Tree (only for val runs with tree data) ---
if tree_data:
    st.subheader("Monte Carlo Tree Search (MCTS)")
    st.markdown(
        "Each node is a `Workflow` class — a Python graph of **operators** "
        "(e.g. `AnswerGenerate`, `ScEnsemble`, `Custom`) wired together with loops and conditionals. "
        "Every round, the optimizer selects a parent node and asks an LLM to make **one modification** "
        "to its code: add an operator, delete one, change a prompt, or restructure control flow. "
        "The child is evaluated on a validation set; a **green edge** means it improved, **red** means it regressed.\n\n"
        "**Scoring:** Each workflow is run on every sample in the validation dataset (e.g. 200 HotpotQA questions). "
        "For each sample, the workflow's output is compared to the ground-truth answer using a dataset-specific "
        "metric (F1 score for HotpotQA, exact match for GSM8K, pass rate for code benchmarks). "
        "The node's score is the **average metric across all samples**.\n\n"
        "Node selection uses a mixed probability: **70% exploitation** (softmax over scores — higher-scoring "
        "workflows are more likely to be picked for refinement) and **30% exploration** (uniform — any node "
        "can be selected, even low-scoring ones, to try a different modification direction). "
        "The tree **deepens** when the optimizer keeps refining a strong branch (e.g. adding a review step "
        "to an already-good ensemble workflow). It **widens** when it backtracks to a different ancestor "
        "and tries an alternative structural change. Failed modifications are tracked in "
        "`processed_experience.json` so the same change is never attempted twice on the same parent."
    )
    experiences = {}
    for r in loader.get_available_rounds(dataset):
        exp = loader.load_round_experience(dataset, r)
        if exp:
            experiences[r] = exp
    display_mcts_tree(tree_data, experiences)
    st.divider()

# --- Workflow Evolution (Diagram + Code side by side) ---
# Use rounds from the selected run if available
run_rounds = sorted(display_df["round"].unique().astype(int).tolist())
# Only show workflow evolution for rounds that have graph.py files
available_rounds = loader.get_available_rounds(dataset)
wf_rounds = [r for r in run_rounds if r in available_rounds]

if wf_rounds:
    st.subheader("Workflow Evolution")
    wf_tabs = st.tabs([f"Round {r}" for r in wf_rounds])
    for wf_tab, r in zip(wf_tabs, wf_rounds):
        with wf_tab:
            # Show modification info
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

    st.divider()

# --- Prediction Details ---
st.subheader("Prediction Details (Incorrect)")
display_predictions(loader, dataset, wf_rounds if wf_rounds else selected_rounds)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
