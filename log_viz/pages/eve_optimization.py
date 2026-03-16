"""Eve Persona Adherence Optimization dashboard page."""

import pandas as pd
import streamlit as st

from components.mcts_tree import display_mcts_tree
from components.sidebar import render_sidebar
from components.workflow_diagram import display_workflow_diagram_single
from components.workflow_viewer import display_workflow_code_single
from data_loader import AFlowDataLoader
from plots import (
    create_cost_all_splits_plot,
    create_cost_progression_plot,
    create_eve_dimension_comparison_bar,
    create_eve_dimension_progression,
    create_split_comparison,
)
from utils.config import EVE_BASELINE, EVE_DIMENSIONS, WORKSPACE_DIR


def _render_context_chat_bubbles(context: str) -> None:
    """Render a 'User: ... / Assistant: ...' context string as Streamlit chat bubbles.

    Handles multi-line assistant responses (numbered lists, paragraphs, etc.)
    by accumulating continuation lines until the next role prefix.
    """
    current_role = None
    current_lines: list[str] = []

    for line in context.split("\n"):
        stripped = line.strip()
        if stripped.startswith("User:"):
            # Flush previous message
            if current_role and current_lines:
                st.chat_message(current_role).write("\n".join(current_lines))
            current_role = "user"
            current_lines = [stripped[len("User:") :].strip()]
        elif stripped.startswith("Assistant:"):
            if current_role and current_lines:
                st.chat_message(current_role).write("\n".join(current_lines))
            current_role = "assistant"
            current_lines = [stripped[len("Assistant:") :].strip()]
        elif stripped:
            # Continuation line — append to current message
            current_lines.append(stripped)

    # Flush last message
    if current_role and current_lines:
        st.chat_message(current_role).write("\n".join(current_lines))


# Scoring criteria from benchmarks/eve.py DIMENSION_CRITERIA — kept in sync manually
DIMENSION_CRITERIA = {
    "verbosity": (
        "Evaluate whether the assistant's response length and detail level is appropriate "
        "for the user's query. The target is MEDIUM verbosity — the response should cover "
        "what's needed without unnecessary filler. Every sentence should earn its place. "
        "Score low if the response is extremely terse and omits critical information, or if "
        "it is excessively verbose with unnecessary padding. Score high if the detail level "
        "is well-calibrated to the complexity of the user's request."
    ),
    "tone_of_voice": (
        "Evaluate whether the assistant's response sounds warm, approachable, and professional. "
        "The target tone is FRIENDLY — like a helpful, approachable colleague. "
        "Score low if the response is cold, robotic, rude, or overly formal and distant. "
        "Score high if the response feels naturally warm and professional without being "
        "unprofessional or overly casual."
    ),
    "assertiveness": (
        "Evaluate whether the assistant provides clear guidance without being pushy or passive. "
        "The target is MEDIUM assertiveness — offering clear suggestions while respecting "
        "user autonomy. Score low if the response is completely passive (only mirrors the user) "
        "or overly hedging (qualifies everything, user unsure what to do). Also score low if "
        "the response is too directive or pushy. Score high if the response confidently guides "
        "without overstepping."
    ),
    "empathy": (
        "Evaluate whether the assistant acknowledges the user's situation and emotional context. "
        "The target is SUPPORTIVE empathy — validating feelings and adjusting the response "
        "accordingly. Score low if the response is dismissive or ignores emotional undertones. "
        "Score high if the response demonstrates genuine understanding and makes the user feel "
        "heard before jumping to solutions."
    ),
}

st.set_page_config(
    page_title="Eve Persona Adherence Optimization - AFlow",
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

st.title("Eve Persona Adherence Optimization")
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

# Load per-dimension data from CSVs — prefer test, fall back to train
test_dim_df = loader.load_eve_dimension_summary(dataset, split="test")
train_dim_df = loader.load_eve_dimension_summary(dataset, split="train")
if not test_dim_df.empty and any(d in test_dim_df.columns for d in EVE_DIMENSIONS):
    dim_df = test_dim_df
    dim_source = "Test"
elif not train_dim_df.empty and any(d in train_dim_df.columns for d in EVE_DIMENSIONS):
    dim_df = train_dim_df
    dim_source = "Train"
else:
    dim_df = pd.DataFrame()
    dim_source = "Train"
has_dimensions = not dim_df.empty and any(d in dim_df.columns for d in EVE_DIMENSIONS)

# --- Section 1: Metrics Cards (from Test partition) ---
test_headline_df = loader.load_test_results(dataset)
has_test_headline = test_headline_df is not None and not test_headline_df.empty

if has_test_headline:
    headline_df = test_headline_df
    headline_label = "Test"
else:
    headline_df = display_df
    headline_label = "Train"

best_score = headline_df["score"].max()
baseline_score = headline_df.iloc[0]["score"]
improvement = best_score - baseline_score
best_round = int(headline_df.loc[headline_df["score"].idxmax(), "round"])

st.caption(f"Headline metrics from **{headline_label}** partition")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Best Score ({headline_label})", f"{best_score:.1%}")
c2.metric(f"Baseline R1 ({headline_label})", f"{baseline_score:.1%}")
c3.metric(
    "Improvement",
    f"+{improvement:.1%}",
    delta=(
        f"{improvement / baseline_score:.0%} relative" if baseline_score > 0 else None
    ),
)
c4.metric("Best Round", f"R{best_round}")

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

# Load all split DataFrames (used by Section 2 cost chart and Section 3)
train_df = loader.load_validation_results(dataset)
dev_df = loader.load_dev_results(dataset)
test_df = loader.load_test_results(dataset)

# --- Section 2: Charts ---
# Determine which data to chart — prefer test, fall back to train
chart_source = headline_label  # "Test" or "Train"

if has_dimensions:
    st.caption(f"Dimension charts from **{dim_source}** partition")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_eve_dimension_progression(
            dim_df, source=dim_source.lower(), baseline=EVE_BASELINE
        )
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
        fig = create_eve_dimension_comparison_bar(EVE_BASELINE, best_scores)
        st.plotly_chart(fig, use_container_width=True, key="dim_comparison_section2")
    with col3:
        fig = create_cost_all_splits_plot(train_df, dev_df, test_df)
        st.plotly_chart(fig, use_container_width=True)
else:
    # Fallback: show overall score progression + cost only
    from plots import create_score_progression_plot

    st.caption(f"Charts from **{chart_source}** partition")
    col1, col2 = st.columns(2)
    with col1:
        fig = create_score_progression_plot(headline_df, source=chart_source.lower())
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_cost_all_splits_plot(train_df, dev_df, test_df)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Section 3: Train / Dev / Test Generalization ---

has_dev = dev_df is not None and not dev_df.empty
has_test = test_df is not None and not test_df.empty

if has_dev or has_test:
    st.subheader("Train / Dev / Test Generalization")

    # Grouped bar chart
    fig = create_split_comparison(
        train_df, dev_df, test_df, baseline_score=EVE_BASELINE.get("score")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

# --- Section 3b: Dataset Generation Details ---
st.subheader("Dataset Generation Details")
st.markdown(
    "Samples were generated using **DeepEval Synthesizer** "
    "(`generate_conversational_goldens_from_scratch`) to produce dimension-targeted "
    "scenarios, then converted into realistic user messages via a second **GPT-4.1** call. "
    "Each partition is balanced across 4 persona dimensions "
    "(verbosity, tone of voice, assertiveness, empathy).\n\n"
    "For multi-turn samples, **user messages are synthetic** (from DeepEval) but "
    "**assistant responses in the conversation history are from the real Eve application** "
    "(replayed via WebSocket). The final user message is sent to the workflow for response "
    "generation and scoring."
)

# Turn distribution table
turn_dist = loader.get_turn_distribution(dataset)
if turn_dist:
    all_turns = sorted({t for counts in turn_dist.values() for t in counts})
    header = "| Partition | Total |" + " | ".join(f"{t}-turn" for t in all_turns) + " |"
    sep = "|-----------|-------|" + " | ".join("-----" for _ in all_turns) + " |"
    tbl_rows = []
    for split_name in ["train", "dev", "test"]:
        if split_name not in turn_dist:
            continue
        counts = turn_dist[split_name]
        total = sum(counts.values())
        cells = [f"{counts.get(t, 0)}" for t in all_turns]
        tbl_rows.append(
            f"| {split_name.capitalize()} | {total} | " + " | ".join(cells) + " |"
        )
    totals = {t: sum(turn_dist[s].get(t, 0) for s in turn_dist) for t in all_turns}
    grand_total = sum(totals.values())
    total_cells = [f"**{totals[t]}**" for t in all_turns]
    tbl_rows.append(
        f"| **Total** | **{grand_total}** | " + " | ".join(total_cells) + " |"
    )
    st.markdown("\n".join([header, sep] + tbl_rows))

st.markdown(
    "\n**Models used:**\n"
    "- **Eve production app:** GPT-4.1 (baseline dimension scores from real Eve via WebSocket)\n"
    "- **Sample generation:** GPT-4.1 via DeepEval Synthesizer (scenarios + synthetic conversation history)\n"
    "- **Message conversion:** GPT-4.1 (converts DeepEval scenarios into realistic user messages)\n"
    "- **AFlow workflow execution:** GPT-4.1 (the model being optimized by AFlow)\n"
    "- **Dimension evaluation (LLM-as-Judge):** GPT-4.1 via DeepEval ConversationalGEval\n"
    "\n*All models standardized to GPT-4.1 for consistent comparison with Eve production.*"
)

# Scenario explorer
samples_df = loader.load_dataset_samples(dataset)
if not samples_df.empty:
    st.markdown("---")
    st.markdown("#### Scenario Explorer")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        split_filter = st.selectbox(
            "Partition",
            ["All"] + sorted(samples_df["split"].unique().tolist()),
            key="scenario_split",
        )
    with fc2:
        dim_options = ["All"] + sorted(
            samples_df["dimension_target"].dropna().unique().tolist()
        )
        dim_filter = st.selectbox("Dimension", dim_options, key="scenario_dim")
    with fc3:
        turn_options = ["All"] + sorted(
            samples_df["turn_number"].dropna().unique().astype(int).tolist()
        )
        turn_filter = st.selectbox(
            "Turns",
            turn_options,
            format_func=lambda x: f"{x}-turn" if x != "All" else "All",
            key="scenario_turns",
        )

    filtered = samples_df.copy()
    if split_filter != "All":
        filtered = filtered[filtered["split"] == split_filter]
    if dim_filter != "All":
        filtered = filtered[filtered["dimension_target"] == dim_filter]
    if turn_filter != "All":
        filtered = filtered[filtered["turn_number"] == int(turn_filter)]

    st.caption(f"Showing {len(filtered)} of {len(samples_df)} samples")

    for _, row in filtered.iterrows():
        with st.expander(
            f"[{row['split'].upper()}] {row.get('conversation_id', '')} — "
            f"{row.get('dimension_target', '')} · {int(row.get('turn_number', 1))}-turn · "
            f"{row.get('scenario', '')}"
        ):
            # Show conversation context if multi-turn
            context = row.get("context", "")
            if pd.notna(context) and context:
                st.markdown("**Prior turns**")
                _render_context_chat_bubbles(str(context))

            st.markdown("**User message**")
            st.chat_message("user").write(str(row["user_message"]))

            ref = row.get("reference_response", "")
            if pd.notna(ref) and ref:
                st.markdown("**Reference response**")
                st.chat_message("assistant").write(str(ref))

st.divider()

# --- Section 4: MCTS Tree ---
tree_data = loader.load_mcts_tree(dataset)
if tree_data:
    st.subheader("MCTS Search Tree")
    experiences = {}
    for r in loader.get_available_rounds(dataset):
        exp = loader.load_round_experience(dataset, r)
        if exp:
            experiences[r] = exp
    # Use dev scores for node labels if available
    dev_round_scores = None
    if dev_df is not None and not dev_df.empty:
        dev_round_scores = dict(zip(dev_df["round"].astype(int), dev_df["score"]))
    display_mcts_tree(
        tree_data,
        experiences,
        dev_scores=dev_round_scores,
        score_label="Dev" if dev_round_scores else "Train",
        baseline_score=EVE_BASELINE.get("score"),
    )
    st.divider()

# --- Section 5: Workflow Evolution ---
available_rounds = loader.get_available_rounds(dataset)
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

            st.markdown("**Structure**")
            display_workflow_diagram_single(loader, dataset, r)
            st.markdown("**Code**")
            display_workflow_code_single(loader, dataset, r)
    st.divider()

# --- Section 6: Comparison View ---
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
    st.plotly_chart(fig, use_container_width=True, key="dim_comparison_section6")

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

# --- Section 6b: Most Improved Example per Dimension ---
baseline_round = 0  # Eve app baseline (R0)
# Use dev split to pick the best round per dimension and show dev examples
best_rounds_per_dim = loader.find_best_round_per_dimension(dataset, split="dev")
if best_rounds_per_dim:
    improved_per_dim = loader.find_most_improved_per_dimension(
        dataset,
        baseline_round,
        split="dev",
        best_rounds_per_dim=best_rounds_per_dim,
    )
    # Filter to dimensions that actually improved
    # v is a dict keyed by turn count {1: {...}, 2: {...}, ...}
    shown_dims = {
        d: v
        for d, v in improved_per_dim.items()
        if v and isinstance(v, dict) and any(ex["delta"] > 0 for ex in v.values())
    }
    if shown_dims:
        st.subheader("Most Improved Example per Dimension")
        st.caption(
            "Best round per dimension selected on **Dev**, examples shown from **Dev**"
        )

        dim_tabs = st.tabs([d.replace("_", " ").title() for d in shown_dims])
        for tab, (dim, turn_examples) in zip(dim_tabs, shown_dims.items()):
            with tab:
                criteria = DIMENSION_CRITERIA.get(dim, "")
                if criteria:
                    st.info(f"**Scoring Criteria:** {criteria}")

                # Let user select turn count
                available_turns = sorted(turn_examples.keys())
                turn_labels = {t: f"{t}-turn" for t in available_turns}
                selected_turns = st.radio(
                    "Conversation length",
                    available_turns,
                    format_func=lambda t: turn_labels[t],
                    horizontal=True,
                    key=f"turns_{dim}",
                )

                info = turn_examples[selected_turns]

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Best Round (Dev)", f"R{info['best_round']}")
                mc2.metric(
                    f"R{baseline_round} {dim.replace('_', ' ').title()}",
                    f"{info['dim_baseline']:.1%}",
                )
                mc3.metric(
                    f"R{info['best_round']} {dim.replace('_', ' ').title()}",
                    f"{info['dim_best']:.1%}",
                )
                mc4.metric(
                    "Dimension Delta",
                    f"{info['delta']:+.1%}",
                )

                # Show full conversation: context (prior turns) + current user message
                context = info.get("context", "")
                if context:
                    st.markdown("**Conversation History**")
                    _render_context_chat_bubbles(context)

                st.markdown("**Current User Message**")
                st.chat_message("user").write(info["user_message"][:1000])

                col_b, col_t = st.columns(2)
                with col_b:
                    st.markdown(f"**Round {baseline_round} Response**")
                    st.chat_message("assistant").write(
                        info["prediction_baseline"][:2000]
                    )
                with col_t:
                    st.markdown(f"**Round {info['best_round']} Response**")
                    st.chat_message("assistant").write(info["prediction_best"][:2000])

        st.divider()
