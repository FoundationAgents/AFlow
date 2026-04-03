"""Sidebar controls for the AFlow dashboard."""

from typing import Any, Dict

import streamlit as st

from data_loader import AFlowDataLoader


def render_sidebar(loader: AFlowDataLoader) -> Dict[str, Any]:
    """Render sidebar with dataset selector, run selector, and refresh controls."""
    with st.sidebar:
        st.header("AFlow Dashboard")

        # Dataset selector
        datasets = loader.get_available_datasets()
        if not datasets:
            st.error("No datasets found in workspace/")
            st.stop()

        dataset = st.selectbox("Dataset", datasets, index=0)

        # Detect runs
        all_results = loader.load_all_results(dataset)
        runs = loader.detect_runs(all_results)

        selected_run_df = None
        if runs:
            run_labels = [label for _, label, _ in runs]
            # Default to the first test run if one exists
            default_idx = 0
            for i, (_, _, rdf) in enumerate(runs):
                if "source" in rdf.columns and rdf["source"].iloc[0] == "test":
                    default_idx = i
                    break
            selected_label = st.selectbox("Run", run_labels, index=default_idx)
            selected_idx = run_labels.index(selected_label)
            _, _, selected_run_df = runs[selected_idx]

        # Dataset split sizes
        split_sizes = loader.get_dataset_split_sizes(dataset)
        if split_sizes:
            cols = st.columns(len(split_sizes))
            for col, (split, count) in zip(cols, split_sizes.items()):
                col.metric(split.capitalize(), f"{count:,}")

        # All rounds selected by default
        selected_rounds = loader.get_available_rounds(dataset)

        # Run configuration
        source = "val"
        if selected_run_df is not None and "source" in selected_run_df.columns:
            source = selected_run_df["source"].iloc[0]
        run_config = loader.load_run_config(dataset, split=source)
        if run_config:
            with st.expander("Run Configuration"):
                st.markdown("**Models**")
                cols = st.columns(2)
                cols[0].metric("Optimizer", run_config.get("opt_model", "N/A"))
                cols[1].metric("Executor", run_config.get("exec_model", "N/A"))

                st.markdown("**Search**")
                cols = st.columns(3)
                cols[0].metric("Sample", run_config.get("sample"))
                cols[1].metric("Max Rounds", run_config.get("max_rounds"))
                cols[2].metric("Val Rounds", run_config.get("validation_rounds"))

                st.markdown("**MCTS**")
                cols = st.columns(2)
                cols[0].metric("Alpha", run_config.get("mcts_alpha"))
                cols[1].metric("Lambda", run_config.get("mcts_lambda"))

                with st.expander("Raw JSON"):
                    st.json(run_config)

        # Operator definitions
        operators = loader.load_operator_definitions(dataset)
        if operators:
            with st.expander("Available Operators"):
                for name, info in operators.items():
                    st.markdown(f"**{name}**")
                    if isinstance(info, dict):
                        st.caption(info.get("description", ""))
                        st.code(info.get("interface", ""), language="python")
                    else:
                        st.caption(str(info))

        st.divider()

        # Auto-refresh controls
        auto_refresh = st.toggle("Auto-refresh", value=False)
        refresh_interval = st.slider(
            "Refresh interval (s)",
            min_value=1,
            max_value=10,
            value=5,
            disabled=not auto_refresh,
        )

        if st.button("Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    return {
        "dataset": dataset,
        "selected_rounds": selected_rounds,
        "selected_run_df": selected_run_df,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
    }
