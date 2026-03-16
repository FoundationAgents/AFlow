"""Prediction-level detail table."""

from typing import List

import pandas as pd
import streamlit as st

from data_loader import AFlowDataLoader
from utils.config import MAX_PREDICTIONS_DISPLAY


def display_predictions(
    loader: AFlowDataLoader,
    dataset: str,
    rounds: List[int],
) -> None:
    """Display prediction-level details from log.json."""
    if not rounds:
        st.info("No rounds to display.")
        return

    selected_round = st.selectbox(
        "Select Round",
        rounds,
        index=0,
        key="predictions_round",
    )

    log_data = loader.load_round_log(dataset, selected_round)
    if not log_data:
        st.info(f"No prediction log for round {selected_round}.")
        return

    df = pd.DataFrame(log_data)

    # Summary
    total = len(df)
    st.caption(
        f"Showing {min(total, MAX_PREDICTIONS_DISPLAY)} of {total} incorrect predictions"
    )

    # Filter
    search = st.text_input(
        "Search questions",
        placeholder="Type to filter...",
        key="predictions_search",
    )
    if search:
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df = df[mask]

    # Display table
    display_df = df.head(MAX_PREDICTIONS_DISPLAY)

    # Rename columns for readability
    column_map = {
        "question": "Question",
        "right_answer": "Expected",
        "model_output": "Model Output",
        "extracted_output": "Extracted",
    }
    display_cols = [c for c in column_map if c in display_df.columns]
    display_df = display_df[display_cols].rename(columns=column_map)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )
