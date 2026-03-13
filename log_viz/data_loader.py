"""Data loading for AFlow optimization results."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.config import (
    CACHE_TTL_DATASETS,
    CACHE_TTL_RESULTS,
    EVE_DIMENSIONS,
    PROJECT_ROOT,
    WORKSPACE_DIR,
    WORKSPACE_DIRS,
)


class AFlowDataLoader:
    """Loads AFlow optimization data from workspace directories."""

    def __init__(self, workspace_root: Path = WORKSPACE_DIR):
        self.workspace_root = workspace_root

    def _workflows_path(self, dataset: str) -> Path:
        return self._resolve_dataset(dataset) / "workflows"

    def _workflows_test_path(self, dataset: str) -> Path:
        return self._resolve_dataset(dataset) / "workflows_test"

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def get_available_datasets(_self) -> List[str]:
        """Discover datasets across all workspace directories.

        Returns labels like 'HotpotQA' or 'HotpotQA (workspace_v2)'.
        """
        datasets = []
        for ws_dir in WORKSPACE_DIRS:
            if not ws_dir.exists():
                continue
            suffix = "" if ws_dir.name == "workspace" else f" ({ws_dir.name})"
            for d in sorted(ws_dir.iterdir()):
                results_path = d / "workflows" / "results.json"
                if (
                    d.is_dir()
                    and results_path.exists()
                    and results_path.stat().st_size > 0
                ):
                    datasets.append(f"{d.name}{suffix}")
        return datasets

    def _resolve_dataset(self, dataset_label: str) -> Path:
        """Resolve a dataset label to its workspace root path."""
        if " (" in dataset_label:
            name, ws = dataset_label.rsplit(" (", 1)
            ws_name = ws.rstrip(")")
            for ws_dir in WORKSPACE_DIRS:
                if ws_dir.name == ws_name:
                    return ws_dir / name
        return self.workspace_root / dataset_label

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_validation_results(_self, dataset: str) -> pd.DataFrame:
        """Load workflows/results.json as DataFrame."""
        path = _self._workflows_path(dataset) / "results.json"
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("round").reset_index(drop=True)
        return df

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_test_results(_self, dataset: str) -> Optional[pd.DataFrame]:
        """Load workflows_test/results.json if it exists."""
        path = _self._workflows_test_path(dataset) / "results.json"
        if not path.exists() or path.stat().st_size == 0:
            return None
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("round").reset_index(drop=True)
        return df

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_mcts_tree(_self, dataset: str) -> Dict:
        """Build MCTS tree from individual round experience.json files.

        The optimizer writes processed_experience.json mid-run, so it can be
        stale (missing the last round).  Building from per-round files is
        always up-to-date.
        """
        from collections import defaultdict

        wf = _self._workflows_path(dataset)
        tree: Dict = defaultdict(lambda: {"score": None, "success": {}, "failure": {}})

        for d in sorted(wf.iterdir()):
            m = re.match(r"round_(\d+)", d.name)
            if not m or not d.is_dir():
                continue
            exp_path = d / "experience.json"
            if not exp_path.exists():
                continue
            with open(exp_path) as f:
                data = json.load(f)

            round_number = int(m.group(1))
            father = data.get("father node", data.get("father_node"))
            if father is None:
                continue

            if tree[father]["score"] is None:
                tree[father]["score"] = data.get("before")

            entry = {
                "modification": data.get("modification", ""),
                "score": data.get("after"),
            }
            if data.get("short_label"):
                entry["short_label"] = data["short_label"]

            bucket = "success" if data.get("succeed") else "failure"
            tree[father][bucket][round_number] = entry

        return dict(tree)

    def get_available_rounds(self, dataset: str) -> List[int]:
        """Discover round_N directories."""
        wf = self._workflows_path(dataset)
        rounds = []
        if not wf.exists():
            return rounds
        for d in wf.iterdir():
            m = re.match(r"round_(\d+)", d.name)
            if m and d.is_dir():
                rounds.append(int(m.group(1)))
        return sorted(rounds)

    def load_round_experience(self, dataset: str, round_num: int) -> Optional[Dict]:
        """Load round_N/experience.json."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "experience.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_round_graph(self, dataset: str, round_num: int) -> Optional[str]:
        """Load round_N/graph.py as string."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "graph.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_round_prompt(self, dataset: str, round_num: int) -> Optional[str]:
        """Load round_N/prompt.py as string."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "prompt.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_round_log(self, dataset: str, round_num: int) -> Optional[List[Dict]]:
        """Load round_N/log.json (failed/incorrect predictions)."""
        path = self._workflows_path(dataset) / f"round_{round_num}" / "log.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_operator_definitions(self, dataset: str) -> Optional[Dict]:
        """Load template/operator.json."""
        path = self._workflows_path(dataset) / "template" / "operator.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_all_results(_self, dataset: str) -> pd.DataFrame:
        """Load and combine validation + test results, tagged with source."""
        frames = []
        for source, path in [
            ("val", _self._workflows_path(dataset) / "results.json"),
            ("test", _self._workflows_test_path(dataset) / "results.json"),
        ]:
            if path.exists() and path.stat().st_size > 0:
                with open(path) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                if not df.empty:
                    df["source"] = source
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "time" in combined.columns:
            combined["time"] = pd.to_datetime(combined["time"])
        return combined

    @staticmethod
    def detect_runs(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
        """Detect distinct runs by clustering timestamps.

        Returns list of (run_id, label, run_df) sorted newest first.
        Entries within 30 minutes of each other belong to the same run.
        """
        if df.empty or "time" not in df.columns:
            return []

        sorted_df = df.sort_values("time").reset_index(drop=True)
        runs = []
        current_run = [sorted_df.iloc[0]]
        for i in range(1, len(sorted_df)):
            row = sorted_df.iloc[i]
            prev = current_run[-1]
            if (row["time"] - prev["time"]) > timedelta(minutes=30):
                runs.append(current_run)
                current_run = [row]
            else:
                current_run.append(row)
        runs.append(current_run)

        result = []
        for run_rows in runs:
            run_df = pd.DataFrame(run_rows).sort_values("round").reset_index(drop=True)
            first_time = run_df["time"].iloc[0]
            run_id = first_time.strftime("%Y%m%d-%H%M")
            source = run_df["source"].iloc[0]
            n_rounds = len(run_df)
            label = f"{run_id} ({source}, {n_rounds} rounds)"
            result.append((run_id, label, run_df))

        # Newest first
        result.reverse()
        return result

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_run_config(_self, dataset: str, split: str = "val") -> Optional[Dict]:
        """Load run_config.json for a dataset's val or test workflows."""
        if split == "test":
            path = _self._workflows_test_path(dataset) / "run_config.json"
        else:
            path = _self._workflows_path(dataset) / "run_config.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def load_operator_source(self, dataset: str) -> Optional[str]:
        """Load template/operator.py as string."""
        path = self._workflows_path(dataset) / "template" / "operator.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_operator_prompts(self, dataset: str) -> Optional[str]:
        """Load template/op_prompt.py as string."""
        path = self._workflows_path(dataset) / "template" / "op_prompt.py"
        if not path.exists():
            return None
        return path.read_text()

    def load_round_csv(self, dataset: str, round_num: int) -> Optional[pd.DataFrame]:
        """Load the per-sample CSV file from a round directory.

        CSV files are named {avg_score}_{timestamp}.csv (e.g. 0.77600_20260313_100000.csv).
        """
        round_dir = self._workflows_path(dataset) / f"round_{round_num}"
        if not round_dir.exists():
            return None
        csv_files = list(round_dir.glob("*.csv"))
        if not csv_files:
            return None
        return pd.read_csv(csv_files[0])

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_eve_dimension_summary(_self, dataset: str) -> pd.DataFrame:
        """Aggregate per-dimension scores from round CSVs.

        Returns DataFrame with columns: round, verbosity, tone, assertiveness, empathy, score.
        Only includes rounds that have CSV files with dimension columns.
        """
        rows = []
        for r in _self.get_available_rounds(dataset):
            csv_df = _self.load_round_csv(dataset, r)
            if csv_df is None:
                continue
            # Check that dimension columns exist
            available_dims = [d for d in EVE_DIMENSIONS if d in csv_df.columns]
            if not available_dims:
                continue
            row = {"round": r}
            for dim in EVE_DIMENSIONS:
                if dim in csv_df.columns:
                    row[dim] = csv_df[dim].mean()
            if "score" in csv_df.columns:
                row["score"] = csv_df["score"].mean()
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def get_dataset_split_sizes(_self, dataset: str) -> Dict[str, int]:
        """Count samples in each dataset split (validate / test).

        Returns e.g. {"validate": 200, "test": 800}.
        """
        # Strip workspace suffix to get the raw dataset name
        name = dataset.split(" (")[0].lower()
        data_dir = PROJECT_ROOT / "data" / "datasets"
        sizes: Dict[str, int] = {}
        for split in ("validate", "test"):
            path = data_dir / f"{name}_{split}.jsonl"
            if path.exists():
                sizes[split] = sum(1 for _ in open(path))
        return sizes
