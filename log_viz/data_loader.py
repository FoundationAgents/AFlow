"""Data loading for AFlow optimization results."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Add log_viz to path if not already there
log_viz_path = str(Path(__file__).parent)
if log_viz_path not in sys.path:
    sys.path.insert(0, log_viz_path)

from utils.config import (
    CACHE_TTL_DATASETS,
    CACHE_TTL_RESULTS,
    EVE_BASELINE,
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

    def _workflows_dev_path(self, dataset: str) -> Path:
        return self._resolve_dataset(dataset) / "workflows_dev"

    def _workflows_test_path(self, dataset: str) -> Path:
        return self._resolve_dataset(dataset) / "workflows_test"

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def get_available_datasets(_self) -> List[str]:
        """Discover datasets across all workspace directories.

        Returns labels like 'HotpotQA' or 'HotpotQA (workspace_v2)'.
        Also includes datasets with only test results (like EVE).
        """
        datasets = []
        for ws_dir in WORKSPACE_DIRS:
            if not ws_dir.exists():
                continue
            suffix = "" if ws_dir.name == "workspace" else f" ({ws_dir.name})"
            for d in sorted(ws_dir.iterdir()):
                # Check for main results
                main_results = d / "workflows" / "results.json"
                test_results = d / "workflows_test" / "results.json"

                if d.is_dir() and (
                    (main_results.exists() and main_results.stat().st_size > 0)
                    or (test_results.exists() and test_results.stat().st_size > 0)
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
    def load_dev_results(_self, dataset: str) -> Optional[pd.DataFrame]:
        """Load workflows_dev/results.json if it exists."""
        path = _self._workflows_dev_path(dataset) / "results.json"
        if not path.exists() or path.stat().st_size == 0:
            return None
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
        """Load and combine train/dev/test results, tagged with source."""
        frames = []
        for source, path in [
            ("train", _self._workflows_path(dataset) / "results.json"),
            ("dev", _self._workflows_dev_path(dataset) / "results.json"),
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
            # Handle timestamps with microseconds and ISO format
            combined["time"] = pd.to_datetime(
                combined["time"], format="ISO8601", errors="coerce"
            )
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

    def load_round_csv(
        self, dataset: str, round_num: int, split: str = "train"
    ) -> Optional[pd.DataFrame]:
        """Load the per-sample CSV file from a round directory.

        CSV files are named {avg_score}_{timestamp}.csv (e.g. 0.77600_20260313_100000.csv).
        split: 'train' uses workflows/, 'dev' uses workflows_dev/, 'test' uses workflows_test/
        """
        if split == "dev":
            base = self._workflows_dev_path(dataset)
        elif split == "test":
            base = self._workflows_test_path(dataset)
        else:
            base = self._workflows_path(dataset)
        round_dir = base / f"round_{round_num}"
        if not round_dir.exists():
            return None
        csv_files = sorted(round_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not csv_files:
            return None
        return pd.read_csv(csv_files[-1])  # most recent

    @st.cache_data(ttl=CACHE_TTL_RESULTS)
    def load_eve_dimension_summary(
        _self, dataset: str, split: str = "train"
    ) -> pd.DataFrame:
        """Aggregate per-dimension scores from round CSVs.

        Returns DataFrame with columns: round, verbosity, tone_of_voice, assertiveness, empathy, score.
        Only includes rounds that have CSV files with dimension columns.
        split: 'train' uses workflows/, 'dev' uses workflows_dev/, 'test' uses workflows_test/
        """
        rows = []
        # Discover rounds from the split-specific directory
        if split == "dev":
            base = _self._workflows_dev_path(dataset)
        elif split == "test":
            base = _self._workflows_test_path(dataset)
        else:
            base = _self._workflows_path(dataset)

        if not base.exists():
            return pd.DataFrame()

        round_dirs = [
            d for d in base.iterdir() if d.is_dir() and re.match(r"round_\d+", d.name)
        ]
        round_nums = sorted(int(d.name.split("_")[1]) for d in round_dirs)

        for r in round_nums:
            csv_df = _self.load_round_csv(dataset, r, split=split)
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

    def find_best_round_per_dimension(
        self, dataset: str, split: str = "dev"
    ) -> Dict[str, int]:
        """Find the round with the highest mean score for each dimension on a split.

        Returns e.g. {"verbosity": 3, "tone_of_voice": 2, ...}.
        """
        dim_df = self.load_eve_dimension_summary(dataset, split=split)
        if dim_df.empty:
            return {}
        best_rounds = {}
        for dim in EVE_DIMENSIONS:
            if dim in dim_df.columns:
                best_idx = dim_df[dim].idxmax()
                best_rounds[dim] = int(dim_df.loc[best_idx, "round"])
        return best_rounds

    def find_most_improved_per_dimension(
        self,
        dataset: str,
        baseline_round: int,
        split: str = "test",
        best_rounds_per_dim: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """Find the most improved sample for each dimension.

        For each dimension, compares baseline_round vs the dev-best round for that
        dimension, then picks the sample with the largest gain on that dimension.

        Returns dict keyed by dimension name, each value is a dict with:
        sample_id, user_message, prediction_baseline, prediction_best,
        dim_baseline, dim_best, delta, best_round.
        """
        if best_rounds_per_dim is None:
            best_rounds_per_dim = self.find_best_round_per_dimension(
                dataset, split="dev"
            )

        if baseline_round == 0:
            # R0 = Eve app baseline; no per-sample CSV exists.
            # Use a scaffold CSV from the first best round and fill with
            # EVE_BASELINE constants so that per-sample deltas can be computed.
            first_best_rnd = next(iter(best_rounds_per_dim.values()), None)
            if first_best_rnd is None:
                return {}
            scaffold_csv = self.load_round_csv(dataset, first_best_rnd, split=split)
            if scaffold_csv is None:
                return {}
            baseline_csv = scaffold_csv.copy()
            for dim in EVE_DIMENSIONS:
                if dim in baseline_csv.columns:
                    baseline_csv[dim] = EVE_BASELINE.get(dim, 0)
            if "score" in baseline_csv.columns:
                baseline_csv["score"] = EVE_BASELINE.get("score", 0)
            if "prediction" in baseline_csv.columns:
                baseline_jsonl = (
                    PROJECT_ROOT
                    / "data"
                    / "datasets"
                    / f"eve_{split}_with_baseline.jsonl"
                )
                if baseline_jsonl.exists() and "user_message" in baseline_csv.columns:
                    response_lookup = {}
                    with open(baseline_jsonl) as f:
                        for line in f:
                            rec = json.loads(line)
                            if rec.get("eve_baseline_response"):
                                response_lookup[rec["user_message"]] = rec[
                                    "eve_baseline_response"
                                ]
                    if response_lookup:
                        baseline_csv["prediction"] = (
                            baseline_csv["user_message"]
                            .map(response_lookup)
                            .fillna("(Eve app baseline)")
                        )
                    else:
                        baseline_csv["prediction"] = "(Eve app baseline)"
                else:
                    baseline_csv["prediction"] = "(Eve app baseline)"
        else:
            baseline_csv = self.load_round_csv(dataset, baseline_round, split=split)
        if baseline_csv is None:
            return {}

        results: Dict[str, Optional[List[Dict]]] = {}
        for dim, best_rnd in best_rounds_per_dim.items():
            if best_rnd == baseline_round:
                results[dim] = None
                continue

            best_csv = self.load_round_csv(dataset, best_rnd, split=split)
            if best_csv is None:
                results[dim] = None
                continue

            # Join on sample_id if available, else positional
            if "sample_id" in baseline_csv.columns and "sample_id" in best_csv.columns:
                merged = baseline_csv.merge(
                    best_csv, on="sample_id", suffixes=("_baseline", "_best")
                )
            else:
                if len(baseline_csv) != len(best_csv):
                    results[dim] = None
                    continue
                merged = baseline_csv.add_suffix("_baseline").join(
                    best_csv.add_suffix("_best")
                )

            dim_b = f"{dim}_baseline"
            dim_t = f"{dim}_best"
            if dim_b not in merged.columns or dim_t not in merged.columns:
                results[dim] = None
                continue

            merged["_delta"] = merged[dim_t] - merged[dim_b]

            def _extract_row(row, idx):
                um_col = (
                    "user_message"
                    if "user_message" in merged.columns
                    else "user_message_best"
                )
                ctx_col = "context" if "context" in merged.columns else "context_best"
                ctx_val = str(row.get(ctx_col, row.get("context_baseline", "")))
                if ctx_val == "nan":
                    ctx_val = ""
                return {
                    "sample_id": row.get("sample_id", str(idx)),
                    "context": ctx_val,
                    "user_message": str(
                        row.get(um_col, row.get("user_message_baseline", ""))
                    ),
                    "prediction_baseline": str(row.get("prediction_baseline", "")),
                    "prediction_best": str(row.get("prediction_best", "")),
                    "dim_baseline": row[dim_b],
                    "dim_best": row[dim_t],
                    "delta": row["_delta"],
                    "best_round": best_rnd,
                    "score_baseline": row.get("score_baseline", 0),
                    "score_best": row.get("score_best", 0),
                }

            # Count turns per row: 1 (no context) or count of "User:" in context + 1
            ctx_col = "context" if "context" in merged.columns else "context_best"
            if ctx_col in merged.columns:

                def _count_turns(ctx):
                    ctx_str = str(ctx) if pd.notna(ctx) else ""
                    if not ctx_str or ctx_str == "nan":
                        return 1
                    return ctx_str.count("User:") + 1

                merged["_turns"] = merged[ctx_col].apply(_count_turns)
            else:
                merged["_turns"] = 1

            # Build examples dict keyed by turn count
            examples = {}
            for n_turns in sorted(merged["_turns"].unique()):
                subset = merged[merged["_turns"] == n_turns]
                if subset.empty:
                    continue
                best_idx = subset["_delta"].idxmax()
                if subset.loc[best_idx, "_delta"] > 0:
                    examples[int(n_turns)] = _extract_row(
                        merged.loc[best_idx], best_idx
                    )

            results[dim] = examples if examples else None

        return results

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

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def get_turn_distribution(_self, dataset: str) -> Dict[str, Dict[int, int]]:
        """Count samples by number of turns per split.

        Returns e.g. {"train": {1: 21, 2: 2, 3: 2}, "dev": {...}, "test": {...}}.
        """
        name = dataset.split(" (")[0].lower()
        data_dir = PROJECT_ROOT / "data" / "datasets"
        result: Dict[str, Dict[int, int]] = {}
        for split in ("train", "dev", "test"):
            path = data_dir / f"{name}_{split}.jsonl"
            if not path.exists():
                continue
            counts: Dict[int, int] = {}
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    t = row.get("turn_number", 1)
                    counts[t] = counts.get(t, 0) + 1
            result[split] = dict(sorted(counts.items()))
        return result

    @st.cache_data(ttl=CACHE_TTL_DATASETS)
    def load_dataset_samples(_self, dataset: str) -> pd.DataFrame:
        """Load all raw JSONL samples across train/dev/test into a DataFrame.

        Returns DataFrame with columns: split, conversation_id, scenario,
        dimension_target, turn_number, user_message, context, reference_response.
        """
        name = dataset.split(" (")[0].lower()
        data_dir = PROJECT_ROOT / "data" / "datasets"
        rows = []
        for split in ("train", "dev", "test"):
            path = data_dir / f"{name}_{split}.jsonl"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    row["split"] = split
                    rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
