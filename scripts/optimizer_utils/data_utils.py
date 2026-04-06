import datetime
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:

    DEFAULT_ALPHA = 0.2
    DEFAULT_LAMBDA = 0.3
    DEFAULT_LOG_SAMPLES = 3

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.top_scores: List[Dict[str, Any]] = []

    @staticmethod
    def get_dataset_size(
        dataset_name: str, split: str, data_path: str = "data/datasets"
    ) -> Optional[int]:
        """Count samples in a dataset JSONL file."""
        path = Path(f"{data_path}/{dataset_name.lower()}_{split}.jsonl")
        if not path.exists():
            return None
        return sum(1 for _ in open(path))

    def load_results(self, path: str) -> list:
        result_path = Path(path) / "results.json"

        if not result_path.exists():
            return []

        try:
            with open(result_path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from {result_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading results from {result_path}: {e}")
            return []

    def get_top_rounds(
        self, sample: int, path: Optional[str] = None, mode: str = "Graph"
    ) -> List[Dict]:

        self._load_scores(path, mode)
        unique_rounds: Dict[int, Dict] = {}

        for item in self.top_scores:
            round_num = item["round"]
            if round_num not in unique_rounds:
                unique_rounds[round_num] = item
                if len(unique_rounds) >= sample:
                    break

        result = []
        if 1 in unique_rounds:
            result.append(unique_rounds[1])
            del unique_rounds[1]

        result.extend(unique_rounds.values())

        return result[:sample]

    def select_round(self, items: List[Dict]) -> Dict:

        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(
            f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}"
        )

        return sorted_items[selected_index]

    def _compute_probabilities(
        self,
        scores: List[float],
        alpha: float = DEFAULT_ALPHA,
        lambda_: float = DEFAULT_LAMBDA,
    ) -> np.ndarray:

        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(
        self, cur_round: int, path: Optional[str] = None, mode: str = "Graph"
    ) -> str:
        if mode == "Graph":
            log_dir = self.root_path / "workflows" / f"round_{cur_round}" / "log.json"
        else:
            log_dir = Path(path)

        if not log_dir.exists():
            logger.warning(f"Log file not found: {log_dir}")
            return ""

        logger.info(f"Loading log from: {log_dir}")

        try:
            data = read_json_file(log_dir, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading log file {log_dir}: {e}")
            return ""

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(self.DEFAULT_LOG_SAMPLES, len(data))
        random_samples = random.sample(data, sample_size)

        log_entries = [
            json.dumps(sample, indent=4, ensure_ascii=False)
            for sample in random_samples
        ]

        return "\n\n".join(log_entries)

    def get_results_file_path(self, graph_path: str) -> str:

        return str(Path(graph_path) / "results.json")

    def create_result_data(
        self, round: int, score: float, avg_cost: float, total_cost: float
    ) -> dict:

        now = datetime.datetime.now()
        return {
            "round": round,
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "time": now,
        }

    def save_results(self, json_file_path: str, data: list) -> None:
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(
        self, path: Optional[str] = None, mode: str = "Graph"
    ) -> List[Dict]:
        if mode == "Graph":
            rounds_dir = self.root_path / "workflows"
        else:
            rounds_dir = Path(path)

        result_file = rounds_dir / "results.json"
        self.top_scores = []

        try:
            data = read_json_file(result_file, encoding="utf-8")
            df = pd.DataFrame(data)

            scores_per_round = df.groupby("round")["score"].mean().to_dict()

            self.top_scores = [
                {"round": round_number, "score": average_score}
                for round_number, average_score in scores_per_round.items()
            ]

            self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Error loading scores from {result_file}: {e}")
            self.top_scores = []

        return self.top_scores
