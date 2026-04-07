# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Tuple

from benchmarks.arc import ARCBenchmark
from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark
from benchmarks.livecodebench import LiveCodeBench
from benchmarks.eve import EveBenchmark
from benchmarks.strategyqa import StrategyQABenchmark

# DatasetType is str so external projects can register their own
# benchmarks at runtime without modifying this file.
DatasetType = str

# Built-in benchmark registry — external projects extend this via
# Evaluator.register_benchmark() before creating an Optimizer.
_BUILTIN_BENCHMARKS: Dict[str, type] = {
    "GSM8K": GSM8KBenchmark,
    "MATH": MATHBenchmark,
    "HumanEval": HumanEvalBenchmark,
    "HotpotQA": HotpotQABenchmark,
    "MBPP": MBPPBenchmark,
    "DROP": DROPBenchmark,
    "LiveCodeBench": LiveCodeBench,
    "StrategyQA": StrategyQABenchmark,
    "ARC": ARCBenchmark,
    "Eve": EveBenchmark,
}


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    _custom_benchmarks: Dict[str, type] = {}

    @classmethod
    def register_benchmark(cls, name: str, benchmark_class: type) -> None:
        """Register a benchmark so the optimizer can use it by name.

        Call this before creating an Optimizer instance::

            from scripts.evaluator import Evaluator
            from my_benchmarks import MyBenchmark
            Evaluator.register_benchmark("MyDataset", MyBenchmark)
        """
        cls._custom_benchmarks[name] = benchmark_class

    def __init__(self, eval_path: str, data_path: str = "data/datasets"):
        self.eval_path = eval_path
        self.data_path = data_path
        self.dataset_configs: Dict[str, type] = {
            **_BUILTIN_BENCHMARKS,
            **self._custom_benchmarks,
        }

    async def graph_evaluate(
        self,
        dataset: DatasetType,
        graph,
        params: dict,
        path: str,
        is_test: bool = False,
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        import os

        base_path = f"{self.data_path}/{dataset.lower()}"
        if test:
            return f"{base_path}_test.jsonl"
        train_path = f"{base_path}_train.jsonl"
        validate_path = f"{base_path}_validate.jsonl"
        return train_path if os.path.exists(train_path) else validate_path
