# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.run_config import (
    RunConfig,
    load_run_config,
    update_run_config,
    write_run_config,
)
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.logs import logger

QuestionType = Literal["math", "code", "qa", "open_ended"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    short_label: str = Field(
        default="",
        description="A 3-4 word Terraform-style summary of the modification, "
        "e.g. '+ Self-review step', '~ Ensemble logic', '- Redundant loop'",
    )
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = (
            create_llm_instance(opt_llm_config) if opt_llm_config else None
        )
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

    def _build_run_config(self, mode: str, test_rounds=None) -> RunConfig:
        """Assemble all hyperparameters into a RunConfig snapshot."""
        return RunConfig(
            dataset=self.dataset,
            mode=mode,
            question_type=self.type,
            opt_model=(
                self.optimize_llm_config.model if self.optimize_llm_config else None
            ),
            exec_model=self.execute_llm_config.model,
            opt_temperature=(
                self.optimize_llm_config.temperature
                if self.optimize_llm_config
                else None
            ),
            opt_top_p=(
                self.optimize_llm_config.top_p if self.optimize_llm_config else None
            ),
            exec_temperature=self.execute_llm_config.temperature,
            exec_top_p=self.execute_llm_config.top_p,
            sample=self.sample,
            max_rounds=self.max_rounds,
            validation_rounds=self.validation_rounds,
            check_convergence=self.check_convergence,
            initial_round=self.round,
            mcts_alpha=DataUtils.DEFAULT_ALPHA,
            mcts_lambda=DataUtils.DEFAULT_LAMBDA,
            log_samples=DataUtils.DEFAULT_LOG_SAMPLES,
            valset_size=DataUtils.get_dataset_size(self.dataset, "validate"),
            testset_size=DataUtils.get_dataset_size(self.dataset, "test"),
            started_at=datetime.now(timezone.utc).isoformat(),
            test_rounds=test_rounds,
        )

    def _finalize_run_config(self, output_dir: str, converged: bool = False) -> None:
        """Update run_config.json with runtime metadata after run completes."""
        results = self.data_utils.load_results(output_dir)
        best_score = (
            max((r["score"] for r in results), default=None) if results else None
        )
        baseline_score = (
            next((r["score"] for r in results if r["round"] == 1), None)
            if results
            else None
        )
        total_cost = sum(r.get("total_cost", 0) for r in results) if results else None
        rounds_completed = len(set(r["round"] for r in results)) if results else 0

        completed_at = datetime.now(timezone.utc)
        # Compute wall-clock runtime from started_at
        runtime_seconds = None
        run_config = load_run_config(output_dir)
        if run_config and run_config.get("started_at"):
            started = datetime.fromisoformat(run_config["started_at"])
            runtime_seconds = (completed_at - started).total_seconds()

        update_run_config(
            output_dir,
            {
                "completed_at": completed_at.isoformat(),
                "rounds_completed": rounds_completed,
                "best_score": best_score,
                "baseline_score": baseline_score,
                "total_cost": total_cost,
                "runtime_seconds": runtime_seconds,
                "converged": converged,
            },
        )

    def optimize(self, mode: OptimizerType = "Graph", test_rounds=None):
        config = self._build_run_config(mode, test_rounds)

        if mode == "Test":
            output_dir = f"{self.root_path}/workflows_test"
            write_run_config(config, output_dir)
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test(rounds=test_rounds))
            self._finalize_run_config(output_dir)
            return None

        output_dir = f"{self.root_path}/workflows"
        write_run_config(config, output_dir)

        convergence_detected = False
        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(
                        f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})"
                    )
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = (
                self.convergence_utils.check_convergence(top_k=3)
            )

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                convergence_detected = True
                break

            time.sleep(5)

        self._finalize_run_config(output_dir, converged=convergence_detected)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(
                self, directory, validation_n, data, initial=True
            )

        # Create a loop until the generated graph meets the check conditions
        max_generation_retries = 20
        for _retry in range(max_generation_retries):
            directory = self.graph_utils.create_round_directory(
                graph_path, self.round + 1
            )

            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(
                sample["round"], graph_path
            )
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(
                processed_experience, sample["round"]
            )

            operator_description = self.graph_utils.load_operators_description(
                self.operators
            )
            log_data = self.data_utils.load_log(sample["round"])

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience,
                sample["score"],
                graph[0],
                prompt,
                operator_description,
                self.type,
                log_data,
                dataset=self.dataset,
            )

            # Replace ActionNode with AsyncLLM and XmlFormatter
            try:
                # Create XmlFormatter based on GraphOptimize model
                graph_formatter = XmlFormatter.from_model(GraphOptimize)

                # Call the LLM with formatter
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt,
                    graph_formatter,
                )

                # If we reach here, response is properly formatted and validated
                logger.info("Graph optimization response received successfully")
            except FormatError as e:
                # Handle format validation errors
                logger.error(f"Format error in graph optimization: {str(e)}")
                # Try again with a fallback approach - direct call with post-processing
                raw_response = await self.optimize_llm(graph_optimize_prompt)

                # Try to extract fields using basic parsing
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error(
                        "Failed to extract fields from raw response, retrying..."
                    )
                    continue

            # Check if the modification meets the conditions
            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )
            if not check:
                logger.info("Duplicate modification detected, retrying...")
                continue

            # Validate that all prompt_custom references in graph have definitions in prompt
            sanitized_prompt = self.graph_utils._sanitize_prompt_content(
                response["prompt"]
            )
            valid, refs, defs = GraphUtils.validate_prompt_references(
                response["graph"], sanitized_prompt
            )
            if not valid:
                missing = refs - defs
                logger.warning(
                    f"Prompt reference validation failed. "
                    f"Graph references {missing} but prompt only defines {defs}. Retrying..."
                )
                continue

            # Write graph files and dry-run on a single sample to catch runtime errors
            self.graph_utils.write_graph_files(
                directory, response, self.round + 1, self.dataset
            )

            try:
                self.graph = self._load_graph_fresh(self.round + 1, graph_path)
                await self._dry_run_graph(self.graph)
                logger.info(f"Dry-run passed for round {self.round + 1}")
            except Exception as e:
                logger.warning(
                    f"Generated graph failed dry-run: {e}. "
                    f"Retrying generation... (attempt {_retry + 1}/{max_generation_retries})"
                )
                continue

            break
        else:
            logger.error(
                f"Exhausted {max_generation_retries} generation retries. "
                f"Skipping round {self.round + 1}."
            )
            return 0.0

        # Full evaluation (only reached if dry-run passed)
        experience = self.experience_utils.create_experience_data(
            sample, response["modification"], response.get("short_label", "")
        )

        logger.info(directory)

        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=False
        )

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    def _load_graph_fresh(self, round_number: int, graph_path: str):
        """Load a graph module from disk, bypassing Python's import cache entirely.

        Uses importlib.util.spec_from_file_location to load directly from the
        file path rather than __import__, which has deep internal caching that
        persists even after sys.modules eviction and importlib.invalidate_caches().

        Key subtlety: WORKFLOW_TEMPLATE hardcodes the import path as
        ``workspace.{dataset}.workflows.round_N.prompt``, regardless of the
        actual workspace directory (e.g. workspace_pilot2/).  We must register
        the freshly-loaded prompt module under that template import name so
        that graph.py's import statement finds it.
        """
        round_dir = Path(graph_path) / f"round_{round_number}"

        # Delete __pycache__ to prevent bytecode reuse
        pycache_dir = round_dir / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)
        importlib.invalidate_caches()

        # The graph.py template always imports:
        #   import workspace.{dataset}.workflows.round_N.prompt as prompt_custom
        # We must register the prompt module under this exact name.
        template_base = f"workspace.{self.dataset}.workflows"
        template_prompt_name = f"{template_base}.round_{round_number}.prompt"
        template_round_name = f"{template_base}.round_{round_number}"

        # Also compute the actual module path for graph (used for loading)
        actual_base = graph_path.replace("\\", ".").replace("/", ".")
        actual_graph_name = f"{actual_base}.round_{round_number}.graph"
        actual_prompt_name = f"{actual_base}.round_{round_number}.prompt"
        actual_round_name = f"{actual_base}.round_{round_number}"

        # Evict stale modules under both actual and template paths
        for name in (
            actual_graph_name,
            actual_prompt_name,
            actual_round_name,
            template_prompt_name,
            template_round_name,
        ):
            sys.modules.pop(name, None)

        # Create the round_N package module so Python's import chain can
        # resolve "workspace.{dataset}.workflows.round_N.prompt".
        # Without this, Python fails to find round_N as a sub-package of
        # workspace.{dataset}.workflows when the actual files live elsewhere.
        round_pkg = types.ModuleType(template_round_name)
        round_pkg.__path__ = [str(round_dir)]
        round_pkg.__package__ = template_round_name
        sys.modules[template_round_name] = round_pkg

        # Also set round_N as an attribute on the parent workflows package
        # so "import workspace.{dataset}.workflows.round_N" resolves
        workflows_module = sys.modules.get(template_base)
        if workflows_module is not None:
            setattr(workflows_module, f"round_{round_number}", round_pkg)

        # Load prompt from actual file, register under the template import name
        # so that graph.py's "import workspace.{dataset}..." finds our fresh copy
        prompt_file = round_dir / "prompt.py"
        prompt_spec = importlib.util.spec_from_file_location(
            template_prompt_name, str(prompt_file)
        )
        prompt_module = importlib.util.module_from_spec(prompt_spec)
        sys.modules[template_prompt_name] = prompt_module
        prompt_spec.loader.exec_module(prompt_module)
        round_pkg.prompt = prompt_module

        # Load graph module from actual file path
        graph_file = round_dir / "graph.py"
        graph_spec = importlib.util.spec_from_file_location(
            actual_graph_name, str(graph_file)
        )
        graph_module = importlib.util.module_from_spec(graph_spec)
        sys.modules[actual_graph_name] = graph_module
        graph_spec.loader.exec_module(graph_module)

        return getattr(graph_module, "Workflow")

    async def _dry_run_graph(self, graph_class):
        """Run the graph on one validation sample to catch runtime errors early."""
        # Use train split if available; fall back to validate for legacy datasets
        train_path = f"data/datasets/{self.dataset.lower()}_train.jsonl"
        validate_path = f"data/datasets/{self.dataset.lower()}_validate.jsonl"
        dataset_path = train_path if os.path.exists(train_path) else validate_path
        with open(dataset_path) as f:
            sample_data = json.loads(f.readline())
        workflow = graph_class(
            name=self.dataset,
            llm_config=self.execute_llm_config,
            dataset=self.dataset,
        )
        await workflow(sample_data["question"])

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing

        Args:
            response: Raw response text from LLM

        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re

            # Initialize result dictionary with default values
            result = {"modification": "", "graph": "", "prompt": ""}

            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()

            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None

            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self, rounds=None):
        if rounds is None:
            rounds = [1]
        data = []

        # Load graphs from the optimization workflows path, save results to workflows_test
        source_graph_path = f"{self.root_path}/workflows"
        output_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(output_path)

        data = self.data_utils.load_results(output_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(output_path, round)
            self.graph = self._load_graph_fresh(round, source_graph_path)

            score, avg_cost, total_cost = (
                await self.evaluation_utils.evaluate_graph_test(
                    self, directory, is_test=True
                )
            )

            new_data = self.data_utils.create_result_data(
                round, score, avg_cost, total_cost
            )
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)
