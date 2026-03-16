"""Evaluate specific AFlow rounds on a given dataset split.

Usage:
    uv run python scripts/eval_rounds.py --dataset Eve --split dev --rounds 1,2,3,4 \
        --exec_model_name gpt-4.1-mini --optimized_path workspace_eve_demo
"""

import argparse
import asyncio
import os
import sys

# Ensure project root is on path (needed when run as scripts/eval_rounds.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.eve import EveBenchmark  # noqa: E402
from scripts.async_llm import LLMsConfig  # noqa: E402
from scripts.optimizer import Optimizer  # noqa: E402


def load_graph_class(optimizer, round_number, workflows_path):
    """Load a graph class for a specific round, handling module remapping."""
    return optimizer._load_graph_fresh(round_number, workflows_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rounds on a dataset split")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "dev", "test"]
    )
    parser.add_argument(
        "--rounds", type=str, required=True, help="Comma-separated round numbers"
    )
    parser.add_argument("--exec_model_name", type=str, default="gpt-4.1")
    parser.add_argument("--optimized_path", type=str, required=True)
    args = parser.parse_args()

    round_numbers = [int(r) for r in args.rounds.split(",")]
    data_path = f"data/datasets/{args.dataset.lower()}_{args.split}.jsonl"

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        exit(1)

    models_config = LLMsConfig.default()
    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        print(f"ERROR: Model {args.exec_model_name} not found in config/config2.yaml")
        exit(1)

    # Create a minimal optimizer just for graph loading
    optimizer = Optimizer.__new__(Optimizer)
    optimizer.dataset = args.dataset
    optimizer.execute_llm_config = exec_llm_config

    workflows_path = f"{args.optimized_path}/{args.dataset}/workflows"

    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split} ({data_path})")
    print(f"Rounds: {round_numbers}")
    print(f"Exec model: {args.exec_model_name}")
    print()

    results = []
    for rnd in round_numbers:
        print(f"Evaluating round {rnd} on {args.split}...")

        graph_class = load_graph_class(optimizer, rnd, workflows_path)
        workflow = graph_class(
            name=args.dataset,
            llm_config=exec_llm_config,
            dataset=args.dataset,
        )

        # Write CSVs to split-specific directory (workflows_dev/round_N or workflows_test/round_N)
        split_dir = (
            f"{args.optimized_path}/{args.dataset}/workflows_{args.split}/round_{rnd}"
        )
        os.makedirs(split_dir, exist_ok=True)

        benchmark = EveBenchmark(
            name=args.dataset,
            file_path=data_path,
            log_path=split_dir,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        score, avg_cost, total_cost = loop.run_until_complete(
            benchmark.run_evaluation(workflow, va_list=None)
        )

        result = {
            "round": rnd,
            "split": args.split,
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
        }
        results.append(result)
        print(
            f"  Round {rnd}: score={score:.4f}, avg_cost=${avg_cost:.4f}, total_cost=${total_cost:.4f}"
        )
        print()

    # Write results.json for the dashboard
    import json
    from datetime import datetime

    results_for_json = [
        {
            "round": r["round"],
            "score": r["score"],
            "avg_cost": r["avg_cost"],
            "total_cost": r["total_cost"],
            "time": datetime.now().isoformat(),
        }
        for r in results
    ]
    results_json_path = (
        f"{args.optimized_path}/{args.dataset}/workflows_{args.split}/results.json"
    )
    with open(results_json_path, "w") as f:
        json.dump(results_for_json, f, indent=4)
    print(f"Wrote {results_json_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Round':>6} | {'Split':>5} | {'Score':>8} | {'Avg Cost':>10}")
    print(f"{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}")
    for r in results:
        print(
            f"{r['round']:>6} | {r['split']:>5} | {r['score']:>8.4f} | ${r['avg_cost']:>9.4f}"
        )
