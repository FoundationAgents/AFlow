"""Export AFlow optimization results to Morse Graph IR (agent-graph.json).

Called automatically at the end of an optimization run. Serializes the
best round's graph architecture (pipeline steps, prompts) into a
portable JSON that a TypeScript runtime can consume.

Each pipeline step is fully specified — model, temperature, and the
exact instruction — so the TS side is a dumb executor with no decisions.
"""

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The exact prompt AFlow's ScEnsemble operator uses
SC_ENSEMBLE_VOTING_PROMPT = (
    "Given the question described as follows: {problem}\n"
    "Several solutions have been generated to address the given question. "
    "They are as follows:\n{solutions}\n\n"
    "Carefully evaluate these solutions and identify the answer that appears "
    "most frequently across them. This consistency in answers is crucial for "
    "determining the most reliable solution.\n\n"
    "Output only the full text of the most consistent solution. "
    "Do not include any additional text or explanation."
)


def _parse_prompts(source: str) -> dict:
    """Extract string constants from Python source using AST."""
    prompts = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return prompts
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    prompts[target.id] = node.value.value.strip()
    return prompts


def _parse_pipeline(graph_source: str) -> list:
    """Extract pipeline steps from graph.py source."""
    steps = []
    for match in re.finditer(
        r"await\s+self\.custom\(\s*input=.*?,\s*instruction=prompt_custom\.(\w+)",
        graph_source,
    ):
        steps.append({"type": "custom", "promptRef": match.group(1)})
    if re.search(r"self\.sc_ensemble|ScEnsemble", graph_source):
        steps.append({"type": "scEnsemble"})
    return steps


def _find_best_round(results_path: Path) -> Optional[dict]:
    if not results_path.exists():
        return None
    results = json.loads(results_path.read_text())
    return max(results, key=lambda r: r["score"]) if results else None


def export_agent_graph(
    workspace_path: str,
    worker_name: str = "agent",
    eval_config: dict = None,
    round_num: int = None,
    prompt_source_path: str = None,
    exec_model: str = "gpt-4.1",
    exec_temperature: float = 0.0,
) -> dict:
    """Export the best (or specified) round to Morse Graph IR.

    Each pipeline step includes the model and temperature so the TS
    executor doesn't need to make any decisions.
    """
    ws = Path(workspace_path)
    workflows = ws / "workflows"
    results_path = workflows / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []

    if round_num is None:
        best = _find_best_round(results_path)
        if best is None:
            raise ValueError(f"No results in {results_path}")
        round_num = best["round"]

    round_score = next((r["score"] for r in results if r["round"] == round_num), 0.0)
    baseline_score = results[0]["score"] if results else 0.0

    round_dir = workflows / f"round_{round_num}"
    graph_source = (round_dir / "graph.py").read_text()
    prompt_path = round_dir / "prompt.py"
    prompts = _parse_prompts(prompt_path.read_text()) if prompt_path.exists() else {}

    raw_pipeline = _parse_pipeline(graph_source)

    pipeline = []
    for step in raw_pipeline:
        if step["type"] == "custom":
            ref = step.get("promptRef")
            instruction = prompts.get(ref, "") if ref else ""
            pipeline.append({
                "type": "custom",
                "model": exec_model,
                "temperature": exec_temperature,
                "instruction": instruction,
                "semantics": "Concatenate instruction + input, call LLM, return the text response.",
            })
        elif step["type"] == "scEnsemble":
            pipeline.append({
                "type": "scEnsemble",
                "model": exec_model,
                "temperature": exec_temperature,
                "candidates": 3,
                "votingPrompt": SC_ENSEMBLE_VOTING_PROMPT,
                "semantics": (
                    "Call the base agent N times with the same input to get N candidate responses. "
                    "Then call the LLM with the votingPrompt + all candidates to select the best one."
                ),
            })

    system_prompt_source = None
    if prompt_source_path:
        ps = Path(prompt_source_path)
        if ps.exists():
            system_prompt_source = ps.read_text()

    lead_node = {
        "type": "llm",
        "pipeline": pipeline,
    }
    if system_prompt_source is not None:
        lead_node["systemPromptSource"] = system_prompt_source

    graph = {
        "version": "1",
        "name": worker_name,
        "entryNode": "leadAgent",
        "nodes": {
            "leadAgent": lead_node,
        },
        "edges": [],
        "metadata": {
            "optimizedBy": "aflow",
            "optimizedAt": datetime.now(timezone.utc).isoformat(),
            "sourceRound": round_num,
            "baselineScore": baseline_score,
            "optimizedScore": round_score,
        },
    }

    if eval_config:
        graph["evalConfig"] = eval_config

    output_path = ws / "agent-graph.json"
    output_path.write_text(json.dumps(graph, indent=2))
    return graph
