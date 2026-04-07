# -*- coding: utf-8 -*-
# @Desc    : Generic Eval Bridge benchmark for E2E agent evaluation.
#
# Loads test cases from a JSONL file with the real_eval_data schema
# (messages, turn_evaluations with rubric criteria + gates), sends
# conversations to an Eval Bridge HTTP server, and scores each turn
# with an LLM judge gated by deterministic checks.
#
# This benchmark is worker-agnostic — it works for any agent that
# speaks the Eval Bridge protocol.

import json
import os
import re
from typing import Any, Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger

BRIDGE_URL = os.environ.get("EVAL_BRIDGE_URL", "http://localhost:9700")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4.1")


def _check_gate(gate: dict, response: str) -> bool:
    """Evaluate a single deterministic gate against a response string."""
    gate_type = gate["type"]
    value = gate["value"]

    if gate_type == "regex":
        return bool(re.search(value, response, re.IGNORECASE))
    elif gate_type == "not_regex":
        return not bool(re.search(value, response, re.IGNORECASE))
    elif gate_type == "contains":
        return value in response
    elif gate_type == "not_contains":
        return value not in response
    else:
        logger.warning(f"Unknown gate type: {gate_type}")
        return True


async def _judge_turn(criteria: dict, response: str, api_key: str) -> float:
    """Score a response using an LLM judge. Returns 0.0-1.0."""
    import openai

    prompt = criteria["prompt"]
    ctype = criteria.get("type", "rubric")

    if ctype == "similarity" and criteria.get("reference"):
        system = (
            "You are an evaluation judge. Score how well the assistant response "
            "matches the reference answer and satisfies the criteria. "
            'Return ONLY a JSON object: {"score": <float 0.0 to 1.0>}'
        )
        user_msg = (
            f"Criteria: {prompt}\n\n"
            f"Reference: {criteria['reference']}\n\n"
            f"Response: {response}"
        )
    else:
        system = (
            "You are an evaluation judge. Score how well the assistant response "
            "satisfies the criteria. "
            'Return ONLY a JSON object: {"score": <float 0.0 to 1.0>}'
        )
        user_msg = f"Criteria: {prompt}\n\nResponse: {response}"

    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        text = completion.choices[0].message.content or ""
        parsed = json.loads(text)
        return float(parsed.get("score", 0.0))
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        return 0.0


async def _call_bridge(messages: List[str], user_id: str, bridge_url: str) -> List[str]:
    """Call the Eval Bridge and return agent responses per turn."""
    import aiohttp

    if len(messages) == 1:
        endpoint = f"{bridge_url}/_eval/run"
        payload = {"userId": user_id, "message": messages[0], "timeoutMs": 90_000}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Bridge error: {data.get('error', resp.status)}"
                    )
                return data.get("messages", [data.get("response", "")])
    else:
        endpoint = f"{bridge_url}/_eval/run-conversation"
        payload = {"userId": user_id, "messages": messages, "perTurnTimeoutMs": 90_000}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120 * len(messages)),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Bridge error: {data.get('error', resp.status)}"
                    )
                return data.get("turns", [])


class EvalBridgeBenchmark(BaseBenchmark):
    """Eval Bridge benchmark using rubric + gate scoring.

    Each JSONL record has:
      - test_id, suite, description, user_id
      - messages: list of user messages
      - turn_evaluations: per-turn scoring (criteria + gates)
      - ground_truth: optional context for interpolation

    Scoring per turn:
      turn_score = llm_judge(criteria) * product(gate_pass ? 1 : 0)
    Sample score = mean(turn_scores)
    """

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.bridge_url = BRIDGE_URL
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def calculate_score(
        self, _expected_output: Any, prediction: Any
    ) -> Tuple[float, Any]:
        return 0.0, prediction

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _run_bridge(self, messages: List[str], user_id: str) -> List[str]:
        return await _call_bridge(messages, user_id, self.bridge_url)

    async def evaluate_problem(
        self, problem: dict, _graph: Callable
    ) -> Tuple[str, str, str, str, str, float, float]:
        test_id = problem.get("test_id", "")
        suite = problem.get("suite", "")
        user_id = problem.get("user_id", "account_1")
        messages = problem.get("messages", [])
        turn_evaluations = problem.get("turn_evaluations", [])

        try:
            agent_turns = await self._run_bridge(messages, user_id)
        except Exception as e:
            logger.info(f"Bridge call failed for {test_id}: {e}")
            return (test_id, suite, json.dumps(messages), "", str(e), 0.0, 0.0)

        turn_scores = []
        gate_details = []

        for te in turn_evaluations:
            turn_idx = te["turn"]
            if turn_idx == -1:
                turn_idx = len(agent_turns) - 1
            if turn_idx < 0 or turn_idx >= len(agent_turns):
                turn_scores.append(0.0)
                gate_details.append(
                    {"turn": te["turn"], "error": "turn index out of range"}
                )
                continue

            response = agent_turns[turn_idx]
            criteria = te.get("criteria", {})
            gates = te.get("gates", [])

            all_gates_pass = True
            failed_gates = []
            for gate in gates:
                if not _check_gate(gate, response):
                    all_gates_pass = False
                    failed_gates.append(gate.get("description", gate["value"]))

            if all_gates_pass:
                judge_score = await _judge_turn(criteria, response, self.api_key)
            else:
                judge_score = 0.0

            turn_score = judge_score if all_gates_pass else 0.0
            turn_scores.append(turn_score)
            gate_details.append(
                {
                    "turn": te["turn"],
                    "judge_score": judge_score,
                    "gates_pass": all_gates_pass,
                    "failed_gates": failed_gates,
                    "turn_score": turn_score,
                }
            )

        sample_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

        prediction_summary = (
            " | ".join(t[:100] for t in agent_turns) if agent_turns else ""
        )

        if sample_score < 0.6:
            self.log_mismatch(
                json.dumps(messages),
                json.dumps(turn_evaluations),
                prediction_summary,
                gate_details,
            )

        return (
            test_id,
            suite,
            json.dumps(messages),
            prediction_summary,
            json.dumps(gate_details),
            sample_score,
            0.0,
        )

    def get_result_columns(self) -> List[str]:
        return [
            "test_id",
            "suite",
            "messages",
            "prediction",
            "gate_details",
            "score",
            "cost",
        ]
