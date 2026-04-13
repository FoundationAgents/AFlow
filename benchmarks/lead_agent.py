# -*- coding: utf-8 -*-
# @Desc    : Lead Agent benchmark with single-judge correctness + persona scoring.
#
# Passes the full JSONL record (as JSON string) to the graph so the graph
# can extract messages, user_id, and handle multi-turn correctly.
# Scores with a single LLM judge covering both correctness and persona.

import json
import os
from typing import Any, Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4.1")


class LeadAgentBenchmark(BaseBenchmark):
    """Benchmark for lead agent optimization.

    Each JSONL record has:
      - test_id, suite, description, comment, user_id
      - messages: [{role, content}, ...]
      - wanted_response: reference answer
      - expected_side_effects: []

    The full record is serialized to JSON and passed to the graph.
    The graph is responsible for extracting the user message and
    calling the Eval Bridge.
    """

    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        persona_description: str = "",
        correctness_weight: float = 0.6,
        persona_weight: float = 0.4,
    ):
        super().__init__(name, file_path, log_path)
        self.persona_description = persona_description
        self.correctness_weight = correctness_weight
        self.persona_weight = persona_weight
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def calculate_score(
        self, expected_output: Any, prediction: Any
    ) -> Tuple[float, Any]:
        return 0.0, prediction

    async def _judge(self, wanted: str, actual: str) -> dict:
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)
        try:
            completion = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluation judge scoring an AI agent's response on two axes.\n\n"
                            "AXIS 1 — CORRECTNESS:\n"
                            "How well does the actual response convey the same information and intent "
                            "as the wanted response? It does NOT need to be word-for-word identical. "
                            "Score based on whether the key facts, actions, and intent match. "
                            "Missing key information, wrong information, or a different action = low score.\n\n"
                            "AXIS 2 — PERSONA:\n"
                            f"{self.persona_description}\n\n"
                            'Return ONLY a JSON object: {"correctness": <float 0.0 to 1.0>, "persona": <float 0.0 to 1.0>}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Wanted response:\n{wanted}\n\n"
                            f"Actual response:\n{actual}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=30,
            )
            text = completion.choices[0].message.content or ""
            parsed = json.loads(text)
            return {
                "correctness": float(parsed.get("correctness", 0.0)),
                "persona": float(parsed.get("persona", 0.0)),
            }
        except Exception as e:
            logger.warning(f"Judge error: {e}")
            return {"correctness": 0.0, "persona": 0.0}

    async def run_evaluation(self, agent, va_list, max_concurrent_tasks=1):
        """Override to force sequential execution — the Eval Bridge
        serializes requests and can't handle concurrent calls."""
        return await super().run_evaluation(agent, va_list, max_concurrent_tasks=1)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(
        self, problem: dict, graph: Callable
    ) -> Tuple[str, str, str, str, float, float, float, float]:
        test_id = problem.get("test_id", "")
        suite = problem.get("suite", "")
        wanted = problem.get("wanted_response", "")

        input_text = json.dumps(problem)

        try:
            prediction, cost = await self._generate_output(graph, input_text)

            scores = await self._judge(wanted, prediction)
            correctness = scores["correctness"]
            persona = scores["persona"]
            score = self.correctness_weight * correctness + self.persona_weight * persona

            if score < 0.5:
                self.log_mismatch(input_text[:200], wanted, prediction, scores)

            return (
                test_id,
                suite,
                prediction[:200],
                wanted[:200],
                correctness,
                persona,
                score,
                cost,
            )

        except Exception as e:
            logger.info(f"Skipping {test_id}: {e}")
            return (test_id, suite, str(e), wanted[:200], 0.0, 0.0, 0.0, 0.0)

    def get_result_columns(self) -> List[str]:
        return [
            "test_id",
            "suite",
            "prediction",
            "wanted",
            "correctness",
            "persona",
            "score",
            "cost",
        ]
