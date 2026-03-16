# -*- coding: utf-8 -*-
# @Desc    : Eve (open-ended) benchmark with DeepEval ConversationalGEval scoring
#             Based on Appendix F of the AFlow paper

import asyncio
import json
from typing import Any, Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, Turn
from scripts.logs import logger

# Scoring criteria for each persona dimension, aligned with Eve's persona.ts
DIMENSION_CRITERIA = {
    "verbosity": (
        "Evaluate whether the assistant's response length and detail level is appropriate "
        "for the user's query. The target is MEDIUM verbosity — the response should cover "
        "what's needed without unnecessary filler. Every sentence should earn its place. "
        "Score low if the response is extremely terse and omits critical information, or if "
        "it is excessively verbose with unnecessary padding. Score high if the detail level "
        "is well-calibrated to the complexity of the user's request."
    ),
    "tone_of_voice": (
        "Evaluate whether the assistant's response sounds warm, approachable, and professional. "
        "The target tone is FRIENDLY — like a helpful, approachable colleague. "
        "Score low if the response is cold, robotic, rude, or overly formal and distant. "
        "Score high if the response feels naturally warm and professional without being "
        "unprofessional or overly casual."
    ),
    "assertiveness": (
        "Evaluate whether the assistant provides clear guidance without being pushy or passive. "
        "The target is MEDIUM assertiveness — offering clear suggestions while respecting "
        "user autonomy. Score low if the response is completely passive (only mirrors the user) "
        "or overly hedging (qualifies everything, user unsure what to do). Also score low if "
        "the response is too directive or pushy. Score high if the response confidently guides "
        "without overstepping."
    ),
    "empathy": (
        "Evaluate whether the assistant acknowledges the user's situation and emotional context. "
        "The target is SUPPORTIVE empathy — validating feelings and adjusting the response "
        "accordingly. Score low if the response is dismissive or ignores emotional undertones. "
        "Score high if the response demonstrates genuine understanding and makes the user feel "
        "heard before jumping to solutions."
    ),
}


def _parse_context_to_turns(context: str) -> List[Turn]:
    """Parse a context string of 'User: ...\nAssistant: ...' pairs into Turn objects."""
    turns = []
    if not context:
        return turns

    current_role = None
    current_content = []

    for line in context.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("User:"):
            if current_role and current_content:
                turns.append(
                    Turn(role=current_role, content="\n".join(current_content).strip())
                )
            current_role = "user"
            current_content = [line_stripped[len("User:") :].strip()]
        elif line_stripped.startswith("Assistant:"):
            if current_role and current_content:
                turns.append(
                    Turn(role=current_role, content="\n".join(current_content).strip())
                )
            current_role = "assistant"
            current_content = [line_stripped[len("Assistant:") :].strip()]
        elif line_stripped:
            current_content.append(line_stripped)

    if current_role and current_content:
        turns.append(
            Turn(role=current_role, content="\n".join(current_content).strip())
        )

    return turns


class EveBenchmark(BaseBenchmark):
    """DeepEval ConversationalGEval benchmark for Eve's open-ended responses.

    Each test case contains:
      - question: pre-formatted input string (for AFlow dry-run compatibility)
      - user_message: the user's raw input
      - context: conversation history (User:/Assistant: pairs)
      - reference_response: a gold-standard response
      - scenario: the type of interaction (job_search, resume, general, multi_turn)

    Scoring: 4 ConversationalGEval metrics (verbosity, tone, assertiveness, empathy),
    each returning 0-1, averaged for the final score.
    """

    DIMENSIONS = ["verbosity", "tone_of_voice", "assertiveness", "empathy"]

    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.metrics = self._build_metrics()

    def _build_metrics(self) -> dict:
        """Create one ConversationalGEval metric per scoring dimension."""
        metrics = {}
        for dim_name, criteria in DIMENSION_CRITERIA.items():
            metrics[dim_name] = ConversationalGEval(
                name=dim_name,
                criteria=criteria,
                model="gpt-4.1",
                threshold=0.5,
                async_mode=True,
                verbose_mode=False,
            )
        return metrics

    def _build_test_case(
        self, context: str, user_message: str, prediction: str
    ) -> ConversationalTestCase:
        """Build a ConversationalTestCase from context, user message, and assistant response."""
        turns = _parse_context_to_turns(context)
        turns.append(Turn(role="user", content=user_message))
        turns.append(Turn(role="assistant", content=prediction))
        return ConversationalTestCase(
            turns=turns,
            chatbot_role="Eve, a friendly AI job search and resume assistant",
        )

    async def _score_dimensions(
        self, test_case: ConversationalTestCase
    ) -> Tuple[float, dict]:
        """Score a test case on all 4 dimensions using DeepEval ConversationalGEval."""
        dim_scores = {}
        for dim_name, metric in self.metrics.items():
            try:
                score = await asyncio.to_thread(
                    metric.measure, test_case, _show_indicator=False
                )
                dim_scores[dim_name] = score
            except Exception as e:
                logger.warning(f"DeepEval scoring failed for {dim_name}: {e}")
                dim_scores[dim_name] = 0.0

        avg_score = sum(dim_scores.values()) / len(dim_scores) if dim_scores else 0.0
        return avg_score, dim_scores

    def calculate_score(
        self, expected_output: Any, prediction: Any  # noqa: ARG002
    ) -> Tuple[float, Any]:
        """Not used directly — scoring happens in evaluate_problem via DeepEval.
        Kept for interface compatibility."""
        return 0.0, prediction

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(
        self, problem: dict, graph: Callable
    ) -> Tuple[
        str, str, str, str, str, str, str, float, float, float, float, float, float
    ]:
        sample_id = problem.get("conversation_id", "")
        user_message = problem["user_message"]
        reference_response = problem["reference_response"]
        context = problem.get("context", "")
        scenario = problem.get("scenario", "general")

        # Build input — use pre-formatted question field if available
        input_text = problem.get("question", user_message)

        try:
            prediction, cost = await self._generate_output(graph, input_text)

            # DeepEval ConversationalGEval scoring
            test_case = self._build_test_case(context, user_message, prediction)
            score, dim_scores = await self._score_dimensions(test_case)

            if score < 0.6:
                self.log_mismatch(
                    user_message, reference_response, prediction, dim_scores
                )

            return (
                sample_id,
                context,
                user_message,
                scenario,
                prediction,
                reference_response,
                json.dumps(dim_scores),
                dim_scores.get("verbosity", 0.0),
                dim_scores.get("tone_of_voice", 0.0),
                dim_scores.get("assertiveness", 0.0),
                dim_scores.get("empathy", 0.0),
                score,
                cost,
            )

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                sample_id,
                context,
                user_message,
                scenario,
                str(e),
                reference_response,
                "{}",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

    def get_result_columns(self) -> List[str]:
        return [
            "sample_id",
            "context",
            "user_message",
            "scenario",
            "prediction",
            "reference_response",
            "dimension_scores",
            "verbosity",
            "tone_of_voice",
            "assertiveness",
            "empathy",
            "score",
            "cost",
        ]
