# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 17:36 PM
# @Author  : didi
# @Desc    : operator demo of aflow
from typing import List

from scripts.async_llm import AsyncLLM
from scripts.logs import logger
from scripts.operators import Operator
from scripts.utils.code import extract_test_cases_from_jsonl
from scripts.utils.lcb_runner import grade_stdio
from workspace.MBPP.workflows.template.op_prompt import *
from workspace.MBPP.workflows.template.operator_an import *


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(
            GenerateOp, prompt, mode="code_fill", function_name=entry_point
        )
        return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point, question_id=""):
        """
        Execute code using LiveCodeBench runner for consistency with official evaluation
        """
        import json

        # For LiveCodeBench, use question_id to find test cases
        search_key = question_id if question_id else entry_point
        test_cases = extract_test_cases_from_jsonl(search_key, dataset="LiveCodeBench")

        # Handle case where no test cases are found
        if test_cases is None:
            return {"exec_fail_case": f"No test cases found for {search_key}"}

        try:
            # Parse test cases - they should be in JSON format for LiveCodeBench
            if isinstance(test_cases, str):
                test_cases = json.loads(test_cases)

            # Extract inputs and outputs for lcb_runner
            inputs = []
            outputs = []

            for test_case in test_cases:
                if isinstance(test_case, dict):
                    inputs.append(test_case.get("input", ""))
                    outputs.append(test_case.get("output", ""))
            print(inputs)
            print(outputs)

            # Use grade_stdio directly to avoid multiprocessing issues
            results, metadata = grade_stdio(
                code=solution, all_inputs=inputs, all_outputs=outputs, timeout=6
            )

            logger.info(f"results: {results} {metadata}")

            # Check if all tests passed
            if isinstance(results, list) and all(r == True or r == 1 for r in results):
                return "no error"
            else:
                # Return error information
                return {"exec_fail_case": f"Test failed: {metadata}"}

        except Exception as e:
            return {"exec_fail_case": f"Error executing tests: {str(e)}"}

    async def __call__(
        self, problem, solution, entry_point, test_loop: int = 3, question_id: str = ""
    ):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the soluion and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point, question_id)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(
                    ReflectionTestOp, prompt, mode="code_fill"
                )
                solution = response["response"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(
                    ReflectionTestOp, prompt, mode="code_fill"
                )
                solution = response["response"]

        result = self.exec_code(solution, entry_point, question_id)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}
