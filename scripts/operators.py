# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : didi & zhaoyang
# @Desc    : operator demo of aflow

import asyncio
import concurrent.futures
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.async_llm import AsyncLLM
from scripts.logs import logger
from scripts.formatter import (
    BaseFormatter,
    FormatError,
    XmlFormatter,
    TextFormatter,
    CodeFormatter,
)
from scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
)  # All BaseModel

from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
)
from scripts.utils.code import (
    extract_test_cases_from_jsonl,
    test_case_2_test_function,
)


class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)

        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(prompt, formatter)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(prompt)

            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}

    def _create_formatter(
        self, op_class, mode=None, **extra_kwargs
    ) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
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

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            result = global_namespace["solve"]()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)
        # Create a class-level process pool, instead of creating a new one for each execution
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        """Ensure the process pool is closed when the object is destroyed"""
        if hasattr(self, "process_pool"):
            self.process_pool.shutdown(wait=True)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()

        try:
            # Use the class-level process pool
            future = loop.run_in_executor(self.process_pool, run_code, code)
            # Wait for the task to complete or timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Only cancel this specific future, not the entire process pool
            future.cancel()
            # Force garbage collection
            import gc

            gc.collect()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # If the process pool is broken, recreate it
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, try again"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem, analysis=analysis, feedback=feedback
        )
        response = await self._fill_node(
            CodeGenerateOp, prompt, mode, function_name="solve"
        )
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(
                problem, analysis, feedback, mode="code_fill"
            )
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )

            # Force garbage collection after each iteration
            import gc

            gc.collect()

        return {"code": code, "output": output}


class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point):
        test_cases = extract_test_cases_from_jsonl(entry_point)

        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(self, problem, solution, entry_point, test_loop: int = 3):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error'; if incorrect, reflect on the solution and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
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

        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}


class Format(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, _mode: str = None):
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, _mode: str = None):
        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        return response


class Revise(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, feedback, _mode: str = None):
        prompt = REVISE_PROMPT.format(
            problem=problem, solution=solution, feedback=feedback
        )
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")
        return response


class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm: AsyncLLM, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {
            chr(65 + i): solutions.index(solution)
            for i, solution in enumerate(shuffled_solutions)
        }
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, _mode: str = None):
        logger.info(f"solution count: {len(solutions)}")
        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(
                solutions=solution_text, question=problem
            )
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}


class EvalBridgeCall:
    """Calls a worker's Eval Bridge for full E2E agent execution.

    The Eval Bridge is an HTTP server that wraps a TemporalHarness and
    runs the complete TypeScript agent graph — tool calling, sub-agents,
    and all post-agent logic — returning the final response.

    Requires ``aiohttp`` (imported lazily so AFlow users who don't need
    the bridge are unaffected).
    """

    def __init__(self, bridge_url: str = "http://localhost:9700"):
        self.bridge_url = bridge_url.rstrip("/")

    async def __call__(
        self,
        *,
        user_id: str = "account_1",
        message: str,
        timeout_ms: int = 60_000,
    ) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "userId": user_id,
                "message": message,
                "timeoutMs": timeout_ms,
            }
            async with session.post(
                f"{self.bridge_url}/_eval/run",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000 + 10),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Eval Bridge error: {data.get('error', resp.status)}"
                    )
                return {
                    "response": data["response"],
                    "latency": data.get("latency", {}),
                }


class EvalBridgeConversation:
    """Calls a worker's Eval Bridge for a multi-turn E2E conversation.

    Requires ``aiohttp`` (imported lazily).
    """

    def __init__(self, bridge_url: str = "http://localhost:9700"):
        self.bridge_url = bridge_url.rstrip("/")

    async def __call__(
        self,
        *,
        user_id: str = "account_1",
        messages: list,
        per_turn_timeout_ms: int = 60_000,
    ) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "userId": user_id,
                "messages": messages,
                "perTurnTimeoutMs": per_turn_timeout_ms,
            }
            total_timeout = (per_turn_timeout_ms / 1000 + 10) * len(messages)
            async with session.post(
                f"{self.bridge_url}/_eval/run-conversation",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=total_timeout),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Eval Bridge error: {data.get('error', resp.status)}"
                    )
                return {
                    "turns": data["turns"],
                    "latency": data.get("latency", {}),
                }
