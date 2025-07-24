# -*- coding: utf-8 -*-
# @Date    : 4/1/2025 21:01 PM
# @Author  : didi
# @Desc    : Run baseline to test prompts 

import asyncio
from typing import Literal
from scripts.operators import AnswerGenerate, CustomCodeGenerate
from scripts.workflow import Workflow
from scripts.async_llm import LLMsConfig
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.mbpp import MBPPBenchmark
from benchmarks.livecodebench import LiveCodeBench
from scripts.logs import logger
from scripts.async_llm import create_llm_instance

# Import the new Workflow from MBPP workspace
import workspace.MBPP.workflows.round_1.graph as mbpp_workflow
import workspace.LiveCodeBench.workflows.round_1.graph as livecodebench_workflow
import workspace.MBPP.workflows.template.operator as mbpp_operator

from scripts.evaluator import DatasetType

class PoWorkflow(Workflow):
    def __init__(
        self,
        test_prompt: str,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.prompt = test_prompt
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.response = AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        full_input = self.prompt + problem
        solution = await self.response(input=full_input)
        return solution['answer'], self.llm.get_usage_summary()["total_cost"]
    
class ModelWorkflow(Workflow):
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom_code_generate = CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str, question_id: str = ""):
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="") # But When you want to get standard code ,you should use customcodegenerator.
        return solution['response'], self.llm.get_usage_summary()["total_cost"]


async def main():
    mini_llm_config = LLMsConfig.default().get("o3-mini")
    
    # Use the new MBPP Workflow instead of PoWorkflow
    mbpp_test_workflow = mbpp_workflow.Workflow(
        name="mbpp_test", 
        llm_config=mini_llm_config, 
        dataset="MBPP"
    )
    livecodebench_test_workflow = livecodebench_workflow.Workflow(
        name="livecodebench_test", 
        llm_config=mini_llm_config, 
        dataset="LiveCodeBench"
    )
    model_test_workflow = ModelWorkflow(name="mbpp_model_test", llm_config=mini_llm_config, dataset="MBPP")
    
    # file path refer to the dataset you want to use.
    # log_path refer to the folder of output csv.
    # test_hotpotqa_benchmark = HotpotQABenchmark(name="HotpotQA", file_path="data/hotpotqa_validate.jsonl", log_path="")
    test_mbpp_benchmark = MBPPBenchmark(name="MBPP", file_path="data/datasets/mbpp_test.jsonl", log_path="")
    test_livecodebench = LiveCodeBench(name="LiveCodeBench", file_path="data/datasets/livecodebench_validate.jsonl", log_path="experiments/lcb")
    # test_livecodebench = LiveCodeBench(name="LiveCodeBench", file_path="data/datasets/livecodebench_validate.jsonl", log_path="")
    
    # results = await test_mbpp_benchmark.run_baseline(mbpp_test_workflow)
    # results = await test_mbpp_benchmark.run_baseline(model_test_workflow)
    # results = await test_livecodebench.run_baseline(model_test_workflow)
    results = await test_livecodebench.run_baseline(livecodebench_test_workflow)

    print(results)

if __name__ == "__main__":
    asyncio.run(main())


# AFLOW Test 
# MBPP o3-mini-raw 0.78886 0.78299 
# MBPP o3-mini-aflow 0.93548 0.94135

# LiveCodeBench o3-mini-raw 
# LiveCodeBench o3-mini-aflow 
