import workspace.LiveCodeBench.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType
from scripts.logs import logger


class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)  # NEW: enable automatic verification

    async def __call__(self, problem: str, entry_point: str, question_id: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom_code_generate(
            problem=problem, entry_point=entry_point, instruction=""
        )
        # NEW: run public tests and, if modified, use the repaired solution
        test_result = await self.test(
            problem=problem,
            solution=solution["response"],
            entry_point=entry_point,
            question_id=question_id,
        )
        final_solution = test_result.get("solution", solution["response"])
        logger.info("-------- test result --------")
        logger.info(final_solution)
        logger.info("-------- test result --------")
        return final_solution, self.llm.get_usage_summary()["total_cost"]
