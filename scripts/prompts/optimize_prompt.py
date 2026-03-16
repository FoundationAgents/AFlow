ANSWER_FORMAT_CONSTRAINTS = {
    "ARC": """ANSWER FORMAT CONSTRAINT (CRITICAL — overrides all other instructions):
The final output of the workflow MUST be a SINGLE LETTER (A, B, C, D, or E).
The evaluation extracts the first standalone letter A-E. If no letter is found, the answer is WRONG.
- GOOD output: "A" or "The answer is B." or "B. kinetic energy"
- BAD output: "kinetic energy" or "20 m/s" or "The hot water dissolved..."
DO NOT use AnswerGenerate — its built-in prompt produces prose answers (e.g. "20 m/s") instead of letters, which ALWAYS fails extraction. Use ONLY Custom operators with explicit "answer with the letter A/B/C/D/E" instructions.
DO NOT add unnecessary operators to this task — the baseline Custom approach already works well. Only add complexity if each step preserves the letter format.""",
    "StrategyQA": """ANSWER FORMAT CONSTRAINT: The final output must contain "yes" or "no".
The evaluation system extracts the first occurrence of "yes" or "no" (case-insensitive).
- GOOD output: "Yes" or "No, because..." or "The answer is yes."
- BAD output: "True" or "Correct" or "The statement is accurate"
All prompts must instruct the model to answer with yes or no.""",
    "HotpotQA": """ANSWER FORMAT CONSTRAINT: The final output should be a concise text answer (entity name, short phrase).
Scoring uses F1 token overlap with the gold answer after removing articles and punctuation.
- GOOD output: "Barack Obama" or "1945" or "the United Kingdom"
- BAD output: Long paragraphs of reasoning (dilutes F1 score)
Prompts should instruct the model to output ONLY the answer, not explanations.""",
    "GSM8K": """ANSWER FORMAT CONSTRAINT: The final output must contain a number.
The evaluation extracts the LAST number found in the output.
- GOOD output: "The answer is 42." or "42"
- BAD output: "forty-two" or text with no digits""",
    "MATH": """ANSWER FORMAT CONSTRAINT: The final answer must be in LaTeX \\boxed{} format.
The evaluation extracts the content of the last \\boxed{} in the output.
- GOOD output: "Therefore \\boxed{\\frac{1}{2}}"
- BAD output: "The answer is 1/2" (no boxed)""",
    "DROP": """ANSWER FORMAT CONSTRAINT: The final output should be a concise text answer.
Scoring uses F1 token overlap. Keep answers short and precise.""",
    "Eve": """ANSWER FORMAT CONSTRAINT: The output should be a natural conversational response from Eve, an AI job search and resume assistant for Australian users.
The evaluation scores the response on 4 behavioral dimensions using LLM-as-Judge:
- Verbosity (target: medium) — cover what's needed without unnecessary filler
- Tone of Voice (target: friendly) — warm, approachable, professional
- Assertiveness (target: medium) — clear guidance without being pushy or passive
- Empathy (target: supportive) — acknowledge the user's situation and feelings
GOOD output: Warm, appropriately detailed response that addresses the user's needs and acknowledges their situation with genuine empathy before jumping to solutions.
BAD output: Cold/robotic response, overly verbose or terse, dismissive of user's feelings, or too passive/aggressive.
Do NOT output just a factual answer — Eve should respond as a helpful, empathetic assistant.""",
}

WORKFLOW_OPTIMIZE_PROMPT = """You are building a Graph and corresponding Prompt to jointly solve {type} problems.
{answer_format}
Referring to the given graph and prompt, which forms a basic example of a {type} solution approach,
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators),
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical
representation.Ensure that all the prompts required by the current graph from prompt_custom are included.Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
CRITICAL CONSISTENCY RULE: Every `prompt_custom.ATTRIBUTE_NAME` referenced in your <graph> output MUST have a corresponding `ATTRIBUTE_NAME = \"""...\"""` definition in your <prompt> output. For example, if the graph uses `prompt_custom.XXX_PROMPT` and `prompt_custom.ALT_PROMPT`, the prompt section MUST define BOTH `XXX_PROMPT` and `ALT_PROMPT`. Missing definitions cause runtime errors.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process."""


WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <short_label>(3-4 word Terraform-style summary using + for add, - for remove, ~ for change. Examples: "+ Self-review step", "~ Ensemble logic", "- Redundant loop")</short_label>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n
IMPORTANT: If you reference `prompt_custom.XXX_PROMPT` or any other attribute in the graph, you MUST define those exact variable names in your <prompt> output. Verify that every `prompt_custom.NAME` used in <graph> has a matching `NAME = \"""...\"""` in <prompt>.

Here is a COMPLETE VALID example of matching <graph> and <prompt> sections:
<graph>
class Workflow:
    ...
    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        review = await self.custom(input=solution["response"], instruction=prompt_custom.REVIEW_PROMPT)
        return review["response"], self.llm.get_usage_summary()["total_cost"]
</graph>
<prompt>
SOLVE_PROMPT = \"""
Solve the following problem step by step.
\"""

REVIEW_PROMPT = \"""
Review the solution for correctness and output only the final answer.
\"""
</prompt>

SELF-CHECK before outputting: (1) Every `prompt_custom.X` in <graph> has a matching `X = \"""...\"""` in <prompt>. (2) <prompt> contains ONLY Python variable assignments — no comments, no markdown, no explanation text.

**Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators are not yet used in the graph, try incorporating them.**
"""

WORKFLOW_TEMPLATE = """from typing import Literal
import workspace.{dataset}.workflows.template.operator as operator
import workspace.{dataset}.workflows.round_{round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

{graph}
"""
