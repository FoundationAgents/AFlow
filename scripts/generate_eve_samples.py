"""Generate Eve evaluation samples targeting persona dimensions.

Reads workspace/Eve/persona_config.yaml and uses DeepEval's Synthesizer
to generate scenarios, then converts them into realistic user messages
via an LLM call. Outputs train/dev/test JSONL partitions.

Usage:
    uv run python scripts/generate_eve_samples.py
    uv run python scripts/generate_eve_samples.py --partition_size 25 --max_turns 3
    uv run python scripts/generate_eve_samples.py --partition_size 10 --max_turns 1 --model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ConversationalStylingConfig, EvolutionConfig
from deepeval.synthesizer.types import Evolution
from openai import AsyncOpenAI

PERSONA_CONFIG_PATH = Path("workspace/Eve/persona_config.yaml")
OUTPUT_DIR = Path("data/datasets")


def load_persona_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_styling_config(persona: dict, dimension: str) -> ConversationalStylingConfig:
    """Build a ConversationalStylingConfig targeting a specific dimension."""
    dim_config = persona["dimensions"][dimension]
    stress_scenarios = "\n".join(f"- {s}" for s in dim_config["stress_scenarios"])
    capabilities = "\n".join(f"- {c}" for c in persona["capabilities"])
    rules = "\n".join(f"- {r}" for r in persona["behavioral_rules"])

    scenario_context = (
        f"{persona['name']} is an AI assistant. Objective: {persona['objective']}\n\n"
        f"Capabilities:\n{capabilities}\n\n"
        f"Behavioral rules:\n{rules}\n\n"
        f"This sample targets the '{dimension}' dimension.\n"
        f"Target level: {dim_config['target']}\n"
        f"Interpretation: {dim_config['interpretation']}\n\n"
        f"Generate scenarios that stress-test this dimension:\n{stress_scenarios}"
    )

    conversational_task = (
        f"The user interacts with {persona['name']} about job search, resume building, "
        f"or career advice. The scenario should make it challenging for the assistant "
        f"to achieve the target '{dim_config['target']}' level of '{dimension}'. "
        f"The user message should feel natural and realistic."
    )

    participant_roles = (
        f"User: An Australian job seeker or career changer with varying emotional states, "
        f"experience levels, and communication styles. "
        f"Assistant: {persona['name']}, a friendly AI job search and resume assistant."
    )

    return ConversationalStylingConfig(
        scenario_context=scenario_context,
        conversational_task=conversational_task,
        participant_roles=participant_roles,
        expected_outcome_format=(
            f"A response that scores high on '{dimension}' ({dim_config['target']}). "
            f"The response should demonstrate: "
            + "; ".join(dim_config["score_high_when"][:3])
        ),
    )


def build_evolution_config() -> EvolutionConfig:
    """Configure evolution to produce diverse, realistic samples."""
    return EvolutionConfig(
        num_evolutions=1,
        evolutions={
            Evolution.REASONING: 0.15,
            Evolution.CONCRETIZING: 0.25,
            Evolution.CONSTRAINED: 0.20,
            Evolution.HYPOTHETICAL: 0.20,
            Evolution.IN_BREADTH: 0.20,
        },
    )


USER_TONES = [
    "neutral and straightforward",
    "friendly and upbeat",
    "frustrated and impatient",
    "anxious and uncertain",
    "demanding and curt",
    "skeptical and challenging",
    "overwhelmed and stressed",
    "confused and rambling",
    "sarcastic and dismissive",
    "grateful but needy",
    "formal and businesslike",
    "casual and chatty",
    "defensive and guarded",
    "enthusiastic but scattered",
    "resigned and low-energy",
]

SCENARIO_TO_MESSAGE_PROMPT = """\
You are generating realistic user messages for testing an AI job search assistant called Eve.

Given a scenario description, generate a realistic user message (or multi-turn conversation) that a real person would type into a chat interface. The message should:
- Sound natural and conversational (not like a description or instruction)
- Be written from the user's perspective in first person
- Include realistic details (Australian cities, real job titles, plausible personal situations)
- Adopt the specified user tone/emotional register throughout

User tone: {tone}

Scenario: {scenario}
Target dimension being tested: {dimension} (target: {target})
Number of conversation turns to generate: {num_turns}

Respond in JSON format:
- If num_turns is 1, return: {{"user_message": "the message", "scenario_label": "<one of: job_search, resume, general>"}}
- If num_turns > 1, return: {{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...], "scenario_label": "multi_turn"}}
  The last turn MUST be a "user" turn. Assistant turns should be brief, natural placeholders.

Return ONLY valid JSON, no markdown fences."""


async def scenario_to_message(
    client: AsyncOpenAI,
    scenario: str,
    dimension: str,
    target: str,
    num_turns: int,
    model: str,
    tone: str = "neutral and straightforward",
) -> dict:
    """Convert a DeepEval scenario description into a realistic user message."""
    prompt = SCENARIO_TO_MESSAGE_PROMPT.format(
        scenario=scenario,
        dimension=dimension,
        target=target,
        num_turns=num_turns,
        tone=tone,
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_scenarios(
    persona: dict,
    dimension: str,
    num_samples: int,
    model: str,
) -> list[dict]:
    """Use DeepEval Synthesizer to generate diverse scenarios for a dimension."""
    styling_config = build_styling_config(persona, dimension)
    evolution_config = build_evolution_config()

    synthesizer = Synthesizer(
        model=model,
        async_mode=True,
        conversational_styling_config=styling_config,
        evolution_config=evolution_config,
    )

    goldens = synthesizer.generate_conversational_goldens_from_scratch(
        num_goldens=num_samples,
    )

    return [
        {
            "scenario": g.scenario,
            "expected_outcome": g.expected_outcome or "",
            "dimension": dimension,
        }
        for g in goldens
    ]


async def convert_scenarios_to_samples(
    scenarios: list[dict],
    persona: dict,
    max_turns: int,
    multi_turn_ratio: float,
    model: str,
) -> list[dict]:
    """Convert scenario descriptions into realistic user messages via LLM."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(20)

    async def convert_one(idx: int, sc: dict) -> dict:
        async with semaphore:
            dimension = sc["dimension"]
            target = persona["dimensions"][dimension]["target"]

            # Decide number of turns
            if random.random() < multi_turn_ratio:
                num_turns = random.randint(2, max_turns) if max_turns > 1 else 1
            else:
                num_turns = 1

            tone = random.choice(USER_TONES)

            result = await scenario_to_message(
                client=client,
                scenario=sc["scenario"],
                dimension=dimension,
                target=target,
                num_turns=num_turns,
                model=model,
                tone=tone,
            )

            # Parse into our JSONL format
            context = ""
            user_message = ""
            turn_number = 1

            if "turns" in result and len(result["turns"]) > 1:
                # Multi-turn: build context from earlier turns, last user turn is the message
                turns = result["turns"]
                context_lines = []
                for turn in turns[:-1]:
                    role_label = "User" if turn["role"] == "user" else "Assistant"
                    context_lines.append(f"{role_label}: {turn['content']}")
                context = "\n".join(context_lines)
                user_message = turns[-1]["content"]
                turn_number = sum(1 for t in turns if t["role"] == "user")
            else:
                user_message = result.get("user_message", sc["scenario"])
                turn_number = 1

            scenario_label = result.get("scenario_label", "general")

            return {
                "question": user_message,
                "user_message": user_message,
                "context": context,
                "reference_response": sc["expected_outcome"],
                "scenario": scenario_label,
                "dimension_target": dimension,
                "conversation_id": f"gen_{dimension}_{idx+1:03d}",
                "turn_number": turn_number,
            }

    tasks = [convert_one(i, sc) for i, sc in enumerate(scenarios)]
    return await asyncio.gather(*tasks)


def partition_samples(
    all_samples: list[dict], partition_size: int  # noqa: ARG001
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test, balanced by dimension."""
    by_dimension: dict[str, list[dict]] = {}
    for s in all_samples:
        dim = s["dimension_target"]
        by_dimension.setdefault(dim, []).append(s)

    train, dev, test = [], [], []

    for dim, samples in by_dimension.items():
        random.shuffle(samples)
        per_partition = len(samples) // 3
        remainder = len(samples) % 3

        sizes = [per_partition] * 3
        for i in range(remainder):
            sizes[i] += 1

        train.extend(samples[: sizes[0]])
        dev.extend(samples[sizes[0] : sizes[0] + sizes[1]])
        test.extend(samples[sizes[0] + sizes[1] :])

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    return train, dev, test


def write_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Eve evaluation samples targeting persona dimensions"
    )
    parser.add_argument(
        "--partition_size",
        type=int,
        default=25,
        help="Number of samples per partition (train/dev/test). Total = 3x this value.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Maximum conversation turns for multi-turn samples.",
    )
    parser.add_argument(
        "--multi_turn_ratio",
        type=float,
        default=0.3,
        help="Fraction of samples that are multi-turn (default 0.3).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="LLM model for generation (default gpt-4.1).",
    )
    parser.add_argument(
        "--persona_config",
        type=str,
        default=str(PERSONA_CONFIG_PATH),
        help="Path to persona config YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partitioning.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    persona = load_persona_config(Path(args.persona_config))
    dimensions = list(persona["dimensions"].keys())
    total_samples = args.partition_size * 3

    # Distribute samples across dimensions (roughly equal, empathy gets remainder)
    per_dimension = total_samples // len(dimensions)
    remainder = total_samples % len(dimensions)
    dimension_counts = {d: per_dimension for d in dimensions}
    if "empathy" in dimension_counts and remainder > 0:
        dimension_counts["empathy"] += remainder
    else:
        for i, d in enumerate(dimensions):
            if i < remainder:
                dimension_counts[d] += 1

    print(f"Persona: {persona['name']}")
    print(f"Total samples: {total_samples} ({args.partition_size} per partition)")
    print(f"Max turns: {args.max_turns}")
    print(f"Multi-turn ratio: {args.multi_turn_ratio}")
    print(f"Model: {args.model}")
    print(f"Distribution: {dimension_counts}")
    print()

    # Step 1: Generate scenarios via DeepEval Synthesizer
    all_scenarios = []
    for dimension, count in dimension_counts.items():
        print(f"Generating {count} scenarios for '{dimension}'...")
        scenarios = generate_scenarios(
            persona=persona,
            dimension=dimension,
            num_samples=count,
            model=args.model,
        )
        all_scenarios.extend(scenarios)
        print(f"  Generated {len(scenarios)} scenarios")

    print(f"\nTotal scenarios: {len(all_scenarios)}")

    # Step 2: Convert scenarios into realistic user messages
    print("Converting scenarios to realistic user messages...")
    all_samples = asyncio.run(
        convert_scenarios_to_samples(
            scenarios=all_scenarios,
            persona=persona,
            max_turns=args.max_turns,
            multi_turn_ratio=args.multi_turn_ratio,
            model=args.model,
        )
    )
    print(f"Converted {len(all_samples)} samples")

    # Step 3: Partition into train/dev/test
    print("Partitioning into train/dev/test...")
    train, dev, test = partition_samples(list(all_samples), args.partition_size)

    write_jsonl(train, output_dir / "eve_train.jsonl")
    write_jsonl(dev, output_dir / "eve_dev.jsonl")
    write_jsonl(test, output_dir / "eve_test.jsonl")

    print(f"\nDone!")
    print(f"  Train: {len(train)} samples")
    print(f"  Dev:   {len(dev)} samples")
    print(f"  Test:  {len(test)} samples")


if __name__ == "__main__":
    main()
