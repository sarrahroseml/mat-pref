"""GRPO training for Qwen3-8B on materials science substitution task.

Uses Tinker's RL framework with a binary reward (correct answer = 1, wrong = 0).
Starts from the SFT v2 checkpoint and runs group-relative policy optimization.

Handles both 3-choice and 4-choice questions, forward and reverse goals.

Usage:
    # Full run (~200 steps, ~2-3 hours)
    python scripts/tinker_grpo.py

    # Resume from checkpoint
    python scripts/tinker_grpo.py load_checkpoint_path=runs/grpo_llama_3p1_8b

    # Different seed for multi-seed runs
    python scripts/tinker_grpo.py dataset_builder.seed=43

Requires:
    pip install tinker-cookbook
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Sequence

import chz
import tinker

from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "questions" / "train.jsonl"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert materials scientist. You will be given a multiple-choice "
    "question about ionic substitution in crystalline materials. Reason briefly "
    "about which substitution best meets the design goal, considering "
    "factors like ionic radius, electronegativity, charge balance, and known "
    "stability trends. Keep your reasoning concise (under 200 words). "
    'End your response with "Answer: (X)" where X is the letter of your chosen option.'
)

# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"(?:Answer:\s*\**\(?([A-Da-d])\)?\**"
    r"|\*\*\(?([A-Da-d])\)?\*\*\s*$"
    r"|^\s*\(?([A-Da-d])\)?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_answer(text: str, n_choices: int = 4) -> str | None:
    """Extract the final Answer: (X) from model output.

    Validates against n_choices: if the model outputs "D" on a 3-choice
    question, returns None (reward=0), not a match.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    search_text = cleaned if cleaned else text
    matches = _ANSWER_RE.findall(search_text)
    if not matches:
        return None
    last = matches[-1]
    letter = next((g for g in last if g), None)
    if letter is None:
        return None
    letter = letter.upper()
    valid = [chr(65 + i) for i in range(n_choices)]
    if letter not in valid:
        return None  # hallucinated option — wrong, not parse error
    return letter


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------


def format_question(q: dict) -> str:
    n = q.get("n_choices", len(q["candidates"]))
    labels = ", ".join(chr(65 + i) for i in range(n))

    lines = [
        f"Host material: {q['host_formula']} ({q['structure_family']})",
        f"Site to substitute: {q['site_label']}",
        f'Design goal: "{q["design_goal"]}"',
        "",
        "Candidates:",
    ]
    for c in q["candidates"]:
        lines.append(f"  ({c['label']}) {c['element']} \u2192 {c['formula']}")
    lines.append("")
    lines.append(
        "Which substitution best achieves the design goal? "
        "Explain your reasoning, then give your final answer as "
        f'"Answer: (X)" where X is {labels}.'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------------------


class SubstitutionEnv(Env):
    """Single-turn environment for ionic substitution questions.

    The model sees the question as the initial observation, generates a
    response, and receives reward = 1 if correct, 0 otherwise.
    """

    def __init__(
        self,
        question: dict,
        renderer: renderers.Renderer,
        tokenizer: Any,
    ):
        self.question = question
        self.correct_answer = question["correct_answer"]
        self.n_choices = question.get("n_choices", len(question["candidates"]))
        self.renderer = renderer
        self.tokenizer = tokenizer

    async def initial_observation(self):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_question(self.question)},
        ]
        prompt = self.renderer.build_generation_prompt(messages)
        stop_sequences = self.renderer.get_stop_sequences()
        return prompt, stop_sequences

    async def step(self, action):
        response_text = self.tokenizer.decode(action)
        predicted = parse_answer(response_text, n_choices=self.n_choices)

        reward = 1.0 if predicted == self.correct_answer else 0.0

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={
                "correct": int(reward),
                "parsed": 1 if predicted is not None else 0,
            },
        )


class SubstitutionEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of substitution environments for one training step."""

    def __init__(
        self,
        questions: list[dict],
        group_size: int,
        renderer: renderers.Renderer,
        tokenizer: Any,
    ):
        self.questions = questions
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer

    async def make_envs(self) -> Sequence[Env]:
        return [
            SubstitutionEnv(q, self.renderer, self.tokenizer)
            for q in self.questions
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(self, trajectory_group, env_group):
        """Return list of (reward, metrics) for each trajectory."""
        results = []
        for traj, env in zip(trajectory_group, env_group):
            total_reward = sum(t.reward for t in traj.transitions)
            correct = 1 if total_reward > 0 else 0
            results.append((total_reward, {"correct": correct}))
        return results

    def logging_tags(self) -> dict[str, str]:
        return {"task": "substitution"}


@chz.chz
class SubstitutionDatasetBuilder(RLDatasetBuilder):
    """Dataset builder for the substitution RL task."""

    batch_size: int = 64      # questions per batch
    group_size: int = 8       # rollouts per question
    n_questions: int = 0      # 0 = all 7,287 training questions
    seed: int = 42
    renderer_name: str | None = None
    model_name_for_tokenizer: str = MODEL_NAME

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        with open(TRAIN_PATH) as f:
            all_questions = [json.loads(l) for l in f if l.strip()]

        if self.n_questions > 0:
            rng = random.Random(self.seed)
            buckets: dict[tuple[str, ...], list[dict]] = {}
            for q in all_questions:
                key = (q["target_property"], q["difficulty"],
                       q.get("direction", "forward"))
                buckets.setdefault(key, []).append(q)
            total = len(all_questions)
            sampled: list[dict] = []
            for key, items in buckets.items():
                k = max(1, round(self.n_questions * len(items) / total))
                sampled.extend(rng.sample(items, min(k, len(items))))
            rng.shuffle(sampled)
            questions = sampled[:self.n_questions]
        else:
            questions = all_questions

        logger.info("Loaded %d questions for GRPO", len(questions))

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        rname = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        renderer = renderers.get_renderer(rname, tokenizer)

        rng = random.Random(self.seed)
        rng.shuffle(questions)
        n_batches = len(questions) // self.batch_size
        if n_batches == 0:
            n_batches = 1

        batches = []
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(questions))
            batch_qs = questions[start:end]
            batches.append(
                SubstitutionEnvGroupBuilder(
                    questions=batch_qs,
                    group_size=self.group_size,
                    renderer=renderer,
                    tokenizer=tokenizer,
                )
            )

        class SubstitutionRLDataset(RLDataset):
            def __init__(self, env_group_builders):
                self._builders = env_group_builders

            def __len__(self):
                return len(self._builders)

            def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
                return [self._builders[index]]

        train_dataset = SubstitutionRLDataset(batches)
        return train_dataset, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# SFT v2 checkpoint (template-based, 7,130 traces, 1 epoch)
SFT_CHECKPOINT = "tinker://fe8023c9-9484-591d-83bd-8a9f8b754be5:train:0/weights/final"


def build_config_blueprint() -> chz.Blueprint[rl_train.Config]:
    builder = SubstitutionDatasetBuilder(
        batch_size=64,
        group_size=8,
        n_questions=0,      # 0 = all 7,287 training questions
    )

    return chz.Blueprint(rl_train.Config).apply(
        {
            "model_name": MODEL_NAME,
            "log_path": str(PROJECT_ROOT / "runs" / "grpo_llama_3p1_8b"),
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 1024,
            "temperature": 0.8,
            "loss_fn": "importance_sampling",
            "lora_rank": 32,
            "eval_every": 20,
            "save_every": 20,
            "remove_constant_reward_groups": True,
            "load_checkpoint_path": SFT_CHECKPOINT,
        }
    )


def main(config: rl_train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(rl_train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(__import__("sys").argv[1:])
    main(blueprint.make())
