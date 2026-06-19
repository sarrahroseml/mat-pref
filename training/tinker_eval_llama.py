"""Evaluate Qwen3-8B via Tinker — base model or fine-tuned checkpoint.

Usage:
    # Base model (no fine-tuning) on all splits
    python scripts/tinker_eval.py --base

    # SFT checkpoint on test_iid
    python scripts/tinker_eval.py --checkpoint runs/sft_qwen3_8b_v2 --split test_iid

    # GRPO checkpoint on all splits
    python scripts/tinker_eval.py --checkpoint runs/grpo_qwen3_8b --split all

    # Sample N questions
    python scripts/tinker_eval.py --base --n 100

Requires:
    pip install tinker-cookbook
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import tinker

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "baselines"

SPLIT_PATHS = {
    "test_iid": DATA_DIR / "questions" / "test_iid.jsonl",
    "test_ood_host": DATA_DIR / "questions" / "test_ood_host.jsonl",
    "test_ood_property": DATA_DIR / "questions" / "test_ood_property.jsonl",
}

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
# Answer parsing
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(
    r"(?:Answer:\s*\**\(?([A-Da-d])\)?\**"
    r"|\*\*\(?([A-Da-d])\)?\*\*\s*$"
    r"|^\s*\(?([A-Da-d])\)?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_answer(text: str, n_choices: int = 4) -> str | None:
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
        return None  # hallucinated option
    return letter


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    sampling_client,
    tokenizer,
    renderer,
    questions: list[dict],
    split_name: str,
    out_path: Path,
    model_tag: str = "llama-3p1-8b-instruct-base",
    max_tokens: int = 1024,
    pipeline_depth: int = 16,
) -> list[dict]:
    """Pipelined Tinker eval: keep `pipeline_depth` futures in flight."""
    from collections import deque

    # Load existing results for resume
    done_ids: set[str] = set()
    results: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r["question_id"])
        log.info("Resuming: %d already evaluated", len(done_ids))

    remaining = [q for q in questions if q["question_id"] not in done_ids]
    correct = sum(1 for r in results if r["correct"])
    total_done = len(results)

    log.info("Evaluating %d remaining (%d done)  pipeline_depth=%d",
             len(remaining), total_done, pipeline_depth)

    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["\n\nHost material:"],
    )

    def issue(q):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_question(q)},
        ]
        prompt = renderer.build_generation_prompt(messages)
        return sampling_client.sample(prompt=prompt, sampling_params=params,
                                      num_samples=1)

    inflight = deque()
    next_idx = 0
    while next_idx < len(remaining) and len(inflight) < pipeline_depth:
        inflight.append((remaining[next_idx], issue(remaining[next_idx])))
        next_idx += 1

    completed = 0
    t_start = time.time()
    while inflight:
        q, future = inflight.popleft()
        qid = q["question_id"]
        expected = q["correct_answer"]
        n_choices = q.get("n_choices", len(q["candidates"]))
        result = future.result()
        response_text = tokenizer.decode(result.sequences[0].tokens)

        predicted = parse_answer(response_text, n_choices=n_choices)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        r = {
            "question_id": qid,
            "split": split_name,
            "regime": "zero_shot",
            "model": model_tag,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "host_formula": q["host_formula"],
            "structure_family": q["structure_family"],
            "site_label": q["site_label"],
            "target_property": q["target_property"],
            "difficulty": q["difficulty"],
            "direction": q.get("direction", "forward"),
            "n_choices": n_choices,
            "gap": q["gap"],
            "response": response_text,
        }
        results.append(r)

        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")

        completed += 1
        if next_idx < len(remaining):
            inflight.append((remaining[next_idx], issue(remaining[next_idx])))
            next_idx += 1

        total_so_far = total_done + completed
        if total_so_far % 20 == 0 or total_so_far == len(questions):
            acc = correct / total_so_far * 100
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - completed) / rate if rate > 0 else 0
            log.info("[%s] %d/%d  acc=%.1f%%  rate=%.2f q/s  ETA=%.0fs",
                     split_name, total_so_far, len(questions), acc, rate, eta)

    return results


def print_summary(all_results: list[dict], model_tag: str) -> None:
    by_split = defaultdict(list)
    for r in all_results:
        by_split[r["split"]].append(r)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS \u2014 {model_tag}")
    print(f"{'='*70}")

    for split_name, rs in sorted(by_split.items()):
        correct = sum(1 for r in rs if r["correct"])
        total = len(rs)
        acc = correct / total * 100
        pf = sum(1 for r in rs if r["predicted"] is None)
        print(f"\n{split_name}: {correct}/{total} = {acc:.1f}%", end="")
        if pf:
            print(f"  ({pf} parse failures)", end="")
        print()

        _breakdown(rs, "target_property", "  Property")
        _breakdown(rs, "direction", "  Direction")

        for nc in sorted(set(r.get("n_choices", 4) for r in rs)):
            sub = [r for r in rs if r.get("n_choices", 4) == nc]
            c = sum(1 for r in sub if r["correct"])
            print(f"  {nc}-choice: {c}/{len(sub)} = {c/len(sub)*100:.1f}%")

        _breakdown(rs, "difficulty", "  Difficulty")

        if split_name == "test_ood_host":
            _breakdown(rs, "structure_family", "  Family")

    print(f"\n{'='*70}")


def _breakdown(results: list[dict], field: str, prefix: str) -> None:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r.get(field, "unknown"), []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        c = sum(1 for r in rs if r["correct"])
        print(f"{prefix} {key}: {c}/{len(rs)} = {c/len(rs)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B via Tinker")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to Tinker run directory (omit for base model)")
    parser.add_argument("--base", action="store_true",
                        help="Evaluate base Qwen3-8B without fine-tuning")
    parser.add_argument("--split", default="all",
                        choices=["test_iid", "test_ood_host", "test_ood_property", "all"])
    parser.add_argument("--n", type=int, default=0, help="Sample N questions (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSONL to resume from")
    args = parser.parse_args()

    if not args.base and not args.checkpoint:
        parser.error("Provide --checkpoint or --base")

    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    if args.base:
        log.info("Creating sampling client for base %s...", MODEL_NAME)
        sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
        model_tag = "llama-3p1-8b-instruct-base"
        checkpoint_name = "base"
    else:
        resume_info = checkpoint_utils.get_last_checkpoint(args.checkpoint)
        if not resume_info:
            log.error("No checkpoint found in %s", args.checkpoint)
            return

        info = resume_info if isinstance(resume_info, dict) else vars(resume_info)
        sampler_path = info.get("sampler_path")
        if not sampler_path:
            log.info("No sampler_path found, creating from state_path...")
            state_path = info.get("state_path")
            training_client = service_client.create_training_client_from_state_with_optimizer(state_path)
            sampling_client = training_client.save_weights_and_get_sampling_client(name="eval")
        else:
            log.info("Using sampler_path: %s", sampler_path)
            sampling_client = service_client.create_sampling_client(model_path=sampler_path)
        model_tag = f"llama-3p1-8b-{Path(args.checkpoint).name}"
        checkpoint_name = Path(args.checkpoint).name

    log.info("Model ready: %s", model_tag)

    splits = list(SPLIT_PATHS.keys()) if args.split == "all" else [args.split]
    all_results = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = (Path(args.resume) if args.resume
                else RESULTS_DIR / f"eval_{checkpoint_name}_{ts}.jsonl")

    for split_name in splits:
        path = SPLIT_PATHS[split_name]
        with open(path) as f:
            questions = [json.loads(l) for l in f if l.strip()]

        if args.n > 0:
            import random
            rng = random.Random(args.seed)
            questions = rng.sample(questions, min(args.n, len(questions)))

        log.info("Evaluating %s: %d questions", split_name, len(questions))
        results = evaluate_model(
            sampling_client, tokenizer, renderer,
            questions, split_name, out_path, model_tag,
        )
        all_results.extend(results)

    print_summary(all_results, model_tag)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
