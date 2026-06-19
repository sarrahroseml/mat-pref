"""Baseline evaluation via OpenAI-compatible APIs.

Runs test splits against models in multiple context regimes:
  1. Zero-shot:          just the question
  2. Few-shot (5-shot):  5 worked examples + question
  3. Chemical-context:   question + Shannon radii / electronegativity
  4. Property-augmented: question + actual MP property values

Handles both 3-choice and 4-choice questions, forward and reverse goals.

Usage:
    python scripts/run_baselines.py                                    # full test splits, all regimes
    python scripts/run_baselines.py --model qwen3-8b                   # specific model
    python scripts/run_baselines.py --split test_iid                   # single split
    python scripts/run_baselines.py --regime zero_shot                 # single regime
    python scripts/run_baselines.py --n 50                             # sample 50 questions per split
    python scripts/run_baselines.py --workers 10                       # concurrency

Requires:
    pip install openai
    export DEEPINFRA_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "baselines"

TRAIN_PATH = DATA_DIR / "questions" / "train.jsonl"
CHEM_DESC_PATH = DATA_DIR / "chem_descriptors.json"

SPLIT_PATHS = {
    "test_iid": DATA_DIR / "questions" / "test_iid.jsonl",
    "test_ood_host": DATA_DIR / "questions" / "test_ood_host.jsonl",
    "test_ood_property": DATA_DIR / "questions" / "test_ood_property.jsonl",
}

# ---------------------------------------------------------------------------
# Provider / model config
# ---------------------------------------------------------------------------

PROVIDERS = {
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}

MODEL_ALIASES = {
    "qwen3-8b": ("deepinfra", "Qwen/Qwen3-14B"),  # Qwen3-8B not on DeepInfra; 14B is closest
    "qwen-72b": ("deepinfra", "Qwen/Qwen2.5-72B-Instruct"),
    "qwen3-235b-thinking": ("deepinfra", "Qwen/Qwen3-235B-A22B-Thinking-2507"),
    "qwen3-235b-instruct": ("deepinfra", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    "llama-3.3-70b": ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "deepseek-r1": ("deepinfra", "deepseek-ai/DeepSeek-R1"),
    "deepseek-v3": ("deepinfra", "deepseek-ai/DeepSeek-V3"),
    "mixtral-8x22b": ("fireworks", "accounts/fireworks/models/mixtral-8x22b-instruct"),
    "qwen-7b": ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "llama-8b": ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    "llama-70b": ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_questions(path: Path, n: int, seed: int = 42) -> list[dict]:
    """Stratified sample: proportional to (target_property x difficulty x direction)."""
    qs = load_jsonl(path)
    rng = random.Random(seed)

    buckets: dict[tuple[str, ...], list[dict]] = {}
    for q in qs:
        key = (q["target_property"], q["difficulty"],
               q.get("direction", "forward"))
        buckets.setdefault(key, []).append(q)

    sampled: list[dict] = []
    total = len(qs)
    for key, items in buckets.items():
        k = max(1, round(n * len(items) / total))
        sampled.extend(rng.sample(items, min(k, len(items))))

    rng.shuffle(sampled)
    return sampled[:n]


def pick_few_shot_examples(
    train_path: Path, n: int = 5, seed: int = 99,
) -> list[dict]:
    """Pick n diverse training questions covering direction, property, n_choices."""
    qs = load_jsonl(train_path)
    rng = random.Random(seed)

    # Target: 1 forward formation_energy, 1 reverse formation_energy,
    #         1 forward band_gap, 1 reverse band_gap, 1 three-choice
    targets = [
        ("formation_energy_per_atom", "forward", None),
        ("formation_energy_per_atom", "reverse", None),
        ("band_gap", "forward", None),
        ("band_gap", "reverse", None),
        (None, None, 3),  # any 3-choice
    ]

    examples: list[dict] = []
    used_ids: set[str] = set()

    for prop, direction, n_choices in targets:
        matches = [
            q for q in qs
            if q["question_id"] not in used_ids
            and (prop is None or q["target_property"] == prop)
            and (direction is None or q.get("direction", "forward") == direction)
            and (n_choices is None or q.get("n_choices", 4) == n_choices)
        ]
        if matches:
            pick = rng.choice(matches)
            examples.append(pick)
            used_ids.add(pick["question_id"])

    # Fill remaining slots randomly
    remaining = [q for q in qs if q["question_id"] not in used_ids]
    rng.shuffle(remaining)
    for q in remaining:
        if len(examples) >= n:
            break
        examples.append(q)
        used_ids.add(q["question_id"])

    return examples[:n]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _answer_labels(q: dict) -> str:
    """Return comma-separated valid labels: 'A, B, C' or 'A, B, C, D'."""
    n = q.get("n_choices", len(q["candidates"]))
    return ", ".join(chr(65 + i) for i in range(n))


def _format_question_text(q: dict) -> str:
    """Render the core question (no property values)."""
    labels = _answer_labels(q)
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


def _format_question_with_properties(q: dict) -> str:
    """Render the question with actual MP property values included."""
    labels = _answer_labels(q)
    prop = q["target_property"]
    prop_label = {
        "formation_energy_per_atom": "Formation energy (eV/atom)",
        "band_gap": "Band gap (eV)",
    }.get(prop, prop)

    lines = [
        f"Host material: {q['host_formula']} ({q['structure_family']})",
        f"Site to substitute: {q['site_label']}",
        f'Design goal: "{q["design_goal"]}"',
        "",
        "Candidates (with computed properties from Materials Project):",
    ]
    for c in q["candidates"]:
        val = c.get("property_value", "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        lines.append(
            f"  ({c['label']}) {c['element']} \u2192 {c['formula']}  "
            f"[{prop_label}: {val}]"
        )
    lines.append("")
    lines.append(
        "Given these property values and your chemistry knowledge, which "
        "substitution best achieves the design goal? Explain your reasoning, "
        f'then give your final answer as "Answer: (X)" where X is {labels}.'
    )
    return "\n".join(lines)


def _make_few_shot_demo(q: dict) -> str:
    """Create a single worked example for the few-shot prompt."""
    prompt = _format_question_text(q)
    correct = q["correct_answer"]
    correct_cand = next(c for c in q["candidates"] if c["label"] == correct)

    answer = (
        f"The best substitution is ({correct}) {correct_cand['element']} "
        f"\u2192 {correct_cand['formula']}.\n\n"
        f"Reasoning: Among the candidates, {correct_cand['element']} at the "
        f"{q['site_label']} produces {correct_cand['formula']}, which best "
        f"satisfies the goal to {q['design_goal'].lower()}. The other "
        f"candidates are less favorable for this property.\n\n"
        f"Answer: ({correct})"
    )
    return f"Question:\n{prompt}\n\nAnswer:\n{answer}"


SYSTEM_PROMPT = (
    "You are an expert materials scientist. You will be given a multiple-choice "
    "question about ionic substitution in crystalline materials. Reason briefly "
    "about which substitution best meets the design goal, considering "
    "factors like ionic radius, electronegativity, charge balance, and known "
    "stability trends. Keep your reasoning concise (under 200 words). "
    'End your response with "Answer: (X)" where X is the letter of your chosen option.'
)


# ---------------------------------------------------------------------------
# Chemical-context augmented prompt
# ---------------------------------------------------------------------------

_CHEM_DESC_CACHE: dict | None = None


def _load_chem_descriptors() -> dict:
    global _CHEM_DESC_CACHE
    if _CHEM_DESC_CACHE is None:
        with open(CHEM_DESC_PATH) as f:
            _CHEM_DESC_CACHE = json.load(f)
    return _CHEM_DESC_CACHE


def _format_chem_descriptor(desc: dict) -> str:
    """Format a single descriptor as a compact string."""
    parts = []
    ox = desc.get("oxidation_state")
    r = desc.get("ionic_radius_A")
    if r is not None:
        parts.append(f"Shannon radius({'+' if ox > 0 else ''}{ox})={r:.3f} \u00c5")
    en = desc.get("pauling_electronegativity")
    if en is not None:
        parts.append(f"Pauling EN={en:.2f}")
    parts.append(f"d-electrons={desc.get('d_electron_count', 0)}")
    parts.append(f"CN={desc.get('coordination_number', '?')}")
    pref = desc.get("coordination_preferences", "")
    if pref:
        parts.append(pref)
    return "; ".join(parts)


def _format_question_with_chem_context(q: dict) -> str:
    """Render question with chemical descriptors (no DFT values)."""
    labels = _answer_labels(q)
    lookup = _load_chem_descriptors()

    lines = [
        f"Host material: {q['host_formula']} ({q['structure_family']})",
        f"Site to substitute: {q['site_label']}",
        f'Design goal: "{q["design_goal"]}"',
        "",
        "Candidates (with chemical descriptors for the substitution site):",
    ]
    for c in q["candidates"]:
        key = f"{c['element']}|{q['structure_family']}|{q['site_label']}"
        desc = lookup.get(key)
        if desc:
            desc_text = _format_chem_descriptor(desc)
        else:
            desc_text = "descriptors unavailable"
        lines.append(
            f"  ({c['label']}) {c['element']} \u2192 {c['formula']}  "
            f"[{desc_text}]"
        )
    lines.append("")
    lines.append(
        "Using these chemical descriptors and your materials science knowledge, "
        "which substitution best achieves the design goal? Explain your reasoning, "
        f'then give your final answer as "Answer: (X)" where X is {labels}.'
    )
    return "\n".join(lines)


def build_messages(
    q: dict,
    regime: str,
    few_shot_examples: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Build the message list for the API call."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if regime == "zero_shot":
        messages.append({"role": "user", "content": _format_question_text(q)})

    elif regime == "few_shot":
        assert few_shot_examples, "few_shot regime requires examples"
        demos = "\n\n---\n\n".join(
            _make_few_shot_demo(ex) for ex in few_shot_examples
        )
        user_content = (
            "Here are some worked examples:\n\n"
            f"{demos}\n\n"
            "---\n\n"
            "Now answer the following question:\n\n"
            f"{_format_question_text(q)}"
        )
        messages.append({"role": "user", "content": user_content})

    elif regime == "chemical_context":
        messages.append(
            {"role": "user", "content": _format_question_with_chem_context(q)}
        )

    elif regime == "property_augmented":
        messages.append(
            {"role": "user", "content": _format_question_with_properties(q)}
        )

    else:
        raise ValueError(f"Unknown regime: {regime}")

    return messages


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

    If the parsed letter is outside the valid range for n_choices
    (e.g., "D" for a 3-choice question), returns None — this counts
    as an incorrect answer, not a parse failure.
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

    # Validate against number of choices
    valid_labels = [chr(65 + i) for i in range(n_choices)]
    if letter not in valid_labels:
        return None  # hallucinated option — wrong, not a parse failure

    return letter


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_model(
    messages: list[dict[str, str]],
    model: str,
    client: Any,
    max_tokens: int = 4096,
    max_retries: int = 5,
) -> str:
    """Call API with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                k in err_str for k in ("rate", "429", "quota", "503", "timeout")
            )
            if not is_retryable or attempt == max_retries:
                raise
            wait = min(2 ** attempt * 5 + random.uniform(0, 2), 120)
            log.warning("Retryable error (attempt %d/%d), waiting %.0fs\u2026",
                        attempt + 1, max_retries, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _eval_one(
    q: dict,
    regime: str,
    model: str,
    client: Any,
    few_shot_examples: list[dict] | None,
    delay: float = 0.0,
) -> dict:
    """Evaluate a single question."""
    qid = q["question_id"]
    expected = q["correct_answer"]
    n_choices = q.get("n_choices", len(q["candidates"]))
    messages = build_messages(q, regime, few_shot_examples)

    if delay:
        time.sleep(delay)
    try:
        response_text = call_model(messages, model, client)
    except Exception as e:
        log.error("All retries failed for %s: %s", qid[:8], e)
        response_text = f"ERROR: {e}"

    predicted = parse_answer(response_text, n_choices=n_choices)
    return {
        "question_id": qid,
        "split": "",  # filled by caller
        "regime": regime,
        "model": model,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
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


def evaluate(
    questions: list[dict],
    regime: str,
    model: str,
    client: Any,
    few_shot_examples: list[dict] | None = None,
    workers: int = 1,
    delay: float = 0.1,
) -> list[dict]:
    """Run evaluation on a list of questions in a given regime."""
    results: list[dict] = []
    correct = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_one, q, regime, model, client, few_shot_examples, delay): q
            for q in questions
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            if r["correct"]:
                correct += 1
            acc = correct / done * 100
            if done % 50 == 0 or done == len(questions):
                log.info(
                    "[%s] %d/%d  acc=%.1f%%  (last: q=%s %s\u2192%s %s)",
                    regime, done, len(questions), acc,
                    r["question_id"][:8], r["expected"], r["predicted"],
                    "\u2713" if r["correct"] else "\u2717",
                )

    return results


def print_summary(all_results: list[dict], model_name: str) -> dict:
    """Print and return accuracy summary with full breakdowns."""
    summary: dict[str, Any] = {"model": model_name}

    by_split_regime: dict[tuple[str, str], list[dict]] = {}
    for r in all_results:
        key = (r["split"], r["regime"])
        by_split_regime.setdefault(key, []).append(r)

    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS \u2014 {model_name}")
    print(f"{'='*70}")

    for (split, regime), results in sorted(by_split_regime.items()):
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        acc = correct / total * 100 if total else 0
        unparsed = sum(1 for r in results if r["predicted"] is None)

        print(f"\n{split} / {regime}: {correct}/{total} = {acc:.1f}%", end="")
        if unparsed:
            print(f"  ({unparsed} parse failures)", end="")
        print()

        summary.setdefault(split, {})[regime] = {
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": total,
            "parse_failures": unparsed,
        }

        # By property
        _print_breakdown(results, "target_property", "  Property")

        # By direction
        _print_breakdown(results, "direction", "  Direction")

        # By n_choices
        for nc in sorted(set(r["n_choices"] for r in results)):
            rs = [r for r in results if r["n_choices"] == nc]
            c = sum(1 for r in rs if r["correct"])
            print(f"  {nc}-choice: {c}/{len(rs)} = {c/len(rs)*100:.1f}%")

        # By difficulty
        _print_breakdown(results, "difficulty", "  Difficulty")

        # Per-family breakdown for OOD-host
        if split == "test_ood_host":
            _print_breakdown(results, "structure_family", "  Family")

    print(f"\n{'='*70}")
    return summary


def _print_breakdown(results: list[dict], field: str, prefix: str) -> None:
    """Print accuracy breakdown by a given field."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r.get(field, "unknown"), []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        c = sum(1 for r in rs if r["correct"])
        print(f"{prefix} {key}: {c}/{len(rs)} = {c/len(rs)*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--model", default="qwen3-8b",
        choices=list(MODEL_ALIASES.keys()),
        help="Model alias (default: qwen3-8b)",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Questions to sample per split (0 = full, default: 0)",
    )
    parser.add_argument(
        "--split", default="all",
        choices=["test_iid", "test_ood_host", "test_ood_property", "all"],
        help="Which test split(s) (default: all)",
    )
    parser.add_argument(
        "--regime", default="all",
        choices=["zero_shot", "few_shot", "chemical_context",
                 "property_augmented", "all"],
        help="Which context regime(s) (default: all)",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    provider_name, model_id = MODEL_ALIASES[args.model]
    base_url, env_var = PROVIDERS[provider_name]

    api_key = os.environ.get(env_var)
    if not api_key:
        log.error("Set %s environment variable.", env_var)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        log.error("Install openai: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    log.info("Using provider=%s model=%s", provider_name, model_id)
    splits = list(SPLIT_PATHS.keys()) if args.split == "all" else [args.split]
    regimes = (
        ["zero_shot", "few_shot", "chemical_context", "property_augmented"]
        if args.regime == "all" else [args.regime]
    )

    few_shot_examples = pick_few_shot_examples(TRAIN_PATH, n=5, seed=99)
    log.info("Picked %d few-shot examples.", len(few_shot_examples))

    all_results: list[dict] = []
    for split_name in splits:
        path = SPLIT_PATHS[split_name]
        if args.n > 0:
            questions = sample_questions(path, args.n, seed=args.seed)
            log.info("Sampled %d questions from %s.", len(questions), split_name)
        else:
            questions = load_jsonl(path)
            log.info("Loaded full %s: %d questions.", split_name, len(questions))

        for regime in regimes:
            log.info("=" * 50)
            log.info("%s / %s \u2014 %d questions, model=%s",
                     split_name, regime, len(questions), args.model)
            log.info("=" * 50)
            results = evaluate(
                questions, regime, model_id, client,
                few_shot_examples=few_shot_examples if regime == "few_shot" else None,
                workers=args.workers,
                delay=args.delay,
            )
            for r in results:
                r["split"] = split_name
            all_results.extend(results)

    summary = print_summary(all_results, args.model)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = args.model.replace("-", "_")

    results_path = RESULTS_DIR / f"eval_{tag}_{ts}.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    summary_path = RESULTS_DIR / f"eval_{tag}_{ts}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Results: %s", results_path)
    log.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
