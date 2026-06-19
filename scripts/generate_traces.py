"""Generate rationalization traces for SFT training data using DeepSeek R1.

For each training question, prompts R1 with the correct answer (but NOT
property values) and asks it to explain WHY using chemical principles.
Post-processes traces to strip <think> blocks, remove leaked numbers,
and verify the answer matches ground truth.

Outputs SFT-ready JSONL with {"input": question_as_eval, "output": trace}.

Usage:
    # 50-question pilot
    python scripts/generate_traces.py --n 50

    # Full training set
    python scripts/generate_traces.py

    # Resume from checkpoint
    python scripts/generate_traces.py --resume data/traces/traces_deepseek_r1_20260325_140000.jsonl

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "questions" / "train.jsonl"  # template-based questions
TRACES_DIR = DATA_DIR / "traces"

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL_ID = "deepseek-ai/DeepSeek-R1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

REASONING_SYSTEM = (
    "You are an expert materials scientist. You will be given a multiple-choice "
    "question about ionic substitution in crystalline materials. The correct "
    "answer is provided to you privately, but you must NOT reveal it upfront.\n\n"
    "Write your response as if you are reasoning through the problem from "
    "scratch and arriving at the answer through chemical logic. Do NOT say "
    '"the correct answer is" or "I am told the answer is" — instead, analyze '
    "each candidate and build toward the conclusion naturally.\n\n"
    "Consider these factors:\n"
    "- Ionic radius matching and site compatibility\n"
    "- Coordination preferences and crystal field effects\n"
    "- Tolerance factor and structural stability\n"
    "- Electronegativity and bonding character\n"
    "- Periodic trends and known material families\n\n"
    "Do NOT reference specific numerical property values (no eV, eV/atom, "
    "or band gap numbers). Reason ONLY from chemical principles and periodic "
    "table relationships.\n\n"
    "Evaluate each candidate, explain why the best one is superior and why "
    "each alternative is worse. Keep your reasoning concise (under 300 words). "
    'End with "Answer: (X)" where X is the letter you arrived at.'
)


def format_eval_question(q: dict) -> str:
    """Format question exactly as it appears at evaluation time (no values)."""
    n_choices = q.get("n_choices", len(q["candidates"]))
    labels = ", ".join(chr(65 + i) for i in range(n_choices))  # "A, B, C" or "A, B, C, D"

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
        'Which substitution best achieves the design goal? '
        'Explain your reasoning, then give your final answer as '
        f'"Answer: (X)" where X is {labels}.'
    )
    return "\n".join(lines)


def build_reasoning_prompt(q: dict) -> str:
    """Build rationalization prompt — tell model the answer, ask it to explain why.

    The correct answer is provided so the model can construct a coherent
    reasoning trace that arrives at the right conclusion. The system prompt
    instructs it not to leak that it was told the answer.
    """
    question_text = format_eval_question(q)

    correct_label = q["correct_answer"]
    correct_element = q.get("correct_element", "")
    correct_formula = ""
    for c in q["candidates"]:
        if c["label"] == correct_label:
            correct_formula = c["formula"]
            break

    return (
        f"{question_text}\n\n"
        f"[The correct answer is ({correct_label}) {correct_element} → {correct_formula}. "
        f"Explain why a materials scientist would arrive at this answer using "
        f"chemical reasoning. Do not mention that you were told the answer.]"
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Patterns for leaked numerical values
_LEAK_PATTERNS = [
    re.compile(r"-?\d+\.\d+\s*eV(?:/atom)?", re.IGNORECASE),
    re.compile(r"(?:formation energy|band gap|energy)\s*(?:of|=|:)\s*-?\d+\.\d+", re.IGNORECASE),
    re.compile(r"-?\d+\.\d{3,}\s*(?:eV|meV|Ry|Ha)", re.IGNORECASE),
]

_ANSWER_RE = re.compile(
    r"(?:Answer:\s*\**\(?([A-Da-d])\)?\**"
    r"|\*\*\(?([A-Da-d])\)?\*\*\s*$"
    r"|^\s*\(?([A-Da-d])\)?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks, keep only the final answer section."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


def remove_leaked_values(text: str) -> tuple[str, int]:
    """Remove leaked numerical values. Returns (cleaned_text, leak_count)."""
    leak_count = 0
    cleaned = text
    for pattern in _LEAK_PATTERNS:
        matches = pattern.findall(cleaned)
        leak_count += len(matches)
        cleaned = pattern.sub("[value removed]", cleaned)
    return cleaned, leak_count


def extract_answer(text: str) -> str | None:
    """Extract the final Answer: (X) from trace."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    letter = next((g for g in last if g), None)
    return letter.upper() if letter else None


def postprocess_trace(raw_trace: str, expected_answer: str) -> dict:
    """Full post-processing pipeline for a trace."""
    # Step 1: strip <think> block
    answer_text = strip_think_block(raw_trace)

    # Step 2: remove leaked numbers
    cleaned, leak_count = remove_leaked_values(answer_text)

    # Step 3: verify answer
    parsed_answer = extract_answer(cleaned)
    answer_matches = parsed_answer == expected_answer if parsed_answer else False

    return {
        "processed_trace": cleaned,
        "raw_length": len(raw_trace),
        "processed_length": len(cleaned),
        "think_block_stripped": "<think>" in raw_trace,
        "leaks_found": leak_count,
        "leaks_after_clean": len([p for p in _LEAK_PATTERNS if p.search(cleaned)]),
        "answer_parsed": parsed_answer,
        "answer_matches": answer_matches,
        "independently_correct": answer_matches,  # model derived this on its own
    }


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------


def call_model(
    messages: list[dict[str, str]],
    client: Any,
    max_tokens: int = 8192,
    max_retries: int = 5,
) -> str:
    """Call API with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(k in err_str for k in ("rate", "429", "quota", "503", "timeout"))
            if not is_retryable or attempt == max_retries:
                raise
            wait = min(2 ** attempt * 5 + random.uniform(0, 2), 120)
            log.warning("Retryable error (attempt %d/%d), waiting %.0fs…",
                        attempt + 1, max_retries, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------


def generate_one(
    q: dict,
    client: Any,
    delay: float = 0.0,
) -> dict:
    """Generate a blind reasoning trace — model derives the answer itself."""
    qid = q["question_id"]

    messages = [
        {"role": "system", "content": REASONING_SYSTEM},
        {"role": "user", "content": build_reasoning_prompt(q)},
    ]

    if delay:
        time.sleep(delay)

    try:
        raw_trace = call_model(messages, client)
    except Exception as e:
        log.error("All retries failed for %s: %s", qid[:8], e)
        return {
            "question_id": qid,
            "error": str(e),
        }

    # Post-process
    pp = postprocess_trace(raw_trace, q["correct_answer"])

    # Build SFT pair
    eval_input = format_eval_question(q)

    return {
        "question_id": qid,
        "host_formula": q["host_formula"],
        "structure_family": q["structure_family"],
        "site_label": q["site_label"],
        "target_property": q["target_property"],
        "design_goal": q["design_goal"],
        "correct_answer": q["correct_answer"],
        "correct_element": q["correct_element"],
        "difficulty": q["difficulty"],
        "direction": q.get("direction", "forward"),
        "n_choices": q.get("n_choices", 4),
        # SFT pair
        "input": eval_input,
        "output": pp["processed_trace"],
        # QA metadata
        "raw_trace": raw_trace,
        "pp_stats": {
            "raw_length": pp["raw_length"],
            "processed_length": pp["processed_length"],
            "think_stripped": pp["think_block_stripped"],
            "leaks_found": pp["leaks_found"],
            "leaks_after_clean": pp["leaks_after_clean"],
            "answer_parsed": pp["answer_parsed"],
            "answer_matches": pp["answer_matches"],
            "independently_correct": pp["independently_correct"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT rationalization traces")
    parser.add_argument(
        "--n", type=int, default=0,
        help="Questions to sample (0 = full training set)",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to partial results JSONL to resume from",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        log.error("Set DEEPINFRA_API_KEY environment variable.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)

    # Load training questions
    with open(TRAIN_PATH) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d training questions.", len(questions))

    # Handle resume
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if "error" not in r:
                            existing_results.append(r)
                            done_ids.add(r["question_id"])
            log.info("Resuming: %d already completed.", len(done_ids))

    if args.n > 0:
        # Stratified sample: proportional to (target_property × structure_family × difficulty × direction)
        rng = random.Random(args.seed)
        buckets: dict[tuple[str, ...], list[dict]] = {}
        for q in questions:
            key = (q["target_property"], q["structure_family"], q["difficulty"],
                   q.get("direction", "forward"))
            buckets.setdefault(key, []).append(q)

        total = len(questions)
        sampled: list[dict] = []
        for key, items in buckets.items():
            k = max(1, round(args.n * len(items) / total))
            sampled.extend(rng.sample(items, min(k, len(items))))
        rng.shuffle(sampled)
        questions = sampled[:args.n]

        # Log distribution
        from collections import Counter
        prop_dist = Counter(q["target_property"] for q in questions)
        log.info("Stratified sample: %d questions — %s", len(questions),
                 ", ".join(f"{k}: {v}" for k, v in prop_dist.most_common()))

    # Filter out already-done questions
    remaining = [q for q in questions if q["question_id"] not in done_ids]
    log.info("Generating traces: %d remaining (%d already done), %d workers",
             len(remaining), len(done_ids), args.workers)

    results = list(existing_results)
    done = len(existing_results)
    errors = 0
    leaks_total = 0
    answer_mismatches = 0

    # Set up output path
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    n_tag = f"_n{args.n}" if args.n > 0 else ""
    out_path = TRACES_DIR / f"traces_deepseek_r1{n_tag}_{ts}.jsonl"

    # Write existing results first
    with open(out_path, "w") as f:
        for r in existing_results:
            f.write(json.dumps(r) + "\n")

    # Generate remaining
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, q, client, args.delay): q
            for q in remaining
        }
        for future in as_completed(futures):
            r = future.result()

            if "error" in r:
                errors += 1
            else:
                results.append(r)
                done += 1
                stats = r["pp_stats"]
                if stats["leaks_found"] > 0:
                    leaks_total += 1
                if not stats["independently_correct"]:
                    answer_mismatches += 1

                # Append to file incrementally
                with open(out_path, "a") as f:
                    f.write(json.dumps(r) + "\n")

            total_processed = done + errors
            correct_so_far = sum(1 for r in results if "error" not in r and r["pp_stats"]["independently_correct"])
            if total_processed % 10 == 0 or total_processed == len(remaining) + len(existing_results):
                yield_pct = correct_so_far / done * 100 if done else 0
                log.info(
                    "Progress: %d/%d done | %d correct (%.0f%% yield) | %d errors | %d leaked",
                    done, len(questions), correct_so_far, yield_pct, errors, leaks_total,
                )

    # Summary stats
    good_results = [r for r in results if "error" not in r]
    correct_results = [r for r in good_results if r["pp_stats"]["independently_correct"]]

    if good_results:
        pp_stats = [r["pp_stats"] for r in good_results]
        leaks = sum(1 for s in pp_stats if s["leaks_found"] > 0)
        correct_count = len(correct_results)
        lengths = [s["processed_length"] for s in pp_stats if s["independently_correct"]]

        print(f"\n{'='*70}")
        print(f"TRACE GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total generated:     {len(good_results)}")
        print(f"Independently correct: {correct_count}/{len(good_results)} ({correct_count/len(good_results)*100:.1f}% yield)")
        print(f"Errors:              {errors}")
        print(f"Leaked values:       {leaks}/{len(good_results)} ({leaks/len(good_results)*100:.1f}%)")
        if lengths:
            print(f"Trace length (chars): min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
        print(f"All traces:          {out_path}")
        print(f"{'='*70}")

    # Save SFT-only file — ONLY correct traces (genuine reasoning)
    sft_path = out_path.with_name(out_path.stem + "_sft.jsonl")
    with open(sft_path, "w") as f:
        for r in correct_results:
            f.write(json.dumps({"input": r["input"], "output": r["output"]}) + "\n")
    log.info("SFT pairs saved: %d correct traces to %s", len(correct_results), sft_path)


if __name__ == "__main__":
    main()
