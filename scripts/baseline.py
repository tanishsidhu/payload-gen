#!/usr/bin/env python3
"""Run the base (unfine-tuned) model on the test set and print baseline metrics."""

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import generate, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generator"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from build_dataset import TEST_SEED, write_jsonl  # noqa: E402
from score import score_prediction, summarize  # noqa: E402

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_TEST_COUNT = 800


def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, row):
    # System + user only — the model fills the assistant turn.
    messages = row["messages"][:2]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )


def ensure_test_set(path, count, seed):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        rows = load_rows(path)
        if rows and "seed" in rows[0]:
            print(f"Using existing test set: {path} ({len(rows)} rows, seed={rows[0]['seed']})")
            return rows

    print(f"Generating test set -> {path} (seed={seed}, count={count})")
    write_jsonl(path, seed, count)
    return load_rows(path)


def run_baseline(args):
    rows = ensure_test_set(args.test_path, args.test_count, args.test_seed)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    scores = []
    for i, row in enumerate(rows, start=1):
        prompt = build_prompt(tokenizer, row)
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        scores.append(score_prediction(row, output))

        if i % 10 == 0 or i == len(rows):
            print(f"  scored {i}/{len(rows)}", flush=True)

    metrics = summarize(scores)
    print()
    print("Baseline results (base model, no adapter)")
    print("=" * 52)
    print(f"  Test rows scored     {metrics['rows']}")
    print(f"  Schema-valid         {metrics['schema_valid_pct']:.1f}%")
    print(f"  Exact match          {metrics['exact_match_pct']:.1f}%")
    print(f"  Refusal accuracy     {metrics['refusal_accuracy_pct']:.1f}%  "
          f"({metrics['refusal_rows']} refusal rows)")
    print("=" * 52)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "scores": scores}, f, indent=2)
        print(f"Saved details -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Baseline eval on unseen test specs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-path", default="data/test.jsonl")
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument(
        "--test-seed",
        type=int,
        default=TEST_SEED,
        help="Seed for test.jsonl generation (default: 2_000_000, never used in train/valid)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Score only first N rows")
    parser.add_argument("--output", default="", help="Optional JSON file for raw scores")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
