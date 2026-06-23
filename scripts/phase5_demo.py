#!/usr/bin/env python3
"""Phase 5: run three fine-tuned demo cases on unseen test specs."""

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import generate, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generator"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from score import score_prediction  # noqa: E402

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Hand-picked from data/test.jsonl — all seeds >= 2_000_000 (never in training).
DEMO_CASES = [
    {
        "case": "a_valid_request",
        "test_index": 2,
        "description": "Valid post_receipt request on an unseen spec",
    },
    {
        "case": "b_distractor_ignored",
        "test_index": 0,
        "description": "Valid book_lease; user mentions in_service_date (distractor)",
    },
    {
        "case": "c_impossible_refusal",
        "test_index": 1,
        "description": "transfer_ownership — no endpoint in this spec",
    },
]


def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, row):
    return tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True)


def run_demos(args):
    rows = load_rows(args.test_path)
    print(f"Loading fine-tuned model: {args.model} + adapter {args.adapter_path}")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    results = []
    for spec in DEMO_CASES:
        row = rows[spec["test_index"]]
        output = generate(
            model,
            tokenizer,
            build_prompt(tokenizer, row),
            max_tokens=args.max_tokens,
            verbose=False,
        )
        score = score_prediction(row, output)
        result = {
            **spec,
            "seed": row.get("seed"),
            "user_request": row["messages"][1]["content"],
            "expected_response": row["messages"][2]["content"],
            "model_response": output,
            "schema_valid": score["schema_valid"],
            "exact_match": score["exact_match"],
        }
        results.append(result)

        print()
        print("=" * 60)
        print(spec["case"].upper(), f"(test index {spec['test_index']}, seed {row.get('seed')})")
        print(spec["description"])
        print("-" * 60)
        print("USER:", row["messages"][1]["content"])
        print()
        print("EXPECTED:", row["messages"][2]["content"])
        print()
        print("MODEL:", output)
        print()
        print(f"schema_valid={score['schema_valid']}  exact_match={score['exact_match']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"demos": results}, indent=2), encoding="utf-8")
    print()
    print(f"Saved -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 fine-tuned demo cases.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--test-path", default="data/test.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/phase5_demos.json")
    args = parser.parse_args()
    run_demos(args)


if __name__ == "__main__":
    main()
