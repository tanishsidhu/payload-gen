#!/usr/bin/env python3
"""Compare base-3B vs fine-tuned-3B vs Claude on seen and unseen specs."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from mlx_lm import generate, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generator"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from score import row_has_distractor, score_prediction, summarize  # noqa: E402

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Rough Sonnet pricing (USD per million tokens) for cost estimates.
CLAUDE_INPUT_PER_M = 3.0
CLAUDE_OUTPUT_PER_M = 15.0
LOCAL_COST_PER_PAYLOAD = 0.0  # on-prem amortized; shown as ~$0


def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, row):
    return tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True)


def eval_local_model(rows, model_id, adapter_path=None, label="model"):
    print(f"  Evaluating {label} on {len(rows)} rows...", flush=True)
    if adapter_path:
        model, tokenizer = load(model_id, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_id)

    scores = []
    for i, row in enumerate(rows, start=1):
        output = generate(
            model,
            tokenizer,
            build_prompt(tokenizer, row),
            max_tokens=256,
            verbose=False,
        )
        scores.append(score_prediction(row, output))
        if i % 25 == 0 or i == len(rows):
            print(f"    {label}: {i}/{len(rows)}", flush=True)
    return scores


def call_claude(row, api_key, model=CLAUDE_MODEL):
    system = row["messages"][0]["content"]
    user = row["messages"][1]["content"]
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 256,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    return text, usage


def eval_claude(rows, api_key, model=CLAUDE_MODEL):
    print(f"  Evaluating Claude ({model}) on {len(rows)} rows...", flush=True)
    scores = []
    total_input = 0
    total_output = 0
    for i, row in enumerate(rows, start=1):
        try:
            output, usage = call_claude(row, api_key, model=model)
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
        except urllib.error.HTTPError as e:
            print(f"    Claude API error: {e.read().decode()[:200]}")
            raise
        scores.append(score_prediction(row, output))
        if i % 10 == 0 or i == len(rows):
            print(f"    claude: {i}/{len(rows)}", flush=True)
        time.sleep(0.2)  # gentle rate limit

    cost = (
        total_input * CLAUDE_INPUT_PER_M / 1_000_000
        + total_output * CLAUDE_OUTPUT_PER_M / 1_000_000
    )
    cost_per_payload = cost / len(rows) if rows else 0.0
    return scores, cost_per_payload


def load_baseline_scores(test_rows):
    path = PROJECT_ROOT / "results" / "baseline.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = data.get("scores", [])
    if len(scores) != len(test_rows):
        return None
    # Re-score metadata (distractor flag) for newer summarize fields.
    enriched = []
    for row, score in zip(test_rows, scores):
        score["has_distractor"] = row_has_distractor(row)
        enriched.append(score)
    return enriched


def print_table(results):
    print()
    print("BENCHMARK (mechanically scored — schema valid % / exact match % / refusal %)")
    print("=" * 95)
    header = f"{'Model':<22} {'Split':<12} {'Rows':>5} {'Schema%':>8} {'Exact%':>8} {'Refusal%':>9} {'DistExact%':>11} {'$/payload':>10}"
    print(header)
    print("-" * 95)
    for row in results:
        dist = row.get("distractor_exact_match_pct")
        dist_str = f"{dist:>10.1f}" if dist is not None else f"{'n/a':>10}"
        cost = row.get("cost_per_payload_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "  ~$0.00"
        print(
            f"{row['model']:<22} {row['split']:<12} {row['rows']:>5} "
            f"{row['schema_valid_pct']:>7.1f}% {row['exact_match_pct']:>7.1f}% "
            f"{row['refusal_accuracy_pct']:>8.1f}% {dist_str} {cost_str:>10}"
        )
    print("=" * 95)
    print("DistExact% = exact match on distractor rows only (lower => distractor problem)")


def run_benchmark(args):
    test_rows = load_rows(args.test_path)
    valid_rows = load_rows(args.valid_path)

    if args.limit:
        test_rows = test_rows[: args.limit]
    if args.eval_seen and args.valid_limit:
        valid_rows = valid_rows[: args.valid_limit]

    if not valid_rows[0].get("seed"):
        print("valid.jsonl missing seed field — regenerating...")
        os.system(
            f"cd {PROJECT_ROOT} && source .venv/bin/activate && "
            f"python src/generator/build_dataset.py --train-count 0 --test-count 0 --valid-count 800"
        )
        valid_rows = load_rows(args.valid_path)
        if args.limit:
            valid_rows = valid_rows[: args.limit]

    table = []
    output = {"table": table, "notes": []}

    # --- Base 3B ---
    base_test = load_baseline_scores(test_rows)
    if base_test and not args.rerun_base:
        print("Reusing base model scores from results/baseline.json (test/unseen)")
        metrics = summarize(base_test)
    else:
        base_test = eval_local_model(test_rows, args.model, label="base-3B")
        metrics = summarize(base_test)
    table.append({"model": "base-3B", "split": "unseen", **metrics, "cost_per_payload_usd": LOCAL_COST_PER_PAYLOAD})

    if args.eval_seen:
        base_valid = eval_local_model(valid_rows, args.model, label="base-3B")
        table.append({"model": "base-3B", "split": "seen", **summarize(base_valid), "cost_per_payload_usd": LOCAL_COST_PER_PAYLOAD})

    # --- Fine-tuned 3B ---
    ft_test = eval_local_model(test_rows, args.model, adapter_path=args.adapter_path, label="fine-tuned-3B")
    table.append({"model": "fine-tuned-3B", "split": "unseen", **summarize(ft_test), "cost_per_payload_usd": LOCAL_COST_PER_PAYLOAD})

    if args.eval_seen:
        ft_valid = eval_local_model(valid_rows, args.model, adapter_path=args.adapter_path, label="fine-tuned-3B")
        table.append({"model": "fine-tuned-3B", "split": "seen", **summarize(ft_valid), "cost_per_payload_usd": LOCAL_COST_PER_PAYLOAD})

    # --- Claude ---
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and args.claude_limit > 0:
        claude_rows = test_rows[: args.claude_limit]
        claude_scores, claude_cost = eval_claude(claude_rows, api_key, model=args.claude_model)
        table.append(
            {
                "model": "Claude",
                "split": "unseen",
                **summarize(claude_scores),
                "cost_per_payload_usd": claude_cost,
            }
        )
    else:
        msg = "Skipped Claude (set ANTHROPIC_API_KEY to include frontier numbers)"
        print(msg)
        output["notes"].append(msg)

    print_table(table)

    # Distractor decision helper
    ft_unseen = next(r for r in table if r["model"] == "fine-tuned-3B" and r["split"] == "unseen")
    if ft_unseen.get("distractor_rows") and ft_unseen.get("non_distractor_exact_match_pct") is not None:
        gap = ft_unseen["non_distractor_exact_match_pct"] - ft_unseen["distractor_exact_match_pct"]
        print()
        print(f"Distractor gap (non-distractor exact % minus distractor exact %): {gap:.1f} points")
        if gap >= args.distractor_gap_threshold:
            print(
                f"  -> Gap >= {args.distractor_gap_threshold}%: recommend adding more distractor "
                "training rows and a short retrain."
            )
            output["notes"].append(f"distractor_retrain_recommended: gap={gap:.1f}")
        else:
            print("  -> Distractor gap is modest; extra distractor training optional.")
            output["notes"].append(f"distractor_retrain_optional: gap={gap:.1f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Three-way benchmark with distractor analysis.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--test-path", default="data/test.jsonl")
    parser.add_argument("--valid-path", default="data/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Limit test rows (0 = all)")
    parser.add_argument("--valid-limit", type=int, default=100, help="Limit valid rows when --eval-seen")
    parser.add_argument("--eval-seen", action="store_true", help="Also eval on valid.jsonl (seen split)")
    parser.add_argument("--rerun-base", action="store_true", help="Ignore cached baseline.json")
    parser.add_argument("--claude-limit", type=int, default=50, help="Claude eval row count (0=skip)")
    parser.add_argument("--claude-model", default=CLAUDE_MODEL)
    parser.add_argument("--distractor-gap-threshold", type=float, default=15.0)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
