"""Loop the generator pipeline and write MLX chat-format JSONL training rows."""

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python src/generator/build_dataset.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_target import emit_target
from invent_spec import invent_spec, render_spec_text
from pick_task import pick_task
from render_request import render_request

SYSTEM_PROMPT = (
    "You are a payload generator. Read the API spec in this message and output "
    "ONLY valid JSON — either the request payload or a refusal object."
)

# Used for test.jsonl only — never overlaps train (1000+) or valid (900_000+).
TEST_SEED = 2_000_000


def build_row(seed):
    spec = invent_spec(seed)
    task = pick_task(spec, seed + 1)
    request = render_request(task, seed + 2)
    target = emit_target(request)

    return {
        "seed": seed,
        "messages": [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nAPI SPEC:\n{render_spec_text(spec)}",
            },
            {"role": "user", "content": request["text"]},
            {"role": "assistant", "content": target},
        ]
    }


def write_jsonl(path, start_seed, count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            row = build_row(start_seed + i)
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate MLX chat JSONL training data.")
    parser.add_argument("--train-count", type=int, default=6000)
    parser.add_argument("--valid-count", type=int, default=800)
    parser.add_argument("--train-seed", type=int, default=1000)
    parser.add_argument("--valid-seed", type=int, default=900_000)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.train_count > 0:
        write_jsonl(data_dir / "train.jsonl", args.train_seed, args.train_count)
        print(f"Wrote {args.train_count} rows -> {data_dir / 'train.jsonl'}")
    if args.valid_count > 0:
        write_jsonl(data_dir / "valid.jsonl", args.valid_seed, args.valid_count)
        print(f"Wrote {args.valid_count} rows -> {data_dir / 'valid.jsonl'}")
    if args.test_count > 0:
        write_jsonl(data_dir / "test.jsonl", args.test_seed, args.test_count)
        print(f"Wrote {args.test_count} rows -> {data_dir / 'test.jsonl'} (seed={args.test_seed})")


if __name__ == "__main__":
    main()
