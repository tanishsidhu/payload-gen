#!/usr/bin/env python3
"""Capture fixed request/response examples at each training stage."""

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from mlx_lm import generate, load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "generator"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from score import score_prediction  # noqa: E402

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_INDICES = [0, 1]
EXAMPLES_FILE = PROJECT_ROOT / "results" / "examples.json"

# Fixed demo rows from data/test.jsonl (seed 2_000_000 + offset).
LABELS = {
    0: "valid request (book lease)",
    1: "impossible request (transfer ownership)",
}


def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tokenizer, row):
    messages = row["messages"][:2]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


@contextmanager
def staged_adapter_weights(adapter_dir, checkpoint_name=None):
    """
    Temporarily swap in a numbered checkpoint without losing final weights.

    mlx_lm always loads adapter_dir/adapters.safetensors.
    """
    if adapter_dir is None:
        yield
        return

    adapter_dir = Path(adapter_dir)
    weights = adapter_dir / "adapters.safetensors"
    backup = adapter_dir / "adapters.safetensors.bak"

    if checkpoint_name is None:
        yield
        return

    checkpoint = adapter_dir / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if weights.exists():
        shutil.copy(weights, backup)
    shutil.copy(checkpoint, weights)
    try:
        yield
    finally:
        if backup.exists():
            shutil.move(backup, weights)
        elif weights.exists() and checkpoint_name is not None:
            weights.unlink()


def capture_stage(stage, model_id, adapter_dir, checkpoint_name, test_path, indices, max_tokens):
    rows = load_rows(test_path)
    examples = []

    with staged_adapter_weights(adapter_dir, checkpoint_name):
        if adapter_dir and Path(adapter_dir, "adapter_config.json").exists():
            model, tokenizer = load(model_id, adapter_path=str(adapter_dir))
        else:
            model, tokenizer = load(model_id)

        for idx in indices:
            row = rows[idx]
            prompt = build_prompt(tokenizer, row)
            output = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
            score = score_prediction(row, output)
            examples.append(
                {
                    "label": LABELS.get(idx, f"test row {idx}"),
                    "test_index": idx,
                    "seed": row.get("seed"),
                    "user_request": row["messages"][1]["content"],
                    "expected_response": row["messages"][2]["content"],
                    "model_response": output,
                    "schema_valid": score["schema_valid"],
                    "exact_match": score["exact_match"],
                }
            )

    return examples


def save_examples(stage, examples, indices):
    EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if EXAMPLES_FILE.exists():
        data = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    else:
        data = {"example_indices": indices, "stages": {}}

    data["example_indices"] = indices
    data["stages"][stage] = examples
    EXAMPLES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved stage '{stage}' ({len(examples)} examples) -> {EXAMPLES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Capture fixed examples for one training stage.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["baseline", "training_iter_300", "fine_tuned"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Numbered checkpoint inside adapter dir, e.g. 0000300_adapters.safetensors",
    )
    parser.add_argument("--test-path", default="data/test.jsonl")
    parser.add_argument("--indices", default="0,1", help="Comma-separated test row indices")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    checkpoint = args.checkpoint or None

    if args.stage == "baseline":
        adapter_dir = None
        checkpoint = None
    elif args.stage == "training_iter_300":
        adapter_dir = args.adapter_dir
        checkpoint = checkpoint or "0000300_adapters.safetensors"
    else:
        adapter_dir = args.adapter_dir
        checkpoint = None

    examples = capture_stage(
        args.stage,
        args.model,
        adapter_dir,
        checkpoint,
        args.test_path,
        indices,
        args.max_tokens,
    )
    save_examples(args.stage, examples, indices)


if __name__ == "__main__":
    main()
