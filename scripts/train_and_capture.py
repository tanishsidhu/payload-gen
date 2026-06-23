#!/usr/bin/env python3
"""Run LoRA training, then capture mid-training and final examples."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TRAIN_CMD = [
    "mlx_lm.lora",
    "--model", MODEL,
    "--train",
    "--data", "./data",
    "--fine-tune-type", "lora",
    "--batch-size", "1",
    "--num-layers", "8",
    "--iters", "600",
    "--learning-rate", "2e-4",
    "--save-every", "100",
    "--grad-checkpoint",
    "--adapter-path", "./adapters",
]


def run(cmd, log_path=None):
    print("$", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
            print(tail)
            raise SystemExit(proc.returncode)
    else:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    results = PROJECT_ROOT / "results"
    run(TRAIN_CMD, results / "training.log")

    capture = [sys.executable, "scripts/capture_examples.py"]
    run(capture + ["--stage", "training_iter_300"])
    run(capture + ["--stage", "fine_tuned"])

    print("Training complete. Examples saved to results/examples.json")
    print("Training log saved to results/training.log")


if __name__ == "__main__":
    main()
