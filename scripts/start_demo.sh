#!/usr/bin/env bash
# Start Layer C (MLX model) and the FastAPI orchestration app for end-to-end demo.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Starting Layer C model server on :8080 ..."
mlx_lm.server \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters \
  --port 8080 &
MODEL_PID=$!

sleep 3
echo "Starting orchestration app on :8000 ..."
echo "Open http://localhost:8000"
uvicorn server:app --app-dir src/app --host 0.0.0.0 --port 8000

kill $MODEL_PID 2>/dev/null || true
