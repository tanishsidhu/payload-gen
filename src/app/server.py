"""FastAPI backend wiring retrieval, mapping, orchestration, and the fine-tuned model."""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mapping"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "orchestration"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from decompose import decompose  # noqa: E402
from excel_map import map_excel_to_values  # noqa: E402
from retrieve import SpecRetriever  # noqa: E402
from score import extract_json  # noqa: E402

SYSTEM_PROMPT = (
    "You are a payload generator. Read the API spec in this message and output "
    "ONLY valid JSON — either the request payload or a refusal object."
)

MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8080/v1/chat/completions")

app = FastAPI(title="Payload Gen")
_retriever: Optional[SpecRetriever] = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = SpecRetriever(specs_dir=PROJECT_ROOT / "specs")
    return _retriever


def _find_endpoint(spec, operation):
    for endpoint in spec["endpoints"]:
        if endpoint["operation"] == operation:
            return endpoint
    return None


def _call_model(system_content, user_content):
    """Call the fine-tuned model served by mlx_lm.server (Layer C)."""
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        MODEL_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Layer C model server not reachable. Start it with:\n"
            "  mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit "
            "--adapter-path ./adapters --port 8080"
        ) from e

    return data["choices"][0]["message"]["content"]


def _run_pipeline(message, excel_path=None):
    retriever = get_retriever()
    retrieval = retriever.retrieve(message)
    operations = decompose(message, hinted_operation=retrieval["operation"])
    spec = retrieval["spec"]
    spec_text = retrieval["spec_text"]

    excel_values = None
    if excel_path:
        endpoint = _find_endpoint(spec, retrieval["operation"])
        if endpoint:
            excel_values = map_excel_to_values(excel_path, endpoint)

    results = []
    for op in operations:
        endpoint = _find_endpoint(spec, op)
        if endpoint is None:
            results.append(
                {
                    "operation": op,
                    "error": "operation not in retrieved spec",
                    "payload": {"refusal": "no endpoint supports this operation"},
                }
            )
            continue

        user_parts = [f"Operation: {op}", message]
        if excel_values and op == retrieval["operation"]:
            user_parts.append(
                f"Use these spreadsheet values where applicable: {json.dumps(excel_values)}"
            )

        raw = _call_model(
            f"{SYSTEM_PROMPT}\n\nAPI SPEC:\n{spec_text}",
            "\n".join(user_parts),
        )
        results.append({"operation": op, "model_raw": raw, "payload": extract_json(raw)})

    return {
        "retrieval": {
            "spec_id": retrieval["id"],
            "operation": retrieval["operation"],
            "score": retrieval["score"],
        },
        "operations": operations,
        "excel_values": excel_values,
        "results": results,
    }


@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "src" / "app" / "chat.html")


@app.post("/chat")
async def chat(request: Request):
    content_type = request.headers.get("content-type", "")
    excel_path = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        message = str(form.get("message", ""))
        upload = form.get("excel")
        if upload and getattr(upload, "filename", None):
            suffix = Path(upload.filename).suffix or ".xlsx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                excel_path = tmp.name
    else:
        body = await request.json()
        message = body.get("message", "")

    try:
        return JSONResponse(_run_pipeline(message, excel_path=excel_path))
    finally:
        if excel_path:
            Path(excel_path).unlink(missing_ok=True)
