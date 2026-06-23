"""Optional FastAPI mock server that validates payloads against a fictional spec."""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))

from invent_spec import invent_spec, render_spec_text
from score import extract_json, is_schema_valid

app = FastAPI(title="Payload Mock Server")

# Demo spec baked into the server for quick manual checks (not used for model training).
DEMO_SEED = 424242


class GenerateRequest(BaseModel):
    seed: int = DEMO_SEED
    model_output: str


@app.get("/spec")
def get_spec(seed: int = DEMO_SEED):
    spec = invent_spec(seed)
    return {"spec": spec, "spec_text": render_spec_text(spec)}


@app.post("/execute")
def execute_payload(req: GenerateRequest):
    spec = invent_spec(req.seed)
    parsed = extract_json(req.model_output)
    if parsed is None:
        raise HTTPException(status_code=400, detail="Model output is not valid JSON")

    if not is_schema_valid(parsed, spec):
        raise HTTPException(status_code=422, detail="Payload fails schema validation")

    if "refusal" in parsed:
        return {"status": "refused", "payload": parsed}

    return {
        "status": "accepted",
        "payload": parsed,
        "message": f"Would POST to {parsed['path']}",
    }
