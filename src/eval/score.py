"""Score model outputs: schema validity, exact match, and refusal accuracy."""

import json

import jsonschema
from jsonschema import Draft7Validator

from invent_spec import invent_spec


def extract_json(text):
    """Pull a JSON object out of raw model text (handles markdown fences)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _field_schema(field):
    if field["type"] == "object":
        props = {}
        required = []
        for child in field["fields"]:
            props[child["name"]] = _field_schema(child)
            if child["required"]:
                required.append(child["name"])
        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    if field["type"] == "string":
        return {"type": "string"}
    if field["type"] == "integer":
        return {"type": "integer"}
    if field["type"] == "date":
        # ISO-8601 date strings
        return {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
    if field["type"] == "enum":
        return {"type": "string", "enum": field["enum"]}

    return {}


def _body_schema(fields):
    props = {}
    required = []
    for field in fields:
        props[field["name"]] = _field_schema(field)
        if field["required"]:
            required.append(field["name"])
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _payload_schema(endpoint):
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "const": endpoint["operation"]},
            "path": {"type": "string", "const": endpoint["path"]},
            "body": _body_schema(endpoint["fields"]),
        },
        "required": ["operation", "path", "body"],
        "additionalProperties": False,
    }


def _refusal_schema():
    return {
        "type": "object",
        "properties": {
            "refusal": {
                "type": "string",
                "const": "no endpoint supports this operation",
            }
        },
        "required": ["refusal"],
        "additionalProperties": False,
    }


def _find_endpoint(spec, operation):
    for endpoint in spec["endpoints"]:
        if endpoint["operation"] == operation:
            return endpoint
    return None


def row_has_distractor(row):
    """True when the user message mentions a decoy field from another endpoint."""
    return "Also include" in row["messages"][1]["content"]


def is_schema_valid(parsed, spec):
    """Return True if parsed output matches the spec structurally."""
    if parsed is None:
        return False

    if "refusal" in parsed:
        validator = Draft7Validator(_refusal_schema())
        return validator.is_valid(parsed)

    operation = parsed.get("operation")
    endpoint = _find_endpoint(spec, operation)
    if endpoint is None:
        return False

    validator = Draft7Validator(_payload_schema(endpoint))
    return validator.is_valid(parsed)


def score_prediction(row, model_text):
    """
    Score one model output against a test row.

    Returns dict with schema_valid, exact_match, expected_refusal, predicted_refusal.
    """
    expected = json.loads(row["messages"][2]["content"])
    predicted = extract_json(model_text)
    spec = invent_spec(row["seed"])

    expected_refusal = "refusal" in expected
    predicted_refusal = predicted is not None and "refusal" in predicted

    schema_valid = is_schema_valid(predicted, spec)
    exact_match = (
        predicted is not None
        and normalize_json(predicted) == normalize_json(expected)
    )

    return {
        "schema_valid": schema_valid,
        "exact_match": exact_match,
        "expected_refusal": expected_refusal,
        "predicted_refusal": predicted_refusal,
        "refusal_correct": expected_refusal and exact_match,
        "has_distractor": row_has_distractor(row),
    }


def summarize(scores):
    """Aggregate per-row scores into headline percentages."""
    total = len(scores)
    if total == 0:
        return {}

    refusal_rows = [s for s in scores if s["expected_refusal"]]
    distractor_rows = [s for s in scores if s.get("has_distractor")]
    non_distractor_rows = [s for s in scores if not s.get("has_distractor")]

    def _pct(items, key):
        if not items:
            return 0.0
        return 100 * sum(s[key] for s in items) / len(items)

    result = {
        "rows": total,
        "schema_valid_pct": _pct(scores, "schema_valid"),
        "exact_match_pct": _pct(scores, "exact_match"),
        "refusal_accuracy_pct": _pct(refusal_rows, "refusal_correct"),
        "refusal_rows": len(refusal_rows),
    }

    if distractor_rows:
        result["distractor_rows"] = len(distractor_rows)
        result["distractor_exact_match_pct"] = _pct(distractor_rows, "exact_match")
        result["distractor_schema_valid_pct"] = _pct(distractor_rows, "schema_valid")
        result["non_distractor_exact_match_pct"] = _pct(non_distractor_rows, "exact_match")

    return result
