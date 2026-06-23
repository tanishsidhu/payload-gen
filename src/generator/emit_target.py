"""Produce the correct JSON payload or structured refusal for a given task."""

import json


def emit_target(request):
    """
    Build the ground-truth assistant response.

    request: output of render_request()
    """
    task = request["task"]

    if request["kind"] == "impossible" or task["kind"] == "impossible":
        return json.dumps({"refusal": "no endpoint supports this operation"})

    payload = {
        "operation": task["operation"],
        "path": task["endpoint"]["path"],
        "body": task["values"],
    }
    return json.dumps(payload)
