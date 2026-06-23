"""Pick one endpoint from a spec and fill in legal field values for it."""

import random
from datetime import date, timedelta

from invent_spec import UNSUPPORTED_OPERATIONS


def _sample_string(rng, prefix):
    suffix = "".join(rng.choices("0123456789", k=4))
    return f"{prefix}-{suffix}"


def _sample_date(rng):
    base = date(2024, 1, 1)
    offset = rng.randint(0, 700)
    return (base + timedelta(days=offset)).isoformat()


def _sample_integer(rng, unit):
    # Keep human-readable numbers; unit is stored separately on the field.
    if unit == "cents":
        return rng.randint(50_000, 5_000_000)  # $500 – $50,000
    return rng.randint(500, 50_000)


def _fill_fields(rng, fields):
    values = {}
    for field in fields:
        if field["type"] == "object":
            values[field["name"]] = _fill_fields(rng, field["fields"])
            continue

        # Skip some optional fields to add variety.
        if not field["required"] and rng.random() < 0.4:
            continue

        if field["type"] == "string":
            if "contract" in field["name"] or "lease" in field["name"] or "agreement" in field["name"]:
                values[field["name"]] = _sample_string(rng, "LF")
            elif "asset" in field["name"] or "equipment" in field["name"]:
                values[field["name"]] = _sample_string(rng, "AST")
            elif "payer" in field["name"] or "remitter" in field["name"] or "source" in field["name"]:
                values[field["name"]] = _sample_string(rng, "PAY")
            elif "lessee" in field["name"] or "borrower" in field["name"] or "customer" in field["name"]:
                values[field["name"]] = rng.choice(
                    ["Acme Corp", "Globex LLC", "Initech Holdings", "Umbrella Finance"]
                )
            else:
                values[field["name"]] = rng.choice(
                    ["standard terms apply", "rate adjusted per schedule", "pending legal review"]
                )
        elif field["type"] == "integer":
            unit = field.get("unit", "dollars")
            values[field["name"]] = _sample_integer(rng, unit)
        elif field["type"] == "enum":
            values[field["name"]] = rng.choice(field["enum"])
        elif field["type"] == "date":
            values[field["name"]] = _sample_date(rng)

    return values


def pick_task(spec, seed):
    """
    Pick one endpoint and legal values, OR flag an impossible operation.

    Returns a dict with kind='valid' or kind='impossible'.
    """
    rng = random.Random(seed)
    roll = rng.random()

    # ~12% impossible requests — no matching endpoint in this spec.
    if roll < 0.12:
        op = rng.choice(UNSUPPORTED_OPERATIONS)
        return {
            "kind": "impossible",
            "operation": op,
            "spec": spec,
            "values": {},
        }

    endpoint = rng.choice(spec["endpoints"])
    values = _fill_fields(rng, endpoint["fields"])

    return {
        "kind": "valid",
        "operation": endpoint["operation"],
        "endpoint": endpoint,
        "values": values,
        "spec": spec,
    }
