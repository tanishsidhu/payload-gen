"""Turn a task into a natural-language request, with optional distractors or impossible ops."""

import random
from datetime import datetime

# Weekday names for "next Monday" style phrasing.
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]


def _number_to_words(n):
    """Convert an integer to spoken English (good enough for our value ranges)."""
    ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n < 20:
        return ones[n]
    if n < 100:
        word = tens[n // 10]
        if n % 10:
            word = f"{word} {ones[n % 10]}"
        return word
    if n < 1_000:
        return f"{_number_to_words(n // 100)} hundred {_number_to_words(n % 100)}".replace(" zero", "")
    if n < 1_000_000:
        return f"{_number_to_words(n // 1000)} thousand {_number_to_words(n % 1000)}".replace(" zero", "")
    return str(n)


def _format_amount(value, unit):
    if unit == "cents":
        dollars = value / 100
        if dollars == int(dollars):
            return f"{_number_to_words(int(dollars))} dollars"
        return f"{_number_to_words(value)} cents"
    return f"{_number_to_words(value)} dollars"


def _format_date(iso_date, rng):
    d = datetime.strptime(iso_date, "%Y-%m-%d")
    style = rng.choice(["iso", "long", "relative"])
    if style == "iso":
        return iso_date
    if style == "long":
        return d.strftime("%B %d, %Y")
    return f"next {rng.choice(WEEKDAYS)}"


def _field_phrase(field_name, value, field_meta, rng):
    if field_meta["type"] == "integer":
        unit = field_meta.get("unit", "dollars")
        amount = _format_amount(value, unit)
        if "monthly" in field_name or "installment" in field_name or "rent" in field_name:
            return f"monthly payment of {amount}"
        if "receipt" in field_name or "collected" in field_name:
            return f"receipt of {amount}"
        return f"payment of {amount}"

    if field_meta["type"] == "date":
        return f"effective {_format_date(value, rng)}"

    if field_meta["type"] == "enum":
        return f"{field_name.replace('_', ' ')} set to {value.replace('_', ' ')}"

    label = field_name.replace("_", " ")
    return f"{label} {value}"


def _find_field_meta(fields, name):
    for field in fields:
        if field["type"] == "object":
            found = _find_field_meta(field["fields"], name)
            if found:
                return found
        elif field["name"] == name:
            return field
    return None


def _flatten_fields(fields):
    flat = []
    for field in fields:
        if field["type"] == "object":
            flat.extend(_flatten_fields(field["fields"]))
        else:
            flat.append(field)
    return flat


def _render_valid_request(task, rng, include_distractor):
    endpoint = task["endpoint"]
    flat_fields = _flatten_fields(endpoint["fields"])

    phrases = []
    for name, value in task["values"].items():
        if isinstance(value, dict):
            for nested_name, nested_value in value.items():
                meta = _find_field_meta(endpoint["fields"], nested_name)
                if meta:
                    phrases.append(_field_phrase(nested_name, nested_value, meta, rng))
            continue
        meta = _find_field_meta(endpoint["fields"], name)
        if meta:
            phrases.append(_field_phrase(name, value, meta, rng))

    op = task["operation"].replace("_", " ")
    templates = [
        f"Please {op} with {', '.join(phrases)}.",
        f"Can you {op}? {', '.join(phrases)}.",
        f"I need to {op}: {', '.join(phrases)}.",
        f"Go ahead and {op} — {', '.join(phrases)}.",
    ]
    text = rng.choice(templates)

    if include_distractor:
        # Mention a field that belongs to a *different* endpoint in this spec.
        other_fields = []
        for ep in task["spec"]["endpoints"]:
            if ep["operation"] == task["operation"]:
                continue
            other_fields.extend(_flatten_fields(ep["fields"]))

        if other_fields:
            decoy = rng.choice(other_fields)
            decoy_phrase = decoy["name"].replace("_", " ")
            text += f" Also include {decoy_phrase} if possible."

    return {
        "text": text,
        "kind": "distractor" if include_distractor else "valid",
        "task": task,
    }


def _render_impossible_request(task, rng):
    op = task["operation"].replace("_", " ")
    contract = f"LF-{rng.randint(1000, 9999)}"
    templates = [
        f"Please {op} for contract {contract}.",
        f"I need to {op} on agreement {contract} effective next Monday.",
        f"Can you {op}? Contract ref is {contract}.",
    ]
    return {
        "text": rng.choice(templates),
        "kind": "impossible",
        "task": task,
    }


def render_request(task, seed):
    """Turn a pick_task result into a natural-language user request."""
    rng = random.Random(seed)

    if task["kind"] == "impossible":
        return _render_impossible_request(task, rng)

    # ~25% of valid tasks get a distractor field mention.
    include_distractor = rng.random() < 0.25
    return _render_valid_request(task, rng, include_distractor)
