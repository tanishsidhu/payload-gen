"""Generate a random fictional asset-finance API spec from a seed."""

import json
import random
import string

# Canonical operations the generator knows how to fill and score.
OPERATIONS = [
    "create_asset",
    "book_lease",
    "post_receipt",
    "amend_contract",
]

# Operations we may ask for that are NOT in every spec (used for impossible requests).
UNSUPPORTED_OPERATIONS = [
    "delete_contract",
    "transfer_ownership",
    "close_fiscal_period",
    "reverse_journal",
]

FIELD_SYNONYMS = {
    "asset_id": ["asset_id", "asset_tag", "equipment_code", "asset_ref"],
    "asset_type": ["asset_type", "category", "asset_class"],
    "purchase_price": ["purchase_price", "acquisition_cost", "asset_cost"],
    "acquisition_date": ["acquisition_date", "purchase_date", "in_service_date"],
    "contract_ref": ["contract_ref", "lease_id", "agreement_no", "contract_id"],
    "lessee_name": ["lessee_name", "borrower_name", "customer_name"],
    "start_date": ["start_date", "commencement_date", "lease_start"],
    "end_date": ["end_date", "maturity_date", "lease_end"],
    "monthly_payment": ["monthly_payment", "installment_amount", "rent_amount"],
    "effective_date": ["effective_date", "value_date", "posting_date"],
    "receipt_amount": ["receipt_amount", "collected_amount", "cash_received"],
    "received_date": ["received_date", "collection_date", "deposit_date"],
    "payer_reference": ["payer_reference", "remitter_ref", "source_ref"],
    "amendment_type": ["amendment_type", "change_type", "modification_kind"],
    "new_terms": ["new_terms", "revised_clause", "updated_conditions"],
}

OPERATION_FIELDS = {
    "create_asset": [
        ("asset_id", "string", True),
        ("asset_type", "enum", True),
        ("purchase_price", "integer", True),
        ("acquisition_date", "date", True),
    ],
    "book_lease": [
        ("contract_ref", "string", True),
        ("asset_id", "string", True),
        ("lessee_name", "string", True),
        ("start_date", "date", True),
        ("end_date", "date", False),
        ("monthly_payment", "integer", True),
    ],
    "post_receipt": [
        ("contract_ref", "string", True),
        ("receipt_amount", "integer", True),
        ("received_date", "date", True),
        ("payer_reference", "string", False),
    ],
    "amend_contract": [
        ("contract_ref", "string", True),
        ("amendment_type", "enum", True),
        ("effective_date", "date", True),
        ("new_terms", "string", False),
    ],
}

ENUM_VALUES = {
    "asset_type": ["vehicle", "equipment", "aircraft", "real_estate"],
    "amendment_type": ["rate_change", "term_extension", "party_change", "covenant_update"],
}

API_NAME_PARTS = [
    ("Northwind", "Capital"),
    ("Summit", "Finance"),
    ("Harbor", "Leasing"),
    ("Atlas", "Asset"),
    ("Pioneer", "Credit"),
    ("BlueRock", "Funding"),
]


def _pick_name(rng, canonical):
    return rng.choice(FIELD_SYNONYMS[canonical])


def _random_path(rng, operation):
    slug = operation.replace("_", "-")
    prefix = rng.choice(["/api", "/v1", "/v2", ""])
    suffix = rng.choice(["", "/submit", "/execute"])
    return f"{prefix}/{slug}{suffix}"


def _maybe_nest_fields(rng, fields):
    """Occasionally group 2 fields under a nested object."""
    if len(fields) < 3 or rng.random() > 0.35:
        return fields

    idx = rng.randint(0, len(fields) - 2)
    nested_name = rng.choice(["details", "metadata", "terms", "payment_info"])
    nested_fields = fields[idx : idx + 2]
    remaining = fields[:idx] + fields[idx + 2 :]
    remaining.append(
        {
            "name": nested_name,
            "type": "object",
            "required": True,
            "fields": nested_fields,
        }
    )
    return remaining


def _build_field(rng, canonical, field_type, required):
    name = _pick_name(rng, canonical)
    field = {"name": name, "type": field_type, "required": required}

    if field_type == "integer" and canonical in {
        "purchase_price",
        "monthly_payment",
        "receipt_amount",
    }:
        # Same meaning, different unit labels — forces the model to read the spec.
        field["unit"] = rng.choice(["cents", "dollars"])

    if field_type == "enum":
        field["enum"] = list(ENUM_VALUES[canonical])

    return field


def _build_endpoint(rng, operation):
    raw_fields = OPERATION_FIELDS[operation]
    fields = [_build_field(rng, c, t, req) for c, t, req in raw_fields]
    rng.shuffle(fields)
    fields = _maybe_nest_fields(rng, fields)

    return {
        "operation": operation,
        "method": "POST",
        "path": _random_path(rng, operation),
        "fields": fields,
    }


def invent_spec(seed):
    """Return a unique fictional API spec as a Python dict."""
    rng = random.Random(seed)
    part_a, part_b = rng.choice(API_NAME_PARTS)
    version = f"{rng.randint(1, 3)}.{rng.randint(0, 9)}"

    count = rng.randint(3, min(6, len(OPERATIONS)))
    chosen_ops = rng.sample(OPERATIONS, k=count)
    endpoints = [_build_endpoint(rng, op) for op in chosen_ops]

    return {
        "api_name": f"{part_a} {part_b} API",
        "version": version,
        "endpoints": endpoints,
    }


def render_spec_text(spec):
    """Render a spec dict as plain text for the model's system prompt."""
    lines = [f"API: {spec['api_name']} v{spec['version']}", ""]

    for ep in spec["endpoints"]:
        lines.append(f"POST {ep['path']}  ({ep['operation']})")
        _render_fields(lines, ep["fields"], indent=2)
        lines.append("")

    return "\n".join(lines).rstrip()


def _render_fields(lines, fields, indent):
    pad = " " * indent
    for field in fields:
        if field["type"] == "object":
            req = "required" if field["required"] else "optional"
            lines.append(f"{pad}{field['name']} (object, {req}):")
            _render_fields(lines, field["fields"], indent + 2)
            continue

        req = "required" if field["required"] else "optional"
        type_label = field["type"]
        if field["type"] == "enum":
            type_label = f"enum {field['enum']}"
        if "unit" in field:
            type_label = f"{type_label}, unit={field['unit']}"

        lines.append(f"{pad}{field['name']} ({type_label}, {req})")


def all_field_names(spec):
    """Flat list of every field name in the spec (including nested)."""
    names = []
    for ep in spec["endpoints"]:
        names.extend(_collect_field_names(ep["fields"]))
    return names


def _collect_field_names(fields):
    names = []
    for field in fields:
        if field["type"] == "object":
            names.extend(_collect_field_names(field["fields"]))
        else:
            names.append(field["name"])
    return names


if __name__ == "__main__":
    for seed in (42, 99):
        spec = invent_spec(seed)
        print(f"=== seed {seed} ===")
        print(json.dumps(spec, indent=2))
        print()
