"""Layer B: fuzzy-match Excel columns to schema fields and coerce types."""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# Endpoint field shapes match invent_spec / specs/*.json.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))


def _normalize_label(text):
    text = str(text).lower().strip()
    text = re.sub(r"[$#/]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _flatten_fields(fields, prefix=""):
    """List leaf fields with dotted paths for nested objects."""
    flat = []
    for field in fields:
        if field["type"] == "object":
            nested_prefix = f"{prefix}{field['name']}."
            flat.extend(_flatten_fields(field["fields"], nested_prefix))
        else:
            flat.append({**field, "path": f"{prefix}{field['name']}"})
    return flat


def _similarity(a, b):
    return SequenceMatcher(None, _normalize_label(a), _normalize_label(b)).ratio()


# Common spreadsheet headers → schema field name hints (beyond fuzzy match).
FIELD_ALIASES = {
    "contract_ref": ["contract", "contract id", "contract #", "agreement", "lease id"],
    "lease_id": ["contract", "contract id", "contract #", "agreement", "lease id"],
    "asset_id": ["equipment", "eqpt", "asset", "asset tag", "asset code", "equipment code"],
    "asset_ref": ["equipment", "eqpt", "asset", "asset tag", "asset code", "equipment code"],
    "asset_tag": ["equipment", "eqpt", "asset tag", "equipment code"],
    "equipment_code": ["equipment", "eqpt", "asset code", "equipment code"],
    "lessee_name": ["lessee", "customer", "borrower", "client"],
    "monthly_payment": ["monthly rent", "rent", "installment", "payment"],
    "start_date": ["lease start", "start date", "commencement", "start dt"],
    "lease_start": ["lease start", "start date", "commencement"],
    "commencement_date": ["lease start", "start date", "commencement"],
    "end_date": ["maturity", "end date", "lease end"],
    "maturity_date": ["maturity", "end date", "lease end"],
    "lease_end": ["maturity", "end date", "lease end"],
    "receipt_amount": ["amount", "cash received", "receipt", "payment amount"],
    "received_date": ["received", "collection date", "deposit date"],
}


def _column_field_score(column, field):
    """Score how well a spreadsheet column matches a schema field."""
    scores = [_similarity(column, field["name"])]
    for alias in FIELD_ALIASES.get(field["name"], []):
        scores.append(_similarity(column, alias))

    col_norm = _normalize_label(column)
    if field["type"] == "date" and any(k in col_norm for k in ("date", "start", "maturity", "end", "commence")):
        scores.append(0.85)
    if field["type"] == "string" and "id" in field["name"] and any(k in col_norm for k in ("start", "maturity", "date")):
        scores.append(0.0)

    return max(scores)


def match_columns(excel_columns, endpoint_fields, threshold=0.45):
    """
    Map Excel column names to schema field paths.

    Returns {excel_column: field_path} for each matched pair.
    """
    leaves = _flatten_fields(endpoint_fields)
    mapping = {}
    used_paths = set()

    for col in excel_columns:
        best_path = None
        best_score = threshold
        for field in leaves:
            if field["path"] in used_paths:
                continue
            score = _column_field_score(col, field)
            if score > best_score:
                best_score = score
                best_path = field["path"]

        if best_path:
            mapping[col] = best_path
            used_paths.add(best_path)

    return mapping


def _parse_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    parsed = pd.to_datetime(value, dayfirst=False, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _parse_integer(value, unit="dollars"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if not text:
            return None
        text = text.replace("$", "").replace(",", "")
        # Handle "20,017.64" style currency in dollars → integer dollars or cents per unit.
        number = float(text)

    if unit == "cents":
        if isinstance(value, str) and "$" in value:
            return int(round(number * 100))
        return int(number)

    # Dollars stored as whole dollars (round 20017.64 → 20018).
    return int(round(number))


def _parse_string(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def coerce_value(raw, field):
    """Coerce one Excel cell to the schema type."""
    if field["type"] == "date":
        return _parse_date(raw)
    if field["type"] == "integer":
        return _parse_integer(raw, field.get("unit", "dollars"))
    if field["type"] == "enum":
        text = _parse_string(raw)
        if text is None:
            return None
        normalized = text.lower().replace(" ", "_").replace("-", "_")
        for option in field.get("enum", []):
            if normalized == option or normalized in option:
                return option
        return normalized
    return _parse_string(raw)


def _set_nested(target, path, value):
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def map_excel_to_values(xlsx_path, endpoint, sheet=0):
    """
    Read a messy spreadsheet row and return cleaned field values for one endpoint.

    endpoint: one entry from spec["endpoints"] (includes fields + operation).
    Uses the first data row by default.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if df.empty:
        return {}

    row = df.iloc[0]
    leaves = {f["path"]: f for f in _flatten_fields(endpoint["fields"])}
    col_map = match_columns(list(df.columns), endpoint["fields"])

    values = {}
    for excel_col, path in col_map.items():
        field = leaves[path]
        raw = row[excel_col]
        if (raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == ""):
            if not field["required"]:
                continue
        coerced = coerce_value(raw, field)
        if coerced is None:
            continue
        _set_nested(values, path, coerced)

    return values


def map_excel_with_report(xlsx_path, endpoint, sheet=0):
    """Same as map_excel_to_values but also returns the column mapping for debugging."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    row = df.iloc[0]
    leaves = {f["path"]: f for f in _flatten_fields(endpoint["fields"])}
    col_map = match_columns(list(df.columns), endpoint["fields"])

    values = {}
    for excel_col, path in col_map.items():
        field = leaves[path]
        coerced = coerce_value(row[excel_col], field)
        if coerced is not None:
            _set_nested(values, path, coerced)

    return {"values": values, "column_map": col_map}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Demo Excel → schema field mapping.")
    parser.add_argument("--xlsx", default="data/sample_deal.xlsx")
    parser.add_argument("--spec", default="specs/harbor_leasing_api_500006.json")
    parser.add_argument("--operation", default="book_lease")
    args = parser.parse_args()

    spec_data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    endpoint = next(ep for ep in spec_data["spec"]["endpoints"] if ep["operation"] == args.operation)

    report = map_excel_with_report(args.xlsx, endpoint)
    print("Column mapping:")
    for col, path in report["column_map"].items():
        print(f"  {col!r} -> {path}")
    print("\nCleaned values:")
    print(json.dumps(report["values"], indent=2))
