"""Layer A: embed specs and retrieve the best match for a natural-language request."""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from invent_spec import invent_spec, render_spec_text  # noqa: E402

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SPECS_DIR = Path(__file__).resolve().parents[2] / "specs"
SPEC_SEEDS = list(range(500_001, 500_013))

# Plain-English hooks for embedding + keyword boost (one row per operation).
OPERATION_HINTS = {
    "create_asset": {
        "description": "create asset, register equipment, add vehicle or aircraft to the fleet",
        "keywords": ["create asset", "register asset", "new asset", "add equipment", "register equipment"],
    },
    "book_lease": {
        "description": "book lease, create lease contract, start rental or finance agreement",
        "keywords": ["book lease", "book a lease", "new lease", "lease booking", "lease contract"],
    },
    "post_receipt": {
        "description": "post receipt, record cash received, log customer payment or collection",
        "keywords": ["post receipt", "post a receipt", "record receipt", "cash received", "collection", "received payment"],
    },
    "amend_contract": {
        "description": "amend contract, modify agreement, change rate term or covenant",
        "keywords": ["amend contract", "amendment", "rate change", "term extension", "modify contract"],
    },
}

KEYWORD_BOOST = 0.30  # added to cosine score when a keyword phrase matches


def _slugify(api_name):
    return api_name.lower().replace(" ", "_").replace("-", "_")


def _endpoint_embed_text(spec, endpoint):
    """One endpoint = one retrieval vector (not the whole spec)."""
    op = endpoint["operation"]
    hints = OPERATION_HINTS.get(op, {"description": op.replace("_", " "), "keywords": []})
    field_names = []
    for field in endpoint["fields"]:
        if field["type"] == "object":
            field_names.extend(child["name"] for child in field["fields"])
        else:
            field_names.append(field["name"])

    return (
        f"Operation: {op}\n"
        f"{hints['description']}\n"
        f"API: {spec['api_name']} v{spec['version']}\n"
        f"POST {endpoint['path']}\n"
        f"Fields: {', '.join(field_names)}"
    )


def _keyword_boost(request, operation):
    text = request.lower()
    for phrase in OPERATION_HINTS.get(operation, {}).get("keywords", []):
        if phrase in text:
            return KEYWORD_BOOST
    return 0.0


def build_spec_library(specs_dir=DEFAULT_SPECS_DIR, seeds=SPEC_SEEDS):
    """Write one JSON file per fictional API spec."""
    specs_dir = Path(specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        spec = invent_spec(seed)
        slug = _slugify(spec["api_name"])
        path = specs_dir / f"{slug}_{seed}.json"
        payload = {"seed": seed, "spec": spec, "spec_text": render_spec_text(spec)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)
    return written


def load_specs(specs_dir=DEFAULT_SPECS_DIR):
    specs_dir = Path(specs_dir)
    entries = []
    for path in sorted(specs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "id": path.stem,
                "path": str(path),
                "seed": data.get("seed"),
                "spec": data["spec"],
                "spec_text": data.get("spec_text") or render_spec_text(data["spec"]),
            }
        )
    if not entries:
        raise FileNotFoundError(f"No specs in {specs_dir}. Run: python src/retrieval/retrieve.py --build")
    return entries


def _build_endpoint_entries(spec_entries):
    """Flatten specs into one searchable row per endpoint."""
    rows = []
    for spec_entry in spec_entries:
        for endpoint in spec_entry["spec"]["endpoints"]:
            rows.append(
                {
                    "spec_id": spec_entry["id"],
                    "seed": spec_entry["seed"],
                    "spec": spec_entry["spec"],
                    "spec_text": spec_entry["spec_text"],
                    "operation": endpoint["operation"],
                    "endpoint": endpoint,
                    "embed_text": _endpoint_embed_text(spec_entry["spec"], endpoint),
                }
            )
    return rows


def embed_texts(model, texts):
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors)


class SpecRetriever:
    """Embed each endpoint separately; rank by cosine similarity + keyword boost."""

    def __init__(self, specs_dir=DEFAULT_SPECS_DIR, model_name=DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.spec_entries = load_specs(specs_dir)
        self.endpoint_entries = _build_endpoint_entries(self.spec_entries)
        self.vectors = embed_texts(self.model, [e["embed_text"] for e in self.endpoint_entries])

    def retrieve(self, request, top_k=1):
        query = embed_texts(self.model, [request])[0]
        ranked = []
        for i, entry in enumerate(self.endpoint_entries):
            embed_score = float(self.vectors[i] @ query)
            kw_boost = _keyword_boost(request, entry["operation"])
            ranked.append((embed_score + kw_boost, embed_score, kw_boost, entry))

        ranked.sort(key=lambda x: x[0], reverse=True)

        results = []
        for total, embed_score, kw_boost, entry in ranked[:top_k]:
            results.append(
                {
                    "id": entry["spec_id"],
                    "operation": entry["operation"],
                    "score": total,
                    "embedding_score": embed_score,
                    "keyword_boost": kw_boost,
                    "seed": entry["seed"],
                    "spec": entry["spec"],
                    "spec_text": entry["spec_text"],
                    "endpoint": entry["endpoint"],
                }
            )
        return results[0] if top_k == 1 else results


def retrieve(request, specs_dir=DEFAULT_SPECS_DIR, model_name=DEFAULT_MODEL):
    return SpecRetriever(specs_dir=specs_dir, model_name=model_name).retrieve(request)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the spec retrieval library.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--specs-dir", default=str(DEFAULT_SPECS_DIR))
    parser.add_argument("--request", default="")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.build:
        paths = build_spec_library(args.specs_dir)
        print(f"Wrote {len(paths)} specs -> {args.specs_dir}")

    if args.request:
        retriever = SpecRetriever(specs_dir=args.specs_dir)
        print(f"Indexed {len(retriever.endpoint_entries)} endpoints across {len(retriever.spec_entries)} specs")
        results = retriever.retrieve(args.request, top_k=args.top_k)
        if args.top_k == 1:
            results = [results]
        print()
        for i, hit in enumerate(results, start=1):
            boost = f"+{hit['keyword_boost']:.2f}" if hit["keyword_boost"] else ""
            print(
                f"#{i}  {hit['operation']} @ {hit['id']}  "
                f"score={hit['score']:.3f} (embed={hit['embedding_score']:.3f}{boost})"
            )
