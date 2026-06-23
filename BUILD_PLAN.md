# Build Plan: Schema-Grounded Payload Generator (Asset-Finance)

> **Instructions for the coding assistant (read this first, applies to the whole project):**
>
> 1. **Keep all code simple and readable.** I am the one who will maintain this. Favor plain, obvious Python over clever one-liners. No premature abstraction, no frameworks where a function will do.
> 2. **Keep the code organized** into the folder structure defined below. One concern per file. Short files are good.
> 3. **Explain everything in simple terms as you go.** Before writing a file, tell me in 2–3 plain sentences what it does and why. After writing it, tell me what to run to see it work. Assume I am a strong .NET engineer who is new to ML tooling — so explain the ML-specific parts (LoRA, tokens, loss, eval) like I'm smart but new to this domain.
> 4. **Work one phase at a time. Stop after each phase and wait for me to confirm before moving on.** Do not race ahead and build everything at once.
> 5. **Comment the non-obvious lines**, especially anything ML-specific. Skip comments on obvious code.
> 6. After each phase, give me a one-line "what we proved" summary so I can see the project taking shape.

---

## The one idea that governs the whole project

The API spec is **always an input given to the model at inference time — never baked into the model's weights.** The model learns the *skill* of reading any spec it is handed and producing a valid payload. We prove it learned a skill (not memory) by testing it on specs it has **never seen during training**. Every design choice below serves this. If anything you build would let the model "memorize" a specific API, that's a bug — flag it to me.

---

## What we're building (plain version)

A small local model (3B parameters) that, given an API spec + a plain-English request, outputs a correct JSON payload — or a clean refusal if the request can't be done. Around it, three thin layers that mimic a real product:

- **Layer A (Retrieval):** find the right spec for a request.
- **Layer B (Mapping):** turn a messy Excel + request into clean field values.
- **Layer C (the fine-tuned model):** generate the payload. *This is the only part we fine-tune.*

The point of the project is not "a model that does payloads" — frontier models already do that. The point is to show **when fine-tuning is the right call (cost at volume, on-prem compliance, tail reliability)** and that I can carve a fuzzy business problem into the one piece worth fine-tuning.

---

## Folder structure (create this in Phase 0)

```
payload-gen/
├── README.md                 # the writeup (final phase)
├── BUILD_PLAN.md             # this file
├── requirements.txt
├── data/                     # generated training data lives here
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl            # generated LAST, with a different seed
├── specs/                    # fictional specs for the retrieval layer (Layer A)
├── adapters/                 # LoRA output from training (created by MLX)
├── src/
│   ├── generator/            # Layer C data generation (the real work)
│   │   ├── invent_spec.py    # makes a random fictional API spec
│   │   ├── pick_task.py      # picks a valid operation + values
│   │   ├── render_request.py # writes the natural-language request
│   │   ├── emit_target.py    # produces the correct payload (or refusal)
│   │   └── build_dataset.py  # ties the above into JSONL rows
│   ├── eval/
│   │   ├── score.py          # schema-validity, exact-match, refusal accuracy
│   │   └── mock_server.py    # optional: executes payloads to confirm validity
│   ├── retrieval/
│   │   └── retrieve.py       # Layer A: find the right spec
│   ├── mapping/
│   │   └── excel_map.py      # Layer B: Excel columns -> schema fields
│   ├── orchestration/
│   │   └── decompose.py      # "book a deal" -> ordered list of operations
│   └── app/
│       └── server.py         # Layer C model server + the full chat flow
└── scripts/
    ├── baseline.py           # run base model BEFORE training (for comparison)
    └── benchmark.py          # base vs fine-tuned vs Claude, final table
```

Keep this structure. If a file gets longer than ~150 lines, tell me and we'll split it.

---

## Phase 0 — Project setup

**Goal:** a working environment and the folder skeleton.

Tasks:
1. Create the folder structure above (empty files with a one-line docstring saying what each will do).
2. Create `requirements.txt` with: `mlx`, `mlx-lm`, `datasets`, `huggingface_hub`, `sentence-transformers`, `numpy`, `pandas`, `openpyxl`, `jsonschema`, `fastapi`, `uvicorn`.
3. Write a 5-line `README.md` stub (one sentence on what the project is).
4. Tell me the exact commands to create the venv and install.

**Explain to me:** what each dependency is for, in one line each.

**Stop here. Wait for me to confirm install worked before Phase 1.**

---

## Phase 1 — Choose and smoke-test the 3B model

**Goal:** confirm the base model runs on my machine (M5 Air, 16GB) before we build anything around it.

Tasks:
1. Use `mlx-community/Llama-3.2-3B-Instruct-4bit` as the model (4-bit so we get QLoRA automatically — the base stays quantized, only the small adapter trains in full precision).
2. Give me the one `mlx_lm.generate` command to run a quick "say hello" generation.
3. Note expected behavior: first run downloads weights; after that it should stream a sentence in a couple seconds.

**Explain to me:** what "4-bit / quantized" means in plain terms, and why a 3B fits in 16GB unified memory when it wouldn't fit a similar discrete GPU.

**Stop here. Wait for me to confirm the model runs.**

---

## Phase 2 — The data generator (Layer C foundation — this is the real work)

**Goal:** a program that emits unlimited correct (spec, request, payload) training rows. Build it in small pieces, one file at a time, testing each before the next.

> Remember the governing idea: every row has a **different** randomly-generated spec, so the model cannot win by memorizing endpoints — only by learning to read whatever spec is in front of it.

Build in this order, stopping to show me output after each:

1. **`invent_spec.py`** — given a random seed, produce a fictional asset-finance API: 3–6 endpoints drawn from a pool (create asset, book lease, post payment, post receipt, amend contract). Each endpoint has fields with types (string, integer, enum, ISO-8601 date), required vs optional flags, and occasional nesting. Randomize field names, ordering, and units (e.g. amount in cents vs dollars) so no two specs look alike. Return it as a Python dict.
   - *Show me:* two example specs printed, so I can see they differ.

2. **`pick_task.py`** — given a spec, pick one endpoint and a legal set of values for its fields. Return the choice.

3. **`render_request.py`** — turn the task into a natural-language sentence ("post a payment of four thousand two hundred against contract LF-8831, effective next Monday"). Vary phrasing. **Inject distractors** sometimes (mention a field the chosen endpoint doesn't accept). **Sometimes produce an impossible request** (asks for an operation no endpoint in this spec supports) — flag these so the next step knows to refuse.

4. **`emit_target.py`** — produce the mechanically-correct payload for the task (your code chose the values, so it knows the answer). For impossible requests, emit a structured refusal like `{"refusal": "no endpoint supports this operation"}`.

5. **`build_dataset.py`** — loop the above to produce JSONL rows in MLX chat format:
   ```json
   {"messages": [
     {"role": "system", "content": "API SPEC:\n<the spec rendered as text>"},
     {"role": "user", "content": "<the natural-language request>"},
     {"role": "assistant", "content": "<the correct payload OR refusal as a JSON string>"}
   ]}
   ```
   Generate ~6,000 rows → `data/train.jsonl`, ~800 → `data/valid.jsonl`.
   **Do NOT generate the test set yet.**

**Explain to me:** why distractors and impossible-requests matter (they teach the model to obey the spec and to refuse cleanly, not to please the user blindly). Show me 3 full example rows so I can read the exact shape.

**Stop here. Wait for me to review the generated data before Phase 3.**

---

## Phase 3 — Baseline (measure BEFORE we train)

**Goal:** record how the untouched base model does, so the improvement later is credible.

Tasks:
1. Generate the **test set now**, with a *different* random seed, into `data/test.jsonl`. This guarantees test specs never appeared in training. State the seed in a comment so it's reproducible.
2. Write `scripts/baseline.py`: run every test row through the **base** model (no adapter) and record schema-validity %, exact-match %, and refusal accuracy.
3. Print a small results table.

**Explain to me:** why measuring before training is the step most people skip, and why a separate-seed test set is the whole proof that we're testing skill not memory.

**Stop here. Show me the baseline numbers.**

---

## Phase 4 — Fine-tune the LoRA adapter

**Goal:** train the small adapter on our data.

Tasks:
1. Give me the `mlx_lm.lora` command. Use: `--batch-size 1` (16GB limit), `--num-layers 8`, `--iters 600`, `--learning-rate 2e-4`, `--save-every 100`, `--grad-checkpoint`, `--adapter-path ./adapters`, `--data ./data`.
2. Tell me what healthy training looks like: loss drops steeply in the first ~100 iterations and flattens by 300–500. If it flatlines high, the data has a problem; if still dropping at 600, raise iters.

**Explain to me, in plain terms:** what LoRA is (we freeze the big model and train tiny add-on layers), what "loss" means (how wrong the model's guesses are), and what an "iteration" is. Keep it concrete.

**Stop here. Wait for training to finish; show me the loss curve / final loss.**

---

## Phase 5 — Test the fine-tuned model on UNSEEN specs

**Goal:** the payoff moment.

Tasks:
1. Give me a `mlx_lm.generate` command that loads the adapter (`--adapter-path ./adapters`) and runs against a spec pulled from the **test** set (unseen during training).
2. Run three demo cases: (a) a valid request → expect a correct payload, (b) a request with a distractor field → expect the distractor ignored, (c) an impossible request → expect a clean refusal.

**Explain to me:** why success on an unseen spec proves skill rather than memory — connect it explicitly back to the governing idea at the top.

**Stop here. Show me the three demo outputs.**

---

## Phase 6 — Eval harness + three-way benchmark

**Goal:** turn anecdotes into a defensible table.

Tasks:
1. `src/eval/score.py` — for each test row, score mechanically: schema-validity (use `jsonschema`), exact-match / JSON-diff against ground truth, refusal accuracy on impossible requests. No human judgment anywhere.
2. `src/eval/mock_server.py` (optional but nice) — a tiny FastAPI app that actually *executes* the generated payload against the fictional spec, to confirm it works and not just looks right.
3. `scripts/benchmark.py` — produce the headline table comparing **base-3B vs fine-tuned-3B vs Claude** (call Claude via the Anthropic API for the frontier number), each split by *seen-spec* vs *unseen-spec*, plus a rough cost-per-payload column.

**Explain to me:** what "mechanically verifiable" means and why it makes this eval trustworthy (no vibes). Remind me the win condition is **fine-tuned-3B landing near Claude on validity while being far cheaper** — not beating Claude on accuracy.

**Stop here. Show me the benchmark table.**

---

## Phase 7 — Layer A: retrieval (thin but real)

**Goal:** prove the spec is *fetched*, not memorized.

Tasks:
1. Put ~12 fictional specs as JSON files in `specs/`.
2. `src/retrieval/retrieve.py` — embed each spec once with `sentence-transformers` (`all-MiniLM-L6-v2`), and given a request, return the best-matching spec by cosine similarity. Keep it to a handful of functions.

**Explain to me:** what an "embedding" is in one plain paragraph, and why retrieval (not fine-tuning) is the right tool for "which spec do I need" (the spec set changes; you don't want it frozen in weights).

**Stop here.**

---

## Phase 8 — Layer B: Excel → fields (the vivid demo piece)

**Goal:** messy spreadsheet in, clean field values out.

Tasks:
1. `src/mapping/excel_map.py` — read an `.xlsx` with `pandas`/`openpyxl`, fuzzy-match its column names to a given schema's fields, coerce types (dates → ISO-8601, currency text → integer cents, blank cells → defaults). Return the cleaned values dict.
2. Include a small sample `.xlsx` to demo with.

**Explain to me:** why this is the part a non-technical bank user actually cares about, and why it's *mapping* (deterministic-ish) rather than something to fine-tune.

**Stop here.**

---

## Phase 9 — Orchestration (thin) + chatbot wiring

**Goal:** assemble the full flow behind a simple chat endpoint.

Tasks:
1. `src/orchestration/decompose.py` — turn a vague request ("book a deal") into an ordered list of operations (create asset → create contract → post booking). For v1 a short rule-based map or a single Claude API call is fine. Keep it small; we're showing the seam exists, not solving general planning.
2. Serve the fine-tuned model: `mlx_lm.server --model <id> --adapter-path ./adapters --port 8080`.
3. `src/app/server.py` — a FastAPI backend implementing the full flow:
   ```
   request (+ optional Excel)
     -> retrieve spec (Layer A)
     -> map Excel -> values (Layer B)
     -> decompose into operations (orchestration)
     -> generate payload per operation (Layer C model @ :8080)
     -> return payloads / refusals
   ```
4. A minimal single-file HTML chat page to drive it.

**Explain to me:** how the four layers hand off to each other, and point out which single layer (C) is the fine-tuned one.

**Stop here. Show me the end-to-end demo working.**

---

## Phase 10 — The writeup (what actually gets me hired)

**Goal:** the README that frames everything.

Tasks:
1. An architecture diagram (the four layers from Phase 9) — ASCII or a simple image.
2. The three-way benchmark table from Phase 6.
3. A decision section sorting three real problems onto the **RAG / fine-tune / prompt** axis with reasoning:
   - Documentation Q&A → RAG (knowledge changes, needs traceability)
   - Ticket dedup / related-ticket → retrieval + embeddings (changes daily)
   - Payload generation → fine-tune (behavior not knowledge; mechanically verifiable; cost + on-prem at scale)
4. The thesis line: *a 3B model matches the frontier model on payload validity for asset-finance transactions at a fraction of the cost, deployable on-prem — and here is exactly which layer was worth fine-tuning and why the rest weren't.*

**Explain to me:** why the decision section (knowing when NOT to fine-tune) is the most senior signal in the whole project.

---

## Suggested order of attack

- **Core (do first):** Phases 0–6. This is the defensible heart — a fine-tuned model proven on unseen specs, with a real benchmark.
- **Architecture story:** Phases 7 and 10.
- **Demo polish:** Phases 8–9, once the core works.

Build the core first. Don't let the demo layers cause scope creep before the model is trained and benchmarked.

---

## Reminders for the assistant (repeat to yourself each phase)

- Simple, readable code over clever code.
- One concern per file; flag files over ~150 lines.
- Explain before and after each file, in plain terms, ML concepts included.
- One phase at a time; stop and wait for my confirmation.
- End each phase with a one-line "what we proved."
- The spec is always an input, never memorized — if any code violates this, stop and tell me.
