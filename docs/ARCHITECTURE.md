# Architecture & Design — Stage-Gated AI Support Pipeline

## Purpose & Scope

This document describes the design of a stage-gated, replayable pipeline that turns
raw customer-support tickets into reviewed, policy-grounded draft replies. Given two
inputs on disk — `tickets.json` and `policy_kb.json` — the pipeline runs each ticket
through six stages (**triage → retrieval → drafting → checks → review → finalise**)
behind an explicit state machine (`src/state.py`), and emits six JSON/JSONL artifacts
plus a full audit log of every LLM call (`out/llm_calls.jsonl`). It is standard-library
only (no `pip`), deterministic by default (an offline mock provider), and structured so
that its correctness guarantees are enforced *in code* rather than trusted to the model.
The scope of this document is the runtime architecture, the module contracts, the
determinism/replayability model, and the production trade-offs considered — it is not a
usage guide (see `README.md` for setup and commands).

---

## Design Principles

The design is driven by six principles, each traceable to specific code:

1. **Deterministic by default.** The default provider is an offline `MockProvider`
   (`src/provider.py`) whose outputs are a pure function of the ticket text and config.
   Two runs on identical inputs produce byte-identical artifacts modulo one field
   (timestamps). Provider resolution defaults to `mock` unless a real key is present
   (`src/run.py::resolve_provider_name`).

2. **Standard-library only (no pip).** No third-party dependency is imported anywhere.
   The real LLM call is made with `urllib.request` from the stdlib
   (`src/provider.py::NvidiaProvider._generate`); hashing uses `hashlib`, config/IO use
   `json` and `pathlib`. A clean checkout runs with just Python 3.10+.

3. **Guardrails enforced in code (structural invariants never trusted to the LLM).**
   The model *proposes*; the code *disposes*. Every LLM output passes through a
   `sanitize_*` function (`sanitize_triage`, `sanitize_draft`, `sanitize_review`) that
   coerces it back onto a fixed contract, and the citation-subset invariant is enforced
   by intersection in `src/drafting.py::sanitize_draft`, not by prompt instruction.

4. **Swap-safety.** No logic keys on sample ticket IDs, policy titles, exact wording,
   input order, or item counts. Input validators check *shape* only
   (`src/io_utils.py::validate_tickets` / `validate_kb`); the mock keys on category
   keywords, never on IDs; retrieval scores generic token overlap. Inputs are freely
   replaceable by an evaluator.

5. **Fail-loud input validation.** Malformed or incomplete input raises `InputError`
   (`src/io_utils.py`) which `main` catches and turns into a stderr message plus a
   non-zero exit **before any artifact is written** (`src/run.py`, the `try` block
   around loading/validation returns exit code `2`).

6. **Replayable / idempotent runs.** Every run deletes stale artifacts first
   (`src/run.py::_reset_artifacts`) and truncates the JSONL log
   (`src/io_utils.py::reset_jsonl`), so artifacts are regenerated from scratch each run
   — never appended to or partially updated.

---

## State Machine

The pipeline order is encoded as an explicit finite state machine in `src/state.py`.
The `Stage` enum (an `IntEnum`) defines ten ordered stages:

```python
class Stage(IntEnum):
    INIT = 0
    INPUTS_LOADED = 1
    TICKETS_PARSED = 2
    KB_INDEXED = 3
    TICKET_TRIAGED = 4
    EVIDENCE_RETRIEVED = 5
    RESPONSE_DRAFTED = 6
    RESPONSE_CHECKED = 7
    RESPONSE_REVIEWED = 8
    RESPONSE_FINALISED = 9
```

**Why it exists.** Ordering guarantees (retrieval precedes drafting; review precedes
finalise) are safety-critical: a draft that cites a policy it never retrieved, or a
"ready" verdict produced before review, would be a silent correctness bug. Encoding the
order as a machine makes the ordering a *construction-time invariant* rather than a
convention a future edit could quietly break.

**How illegal transitions raise.** `StateMachine.advance(target)` permits *only* the
immediate successor:

```python
def advance(self, target: Stage) -> None:
    if int(target) != int(self.stage) + 1:
        raise IllegalTransition(f"illegal transition {self.stage.name} -> {target.name}")
    self.stage = target
    self.history.append(target)
```

Any skip, repeat, or reversal raises `IllegalTransition`, so an orchestrator bug fails
loud instead of emitting artifacts out of order. `src/run.py` walks the machine
step-by-step: `INPUTS_LOADED → TICKETS_PARSED → KB_INDEXED` during load, then
`TICKET_TRIAGED → EVIDENCE_RETRIEVED → RESPONSE_DRAFTED → RESPONSE_CHECKED →
RESPONSE_REVIEWED → RESPONSE_FINALISED` as each stage completes.

**The finalize guard.** Before finalising, `run.py` calls `sm.require(Stage.RESPONSE_REVIEWED)`:

```python
def require(self, minimum: Stage) -> None:
    if int(self.stage) < int(minimum):
        raise IllegalTransition(f"stage {self.stage.name} is before required {minimum.name}")
```

This asserts the review stage has completed before any final status is computed, so a
"ready" decision can never be produced ahead of the reviewer's verdict.

---

## Stage-by-Stage Walkthrough

Each per-ticket stage owns exactly one module, writes exactly one artifact, and (for the
three LLM stages) makes exactly one model call per ticket. The deterministic stages
(retrieval, checks, finalise) never touch the LLM.

| Stage | Module (function) | LLM? | Output artifact | Invariant enforced in code |
|-------|-------------------|------|-----------------|-----------------------------|
| Triage | `src/triage.py::run_triage` | yes (1/ticket) | `out/triage.json` | `category ∈ allowed_categories`; `priority ∈ {low,medium,high,urgent}`; all fields present (`sanitize_triage`) |
| Retrieval | `src/retrieval.py::run_retrieval` | no | `out/retrieval_results.json` | Deterministic top-k; ≥1 policy available to cite; stable ordering |
| Drafting | `src/drafting.py::run_drafting` | yes (1/ticket) | `out/draft_responses.json` | `cited_policy_ids ⊆ retrieved_policy_ids` and non-empty (`sanitize_draft`) |
| Checks | `src/checks.py::run_checks` | no | `out/response_checks.json` | Five guardrails; `passed` iff zero issues |
| Review | `src/review.py::run_review` | yes (1/ticket) | `out/review_results.json` | `verdict ∈ {approve,revise,escalate}`, else fail-safe to `revise` (`sanitize_review`) |
| Finalise | `src/finalize.py::run_finalize` | no | `out/final_responses.json` | `ready` iff checks passed **and** verdict is `approve` |

### Triage — `src/triage.py`

**Input:** each ticket. **Output:** `triage.json` (one record/ticket). **LLM:** yes.
`render_triage_prompt` builds a deterministic prompt string (its sha256 becomes the
`prompt_hash`), the provider proposes a classification, and `sanitize_triage` forces the
result back onto the contract: an unknown `category` collapses to `general` (or the first
allowed category), an unknown `priority` becomes `medium`, and `missing_information` is
coerced to a list of strings. So a hallucinated category can never leak downstream. On
provider failure the code falls back to `mock_triage`.

### Retrieval — `src/retrieval.py`

**Input:** each ticket + the pre-built KB index. **Output:** `retrieval_results.json`.
**LLM:** no — fully deterministic. This stage must precede drafting (enforced by the
state machine) because drafting may cite only what retrieval selected.

The scoring model (`retrieve_for_ticket`):

- **Tokenisation.** Both sides are lowercased and split on non-alphanumerics; tokens
  shorter than 3 chars and a fixed stopword set are dropped (`_tokenize`). The ticket
  side is `subject + message`; the policy side is `title + content + tags` (precomputed
  once per policy in `build_index`, the `KB_INDEXED` stage).
- **Score.** For each policy, `score = token_overlap + tag_bonus`, where
  `token_overlap = |ticket_tokens ∩ policy_tokens|` and
  `tag_bonus = 2 × |ticket_tokens ∩ tag_tokens|` — a tag match counts double.
- **Stable ranking / tie-break.** `scored.sort(key=lambda x: (-x[1], x[0]))` sorts by
  score descending, then `policy_id` ascending, so ties resolve deterministically and
  runs are reproducible.
- **Tone/safety inclusion for complaint-like tickets.** If the ticket text contains a
  complaint hint (`_COMPLAINT_HINTS`, matched against *tokens* so "sue" never fires
  inside "issue") and no tone/safety policy (tags in `{tone, safety, communication}`) is
  already in the top-k, the lowest-ranked slot is swapped for the best safety candidate
  and the list is re-sorted. This guarantees de-escalation guidance is on hand for angry
  tickets.
- **Zero-overlap fallback.** If *every* policy scores 0, the sort by `policy_id`
  ascending still yields a deterministic top-k, and each explanation notes "default
  policy selected by id order". This guarantees drafting always has at least one policy
  to cite, even for a ticket with no lexical overlap.

Each result carries `policy_id`, `score`, `rank`, and a human-readable
`ranking_explanation`.

### Drafting — `src/drafting.py`

**Input:** ticket, its triage record, and **only** its retrieved policies. **Output:**
`draft_responses.json`. **LLM:** yes. `run_drafting` resolves `retrieved_ids` for the
ticket and passes *only* those policy objects into `render_draft_prompt` and the model
context — the model physically cannot see policies it did not retrieve. `sanitize_draft`
then enforces the citation invariant structurally:

```python
cited = [c for c in cited if c in retrieved_set]   # drop out-of-set citations
if not cited and retrieved:                        # guarantee >= 1 citation
    cited = list(retrieved)
```

So `cited_policy_ids ⊆ retrieved_policy_ids` and is non-empty whenever any policy was
retrieved — regardless of what the model returned. On provider failure it falls back to
`mock_draft`.

### Checks — `src/checks.py`

**Input:** drafts, retrieval, triage, config, tickets. **Output:**
`response_checks.json`. **LLM:** no. `run_checks` runs five deterministic guardrails per
ticket and marks `passed` only when the issue list is empty:

1. **`no_citations`** — the draft cites nothing.
2. **`citation_out_of_set:<ids>`** — a citation is not in the retrieved set (defence in
   depth behind `sanitize_draft`).
3. **`banned_phrase:<phrase>`** — the reply contains a phrase from
   `cfg["banned_phrases"]` (e.g. "full refund", "we guarantee"), matched
   case-insensitively.
4. **`reply_too_short`** — the trimmed reply is shorter than `min_reply_len` (default 40).
5. **`escalation_not_communicated`** — the ticket should escalate but the reply lacks
   both an `escalation_note` and any escalation marker (`_ESCALATION_MARKERS`).

### Review — `src/review.py`

**Input:** ticket, triage, draft, check result, retrieved ids. **Output:**
`review_results.json`. **LLM:** yes — an *independent* QA pass. `render_review_prompt`
shows the reviewer the check results and drafted reply; `sanitize_review` clamps the
verdict to `{approve, revise, escalate}` and **fails safe** — any unknown verdict becomes
`revise` (a human review), never `approve`. On provider failure it falls back to
`mock_review`, whose deterministic logic is: failed checks → `revise`; else should-escalate
→ `escalate`; else → `approve`.

### Finalise — `src/finalize.py`

**Input:** triage, drafts, checks, review. **Output:** `final_responses.json`. **LLM:**
no. `run_finalize` computes `final_status` from two AND-ed gates:

```python
if not check["passed"]:
    final_status = "needs_human_review"       # failed guardrail
elif verdict != "approve":
    final_status = "needs_human_review"        # reviewer said revise/escalate
else:
    final_status = "ready"
```

Because escalations resolve to `escalate`/`revise` (never `approve`), an escalated ticket
is never `ready` — so the "an escalation may only be ready if it communicates the next
step" criterion holds by construction. `missing_information` items from triage are appended
to the final `notes` for visibility.

---

## Provider Abstraction

All model access sits behind one small interface in `src/provider.py`, so the pipeline
logic is provider-agnostic.

**`Provider` (base).** Owns the *call-log record assembly*, not generation. Its
`complete(stage, ticket_id, prompt, context, input_artifacts, output_artifact)` method:

- calls the subclass `_generate` inside a `try/except` that catches **any** exception,
- writes exactly one log record per call (`append_jsonl`) **regardless of success or
  failure**, attaching a `warning` field and printing a `[warn] … -> using deterministic
  fallback` line on failure,
- returns the structured dict, or `None` on failure so the caller can substitute a
  deterministic fallback.

`_generate` is abstract (`raise NotImplementedError`), so each provider implements only
generation.

**`MockProvider` (default).** Deterministic and offline; `_generate` dispatches by stage
to `mock_triage` / `mock_draft` / `mock_review`. These key on **category keywords**
(`_CATEGORY_KEYWORDS`, `_URGENT_KEYWORDS`, `_ESCALATE_KEYWORDS`, …) matched as whole words
(`\b…\b`, so "sue" never matches "issue"), never on sample IDs — preserving swap-safety.
Its model label is `cfg["mock_model"]` (default `mock-v1`).

**`NvidiaProvider`.** An OpenAI-compatible chat-completions call to NVIDIA NIM made purely
with `urllib.request` (no SDK). It posts to `{base_url}/chat/completions` with a
per-stage system prompt (`_SYSTEM_PROMPTS`) plus the rendered user prompt, model
`z-ai/glm-5.2`, and `temperature = 0` (from config) to minimise variance. The response
content is parsed by `_extract_json`, which strips code fences and, failing a direct
`json.loads`, extracts the first `{…}` block. Defaults: `base_url =
https://integrate.api.nvidia.com/v1`, `request_timeout = 60`.

**`provider: auto` resolution.** `resolve_provider_name` maps the config/`PROVIDER`
value: `auto` → `nvidia` **iff** `NVIDIA_API_KEY` is present in `os.environ`, else `mock`;
`mock`/`nvidia` force the respective provider. `make_provider` then instantiates the
chosen class.

**`prompt_hash`.** `prompt_hash(prompt)` is the sha256 hex digest of the exact rendered
prompt string. It is provider-independent (mock and NVIDIA hash the same prompt
identically) and is what proves each stage is a *distinct* call — triage and drafting have
different rendered prompts, hence different hashes, for the same ticket.

**Skip-with-warning fallback.** Each stage makes **one** call, with **no retry loop**. Any
failure (missing key, network error, timeout, unparseable body, empty result) is caught
in `Provider.complete`, logged with a `warning`, and surfaced as `None`; the stage then
substitutes its deterministic mock fallback (`if result is None: result = mock_…`). The
pipeline never crashes on an LLM failure and never blocks in a retry storm.

**Key handling.** `NVIDIA_API_KEY` is read *only* from `os.environ`
(`NvidiaProvider.__init__`), is never hardcoded or committed, and is **never written to
the log** — the log record schema contains no key field, and the `Authorization` header is
constructed at call time and discarded. A gitignored `.env` may populate the environment
via `src/io_utils.py::load_dotenv`, which never overrides an existing variable.

---

## Determinism & Replayability

**Timestamps are the only per-run variance.** Every artifact is a pure function of the
inputs and config on the mock path, except the `timestamp` field on each log record, set
by `_now_iso()` (`datetime.now(timezone.utc).isoformat()`). The code comment in
`provider.py` states this explicitly: "Timestamp is the ONLY per-run variance in the
artifacts."

**Artifacts regenerated from scratch each run.** `_reset_artifacts` deletes all six JSON
artifacts and truncates `llm_calls.jsonl` (via `reset_jsonl`) at the start of every run,
so there is no stale state, no accumulation, and no partial-update path. `make clean &&
make run` and re-running in place produce identical trees.

**How `validate.py` proves a double-run is identical modulo timestamps.** The validator
forces `PROVIDER=mock` for reproducibility, runs the pipeline twice, and `snapshot()`s the
output tree both times. For the JSONL log it pops the `timestamp` field from every record
before comparison; the six JSON artifacts are compared verbatim. The
`determinism_modulo_timestamps` check asserts `snap1 == snap2`. The validator also runs a
negative test (injecting a banned phrase into a draft and confirming the ticket flips to
`needs_human_review` through the real `run_checks`/`run_finalize`), and asserts the full
grading surface (artifact existence, one record per ticket-stage, the citation subset
invariant, distinct triage/drafting `prompt_hash`, and the status logic), printing
`N/N checks passed`.

---

## Data Contracts

Each stage writes a JSON array of per-ticket records (via `dump_json`, `ensure_ascii=False`
so non-ASCII round-trips as UTF-8). The audit log is JSONL.

| Artifact | Per-record fields |
|----------|-------------------|
| `triage.json` | `ticket_id`, `category`, `priority`, `should_escalate`, `reason`, `missing_information[]` |
| `retrieval_results.json` | `ticket_id`, `retrieved_policy_ids[]`, `results[]` (each: `policy_id`, `score`, `rank`, `ranking_explanation`) |
| `draft_responses.json` | `ticket_id`, `subject`, `reply`, `cited_policy_ids[]` (⊆ retrieved), `escalation_note` (string or null) |
| `response_checks.json` | `ticket_id`, `passed` (bool), `issues[]` |
| `review_results.json` | `ticket_id`, `verdict` (`approve`/`revise`/`escalate`), `issues[]`, `reviewer_notes` |
| `final_responses.json` | `ticket_id`, `category`, `priority`, `final_status` (`ready`/`needs_human_review`), `reply`, `supporting_policy_ids[]`, `review_verdict`, `notes[]` |

**`llm_calls.jsonl` — the eight required fields** (asserted present and non-empty by
`validate.py::LOG_FIELDS`):

1. `stage` — `triage` / `drafting` / `review`
2. `ticket_id`
3. `timestamp` — ISO-8601 UTC (the only per-run variance)
4. `provider` — `mock` / `nvidia`
5. `model` — e.g. `mock-v1` or `z-ai/glm-5.2`
6. `prompt_hash` — sha256 of the rendered prompt
7. `input_artifacts` — list of the on-disk input paths whose contents fed the call
8. `output_artifact` — the artifact this call contributes to

On a fallback, an optional ninth `warning` field is added; it is not part of the required
set.

---

## Extension Points

The design isolates the three axes an evaluator or maintainer is most likely to extend:

- **Add a provider.** Subclass `Provider` and implement `_generate(self, stage, prompt,
  context) -> dict | None`. The base class already handles logging, the record schema,
  the `prompt_hash`, and the skip-with-warning contract, so a new backend only implements
  generation. Wire it into `make_provider` / `resolve_provider_name` in `src/run.py`.

- **Add a guardrail check.** Append a new rule inside the per-ticket loop of
  `src/checks.py::run_checks` that pushes a string onto `issues`. Because `passed = not
  issues` and any failed check forces `needs_human_review` in `run_finalize`, a new check
  is automatically wired into the final status logic — no other file changes.

- **Add a category.** Add the string to `allowed_categories` in `config/settings.json`.
  Triage's `sanitize_triage` validates against this list, so the new category is accepted
  immediately; optionally add keyword hints to `_CATEGORY_KEYWORDS` in `src/provider.py`
  to improve mock classification. No code change is required for the contract to hold.

---

## Production Considerations & Alternatives

- **Orchestration: hand-rolled state machine vs. LangGraph.** A graph framework such as
  LangGraph would give typed nodes, edges, and a persisted execution graph out of the box.
  It was deliberately *not* used here because (a) the stdlib-only constraint forbids the
  dependency, and (b) the pipeline is a strictly linear, ten-node DAG whose ordering
  guarantees are fully captured by a ~40-line `IntEnum` + `advance`/`require`
  (`src/state.py`). A hand-rolled machine keeps the failure mode explicit
  (`IllegalTransition`), the audit trail trivial (`history`), and the whole system
  auditable in one file — the framework's generality would add surface area without buying
  a guarantee we do not already enforce. LangGraph (or a Temporal-style workflow engine)
  becomes attractive once the graph branches, fans out, or needs durable
  suspend/resume — a natural future migration path.

- **Response caching for reproducible real-LLM runs.** The mock path is already
  reproducible; the NVIDIA path is not (network + model non-determinism, mitigated only by
  `temperature = 0`). A content-addressed cache keyed on `prompt_hash` (already computed
  for every call) would make real-LLM runs replayable and cheap — a cache hit returns the
  prior response, and `prompt_hash` is the natural cache key.

- **Batching & rate-limit handling.** Today each ticket-stage is a single, serial,
  no-retry call (`Provider.complete`). For high ticket volumes this should grow request
  batching and a bounded retry-with-backoff around `urllib` for `429`/`5xx`, plus a
  concurrency limit. The single-shot design is intentional for the current scope: it keeps
  runs fast, cheap, and free of retry storms, and any transient failure degrades to a
  deterministic fallback rather than blocking the batch.

- **Observability via `llm_calls.jsonl`.** The audit log is the observability backbone:
  every call records its stage, ticket, provider, model, `prompt_hash`, input/output
  artifacts, timestamp, and any `warning`. This supports cost/latency attribution, fallback
  detection (count records with a `warning`), and per-ticket replay. In production it would
  feed a metrics pipeline (calls per stage, fallback rate, p50/p95 latency).

---

## Known Limitations & Trade-offs

- **Keyword retrieval vs. embeddings.** Retrieval scores lexical token overlap plus a tag
  bonus (`src/retrieval.py`), not semantic similarity. It is deterministic, dependency-free,
  and fully explainable (`ranking_explanation`), but it misses synonymy and paraphrase — a
  ticket that says "can't sign in" will not lexically match a policy titled
  "authentication reset". An embedding retriever would improve recall at the cost of a
  dependency, a model, and non-determinism. The tone/safety inclusion rule and the
  zero-overlap fallback are deliberate hedges against the weakest lexical cases.

- **Mock reply templating.** `mock_draft` produces a fixed, templated reply. It exists to
  make the pipeline reproducible and testable offline, not to be a good customer reply — it
  guarantees valid *structure* (citations, escalation language, length) but not natural
  prose. Real quality requires the NVIDIA provider.

- **Single-shot LLM calls.** One call per ticket-stage with no retry means a transient
  failure yields a deterministic fallback rather than a retried real-model answer. This
  trades peak answer quality for guaranteed valid artifacts and bounded cost/latency — the
  right trade for a graded, replayable pipeline, but one a production deployment would
  revisit with caching, batching, and backoff as noted above.

- **Note on the state-guard comment.** The `src/state.py` module docstring mentions
  refusing finals before `RESPONSE_CHECKED`; the orchestrator's actual guard is stricter —
  `sm.require(Stage.RESPONSE_REVIEWED)` in `src/run.py` — so finals cannot be produced
  before the review stage. The enforced behaviour (review-gated) is the authoritative one.
