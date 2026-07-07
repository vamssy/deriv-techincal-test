# Stage-Gated AI Support Pipeline

A replayable, stage-gated pipeline that turns raw support tickets into reviewed,
policy-grounded draft replies. It reads `tickets.json` and `policy_kb.json` from
disk, then for every ticket runs **triage → retrieval → drafting → checks →
review → finalise** behind an explicit state machine. It produces six JSON/JSONL
artifacts — including per-ticket triage, retrieved evidence, drafted replies,
guardrail results, reviewer verdicts, and a final response pack — plus a full
audit log of every LLM call. It is **stdlib-only**, **deterministic by default**,
and **fully regenerable from a clean checkout with zero secrets**.

## Table of Contents

- [Overview / How It Works](#overview--how-it-works)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Setup](#setup)
- [Usage](#usage)
- [Output Artifacts](#output-artifacts)
- [Project Structure](#project-structure)
- [How It's Validated / Graded](#how-its-validated--graded)
- [Troubleshooting / Notes](#troubleshooting--notes)

## Overview / How It Works

Each ticket flows through an explicit `Stage` state machine (`src/state.py`).
Illegal transitions raise, and finals cannot be produced before the review stage —
so ordering guarantees are enforced by construction, not by convention.

| # | Stage                | Module          | LLM? | Output                        |
|---|----------------------|-----------------|------|-------------------------------|
| 1 | Load + validate inputs | `io_utils.py` | no   | *(in-memory, schema-checked)* |
| 2 | Index the policy KB    | `retrieval.py`| no   | *(in-memory token index)*     |
| 3 | **Triage** (per ticket) | `triage.py`  | #1   | `out/triage.json`             |
| 4 | **Retrieval** (per ticket) | `retrieval.py` | no | `out/retrieval_results.json` |
| 5 | **Drafting** (per ticket) | `drafting.py` | #2 | `out/draft_responses.json`   |
| 6 | **Checks** (guardrails) | `checks.py`   | no   | `out/response_checks.json`    |
| 7 | **Review** (per ticket) | `review.py`   | #3   | `out/review_results.json`     |
| 8 | **Finalise**            | `finalize.py` | no   | `out/final_responses.json`    |
| – | Every LLM call is logged | `provider.py`| –   | `out/llm_calls.jsonl`         |

**Flow:** `tickets.json` + `policy_kb.json` → triage classifies each ticket →
retrieval selects the top policies deterministically → drafting writes a reply
seeing **only** that ticket's retrieved snippets → checks run five deterministic
guardrails → the reviewer gives an independent verdict → finalise decides the
status.

**Key design guarantees (enforced in code, never left to the LLM):**

- Drafting is handed **only** its ticket's retrieved snippets, and citations are
  intersected with the retrieved set — so `cited_policy_ids ⊆ retrieved_policy_ids`
  and every draft cites ≥1 policy, by construction.
- Triage, drafting, and review are **distinct stages** with a distinct
  `prompt_hash` per ticket-stage.
- A ticket is `ready` **only if** it passes all checks **and** the reviewer
  returns `approve`; otherwise `final_status = needs_human_review`.
- **Deterministic by default:** the default provider is an offline mock, so two
  runs on identical inputs produce identical artifacts (timestamps aside).
- **Swap-safe:** no logic keys on sample IDs, titles, wording, order, or counts.
- **Fail loud:** malformed or incomplete input aborts with an actionable message
  and a non-zero exit, before any artifact is written.

## Prerequisites

- **Python 3.10+** (standard library only).
- **No dependencies to install** — no `pip install`, no virtualenv required. The
  NVIDIA call uses `urllib` from the stdlib.
- *(Optional)* `make` for the convenience targets, and `git` to clone.

Verify your interpreter:

```bash
python --version   # or: python3 --version  -> expect 3.10 or newer
```

## Configuration

The pipeline runs fully offline on a deterministic **mock** provider by default.
To use the real LLM, set one environment variable.

**Required for the live LLM — `NVIDIA_API_KEY`:**

```bash
export NVIDIA_API_KEY="nvapi-..."   # your NVIDIA NIM key; never commit this
```

Alternatively, put it in a gitignored `.env` file at the repo root:

```bash
echo 'NVIDIA_API_KEY="nvapi-..."' > .env
```

- **Model:** `z-ai/glm-5.2`, served via NVIDIA NIM (OpenAI-compatible) at
  `https://integrate.api.nvidia.com/v1`.
- **Where to change it:** `config/settings.json` (`model`, `base_url`).
- The key is read only from `os.environ`, never hardcoded, committed, or logged.

**Provider selection** (`provider` in `config/settings.json`, or the `PROVIDER`
env var):

| Value    | Behaviour                                                        |
|----------|-----------------------------------------------------------------|
| `auto`   | *(default)* uses NVIDIA if `NVIDIA_API_KEY` is set, else `mock`  |
| `mock`   | forces the deterministic offline provider (no network, no key)  |
| `nvidia` | forces the NVIDIA NIM provider                                  |

**Other settings** (`config/settings.json`): `top_k` (policies retrieved per
ticket), `min_reply_len`, `banned_phrases`, `allowed_categories`, `temperature`,
`request_timeout`, and input/output paths.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/vamssy/deriv-techincal-test.git
cd deriv-techincal-test

# 2. Confirm Python 3.10+ (no other install step is needed)
python --version

# 3. (Optional) Configure the live LLM. Skip this to run deterministically on mock.
export NVIDIA_API_KEY="nvapi-..."

# 4. Run the pipeline
python -m src.run
```

## Usage

**Run the pipeline** (regenerates all artifacts in `out/`):

```bash
python -m src.run          # or: make run
```

Expect a summary line such as `[done] artifacts in out/  ready=5
needs_human_review=1`, and seven files written to `out/`. With no key set it runs
on the deterministic mock; with `NVIDIA_API_KEY` set it calls the real model and
logs `provider=nvidia`.

**Validate** (asserts the full grading surface; forces the mock provider so the
result is reproducible):

```bash
python validate.py         # or: make validate
```

Expect a per-check `[PASS]/[FAIL]` list ending in `24/24 checks passed` and exit
code `0`.

**Run on different inputs** (defaults read from the repo root):

```bash
python -m src.run --tickets path/to/tickets.json --kb path/to/policy_kb.json
```

**Clean generated artifacts:**

```bash
make clean                 # removes out/
```

## Output Artifacts

All generated files land in `out/` (gitignored; regenerated on every run).

| Path                            | Contents                                                                 |
|---------------------------------|--------------------------------------------------------------------------|
| `out/triage.json`               | Per ticket: `category`, `priority`, `should_escalate`, `reason`, `missing_information`. |
| `out/retrieval_results.json`    | Per ticket: `retrieved_policy_ids` and per-policy `score` + `ranking_explanation`. |
| `out/draft_responses.json`      | Per ticket: `subject`, `reply`, `cited_policy_ids` (⊆ retrieved), `escalation_note`. |
| `out/response_checks.json`      | Per ticket: `passed` and the list of guardrail `issues`.                 |
| `out/review_results.json`       | Per ticket: reviewer `verdict` (`approve`/`revise`/`escalate`), `issues`, `reviewer_notes`. |
| `out/final_responses.json`      | Per ticket: `final_status` (`ready`/`needs_human_review`), `reply`, `supporting_policy_ids`, `review_verdict`, `notes`. |
| `out/llm_calls.jsonl`           | One line per LLM call: `stage`, `ticket_id`, `timestamp`, `provider`, `model`, `prompt_hash`, `input_artifacts`, `output_artifact`. |

## Project Structure

```
.
├── src/
│   ├── run.py             # Entrypoint + orchestrator; drives the state machine end to end
│   ├── state.py           # Stage(Enum) + StateMachine (illegal transitions raise)
│   ├── io_utils.py        # JSON/JSONL I/O, schema guards, .env loader
│   ├── provider.py        # Provider base, MockProvider, NvidiaProvider; prompt_hash + call logging
│   ├── triage.py          # Stage 1: per-ticket classification
│   ├── retrieval.py       # Deterministic top-k policy retrieval + KB index
│   ├── drafting.py        # Stage 2: per-ticket reply drafting (sees only retrieved snippets)
│   ├── checks.py          # Deterministic guardrail checks
│   ├── review.py          # Stage 3: per-ticket reviewer verdict
│   └── finalize.py        # Final status decision
├── config/
│   └── settings.json      # Provider, model, base_url, paths, top_k, guardrail config
├── fixtures/              # Alternate sample inputs used for swap testing
├── tickets.json           # Sample input (evaluator-replaceable)
├── policy_kb.json         # Sample input (evaluator-replaceable)
├── validate.py            # Standalone validator asserting the grading surface
├── Makefile               # run / validate / clean convenience targets
├── out/                   # Generated artifacts (gitignored)
└── README.md
```

## How It's Validated / Graded

`python validate.py` runs the pipeline from a clean state (forcing the mock
provider) and asserts, among others:

- **Artifacts:** all required files exist; every JSON is valid; `llm_calls.jsonl`
  is valid JSONL with all eight fields on every record.
- **Inputs from disk:** each call's `input_artifacts` point at real input paths;
  artifacts reflect the current inputs, not embedded samples.
- **Per-ticket separation:** exactly one `triage`, one `drafting`, and one
  `review` record per ticket, in both the artifact and the log.
- **Citation invariant:** every draft's `cited_policy_ids` is a non-empty subset
  of that ticket's `retrieved_policy_ids` (retrieval precedes drafting).
- **Distinct stages:** triage vs. drafting have distinct `prompt_hash` per ticket.
- **Status logic:** any failed check ⇒ `needs_human_review`; a `ready` final
  requires both a passing check and reviewer `approve`.
- **Determinism:** two runs produce identical artifacts modulo timestamps.
- **Negative test:** injecting a banned phrase flips that ticket to
  `needs_human_review`.

A passing run prints `24/24 checks passed` and exits `0`.

## Troubleshooting / Notes

- **No key set?** That's fine — the run uses the deterministic mock and produces
  valid artifacts. `provider=mock` in the summary confirms this.
- **Live LLM not triggering?** Ensure `NVIDIA_API_KEY` is exported in the same
  shell (or present in `.env`), or set `PROVIDER=nvidia`. `provider=auto` only
  switches to NVIDIA when the key is present.
- **LLM errors / rate limits / timeouts:** each call runs once; on any failure the
  pipeline logs a warning and falls back to a deterministic safe output for that
  ticket-stage (skip-with-warning) — it never crashes or retries in a loop, so
  artifacts stay valid.
- **Clean-run regeneration:** every run deletes stale artifacts first, so
  `make clean && make run` (or deleting `out/` and re-running) rebuilds everything
  from disk identically (mock path).
- **Malformed input:** a clear error is printed to stderr and the process exits
  non-zero without writing partial artifacts.
- **Non-English tickets** are flagged in `missing_information` (not crashed);
  non-ASCII content round-trips as UTF-8.
```
