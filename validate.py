#!/usr/bin/env python3
"""Standalone validator mirroring the grading surface.

Runs the pipeline from a clean state (forcing the deterministic mock provider so
the result is reproducible regardless of NVIDIA_API_KEY), then asserts the 12
checks the evaluator cares about — including a negative test that a banned phrase
flips a ticket to needs_human_review, and a determinism double-run.

    python validate.py            # exits 0 if all checks pass, 1 otherwise
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
REQUIRED = ["triage.json", "retrieval_results.json", "draft_responses.json",
            "response_checks.json", "review_results.json", "final_responses.json",
            "llm_calls.jsonl"]
LOG_FIELDS = ["stage", "ticket_id", "timestamp", "provider", "model",
              "prompt_hash", "input_artifacts", "output_artifact"]

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))


def run_pipeline(tickets=None, kb=None):
    env = dict(os.environ)
    env["PROVIDER"] = "mock"  # force determinism / no secrets
    cmd = [sys.executable, "-m", "src.run"]
    if tickets:
        cmd += ["--tickets", tickets]
    if kb:
        cmd += ["--kb", kb]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"pipeline exited {r.returncode}\n{r.stdout}\n{r.stderr}")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_jsonl(name):
    return [json.loads(l) for l in (OUT / name).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def snapshot():
    snap = {}
    for name in REQUIRED:
        if name == "llm_calls.jsonl":
            rows = load_jsonl(name)
            for row in rows:
                row.pop("timestamp", None)
            snap[name] = json.dumps(rows, sort_keys=True)
        else:
            snap[name] = (OUT / name).read_text(encoding="utf-8")
    return snap


def config_inputs():
    cfg = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
    return cfg["input_paths"]["tickets"], cfg["input_paths"]["policy_kb"]


def validate_run(ticket_ids):
    triage = load("triage.json")
    retrieval = load("retrieval_results.json")
    drafts = load("draft_responses.json")
    checks = load("response_checks.json")
    review = load("review_results.json")
    finals = load("final_responses.json")
    calls = load_jsonl("llm_calls.jsonl")

    # 1. required artifacts exist
    for name in REQUIRED:
        check(f"artifact_exists:{name}", (OUT / name).exists())
    # 2. valid JSON/JSONL (loads above would have thrown otherwise)
    check("valid_json_and_jsonl", True)
    # 3. input_artifacts point at real paths
    check("input_artifacts_exist",
          all(all((ROOT / p).exists() for p in c["input_artifacts"]) for c in calls))
    # 4. exactly one triage record per ticket (artifact + log)
    tri_calls = [c for c in calls if c["stage"] == "triage"]
    check("one_triage_per_ticket_artifact",
          sorted(r["ticket_id"] for r in triage) == sorted(ticket_ids))
    check("one_triage_per_ticket_log",
          sorted(c["ticket_id"] for c in tri_calls) == sorted(ticket_ids))
    # 5. exactly one draft record per ticket (artifact + log)
    dr_calls = [c for c in calls if c["stage"] == "drafting"]
    check("one_draft_per_ticket_artifact",
          sorted(r["ticket_id"] for r in drafts) == sorted(ticket_ids))
    check("one_draft_per_ticket_log",
          sorted(c["ticket_id"] for c in dr_calls) == sorted(ticket_ids))
    # 5b. exactly one review record per ticket (artifact + log) — Stage 3
    rv_calls = [c for c in calls if c["stage"] == "review"]
    check("one_review_per_ticket_artifact",
          sorted(r["ticket_id"] for r in review) == sorted(ticket_ids))
    check("one_review_per_ticket_log",
          sorted(c["ticket_id"] for c in rv_calls) == sorted(ticket_ids))
    # 6/7. citations subset of retrieved + at least one
    retr_by = {r["ticket_id"]: set(r["retrieved_policy_ids"]) for r in retrieval}
    subset_ok = atleast_ok = True
    for d in drafts:
        cited = set(d["cited_policy_ids"])
        retr = retr_by.get(d["ticket_id"], set())
        subset_ok = subset_ok and cited <= retr
        if retr:
            atleast_ok = atleast_ok and len(cited) >= 1
    check("cited_subset_of_retrieved", subset_ok)
    check("at_least_one_citation_when_policies_exist", atleast_ok)
    # 8. failed check -> needs_human_review
    final_by = {f["ticket_id"]: f for f in finals}
    check("failed_check_needs_human_review",
          all(final_by[c["ticket_id"]]["final_status"] == "needs_human_review"
              for c in checks if not c["passed"]))
    # 8b. a 'ready' final requires both a passing check and reviewer 'approve'
    review_by = {r["ticket_id"]: r for r in review}
    check_by = {c["ticket_id"]: c for c in checks}
    check("ready_requires_check_pass_and_review_approve",
          all(check_by[f["ticket_id"]]["passed"]
              and review_by[f["ticket_id"]]["verdict"] == "approve"
              for f in finals if f["final_status"] == "ready"))
    # 9. every log record has all required fields
    check("log_has_all_required_fields",
          all(all(c.get(f) not in (None, "") for f in LOG_FIELDS) for c in calls))
    # 11. triage != drafting (distinct prompt_hash per ticket)
    th = {c["ticket_id"]: c["prompt_hash"] for c in tri_calls}
    dh = {c["ticket_id"]: c["prompt_hash"] for c in dr_calls}
    check("triage_drafting_distinct_prompt_hash",
          all(th.get(t) != dh.get(t) for t in ticket_ids))
    # escalation ready-only-if-communicated
    tri_by = {t["ticket_id"]: t for t in triage}
    markers = ["escalat", "specialist", "follow up", "next step", "prioritis"]
    esc_ok = True
    for f in finals:
        t = tri_by[f["ticket_id"]]
        if t["should_escalate"] and f["final_status"] == "ready":
            esc_ok = esc_ok and any(m in f["reply"].lower() for m in markers)
    check("escalation_ready_only_if_communicated", esc_ok)


def main():
    tickets_path, kb_path = config_inputs()
    ticket_ids = [t["ticket_id"]
                  for t in json.loads((ROOT / tickets_path).read_text(encoding="utf-8"))]

    # 12. determinism: two runs identical modulo timestamps
    run_pipeline()
    snap1 = snapshot()
    validate_run(ticket_ids)
    run_pipeline()
    snap2 = snapshot()
    check("determinism_modulo_timestamps", snap1 == snap2,
          "" if snap1 == snap2 else "artifacts differ across runs")

    # Negative test: inject a banned phrase -> that ticket becomes needs_human_review.
    if ticket_ids:
        drafts = load("draft_responses.json")
        checks = load("response_checks.json")
        finals = load("final_responses.json")
        cfg = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
        banned = cfg["banned_phrases"][0]
        # simulate the check on a mutated reply using the real checks module
        sys.path.insert(0, str(ROOT))
        from src.checks import run_checks
        from src.finalize import run_finalize
        from src.provider import mock_review
        retrieval = load("retrieval_results.json")
        triage = load("triage.json")
        victim = drafts[0]["ticket_id"]
        mutated = {d["ticket_id"]: dict(d) for d in drafts}
        mutated[victim]["reply"] = mutated[victim]["reply"] + " " + banned
        tickets = json.loads((ROOT / tickets_path).read_text(encoding="utf-8"))
        triage_by = {t["ticket_id"]: t for t in triage}
        rerun_checks = run_checks(mutated,
                                  {r["ticket_id"]: r for r in retrieval},
                                  triage_by, cfg, tickets)
        checks_by = {c["ticket_id"]: c for c in rerun_checks}
        review_by = {t["ticket_id"]: mock_review(t, triage_by[t["ticket_id"]],
                                                  mutated[t["ticket_id"]],
                                                  checks_by[t["ticket_id"]])
                     for t in tickets}
        rerun_final = run_finalize(tickets, triage_by, mutated, checks_by, review_by)
        vf = {f["ticket_id"]: f for f in rerun_final}[victim]
        check("banned_phrase_flips_to_needs_human_review",
              vf["final_status"] == "needs_human_review",
              f"(injected '{banned}' into {victim})")

    passed = [r for r in _RESULTS if r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
    print(f"\n{len(passed)}/{len(_RESULTS)} checks passed")
    return 0 if len(passed) == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
