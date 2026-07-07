"""Entrypoint + orchestrator: drives the StateMachine end to end.

    python -m src.run [--tickets PATH] [--kb PATH] [--config PATH] [--out DIR]

Provider is resolved at runtime: MockProvider by default (deterministic, no
secrets); NvidiaProvider only when NVIDIA_API_KEY is present (or PROVIDER=nvidia).
"""

import argparse
import os
import sys
from pathlib import Path

from .checks import run_checks
from .drafting import run_drafting
from .finalize import run_finalize
from .io_utils import (InputError, dump_json, load_dotenv, load_json,
                       reset_jsonl, validate_kb, validate_tickets)
from .provider import MockProvider, NvidiaProvider
from .retrieval import build_index, run_retrieval
from .review import run_review
from .state import Stage, StateMachine
from .triage import run_triage

CONFIG_PATH = "config/settings.json"
_ARTIFACTS = ("triage.json", "retrieval_results.json", "draft_responses.json",
              "response_checks.json", "review_results.json",
              "final_responses.json", "llm_calls.jsonl")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage-gated AI support pipeline")
    ap.add_argument("--tickets", help="path to tickets.json (default from config)")
    ap.add_argument("--kb", help="path to policy_kb.json (default from config)")
    ap.add_argument("--config", default=CONFIG_PATH, help="path to settings.json")
    ap.add_argument("--out", help="output directory (default from config)")
    return ap.parse_args(argv)


def load_config(args):
    cfg = load_json(args.config)
    cfg.setdefault("input_paths", {})
    if args.tickets:
        cfg["input_paths"]["tickets"] = args.tickets
    if args.kb:
        cfg["input_paths"]["policy_kb"] = args.kb
    if args.out:
        cfg["out_dir"] = args.out
    if os.environ.get("PROVIDER"):
        cfg["provider"] = os.environ["PROVIDER"]
    return cfg


def resolve_provider_name(cfg):
    name = cfg.get("provider", "auto")
    if name == "auto":
        return "nvidia" if os.environ.get("NVIDIA_API_KEY") else "mock"
    return name


def make_provider(cfg, log_path):
    if resolve_provider_name(cfg) == "nvidia":
        return NvidiaProvider(cfg, log_path)
    return MockProvider(cfg, log_path)


def _reset_artifacts(out_dir, log_path):
    """Delete stale artifacts so a re-run regenerates everything from scratch."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name in _ARTIFACTS:
        fp = Path(out_dir) / name
        if fp.exists():
            fp.unlink()
    reset_jsonl(log_path)  # ensure an (empty) log exists even for zero tickets


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(".env")  # optional local key file; never committed

    try:
        cfg = load_config(args)
        tickets_path = cfg["input_paths"]["tickets"]
        kb_path = cfg["input_paths"]["policy_kb"]
        out_dir = cfg.get("out_dir", "out")

        paths = {name: str(Path(out_dir) / name) for name in _ARTIFACTS}
        log_path = paths["llm_calls.jsonl"]

        sm = StateMachine()

        tickets = validate_tickets(load_json(tickets_path))
        policies = validate_kb(load_json(kb_path))
        sm.advance(Stage.INPUTS_LOADED)
        sm.advance(Stage.TICKETS_PARSED)
        index = build_index(policies)
        sm.advance(Stage.KB_INDEXED)
    except InputError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    _reset_artifacts(out_dir, log_path)
    provider = make_provider(cfg, log_path)
    print(f"[info] provider={provider.name} model={provider.model} "
          f"tickets={len(tickets)} policies={len(policies)}")

    # Stage 1: Triage (one LLM call/ticket)
    triage = run_triage(provider, tickets, cfg, tickets_path, paths["triage.json"])
    dump_json(paths["triage.json"], triage)
    sm.advance(Stage.TICKET_TRIAGED)

    # Retrieval (deterministic; must precede drafting)
    retrieval = run_retrieval(tickets, policies, index, cfg)
    dump_json(paths["retrieval_results.json"], retrieval)
    sm.advance(Stage.EVIDENCE_RETRIEVED)

    triage_by_id = {r["ticket_id"]: r for r in triage}
    retrieval_by_id = {r["ticket_id"]: r for r in retrieval}
    policy_by_id = {p["policy_id"]: p for p in policies}

    # Stage 2: Drafting (one LLM call/ticket; sees ONLY its retrieved snippets).
    # input_artifacts are the real on-disk inputs whose contents feed this call.
    draft_inputs = [tickets_path, kb_path]
    drafts = run_drafting(provider, tickets, triage_by_id, retrieval_by_id,
                          policy_by_id, cfg, draft_inputs,
                          paths["draft_responses.json"])
    dump_json(paths["draft_responses.json"], drafts)
    sm.advance(Stage.RESPONSE_DRAFTED)

    drafts_by_id = {r["ticket_id"]: r for r in drafts}

    # Deterministic checks
    checks = run_checks(drafts_by_id, retrieval_by_id, triage_by_id, cfg, tickets)
    dump_json(paths["response_checks.json"], checks)
    sm.advance(Stage.RESPONSE_CHECKED)

    checks_by_id = {r["ticket_id"]: r for r in checks}

    # Stage 3: Review (one LLM call/ticket; verdict folded into final status)
    review = run_review(provider, tickets, triage_by_id, drafts_by_id,
                        checks_by_id, retrieval_by_id, [tickets_path, kb_path],
                        paths["review_results.json"])
    dump_json(paths["review_results.json"], review)
    sm.advance(Stage.RESPONSE_REVIEWED)

    review_by_id = {r["ticket_id"]: r for r in review}

    # Finalise (guarded: cannot run before RESPONSE_REVIEWED)
    sm.require(Stage.RESPONSE_REVIEWED)
    finals = run_finalize(tickets, triage_by_id, drafts_by_id, checks_by_id,
                          review_by_id)
    dump_json(paths["final_responses.json"], finals)
    sm.advance(Stage.RESPONSE_FINALISED)

    ready = sum(1 for f in finals if f["final_status"] == "ready")
    nhr = sum(1 for f in finals if f["final_status"] == "needs_human_review")
    print(f"[done] artifacts in {out_dir}/  ready={ready} "
          f"needs_human_review={nhr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
