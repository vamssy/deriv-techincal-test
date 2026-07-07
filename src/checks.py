"""Deterministic guardrail checks (NO LLM) -> response_checks.json.

Five checks per ticket: missing citations, out-of-set citations, banned phrases,
too-short reply, and escalation-communicated mismatch. A ticket passes only if
it raises zero issues.
"""

_ESCALATION_MARKERS = ["escalat", "specialist", "follow up", "follow-up",
                       "next step", "forwarded", "prioritis", "prioritiz"]


def run_checks(drafts_by_id, retrieval_by_id, triage_by_id, cfg, tickets):
    banned = [b.lower() for b in cfg.get("banned_phrases", []) if b]
    min_len = int(cfg.get("min_reply_len", 40))
    records = []

    for ticket in tickets:
        tid = ticket["ticket_id"]
        draft = drafts_by_id[tid]
        retrieved = set(retrieval_by_id[tid]["retrieved_policy_ids"])
        triage = triage_by_id[tid]

        cited = draft.get("cited_policy_ids") or []
        reply = draft.get("reply", "") or ""
        reply_l = reply.lower()
        issues = []

        # 1. missing citations
        if not cited:
            issues.append("no_citations")

        # 2. cited not subset of retrieved
        out_of_set = [c for c in cited if c not in retrieved]
        if out_of_set:
            issues.append("citation_out_of_set:" + ",".join(map(str, out_of_set)))

        # 3. banned phrases
        for phrase in banned:
            if phrase in reply_l:
                issues.append(f"banned_phrase:{phrase}")

        # 4. empty / too-short reply
        if len(reply.strip()) < min_len:
            issues.append("reply_too_short")

        # 5. escalation mismatch
        if triage.get("should_escalate"):
            communicated = bool(draft.get("escalation_note")) and \
                any(m in reply_l for m in _ESCALATION_MARKERS)
            if not communicated:
                issues.append("escalation_not_communicated")

        records.append({"ticket_id": tid, "passed": not issues, "issues": issues})

    return records
