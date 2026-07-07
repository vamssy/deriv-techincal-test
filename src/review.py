"""Stage 3 — Review. One LLM call per ticket -> review_results.json.

An independent QA pass over each draft. The verdict is folded into the final
status: a ticket is 'ready' only if it passed the deterministic checks AND the
reviewer approved. The verdict is sanitised in code to a fixed vocabulary so a
malformed/hallucinated response cannot corrupt the pipeline.
"""

from .provider import mock_review

_VERDICTS = ("approve", "revise", "escalate")


def render_review_prompt(ticket, triage, draft, check, retrieved_ids):
    return "\n".join([
        "Review this drafted support reply for safety before it is sent.",
        f"ticket_id: {ticket['ticket_id']}",
        f"customer_message: {ticket.get('message', '')}",
        f"triage_category: {triage['category']}",
        f"triage_priority: {triage['priority']}",
        f"should_escalate: {triage['should_escalate']}",
        f"retrieved_policy_ids: {retrieved_ids}",
        f"cited_policy_ids: {draft.get('cited_policy_ids', [])}",
        f"automated_checks_passed: {check.get('passed')}",
        f"automated_check_issues: {check.get('issues', [])}",
        "drafted_reply:",
        draft.get("reply", ""),
        "Decide a verdict in [approve, revise, escalate].",
    ])


def sanitize_review(result, ticket):
    result = result if isinstance(result, dict) else {}
    verdict = result.get("verdict")
    if verdict not in _VERDICTS:
        verdict = "revise"  # fail safe: unknown verdict -> human review
    issues = result.get("issues")
    if not isinstance(issues, list):
        issues = [] if issues in (None, "") else [str(issues)]
    return {
        "ticket_id": ticket["ticket_id"],
        "verdict": verdict,
        "issues": [str(i) for i in issues],
        "reviewer_notes": str(result.get("reviewer_notes", "")) or "n/a",
    }


def run_review(provider, tickets, triage_by_id, drafts_by_id, checks_by_id,
               retrieval_by_id, input_artifacts, output_artifact):
    records = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        triage = triage_by_id[tid]
        draft = drafts_by_id[tid]
        check = checks_by_id[tid]
        retrieved_ids = retrieval_by_id[tid]["retrieved_policy_ids"]

        prompt = render_review_prompt(ticket, triage, draft, check, retrieved_ids)
        context = {"ticket": ticket, "triage": triage, "draft": draft,
                   "check": check, "retrieved": retrieved_ids}
        result = provider.complete(
            "review", tid, prompt, context,
            input_artifacts=input_artifacts, output_artifact=output_artifact,
        )
        if result is None:  # provider failed -> deterministic safe fallback
            result = mock_review(ticket, triage, draft, check)
        records.append(sanitize_review(result, ticket))
    return records
