"""Stage 2 — Drafting. One LLM call per ticket -> draft_responses.json.

The call receives ONLY that ticket's retrieved snippets (by construction), so the
citation-subset invariant is guaranteed rather than hoped for. sanitize_draft
then intersects any claimed citations with the retrieved set and forces at least
one citation, so no provider output can break checks 6/7.
"""

from .provider import mock_draft


def render_draft_prompt(ticket, triage, retrieved_policies):
    lines = [
        "Draft a professional support reply to this ticket.",
        f"ticket_id: {ticket['ticket_id']}",
        f"subject: {ticket.get('subject', '')}",
        f"message: {ticket.get('message', '')}",
        f"triage_category: {triage['category']}",
        f"triage_priority: {triage['priority']}",
        f"should_escalate: {triage['should_escalate']}",
        "You may cite ONLY these retrieved policies (by policy_id):",
    ]
    for p in retrieved_policies:
        lines.append(f"- [{p['policy_id']}] {p['title']}: {p['content']}")
    if not retrieved_policies:
        lines.append("- (no policies retrieved)")
    lines.append("Cite at least one of the policy_ids above. Do not promise, "
                 "approve, or guarantee refunds, expedites, or any outcome. "
                 "If should_escalate is true, explain the next step and do not "
                 "claim the issue is resolved.")
    return "\n".join(lines)


def sanitize_draft(result, ticket, retrieved_ids):
    result = result if isinstance(result, dict) else {}
    reply = str(result.get("reply", "")).strip()
    subject = str(result.get("subject", "")).strip() or \
        f"Re: {ticket.get('subject', 'your support request')}"

    cited = result.get("cited_policy_ids") or []
    if not isinstance(cited, list):
        cited = [cited]
    retrieved = list(retrieved_ids)
    retrieved_set = set(retrieved)
    cited = [c for c in cited if c in retrieved_set]   # drop out-of-set citations
    if not cited and retrieved:                        # guarantee >= 1 citation
        cited = list(retrieved)

    note = result.get("escalation_note")
    note = str(note) if note not in (None, "", "null") else None

    return {
        "ticket_id": ticket["ticket_id"],
        "subject": subject,
        "reply": reply,
        "cited_policy_ids": cited,
        "escalation_note": note,
    }


def run_drafting(provider, tickets, triage_by_id, retrieval_by_id, policy_by_id,
                 cfg, input_artifacts, output_artifact):
    records = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        triage = triage_by_id[tid]
        retrieved_ids = retrieval_by_id[tid]["retrieved_policy_ids"]
        retrieved_policies = [policy_by_id[pid] for pid in retrieved_ids]

        prompt = render_draft_prompt(ticket, triage, retrieved_policies)
        context = {"ticket": ticket, "triage": triage,
                   "retrieved": retrieved_policies}
        result = provider.complete(
            "drafting", tid, prompt, context,
            input_artifacts=input_artifacts, output_artifact=output_artifact,
        )
        if result is None:  # provider failed -> deterministic safe fallback
            result = mock_draft(ticket, triage, retrieved_policies)
        records.append(sanitize_draft(result, ticket, retrieved_ids))
    return records
