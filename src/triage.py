"""Stage 1 — Triage. One LLM call per ticket -> triage.json.

The LLM (mock or NVIDIA) proposes the classification; this module sanitises the
result in code so a malformed/hallucinated response can never violate the
structural contract (category always in the allowed set, all fields present).
"""

from .provider import mock_triage

_PRIORITIES = ("low", "medium", "high", "urgent")


def render_triage_prompt(ticket, allowed_categories):
    """Deterministic prompt string; its sha256 is the prompt_hash for this call."""
    return (
        "Triage this customer support ticket.\n"
        f"Allowed categories: {', '.join(allowed_categories)}.\n"
        "Choose exactly one category and a priority in [low, medium, high, urgent].\n"
        f"ticket_id: {ticket['ticket_id']}\n"
        f"subject: {ticket.get('subject', '')}\n"
        f"message: {ticket.get('message', '')}\n"
        f"language: {ticket.get('language', 'en')}\n"
    )


def sanitize_triage(result, ticket, allowed_categories):
    result = result if isinstance(result, dict) else {}
    category = result.get("category")
    if category not in allowed_categories:
        category = "general" if "general" in allowed_categories else (
            allowed_categories[0] if allowed_categories else "general")
    priority = result.get("priority")
    if priority not in _PRIORITIES:
        priority = "medium"
    missing = result.get("missing_information")
    if not isinstance(missing, list):
        missing = [] if missing in (None, "") else [str(missing)]
    return {
        "ticket_id": ticket["ticket_id"],
        "category": category,
        "priority": priority,
        "should_escalate": bool(result.get("should_escalate", False)),
        "reason": str(result.get("reason", "")) or "n/a",
        "missing_information": [str(m) for m in missing],
    }


def run_triage(provider, tickets, cfg, tickets_path, output_artifact):
    allowed = cfg["allowed_categories"]
    records = []
    for ticket in tickets:
        prompt = render_triage_prompt(ticket, allowed)
        context = {"ticket": ticket, "allowed_categories": allowed}
        result = provider.complete(
            "triage", ticket["ticket_id"], prompt, context,
            input_artifacts=[tickets_path], output_artifact=output_artifact,
        )
        if result is None:  # provider failed -> deterministic safe fallback
            result = mock_triage(ticket, allowed)
        records.append(sanitize_triage(result, ticket, allowed))
    return records
