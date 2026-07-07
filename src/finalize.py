"""Final status decision -> final_responses.json.

A ticket is 'ready' only when BOTH gates agree:
  * deterministic checks passed, AND
  * the Stage 3 reviewer returned verdict 'approve'.

Otherwise final_status = needs_human_review, with notes explaining why:
  * failed check           -> needs_human_review
  * reviewer 'escalate'    -> needs_human_review (a human owns escalations)
  * reviewer 'revise'      -> needs_human_review

Because escalations never come back 'approve', they are never 'ready', so the
"escalation may stay ready only if it communicates the next step" criterion holds
by construction.
"""


def run_finalize(tickets, triage_by_id, drafts_by_id, checks_by_id, review_by_id):
    records = []
    for ticket in tickets:
        tid = ticket["ticket_id"]
        triage = triage_by_id[tid]
        draft = drafts_by_id[tid]
        check = checks_by_id[tid]
        review = review_by_id[tid]
        notes = []

        verdict = review.get("verdict", "revise")
        if not check["passed"]:
            final_status = "needs_human_review"
            notes.append("failed_checks: " + "; ".join(check["issues"]))
        elif verdict != "approve":
            final_status = "needs_human_review"
            notes.append(f"reviewer_verdict: {verdict}")
            if review.get("reviewer_notes"):
                notes.append(f"reviewer_notes: {review['reviewer_notes']}")
        else:
            final_status = "ready"

        for item in triage.get("missing_information", []):
            notes.append(f"missing_information: {item}")

        records.append({
            "ticket_id": tid,
            "category": triage["category"],
            "priority": triage["priority"],
            "final_status": final_status,
            "reply": draft["reply"],
            "supporting_policy_ids": draft["cited_policy_ids"],
            "review_verdict": verdict,
            "notes": notes,
        })
    return records
