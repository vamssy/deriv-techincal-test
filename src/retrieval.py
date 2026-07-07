"""Deterministic evidence retrieval (NO LLM).

Scores each policy against a ticket by lowercased token overlap of
subject+message vs title+content+tags, with a small bonus for tag hits. Ranking
is stable (score desc, then policy_id asc). Guarantees drafting always has at
least one policy to cite: a tone/safety policy is pulled in for complaint-like
tickets, and a zero-overlap ticket falls back to the first top_k by policy_id.
"""

import re

_STOPWORDS = {
    "the", "and", "for", "you", "your", "our", "with", "this", "that", "have",
    "has", "was", "were", "are", "not", "but", "can", "cannot", "will", "would",
    "please", "help", "from", "about", "there", "their", "them", "they", "when",
    "what", "why", "how", "who", "into", "out", "get", "got", "any", "all",
    "just", "now", "been", "being", "did", "does", "done", "also", "than",
    "then", "some", "more", "over", "under", "very", "much", "hi", "hello",
}

_SAFETY_TAGS = {"tone", "safety", "communication"}
_COMPLAINT_HINTS = ["angry", "unacceptable", "legal", "lawyer", "sue", "lawsuit",
                    "manager", "complaint", "furious", "escalate", "worst",
                    "terrible", "regulator", "supervisor"]


def _tokenize(text):
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower())
            if len(tok) >= 3 and tok not in _STOPWORDS}


def _ticket_text(ticket):
    return f"{ticket.get('subject', '')} {ticket.get('message', '')}"


def build_index(policies):
    """Precompute token sets per policy (KB_INDEXED stage)."""
    index = {}
    for p in policies:
        tag_text = " ".join(str(t) for t in p.get("tags", []))
        index[p["policy_id"]] = {
            "policy": p,
            "tokens": _tokenize(f"{p.get('title', '')} {p.get('content', '')} {tag_text}"),
            "tag_tokens": _tokenize(tag_text),
        }
    return index


def _is_safety(policy):
    return bool({str(t).lower() for t in policy.get("tags", [])} & _SAFETY_TAGS)


def retrieve_for_ticket(ticket, policies, index, cfg):
    k = min(int(cfg.get("top_k", 3)), len(policies))
    if k <= 0:
        return {"ticket_id": ticket["ticket_id"], "retrieved_policy_ids": [],
                "results": []}

    tt = _tokenize(_ticket_text(ticket))
    scored = []
    for pid, entry in index.items():
        overlap = len(tt & entry["tokens"])
        tag_bonus = 2 * len(tt & entry["tag_tokens"])
        scored.append((pid, overlap + tag_bonus, overlap, tag_bonus))
    scored.sort(key=lambda x: (-x[1], x[0]))  # stable: score desc, policy_id asc

    zero_overlap = all(s[1] == 0 for s in scored)
    top = scored[:k]

    # Pull in a tone/safety policy for complaint-like tickets if none present.
    # Match against tokens (not raw substrings) so 'sue' never hits 'issue'.
    safety_relevant = any(h in tt for h in _COMPLAINT_HINTS)
    if safety_relevant and not any(_is_safety(index[s[0]]["policy"]) for s in top):
        safety_candidates = [s for s in scored if _is_safety(index[s[0]]["policy"])]
        if safety_candidates:
            top = top[:-1] + [safety_candidates[0]]
            top.sort(key=lambda x: (-x[1], x[0]))

    results = []
    for rank, (pid, score, overlap, tag_bonus) in enumerate(top, start=1):
        if zero_overlap:
            expl = (f"no keyword/tag overlap; default policy selected by id order "
                    f"(rank {rank})")
        else:
            expl = (f"score {score} = {overlap} token overlap + {tag_bonus} tag "
                    f"bonus (rank {rank})")
        if safety_relevant and _is_safety(index[pid]["policy"]):
            expl += "; included as tone/safety policy"
        results.append({"policy_id": pid, "score": score, "rank": rank,
                        "ranking_explanation": expl})

    return {
        "ticket_id": ticket["ticket_id"],
        "retrieved_policy_ids": [r["policy_id"] for r in results],
        "results": results,
    }


def run_retrieval(tickets, policies, index, cfg):
    return [retrieve_for_ticket(t, policies, index, cfg) for t in tickets]
