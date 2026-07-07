"""LLM providers + call logging.

Two providers behind one interface:
  * MockProvider   — deterministic, offline, no secrets. The DEFAULT so a clean
                     checkout reproduces identical artifacts (modulo timestamps).
  * NvidiaProvider — OpenAI-compatible NVIDIA NIM call over stdlib urllib. Used
                     only when NVIDIA_API_KEY is present. One call per ticket-stage;
                     on any failure it returns None (skip-with-warning) and the
                     caller falls back to a deterministic safe output.

prompt_hash is sha256 of the exact rendered prompt and is provider-independent.
The API key is read from os.environ only and is NEVER written to the log.
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

from .io_utils import append_jsonl


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    # Timestamp is the ONLY per-run variance in the artifacts.
    return datetime.now(timezone.utc).isoformat()


class Provider:
    """Base provider: renders nothing, but owns the call log record assembly."""

    name = "base"

    def __init__(self, cfg, log_path):
        self.cfg = cfg
        self.log_path = log_path
        self.model = "base"

    def complete(self, stage, ticket_id, prompt, context,
                 input_artifacts, output_artifact):
        """Generate a structured dict for this ticket-stage and log the call.

        Returns the generated dict, or None if the provider failed (caller
        handles the fallback). The log record is written regardless.
        """
        result = None
        note = None
        try:
            result = self._generate(stage, prompt, context)
        except Exception as exc:  # noqa: BLE001 - skip-with-warning, never crash
            note = f"{type(exc).__name__}: {exc}"
            result = None
        if result is None and note is None:
            note = "empty_or_unparseable_response"

        record = {
            "stage": stage,
            "ticket_id": ticket_id,
            "timestamp": _now_iso(),
            "provider": self.name,
            "model": self.model,
            "prompt_hash": prompt_hash(prompt),
            "input_artifacts": list(input_artifacts),
            "output_artifact": output_artifact,
        }
        if note is not None:
            record["warning"] = note
            print(f"[warn] {self.name} {stage} {ticket_id}: {note} "
                  f"-> using deterministic fallback", file=sys.stderr)
        append_jsonl(self.log_path, record)
        return result

    def _generate(self, stage, prompt, context):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic mock generation (keyed to keywords/categories, never sample IDs)
# --------------------------------------------------------------------------- #
_CATEGORY_KEYWORDS = {
    "billing": ["refund", "charge", "charged", "billed", "billing", "invoice",
                "payment", "subscription", "overcharge", "price", "card"],
    "technical": ["error", "bug", "crash", "crashing", "broken", "not working",
                  "fails", "failed", "glitch", "load", "startup", "freeze"],
    "account": ["login", "log in", "sign in", "password", "locked", "access",
                "account", "2fa", "authenticate", "recover"],
    "shipping": ["ship", "shipping", "delivery", "deliver", "package", "parcel",
                 "tracking", "courier", "arrive", "dispatch", "order"],
    "product": ["feature", "how do i", "does it", "can i", "upgrade", "plan",
                "available", "supported"],
    "complaint": ["unacceptable", "terrible", "awful", "worst", "angry",
                  "disappointed", "furious", "complaint", "legal", "lawyer",
                  "lawsuit", "sue", "manager", "supervisor", "regulator"],
}

_URGENT_KEYWORDS = ["urgent", "asap", "immediately", "critical", "outage",
                    "down", "lawsuit", "legal", "sue", "data loss", "breach"]
_HIGH_KEYWORDS = ["angry", "unacceptable", "escalate", "complaint", "refund",
                  "locked", "cannot", "broken"]
_LOW_KEYWORDS = ["question", "wondering", "curious", "how do i", "just checking"]

_ESCALATE_KEYWORDS = ["legal", "lawyer", "lawsuit", "sue", "manager",
                      "supervisor", "regulator", "gdpr", "data breach",
                      "chargeback", "escalate", "unacceptable"]


def _text(ticket):
    return f"{ticket.get('subject', '')} {ticket.get('message', '')}".lower()


def _kw_hits(text, keywords):
    """Count keywords present as whole words/phrases (avoids 'sue' in 'issue')."""
    return sum(1 for k in keywords
               if re.search(r"\b" + re.escape(k) + r"\b", text))


def _kw_any(text, keywords):
    return any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keywords)


def mock_triage(ticket, allowed_categories):
    text = _text(ticket)
    # Pick the category with the most keyword hits; ties break by definition order.
    best_cat, best_hits = "general", 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if cat not in allowed_categories:
            continue
        hits = _kw_hits(text, kws)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    category = best_cat if best_hits > 0 else (
        "general" if "general" in allowed_categories
        else (allowed_categories[0] if allowed_categories else "general"))

    if _kw_any(text, _URGENT_KEYWORDS):
        priority = "urgent"
    elif _kw_any(text, _HIGH_KEYWORDS):
        priority = "high"
    elif _kw_any(text, _LOW_KEYWORDS):
        priority = "low"
    else:
        priority = "medium"

    should_escalate = (
        category == "complaint"
        or priority == "urgent"
        or _kw_any(text, _ESCALATE_KEYWORDS)
    )

    missing = []
    lang = str(ticket.get("language", "en")).lower()
    if lang and lang != "en":
        missing.append(f"non_english_ticket:{lang}")
    if category in ("billing", "shipping", "account") and not re.search(r"\d", text):
        missing.append("missing_identifier")

    return {
        "ticket_id": ticket["ticket_id"],
        "category": category,
        "priority": priority,
        "should_escalate": bool(should_escalate),
        "reason": (f"Classified as '{category}' at '{priority}' priority based on "
                   f"message keywords; escalate={should_escalate}."),
        "missing_information": missing,
    }


def mock_draft(ticket, triage, retrieved):
    name = ticket.get("customer_name")
    greeting = f"Hi {name}," if name else "Hello,"
    category_readable = triage.get("category", "general").replace("_", " ")

    if retrieved:
        cited = ", ".join(f"{p['title']} [{p['policy_id']}]" for p in retrieved)
        policy_line = (f"Based on our relevant policies ({cited}), here is how we "
                       f"will proceed:")
    else:
        policy_line = "Here is how we will proceed:"

    steps = ("- I have reviewed the details you shared.\n"
             "- Our team will look into this in line with the policies above and "
             "keep you informed.")

    escalation_block = ""
    escalation_note = None
    if triage.get("should_escalate"):
        escalation_block = (
            "\n\nGiven the nature of your request, I have escalated this to our "
            "specialist team, who will follow up with the next steps. I cannot "
            "confirm a resolution yet, but your case has been prioritised for "
            "review.")
        escalation_note = ("Escalated to the specialist team for manual "
                           "follow-up; awaiting resolution.")

    reply = (
        f"{greeting}\n\n"
        f"Thank you for contacting us about your {category_readable} issue. "
        f"I understand how important this is and I want to help.\n\n"
        f"{policy_line}\n{steps}"
        f"{escalation_block}\n\n"
        f"We will keep you updated on the outcome. Thank you for your patience.\n\n"
        f"Best regards,\nSupport Team"
    )

    return {
        "ticket_id": ticket["ticket_id"],
        "subject": f"Re: {ticket.get('subject', 'your support request')}",
        "reply": reply,
        "cited_policy_ids": [p["policy_id"] for p in retrieved],
        "escalation_note": escalation_note,
    }


def mock_review(ticket, triage, draft, check):
    """Stage 3 reviewer (deterministic). Verdict drives the final status.

    approve  -> safe to send        (checks passed, no escalation)
    escalate -> needs a human       (should_escalate)
    revise   -> needs a human       (a deterministic check failed)
    """
    if not check.get("passed", False):
        verdict = "revise"
        issues = list(check.get("issues", []))
        notes = "Automated checks failed; a human should revise before sending."
    elif triage.get("should_escalate"):
        verdict = "escalate"
        issues = []
        notes = "Escalation required; routing to a specialist for human review."
    else:
        verdict = "approve"
        issues = []
        notes = "Draft cites only retrieved policies and passes all guardrails."
    return {
        "ticket_id": ticket["ticket_id"],
        "verdict": verdict,
        "issues": issues,
        "reviewer_notes": notes,
    }


class MockProvider(Provider):
    name = "mock"

    def __init__(self, cfg, log_path):
        super().__init__(cfg, log_path)
        self.model = cfg.get("mock_model", "mock-v1")

    def _generate(self, stage, prompt, context):
        if stage == "triage":
            return mock_triage(context["ticket"], context["allowed_categories"])
        if stage == "drafting":
            return mock_draft(context["ticket"], context["triage"],
                              context["retrieved"])
        if stage == "review":
            return mock_review(context["ticket"], context["triage"],
                               context["draft"], context["check"])
        raise ValueError(f"unknown stage {stage}")


# --------------------------------------------------------------------------- #
# NVIDIA NIM provider (OpenAI-compatible, stdlib urllib) — non-deterministic
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPTS = {
    "triage": ("You are a support triage assistant. Respond with ONLY a JSON "
               "object with keys ticket_id, category, priority, should_escalate "
               "(boolean), reason, missing_information (array of strings). "
               "No prose, no code fences."),
    "drafting": ("You are a support drafting assistant. Draft a reply that cites "
                 "ONLY the provided policies by id. Never promise, approve, or "
                 "guarantee a refund, expedite, or any outcome. If escalation is "
                 "needed, explain the next step without claiming resolution. "
                 "Respond with ONLY a JSON object with keys ticket_id, subject, "
                 "reply, cited_policy_ids (array), escalation_note (string or "
                 "null). No prose, no code fences."),
    "review": ("You are a support QA reviewer. Judge whether the drafted reply is "
               "safe to send. Verdict must be one of approve, revise, escalate. "
               "Use 'escalate' if the case needs a human specialist, 'revise' if "
               "the reply is unsafe/incorrect, otherwise 'approve'. Respond with "
               "ONLY a JSON object with keys ticket_id, verdict, issues (array of "
               "strings), reviewer_notes. No prose, no code fences."),
}


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class NvidiaProvider(Provider):
    name = "nvidia"

    def __init__(self, cfg, log_path):
        super().__init__(cfg, log_path)
        self.model = cfg.get("model", "z-ai/glm-5.2")
        self.base_url = cfg.get("base_url",
                                "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.temperature = cfg.get("temperature", 0)
        self.timeout = cfg.get("request_timeout", 60)
        self.api_key = os.environ.get("NVIDIA_API_KEY")

    def _generate(self, stage, prompt, context):
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPTS[stage]},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return _extract_json(content)
