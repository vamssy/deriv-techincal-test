"""Unit tests for the deterministic core of the stage-gated support pipeline.

stdlib-only (unittest, no pytest). All fixtures are built in memory: these tests
never touch out/ nor the repo's sample JSON, so they exercise the pure functions
in isolation and stay reproducible.

Run from the repo root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

# Make the `src` package importable when this file is run directly as well as
# via `python -m unittest discover` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.state import Stage, StateMachine, IllegalTransition
from src.retrieval import build_index, retrieve_for_ticket
from src.checks import run_checks
from src.drafting import sanitize_draft
from src.triage import sanitize_triage
from src.finalize import run_finalize
from src.provider import mock_draft, prompt_hash


# --------------------------------------------------------------------------- #
# Shared in-memory fixtures
# --------------------------------------------------------------------------- #
def make_tickets():
    return [
        {
            "ticket_id": "T1",
            "subject": "Refund for duplicate charge",
            "message": ("I was charged twice for my subscription and would "
                        "like a refund for the extra charge."),
            "customer_name": "Sam",
        },
        {
            "ticket_id": "T2",
            "subject": "This is unacceptable",
            "message": ("Your service is terrible and I want to speak to a "
                        "manager, this is unacceptable."),
            "customer_name": "Alex",
        },
    ]


def make_policies():
    return [
        {
            "policy_id": "P1",
            "title": "Refund Policy",
            "content": ("Customers may request a refund for a duplicate charge "
                        "on a subscription within 30 days."),
            "tags": ["billing", "refund"],
        },
        {
            "policy_id": "P2",
            "title": "Technical Support",
            "content": ("For errors, crashes, or bugs, restart the app and "
                        "clear the cache before contacting support."),
            "tags": ["technical"],
        },
        {
            "policy_id": "P3",
            "title": "Communication Tone",
            "content": ("Always communicate with empathy and professionalism, "
                        "especially with upset customers."),
            "tags": ["tone", "safety"],
        },
        {
            "policy_id": "P4",
            "title": "Account Access",
            "content": ("If a customer is locked out, verify identity before "
                        "resetting the password."),
            "tags": ["account"],
        },
    ]


def make_cfg():
    return {
        "top_k": 3,
        "min_reply_len": 40,
        "banned_phrases": ["full refund", "we guarantee",
                           "refund has been processed"],
        "allowed_categories": ["billing", "technical", "account", "shipping",
                               "product", "complaint", "general"],
    }


# --------------------------------------------------------------------------- #
# 1. State machine
# --------------------------------------------------------------------------- #
class TestStateMachine(unittest.TestCase):
    def test_legal_sequential_advance(self):
        sm = StateMachine()
        self.assertEqual(sm.stage, Stage.INIT)
        ordered = [
            Stage.INPUTS_LOADED, Stage.TICKETS_PARSED, Stage.KB_INDEXED,
            Stage.TICKET_TRIAGED, Stage.EVIDENCE_RETRIEVED,
            Stage.RESPONSE_DRAFTED, Stage.RESPONSE_CHECKED,
            Stage.RESPONSE_REVIEWED, Stage.RESPONSE_FINALISED,
        ]
        for target in ordered:
            sm.advance(target)
            self.assertEqual(sm.stage, target)
        self.assertEqual(sm.stage, Stage.RESPONSE_FINALISED)
        self.assertEqual(sm.history[0], Stage.INIT)
        self.assertEqual(sm.history[-1], Stage.RESPONSE_FINALISED)

    def test_skipping_a_stage_raises(self):
        sm = StateMachine()
        # INIT -> TICKETS_PARSED skips INPUTS_LOADED.
        with self.assertRaises(IllegalTransition):
            sm.advance(Stage.TICKETS_PARSED)
        # Stage is unchanged after the failed transition.
        self.assertEqual(sm.stage, Stage.INIT)

    def test_backwards_advance_raises(self):
        sm = StateMachine()
        sm.advance(Stage.INPUTS_LOADED)
        with self.assertRaises(IllegalTransition):
            sm.advance(Stage.INIT)

    def test_require_below_minimum_raises(self):
        sm = StateMachine()  # at INIT
        with self.assertRaises(IllegalTransition):
            sm.require(Stage.KB_INDEXED)

    def test_require_at_or_above_minimum_passes(self):
        sm = StateMachine()
        sm.advance(Stage.INPUTS_LOADED)
        sm.advance(Stage.TICKETS_PARSED)
        sm.advance(Stage.KB_INDEXED)
        # At the minimum: allowed.
        self.assertIsNone(sm.require(Stage.KB_INDEXED))
        # Above the minimum: allowed.
        self.assertIsNone(sm.require(Stage.INPUTS_LOADED))


# --------------------------------------------------------------------------- #
# 2. Retrieval
# --------------------------------------------------------------------------- #
class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.tickets = make_tickets()
        self.policies = make_policies()
        self.cfg = make_cfg()
        self.index = build_index(self.policies)
        self.policy_ids = {p["policy_id"] for p in self.policies}

    def test_returns_at_most_top_k_all_in_kb(self):
        res = retrieve_for_ticket(self.tickets[0], self.policies,
                                  self.index, self.cfg)
        ids = res["retrieved_policy_ids"]
        self.assertLessEqual(len(ids), self.cfg["top_k"])
        self.assertTrue(set(ids) <= self.policy_ids)
        self.assertEqual(res["ticket_id"], "T1")

    def test_deterministic_across_repeated_calls(self):
        r1 = retrieve_for_ticket(self.tickets[0], self.policies,
                                 self.index, self.cfg)
        r2 = retrieve_for_ticket(self.tickets[0], self.policies,
                                 self.index, self.cfg)
        self.assertEqual(r1, r2)

    def test_zero_overlap_ticket_still_returns_top_k(self):
        # Neutral gibberish: no keyword/tag overlap and no complaint hints.
        ticket = {"ticket_id": "TZ", "subject": "zzxqvv wgblmp",
                  "message": "qwrtzp jklvbn mnbvcx plkoij"}
        res = retrieve_for_ticket(ticket, self.policies, self.index, self.cfg)
        ids = res["retrieved_policy_ids"]
        expected = min(self.cfg["top_k"], len(self.policies))
        self.assertEqual(len(ids), expected)
        self.assertTrue(set(ids) <= self.policy_ids)

    def test_top_k_clamps_to_policy_count(self):
        cfg = dict(self.cfg)
        cfg["top_k"] = 10  # more than the 4 policies available
        res = retrieve_for_ticket(self.tickets[0], self.policies,
                                  self.index, cfg)
        ids = res["retrieved_policy_ids"]
        self.assertEqual(len(ids), len(self.policies))
        self.assertTrue(set(ids) <= self.policy_ids)


# --------------------------------------------------------------------------- #
# 3. Checks
# --------------------------------------------------------------------------- #
class TestChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.tickets = make_tickets()
        self.t1 = self.tickets[0]
        policies = make_policies()
        self.policy_by_id = {p["policy_id"]: p for p in policies}
        index = build_index(policies)
        self.retr = retrieve_for_ticket(self.t1, policies, index, self.cfg)
        retrieved_policies = [self.policy_by_id[pid]
                              for pid in self.retr["retrieved_policy_ids"]]
        # Non-escalating triage so the escalation-communication check is inert.
        self.triage = {
            "ticket_id": "T1", "category": "billing", "priority": "medium",
            "should_escalate": False, "reason": "n/a", "missing_information": [],
        }
        self.draft = mock_draft(self.t1, self.triage, retrieved_policies)

    def _run(self, draft):
        return run_checks({"T1": draft}, {"T1": self.retr},
                          {"T1": self.triage}, self.cfg, [self.t1])[0]

    def test_clean_draft_passes(self):
        rec = self._run(self.draft)
        self.assertTrue(rec["passed"], msg=f"unexpected issues: {rec['issues']}")
        self.assertEqual(rec["issues"], [])

    def test_banned_phrase_fails(self):
        bad = dict(self.draft)
        bad["reply"] = self.draft["reply"] + " full refund"
        rec = self._run(bad)
        self.assertFalse(rec["passed"])
        self.assertTrue(any(i.startswith("banned_phrase") for i in rec["issues"]))

    def test_out_of_set_citation_fails(self):
        bad = dict(self.draft)
        bad["cited_policy_ids"] = list(self.draft["cited_policy_ids"]) + ["P_NOPE"]
        rec = self._run(bad)
        self.assertFalse(rec["passed"])
        self.assertTrue(any(i.startswith("citation_out_of_set")
                            for i in rec["issues"]))

    def test_too_short_reply_fails(self):
        bad = dict(self.draft)
        bad["reply"] = "too short"
        rec = self._run(bad)
        self.assertFalse(rec["passed"])
        self.assertIn("reply_too_short", rec["issues"])


# --------------------------------------------------------------------------- #
# 4. sanitize_draft
# --------------------------------------------------------------------------- #
class TestSanitizeDraft(unittest.TestCase):
    def setUp(self):
        self.ticket = {"ticket_id": "T1", "subject": "Refund"}

    def test_out_of_set_citation_dropped(self):
        result = {"reply": "A sufficiently long reply body for the customer.",
                  "cited_policy_ids": ["P1", "P_BAD"]}
        out = sanitize_draft(result, self.ticket, ["P1", "P2"])
        self.assertEqual(out["cited_policy_ids"], ["P1"])
        self.assertTrue(set(out["cited_policy_ids"]) <= {"P1", "P2"})

    def test_empty_cite_forced_to_retrieved_set(self):
        result = {"reply": "Reply text.", "cited_policy_ids": []}
        out = sanitize_draft(result, self.ticket, ["P1", "P2"])
        self.assertGreaterEqual(len(out["cited_policy_ids"]), 1)
        self.assertTrue(set(out["cited_policy_ids"]) <= {"P1", "P2"})

    def test_always_subset_of_retrieved(self):
        result = {"reply": "Reply text.",
                  "cited_policy_ids": ["P2", "PX", "PY"]}
        out = sanitize_draft(result, self.ticket, ["P1", "P2", "P3"])
        self.assertTrue(set(out["cited_policy_ids"]) <= {"P1", "P2", "P3"})

    def test_empty_retrieved_leaves_no_citations(self):
        result = {"reply": "Reply text.", "cited_policy_ids": ["P1"]}
        out = sanitize_draft(result, self.ticket, [])
        self.assertEqual(out["cited_policy_ids"], [])


# --------------------------------------------------------------------------- #
# 5. sanitize_triage
# --------------------------------------------------------------------------- #
class TestSanitizeTriage(unittest.TestCase):
    def setUp(self):
        self.ticket = {"ticket_id": "T1"}
        self.allowed = make_cfg()["allowed_categories"]

    def test_unknown_category_coerced_to_general(self):
        out = sanitize_triage({"category": "wibble", "priority": "medium"},
                              self.ticket, self.allowed)
        self.assertEqual(out["category"], "general")

    def test_unknown_priority_coerced_to_medium(self):
        out = sanitize_triage({"category": "billing", "priority": "superurgent"},
                              self.ticket, self.allowed)
        self.assertEqual(out["priority"], "medium")

    def test_known_values_preserved(self):
        out = sanitize_triage({"category": "billing", "priority": "high"},
                              self.ticket, self.allowed)
        self.assertEqual(out["category"], "billing")
        self.assertEqual(out["priority"], "high")


# --------------------------------------------------------------------------- #
# 6. finalize
# --------------------------------------------------------------------------- #
class TestFinalize(unittest.TestCase):
    def setUp(self):
        self.tickets = [{"ticket_id": "T1", "subject": "s", "message": "m"}]
        self.triage_by = {"T1": {"ticket_id": "T1", "category": "billing",
                                 "priority": "medium", "missing_information": []}}
        self.drafts_by = {"T1": {"ticket_id": "T1", "reply": "some reply body",
                                 "cited_policy_ids": ["P1"]}}

    def _finalize(self, passed, verdict, issues=None):
        checks_by = {"T1": {"ticket_id": "T1", "passed": passed,
                            "issues": issues or []}}
        review_by = {"T1": {"ticket_id": "T1", "verdict": verdict}}
        return run_finalize(self.tickets, self.triage_by, self.drafts_by,
                            checks_by, review_by)[0]

    def test_failed_check_needs_human_review(self):
        rec = self._finalize(passed=False, verdict="approve",
                             issues=["reply_too_short"])
        self.assertEqual(rec["final_status"], "needs_human_review")

    def test_passed_check_and_approve_is_ready(self):
        rec = self._finalize(passed=True, verdict="approve")
        self.assertEqual(rec["final_status"], "ready")

    def test_passed_check_and_escalate_needs_human_review(self):
        rec = self._finalize(passed=True, verdict="escalate")
        self.assertEqual(rec["final_status"], "needs_human_review")


# --------------------------------------------------------------------------- #
# 7. prompt_hash
# --------------------------------------------------------------------------- #
class TestPromptHash(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(prompt_hash("a support prompt"),
                         prompt_hash("a support prompt"))

    def test_differs_for_different_prompts(self):
        self.assertNotEqual(prompt_hash("prompt one"), prompt_hash("prompt two"))


if __name__ == "__main__":
    unittest.main()
