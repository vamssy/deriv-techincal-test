"""I/O helpers, schema guards, and a tiny stdlib .env loader.

All JSON is written with ensure_ascii=False so non-ASCII content round-trips as
UTF-8. Input validation fails loud with an actionable message.
"""

import json
import os
from pathlib import Path


class InputError(Exception):
    """Raised on missing/malformed input so the run aborts before writing artifacts."""


# --------------------------------------------------------------------------- #
# JSON / JSONL
# --------------------------------------------------------------------------- #
def load_json(path):
    p = Path(path)
    if not p.exists():
        raise InputError(f"Input file not found: {p}. Provide it at the repo root "
                         f"or pass --tickets/--kb.")
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise InputError(f"Malformed JSON in {p}: {exc}") from exc


def dump_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def reset_jsonl(path):
    """Truncate the JSONL log so re-runs never accumulate stale records."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.open("w", encoding="utf-8").close()


def append_jsonl(path, record):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Schema guards — swap-safe (validate shape, never sample IDs/wording)
# --------------------------------------------------------------------------- #
def validate_tickets(data):
    if not isinstance(data, list):
        raise InputError("tickets.json must be a JSON array of ticket objects.")
    required = ("ticket_id", "subject", "message")
    seen = set()
    tickets = []
    for i, t in enumerate(data):
        if not isinstance(t, dict):
            raise InputError(f"Ticket at index {i} is not an object.")
        for field in required:
            if field not in t or t[field] in (None, ""):
                raise InputError(
                    f"Ticket at index {i} is missing required field '{field}'."
                )
        tid = t["ticket_id"]
        if tid in seen:
            raise InputError(f"Duplicate ticket_id '{tid}'.")
        seen.add(tid)
        tickets.append(t)
    return tickets


def validate_kb(data):
    if not isinstance(data, list):
        raise InputError("policy_kb.json must be a JSON array of policy objects.")
    required = ("policy_id", "title", "content")
    seen = set()
    policies = []
    for i, p in enumerate(data):
        if not isinstance(p, dict):
            raise InputError(f"Policy at index {i} is not an object.")
        for field in required:
            if field not in p or p[field] in (None, ""):
                raise InputError(
                    f"Policy at index {i} is missing required field '{field}'."
                )
        pid = p["policy_id"]
        if pid in seen:
            raise InputError(f"Duplicate policy_id '{pid}'.")
        seen.add(pid)
        # normalise tags to a list
        p = dict(p)
        p["tags"] = p.get("tags") or []
        if not isinstance(p["tags"], list):
            raise InputError(f"Policy '{pid}' has non-list tags.")
        policies.append(p)
    return policies


# --------------------------------------------------------------------------- #
# .env loader (stdlib) — convenience so the key can live in a gitignored file
# --------------------------------------------------------------------------- #
def load_dotenv(path=".env"):
    """Populate os.environ from a KEY=VALUE file WITHOUT overriding existing env."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
