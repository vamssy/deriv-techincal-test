"""Explicit stage state machine.

Enforces the pipeline order and refuses to produce finals before RESPONSE_CHECKED.
Illegal transitions raise, so a bug in the orchestrator fails loud instead of
silently emitting artifacts out of order.
"""

from enum import IntEnum


class Stage(IntEnum):
    INIT = 0
    INPUTS_LOADED = 1
    TICKETS_PARSED = 2
    KB_INDEXED = 3
    TICKET_TRIAGED = 4
    EVIDENCE_RETRIEVED = 5
    RESPONSE_DRAFTED = 6
    RESPONSE_CHECKED = 7
    RESPONSE_REVIEWED = 8
    RESPONSE_FINALISED = 9


class IllegalTransition(Exception):
    """Raised when the pipeline attempts an out-of-order stage transition."""


class StateMachine:
    def __init__(self):
        self.stage = Stage.INIT
        self.history = [Stage.INIT]

    def advance(self, target: Stage) -> None:
        """Move to the immediate successor stage; anything else is illegal."""
        if int(target) != int(self.stage) + 1:
            raise IllegalTransition(
                f"illegal transition {self.stage.name} -> {target.name}"
            )
        self.stage = target
        self.history.append(target)

    def require(self, minimum: Stage) -> None:
        """Guard: assert we have at least reached `minimum` before proceeding."""
        if int(self.stage) < int(minimum):
            raise IllegalTransition(
                f"stage {self.stage.name} is before required {minimum.name}"
            )
