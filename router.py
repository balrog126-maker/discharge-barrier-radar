"""
Stage 4 — confidence router.

Decides where a classification may appear, from its confidence alone:

    confidence >= AUTO_POST_THRESHOLD   -> AUTO_POST           (live on the board)
    confidence >= CONFIRM_THRESHOLD     -> NEEDS_CONFIRMATION  (on the board, visibly unverified)
    otherwise, or an API error          -> REVIEW_QUEUE        (never a live board entry)

The threshold values are defined once, in questions.py, alongside the
questions they calibrate; they are bound here as named module constants so
routing logic never contains a bare number.
"""

from __future__ import annotations

from enum import Enum

from questions import AUTO_POST_THRESHOLD, CONFIRM_THRESHOLD

AUTO_POST_THRESHOLD = AUTO_POST_THRESHOLD      # 0.9 — see questions.py
CONFIRM_THRESHOLD = CONFIRM_THRESHOLD          # 0.5 — see questions.py


class Route(str, Enum):
    AUTO_POST = "auto_post"
    NEEDS_CONFIRMATION = "needs_confirmation"
    REVIEW_QUEUE = "review_queue"


def route(confidence: float, *, error: bool = False) -> Route:
    if error:
        return Route.REVIEW_QUEUE
    if confidence >= AUTO_POST_THRESHOLD:
        return Route.AUTO_POST
    if confidence >= CONFIRM_THRESHOLD:
        return Route.NEEDS_CONFIRMATION
    return Route.REVIEW_QUEUE
