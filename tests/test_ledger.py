import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunker import Chunk  # noqa: E402
from jev_client import EscalationAnswer, JevResult, LabelAnswer  # noqa: E402
from ledger import Ledger, Verification  # noqa: E402
from router import AUTO_POST_THRESHOLD, CONFIRM_THRESHOLD, Route, route  # noqa: E402


def chunk(pid="P1", day=0, hour=9, text="x", sid=None):
    ts = f"2026-09-{day + 1:02d}T{hour:02d}:00"
    return Chunk(sid or f"{pid}-d{day}-{hour}", pid, ts, "note", "case_manager", text, [text])


def jev(category="post_acute_placement", status="newly_raised", conf=0.95,
        readiness="not_stated", readiness_conf=0.9, owner="external_facility",
        relevant=True, error=None):
    lab = lambda l, c=conf: LabelAnswer(l, c, {l: c})
    return JevResult("x", relevant, 0.99 if relevant else 0.01, conf,
                     category=lab(category) if relevant else None,
                     status=lab(status) if relevant else None,
                     readiness=lab(readiness, readiness_conf) if relevant else None,
                     owner=lab(owner) if relevant else None,
                     escalation=EscalationAnswer("routine", 0.1, 0.9, {}) if relevant else None,
                     error=error)


# ---- router ----------------------------------------------------------------

@pytest.mark.parametrize("conf,expected", [
    (AUTO_POST_THRESHOLD, Route.AUTO_POST), (0.89, Route.NEEDS_CONFIRMATION),
    (CONFIRM_THRESHOLD, Route.NEEDS_CONFIRMATION), (0.49, Route.REVIEW_QUEUE)])
def test_router_thresholds(conf, expected):
    assert route(conf) is expected


def test_router_sends_errors_to_review():
    assert route(0.99, error=True) is Route.REVIEW_QUEUE


# ---- ledger ----------------------------------------------------------------

def test_new_barrier_is_auto_posted_with_timestamps():
    led = Ledger()
    led.apply(chunk(day=0), jev())
    b = led.patient("P1").barriers["post_acute_placement"]
    assert (b.status, b.verification, b.first_seen) == ("newly_raised", Verification.AUTO,
                                                        "2026-09-01T09:00")
    assert b.on_board


def test_second_mention_is_ongoing_and_age_is_python_date_math():
    led = Ledger()
    led.apply(chunk(day=0), jev())
    # Jev reads the second mention as "newly_raised" from wording; history says ongoing.
    led.apply(chunk(day=2, hour=21), jev(status="newly_raised"))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.status == "ongoing" and b.first_seen == "2026-09-01T09:00"
    assert b.last_seen == "2026-09-03T21:00"
    assert b.age_days(led.clock) == pytest.approx(2.5)


def test_resolution_takes_barrier_off_board_and_freezes_age():
    led = Ledger()
    led.apply(chunk(day=0), jev())
    led.apply(chunk(day=1), jev(status="resolved"))
    led.apply(chunk(day=5, text="later unrelated"), jev(category="none", status="not_applicable"))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert not b.is_open and not b.on_board
    assert b.age_days(led.clock) == pytest.approx(1.0)


def test_medium_confidence_needs_confirmation_then_confirm():
    led = Ledger()
    led.apply(chunk(), jev(conf=0.7))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.verification is Verification.NEEDS_CONFIRMATION and b.on_board
    led.confirm("P1", "post_acute_placement")
    assert b.verification is Verification.CONFIRMED


def test_dismissing_an_open_barrier_hides_it():
    led = Ledger()
    led.apply(chunk(), jev(conf=0.7))
    led.dismiss("P1", "post_acute_placement")
    assert not led.patient("P1").board_barriers


def test_dismissing_unconfirmed_resolution_reopens_barrier():
    led = Ledger()
    led.apply(chunk(day=0), jev())
    led.apply(chunk(day=1), jev(status="resolved", conf=0.6))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.resolution_unconfirmed and b.on_board          # still visible until verified
    led.dismiss("P1", "post_acute_placement")
    assert b.is_open and b.verification is Verification.AUTO and b.resolved_at is None


def test_low_confidence_goes_to_review_never_the_board():
    led = Ledger()
    led.apply(chunk(), jev(conf=0.3))
    assert not led.patient("P1").board_barriers
    [item] = led.review_queue()
    assert item.reason == "low_confidence"
    led.accept_review(item.id)
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.verification is Verification.CONFIRMED and not led.review_queue()


def test_api_error_goes_to_review():
    led = Ledger()
    led.apply(chunk(), jev(relevant=False, conf=0.0, error="403"))
    assert led.review_queue()[0].reason == "api_error"


def test_irrelevant_and_no_barrier_chunks_create_no_barriers():
    led = Ledger()
    led.apply(chunk(sid="a"), jev(relevant=False, conf=0.98))
    led.apply(chunk(sid="b"), jev(category="none", status="resolved", conf=0.95))
    p = led.patient("P1")
    assert p is not None and p.barriers == {}


def test_reopen_after_resolution_restarts_age():
    led = Ledger()
    led.apply(chunk(day=0), jev())
    led.apply(chunk(day=1), jev(status="resolved"))
    led.apply(chunk(day=3), jev(status="ongoing"))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.status == "newly_raised" and b.first_seen == "2026-09-04T09:00"


def test_readiness_needs_minimum_confidence():
    led = Ledger()
    led.apply(chunk(sid="a"), jev(category="none", readiness="ready", readiness_conf=0.5))
    assert led.patient("P1").readiness is None
    led.apply(chunk(sid="b"), jev(category="none", readiness="ready", readiness_conf=0.95))
    assert led.patient("P1").medically_ready


def test_board_puts_ready_but_blocked_first_then_oldest():
    led = Ledger()
    none = dict(category="none", status="not_applicable")
    led.apply(chunk("A", day=0), jev())                                   # blocked, not ready
    led.apply(chunk("A", day=4, sid="a2"), jev(**none))                   #   4d old
    led.apply(chunk("B", day=2), jev(readiness="ready"))                  # ready + blocked
    led.apply(chunk("B", day=3, sid="b2"), jev(**none))                   #   1d old
    led.apply(chunk("C", day=1), jev(readiness="ready"))                  # ready + blocked
    led.apply(chunk("C", day=3, sid="c2"), jev(**none))                   #   2d old
    led.apply(chunk("D", day=3), jev(readiness="ready", **none))          # ready, no barrier
    assert [p.patient_id for p in led.board()] == ["C", "B", "A", "D"]


def test_barrier_first_heard_as_resolved_is_not_a_board_entry():
    led = Ledger()
    led.apply(chunk(), jev(status="resolved", conf=0.7))
    b = led.patient("P1").barriers["post_acute_placement"]
    assert not b.is_open and not b.resolution_unconfirmed and not b.on_board


def test_age_is_measured_to_the_patients_own_latest_note():
    led = Ledger()
    led.apply(chunk("A", day=0), jev())
    led.apply(chunk("A", day=2), jev(category="none", status="not_applicable"))
    led.apply(chunk("B", day=9), jev())                  # another patient, much later
    a = led.patient("A")
    assert a.barriers["post_acute_placement"].age_days(a.as_of) == pytest.approx(2.0)


def test_badge_confidence_is_the_reading_that_earned_the_verification():
    led = Ledger()
    led.apply(chunk(day=0), jev(conf=0.95))
    led.apply(chunk(day=1), jev(conf=0.74))               # weaker later mention
    b = led.patient("P1").barriers["post_acute_placement"]
    assert b.verification is Verification.AUTO
    assert b.posted_confidence == pytest.approx(0.95) and b.confidence == pytest.approx(0.74)
