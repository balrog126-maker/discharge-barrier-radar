"""
Stage 4 — per-patient barrier ledger.

The ledger turns a stream of (chunk, JevResult) pairs into the current state
of every patient: which barriers are open, since when, who owns them, how
sure we are, and whether a person has verified them.

Division of labour:
  * Jev says what a piece of text is about (category, resolved or not, owner...).
  * This module decides everything that needs memory or arithmetic, in plain
    Python: whether a barrier is new or ongoing (from history, not from
    wording), first/last-seen timestamps, barrier age, board sort order.

Storage is behind `LedgerStore`; `InMemoryLedgerStore` is the POC
implementation and a database-backed store can replace it without touching
the ledger logic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol

from dateutil.parser import isoparse

from jev_client import JevResult
from questions import (
    NO_BARRIER_CATEGORY,
    READINESS_MIN_CONFIDENCE,
    RESOLVED_STATUS,
)
from router import Route, route


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

class Verification(str, Enum):
    AUTO = "auto"                                # posted automatically (high confidence)
    NEEDS_CONFIRMATION = "needs_confirmation"    # on the board, unverified
    CONFIRMED = "confirmed"                      # a person confirmed it
    DISMISSED = "dismissed"                      # a person dismissed it (hidden)


_ROUTE_TO_VERIFICATION = {Route.AUTO_POST: Verification.AUTO,
                          Route.NEEDS_CONFIRMATION: Verification.NEEDS_CONFIRMATION}
# When a later mention of an open barrier arrives, keep the strongest of the two.
_STRENGTH = {Verification.DISMISSED: 0, Verification.NEEDS_CONFIRMATION: 1,
             Verification.AUTO: 2, Verification.CONFIRMED: 3}


@dataclass
class Evidence:
    """The text that justified an update — what the UI shows when a card is opened."""
    snippet_id: str
    timestamp: str
    source: str
    author_role: str
    text: str                              # exactly what Jev classified
    ignored_text: list[str]                # copy-forward sentences the chunker dropped
    status_said: str                       # Jev's status reading of this text
    confidence: float
    route: Route


@dataclass
class BarrierEntry:
    patient_id: str
    category: str
    status: str                            # newly_raised | ongoing | resolved
    owner: str
    escalation: str
    first_seen: str
    last_seen: str
    confidence: float                      # of the latest reading
    verification: Verification
    resolved_at: str | None = None
    posted_confidence: float = 0.0         # the reading that earned the current verification
    evidence: list[Evidence] = field(default_factory=list)
    # Kept so an unconfirmed resolution can be dismissed (undone).
    _before_resolution: tuple[str, Verification] | None = None

    @property
    def is_open(self) -> bool:
        return self.status != RESOLVED_STATUS

    @property
    def resolution_unconfirmed(self) -> bool:
        """Resolved an open barrier, but nobody has verified the resolution yet. (A barrier
        first heard of as already resolved has nothing to un-resolve, so it isn't this.)"""
        return (not self.is_open and self._before_resolution is not None
                and self.verification is Verification.NEEDS_CONFIRMATION)

    @property
    def on_board(self) -> bool:
        """Shown as a live board entry: open and not dismissed, or a resolution
        still awaiting confirmation (so nobody loses sight of it)."""
        if self.verification is Verification.DISMISSED:
            return False
        return self.is_open or self.resolution_unconfirmed

    def age(self, now: str) -> timedelta:
        end = self.resolved_at if not self.is_open else now
        return isoparse(end) - isoparse(self.first_seen)

    def age_days(self, now: str) -> float:
        return self.age(now).total_seconds() / 86400

    def since_last_update(self, now: str) -> timedelta:
        return isoparse(now) - isoparse(self.last_seen)


@dataclass
class Readiness:
    label: str                             # ready | not_ready
    as_of: str
    confidence: float
    snippet_id: str
    text: str


@dataclass
class PatientRecord:
    patient_id: str
    barriers: dict[str, BarrierEntry] = field(default_factory=dict)
    readiness: Readiness | None = None
    last_activity: str | None = None

    @property
    def board_barriers(self) -> list[BarrierEntry]:
        return [b for b in self.barriers.values() if b.on_board]

    @property
    def open_barriers(self) -> list[BarrierEntry]:
        return [b for b in self.board_barriers if b.is_open]

    @property
    def medically_ready(self) -> bool:
        return self.readiness is not None and self.readiness.label == "ready"

    @property
    def ready_but_blocked(self) -> bool:
        return self.medically_ready and bool(self.open_barriers)

    @property
    def as_of(self) -> str | None:
        """Reference time for this patient's barrier ages: their latest note or message.
        (A global clock would keep ageing barriers of patients whose notes have stopped —
        in the replayed corpus, patients admitted weeks apart.)"""
        return self.last_activity


@dataclass
class ReviewItem:
    id: int
    patient_id: str
    snippet_id: str
    timestamp: str
    text: str
    reason: str                            # low_confidence | api_error
    confidence: float
    result: JevResult
    chunk: object
    state: str = "open"                    # open | accepted | rejected


@dataclass(frozen=True)
class LedgerUpdate:
    """One thing that changed — used for the UI's activity feed and for eval."""
    patient_id: str
    snippet_id: str
    action: str        # raised | continued | resolved | reopened | readiness | review | no_barrier
    category: str | None
    route: Route | None
    confidence: float


# --------------------------------------------------------------------------- #
# Storage interface
# --------------------------------------------------------------------------- #

class LedgerStore(Protocol):
    def get_patient(self, patient_id: str) -> PatientRecord | None: ...
    def save_patient(self, record: PatientRecord) -> None: ...
    def all_patients(self) -> list[PatientRecord]: ...
    def add_review(self, item: ReviewItem) -> None: ...
    def get_review(self, item_id: int) -> ReviewItem | None: ...
    def all_reviews(self) -> list[ReviewItem]: ...


class InMemoryLedgerStore:
    def __init__(self) -> None:
        self._patients: dict[str, PatientRecord] = {}
        self._reviews: dict[int, ReviewItem] = {}

    def get_patient(self, patient_id: str) -> PatientRecord | None:
        return self._patients.get(patient_id)

    def save_patient(self, record: PatientRecord) -> None:
        self._patients[record.patient_id] = record

    def all_patients(self) -> list[PatientRecord]:
        return list(self._patients.values())

    def add_review(self, item: ReviewItem) -> None:
        self._reviews[item.id] = item

    def get_review(self, item_id: int) -> ReviewItem | None:
        return self._reviews.get(item_id)

    def all_reviews(self) -> list[ReviewItem]:
        return list(self._reviews.values())


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

class Ledger:
    def __init__(self, store: LedgerStore | None = None):
        self.store = store or InMemoryLedgerStore()
        self.clock: str | None = None          # latest timestamp applied (simulation "now")
        self._review_ids = itertools.count(1)

    # -- ingest ------------------------------------------------------------ #

    def apply(self, chunk, result: JevResult) -> list[LedgerUpdate]:
        """Apply one classified chunk. `chunk` is a chunker.Chunk (or anything with
        snippet_id, patient_id, timestamp, source, author_role, text, dropped_sentences)."""
        self._tick(chunk.timestamp)
        patient = self._patient(chunk.patient_id)
        patient.last_activity = chunk.timestamp
        self.store.save_patient(patient)      # every patient seen appears on the board

        if result.error:
            return [self._to_review(chunk, result, "api_error")]
        if not result.is_discharge_related or not result.classified:
            return []

        updates: list[LedgerUpdate] = []
        r = route(result.confidence)

        # Medical readiness is independent of whether this text names a barrier.
        rd = result.readiness
        if rd.label in ("ready", "not_ready") and rd.confidence >= READINESS_MIN_CONFIDENCE:
            patient.readiness = Readiness(rd.label, chunk.timestamp, rd.confidence,
                                          chunk.snippet_id, chunk.text)
            updates.append(LedgerUpdate(patient.patient_id, chunk.snippet_id, "readiness",
                                        None, None, rd.confidence))

        if result.category.label == NO_BARRIER_CATEGORY:
            updates.append(LedgerUpdate(patient.patient_id, chunk.snippet_id, "no_barrier",
                                        None, r, result.confidence))
        elif r is Route.REVIEW_QUEUE:
            updates.append(self._to_review(chunk, result, "low_confidence"))
        else:
            updates.append(self._apply_barrier(patient, chunk, result, r,
                                               _ROUTE_TO_VERIFICATION[r]))
        self.store.save_patient(patient)
        return updates

    def _apply_barrier(self, patient: PatientRecord, chunk, result: JevResult, r: Route,
                       verification: Verification) -> LedgerUpdate:
        cat, ts = result.category.label, chunk.timestamp
        says_resolved = result.status.label == RESOLVED_STATUS
        owner = result.owner.label
        entry = patient.barriers.get(cat)

        if entry is None:
            entry = BarrierEntry(patient.patient_id, cat,
                                 RESOLVED_STATUS if says_resolved else "newly_raised",
                                 owner, result.escalation.label, ts, ts,
                                 result.confidence, verification,
                                 resolved_at=ts if says_resolved else None,
                                 posted_confidence=result.confidence)
            patient.barriers[cat] = entry
            action = "resolved" if says_resolved else "raised"
        elif says_resolved and entry.is_open:
            entry._before_resolution = (entry.status, entry.verification)
            entry.status, entry.resolved_at = RESOLVED_STATUS, ts
            entry.verification, entry.posted_confidence = verification, result.confidence
            action = "resolved"
        elif says_resolved:
            action = "continued"                       # another note confirming it's done
        elif not entry.is_open:
            # Raised again after being resolved: a new episode, so age restarts.
            entry.status, entry.first_seen, entry.resolved_at = "newly_raised", ts, None
            entry.verification, entry._before_resolution = verification, None
            entry.posted_confidence = result.confidence
            action = "reopened"
        else:
            entry.status = "ongoing"
            if entry.verification is Verification.DISMISSED or \
                    _STRENGTH[verification] > _STRENGTH[entry.verification]:
                entry.verification, entry.posted_confidence = verification, result.confidence
            action = "continued"

        entry.last_seen = ts
        entry.confidence = result.confidence
        if owner != "not_stated":
            entry.owner = owner
        entry.escalation = result.escalation.label
        entry.evidence.append(Evidence(chunk.snippet_id, ts, chunk.source, chunk.author_role,
                                       chunk.text, list(getattr(chunk, "dropped_sentences", [])),
                                       result.status.label, result.confidence, r))
        return LedgerUpdate(patient.patient_id, chunk.snippet_id, action, cat, r,
                            result.confidence)

    def _to_review(self, chunk, result: JevResult, reason: str) -> LedgerUpdate:
        item = ReviewItem(next(self._review_ids), chunk.patient_id, chunk.snippet_id,
                          chunk.timestamp, chunk.text, reason, result.confidence, result, chunk)
        self.store.add_review(item)
        cat = result.category.label if result.category else None
        return LedgerUpdate(chunk.patient_id, chunk.snippet_id, "review", cat,
                            Route.REVIEW_QUEUE, result.confidence)

    # -- human actions ----------------------------------------------------- #

    def confirm(self, patient_id: str, category: str) -> None:
        entry = self._entry(patient_id, category)
        entry.verification = Verification.CONFIRMED
        entry._before_resolution = None

    def dismiss(self, patient_id: str, category: str) -> None:
        """Dismissing an unconfirmed resolution means 'not actually resolved' ->
        the barrier reopens as it was. Dismissing an open barrier hides it."""
        entry = self._entry(patient_id, category)
        if entry.resolution_unconfirmed and entry._before_resolution:
            entry.status, entry.verification = entry._before_resolution
            entry.resolved_at, entry._before_resolution = None, None
        else:
            entry.verification = Verification.DISMISSED

    def accept_review(self, item_id: int) -> LedgerUpdate | None:
        """A reviewer agrees with Jev's reading: apply it as confirmed."""
        item = self.store.get_review(item_id)
        if item is None or item.state != "open" or not item.result.classified:
            return None
        item.state = "accepted"
        patient = self._patient(item.patient_id)
        update = self._apply_barrier(patient, item.chunk, item.result, Route.REVIEW_QUEUE,
                                     Verification.CONFIRMED)
        self.store.save_patient(patient)
        return update

    def reject_review(self, item_id: int) -> None:
        item = self.store.get_review(item_id)
        if item is not None:
            item.state = "rejected"

    # -- queries ----------------------------------------------------------- #

    def board(self) -> list[PatientRecord]:
        """Patients in board order: medically ready but blocked first (longest-blocked
        first), then blocked but not yet ready, then everyone else."""
        def key(p: PatientRecord):
            now = p.as_of or datetime.now().isoformat(timespec="minutes")
            oldest = max((b.age_days(now) for b in p.open_barriers), default=0.0)
            tier = 0 if p.ready_but_blocked else 1 if p.open_barriers else 2
            return (tier, -oldest, p.patient_id)

        return sorted(self.store.all_patients(), key=key)

    def review_queue(self) -> list[ReviewItem]:
        return [r for r in self.store.all_reviews() if r.state == "open"]

    def patient(self, patient_id: str) -> PatientRecord | None:
        return self.store.get_patient(patient_id)

    # -- helpers ----------------------------------------------------------- #

    def _tick(self, ts: str) -> None:
        if self.clock is None or isoparse(ts) > isoparse(self.clock):
            self.clock = ts

    def _patient(self, patient_id: str) -> PatientRecord:
        return self.store.get_patient(patient_id) or PatientRecord(patient_id)

    def _entry(self, patient_id: str, category: str) -> BarrierEntry:
        patient = self.store.get_patient(patient_id)
        if patient is None or category not in patient.barriers:
            raise KeyError(f"No {category} barrier for {patient_id}")
        return patient.barriers[category]
