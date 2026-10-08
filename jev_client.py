"""
Stage 3 — the one and only place the TypeSafe SDK is called.

    with JevClient() as jev:
        result = jev.classify("Auth still pending with Summit Health Plan.")
        result.category.label, result.confidence, jev.meter.summary()

Flow per chunk:
  1. relevance gate (questions.RELEVANCE_QUESTIONS). Not discharge-related ->
     stop, return a result with no classification.
  2. otherwise one batched call with every question in
     questions.CLASSIFICATION_QUESTIONS.

Every call's latency, token usage and cost is recorded on `jev.meter` (for the
live cost counter) and appended to logs/jev_calls.jsonl.

CLI:
  python jev_client.py "text to classify"
  python jev_client.py --samples       # classify a few hard cases from the corpus
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx2
from typesafe_sdk import TypeSafeClient, TypeSafeError

from questions import (
    CLASSIFICATION_QUESTIONS,
    CONFIDENCE_DRIVERS,
    ESCALATION_LEVELS,
    RELEVANCE_QUESTIONS,
    RELEVANCE_THRESHOLD,
    RESOLVED_STATUS,
    STATUS_IGNORED_FOR_CATEGORIES,
)

# Jev via OpenRouter's System One API (the SDK appends /v1/systemone).
BASE_URL = "https://openrouter.ai/api"
API_KEY_ENV = "OPENROUTER_API_KEY"
# Pinned so eval numbers are reproducible; "jev-latest" tracks the newest release.
MODEL = "jev-1.13"
# OpenRouter reports the billed amount per call in usage.cost, and that is what
# we meter. This list price (input only; output is free) is the fallback when
# a response carries no cost field.
PRICE_PER_MILLION_INPUT_TOKENS = 0.042

ROOT = Path(__file__).parent
LOG_PATH = ROOT / "logs" / "jev_calls.jsonl"

log = logging.getLogger("radar.jev")
# pipeline.run() classifies from several threads; appends to the call log must not interleave.
_LOG_WRITE_LOCK = threading.Lock()


def _load_api_key() -> str:
    """Shell environment first, then the project's .env file."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == API_KEY_ENV:
                key = value.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set (checked the environment and .env).")
    return key


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LabelAnswer:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EscalationAnswer:
    label: str                  # most probable level name
    score: float                # Jev's expected score, 0 (routine) .. 2 (stuck)
    confidence: float
    probabilities: dict[str, float]      # keyed by level name


@dataclass
class JevResult:
    text: str
    is_discharge_related: bool
    relevance_probability: float
    confidence: float                    # min over questions.CONFIDENCE_DRIVERS (see there)
    category: LabelAnswer | None = None
    status: LabelAnswer | None = None
    readiness: LabelAnswer | None = None
    owner: LabelAnswer | None = None
    escalation: EscalationAnswer | None = None
    latency_ms: float = 0.0              # summed over this chunk's calls
    input_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def classified(self) -> bool:
        return self.category is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CallRecord:
    timestamp: str
    purpose: str                 # "gate" | "classify"
    ref: str | None              # caller's id for the chunk (e.g. snippet_id)
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ok: bool
    error: str | None = None


# --------------------------------------------------------------------------- #
# Usage meter — feeds the live cost counter
# --------------------------------------------------------------------------- #

@dataclass
class UsageMeter:
    records: list[CallRecord] = field(default_factory=list)
    chunks: int = 0                      # chunks classified (one chunk = 1 or 2 calls)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)

    def count_chunk(self) -> None:
        with self._lock:
            self.chunks += 1

    def summary(self) -> dict:
        with self._lock:
            recs = list(self.records)
            chunks = self.chunks
        ok = [r for r in recs if r.ok]
        return {
            "chunks": chunks,
            "calls": len(recs),
            "errors": len(recs) - len(ok),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "cost_usd": sum(r.cost_usd for r in recs),
            "avg_latency_ms": (sum(r.latency_ms for r in ok) / len(ok)) if ok else 0.0,
        }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class JevClient:
    def __init__(self, api_key: str | None = None, *, meter: UsageMeter | None = None,
                 log_path: Path | None = LOG_PATH,
                 transport: httpx2.BaseTransport | None = None):
        """`transport` exists for tests (httpx2.MockTransport); leave it None in the app."""
        self._client = TypeSafeClient(api_key=api_key or _load_api_key(),
                                      base_url=BASE_URL, model=MODEL,
                                      transport=transport)
        self.meter = meter or UsageMeter()
        self._log_path = log_path
        if log_path:
            log_path.parent.mkdir(exist_ok=True)

    # -- plumbing ---------------------------------------------------------- #

    def _call(self, purpose: str, text: str, questions: dict, ref: str | None):
        start = time.perf_counter()
        try:
            resp = self._client.system_one(state=text, questions=questions)
        except TypeSafeError as exc:
            self._record(CallRecord(datetime.now().isoformat(timespec="seconds"), purpose, ref,
                                    (time.perf_counter() - start) * 1000, 0, 0, 0.0, False,
                                    f"{type(exc).__name__}: {exc}"))
            raise
        latency = (time.perf_counter() - start) * 1000
        tokens_in = resp.usage.input_tokens or 0
        rec = CallRecord(datetime.now().isoformat(timespec="seconds"), purpose, ref, latency,
                         tokens_in, resp.usage.output_tokens or 0,
                         self._billed_cost(resp, tokens_in), True)
        self._record(rec)
        return resp, rec

    @staticmethod
    def _billed_cost(resp, tokens_in: int) -> float:
        """OpenRouter's usage.cost (the SDK's Usage model drops unknown fields, so
        read it from the raw body); list price if it's missing."""
        try:
            cost = json.loads(resp.raw_http_response.content)["usage"]["cost"]
            if isinstance(cost, (int, float)):
                return float(cost)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
        return tokens_in * PRICE_PER_MILLION_INPUT_TOKENS / 1_000_000

    def _record(self, rec: CallRecord) -> None:
        self.meter.add(rec)
        log.info("jev %-8s ref=%s %.0fms in=%d cost=$%.8f%s", rec.purpose, rec.ref,
                 rec.latency_ms, rec.input_tokens, rec.cost_usd,
                 "" if rec.ok else f" ERROR {rec.error}")
        if self._log_path:
            line = json.dumps(asdict(rec)) + "\n"
            with _LOG_WRITE_LOCK, self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    @staticmethod
    def _label(ans) -> LabelAnswer:
        return LabelAnswer(ans.choice, ans.confidence, dict(ans.probabilities))

    @staticmethod
    def _escalation(ans) -> EscalationAnswer:
        probs = {ESCALATION_LEVELS[int(k)]: v for k, v in ans.probabilities.items()}
        return EscalationAnswer(max(probs, key=probs.get), ans.score, ans.confidence, probs)

    # -- public ------------------------------------------------------------ #

    def classify(self, text: str, *, ref: str | None = None) -> JevResult:
        """Gate, then (if relevant) classify. Never raises on API errors — the
        result carries `error` and zero confidence so the router sends it to review."""
        self.meter.count_chunk()
        try:
            gate, gate_rec = self._call("gate", text, RELEVANCE_QUESTIONS, ref)
        except TypeSafeError as exc:
            return JevResult(text, False, 0.0, 0.0, error=str(exc))

        p = gate.nouls["is_discharge_related"].noul
        relevant = p >= RELEVANCE_THRESHOLD
        result = JevResult(text, relevant, p, confidence=p if relevant else 1 - p,
                           latency_ms=gate_rec.latency_ms, input_tokens=gate_rec.input_tokens,
                           cost_usd=gate_rec.cost_usd)
        if not relevant:
            return result

        try:
            resp, rec = self._call("classify", text, CLASSIFICATION_QUESTIONS, ref)
        except TypeSafeError as exc:
            result.error, result.confidence = str(exc), 0.0
            return result

        c = resp.choices
        result.category = self._label(c["barrier_category"])
        result.status = self._label(c["barrier_status"])
        result.readiness = self._label(c["medical_readiness"])
        result.owner = self._label(c["barrier_owner"])
        result.escalation = self._escalation(resp.scores["escalation_need"])
        result.latency_ms += rec.latency_ms
        result.input_tokens += rec.input_tokens
        result.cost_usd += rec.cost_usd

        p_resolved = result.status.probabilities.get(RESOLVED_STATUS, 0.0)
        driver_conf = {"is_discharge_related": p,
                       "barrier_category": result.category.confidence,
                       # resolved vs not — see questions.CONFIDENCE_DRIVERS
                       "barrier_status": max(p_resolved, 1 - p_resolved)}
        drivers = [d for d in CONFIDENCE_DRIVERS
                   if not (d == "barrier_status"
                           and result.category.label in STATUS_IGNORED_FOR_CATEGORIES)]
        result.confidence = min(driver_conf[name] for name in drivers)
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_result(r: JevResult) -> None:
    print(f"\n\"{r.text}\"")
    if r.error:
        print(f"  ERROR: {r.error}")
        return
    print(f"  discharge-related: {r.is_discharge_related} (p={r.relevance_probability:.2f})"
          f"   confidence={r.confidence:.2f}   {r.latency_ms:.0f}ms  {r.input_tokens} tok")
    if r.classified:
        for name, a in (("category", r.category), ("status", r.status),
                        ("readiness", r.readiness), ("owner", r.owner)):
            print(f"  {name:<10} {a.label:<22} ({a.confidence:.2f})")
        e = r.escalation
        print(f"  {'escalation':<10} {e.label:<22} (score {e.score:.2f}, conf {e.confidence:.2f})")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--samples", action="store_true", help="classify a few corpus hard cases")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console

    if args.samples:
        from chunker import chunk_corpus
        from corpus import load_corpus
        chunks = [c for c in chunk_corpus(load_corpus()) if not c.is_empty]
        # Pick by surface cues only (the pipeline never reads the answer key).
        cues = ["can now", "may need", "Possible", "declined", "might", "Auth approved",
                "IV pump", "no needs identified"]
        texts = [next(c.text for c in chunks if cue in c.text) for cue in cues
                 if any(cue in c.text for c in chunks)]
    else:
        texts = [args.text or "Auth still pending with Summit Health Plan; pt medically cleared."]

    with JevClient() as jev:
        for t in texts:
            _print_result(jev.classify(t))
        s = jev.meter.summary()
    print(f"\n{s['chunks']} chunks · {s['calls']} calls · {s['input_tokens']} input tokens · "
          f"${s['cost_usd']:.6f} · avg {s['avg_latency_ms']:.0f}ms · {s['errors']} errors")


if __name__ == "__main__":
    main()
