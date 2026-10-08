"""
Stage 2 — Change-detection chunker.

Clinical notes are copy-forwarded constantly: yesterday's "Awaiting SNF bed
offer" is pasted into today's note verbatim. If we sent the whole note to Jev
each day, the same barrier would be re-detected (and re-timestamped) forever.

The chunker keeps a per-patient memory of every sentence already seen and
emits only the sentences that are new or changed. Pure Python, no Jev.

Rules:
  * Notes are split into sentences and compared against all sentences from the
    patient's previous notes (not just the last one — text copied from two
    notes back is still stale).
  * Comparison is on a normalized form (case, whitespace, punctuation), with a
    near-duplicate threshold so trivial re-typing ("Auth pending w/ X." vs
    "auth pending w/ X") counts as the same sentence, while a real edit
    ("pending" -> "approved") counts as changed and is emitted.
  * Chat messages are conversational and are never copy-forwarded, so each
    message is one chunk; only an exact repeat of an earlier message is dropped.

Usage:
  python chunker.py            # run the copy-forward self-test against the corpus
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Sentences at least this similar (after normalization) to something already
# seen are treated as copy-forward. Kept high so small factual edits survive.
NEAR_DUPLICATE_RATIO = 0.93

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_PUNCT = str.maketrans("", "", string.punctuation.replace("#", ""))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def normalize(sentence: str) -> str:
    return " ".join(sentence.lower().translate(_PUNCT).split())


@dataclass
class Chunk:
    """New/changed text from one snippet — the unit that gets sent to Jev."""
    snippet_id: str
    patient_id: str
    timestamp: str
    source: str
    author_role: str
    text: str                                   # new sentences, joined
    new_sentences: list[str]
    dropped_sentences: list[str] = field(default_factory=list)   # stale, kept for audit

    @property
    def is_empty(self) -> bool:
        return not self.new_sentences


class ChangeDetector:
    """Stateful, per-patient. Feed snippets in chronological order with `process()`
    — works equally for a batch replay and for notes arriving one at a time."""

    def __init__(self, near_duplicate_ratio: float = NEAR_DUPLICATE_RATIO):
        self.ratio = near_duplicate_ratio
        self._seen_note: dict[str, list[str]] = {}    # patient -> normalized sentences
        self._seen_chat: dict[str, set[str]] = {}

    def _is_seen(self, norm: str, history: list[str]) -> bool:
        if norm in history:
            return True
        return any(SequenceMatcher(None, norm, h).ratio() >= self.ratio for h in history)

    def process(self, snippet: dict) -> Chunk:
        pid = snippet["patient_id"]
        if snippet["source"] == "chat":
            seen = self._seen_chat.setdefault(pid, set())
            norm = normalize(snippet["text"])
            new, dropped = ([], [snippet["text"]]) if norm in seen else ([snippet["text"]], [])
            seen.add(norm)
        else:
            history = self._seen_note.setdefault(pid, [])
            new, dropped, today = [], [], []
            for sent in split_sentences(snippet["text"]):
                norm = normalize(sent)
                (dropped if self._is_seen(norm, history) else new).append(sent)
                today.append(norm)
            history.extend(today)      # add after the loop: a sentence can't stale itself

        return Chunk(snippet_id=snippet["snippet_id"], patient_id=pid,
                     timestamp=snippet["timestamp"], source=snippet["source"],
                     author_role=snippet["author_role"], text=" ".join(new),
                     new_sentences=new, dropped_sentences=dropped)


def chunk_corpus(snippets: list[dict]) -> list[Chunk]:
    """Chunk every snippet in chronological order. Empty chunks (nothing new)
    are returned too so callers can see what was suppressed; filter on `is_empty`."""
    detector = ChangeDetector()
    return [detector.process(s) for s in sorted(snippets, key=lambda s: s["timestamp"])]


# --------------------------------------------------------------------------- #
# Self-test against the corpus's copy-forward cases.
# Uses the ground truth ONLY to grade the chunker — the chunker never sees it.
# --------------------------------------------------------------------------- #

def _self_test() -> bool:
    from corpus import load_corpus, load_ground_truth

    corpus, truth = load_corpus(), load_ground_truth()
    chunks = chunk_corpus(corpus)

    def dropped_all(seg_text: str, chunk: Chunk) -> bool:
        return all(normalize(s) in {normalize(d) for d in chunk.dropped_sentences}
                   for s in split_sentences(seg_text))

    stale_total = stale_dropped = live_total = live_kept = 0
    header_total = header_dropped = 0
    misses: list[str] = []
    first_note: set[str] = set()

    for ch in chunks:
        gt = truth[ch.snippet_id]
        is_first_note = ch.source == "note" and ch.patient_id not in first_note
        if ch.source == "note":
            first_note.add(ch.patient_id)
        for seg in gt["segments"]:
            if seg["stale"]:
                stale_total += 1
                if dropped_all(seg["text"], ch):
                    stale_dropped += 1
                else:
                    misses.append(f"  STALE KEPT   {ch.snippet_id}: {seg['text']}")
            elif seg["kind"] == "barrier":
                live_total += 1
                if not dropped_all(seg["text"], ch):
                    live_kept += 1
                else:
                    misses.append(f"  LIVE DROPPED {ch.snippet_id}: {seg['text']}")
            elif seg["kind"] == "header" and not is_first_note:
                header_total += 1
                header_dropped += dropped_all(seg["text"], ch)

    sent_in = sum(len(c.new_sentences) + len(c.dropped_sentences) for c in chunks)
    sent_out = sum(len(c.new_sentences) for c in chunks)
    empty = sum(c.is_empty for c in chunks)
    cf_snippets = [c for c in chunks if "copy_forward" in truth[c.snippet_id]["hard_case_tags"]]

    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"
    print("\nChange-detection chunker — self-test")
    print("-" * 52)
    print(f"  copy-forward barrier sentences suppressed  {pct(stale_dropped, stale_total)}")
    print(f"  new barrier sentences kept                 {pct(live_kept, live_total)}")
    print(f"  repeated note headers suppressed           {pct(header_dropped, header_total)}")
    print(f"  sentences sent to Jev                      {pct(sent_out, sent_in)}")
    print(f"  snippets with nothing new (Jev skipped)    {empty}")
    print(f"  snippets containing copy-forward text      {len(cf_snippets)}")
    if misses:
        print("\nMisses:")
        print("\n".join(misses))

    # Show one worked example.
    ex = next((c for c in cf_snippets if c.new_sentences), None)
    if ex:
        print(f"\nExample — {ex.snippet_id}")
        print(f"  original: {next(s for s in corpus if s['snippet_id'] == ex.snippet_id)['text']}")
        print(f"  dropped : {ex.dropped_sentences}")
        print(f"  sent    : {ex.new_sentences}")
    print()
    return stale_dropped == stale_total and live_kept == live_total


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console
    raise SystemExit(0 if _self_test() else 1)
