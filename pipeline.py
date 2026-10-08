"""
The full pipeline: snippet -> chunker -> Jev -> ledger. Shared by eval.py and app.py.

  step(snippet)   one snippet at a time (the UI's "notes arriving" simulation)
  run(snippets)   a whole corpus: chunk in order, classify in parallel (Jev calls
                  don't depend on each other), then apply to the ledger in time order

Granularity:
  "sentence"  (default) one Jev classification per new sentence of a note; chats
              stay whole. A Choice answer can only name one category, so this is
              what lets a note that mentions two barriers produce two entries.
              Measured on the corpus: recall 0.95 vs 0.89 per-chunk at the same
              precision, for ~40% more calls (about half a cent per corpus run).
  "chunk"     one classification per snippet's new text

CLI (makes live Jev calls):
  python pipeline.py                     # whole corpus, print board + cost
  python pipeline.py --patients 5        # first 5 patients only
  python pipeline.py --granularity chunk
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from chunker import ChangeDetector, Chunk
from jev_client import JevClient, JevResult
from ledger import Ledger, LedgerUpdate

GRANULARITIES = ("sentence", "chunk")
DEFAULT_GRANULARITY = "sentence"
MAX_PARALLEL_CALLS = 8


@dataclass
class StepResult:
    snippet: dict
    chunk: Chunk
    units: list[Chunk]                 # what was actually sent to Jev
    results: list[JevResult]
    updates: list[LedgerUpdate]


class Pipeline:
    def __init__(self, jev: JevClient, ledger: Ledger | None = None,
                 granularity: str = DEFAULT_GRANULARITY):
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        self.jev = jev
        self.ledger = ledger or Ledger()
        self.granularity = granularity
        self.detector = ChangeDetector()

    def _units(self, chunk: Chunk) -> list[Chunk]:
        if chunk.is_empty:
            return []
        if self.granularity == "chunk" or chunk.source == "chat":
            return [chunk]
        # One unit per new sentence; copy-forward context stays on the first unit only.
        return [replace(chunk, text=s, new_sentences=[s],
                        dropped_sentences=chunk.dropped_sentences if i == 0 else [])
                for i, s in enumerate(chunk.new_sentences)]

    def _apply(self, snippet, chunk, units, results) -> StepResult:
        updates = [u for unit, res in zip(units, results) for u in self.ledger.apply(unit, res)]
        if not units:
            self.ledger._tick(snippet["timestamp"])
        return StepResult(snippet, chunk, units, results, updates)

    def step(self, snippet: dict) -> StepResult:
        chunk = self.detector.process(snippet)
        units = self._units(chunk)
        results = [self.jev.classify(u.text, ref=u.snippet_id) for u in units]
        return self._apply(snippet, chunk, units, results)

    def run(self, snippets: list[dict], progress=None) -> list[StepResult]:
        ordered = sorted(snippets, key=lambda s: s["timestamp"])
        chunks = [self.detector.process(s) for s in ordered]
        units = [self._units(c) for c in chunks]
        flat = [u for us in units for u in us]
        done = 0

        def classify(u: Chunk) -> JevResult:
            nonlocal done
            res = self.jev.classify(u.text, ref=u.snippet_id)
            done += 1
            if progress:
                progress(done, len(flat))
            return res

        with ThreadPoolExecutor(MAX_PARALLEL_CALLS) as pool:
            flat_results = list(pool.map(classify, flat))

        out, i = [], 0
        for snippet, chunk, us in zip(ordered, chunks, units):
            out.append(self._apply(snippet, chunk, us, flat_results[i:i + len(us)]))
            i += len(us)
        return out


# --------------------------------------------------------------------------- #
# CLI — print the resulting board
# --------------------------------------------------------------------------- #

def _print_board(ledger: Ledger) -> None:
    badge = {"auto": "AUTO     ", "needs_confirmation": "CONFIRM? ",
             "confirmed": "CONFIRMED", "dismissed": "dismissed"}
    print(f"\nBoard as of {ledger.clock}  (barrier ages are as of each patient's latest note)")
    print("=" * 78)
    for p in ledger.board():
        ready = {"ready": "READY", "not_ready": "not ready"}.get(
            p.readiness.label if p.readiness else "", "readiness ?")
        flag = "  <-- ready but blocked" if p.ready_but_blocked else ""
        print(f"{p.patient_id}  [{ready}]{flag}")
        now = p.as_of
        for b in sorted(p.board_barriers, key=lambda b: -b.age_days(now)):
            state = "RESOLVED? " if b.resolution_unconfirmed else ""
            print(f"    {badge[b.verification.value]} {state}{b.category:<21} {b.status:<12} "
                  f"age {b.age_days(now):4.1f}d  owner={b.owner:<17} conf={b.confidence:.2f}"
                  f"  esc={b.escalation}")
    queue = ledger.review_queue()
    print(f"\nReview queue: {len(queue)} item(s)")
    for item in queue[:10]:
        cat = item.result.category.label if item.result.category else "-"
        print(f"    {item.snippet_id:<18} {item.reason:<15} conf={item.confidence:.2f}  {cat}: "
              f"\"{item.text[:60]}\"")


def main() -> None:
    import argparse
    from corpus import load_corpus
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console

    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, help="limit to the first N patients")
    ap.add_argument("--granularity", choices=GRANULARITIES, default=DEFAULT_GRANULARITY)
    args = ap.parse_args()

    snippets = load_corpus()
    if args.patients:
        keep = sorted({s["patient_id"] for s in snippets})[:args.patients]
        snippets = [s for s in snippets if s["patient_id"] in keep]

    with JevClient() as jev:
        pipe = Pipeline(jev, granularity=args.granularity)
        pipe.run(snippets, progress=lambda d, n: print(f"\r  classifying {d}/{n}", end=""))
        s = jev.meter.summary()
    _print_board(pipe.ledger)
    print(f"\n{s['chunks']} classifications · {s['calls']} calls · ${s['cost_usd']:.4f} · "
          f"avg {s['avg_latency_ms']:.0f}ms · {s['errors']} errors")


if __name__ == "__main__":
    main()
