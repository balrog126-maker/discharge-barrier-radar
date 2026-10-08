"""
Stage 5 — evaluation harness.

Runs the real pipeline (chunker -> Jev -> ledger) over the synthetic corpus and
grades it against the answer key, which only this module reads.

What "correct" means
  * Snippet level: the set of (barrier category, resolved?) pairs predicted from a
    snippet's new text must equal the answer key's set for that snippet. Copy-
    forwarded text is excluded from the key, so re-detecting it counts as an
    error. "newly_raised" vs "ongoing" is not graded here — the ledger decides
    that from history, not Jev.
  * Unit level (calibration): each classified unit (a sentence, or a whole
    chunk) is right if its category matches a barrier in the text it covers (or
    "none" when there is none), with resolved-ness matching too. Units the gate
    rejected count as a "none" prediction at the gate's confidence.
  * Board level: each patient's final open barriers, and the "medically ready
    but blocked" flag, versus the answer key replayed to the end of the stay.

All counting here is plain Python.

CLI (makes live Jev calls, about 2 cents per run):
  python eval.py                        # sentence granularity (the default)
  python eval.py --granularity chunk
  python eval.py --compare              # both, with a side-by-side headline
Writes data/eval_report.json (the UI's evaluation panel reads it).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from chunker import normalize, split_sentences
from corpus import load_corpus, load_ground_truth
from jev_client import MODEL, JevClient, JevResult
from pipeline import DEFAULT_GRANULARITY, GRANULARITIES, Pipeline
from questions import (
    BARRIER_CATEGORIES,
    NO_BARRIER_CATEGORY,
    READINESS_MIN_CONFIDENCE,
    RESOLVED_STATUS,
)
from router import AUTO_POST_THRESHOLD, CONFIRM_THRESHOLD, Route, route

REPORT_PATH = Path(__file__).parent / "data" / "eval_report.json"

# The hard-case subset the brief asks to be reported on its own.
HARD_CASE_TAGS = ("negation", "copy_forward", "hedged")
EXTRA_TAGS = ("multi_barrier", "chat_only", "stuck", "no_barrier")

CALIBRATION_BUCKETS = [          # (label, low inclusive, high exclusive)
    ("0.9–1.0", AUTO_POST_THRESHOLD, 1.01),
    ("0.7–0.9", 0.7, AUTO_POST_THRESHOLD),
    ("0.5–0.7", CONFIRM_THRESHOLD, 0.7),
    ("< 0.5", 0.0, CONFIRM_THRESHOLD),
]


# --------------------------------------------------------------------------- #
# Answer-key helpers
# --------------------------------------------------------------------------- #

def _sentence_index(truth: dict) -> dict[str, dict]:
    """Map each normalized sentence of every snippet to its answer-key segment info."""
    index = {}
    for sid, t in truth.items():
        barriers = {b["text"]: b for b in t["barriers"]}
        for seg in t["segments"]:
            info = {"category": seg["category"] if seg["kind"] == "barrier" and not seg["stale"]
                    else NO_BARRIER_CATEGORY,
                    "barrier": barriers.get(seg["text"]) if not seg["stale"] else None}
            # By sentence (notes are split) and whole (chat messages are classified whole).
            for sent in split_sentences(seg["text"]) + [seg["text"]]:
                index[(sid, normalize(sent))] = info
    return index


def _pred_pair(r: JevResult) -> tuple[str, bool] | None:
    """(category, resolved?) for a unit that names a barrier, else None."""
    if r.error or not r.is_discharge_related or not r.classified:
        return None
    if r.category.label == NO_BARRIER_CATEGORY:
        return None
    return r.category.label, r.status.label == RESOLVED_STATUS


def _truth_pairs(t: dict) -> set[tuple[str, bool]]:
    return {(b["category"], b["status"] == RESOLVED_STATUS) for b in t["barriers"]}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(granularity: str = DEFAULT_GRANULARITY, progress=None) -> dict:
    corpus, truth = load_corpus(), load_ground_truth()
    with JevClient() as jev:
        pipe = Pipeline(jev, granularity=granularity)
        steps = pipe.run(corpus, progress=progress)
        usage = jev.meter.summary()

    seg_index = _sentence_index(truth)
    per_snippet_pred: dict[str, set] = {}
    per_snippet_ready: dict[str, str] = {}
    units = []                     # calibration rows
    field_hits = Counter()         # owner / escalation / status agreement on matched units
    field_n = 0

    for st in steps:
        sid = st.snippet["snippet_id"]
        preds, ready = set(), "not_stated"
        for unit, r in zip(st.units, st.results):
            pair = _pred_pair(r)
            if pair:
                preds.add(pair)
            if (r.classified and r.readiness.label != "not_stated"
                    and r.readiness.confidence >= READINESS_MIN_CONFIDENCE):
                ready = r.readiness.label

            # unit-level truth from the answer-key segments this unit covers
            infos = [seg_index.get((sid, normalize(s))) for s in unit.new_sentences]
            infos = [i for i in infos if i]
            t_pairs = {(i["category"], i["barrier"]["status"] == RESOLVED_STATUS)
                       for i in infos if i["barrier"]}
            pred_cat = pair[0] if pair else NO_BARRIER_CATEGORY
            if pair:
                correct = pair in t_pairs
            else:
                correct = not t_pairs
            units.append({"snippet_id": sid, "text": unit.text, "confidence": r.confidence,
                          "route": route(r.confidence, error=bool(r.error)).value,
                          "predicted": pred_cat, "correct": correct,
                          "truth": sorted(c for c, _ in t_pairs) or [NO_BARRIER_CATEGORY]})

            # secondary questions, graded only where the category is right
            if pair and correct:
                b = next(i["barrier"] for i in infos
                         if i["barrier"] and i["category"] == pair[0])
                field_n += 1
                field_hits["owner"] += r.owner.label == b["owner"]
                field_hits["escalation"] += r.escalation.label == b["escalation"]
                field_hits["status"] += r.status.label == b["status"]
        per_snippet_pred[sid] = preds
        per_snippet_ready[sid] = ready

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "granularity": granularity,
        "n_snippets": len(corpus),
        "n_units": len(units),
        "per_category": _per_category(truth, per_snippet_pred),
        "snippet_accuracy": _snippet_accuracy(truth, per_snippet_pred),
        "calibration": _calibration(units),
        "routes": _routes(units),
        "secondary": {k: (field_hits[k] / field_n if field_n else None)
                      for k in ("status", "owner", "escalation")} | {"n": field_n},
        "readiness_accuracy": _readiness(truth, per_snippet_ready),
        "board": _board(corpus, truth, pipe.ledger),
        "usage": usage,
        "errors_sample": [u for u in units if not u["correct"]][:25],
    }


def _per_category(truth, pred) -> list[dict]:
    tp, fp, fn = Counter(), Counter(), Counter()
    for sid, t in truth.items():
        t_cats = {c for c, _ in _truth_pairs(t)} or {NO_BARRIER_CATEGORY}
        p_cats = {c for c, _ in pred.get(sid, set())} or {NO_BARRIER_CATEGORY}
        for c in p_cats & t_cats:
            tp[c] += 1
        for c in p_cats - t_cats:
            fp[c] += 1
        for c in t_cats - p_cats:
            fn[c] += 1
    rows = []
    for c in BARRIER_CATEGORIES:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else None
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else None
        f1 = 2 * p * r / (p + r) if p and r else None
        rows.append({"category": c, "support": tp[c] + fn[c], "tp": tp[c], "fp": fp[c],
                     "fn": fn[c], "precision": p, "recall": r, "f1": f1})
    return rows


def _snippet_accuracy(truth, pred) -> dict:
    ok = {sid: pred.get(sid, set()) == _truth_pairs(t) for sid, t in truth.items()}

    def acc(ids):
        ids = list(ids)
        return {"n": len(ids), "correct": sum(ok[i] for i in ids),
                "accuracy": (sum(ok[i] for i in ids) / len(ids)) if ids else None}

    tagged = lambda tag: [s for s, t in truth.items() if tag in t["hard_case_tags"]]
    hard = [s for s, t in truth.items() if set(HARD_CASE_TAGS) & set(t["hard_case_tags"])]
    easy = [s for s in truth if s not in set(hard)]
    return {
        "overall": acc(truth),
        "hard_subset": acc(hard),
        "rest": acc(easy),
        "by_tag": {tag: acc(tagged(tag)) for tag in HARD_CASE_TAGS + EXTRA_TAGS},
    }


def _calibration(units) -> list[dict]:
    rows = []
    for label, lo, hi in CALIBRATION_BUCKETS:
        b = [u for u in units if lo <= u["confidence"] < hi]
        rows.append({"bucket": label, "n": len(b),
                     "mean_confidence": sum(u["confidence"] for u in b) / len(b) if b else None,
                     "accuracy": sum(u["correct"] for u in b) / len(b) if b else None})
    return rows


def _routes(units) -> list[dict]:
    rows = []
    for rt in Route:
        b = [u for u in units if u["route"] == rt.value]
        rows.append({"route": rt.value, "n": len(b),
                     "accuracy": sum(u["correct"] for u in b) / len(b) if b else None})
    return rows


def _readiness(truth, pred) -> dict:
    stated = [s for s, t in truth.items() if t["medical_readiness"] != "not_stated"]
    return {
        "all": sum(pred[s] == truth[s]["medical_readiness"] for s in truth) / len(truth),
        "stated_only": (sum(pred[s] == truth[s]["medical_readiness"] for s in stated) / len(stated))
        if stated else None,
        "n_stated": len(stated),
    }


def _board(corpus, truth, ledger) -> dict:
    """Final state per patient: open barriers and ready-but-blocked, vs the answer key replayed."""
    t_open, t_ready = defaultdict(set), {}
    for s in sorted(corpus, key=lambda s: s["timestamp"]):
        t = truth[s["snippet_id"]]
        for b in t["barriers"]:
            if b["status"] == RESOLVED_STATUS:
                t_open[s["patient_id"]].discard(b["category"])
            else:
                t_open[s["patient_id"]].add(b["category"])
        if t["medical_readiness"] != "not_stated":
            t_ready[s["patient_id"]] = t["medical_readiness"]

    tp = fp = fn = 0
    rb = Counter()
    for pid in {s["patient_id"] for s in corpus}:
        p = ledger.patient(pid)
        pred = {b.category for b in p.open_barriers} if p else set()
        tp += len(pred & t_open[pid])
        fp += len(pred - t_open[pid])
        fn += len(t_open[pid] - pred)
        truth_rb = t_ready.get(pid) == "ready" and bool(t_open[pid])
        pred_rb = bool(p and p.ready_but_blocked)
        rb[(truth_rb, pred_rb)] += 1
    prec = lambda a, b: a / (a + b) if a + b else None
    return {
        "open_barriers": {"tp": tp, "fp": fp, "fn": fn,
                          "precision": prec(tp, fp), "recall": prec(tp, fn)},
        "ready_but_blocked": {"tp": rb[(True, True)], "fp": rb[(False, True)],
                              "fn": rb[(True, False)],
                              "precision": prec(rb[(True, True)], rb[(False, True)]),
                              "recall": prec(rb[(True, True)], rb[(True, False)])},
        "review_queue": len(ledger.review_queue()),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _f(x, pct=True) -> str:
    if x is None:
        return "   –  "
    return f"{100 * x:5.1f}%" if pct else f"{x:.2f}"


def format_report(rep: dict) -> str:
    L = []
    w = 72
    L.append("=" * w)
    L.append(f" Discharge-Barrier Radar — evaluation   ({rep['model']}, {rep['granularity']}-level)")
    L.append(f" {rep['n_snippets']} snippets · {rep['n_units']} classified units · "
             f"{rep['generated_at']}")
    L.append("=" * w)

    sa = rep["snippet_accuracy"]
    L.append("\nSNIPPET ACCURACY  (predicted barrier set == answer key)")
    L.append(f"  overall                      {_f(sa['overall']['accuracy'])}   "
             f"({sa['overall']['correct']}/{sa['overall']['n']})")
    L.append(f"  hard cases (negation, copy-forward, hedged)"
             f"  {_f(sa['hard_subset']['accuracy'])}   "
             f"({sa['hard_subset']['correct']}/{sa['hard_subset']['n']})")
    L.append(f"  everything else              {_f(sa['rest']['accuracy'])}   "
             f"({sa['rest']['correct']}/{sa['rest']['n']})")
    L.append("  by tag:")
    for tag, a in sa["by_tag"].items():
        L.append(f"    {tag:<16} {_f(a['accuracy'])}   ({a['correct']}/{a['n']})")

    L.append("\nPER-CATEGORY PRECISION / RECALL  (snippet level)")
    L.append(f"  {'category':<22}{'support':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    for r in rep["per_category"]:
        L.append(f"  {r['category']:<22}{r['support']:>8}{_f(r['precision']):>11}"
                 f"{_f(r['recall']):>9}{_f(r['f1']):>8}")

    L.append("\nCALIBRATION  (every classified unit, bucketed by confidence)")
    L.append(f"  {'confidence':<12}{'n':>6}{'mean conf':>12}{'accuracy':>11}")
    for r in rep["calibration"]:
        L.append(f"  {r['bucket']:<12}{r['n']:>6}{_f(r['mean_confidence']):>12}"
                 f"{_f(r['accuracy']):>11}")
    L.append("  by route:")
    for r in rep["routes"]:
        L.append(f"    {r['route']:<20}{r['n']:>6}{_f(r['accuracy']):>11}")

    sec = rep["secondary"]
    L.append(f"\nOTHER QUESTIONS  (on {sec['n']} units with the category right)")
    L.append(f"  status {_f(sec['status'])}   owner {_f(sec['owner'])}   "
             f"escalation {_f(sec['escalation'])}")
    ra = rep["readiness_accuracy"]
    L.append(f"  medical readiness {_f(ra['all'])} all snippets, "
             f"{_f(ra['stated_only'])} where the note states it (n={ra['n_stated']})")

    bd = rep["board"]
    ob, rb = bd["open_barriers"], bd["ready_but_blocked"]
    L.append("\nBOARD  (final state per patient vs answer key)")
    L.append(f"  open barriers        precision {_f(ob['precision'])}  recall {_f(ob['recall'])}"
             f"   (tp {ob['tp']}, fp {ob['fp']}, fn {ob['fn']})")
    L.append(f"  ready-but-blocked    precision {_f(rb['precision'])}  recall {_f(rb['recall'])}"
             f"   (tp {rb['tp']}, fp {rb['fp']}, fn {rb['fn']})")
    L.append(f"  review queue items   {bd['review_queue']}")

    u = rep["usage"]
    L.append("\nCOST")
    L.append(f"  {u['chunks']} classifications · {u['calls']} Jev calls · "
             f"{u['input_tokens']:,} input tokens · ${u['cost_usd']:.4f} · "
             f"avg {u['avg_latency_ms']:.0f} ms/call · {u['errors']} errors")
    if u["chunks"]:
        L.append(f"  ${1000 * u['cost_usd'] / u['chunks']:.4f} per 1,000 classifications")
    L.append("=" * w)
    return "\n".join(L)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console
    ap = argparse.ArgumentParser(description="Evaluate the pipeline against the answer key.")
    ap.add_argument("--granularity", choices=GRANULARITIES, default=DEFAULT_GRANULARITY)
    ap.add_argument("--compare", action="store_true", help="run every granularity")
    args = ap.parse_args()

    progress = lambda d, n: print(f"\r  classifying {d}/{n}", end="", flush=True)
    grans = GRANULARITIES if args.compare else (args.granularity,)
    reports = {}
    for g in grans:
        print(f"\nRunning pipeline ({g}-level)…")
        reports[g] = evaluate(g, progress)
        print("\r" + " " * 40 + "\r", end="")
        print(format_report(reports[g]))

    if args.compare:
        print("\nSIDE BY SIDE")
        print(f"  {'':<24}" + "".join(f"{g:>12}" for g in grans))
        rows = [("snippet accuracy", lambda r: r["snippet_accuracy"]["overall"]["accuracy"]),
                ("hard-case accuracy", lambda r: r["snippet_accuracy"]["hard_subset"]["accuracy"]),
                ("board precision", lambda r: r["board"]["open_barriers"]["precision"]),
                ("board recall", lambda r: r["board"]["open_barriers"]["recall"])]
        for name, fn in rows:
            print(f"  {name:<24}" + "".join(f"{_f(fn(reports[g])):>12}" for g in grans))
        print(f"  {'cost':<24}" + "".join(f"{'$%.4f' % reports[g]['usage']['cost_usd']:>12}"
                                          for g in grans))

    REPORT_PATH.write_text(json.dumps(reports[grans[0]], indent=2), encoding="utf-8")
    print(f"\nSaved {REPORT_PATH.relative_to(Path.cwd()) if REPORT_PATH.is_relative_to(Path.cwd()) else REPORT_PATH}")


if __name__ == "__main__":
    main()
