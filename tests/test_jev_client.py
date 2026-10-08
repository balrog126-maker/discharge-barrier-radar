"""Offline tests for the Jev wrapper — a mock transport stands in for the API."""

import json
import re
import sys
from pathlib import Path

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_client import PRICE_PER_MILLION_INPUT_TOKENS, JevClient  # noqa: E402
from questions import CLASSIFICATION_QUESTIONS, RELEVANCE_QUESTIONS  # noqa: E402


def fake_jev(relevance: float, category_conf: float = 0.97, status_conf: float = 0.95,
             status: int = 200, tokens: int = 250, cost: float | None = None,
             first_labels: dict | None = None):
    """Build a MockTransport that answers whatever questions it's sent."""
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        calls.append(body)
        if status != 200:
            return httpx2.Response(status, json={"detail": "nope"})
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": relevance}
            elif q["type"] == "choice":
                labels = list(q["criteria"])
                if first_labels and name in first_labels:          # force the winning label
                    labels.remove(first_labels[name])
                    labels.insert(0, first_labels[name])
                conf = {"barrier_category": category_conf, "barrier_status": status_conf}.get(name, 0.9)
                rest = (1 - conf) / (len(labels) - 1)
                probs = {lab: (conf if i == 0 else rest) for i, lab in enumerate(labels)}
                answers[name] = {"type": "choice", "choice": labels[0], "confidence": conf,
                                 "probabilities": probs}
            else:
                answers[name] = {"type": "score", "score": 1.8, "confidence": 0.8,
                                 "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                 "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}}
        usage = {"input_tokens": tokens, "output_tokens": 5}
        if cost is not None:                       # OpenRouter adds the billed amount
            usage["cost"] = cost
        return httpx2.Response(200, json={"model": "typesafe/jev-1.13", "answers": answers,
                                          "usage": usage})

    return httpx2.MockTransport(handler), calls


def client(transport):
    return JevClient(api_key="test-key", transport=transport, log_path=None)


def test_irrelevant_chunk_stops_at_gate():
    transport, calls = fake_jev(relevance=0.08)
    with client(transport) as jev:
        r = jev.classify("pt asking for a warm blanket")
    assert len(calls) == 1 and set(calls[0]["questions"]) == set(RELEVANCE_QUESTIONS)
    assert not r.is_discharge_related and not r.classified
    assert r.confidence == pytest.approx(0.92)


def test_relevant_chunk_asks_all_questions_in_one_batched_call():
    transport, calls = fake_jev(relevance=0.99)
    with client(transport) as jev:
        r = jev.classify("Prior auth for SNF submitted to Summit Health Plan.")
    assert len(calls) == 2
    assert set(calls[1]["questions"]) == set(CLASSIFICATION_QUESTIONS)
    assert r.category.label == "post_acute_placement"      # first label in fake
    assert r.escalation.label == "stuck" and r.escalation.score == pytest.approx(1.8)


def test_confidence_is_weakest_driver():
    # Jev is only 62% sure the barrier is resolved -> that uncertainty caps confidence.
    transport, _ = fake_jev(relevance=0.99, category_conf=0.97, status_conf=0.62,
                            first_labels={"barrier_status": "resolved"})
    with client(transport) as jev:
        assert jev.classify("x").confidence == pytest.approx(0.62)


def test_newly_raised_vs_ongoing_split_does_not_lower_confidence():
    # 62% newly_raised, rest spread over ongoing/resolved/n.a. -> P(resolved) ~0.13,
    # so "not resolved" is ~0.87 sure; the ledger decides new vs ongoing itself.
    transport, _ = fake_jev(relevance=0.99, category_conf=0.97, status_conf=0.62)
    with client(transport) as jev:
        assert jev.classify("x").confidence == pytest.approx(1 - 0.38 / 3)


def test_status_is_ignored_in_confidence_when_no_barrier():
    transport, _ = fake_jev(relevance=0.98, category_conf=0.99, status_conf=0.50,
                            first_labels={"barrier_category": "none"})
    with client(transport) as jev:
        r = jev.classify("Dc plan is home with spouse; no needs identified.")
    assert r.category.label == "none" and r.confidence == pytest.approx(0.98)


def test_usage_and_cost_are_metered():
    transport, _ = fake_jev(relevance=0.99, tokens=300)
    with client(transport) as jev:
        r = jev.classify("x")
        s = jev.meter.summary()
    assert r.input_tokens == 600 and s["calls"] == 2 and s["chunks"] == 1
    assert s["cost_usd"] == pytest.approx(600 * PRICE_PER_MILLION_INPUT_TOKENS / 1e6)
    assert s["avg_latency_ms"] > 0


def test_openrouter_reported_cost_is_used_over_list_price():
    transport, _ = fake_jev(relevance=0.99, tokens=300, cost=0.00003)
    with client(transport) as jev:
        jev.classify("x")
        s = jev.meter.summary()
    assert s["cost_usd"] == pytest.approx(0.00006)       # two calls x reported cost


def test_requests_go_to_openrouter_with_pinned_model():
    transport, calls = fake_jev(relevance=0.1)
    seen = []
    inner = transport.handler
    transport.handler = lambda req: (seen.append(req), inner(req))[1]
    with client(transport) as jev:
        jev.classify("x")
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/systemone"
    assert calls[0]["model"] == "jev-1.13"


def test_api_error_is_captured_not_raised():
    transport, _ = fake_jev(relevance=0.99, status=403)
    with client(transport) as jev:
        r = jev.classify("x")
        s = jev.meter.summary()
    assert r.error and r.confidence == 0.0 and not r.classified
    assert s["errors"] == 1


def test_call_log_stays_valid_jsonl_under_parallel_calls(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    transport, _ = fake_jev(relevance=0.99)
    log_file = tmp_path / "calls.jsonl"
    with JevClient(api_key="k", transport=transport, log_path=log_file) as jev:
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(jev.classify, ["x"] * 80))
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 160 and all(json.loads(l)["ok"] for l in lines)


# ---- hard-constraint guards ------------------------------------------------ #

def _project_py_files():
    return [p for p in ROOT.glob("*.py")]


def test_only_questions_py_defines_questions():
    pattern = re.compile(r"\b(Choice|Noul|Score)\(")
    offenders = [p.name for p in _project_py_files()
                 if p.name != "questions.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_only_wrapper_touches_the_sdk_client():
    offenders = [p.name for p in _project_py_files()
                 if p.name != "jev_client.py"
                 and re.search(r"TypeSafeClient|system_one", p.read_text(encoding="utf-8"))]
    assert offenders == []
