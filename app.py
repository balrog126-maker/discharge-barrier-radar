"""
Stage 6 — Discharge-Barrier Radar board.

    streamlit run app.py

Replays the synthetic corpus as if notes were arriving during the patients'
stays: each step runs the real pipeline (chunker -> Jev -> ledger), so the
board, the cost counter and the review queue all update live.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import streamlit as st

import eval as evaluation
import ui
from corpus import load_corpus, load_patients
from jev_client import JevClient
from pipeline import Pipeline
from router import CONFIRM_THRESHOLD

AUTOPLAY_SECONDS = 3          # one simulated day per tick
FEED_LENGTH = 14

st.set_page_config(page_title="Discharge-Barrier Radar", page_icon="◎", layout="wide",
                   initial_sidebar_state="expanded")
ss = st.session_state

# Theme: ?theme=dark opens in dark mode (and the toggle keeps the URL in sync, so a reload
# or a shared link keeps the choice). Decided before any CSS is emitted.
if "dark_mode" not in ss:
    ss.dark_mode = st.query_params.get("theme") == "dark"
DARK = bool(ss.dark_mode)


def _sync_theme_param() -> None:
    st.query_params["theme"] = "dark" if ss.dark_mode else "light"


st.markdown(ui.theme_css(DARK) + ui.CSS, unsafe_allow_html=True)
st.toggle(":material/dark_mode: Dark mode", key="dark_mode", on_change=_sync_theme_param)


# --------------------------------------------------------------------------- #
# Data & state
# --------------------------------------------------------------------------- #

@st.cache_data
def corpus() -> list[dict]:
    return sorted(load_corpus(), key=lambda s: s["timestamp"])


@st.cache_data
def patient_meta() -> dict[str, dict]:
    return {p["patient_id"]: p for p in load_patients()}


@st.cache_data
def originals() -> dict[str, str]:
    return {s["snippet_id"]: s["text"] for s in corpus()}


def jev() -> JevClient:
    if "jev" not in ss:
        try:
            ss.jev = JevClient()
        except RuntimeError as exc:
            st.error(f"**Jev isn't configured.** {exc} Add `OPENROUTER_API_KEY=…` to the "
                     "project's `.env` file and reload.")
            st.stop()
    return ss.jev


def reset_board() -> None:
    ss.pipe = Pipeline(jev())
    ss.cursor, ss.feed, ss.recent = 0, [], set()


if "pipe" not in ss:
    reset_board()
    ss.autoplay, ss.last_tick = False, 0.0

ledger = ss.pipe.ledger
N = len(corpus())


def _replay_from_url() -> None:
    """?replay=end opens the board fully replayed (for demos); ?replay=<n> replays n days.
    Runs once per session, and only when the parameter is present — it makes Jev calls."""
    arg = st.query_params.get("replay")
    if not arg or ss.get("replayed_from_url"):
        return
    ss.replayed_from_url = True
    with st.spinner("Replaying the synthetic stays…"):
        if arg == "end":
            advance("end")
        elif arg.isdigit():
            for _ in range(int(arg)):
                advance("day")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def _describe(u) -> tuple[str, str] | None:
    label = ui.CATEGORY_LABEL.get(u.category or "", "")
    route = {"auto_post": "auto-posted", "needs_confirmation": "needs confirmation",
             "review_queue": "sent to review"}.get(u.route.value if u.route else "", "")
    if u.action == "raised":
        return f"{label} raised", route
    if u.action == "resolved":
        return f"{label} resolved", route
    if u.action == "reopened":
        return f"{label} reopened", route
    if u.action == "continued":
        return f"{label} updated", route
    if u.action == "review":
        return "Low-confidence reading held for review", "review queue"
    if u.action == "readiness":
        p = ledger.patient(u.patient_id)
        state = "medically ready" if p and p.medically_ready else "not medically ready"
        return f"Marked {state}", "readiness"
    return None


def _record(steps) -> None:
    new, recent = [], set()
    for step in steps:
        for u in step.updates:
            d = _describe(u)
            if d:
                new.append({"patient_id": u.patient_id, "what": d[0], "route": d[1],
                            "category": u.category, "ts": step.snippet["timestamp"]})
                recent.add(u.patient_id)
    ss.feed = (list(reversed(new)) + ss.feed)[:FEED_LENGTH]
    ss.recent = recent


def advance(mode: str) -> None:
    """mode: 'note' | 'day' | 'end'."""
    items = corpus()
    if ss.cursor >= N:
        return
    if mode == "end":
        batch = items[ss.cursor:]
    elif mode == "day":
        day = items[ss.cursor]["timestamp"][:10]
        batch = [s for s in items[ss.cursor:] if s["timestamp"][:10] == day]
    else:
        batch = items[ss.cursor:ss.cursor + 1]
    steps = ss.pipe.run(batch) if len(batch) > 1 else [ss.pipe.step(batch[0])]
    ss.cursor += len(batch)
    _record(steps)


def confirm(pid: str, cat: str) -> None:
    ledger.confirm(pid, cat)


def dismiss(pid: str, cat: str) -> None:
    ledger.dismiss(pid, cat)


_replay_from_url()


# --------------------------------------------------------------------------- #
# Sidebar — simulation controls, legend, activity
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="sect">Simulation</div>', unsafe_allow_html=True)
    st.progress(ss.cursor / N if N else 0.0,
                text=f"{ss.cursor} of {N} notes & messages replayed")
    done = ss.cursor >= N
    c1, c2 = st.columns(2)
    if c1.button("Next note", width="stretch", disabled=done):
        with st.spinner("Classifying…"):
            advance("note")
        st.rerun()
    if c2.button("Next day", width="stretch",
                 disabled=done, type="primary"):
        with st.spinner("Classifying a day of notes…"):
            advance("day")
        st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("Run to end", width="stretch",
                 disabled=done):
        with st.spinner("Classifying the rest of the stay…"):
            advance("end")
        st.rerun()
    if c4.button("Reset", width="stretch"):
        ss.autoplay = False
        reset_board()
        st.rerun()
    st.toggle(f"Auto-play · 1 day every {AUTOPLAY_SECONDS}s", key="autoplay", disabled=done)

    st.markdown('<div class="sect" style="margin-top:22px">Legend</div>', unsafe_allow_html=True)
    st.markdown(ui.legend(), unsafe_allow_html=True)

    st.markdown('<div class="sect" style="margin-top:22px">Latest activity</div>',
                unsafe_allow_html=True)
    st.markdown(ui.feed(ss.feed), unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div><div class="hint">All patients, notes and '
                'organisations are synthetic. Jev classifies text only; ages, counts and '
                'sorting are computed in Python.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Top bar
# --------------------------------------------------------------------------- #

st.markdown(ui.topbar(jev().meter.summary(), ledger.clock, ss.cursor, N), unsafe_allow_html=True)

# If Jev calls are failing, say why — otherwise the demo just fills the review queue silently.
_records = jev().meter.records
if _records and not _records[-1].ok:
    _err = _records[-1].error or "unknown error"
    _hint = (" Add credits to the OpenRouter account behind OPENROUTER_API_KEY."
             if " 402 " in _err else "")
    st.error(f"**Jev calls are failing** — the latest call returned: {_err.split(': ', 1)[-1][:220]}"
             f"{_hint} Failed readings are held in the review queue, never posted.",
             icon=":material/error:")

patients = ledger.board()
review_items = ledger.review_queue()
tab_board, tab_insights, tab_review, tab_eval = st.tabs(
    ["Board", "Where time is lost", f"Review queue · {len(review_items)}", "Evaluation"])


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #

def render_card(p) -> None:
    now = p.as_of
    key = f"cardrb_{p.patient_id}" if p.ready_but_blocked else f"card_{p.patient_id}"
    with st.container(key=key):
        st.markdown(ui.card_head(p, patient_meta().get(p.patient_id), p.patient_id in ss.recent),
                    unsafe_allow_html=True)
        rows = sorted(p.board_barriers, key=lambda b: (not b.is_open, -b.age_days(now)))
        for b in rows:
            st.markdown(ui.barrier_row(b, now), unsafe_allow_html=True)
            unverified = b.verification.value == "needs_confirmation"
            if unverified:
                pad, a, r = st.columns([2.6, 2.3, 2.1])
                resolving = b.resolution_unconfirmed
                a.button("Confirm resolved" if resolving else "Confirm", icon=":material/check:",
                         key=f"ok_{p.patient_id}_{b.category}", on_click=confirm,
                         args=(p.patient_id, b.category), width="stretch")
                r.button("Not resolved" if resolving else "Dismiss", icon=":material/close:",
                         key=f"no_{p.patient_id}_{b.category}", on_click=dismiss,
                         args=(p.patient_id, b.category), width="stretch")
        n_ev = sum(len(b.evidence) for b in rows)
        with st.expander(f"Why? Show the source text ({n_ev} note{'s' if n_ev != 1 else ''})"):
            for b in rows:
                st.markdown(ui.evidence(b, originals()), unsafe_allow_html=True)
            if p.readiness:
                st.markdown(
                    f'<div class="ev"><div class="ev-cat">Medical readiness</div>'
                    f'<div class="ev-head">{ui.escape(ui.fmt_ts(p.readiness.as_of))} · '
                    f'{ui.escape(p.readiness.snippet_id)}</div>'
                    f'<div class="ev-text"><mark>{ui.escape(p.readiness.text)}</mark></div>'
                    f'<div class="ev-read">Jev: {p.readiness.label.replace("_", " ")}, '
                    f'confidence {p.readiness.confidence:.2f}</div></div>',
                    unsafe_allow_html=True)


with tab_board:
    if ss.cursor == 0:
        st.markdown("""
<div class="empty"><h3>No notes have arrived yet</h3>
<p>Replay the synthetic stays note by note from the sidebar — <b>Next day</b> is a good pace —
or fast-forward to the end. Every note is classified live by Jev as it arrives.</p></div>""",
                    unsafe_allow_html=True)
        _, b1, b2, _ = st.columns([2, 1.3, 1.3, 2])
        if b1.button("Replay day 1", type="primary", width="stretch"):
            with st.spinner("Classifying…"):
                advance("day")
            st.rerun()
        if b2.button("Fast-forward to end", width="stretch"):
            with st.spinner(f"Classifying {N} notes…"):
                advance("end")
            st.rerun()
    else:
        blocked = [p for p in patients if p.board_barriers]
        rb = [p for p in patients if p.ready_but_blocked]
        unverified = sum(1 for p in patients for b in p.board_barriers
                         if b.verification.value == "needs_confirmation")
        open_days = sum(b.age_days(p.as_of) for p in patients for b in p.open_barriers)
        n_open = sum(len(p.open_barriers) for p in patients)
        st.markdown(ui.kpis([
            ("Medically ready, still blocked", str(len(rb)), "patients — the list to work first", True),
            ("Open barriers", str(n_open), f"across {len(blocked)} patients", False),
            ("Needs confirmation", str(unverified), "unverified entries on the board", False),
            ("Barrier-days open", f"{open_days:.0f}", "sum of open barrier ages", False),
            ("Review queue", str(len(review_items)), "low confidence, not on the board", False),
        ]), unsafe_allow_html=True)

        view = st.segmented_control("Show", ["Ready but blocked", "All open barriers"],
                                    default="All open barriers", label_visibility="collapsed")
        shown = rb if view == "Ready but blocked" else blocked
        if not shown:
            st.markdown('<div class="hint">Nobody in this view right now.</div>',
                        unsafe_allow_html=True)
        cols = st.columns(2, gap="medium")
        for i, p in enumerate(shown):
            with cols[i % 2]:
                render_card(p)

        clear = [p for p in patients if not p.board_barriers]
        if clear:
            st.markdown('<div class="sect" style="margin-top:10px">No open barriers</div>',
                        unsafe_allow_html=True)
            chips = "".join(
                f'<span class="clearchip"><b>{ui.escape(p.patient_id)}</b> · '
                f'{"medically ready" if p.medically_ready else "not ready" if p.readiness else "readiness not stated"}</span>'
                for p in clear)
            st.markdown(f'<div class="clearlist">{chips}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Where time is lost
# --------------------------------------------------------------------------- #

with tab_insights:
    if ss.cursor == 0:
        st.markdown('<div class="hint">Replay some notes to see where discharge time goes.</div>',
                    unsafe_allow_html=True)
    else:
        by_cat = defaultdict(lambda: {"barrier_days": 0.0, "open": 0, "patients": set()})
        owners = Counter()
        for p in patients:
            for b in p.barriers.values():
                if b.verification.value == "dismissed":
                    continue
                row = by_cat[b.category]
                row["barrier_days"] += b.age_days(p.as_of)
                row["patients"].add(p.patient_id)
                if b.is_open:
                    row["open"] += 1
                    owners[b.owner] += 1
        rows = [{"category": c, "barrier_days": v["barrier_days"], "open": v["open"],
                 "patients": len(v["patients"])} for c, v in by_cat.items()]

        left, right = st.columns([1.35, 1], gap="large")
        with left:
            st.markdown('<div class="sect">Barrier-days by category</div>', unsafe_allow_html=True)
            fam = "".join(f'<span class="row" style="display:inline-flex;margin-right:14px">'
                          f'<i style="--c:{c}"></i>&nbsp;{ui.escape(n)}</span>'
                          for n, (c, _) in ui.FAMILIES.items())
            st.markdown(f'<div class="legend" style="flex-direction:row;flex-wrap:wrap">{fam}</div>',
                        unsafe_allow_html=True)
            if rows:
                st.plotly_chart(ui.time_lost_chart(rows, dark=DARK), width="stretch",
                                config={"displayModeBar": False})
            st.markdown('<div class="hint">Each barrier contributes its age — open barriers up to '
                        "the patient's latest note, resolved ones up to resolution. Computed in "
                        'Python from the ledger timestamps.</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="sect">From admitted to stuck</div>', unsafe_allow_html=True)
            stages = [("Patients seen", len(patients)),
                      ("With an open barrier", sum(1 for p in patients if p.open_barriers)),
                      ("Medically ready", sum(1 for p in patients if p.medically_ready)),
                      ("Ready but blocked", sum(1 for p in patients if p.ready_but_blocked))]
            st.plotly_chart(ui.funnel_chart(stages, dark=DARK), width="stretch",
                            config={"displayModeBar": False})
            st.markdown('<div class="sect" style="margin-top:6px">Who holds the open barriers</div>',
                        unsafe_allow_html=True)
            if owners:
                st.plotly_chart(ui.owner_chart(dict(owners), dark=DARK), width="stretch",
                                config={"displayModeBar": False})
        with st.expander("Show as a table"):
            st.markdown(ui.table(
                ["Category", "Barrier-days", "Open now", "Patients"],
                [[ui.CATEGORY_LABEL[r["category"]], f"{r['barrier_days']:.1f}", str(r["open"]),
                  str(r["patients"])] for r in sorted(rows, key=lambda r: -r["barrier_days"])],
                num_cols={1, 2, 3}), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #

with tab_review:
    st.markdown('<div class="hint" style="margin-bottom:12px">Readings below the '
                f'{CONFIRM_THRESHOLD:.0%} confidence floor (and any failed calls) wait '
                'here. They never appear on the board unless a reviewer accepts them.</div>',
                unsafe_allow_html=True)
    if not review_items:
        st.markdown('<div class="empty"><h3>Nothing to review</h3><p>Low-confidence '
                    'readings will collect here.</p></div>', unsafe_allow_html=True)
    for item in review_items:
        r = item.result
        with st.container(key=f"card_rev{item.id}"):
            if r.classified:
                reading = (f"Jev's best guess: <b>{ui.escape(ui.CATEGORY_LABEL.get(r.category.label, r.category.label))}</b>, "
                           f"status “{ui.escape(r.status.label.replace('_', ' '))}” · confidence "
                           f"{item.confidence:.2f}")
            else:
                reading = f"Call failed: {ui.escape(r.error or 'unknown error')}"
            st.markdown(
                f'<div class="phead"><div><span class="pid">{ui.escape(item.patient_id)}</span>'
                f'<span class="loc">{ui.escape(ui.fmt_ts(item.timestamp))} · {ui.escape(item.snippet_id)}</span></div>'
                f'<span class="badge unverified">{"Low confidence" if item.reason == "low_confidence" else "API error"}</span></div>'
                f'<div class="ev-text" style="margin:8px 0 6px">“{ui.escape(item.text)}”</div>'
                f'<div class="ev-read">{reading}</div>', unsafe_allow_html=True)
            _, a, b = st.columns([5, 2.4, 2.1])
            if r.classified:
                a.button("Accept to board", icon=":material/check:", key=f"acc_{item.id}",
                         on_click=ledger.accept_review, args=(item.id,), width="stretch")
            b.button("Reject", icon=":material/close:", key=f"rej_{item.id}",
                     on_click=ledger.reject_review, args=(item.id,), width="stretch")


# --------------------------------------------------------------------------- #
# Evaluation — the credibility section
# --------------------------------------------------------------------------- #

def load_report() -> dict | None:
    if not evaluation.REPORT_PATH.exists():
        return None
    return json.loads(evaluation.REPORT_PATH.read_text(encoding="utf-8"))


with tab_eval:
    rep = load_report()
    head_l, head_r = st.columns([3, 1.2], vertical_alignment="bottom")
    with head_l:
        st.markdown('<div class="sect">How accurate is it?</div>', unsafe_allow_html=True)
        if rep:
            st.markdown(
                f'<div class="hint">Full pipeline (change-detection → Jev {ui.escape(rep["model"])} → '
                f'ledger) over {rep["n_snippets"]} synthetic notes and messages, graded against a '
                f'separate answer key the pipeline never sees · {rep["granularity"]}-level · '
                f'run {ui.escape(ui.fmt_ts(rep["generated_at"]))}</div>', unsafe_allow_html=True)
    with head_r:
        rerun = st.button("Re-run evaluation live", icon=":material/refresh:",
                          width="stretch", help="About 560 Jev calls, ~2¢, under a minute")
    if rerun:
        with st.spinner("Running the full pipeline against the answer key…"):
            new = evaluation.evaluate()
            evaluation.REPORT_PATH.write_text(json.dumps(new, indent=2), encoding="utf-8")
        st.rerun()

    if not rep:
        st.markdown('<div class="empty"><h3>No evaluation yet</h3><p>Run it here, or '
                    '<code>python eval.py</code> in a terminal.</p></div>', unsafe_allow_html=True)
    else:
        sa = rep["snippet_accuracy"]
        auto = next(r for r in rep["routes"] if r["route"] == "auto_post")
        ob = rep["board"]["open_barriers"]
        u = rep["usage"]
        st.markdown(ui.kpis([
            ("Hard-case accuracy", ui.pct(sa["hard_subset"]["accuracy"]),
             f"negation, copy-forward, hedged · n={sa['hard_subset']['n']}", True),
            ("Overall accuracy", ui.pct(sa["overall"]["accuracy"]),
             f"exact barrier set per note · n={sa['overall']['n']}", False),
            ("Auto-posted & correct", ui.pct(auto["accuracy"]),
             f"{auto['n']} entries posted without a human", False),
            ("Board recall", ui.pct(ob["recall"]),
             f"real open barriers found · precision {ui.pct(ob['precision'])}", False),
            ("Cost of this run", ui.money(u["cost_usd"]),
             f"{u['calls']} calls · avg {u['avg_latency_ms']:.0f} ms", False),
        ]), unsafe_allow_html=True)

        c_left, c_right = st.columns([1.25, 1], gap="large")
        with c_left:
            st.markdown('<div class="sect">Calibration — does confidence mean what it says?</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ui.calibration_chart(rep["calibration"], dark=DARK), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(ui.table(
                ["Confidence", "Units", "Mean confidence", "Actual accuracy"],
                [[r["bucket"], str(r["n"]), ui.pct(r["mean_confidence"]), ui.pct(r["accuracy"])]
                 for r in rep["calibration"]], num_cols={1, 2, 3}), unsafe_allow_html=True)
        with c_right:
            st.markdown('<div class="sect">Hard cases, reported separately</div>',
                        unsafe_allow_html=True)
            names = {"negation": "Negation (barrier resolved)", "copy_forward": "Copy-forward text",
                     "hedged": "Hedged language", "multi_barrier": "Several barriers in one note",
                     "chat_only": "Only in chat", "stuck": "Stuck / escalation",
                     "no_barrier": "No barrier at all"}
            tag_rows = [[names.get(t, t), str(a["n"]), ui.pct(a["accuracy"])]
                        for t, a in sa["by_tag"].items()]
            weak = {(i, 2) for i, (_, a) in enumerate(sa["by_tag"].items())
                    if a["accuracy"] is not None and a["accuracy"] < 0.7}
            st.markdown(ui.table(["Case", "Notes", "Accuracy"], tag_rows, num_cols={1, 2},
                                 weak=weak), unsafe_allow_html=True)
            st.markdown('<div class="sect" style="margin-top:18px">By route</div>',
                        unsafe_allow_html=True)
            rnames = {"auto_post": "Auto-posted (≥ 0.9)", "needs_confirmation": "Needs confirmation",
                      "review_queue": "Review queue (< 0.5)"}
            st.markdown(ui.table(["Route", "Units", "Accuracy"],
                                 [[rnames[r["route"]], str(r["n"]), ui.pct(r["accuracy"])]
                                  for r in rep["routes"]], num_cols={1, 2}),
                        unsafe_allow_html=True)

        st.markdown('<div class="sect" style="margin-top:22px">Per category</div>',
                    unsafe_allow_html=True)
        cat_rows, weak = [], set()
        for i, r in enumerate(rep["per_category"]):
            cat_rows.append([ui.CATEGORY_LABEL.get(r["category"], r["category"]), str(r["support"]),
                             ui.pct(r["precision"]), ui.pct(r["recall"]), ui.pct(r["f1"])])
            for j, key in ((2, "precision"), (3, "recall")):
                if r[key] is not None and r[key] < 0.6:
                    weak.add((i, j))
        st.markdown(ui.table(["Category", "Notes", "Precision", "Recall", "F1"], cat_rows,
                             num_cols={1, 2, 3, 4}, weak=weak), unsafe_allow_html=True)

        sec, ra, bd = rep["secondary"], rep["readiness_accuracy"], rep["board"]
        st.markdown(f"""
<div class="note" style="margin-top:18px">
<b>Also measured.</b> Medical readiness {ui.pct(ra['stated_only'])} where a note states it ·
ready-but-blocked flag precision {ui.pct(bd['ready_but_blocked']['precision'])}, recall
{ui.pct(bd['ready_but_blocked']['recall'])} · owner {ui.pct(sec['owner'])} and escalation
{ui.pct(sec['escalation'])} (shown on cards, don't gate posting) ·
{ui.money(1000 * u['cost_usd'] / max(u['chunks'], 1))} per 1,000 classifications.<br>
<b>Read with care.</b> These are 100% synthetic, template-written notes; real documentation is
messier and will score lower. Red cells are the known weak spots.
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Auto-play: one simulated day per tick
# --------------------------------------------------------------------------- #

if ss.autoplay and ss.cursor < N:
    @st.fragment(run_every=AUTOPLAY_SECONDS)
    def _autoplay_tick():
        if time.time() - ss.last_tick >= AUTOPLAY_SECONDS - 0.25:
            ss.last_tick = time.time()
            advance("day")
            st.rerun()
    _autoplay_tick()
