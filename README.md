# Discharge-Barrier Radar

A proof of concept that reads hospital case-manager notes and secure-chat messages
as they arrive, classifies **discharge barriers** with **Jev** (TypeSafe AI's typed
decision model), and shows a live board of patients who are **medically ready but
still can't leave — and why**.

> **Synthetic data only.** Every patient, note, facility, payer and vendor in this
> repository is invented. No real patient data is used anywhere. This is a portfolio
> demo for an IT consulting engagement, not a production or clinical system.

Product requirements: [Discharge-Barrier Radar — PRD](https://claude.ai/code/artifact/2527c9d9-6da0-4615-ae3a-3f065103d19f)

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [The board (UI)](#the-board-ui)
- [Evaluation results](#evaluation-results)
- [Design rules and how they are enforced](#design-rules-and-how-they-are-enforced)
- [Data files](#data-files)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations and next steps](#limitations-and-next-steps)

---

## What it does

```
notes & chat ──► change detection ──► Jev relevance gate ──► Jev classification ──► ledger ──► confidence router
                 (drops copy-forward)   (1 question)   no ─► skipped   (5 questions, 1 call)   (plain Python:     │
                                                                                               age, new/ongoing) │
                                              ┌───────────────────────────────────────────────────────────────────┘
                                              ├── ≥ 0.9     ► auto-posted to the board
                                              ├── 0.5–0.9   ► on the board, visibly "needs confirmation"
                                              └── < 0.5     ► review queue (never a live board entry)
```

For every patient it tracks each barrier's **category**, **status**, **owner**,
**first seen / last seen**, **age**, **escalation need** and **confidence**, and it
keeps the exact note text behind each entry so anyone can audit why the tool said
what it said.

Barrier categories: `post_acute_placement`, `payer_authorization`,
`equipment_services`, `family_caregiver`, `transport`, `clinical_pending`,
`legal_capacity`, `social_housing`, `other`, `none`.

---

## Quick start

Requires Python 3.11+ and an [OpenRouter](https://openrouter.ai) API key with credit
(a full replay of the corpus costs about 2 cents).

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. API key
copy .env.example .env           # macOS/Linux: cp .env.example .env
#    then edit .env:  OPENROUTER_API_KEY=sk-or-v1-...

# 3. Synthetic corpus (already committed in data/; re-run to regenerate)
python corpus.py

# 4. Evaluate the full pipeline against the answer key (~560 Jev calls, ~2¢)
python eval.py

# 5. Open the board
streamlit run app.py
```

Useful extras:

| Command | What it does |
|---|---|
| `python corpus.py --show P-1003` | Print one patient's timeline with its ground-truth labels |
| `python corpus.py --seed 7` | Generate a different (still deterministic) corpus |
| `python chunker.py` | Self-test the change detector on the corpus's copy-forward cases (no API calls) |
| `python jev_client.py "Family can now pick up Thursday."` | Classify one piece of text and print every answer |
| `python jev_client.py --samples` | Classify a handful of hard cases from the corpus |
| `python pipeline.py` | Run the whole corpus and print the resulting board in the terminal |
| `python eval.py --compare` | Evaluate sentence-level and chunk-level classification side by side |
| `http://localhost:8501/?replay=end` | Open the app with the whole stay already replayed (makes Jev calls) |
| `http://localhost:8501/?theme=dark` | Open the app in dark mode (combine with `&replay=end`) |
| `pytest -q` | Run the test suite (offline, no API calls) |

---

## How it works

The build followed six stages; each is one module.

### 1. Synthetic corpus — `corpus.py`

Generates **40 patients**, 3–6 days each, **214 snippets** (case-manager notes and
secure-chat messages) from hand-written templates. Each patient gets a scripted
storyline: which barriers appear, the day each is raised / resolved / gets stuck,
and when the patient becomes medically ready. Generation is seeded and deterministic.

Hard cases are deliberately spread across the corpus and tagged in the answer key:

| Tag | Example | Snippets |
|---|---|---|
| `negation` | "Family can now pick up Thursday." / "SNF confirmed, transport arranged." | 22 |
| `copy_forward` | Yesterday's sentence pasted verbatim into today's note | 18 |
| `hedged` | "May need SNF depending on PT eval tomorrow." | 14 |
| `multi_barrier` | Two barrier categories in one note | 20 |
| `chat_only` | A barrier that only ever appears in secure chat | 20 |
| `stuck` | "Three HH agencies declined… no agency accepting." | 16 |
| `no_barrier` | Routine care, or "no needs identified" (false-positive check) | 86 |

The clean text and the answer key are written to **separate files**
(`corpus.jsonl` vs `ground_truth.jsonl`). The pipeline only ever calls
`load_corpus()`; `load_ground_truth()` is used by `eval.py` and the chunker
self-test and nowhere else.

### 2. Change-detection chunker — `chunker.py`

Clinical notes are copy-forwarded constantly. The chunker keeps a per-patient memory
of every sentence already seen and emits **only new or changed sentences**:

- Notes are split into sentences and compared against *all* of the patient's earlier
  notes (text copied from two notes back is still stale).
- Comparison is on normalised text (case, whitespace, punctuation) with a
  near-duplicate ratio of `0.93`, so trivial re-typing counts as the same sentence
  but a real edit ("pending" → "approved") is sent on.
- Chat messages are one chunk each; only an exact repeat is dropped.
- Dropped sentences are kept on the chunk for the audit view.

Self-test result: **100%** of copy-forward barrier sentences suppressed, **100%** of
new barrier sentences kept, **100%** of repeated note headers suppressed; 27% of all
sentences never reach Jev.

### 3. Jev questions and client — `questions.py`, `jev_client.py`

`questions.py` is the **only** place a Jev question or a confidence threshold is
defined. Two question sets:

| Call | Question | Type | Labels |
|---|---|---|---|
| Gate | `is_discharge_related` | Noul | yes / no |
| Classify | `barrier_category` | Choice | the 10 categories above |
| Classify | `barrier_status` | Choice | newly_raised / ongoing / resolved / not_applicable |
| Classify | `medical_readiness` | Choice | ready / not_ready / not_stated |
| Classify | `barrier_owner` | Choice | case_management / physician_team / family / payer / external_facility / vendor / not_stated |
| Classify | `escalation_need` | Score (3 levels) | routine / delayed / stuck — each worded as a situation |

`JevClient.classify(text)` is the **only** code that calls the SDK. It runs the gate
first and stops if the text isn't discharge-related; otherwise it asks all five
classification questions in **one batched call** (Jev answers them in parallel).
It returns a typed `JevResult` with every label, its confidence, the raw
probabilities, and an overall confidence:

- Overall confidence = the **weakest** of: gate probability, category confidence,
  and status confidence (one shaky answer is enough to hold an entry back).
- Status confidence is measured on the decision the ledger actually uses —
  resolved vs not resolved — so a split between "newly_raised" and "ongoing" doesn't
  lower it.
- When the category is `none`, status is left out (there is no barrier to describe).

Every call's latency, tokens and **billed cost** (OpenRouter's `usage.cost`, falling
back to list price) is recorded on `jev.meter` for the live counter and appended to
`logs/jev_calls.jsonl`. API errors never raise out of `classify()`; the result
carries the error and zero confidence so the router sends it to review.

### 4. Ledger and router — `ledger.py`, `router.py`

The **router** maps confidence to a route. Thresholds are named constants at the top
of `router.py`, bound from `questions.py`:

| Confidence | Route | Where it appears |
|---|---|---|
| ≥ 0.9 | `AUTO_POST` | Board, solid "Auto-posted" badge |
| 0.5 – 0.9 | `NEEDS_CONFIRMATION` | Board, dashed/hatched "Unverified" row with Confirm / Dismiss |
| < 0.5, or an API error | `REVIEW_QUEUE` | Review queue only — never the board |

The **ledger** turns (chunk, JevResult) pairs into per-patient state. Jev says what the
text is about; everything needing memory or arithmetic is plain Python:

- **New vs ongoing** comes from history, not wording. Jev's status is used to detect
  resolutions.
- **Barrier age** is `dateutil` date arithmetic, measured to each patient's own latest
  note (so patients admitted weeks apart are comparable).
- A resolution with only medium confidence stays on the board as "Resolved? — needs
  confirmation"; dismissing it reopens the barrier.
- A barrier first heard of as already resolved is recorded but kept off the board.
- The badge shows the confidence of the reading that earned the verification, not a
  weaker later mention.
- Storage sits behind a small `LedgerStore` protocol; `InMemoryLedgerStore` is the POC
  implementation and a database-backed store can replace it without touching ledger logic.

### 5. Pipeline and evaluation — `pipeline.py`, `eval.py`

`Pipeline` wires chunker → Jev → ledger. `step(snippet)` handles one note at a time
(the UI's replay); `run(snippets)` chunks in order, classifies in parallel (8 threads),
then applies results to the ledger in time order.

Classification granularity defaults to **per sentence**: a Choice answer can name only
one category, so classifying each new sentence separately is what lets a note with two
barriers produce two entries. Measured on the corpus before prompt tuning, per-sentence
scored 86.4% vs 84.1% overall and 60% vs 0% on multi-barrier notes, for ~40% more calls.

`eval.py` runs the real pipeline and grades it against the answer key:

- Per-category precision / recall (note level).
- **Hard-case accuracy reported on its own** (negation, copy-forward, hedged), plus per tag.
- A **calibration table**: every classified unit bucketed by confidence (0.9–1.0,
  0.7–0.9, 0.5–0.7, < 0.5) with actual accuracy, and accuracy by route.
- Board-level precision / recall of final open barriers and of the "ready but blocked" flag.
- Total tokens, cost, average latency per call.

It prints a terminal report and writes `data/eval_report.json`, which the app's
Evaluation tab reads.

### 6. Streamlit UI — `app.py`, `ui.py`

See [The board (UI)](#the-board-ui).

---

## Project layout

```
.
├── app.py              Streamlit app: layout, session state, replay, tabs
├── ui.py               Palette, CSS, HTML builders, Plotly charts (presentation only)
├── corpus.py           Stage 1 – synthetic corpus generator + loaders
├── chunker.py          Stage 2 – change-detection chunker (+ self-test)
├── questions.py        Stage 3 – every Jev question and every threshold
├── jev_client.py       Stage 3 – the only Jev/SDK wrapper; usage meter; call log
├── router.py           Stage 4 – confidence → route
├── ledger.py           Stage 4 – per-patient barrier ledger + storage interface
├── pipeline.py         chunker → Jev → ledger, shared by eval and app
├── eval.py             Stage 5 – evaluation harness and report
├── data/
│   ├── corpus.jsonl        clean text the pipeline reads
│   ├── ground_truth.jsonl  answer key (eval only)
│   ├── patients.json       non-label metadata (unit, room, admit date)
│   └── eval_report.json    latest evaluation, shown in the app
├── logs/jev_calls.jsonl    one line per Jev call (git-ignored)
├── tests/                  pytest suite (offline; Jev is mocked)
├── .streamlit/config.toml  base light theme, Inter font (dark mode is applied by ui.py)
├── requirements.txt
└── .env.example
```

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| API key | `OPENROUTER_API_KEY` in the shell, else `.env` | — (required) |
| Jev endpoint | `BASE_URL` in `jev_client.py` | `https://openrouter.ai/api` |
| Model | `MODEL` in `jev_client.py` | `jev-1.13` (pinned for reproducible eval; `jev-latest` tracks releases) |
| Fallback price | `PRICE_PER_MILLION_INPUT_TOKENS` in `jev_client.py` | `0.042` (used only if a response has no `usage.cost`) |
| Relevance gate | `RELEVANCE_THRESHOLD` in `questions.py` | `0.5` |
| Auto-post threshold | `AUTO_POST_THRESHOLD` in `questions.py` | `0.9` |
| Confirm threshold | `CONFIRM_THRESHOLD` in `questions.py` | `0.5` (see note below) |
| Readiness update floor | `READINESS_MIN_CONFIDENCE` in `questions.py` | `0.7` |
| Copy-forward similarity | `NEAR_DUPLICATE_RATIO` in `chunker.py` | `0.93` |
| Granularity | `DEFAULT_GRANULARITY` in `pipeline.py` | `sentence` |
| Parallel Jev calls | `MAX_PARALLEL_CALLS` in `pipeline.py` | `8` |
| Auto-play speed | `AUTOPLAY_SECONDS` in `app.py` | `3` (one simulated day per tick) |

**On the confirm threshold:** calibration shows readings between 0.5 and 0.7 are right
only ~44% of the time. Raising `CONFIRM_THRESHOLD` to `0.7` would move them to the
review queue and make "needs confirmation" entries right ~92% of the time. It is left at
`0.5` because that is what the original brief specifies.

---

## The board (UI)

`streamlit run app.py`, then use the sidebar to replay the synthetic stays.

- **Top bar** — product name, simulated clock, and a quiet live counter:
  `360 classifications · $0.0184 · avg 151 ms`.
- **KPI tiles** — *Medically ready, still blocked* (the lead tile), open barriers,
  needs confirmation, barrier-days open, review queue.
- **Board** — one card per patient, ready-but-blocked first (dark left edge), then
  longest-blocked. Each barrier row shows its category (colour by family), age, owner,
  escalation ("⚠ Stuck — escalate") and a confidence badge:
  - **Auto-posted** — solid dark pill.
  - **Needs confirmation** — dashed amber outline, hatched background, italic category,
    "UNVERIFIED" tag, "?" icon, with one-click **Confirm** / **Dismiss**.
  - **Confirmed** — green pill.
- **"Why? Show the source text"** — expands each card to the original note: the sentence
  Jev classified is highlighted, copy-forward text the chunker ignored is struck through,
  and Jev's reading (category, status, confidence, route) is shown underneath.
- **Where time is lost** — barrier-days by category, a patient funnel (seen → with an open
  barrier → medically ready → ready but blocked), and who holds the open barriers; with a
  table view.
- **Review queue** — low-confidence readings and failed calls, with **Accept to board** /
  **Reject**.
- **Evaluation** — hard-case accuracy, overall accuracy, auto-posted accuracy, board recall,
  cost; the calibration chart and table; hard cases and categories with weak cells in red;
  a **Re-run evaluation live** button.
- **Dark mode** — a toggle pinned top-right switches the whole app, including the charts,
  between light and dark. The choice is kept in the URL (`?theme=dark`), so a reload or a
  shared link opens in the same theme, and it doesn't disturb the replay, the ledger or any
  confirmations already made.
- **Simulation** — Next note, Next day, Run to end, Reset, and auto-play. Every step runs
  the real pipeline, so the board, counter and review queue update live.
- **Errors** — if Jev calls fail, a red banner shows the API error instead of the board
  silently emptying.

Colour: three category families plus neutral gray (Placement & payer · Home, family &
logistics · Clinical & legal · Other), validated for colour-blind separation. Category
colour is always paired with a text label. "Needs confirmation" uses a reserved warning
colour and a different shape, so it never reads as a category or as a confirmed entry.

Both themes come from one token set in `ui.py` (`THEMES`). The HTML reads the tokens as
CSS variables and the Plotly charts take them directly, so nothing is hardcoded per theme.
Dark mode uses the dark steps of the same palette, validated against the dark surface, and
keeps the same shape cues for "needs confirmation" (dashed outline, hatched row, "?" icon).

---

## Evaluation results

Latest run: `jev-1.13`, sentence-level, 214 snippets, 360 classified units.

| Measure | Result |
|---|---|
| Note-level accuracy, overall | **89.7%** (192/214) |
| **Hard cases** (negation, copy-forward, hedged) | **79.2%** (42/53) |
| Auto-posted entries that were correct | **100%** (243/243) |
| Board: open barriers — precision / recall | 78.3% / 97.3% |
| Board: ready-but-blocked — precision / recall | 89.5% / 100% |
| Medical readiness, where the note states it | 92.9% |
| Cost | $0.0184 for 557 calls (~$0.05 per 1,000 classifications), avg 140 ms/call |

Hard cases by tag: copy-forward 100% · chat-only 100% · no barrier 98.8% · stuck 87.5% ·
hedged 78.6% · multi-barrier 65.0% · negation 63.6%.

Calibration:

| Confidence | Units | Mean confidence | Actual accuracy |
|---|---|---|---|
| 0.9 – 1.0 | 243 | 94.7% | 100.0% |
| 0.7 – 0.9 | 74 | 83.1% | 91.9% |
| 0.5 – 0.7 | 39 | 59.2% | 43.6% |
| < 0.5 | 4 | 41.0% | 25.0% |

Known weak spots: negation (some resolutions sit in a sentence that loses context when
split, e.g. "Clear to transfer."), clinical-pending recall (52.6%: "Echo done, read
pending" reads as routine status), owner (55%) and escalation (64%) labels, and the
0.5–0.7 confidence band.

**Read with care:** these numbers come from template-written synthetic notes and were used
to tune the question wording over two rounds. Real documentation is messier and will score
lower.

---

## Design rules and how they are enforced

| Rule | Enforcement |
|---|---|
| Jev only classifies text — never arithmetic, counting or date comparison | All ages, counts, sorting and new-vs-ongoing logic live in `ledger.py` / `eval.py` / `app.py` (plain Python, `dateutil`) |
| Every Jev question and threshold lives in `questions.py` | `tests/test_jev_client.py::test_only_questions_py_defines_questions` fails if `Choice(`, `Noul(` or `Score(` appears in any other module |
| Every Jev call goes through one wrapper | `test_only_wrapper_touches_the_sdk_client` fails if any other module references `TypeSafeClient` or `system_one` |
| Log every call's latency and token usage | `UsageMeter` + `logs/jev_calls.jsonl`; thread-safe writes are tested |
| The pipeline never reads its answer key | Separate files; only `eval.py` (and the chunker self-test) call `load_ground_truth()` |
| Unverified must never look confirmed | Separate badge shapes and row styling; < 0.5 never reaches the board |

---

## Data files

**`data/corpus.jsonl`** — one clean snippet per line:

```json
{"snippet_id": "P-1003-d1-note", "patient_id": "P-1003", "timestamp": "2026-09-12T09:14",
 "source": "note", "author_role": "case_manager",
 "text": "79M adm 9/11 w/ urosepsis. Auth approved by Keystone Medicaid MCO this afternoon. ..."}
```

**`data/ground_truth.jsonl`** — the answer key, keyed by `snippet_id`: `is_discharge_related`,
`medical_readiness`, `primary_category`, `barriers[]` (category, status, owner, escalation,
text, tags), `stale_barriers[]` (copy-forward text, with `copied_from`), `hard_case_tags[]`,
and per-sentence `segments[]`.

**`logs/jev_calls.jsonl`** — one line per Jev call: timestamp, purpose (`gate` / `classify`),
ref (snippet id), latency_ms, input/output tokens, cost_usd, ok, error.

---

## Testing

```bash
pytest -q
```

39 offline tests (Jev is replaced by an `httpx2.MockTransport`):

- `test_chunker.py` — copy-forward, near-duplicates, changed sentences, per-patient isolation, chat handling, sentence splitting.
- `test_jev_client.py` — gate short-circuit, single batched call, confidence rules, cost metering (billed and fallback), OpenRouter URL and pinned model, error capture, concurrent log writes, and the two hard-constraint guards.
- `test_ledger.py` — router thresholds, new/ongoing from history, date-math ages, resolution/reopen, confirm/dismiss, review queue, readiness floor, board ordering, badge confidence.

`python chunker.py` is an additional self-test against the corpus's copy-forward cases.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OPENROUTER_API_KEY is not set` | Add it to `.env` (see `.env.example`) or export it in your shell |
| Red banner: `402 Insufficient credits` | The OpenRouter account behind the key needs credit — openrouter.ai/settings/credits |
| Everything lands in the review queue | Jev calls are failing; the banner and `logs/jev_calls.jsonl` show the error |
| Garbled `·` / `—` in terminal output on Windows | CLIs reconfigure stdout to UTF-8; if piping, set `PYTHONIOENCODING=utf-8` |
| `client.models.list()` fails | Expected on OpenRouter (it returns OpenRouter's model shape); the pipeline doesn't use it |

---

## Limitations and next steps

- **Synthetic, template-based corpus.** Results will not transfer one-to-one to real notes;
  a shadow pilot graded by case managers is the next proof point.
- **Sentence splitting loses context** for some resolutions; passing the neighbouring
  sentence as context is the likely fix.
- **In-memory ledger.** State resets when the app restarts; swap `InMemoryLedgerStore` for a
  database-backed `LedgerStore`.
- **Dark mode re-skins Streamlit's own widgets with CSS** (buttons, tabs, progress bar,
  toggles) because Streamlit can't switch its theme at runtime. Those selectors are tuned to
  Streamlit 1.6x's DOM and may need a touch-up after a major Streamlit upgrade.
- **Before any real data:** BAA with the model provider, zero-retention settings, access
  control and a privacy review. The product is read-only by design — it never writes to the
  EHR or contacts anyone.

See the [PRD](https://claude.ai/code/artifact/2527c9d9-6da0-4615-ae3a-3f065103d19f) for goals,
users, requirements, success metrics and the pilot roadmap.
