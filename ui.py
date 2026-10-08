"""
Presentation layer for app.py: palette, CSS, HTML builders and Plotly charts.
No pipeline logic lives here — only how things look.

Palette: three category families plus neutral gray (validated with the dataviz
palette checker: all-pairs CVD dE >= 9.2, normal-vision dE >= 24; aqua is below
3:1 on the surface, so category colour is always paired with a text label).
"Needs confirmation" uses the reserved status-warning colour with a distinct
shape (dashed outline, hatched fill, "?" icon + label) so it never reads as a
category or as a confirmed entry.
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
from dateutil.parser import isoparse

from chunker import normalize, split_sentences

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

# Every colour comes from one of two token sets. HTML/CSS reads them as CSS variables
# (see theme_css), so the whole board re-themes without rebuilding any markup; Plotly
# needs literal values, so charts take the token dict directly.
THEMES = {
    "light": dict(
        scheme="light", ink="#111418", ink_2="#52514e", muted="#898781", line="#e6e4dd",
        surface="#ffffff", plane="#f6f6f3", on_ink="#ffffff", ink_hover="#2b2f36",
        hero_label="#c3c2b7", warn="#fab219", warn_ink="#6b4400", warn_bg="#fff7e0",
        hatch_a="#fffdf6", hatch_b="#fff8e6", critical="#d03b3b", good="#0ca30c",
        good_ink="#0b5d0b", good_bg="#e8f6e8", good_line="#bfe3bf", tint="#eef3fb",
        tint_line="#9ec5f4", chip="#efeee9", th_bg="#fafaf8", row_line="#f0efea",
        strike="#c3c2b7", grid="#eeede7", track="#c3c2b7",
        fam_1="#2a78d6", fam_2="#eb6834", fam_3="#1baf7a", fam_4="#898781",
        funnel=("#86b6ef", "#5598e7", "#2a78d6", "#111418"),
        funnel_text=("#111418", "#111418", "#ffffff", "#ffffff")),
    "dark": dict(
        scheme="dark", ink="#f3f2ee", ink_2="#c3c2b7", muted="#898781", line="#2c2c2a",
        surface="#1a1a19", plane="#0d0d0d", on_ink="#111418", ink_hover="#d9d8d2",
        hero_label="#52514e", warn="#fab219", warn_ink="#fbd27a", warn_bg="#2b2410",
        hatch_a="#1f1b10", hatch_b="#272113", critical="#e66767", good="#0ca30c",
        good_ink="#6fd36f", good_bg="#12301a", good_line="#1f5a2b", tint="#17263a",
        tint_line="#3987e5", chip="#262624", th_bg="#212120", row_line="#262624",
        strike="#5a5953", grid="#2c2c2a", track="#4a4a46",
        fam_1="#3987e5", fam_2="#d95926", fam_3="#199e70", fam_4="#898781",
        funnel=("#184f95", "#256abf", "#3987e5", "#f3f2ee"),
        funnel_text=("#ffffff", "#ffffff", "#ffffff", "#111418")),
}


def tokens(dark: bool) -> dict:
    return THEMES["dark" if dark else "light"]


FAMILIES = {                       # name -> (CSS colour variable, categories)
    "Placement & payer": ("var(--fam-1)", ("post_acute_placement", "payer_authorization")),
    "Home, family & logistics": ("var(--fam-2)", ("equipment_services", "family_caregiver",
                                                  "transport", "social_housing")),
    "Clinical & legal": ("var(--fam-3)", ("clinical_pending", "legal_capacity")),
    "Other": ("var(--fam-4)", ("other",)),
}
_FAMILY_TOKEN = dict(zip(FAMILIES, ("fam_1", "fam_2", "fam_3", "fam_4")))
FAMILY_OF = {c: f for f, (_, cats) in FAMILIES.items() for c in cats}
COLOR_OF = {c: FAMILIES[f][0] for c, f in FAMILY_OF.items()}      # CSS variables (HTML)


def chart_color(category: str, dark: bool) -> str:
    """Literal hex for a category's family colour (Plotly can't read CSS variables)."""
    return tokens(dark)[_FAMILY_TOKEN[FAMILY_OF.get(category, "Other")]]

CATEGORY_LABEL = {
    "post_acute_placement": "Post-acute placement", "payer_authorization": "Payer authorization",
    "equipment_services": "Equipment & services", "family_caregiver": "Family / caregiver",
    "transport": "Transport", "clinical_pending": "Clinical pending",
    "legal_capacity": "Legal / capacity", "social_housing": "Social / housing",
    "other": "Other", "none": "No barrier",
}
OWNER_LABEL = {
    "case_management": "Case management", "physician_team": "Physician team",
    "family": "Family", "payer": "Payer", "external_facility": "Receiving facility",
    "vendor": "Vendor", "not_stated": "Not stated",
}
ROLE_LABEL = {
    "case_manager": "Case manager", "rn": "RN", "charge_rn": "Charge RN", "rt": "RT",
    "pt_ot": "PT/OT", "hospitalist": "Hospitalist", "social_worker": "Social worker",
    "utilization_review": "Utilization review", "psychiatry": "Psychiatry",
    "pharmacy": "Pharmacy",
}

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}}
.stApp {{ background: var(--plane); }}
header[data-testid="stHeader"] {{ display:none; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1320px; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
section[data-testid="stSidebar"] button p {{ font-size: .84rem; white-space: nowrap; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

/* ---------- top bar ---------- */
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px;
  flex-wrap:wrap; padding: 4px 2px 14px; border-bottom:1px solid var(--line); margin-bottom:18px; }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:650; font-size:1.12rem;
  color:var(--ink); letter-spacing:-0.01em; }}
.brand .mark {{ width:26px; height:26px; border-radius:8px; background:var(--ink);
  display:inline-flex; align-items:center; justify-content:center; color:var(--on-ink); font-size:14px; }}
.brand .tag {{ font-weight:500; font-size:.72rem; color:var(--muted); border:1px solid var(--line);
  border-radius:999px; padding:2px 8px; letter-spacing:0; }}
.topbar-right {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.clock {{ font-size:.8rem; color:var(--ink-2); }}
.meter {{ font-size:.78rem; color:var(--ink-2); font-variant-numeric: tabular-nums;
  background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:5px 12px;
  display:inline-flex; align-items:center; gap:8px; }}
.meter .live {{ width:7px; height:7px; border-radius:50%; background:var(--good); }}
.meter .idle {{ width:7px; height:7px; border-radius:50%; background:var(--line); }}
.meter b {{ color:var(--ink); font-weight:600; }}

/* ---------- KPI tiles ---------- */
.kpis {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
.kpi .label {{ font-size:.74rem; color:var(--ink-2); font-weight:500; }}
.kpi .value {{ font-size:1.7rem; font-weight:650; color:var(--ink); letter-spacing:-0.02em; line-height:1.2; margin-top:4px; }}
.kpi .sub {{ font-size:.72rem; color:var(--muted); margin-top:2px; }}
.kpi.hero {{ background:var(--ink); border-color:var(--ink); }}
.kpi.hero .label, .kpi.hero .sub {{ color:var(--hero-label); }}
.kpi.hero .value {{ color:var(--on-ink); }}

/* ---------- patient cards ---------- */
div[class*="st-key-card"] {{ background:var(--surface); border:1px solid var(--line);
  border-radius:16px; padding:16px 18px 6px; margin-bottom:14px; gap:.45rem; }}
div[class*="st-key-card"].blocked, div[class*="st-key-cardrb_"] {{ box-shadow: inset 4px 0 0 var(--ink); }}
.phead {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }}
.pid {{ font-weight:650; font-size:1.02rem; color:var(--ink); letter-spacing:-0.01em; }}
.loc {{ font-size:.76rem; color:var(--muted); margin-left:6px; }}
.newdot {{ display:inline-block; font-size:.66rem; font-weight:600; color:var(--ink); background:var(--tint);
  border-radius:999px; padding:1px 7px; margin-left:6px; vertical-align:2px; }}
.ready {{ font-size:.72rem; font-weight:600; border-radius:999px; padding:3px 9px; white-space:nowrap; }}
.ready.yes {{ color:var(--good-ink); background:var(--good-bg); }}
.ready.no  {{ color:var(--ink-2); background:var(--chip); }}
.ready.unk {{ color:var(--muted); background:transparent; border:1px solid var(--line); }}
.blockline {{ font-size:.8rem; color:var(--ink-2); margin:6px 0 4px; }}
.blockline b {{ color:var(--ink); }}

/* barrier rows */
.brow {{ display:grid; grid-template-columns: 1fr auto; gap:4px 12px; align-items:center;
  border:1px solid var(--line); border-radius:10px; padding:9px 11px; }}
.brow .main {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.brow .meta {{ grid-column: 1 / -1; font-size:.74rem; color:var(--ink-2); display:flex; gap:12px; flex-wrap:wrap; }}
.cat {{ display:inline-flex; align-items:center; gap:7px; font-weight:600; font-size:.86rem; color:var(--ink); }}
.cat i {{ width:10px; height:10px; border-radius:3px; background:var(--c); display:inline-block; flex:none; }}
.age {{ font-variant-numeric: tabular-nums; font-weight:600; color:var(--ink); font-size:.82rem; }}
.esc-stuck {{ color:var(--critical); font-weight:600; }}
.esc-delayed {{ color:var(--ink-2); }}

/* confidence badges — shape AND colour differ */
.badge {{ font-size:.7rem; font-weight:600; border-radius:999px; padding:3px 9px; white-space:nowrap;
  display:inline-flex; align-items:center; gap:5px; font-variant-numeric: tabular-nums; }}
.badge.auto {{ background:var(--ink); color:var(--on-ink); }}
.badge.confirmed {{ background:var(--good-bg); color:var(--good-ink); border:1px solid var(--good-line); }}
.badge.unverified {{ background:var(--warn-bg); color:var(--warn-ink); border:1.5px dashed var(--warn); border-radius:6px; }}

/* the unverified row itself is unmistakable */
.brow.unverified {{ border:1.5px dashed var(--warn);
  background: repeating-linear-gradient(135deg, var(--hatch-a) 0 8px, var(--hatch-b) 8px 16px); }}
.brow.unverified .cat {{ font-style: italic; }}
.brow.unverified .flag {{ font-size:.7rem; font-weight:700; color:var(--warn-ink); letter-spacing:.04em; text-transform:uppercase; }}
.brow.resolving .cat span {{ text-decoration: line-through; text-decoration-color: var(--muted); }}

/* compact confirm / dismiss buttons */
div[class*="st-key-ok_"] button, div[class*="st-key-no_"] button,
div[class*="st-key-acc_"] button, div[class*="st-key-rej_"] button {{
  min-height: 0; padding: 4px 10px; border-radius: 8px; font-size: .76rem; font-weight: 600; white-space: nowrap; }}
div[class*="st-key-ok_"], div[class*="st-key-no_"] {{ margin-top: 2px; margin-bottom: 6px; }}
div[class*="st-key-ok_"] button, div[class*="st-key-acc_"] button {{ background: var(--ink); color:var(--on-ink); border-color: var(--ink); }}
div[class*="st-key-ok_"] button:hover, div[class*="st-key-acc_"] button:hover {{ background:var(--ink-hover); color:var(--on-ink); border-color:var(--ink-hover); }}
div[class*="st-key-no_"] button, div[class*="st-key-rej_"] button {{ background: var(--surface); color:var(--ink-2); }}
div[class*="st-key-ok_"] button p, div[class*="st-key-acc_"] button p {{ color: var(--on-ink) !important; }}
div[class*="st-key-no_"] button p, div[class*="st-key-rej_"] button p {{ color: var(--ink-2) !important; }}

/* evidence expander */
div[class*="st-key-card"] details {{ border:none !important; border-top:1px solid var(--line) !important;
  border-radius:0 !important; margin-top:4px; }}
div[class*="st-key-card"] details summary {{ padding-left:0 !important; }}
div[class*="st-key-card"] details summary p {{ font-size:.78rem !important; color:var(--ink-2); font-weight:500; }}
.ev {{ border-left:3px solid var(--c, var(--line)); padding:2px 0 2px 12px; margin:6px 0 14px; }}
.ev-cat {{ font-size:.78rem; font-weight:650; color:var(--ink); margin-bottom:6px; }}
.ev-head {{ font-size:.72rem; color:var(--muted); margin-bottom:4px; }}
.ev-text {{ font-size:.84rem; color:var(--ink-2); line-height:1.5; }}
.ev-text mark {{ background:var(--tint); color:var(--ink); padding:1px 3px; border-radius:4px;
  box-shadow: inset 0 -2px 0 var(--tint-line); }}
.ev-text .stale {{ color:var(--muted); text-decoration: line-through; text-decoration-color:var(--strike); }}
.ev-read {{ font-size:.72rem; color:var(--ink-2); margin-top:4px; }}

/* clear patients */
.clearlist {{ display:flex; flex-wrap:wrap; gap:8px; }}
.clearchip {{ background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:4px 10px;
  font-size:.76rem; color:var(--ink-2); }}
.clearchip b {{ color:var(--ink); font-weight:600; }}

/* section headings */
.sect {{ font-size:.78rem; font-weight:650; color:var(--ink-2); text-transform:uppercase;
  letter-spacing:.06em; margin: 8px 0 10px; }}
.hint {{ font-size:.8rem; color:var(--muted); }}

/* legend */
.legend {{ display:flex; flex-direction:column; gap:7px; font-size:.78rem; color:var(--ink-2); }}
.legend .row {{ display:flex; align-items:center; gap:8px; }}
.legend i {{ width:10px; height:10px; border-radius:3px; background:var(--c); display:inline-block; }}

/* activity feed */
.feed {{ display:flex; flex-direction:column; gap:8px; }}
.feed .item {{ font-size:.75rem; color:var(--ink-2); line-height:1.35; border-left:2px solid var(--c, var(--line)); padding-left:8px; }}
.feed .item b {{ color:var(--ink); font-weight:600; }}

/* empty state */
.empty {{ background:var(--surface); border:1px dashed var(--line); border-radius:16px; padding:40px 28px; text-align:center; }}
.empty h3 {{ margin:0 0 6px; font-size:1.1rem; color:var(--ink); font-weight:650; }}
.empty p {{ margin:0; color:var(--ink-2); font-size:.88rem; }}

/* eval tables */
table.t {{ width:100%; border-collapse:collapse; font-size:.82rem; background:var(--surface);
  border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
table.t th {{ text-align:left; font-weight:600; color:var(--ink-2); font-size:.74rem; padding:9px 12px;
  border-bottom:1px solid var(--line); background:var(--th-bg); }}
table.t td {{ padding:8px 12px; border-bottom:1px solid var(--row-line); color:var(--ink); font-variant-numeric: tabular-nums; }}
table.t tr:last-child td {{ border-bottom:none; }}
table.t td.num, table.t th.num {{ text-align:right; }}
table.t td.weak {{ color:var(--critical); font-weight:600; }}
.note {{ font-size:.78rem; color:var(--ink-2); background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:12px 14px; line-height:1.5; }}

/* tabs */
button[data-baseweb="tab"] p {{ font-weight:600; font-size:.88rem; }}

/* theme toggle: pinned top-right, on its own strip above the top bar */
div[class*="st-key-dark_mode"] {{ position: fixed; top: 12px; right: 28px; z-index: 1000;
  width: auto !important; }}
div[class*="st-key-dark_mode"] label p {{ font-size:.8rem; font-weight:600; color:var(--ink-2); }}
</style>
"""


def theme_css(dark: bool) -> str:
    """The CSS variables for one theme, plus (dark only) the overrides that re-skin
    Streamlit's own widgets, which otherwise follow the light theme in config.toml."""
    t = tokens(dark)
    names = {"ink_2": "ink-2", "on_ink": "on-ink", "ink_hover": "ink-hover",
             "hero_label": "hero-label", "warn_ink": "warn-ink", "warn_bg": "warn-bg",
             "hatch_a": "hatch-a", "hatch_b": "hatch-b", "good_ink": "good-ink",
             "good_bg": "good-bg", "good_line": "good-line", "tint_line": "tint-line",
             "th_bg": "th-bg", "row_line": "row-line", "fam_1": "fam-1", "fam_2": "fam-2",
             "fam_3": "fam-3", "fam_4": "fam-4"}
    skip = {"scheme", "funnel", "funnel_text"}
    decls = " ".join(f"--{names.get(k, k)}:{v};" for k, v in t.items() if k not in skip)
    css = f":root {{ {decls} color-scheme:{t['scheme']}; }}"
    return f"<style>{css}{DARK_WIDGET_CSS if dark else ''}</style>"


# Streamlit's own widgets keep the light palette from config.toml; in dark mode they are
# re-skinned from the same variables. (Tuned against Streamlit 1.6x's DOM.)
DARK_WIDGET_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--plane); color: var(--ink); }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div { background: var(--surface); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--surface); color: var(--ink); border-color: var(--line); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--ink-2); color: var(--ink); background: var(--chip); }
[data-testid="stBaseButton-secondary"]:disabled { background: var(--surface); color: var(--muted); border-color: var(--line); opacity: .6; }
[data-testid="stBaseButton-primary"] { background: var(--ink); color: var(--on-ink); border-color: var(--ink); }
[data-testid="stBaseButton-primary"]:hover { background: var(--ink-hover); border-color: var(--ink-hover); color: var(--on-ink); }
[data-testid="stBaseButton-primary"]:disabled { background: var(--chip); color: var(--muted); border-color: var(--line); }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span { color: var(--on-ink); }
[data-testid="stBaseButton-secondary"] p, [data-testid="stBaseButton-secondary"] span { color: inherit; }
button[data-baseweb="tab"] { color: var(--ink-2); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink); }
button[data-baseweb="tab"] p { color: inherit; }
div[data-baseweb="tab-highlight"] { background-color: var(--ink); }
div[data-baseweb="tab-border"] { background-color: var(--line); }
[data-testid="stProgressBarTrack"] { background-color: var(--line) !important; }
[data-testid="stProgressBarTrack"] > div { background-color: var(--ink) !important; }
[data-testid="stProgress"] p { color: var(--ink-2); }
[data-testid="stExpander"] details { background: transparent; color: var(--ink); }
[data-testid="stExpander"] summary { color: var(--ink-2); }
[data-testid="stExpander"] summary:hover { color: var(--ink); }
[data-testid="stButtonGroup"] button { background: var(--surface); color: var(--ink-2); border-color: var(--line); }
[data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--chip) !important; color: var(--ink) !important; border-color: var(--ink-2) !important; }
[data-testid="stButtonGroup"] button p { color: var(--ink-2); }
[data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--ink) !important; }
[data-testid="stAlert"] { background: var(--surface); color: var(--ink); border: 1px solid var(--line); }
[data-testid="stAlert"] p { color: var(--ink); }
[data-testid="stSpinner"] p, [data-testid="stSpinner"] { color: var(--ink-2); }
.stApp code { background: var(--chip); color: var(--ink); }
[data-testid="stCheckbox"] label > div:nth-of-type(1) { background-color: var(--track) !important; }
[data-testid="stCheckbox"] label[data-selected="true"] > div:nth-of-type(1) { background-color: var(--fam-1) !important; }
[data-testid="stCheckbox"] label > div:nth-of-type(1) > div { background-color: #ffffff !important; }
"""


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def fmt_age(days: float) -> str:
    if days < 1:
        return f"{max(days * 24, 0):.0f} h"
    return f"{days:.1f} d"


def fmt_ts(ts: str | None) -> str:
    """'Tue Sep 8, 14:05'. (strftime's %-d isn't portable to Windows, hence d.day.)"""
    if not ts:
        return "—"
    d = isoparse(ts)
    return f"{d:%a} {d:%b} {d.day}, {d:%H:%M}"


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def money(x: float) -> str:
    return f"${x:.4f}" if x < 1 else f"${x:,.2f}"


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #

def topbar(meter: dict, clock: str | None, cursor: int, total: int) -> str:
    live = "live" if meter["calls"] else "idle"
    lat = f"avg {meter['avg_latency_ms']:.0f} ms" if meter["calls"] else "avg — ms"
    err = f" · {meter['errors']} errors" if meter["errors"] else ""
    return f"""
<div class="topbar">
  <div class="brand"><span class="mark">◎</span>Discharge-Barrier Radar
    <span class="tag">synthetic data · proof of concept</span></div>
  <div class="topbar-right">
    <span class="clock">Simulated time <b>{escape(fmt_ts(clock))}</b> · {cursor} / {total} notes</span>
    <span class="meter" title="Every Jev call this session: count, billed cost, mean latency">
      <span class="{live}"></span><b>{meter['chunks']:,}</b> classifications · <b>{money(meter['cost_usd'])}</b> · {lat}{err}
    </span>
  </div>
</div>"""


def kpis(tiles: list[tuple[str, str, str, bool]]) -> str:
    cells = "".join(
        f'<div class="kpi{" hero" if hero else ""}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="sub">{escape(sub)}</div></div>'
        for label, value, sub, hero in tiles)
    return f'<div class="kpis">{cells}</div>'


def card_head(p, meta: dict | None, is_new: bool) -> str:
    loc = f"{meta['unit']} · Rm {meta['room']}" if meta else ""
    if p.readiness is None:
        ready = '<span class="ready unk">Readiness not stated</span>'
    elif p.readiness.label == "ready":
        ready = '<span class="ready yes">✓ Medically ready</span>'
    else:
        ready = '<span class="ready no">Not medically ready</span>'
    new = '<span class="newdot">updated</span>' if is_new else ""
    head = (f'<div class="phead"><div><span class="pid">{escape(p.patient_id)}</span>'
            f'<span class="loc">{escape(loc)}</span>{new}</div>{ready}</div>')
    opens = p.open_barriers
    if opens:
        oldest = max(b.age_days(p.as_of) for b in opens)
        lead = "Ready but blocked" if p.ready_but_blocked else "Blocked"
        head += (f'<div class="blockline"><b>{lead}</b> · {len(opens)} open barrier'
                 f'{"s" if len(opens) != 1 else ""} · oldest {fmt_age(oldest)}</div>')
    return head


def barrier_row(b, now: str) -> str:
    color = COLOR_OF.get(b.category, "var(--muted)")
    v = b.verification.value
    if b.resolution_unconfirmed:
        cls, badge = "unverified resolving", (
            f'<span class="badge unverified">? Resolved — needs confirmation · {b.posted_confidence:.2f}</span>')
    elif v == "needs_confirmation":
        cls, badge = "unverified", (
            f'<span class="badge unverified">? Needs confirmation · {b.posted_confidence:.2f}</span>')
    elif v == "confirmed":
        cls, badge = "confirmed", '<span class="badge confirmed">✓✓ Confirmed</span>'
    else:
        cls, badge = "auto", f'<span class="badge auto">✓ Auto-posted · {b.posted_confidence:.2f}</span>'
    esc = {"stuck": '<span class="esc-stuck">⚠ Stuck — escalate</span>',
           "delayed": '<span class="esc-delayed">◷ Delayed</span>'}.get(b.escalation, "")
    flag = '<span class="flag">Unverified</span>' if "unverified" in cls else ""
    since = "resolved" if not b.is_open else "open"
    return f"""
<div class="brow {cls}">
  <div class="main"><span class="cat" style="--c:{color}"><i></i><span>{escape(CATEGORY_LABEL.get(b.category, b.category))}</span></span>{flag}</div>
  {badge}
  <div class="meta"><span class="age">{fmt_age(b.age_days(now))} {since}</span>
    <span>Owner: {escape(OWNER_LABEL.get(b.owner, b.owner))}</span>{esc}
    <span>First seen {escape(fmt_ts(b.first_seen))}</span></div>
</div>"""


def evidence(b, originals: dict[str, str]) -> str:
    """The note text behind a barrier: the sentence Jev classified is highlighted,
    copy-forward text the chunker ignored is struck through."""
    color = COLOR_OF.get(b.category, "var(--muted)")
    items = []
    for ev in reversed(b.evidence):
        original = originals.get(ev.snippet_id, ev.text)
        target = {normalize(s) for s in split_sentences(ev.text)} | {normalize(ev.text)}
        stale = {normalize(s) for s in ev.ignored_text}
        parts = []
        for s in split_sentences(original) if ev.source == "note" else [original]:
            n = normalize(s)
            if n in target:
                parts.append(f"<mark>{escape(s)}</mark>")
            elif n in stale:
                parts.append(f'<span class="stale" title="Copied forward from an earlier note — ignored">{escape(s)}</span>')
            else:
                parts.append(escape(s))
        route = {"auto_post": "auto-posted", "needs_confirmation": "needs confirmation",
                 "review_queue": "accepted from review"}[ev.route.value]
        items.append(f"""
<div class="ev-head">{escape(fmt_ts(ev.timestamp))} · {"Note" if ev.source == "note" else "Secure chat"} ·
  {escape(ROLE_LABEL.get(ev.author_role, ev.author_role))} · {escape(ev.snippet_id)}</div>
<div class="ev-text">{" ".join(parts)}</div>
<div class="ev-read">Jev read this as <b>{escape(CATEGORY_LABEL.get(b.category, b.category))}</b>,
  status “{escape(ev.status_said.replace('_', ' '))}”, confidence {ev.confidence:.2f} → {route}</div>""")
    return (f'<div class="ev" style="--c:{color}"><div class="ev-cat">'
            f'{escape(CATEGORY_LABEL.get(b.category, b.category))}</div>{"<br>".join(items)}</div>')


def legend() -> str:
    rows = "".join(f'<div class="row"><i style="--c:{c}"></i>{escape(name)}</div>'
                   for name, (c, _) in FAMILIES.items())
    badges = ('<div class="row"><span class="badge auto">✓ Auto-posted</span> confidence ≥ 0.9</div>'
              '<div class="row"><span class="badge unverified">? Needs confirmation</span> 0.5–0.9</div>'
              '<div class="row"><span class="badge confirmed">✓✓ Confirmed</span> by a person</div>'
              '<div class="row hint">Below 0.5 → review queue, never on the board</div>')
    return f'<div class="legend">{rows}<div style="height:6px"></div>{badges}</div>'


def feed(entries: list[dict]) -> str:
    if not entries:
        return '<div class="hint">Updates appear here as notes arrive.</div>'
    items = "".join(
        f'<div class="item" style="--c:{COLOR_OF.get(e["category"], "var(--muted)")}">'
        f'<b>{escape(e["patient_id"])}</b> · {escape(e["what"])}<br>'
        f'<span class="hint">{escape(fmt_ts(e["ts"]))} · {escape(e["route"])}</span></div>'
        for e in entries)
    return f'<div class="feed">{items}</div>'


def table(headers: list[str], rows: list[list[str]], num_cols: set[int] = frozenset(),
          weak: set[tuple[int, int]] = frozenset()) -> str:
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{escape(h)}</th>'
                 for i, h in enumerate(headers))
    body = ""
    for r_i, row in enumerate(rows):
        tds = "".join(
            f'<td class="{"num " if i in num_cols else ""}{"weak" if (r_i, i) in weak else ""}">{cell}</td>'
            for i, cell in enumerate(row))
        body += f"<tr>{tds}</tr>"
    return f'<table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


# --------------------------------------------------------------------------- #
# Charts (Plotly) — thin marks, rounded ends, recessive axes, direct labels
# --------------------------------------------------------------------------- #

def _layout(fig: go.Figure, height: int, t: dict, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=96, t=8, b=8),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12,
                  color=t["ink_2"]),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["line"],
                        font=dict(color=t["ink"], size=12)),
        showlegend=False, bargap=0.45, barcornerradius=4, **kw)
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=False, linecolor=t["line"],
                     tickfont=dict(color=t["muted"]))
    fig.update_yaxes(showgrid=False, linecolor=t["line"], tickfont=dict(color=t["ink_2"]))
    return fig


def time_lost_chart(rows: list[dict], dark: bool = False) -> go.Figure:
    """rows: category, barrier_days, open, patients — horizontal bars by family colour."""
    t = tokens(dark)
    rows = sorted(rows, key=lambda r: r["barrier_days"])
    fig = go.Figure(go.Bar(
        x=[r["barrier_days"] for r in rows],
        y=[CATEGORY_LABEL[r["category"]] for r in rows],
        orientation="h",
        marker=dict(color=[chart_color(r["category"], dark) for r in rows],
                    line=dict(color=t["surface"], width=2)),
        text=[f"{r['barrier_days']:.1f} d · {r['open']} open" for r in rows],
        textposition="outside", textfont=dict(color=t["ink_2"], size=11), cliponaxis=False,
        customdata=[[r["open"], r["patients"], FAMILY_OF.get(r["category"], "")] for r in rows],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} barrier-days<br>%{customdata[0]} open now · "
                      "%{customdata[1]} patients<br>%{customdata[2]}<extra></extra>"))
    fig = _layout(fig, 36 * max(len(rows), 3) + 40, t)
    fig.update_xaxes(title=dict(text="barrier-days (sum of each barrier's age)",
                                font=dict(size=11, color=t["muted"])))
    return fig


def owner_chart(counts: dict[str, int], dark: bool = False) -> go.Figure:
    t = tokens(dark)
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[OWNER_LABEL.get(k, k) for k, _ in items], orientation="h",
        marker=dict(color=t["ink_2"], line=dict(color=t["surface"], width=2)),
        text=[str(v) for _, v in items], textposition="outside", cliponaxis=False,
        textfont=dict(color=t["ink_2"], size=11),
        hovertemplate="<b>%{y}</b><br>%{x} open barriers<extra></extra>"))
    fig = _layout(fig, 34 * max(len(items), 3) + 40, t)
    fig.update_xaxes(rangemode="tozero", nticks=6, tickformat="d", tickangle=0,
                     title=dict(text="open barriers", font=dict(size=11, color=t["muted"])))
    return fig


def funnel_chart(stages: list[tuple[str, int]], dark: bool = False) -> go.Figure:
    t = tokens(dark)
    fig = go.Figure(go.Funnel(
        y=[s for s, _ in stages], x=[v for _, v in stages],
        marker=dict(color=list(t["funnel"]), line=dict(color=t["surface"], width=2)),
        textinfo="value", textfont=dict(color=list(t["funnel_text"]), size=13),
        connector=dict(fillcolor=t["row_line"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x} patients<extra></extra>"))
    return _layout(fig, 240, t)


def calibration_chart(rows: list[dict], dark: bool = False) -> go.Figure:
    """Reliability view: bar = actual accuracy per confidence bucket; dot = the
    confidence Jev claimed. Same 0–100% scale, one axis."""
    t = tokens(dark)
    rows = [r for r in rows if r["n"]]
    labels = [f"{r['bucket']}<br><span style='font-size:10px'>n={r['n']}</span>" for r in rows]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[100 * r["accuracy"] for r in rows], name="actual accuracy",
                marker=dict(color=t["fam_1"], line=dict(color=t["surface"], width=2)),
                text=[pct(r["accuracy"], 0) for r in rows], textposition="outside",
                textfont=dict(color=t["ink"], size=12), cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>actual accuracy %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=labels, y=[100 * r["mean_confidence"] for r in rows], name="mean confidence",
                    mode="markers", marker=dict(size=11, color=t["ink"], symbol="diamond",
                                                line=dict(color=t["surface"], width=2)),
                    hovertemplate="<b>%{x}</b><br>mean confidence %{y:.1f}%<extra></extra>")
    fig = _layout(fig, 280, t)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14, x=0,
                                                   font=dict(size=11, color=t["ink_2"])))
    fig.update_yaxes(range=[0, 112], showgrid=True, gridcolor=t["grid"], ticksuffix="%",
                     tickfont=dict(color=t["muted"]))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=t["ink_2"]))
    return fig
