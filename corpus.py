"""
Stage 1 — Synthetic corpus generator for Discharge-Barrier Radar.

100% synthetic. No real patients, facilities, payers, or people.

Each patient gets a scripted *storyline* (which barriers appear, on which day
they are raised / resolved / get stuck, when the patient becomes medically
ready). Daily case-manager notes and secure-chat messages are then rendered
from that storyline using hand-written templates.

Outputs (in ./data):
  corpus.jsonl        -- the CLEAN text the pipeline is allowed to read
  ground_truth.jsonl  -- the answer key, keyed by snippet_id (eval only)
  patients.json       -- non-label patient metadata (unit, admit date)

The pipeline must only ever call `load_corpus()`. `load_ground_truth()` is for
eval.py and nothing else.

Usage:
  python corpus.py                  # generate + print summary
  python corpus.py --show P-1007    # print one patient's timeline with labels
  python corpus.py --seed 7         # different (still deterministic) corpus
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.jsonl"
PATIENTS_PATH = DATA_DIR / "patients.json"

DEFAULT_SEED = 42
N_PATIENTS = 40
BASE_DATE = datetime(2026, 9, 1)

CATEGORIES = [
    "post_acute_placement", "payer_authorization", "equipment_services",
    "family_caregiver", "transport", "clinical_pending", "legal_capacity",
    "social_housing", "none", "other",
]
BARRIER_CATEGORIES = [c for c in CATEGORIES if c != "none"]

# Relative frequency of each barrier type (placement/auth/family dominate in
# real discharge delays; legal/other are rare).
CATEGORY_WEIGHTS = {
    "post_acute_placement": 5, "payer_authorization": 4, "equipment_services": 3,
    "family_caregiver": 4, "transport": 4, "clinical_pending": 3,
    "legal_capacity": 1, "social_housing": 4, "other": 1,
}

MIN_PATIENTS_PER_CATEGORY = 3

DEFAULT_OWNER = {
    "post_acute_placement": "external_facility",
    "payer_authorization": "payer",
    "equipment_services": "vendor",
    "family_caregiver": "family",
    "transport": "case_management",
    "clinical_pending": "physician_team",
    "legal_capacity": "case_management",
    "social_housing": "case_management",
    "other": "not_stated",
}

CHAT_ROLES = {
    "post_acute_placement": ["pt_ot", "case_manager"],
    "payer_authorization": ["utilization_review", "case_manager"],
    "equipment_services": ["rt", "rn"],
    "family_caregiver": ["rn", "social_worker"],
    "transport": ["case_manager", "charge_rn"],
    "clinical_pending": ["hospitalist"],
    "legal_capacity": ["social_worker", "psychiatry"],
    "social_housing": ["social_worker", "rn"],
    "other": ["rn"],
}

# Phase -> (status, escalation) ground-truth labels.
PHASE_LABELS = {
    "raised":    ("newly_raised", "routine"),
    "hedged":    ("newly_raised", "routine"),
    "ongoing":   ("ongoing", "delayed"),
    "stuck":     ("ongoing", "stuck"),
    "resolved":  ("resolved", "routine"),
    "dismissed": ("resolved", "routine"),   # hedged barrier that turned out not needed
}

# Fictional names only.
FACILITIES = ["Maple Grove SNF", "Riverside Rehab Center", "Cedar Hills Care Center",
              "Oakview Nursing & Rehab", "Harborview Post-Acute", "Willow Creek SNF",
              "Brookside Acute Rehab"]
PAYERS = ["Summit Health Plan", "BlueRiver Advantage", "Keystone Medicaid MCO",
          "Pinnacle MA"]
VENDORS = ["Apex Medical Supply", "AirLife O2", "HomeWell DME"]
HH_AGENCIES = ["CareBridge Home Health", "Northstar HH", "Lakeside Home Care"]
UNITS = ["4 West", "5 East", "6 North", "Stepdown"]
DIAGNOSES = ["CHF exacerbation", "COPD exacerbation", "hip fx s/p ORIF", "L MCA CVA",
             "urosepsis", "CAP", "LLE cellulitis", "DKA", "upper GI bleed",
             "fall w/ rib fxs", "AKI on CKD", "SBO, conservative mgmt"]


# --------------------------------------------------------------------------- #
# Templates
#
# TEMPLATES[category][phase][channel] -> list of entries.
# An entry is either a string (owner = DEFAULT_OWNER[category]) or a
# (string, owner) tuple. Placeholders are filled from the patient context.
# --------------------------------------------------------------------------- #

TEMPLATES: dict[str, dict[str, dict[str, list]]] = {
    "post_acute_placement": {
        "raised": {
            "note": [("PT/OT recommending SNF. Referrals sent to {facility_a} and {facility_b}.", "case_management"),
                     ("Pt will need SNF placement for rehab; referral packet faxed to {facility_a}.", "case_management"),
                     "Recommendation is acute rehab (IRF). Referral submitted to {facility_a}, awaiting clinical review."],
            "chat": [("heads up PT says she won't be safe going home, they're recommending SNF", "case_management")],
        },
        "hedged": {
            "note": ["May need SNF depending on PT eval tomorrow.",
                     "Possible SNF vs home with HH — will reassess after PT sees pt."],
            "chat": ["PT thinks he might need SNF, not sure yet"],
        },
        "ongoing": {
            "note": ["Still awaiting bed offer from {facility_a}.",
                     "{facility_a} reviewing clinicals, no decision yet.",
                     ("SNF referral pending; followed up with {facility_a} admissions this AM.", "case_management")],
            "chat": ["still no bed at {facility_a}, called twice today", "{facility_a} still reviewing, no answer on the bed"],
        },
        "stuck": {
            "note": ["{facility_a} and {facility_b} both declined. Third referral sent; no accepting facility yet.",
                     "No SNF bed available in network after {n} referrals. Escalated to CM leadership."],
            "chat": ["another decline, that's {n} facilities now. we have nowhere for her to go", "{facility_b} said no too. running out of places to send her"],
        },
        "resolved": {
            "note": ["SNF confirmed at {facility_a}, bed available {weekday}, transport arranged.",
                     "{facility_a} accepted pt; bed ready tomorrow.",
                     "Accepting facility confirmed — {facility_a} will take pt {weekday}."],
            "chat": ["{facility_a} just called, they accepted him! bed {weekday}"],
        },
        "dismissed": {
            "note": ["PT now recommends home with HH; SNF no longer needed."],
            "chat": ["PT re-eval went great, no SNF needed, she can go home"],
        },
    },
    "payer_authorization": {
        "raised": {
            "note": ["Prior auth for SNF submitted to {payer}.",
                     "Insurance auth required for IRF; {payer} request initiated."],
            "chat": ["{payer} wants a peer to peer before they'll auth"],
        },
        "hedged": {
            "note": ["{payer} may require peer-to-peer before approving.",
                     "Possible auth issue — {payer} plan might not cover acute rehab."],
            "chat": ["heard {payer} might push back on the rehab auth"],
        },
        "ongoing": {
            "note": ["Auth still pending with {payer}, ref #{ref}.",
                     "Awaiting {payer} determination; additional clinicals submitted."],
            "chat": ["still waiting on {payer} auth, nothing back yet", "{payer} auth still pending, UR following up"],
        },
        "stuck": {
            "note": ["{payer} denied auth. Peer-to-peer missed, appeal filed, no timeline for decision.",
                     "Second auth denial from {payer}; appeal pending, pt remains in house."],
            "chat": ["{payer} denied again. appeal is in but they're saying 5-7 business days", "appeal w {payer} still sitting there, no decision, pt stuck here"],
        },
        "resolved": {
            "note": ["{payer} approved auth, ref #{ref}. Clear to transfer.",
                     "Auth approved by {payer} this afternoon."],
            "chat": ["auth came through from {payer}!!"],
        },
        "dismissed": {
            "note": ["Confirmed with {payer} that no prior auth is required for this level of care."],
            "chat": ["good news, {payer} says no auth needed for SNF on this plan"],
        },
    },
    "equipment_services": {
        "raised": {
            "note": ["Pt will require home O2; order sent to {vendor}.",
                     "Needs hospital bed and wheelchair at home — DME order placed with {vendor}.",
                     ("Home health referral sent to {hh}; start of care needed prior to dc.", "case_management")],
            "chat": ["RT here, he'll need home O2, sats drop to 86% walking"],
        },
        "hedged": {
            "note": ["Possible equipment issue — may need home O2 pending ambulatory sat.",
                     "Might need a hospital bed at home, family unsure of setup."],
            "chat": ["might need O2 at home, doing walk test later"],
        },
        "ongoing": {
            "note": ["Awaiting {vendor} delivery of home O2.",
                     "{hh} has not yet accepted the HH referral."],
            "chat": ["{vendor} still hasn't confirmed the delivery", "waiting on {hh} to accept, nothing yet"],
        },
        "stuck": {
            "note": ["{vendor} unable to deliver concentrator until next week; second vendor also unable. No alternative identified.",
                     "Three HH agencies declined due to staffing; no agency accepting in pt's zip."],
            "chat": ["no HH agency will take her, 3 declines, not sure what else to try", "{vendor} pushed the O2 delivery again, no other vendor can do it"],
        },
        "resolved": {
            "note": ["O2 delivered to home by {vendor}; confirmed with spouse.",
                     "{hh} accepted, start of care {weekday}.",
                     "DME delivered, hospital bed set up at home per daughter."],
            "chat": ["{vendor} confirmed O2 dropped off at the house"],
        },
        "dismissed": {
            "note": ["Ambulatory sat 94% on RA — no home O2 needed."],
            "chat": ["walk test fine, no O2 needed"],
        },
    },
    "family_caregiver": {
        "raised": {
            "note": ["Family unable to provide the 24h supervision recommended by PT.",
                     "Daughter is primary caregiver but works nights; cannot pick up until the weekend."],
            "chat": ["daughter just told me she can't take him home, she works doubles"],
        },
        "hedged": {
            "note": ["Family may not be able to take pt home — son 'thinking about it'.",
                     "Unclear if spouse can manage care at home."],
            "chat": ["wife seemed hesitant about taking him home, might be an issue"],
        },
        "ongoing": {
            "note": ["Still waiting on family decision re: dc plan.",
                     ("Left VM for son again re: caregiver plan.", "case_management")],
            "chat": ["tried the son again, no answer", "family still hasn't decided on the plan"],
        },
        "stuck": {
            "note": ["Family meeting cancelled twice; unable to reach HCP after {n} attempts.",
                     "Daughter refusing to take pt home and declining SNF; no agreed plan."],
            "chat": ["family no-showed the meeting again. {n} tries now, can't get anyone on the phone", "daughter won't take her and won't agree to SNF. at a standstill"],
        },
        "resolved": {
            "note": ["Family can now pick up {weekday}.",
                     "Spoke with daughter, she will stay with pt 24/7 for the first 2 weeks; caregiver plan in place.",
                     "Son confirmed he will pick up pt {weekday} after work."],
            "chat": ["wife said she's fine to pick up {weekday} morning"],
        },
        "dismissed": {
            "note": ["Spoke with spouse — comfortable managing care at home, no concerns."],
            "chat": ["talked to the wife, she's totally fine taking him home"],
        },
    },
    "transport": {
        "raised": {
            "note": ["Pt will need stretcher transport; request placed.",
                     "Needs wheelchair van to SNF, no family available to drive."],
            "chat": ["no ride home for her, family all out of state"],
        },
        "hedged": {
            "note": ["May need ambulance transport depending on O2 needs."],
            "chat": ["might need a stretcher for transport, will see"],
        },
        "ongoing": {
            "note": ["Awaiting confirmation of wheelchair van pickup time."],
            "chat": ["transport still hasn't called back w a time", "no pickup time from transport yet"],
        },
        "stuck": {
            "note": [("Stretcher transport cancelled twice by vendor; no transport available until {weekday}.", "vendor")],
            "chat": [("transport cancelled on us AGAIN, earliest they have is {weekday}", "vendor"),
                     ("2nd transport cancellation, no van available til {weekday}", "vendor")],
        },
        "resolved": {
            "note": ["Transport arranged for {weekday} 10am.",
                     "Wheelchair van confirmed for 1400 today."],
            "chat": ["van booked for 2pm tmrw"],
        },
        "dismissed": {
            "note": ["Pt able to go by car; no special transport needed."],
            "chat": ["she can ride in the car, no stretcher needed"],
        },
    },
    "clinical_pending": {
        "raised": {
            "note": ["Awaiting echo read before team will clear for dc.",
                     "Pending MRI; neuro wants results prior to dc.",
                     "Needs repeat INR in AM before dc decision."],
            "chat": ["neuro wants the MRI before we send him out"],
        },
        "hedged": {
            "note": ["Possible need for GI scope before dc, GI to weigh in."],
            "chat": ["GI might want to scope before she goes"],
        },
        "ongoing": {
            "note": ["MRI still not read.", "Echo done, read pending."],
            "chat": ["MRI still in the queue", "echo still not read"],
        },
        "stuck": {
            "note": ["MRI bumped for the third day in a row due to scanner queue."],
            "chat": ["MRI got bumped AGAIN. 3rd day", "scanner backlog, MRI bumped again, no slot yet"],
        },
        "resolved": {
            "note": ["MRI negative; neuro cleared pt for dc.",
                     "Echo reviewed, no change to plan — team clearing pt."],
            "chat": ["MRI clean, ok to dc from neuro standpoint"],
        },
        "dismissed": {
            "note": ["GI signed off, no scope needed prior to dc."],
            "chat": ["GI says no scope needed"],
        },
    },
    "legal_capacity": {
        "raised": {
            "note": ["Capacity eval completed — pt lacks capacity. No HCP identified; initiating guardianship process.",
                     "POA paperwork incomplete; pt cannot sign SNF admission docs."],
            "chat": ["psych says she doesn't have capacity, and no HCP on file"],
        },
        "hedged": {
            "note": ["Capacity questionable — may need formal eval before pt can sign for placement."],
            "chat": ["not sure he has capacity to sign, might need psych"],
        },
        "ongoing": {
            "note": ["Guardianship petition pending with court."],
            "chat": ["legal still working on the guardianship", "guardianship petition still with the court"],
        },
        "stuck": {
            "note": ["Court date for emergency guardianship not until next month; no surrogate available."],
            "chat": ["court can't hear the guardianship til next month. no one else to sign", "still no surrogate, court date weeks out. nobody can sign"],
        },
        "resolved": {
            "note": ["Temporary guardian appointed; can now sign facility paperwork.",
                     "Son located with valid HCP document; consent obtained."],
            "chat": ["court appointed the temp guardian today"],
        },
        "dismissed": {
            "note": [("Psych eval: pt has capacity to make dc decisions.", "physician_team")],
            "chat": [("psych saw him, he has capacity", "physician_team")],
        },
    },
    "social_housing": {
        "raised": {
            "note": ["Pt experiencing homelessness; no safe dc address.",
                     "Pt lives alone in 3rd-floor walk-up, unable to manage stairs; housing options being explored.",
                     "Pt reports eviction notice; no stable housing to return to."],
            "chat": ["he told me he's been staying in his car"],
        },
        "hedged": {
            "note": ["Possible housing instability — pt vague about where he is staying."],
            "chat": ["not sure she actually has a place to go, story keeps changing"],
        },
        "ongoing": {
            "note": ["Shelter referral pending bed availability."],
            "chat": ["still waiting on the respite program", "shelter referral still pending a bed"],
        },
        "stuck": {
            "note": ["All shelters at capacity; medical respite declined; no safe dc location identified."],
            "chat": ["respite said no, shelters full. we have nowhere to send him", "all shelters full again today, no safe place to dc him"],
        },
        "resolved": {
            "note": ["Medical respite bed secured starting {weekday}.",
                     "Pt's cousin agreed to let pt stay with her; safe address confirmed."],
            "chat": ["respite has a bed for him {weekday}"],
        },
        "dismissed": {
            "note": ["Confirmed pt has stable housing with sister; no housing concerns."],
            "chat": ["confirmed she lives with her sister, housing is fine"],
        },
    },
    "other": {
        "raised": {
            "note": ["Pt declining dc until arrangements are made for dog at home.",
                     "Pt requesting to stay until son arrives from overseas."],
            "chat": ["she won't leave until someone feeds her cats"],
        },
        "hedged": {
            "note": ["Pt may push back on dc timing — mentioned wanting to wait for family visit."],
            "chat": ["he might refuse to leave before the weekend, we'll see"],
        },
        "ongoing": {
            "note": ["Pt still declining dc; ongoing discussion with pt."],
            "chat": ["still refusing to go", "she's still saying she won't leave"],
        },
        "stuck": {
            "note": ["Pt continues to refuse dc despite multiple conversations; ethics consult requested."],
            "chat": ["talked to her 4 times, still refusing. ethics consult put in", "still refusing dc after multiple talks, waiting on ethics"],
        },
        "resolved": {
            "note": ["Neighbor agreed to care for dog; pt agreeable to dc."],
            "chat": ["neighbor's got the cats, she's ok to go now"],
        },
        "dismissed": {
            "note": ["Pt now agreeable to dc as planned."],
            "chat": ["he's fine with leaving tomorrow now"],
        },
    },
}

# Readiness statements (discharge-related, category none unless paired).
READINESS = {
    "ready": {
        "note": ["Hospitalist confirms pt is medically ready for discharge.",
                 "Medically stable for dc per team.",
                 "Pt medically cleared as of today."],
        "chat": ["from medicine standpoint he's good to go",
                 "cleared medically, just need a dispo"],
    },
    "not_ready": {
        "note": ["Not medically ready; remains on IV diuresis.",
                 "Team expects 2-3 more days of IV abx before dc.",
                 "Not yet medically stable for dc — still requiring O2 titration."],
        "chat": ["not ready yet, still on the drip",
                 "she's not medically ready, give it a couple days"],
    },
}

# Discharge-related, but no barrier (the "none" case that still talks about dispo).
PLAN_NO_BARRIER = [
    "Dc plan is home with spouse; no needs identified.",
    "Anticipate dc home, pt independent with ADLs, no services needed.",
    "Met with pt; plans to return home, family supportive, no barriers identified.",
]

# Not discharge-related at all.
FILLER_NOTE = [
    "Pain controlled on PO meds.", "Afebrile overnight.", "Tolerating regular diet.",
    "Labs stable this AM.", "Ambulated in hall with PT, steady gait.",
    "Pt resting comfortably, no acute events overnight.",
]
FILLER_CHAT = [
    "can someone check the IV pump in {room}, keeps alarming",
    "pt in {room} asking for a warm blanket",
    "labs back, K 3.4, repleting per protocol",
    "pharmacy verified the abx dose",
    "pt asking for pain meds, due at 1400",
]


# --------------------------------------------------------------------------- #
# Storyline model
# --------------------------------------------------------------------------- #

@dataclass
class BarrierPlan:
    category: str
    raise_day: int
    resolve_day: int | None = None
    hedged: bool = False
    hedge_outcome: str = "confirm"        # "confirm" | "dismiss"
    channel: str = "mixed"                # "mixed" (mostly notes) | "chat" (chat-only)
    stuck: bool = False

    def phase_on(self, day: int) -> str | None:
        """Which phase this barrier is in on `day`, or None if not active."""
        if day < self.raise_day:
            return None
        if self.resolve_day is not None and day > self.resolve_day:
            return None
        if day == self.raise_day:
            return "hedged" if self.hedged else "raised"
        if day == self.resolve_day:
            return "dismissed" if (self.hedged and self.hedge_outcome == "dismiss") else "resolved"
        if self.stuck and day - self.raise_day >= 2:
            return "stuck"
        return "ongoing"


@dataclass
class PatientPlan:
    patient_id: str
    n_days: int
    admit: datetime
    unit: str
    room: str
    header: str
    barriers: list[BarrierPlan] = field(default_factory=list)
    ready_day: int | None = None
    copy_forward: bool = False
    ctx: dict = field(default_factory=dict)
    used: set = field(default_factory=set)      # templates already used for this patient


@dataclass
class Segment:
    """One sentence/message fragment with its ground-truth labels."""
    text: str
    kind: str                         # header | filler | plan | readiness | barrier
    discharge_related: bool
    category: str = "none"
    status: str = "not_applicable"
    owner: str | None = None
    escalation: str | None = None
    readiness: str | None = None
    tags: list[str] = field(default_factory=list)
    stale: bool = False
    copied_from: str | None = None


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

class CorpusGenerator:
    def __init__(self, seed: int = DEFAULT_SEED):
        self.rng = random.Random(seed)

    # ---- storyline construction ------------------------------------------ #

    def _feature_sets(self) -> dict[str, set[int]]:
        """Spread each hard case across a quota of patients (deterministic)."""
        idx = list(range(N_PATIENTS))
        self.rng.shuffle(idx)
        clean = set(idx[:5])                   # no barriers at all
        barrier_pts = [i for i in range(N_PATIENTS) if i not in clean]

        def pick(k: int) -> set[int]:
            return set(self.rng.sample(barrier_pts, k))

        return {
            "clean": clean,
            "copy_forward": pick(10),
            "hedged": pick(14),
            "chat_only": pick(9),
            "negation": pick(17),
            "multi": pick(10),
            "stuck": pick(11),
        }

    def _category_deck(self, n: int) -> list[str]:
        """Primary barrier per patient: every category at least MIN_PATIENTS_PER_CATEGORY
        times, the rest filled by weight — so small categories never vanish."""
        deck = BARRIER_CATEGORIES * MIN_PATIENTS_PER_CATEGORY
        cats, w = zip(*CATEGORY_WEIGHTS.items())
        deck += self.rng.choices(cats, weights=w, k=max(0, n - len(deck)))
        self.rng.shuffle(deck)
        return deck

    def _weighted_categories(self, k: int, first: str) -> list[str]:
        pool = {c: w for c, w in CATEGORY_WEIGHTS.items() if c != first}
        out = [first]
        for _ in range(k - 1):
            cats, w = zip(*pool.items())
            c = self.rng.choices(cats, weights=w)[0]
            out.append(c)
            pool.pop(c)
        return out

    def build_plans(self) -> list[PatientPlan]:
        f = self._feature_sets()
        deck = iter(self._category_deck(N_PATIENTS - len(f["clean"])))
        plans = []
        for i in range(N_PATIENTS):
            rng = self.rng
            pid = f"P-{1001 + i}"
            n_days = rng.randint(3, 6)
            if i in f["stuck"]:
                n_days = max(n_days, 5)
            if i in f["copy_forward"] or i in f["negation"]:
                n_days = max(n_days, 4)

            admit = BASE_DATE + timedelta(days=rng.randint(0, 10))
            age, sex = rng.randint(58, 94), rng.choice("MF")
            header = f"{age}{sex} adm {admit.month}/{admit.day} w/ {rng.choice(DIAGNOSES)}."
            unit = rng.choice(UNITS)
            fa, fb = rng.sample(FACILITIES, 2)
            plan = PatientPlan(
                patient_id=pid, n_days=n_days, admit=admit, unit=unit,
                room=f"{rng.randint(1, 30)}", header=header,
                copy_forward=i in f["copy_forward"],
                ctx={"facility_a": fa, "facility_b": fb, "payer": rng.choice(PAYERS),
                     "vendor": rng.choice(VENDORS), "hh": rng.choice(HH_AGENCIES),
                     "ref": f"{rng.randint(100000, 999999)}", "n": rng.choice([3, 4, 5])},
            )

            if i in f["clean"]:
                plan.ready_day = rng.randint(max(0, n_days - 3), n_days - 1)
                plans.append(plan)
                continue

            n_barriers = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
            if i in f["multi"]:
                n_barriers = max(n_barriers, 2)
            cats = self._weighted_categories(n_barriers, first=next(deck))
            first_raise = rng.randint(0, 1)
            for j, cat in enumerate(cats):
                if j == 0 or (j == 1 and i in f["multi"]):
                    raise_day = first_raise           # multi -> two barriers same day
                else:
                    raise_day = rng.randint(0, n_days - 2)
                plan.barriers.append(BarrierPlan(category=cat, raise_day=raise_day))

            b = plan.barriers
            if i in f["negation"]:
                tgt = b[0]
                tgt.resolve_day = rng.randint(tgt.raise_day + 1, n_days - 1)
            for other in b[1:]:
                if other.resolve_day is None and rng.random() < 0.3 and other.raise_day < n_days - 1:
                    other.resolve_day = rng.randint(other.raise_day + 1, n_days - 1)
            if i in f["hedged"]:
                tgt = b[-1]
                tgt.hedged = True
                tgt.hedge_outcome = rng.choice(["confirm", "dismiss"])
                if tgt.hedge_outcome == "dismiss" and tgt.resolve_day is None:
                    if tgt.raise_day >= n_days - 1:
                        tgt.raise_day = n_days - 2
                    tgt.resolve_day = rng.randint(tgt.raise_day + 1, n_days - 1)
            if i in f["chat_only"]:
                b[-1].channel = "chat"
            if i in f["stuck"]:
                candidates = [x for x in b if x.resolve_day is None and not x.hedged] or [b[0]]
                tgt = candidates[0]
                tgt.resolve_day = None
                tgt.raise_day = min(tgt.raise_day, n_days - 3)
                tgt.stuck = True

            # Medical readiness. clinical_pending gates readiness on its resolution.
            clinical = [x for x in b if x.category == "clinical_pending"]
            if clinical:
                plan.ready_day = clinical[0].resolve_day
            else:
                plan.ready_day = rng.randint(0, n_days - 2) if rng.random() < 0.85 else None
            plans.append(plan)
        return plans

    # ---- rendering ------------------------------------------------------- #

    def _pick(self, pool: list, plan: PatientPlan, allow_repeat: bool = False):
        """Choose a template, avoiding reuse within one patient so the only verbatim
        repetition in the corpus is the deliberate (labelled) copy-forward kind.
        Returns None when the pool is exhausted and repeats aren't allowed — the
        caller then simply doesn't mention it that day."""
        fresh = [e for e in pool if e not in plan.used]
        if not fresh and not allow_repeat:
            return None
        entry = self.rng.choice(fresh or pool)
        plan.used.add(entry)
        return entry

    def _fill(self, template: str, plan: PatientPlan, ts: datetime) -> str:
        weekday = (ts + timedelta(days=self.rng.choice([1, 2]))).strftime("%A")
        return template.format(weekday=weekday, room=plan.room, **plan.ctx)

    def _barrier_segment(self, b: BarrierPlan, phase: str, channel: str,
                         plan: PatientPlan, ts: datetime) -> Segment | None:
        entry = self._pick(TEMPLATES[b.category][phase][channel], plan)
        if entry is None:
            return None
        text, owner = (entry if isinstance(entry, tuple) else (entry, DEFAULT_OWNER[b.category]))
        status, escalation = PHASE_LABELS[phase]
        tags = []
        if phase == "hedged":
            tags.append("hedged")
        if phase in ("resolved", "dismissed"):
            tags.append("negation")
        if phase == "stuck":
            tags.append("stuck")
        if b.channel == "chat":
            tags.append("chat_only")
        readiness = None
        if b.category == "clinical_pending":
            readiness = "ready" if phase in ("resolved", "dismissed") else "not_ready"
        return Segment(text=self._fill(text, plan, ts), kind="barrier", discharge_related=True,
                       category=b.category, status=status, owner=owner,
                       escalation=escalation, readiness=readiness, tags=tags)

    def render(self, plan: PatientPlan) -> list[tuple[dict, dict]]:
        """Return [(clean_snippet, ground_truth)] for one patient, chronological."""
        rng = self.rng
        out: list[tuple[datetime, str, str, list[Segment], str]] = []
        # carry[barrier_index] = (Segment, source_snippet_id, times_carried)
        carry: dict[int, tuple[Segment, str, int]] = {}

        for day in range(plan.n_days):
            day_start = plan.admit + timedelta(days=day)
            note_ts = day_start.replace(hour=rng.randint(8, 11), minute=rng.randint(0, 59))
            note_id = f"{plan.patient_id}-d{day}-note"
            note_segs: list[Segment] = [Segment(plan.header, "header", False)]
            fresh_note: list[Segment] = []
            chats: list[tuple[Segment, str]] = []        # (segment, author_role)

            # -- barriers -------------------------------------------------- #
            for bi, b in enumerate(plan.barriers):
                phase = b.phase_on(day)
                if phase is None:
                    carry.pop(bi, None)
                    continue
                pivotal = phase in ("raised", "hedged", "resolved", "dismissed")

                # Copy-forward: re-paste yesterday's sentence verbatim instead of writing new text.
                if (plan.copy_forward and not pivotal and bi in carry and carry[bi][2] < 2):
                    seg, src, times = carry[bi]
                    stale = Segment(**{**seg.__dict__, "status": "ongoing",
                                       "escalation": seg.escalation if seg.escalation != "routine" else "delayed",
                                       "tags": sorted(set(seg.tags) - {"hedged", "negation"} | {"copy_forward"}),
                                       "stale": True, "copied_from": src})
                    note_segs.append(stale)
                    carry[bi] = (seg, src, times + 1)
                    continue

                mention_p = 1.0 if pivotal else (0.8 if phase == "stuck" else 0.4)
                if rng.random() > mention_p:
                    continue
                if b.channel == "chat" or (not plan.copy_forward and rng.random() < 0.15):
                    seg = self._barrier_segment(b, phase, "chat", plan, note_ts)
                    if seg is None:
                        continue
                    chats.append((seg, rng.choice(CHAT_ROLES[b.category])))
                else:
                    seg = self._barrier_segment(b, phase, "note", plan, note_ts)
                    if seg is None:
                        continue
                    fresh_note.append(seg)
                    if phase in ("resolved", "dismissed"):
                        carry.pop(bi, None)
                    else:
                        carry[bi] = (seg, note_id, 0)

            # -- medical readiness ----------------------------------------- #
            clinical_today = any(s.category == "clinical_pending" for s in fresh_note) or \
                any(s.category == "clinical_pending" for s, _ in chats)
            readiness_state = None
            if not clinical_today:
                if plan.ready_day is not None and day == plan.ready_day:
                    readiness_state = "ready"
                elif plan.ready_day is not None and day > plan.ready_day and rng.random() < 0.2:
                    readiness_state = "ready"
                elif (plan.ready_day is None or day < plan.ready_day) and rng.random() < 0.25 \
                        and not any(b.category == "clinical_pending" for b in plan.barriers):
                    readiness_state = "not_ready"
            if readiness_state:
                channel = "note" if rng.random() < 0.7 else "chat"
                text = self._pick(READINESS[readiness_state][channel], plan)
                if text and channel == "note":
                    fresh_note.append(Segment(text, "readiness", True, readiness=readiness_state))
                elif text:
                    chats.append((Segment(text, "readiness", True, readiness=readiness_state),
                                  "hospitalist"))

            # -- plan / filler --------------------------------------------- #
            if not plan.barriers and rng.random() < 0.6 and (text := self._pick(PLAN_NO_BARRIER, plan)):
                fresh_note.append(Segment(text, "plan", True))
            if rng.random() < 0.5 or not fresh_note:
                fresh_note.insert(0, Segment(self._pick(FILLER_NOTE, plan, allow_repeat=True), "filler", False))
            if rng.random() < 0.08:
                chats.append((Segment(self._fill(self._pick(FILLER_CHAT, plan, allow_repeat=True), plan, note_ts),
                                      "filler", False), rng.choice(["rn", "charge_rn", "pharmacy"])))

            # -- decide whether a formal note is written today ------------- #
            has_new_info = any(s.kind in ("barrier", "readiness", "plan") for s in fresh_note)
            write_note = (plan.copy_forward or has_new_info or rng.random() < 0.3
                          or (not plan.barriers and day in (0, plan.n_days - 1)))
            if write_note:
                out.append((note_ts, note_id, "case_manager", note_segs + fresh_note, "note"))
            else:
                # No note today: anything written for the note goes out as a chat instead.
                for s in fresh_note:
                    if s.kind == "barrier":
                        chats.append((s, "case_manager"))
                for bi in list(carry):
                    if carry[bi][1] == note_id:
                        carry.pop(bi)

            for k, (seg, role) in enumerate(chats):
                cts = day_start.replace(hour=rng.randint(10, 18), minute=rng.randint(0, 59))
                out.append((cts, f"{plan.patient_id}-d{day}-chat{k}", role, [seg], "chat"))

        out.sort(key=lambda r: r[0])
        return [self._to_records(plan, *r) for r in out]

    @staticmethod
    def _to_records(plan: PatientPlan, ts: datetime, sid: str, role: str,
                    segs: list[Segment], source: str) -> tuple[dict, dict]:
        text = " ".join(s.text for s in segs)
        clean = {"snippet_id": sid, "patient_id": plan.patient_id,
                 "timestamp": ts.isoformat(timespec="minutes"),
                 "source": source, "author_role": role, "text": text}

        live = [s for s in segs if s.kind == "barrier" and not s.stale]
        stale = [s for s in segs if s.kind == "barrier" and s.stale]
        readiness = next((s.readiness for s in reversed(segs) if s.readiness), "not_stated")
        # Snippet-level tags describe the NEW content; copied text only contributes "copy_forward".
        tags = sorted({t for s in live for t in s.tags} | ({"copy_forward"} if stale else set()))
        if len({s.category for s in live}) >= 2:
            tags.append("multi_barrier")
        if not live and not stale:
            tags.append("no_barrier")

        def barrier_dict(s: Segment) -> dict:
            d = {"category": s.category, "status": s.status, "owner": s.owner,
                 "escalation": s.escalation, "text": s.text, "tags": s.tags}
            if s.copied_from:
                d["copied_from"] = s.copied_from
            return d

        truth = {
            "snippet_id": sid, "patient_id": plan.patient_id,
            "is_discharge_related": any(s.discharge_related and not s.stale for s in segs),
            "medical_readiness": readiness,
            "primary_category": live[0].category if live else "none",
            "barriers": [barrier_dict(s) for s in live],
            "stale_barriers": [barrier_dict(s) for s in stale],
            "hard_case_tags": tags,
            "segments": [{"text": s.text, "kind": s.kind, "category": s.category,
                          "stale": s.stale} for s in segs],
        }
        return clean, truth

    def generate(self) -> tuple[list[dict], list[dict], list[dict]]:
        plans = self.build_plans()
        corpus, truth = [], []
        for p in plans:
            for c, t in self.render(p):
                corpus.append(c)
                truth.append(t)
        patients = [{"patient_id": p.patient_id, "unit": p.unit, "room": p.room,
                     "admit_date": p.admit.date().isoformat(), "los_days": p.n_days}
                    for p in plans]
        return corpus, truth, patients


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus() -> list[dict]:
    """Clean snippets only — the only thing the pipeline may read."""
    return _read_jsonl(CORPUS_PATH)


def load_ground_truth() -> dict[str, dict]:
    """Answer key keyed by snippet_id. For eval.py ONLY."""
    return {r["snippet_id"]: r for r in _read_jsonl(GROUND_TRUTH_PATH)}


def load_patients() -> list[dict]:
    return json.loads(PATIENTS_PATH.read_text(encoding="utf-8"))


def write_corpus(seed: int = DEFAULT_SEED) -> tuple[list[dict], list[dict], list[dict]]:
    corpus, truth, patients = CorpusGenerator(seed).generate()
    DATA_DIR.mkdir(exist_ok=True)
    _write_jsonl(CORPUS_PATH, corpus)
    _write_jsonl(GROUND_TRUTH_PATH, truth)
    PATIENTS_PATH.write_text(json.dumps(patients, indent=2), encoding="utf-8")
    return corpus, truth, patients


# --------------------------------------------------------------------------- #
# CLI summary
# --------------------------------------------------------------------------- #

def print_summary(corpus: list[dict], truth: list[dict], patients: list[dict]) -> None:
    src = Counter(c["source"] for c in corpus)
    per_pt = Counter(c["patient_id"] for c in corpus)
    cat = Counter(b["category"] for t in truth for b in t["barriers"])
    status = Counter(b["status"] for t in truth for b in t["barriers"])
    tags = Counter(tag for t in truth for tag in t["hard_case_tags"])
    ready = Counter(t["medical_readiness"] for t in truth)
    related = Counter(t["is_discharge_related"] for t in truth)

    print(f"\nWrote {len(corpus)} snippets for {len(patients)} patients -> {DATA_DIR}")
    print(f"  notes: {src['note']}   chats: {src['chat']}   "
          f"per patient: {min(per_pt.values())}-{max(per_pt.values())}")
    print(f"  discharge-related: {related[True]}   not related: {related[False]}")

    def block(title: str, counter: Counter, order=None) -> None:
        print(f"\n{title}")
        for k in (order or sorted(counter, key=lambda x: -counter[x])):
            if counter.get(k):
                print(f"  {k:<22}{counter[k]:>4}")

    block("Barrier mentions by category (excl. copy-forward)", cat, BARRIER_CATEGORIES)
    block("Barrier status", status)
    block("Hard-case tags (snippets containing)", tags)
    block("Medical readiness", ready)
    print()


def show_patient(pid: str, corpus: list[dict], truth: list[dict]) -> None:
    gt = {t["snippet_id"]: t for t in truth}
    for c in (c for c in corpus if c["patient_id"] == pid):
        t = gt[c["snippet_id"]]
        print(f"\n[{c['timestamp']}] {c['source'].upper():<4} ({c['author_role']})  {c['snippet_id']}")
        print(f"  \"{c['text']}\"")
        print(f"  readiness={t['medical_readiness']}  tags={t['hard_case_tags']}")
        for b in t["barriers"]:
            print(f"   + {b['category']:<21} {b['status']:<13} owner={b['owner']:<17} esc={b['escalation']}")
        for b in t["stale_barriers"]:
            print(f"   ~ (copied from {b['copied_from']}) {b['category']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--show", metavar="PATIENT_ID", help="print one patient's timeline with labels")
    args = ap.parse_args()

    corpus, truth, patients = write_corpus(args.seed)
    if args.show:
        show_patient(args.show, corpus, truth)
    else:
        print_summary(corpus, truth, patients)


if __name__ == "__main__":
    main()
