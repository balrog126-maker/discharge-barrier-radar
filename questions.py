"""
Every Jev question and every confidence threshold in the project lives here.
Nothing else in the codebase defines a question or hard-codes a threshold.

Jev only ever *classifies text*. It is never asked to count, add, compare
dates, or work out how long something has taken — that is plain Python in
ledger.py / eval.py.

Two calls per discharge-related chunk:
  1. RELEVANCE_QUESTIONS  — the gate. If the chunk isn't about discharge,
     stop here (one cheap call, nothing posted).
  2. CLASSIFICATION_QUESTIONS — all remaining questions in ONE batched call;
     Jev answers them in parallel, so five questions cost about the same
     latency as one.
"""

from typesafe_sdk import Choice, Noul, Score

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #

# Gate: probability that a chunk is discharge-related above which we go on to classify it.
RELEVANCE_THRESHOLD = 0.5

# Confidence router (used by router.py).
AUTO_POST_THRESHOLD = 0.9       # >= this: posted to the board automatically
CONFIRM_THRESHOLD = 0.5         # >= this (and < auto-post): shown as "needs confirmation"
                                # <  this: review queue only, never a live board entry

# Which answers the overall confidence is built from. The posted board entry
# is "this category, with this status", so those plus the gate are what must
# be right; owner / readiness / escalation are displayed but don't gate posting.
# Overall confidence = the weakest of these (min), not an average — one shaky
# answer is enough to hold an entry back for confirmation.
# For barrier_status the confidence is taken on the decision the ledger actually
# uses — resolved vs not resolved — i.e. max(P(resolved), 1 - P(resolved)). Jev
# splitting "newly_raised" vs "ongoing" is harmless (the ledger decides that
# from history) and shouldn't hold an entry back.
CONFIDENCE_DRIVERS = ("is_discharge_related", "barrier_category", "barrier_status")
# When the category is "none" there is no barrier for a status to describe, and
# Jev's status answer is noise (it splits between resolved / not_applicable) —
# so barrier_status is left out of the confidence for these categories.
STATUS_IGNORED_FOR_CATEGORIES = ("none",)

# Ledger: a readiness statement updates the patient's medical-readiness flag
# only when Jev is at least this sure of it (it drives the board's sort order).
READINESS_MIN_CONFIDENCE = 0.7


# --------------------------------------------------------------------------- #
# Label sets (shared with the ledger, router, eval and UI)
# --------------------------------------------------------------------------- #

BARRIER_CATEGORIES = (
    "post_acute_placement", "payer_authorization", "equipment_services",
    "family_caregiver", "transport", "clinical_pending", "legal_capacity",
    "social_housing", "other", "none",
)
NO_BARRIER_CATEGORY = "none"
BARRIER_STATUSES = ("newly_raised", "ongoing", "resolved", "not_applicable")
RESOLVED_STATUS = "resolved"
READINESS_LABELS = ("ready", "not_ready", "not_stated")
OWNERS = ("case_management", "physician_team", "family", "payer",
          "external_facility", "vendor", "not_stated")
ESCALATION_LEVELS = ("routine", "delayed", "stuck")       # index == Jev score level


# --------------------------------------------------------------------------- #
# Stage 1: relevance gate
# --------------------------------------------------------------------------- #

RELEVANCE_QUESTIONS = {
    "is_discharge_related": Noul(
        instructions=(
            "This hospital case-management note or staff chat message discusses the "
            "patient's discharge or disposition planning."
        ),
        criteria={
            "true": (
                "It mentions where or when the patient will go after the hospital, anything "
                "that could delay or enable leaving (placement, insurance authorization, "
                "equipment, home health, family or caregivers, transport, pending tests that "
                "hold discharge, capacity or guardianship, housing, patient refusal), or "
                "whether the patient is medically ready or not ready for discharge. Also true "
                "when it says there are no discharge needs or barriers."
            ),
            "false": (
                "It is only routine bedside care or clinical status with no bearing on "
                "leaving the hospital: pain, diet, vitals, labs, IV pumps, medication doses, "
                "comfort requests, a demographic header line."
            ),
        },
    ),
}


# --------------------------------------------------------------------------- #
# Stage 2: classification (one batched call)
# --------------------------------------------------------------------------- #

CLASSIFICATION_QUESTIONS = {
    "barrier_category": Choice(
        instructions=(
            "What kind of discharge barrier does this text discuss? If it names a specific "
            "barrier, give that barrier's topic even when the text says it is now resolved, "
            "approved, arranged or no longer needed. If it says there are no discharge needs "
            "or barriers at all, the answer is none."
        ),
        criteria={
            "post_acute_placement": "Placement in a skilled nursing facility (SNF), acute rehab (IRF), LTACH or similar: referrals, bed offers, acceptances, declines.",
            "payer_authorization": "Insurance or payer prior authorization, peer-to-peer review, denials or appeals.",
            "equipment_services": "Home equipment or services: home oxygen, DME such as hospital beds or wheelchairs, home health agency start of care.",
            "family_caregiver": "Whether family or caregivers can or will care for, take home, or pick up the patient, or their decision about the plan.",
            "transport": "Getting the patient physically out: ambulance, stretcher, wheelchair van, or no ride.",
            "clinical_pending": "A specific named test, imaging read, consult or lab result that discharge waits on, or that result coming back. A general statement that the patient is medically cleared or stable is not this.",
            "legal_capacity": "Decision-making capacity, health-care proxy, power of attorney, guardianship, or who can sign.",
            "social_housing": "Homelessness, eviction, unsafe or inaccessible home, shelter or medical respite.",
            "other": "A barrier that fits none of the above, including the patient's own wishes or refusal to leave (pets at home, waiting for a visitor or an event).",
            "none": "No specific barrier is named: routine care only; a plan saying no needs, services or barriers are identified; or only a general statement that the patient is or is not medically ready, or that treatment is still in progress. Not for a specific barrier that is being resolved.",
        },
    ),
    "barrier_status": Choice(
        instructions="Based only on the wording of this text, what is the state of the discharge barrier it discusses?",
        criteria={
            "newly_raised": "The text introduces the barrier as a new need or problem, including tentative or possible needs ('may need SNF', 'possible equipment issue').",
            "ongoing": "The text says the barrier is still pending or unresolved: waiting, still, again, no response yet, declined, appeal pending.",
            "resolved": "The text says the barrier is now cleared or no longer applies: approved, accepted, confirmed, arranged, delivered, 'can now', 'no longer needed', signed off. Taking a step toward it (order sent, referral faxed, request placed) is not resolved.",
            "not_applicable": "The text does not discuss a discharge barrier.",
        },
    ),
    "medical_readiness": Choice(
        instructions="Does this text say whether the patient is medically ready for discharge?",
        criteria={
            "ready": "It states the patient is medically ready, stable, or cleared for discharge from a medical standpoint.",
            "not_ready": "It states the patient is not yet medically ready, or that the medical team is waiting on something before clearing discharge.",
            "not_stated": "It says nothing about the patient's medical readiness.",
        },
    ),
    "barrier_owner": Choice(
        instructions="Who currently has to act to move this discharge barrier forward?",
        criteria={
            "case_management": "The hospital case manager or social worker (sending referrals, making calls, arranging transport, pursuing housing or guardianship).",
            "physician_team": "The hospital physicians or consultants (reading a test, clearing the patient, a capacity evaluation).",
            "family": "The patient's family or caregivers (deciding, agreeing, picking up).",
            "payer": "The insurance company or health plan (authorizing, reviewing, deciding an appeal).",
            "external_facility": "A SNF, rehab or other receiving facility (reviewing, offering or declining a bed).",
            "vendor": "An outside vendor or agency (equipment delivery, home health agency, transport company).",
            "not_stated": "No one in particular, or there is no barrier.",
        },
    ),
    # Score levels are worded as situations, not degrees ("somewhat delayed") —
    # Jev reads concrete scenarios more reliably than an abstract scale.
    "escalation_need": Score(
        instructions="Which situation best describes this discharge barrier right now?",
        criteria=[
            "routine: a normal next step is underway or just started, or the barrier was just resolved; nothing has gone wrong.",
            "delayed: still waiting on someone with no answer yet, or a follow-up was needed, but there is still a clear path forward.",
            "stuck: multiple failed attempts, repeated declines or denials, cancellations, or no path forward identified; someone needs to escalate.",
        ],
    ),
}
