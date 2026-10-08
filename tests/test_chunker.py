import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunker import ChangeDetector, split_sentences  # noqa: E402


def note(pid, day, text, source="note"):
    return {"snippet_id": f"{pid}-d{day}-{source}", "patient_id": pid,
            "timestamp": f"2026-09-0{day + 1}T09:00", "source": source,
            "author_role": "case_manager", "text": text}


def test_verbatim_copy_forward_is_dropped():
    d = ChangeDetector()
    d.process(note("P1", 0, "80F adm w/ CHF. Awaiting SNF bed offer."))
    ch = d.process(note("P1", 1, "80F adm w/ CHF. Awaiting SNF bed offer. Family meeting set."))
    assert ch.new_sentences == ["Family meeting set."]
    assert len(ch.dropped_sentences) == 2


def test_text_copied_from_two_notes_back_is_still_stale():
    d = ChangeDetector()
    d.process(note("P1", 0, "Prior auth submitted."))
    d.process(note("P1", 1, "Labs stable."))
    assert d.process(note("P1", 2, "Prior auth submitted.")).is_empty


def test_trivial_retyping_counts_as_same_sentence():
    d = ChangeDetector()
    d.process(note("P1", 0, "Auth still pending with Summit Health Plan, ref #123456."))
    assert d.process(note("P1", 1, "auth still pending with  Summit Health Plan ref #123456")).is_empty


def test_changed_sentence_is_emitted():
    d = ChangeDetector()
    d.process(note("P1", 0, "Auth still pending with Summit Health Plan."))
    ch = d.process(note("P1", 1, "Auth approved by Summit Health Plan."))
    assert ch.new_sentences == ["Auth approved by Summit Health Plan."]


def test_patients_are_independent():
    d = ChangeDetector()
    d.process(note("P1", 0, "Awaiting SNF bed offer."))
    assert d.process(note("P2", 0, "Awaiting SNF bed offer.")).new_sentences


def test_chat_is_one_chunk_and_only_exact_repeats_drop():
    d = ChangeDetector()
    first = d.process(note("P1", 0, "still no bed. called twice", source="chat"))
    assert first.new_sentences == ["still no bed. called twice"]
    assert d.process(note("P1", 1, "still no bed. called twice", source="chat")).is_empty


def test_sentence_split_keeps_abbreviations_intact():
    assert split_sentences("91F adm 9/5 w/ L MCA CVA. Pt will need SNF; referral faxed.") == \
        ["91F adm 9/5 w/ L MCA CVA.", "Pt will need SNF; referral faxed."]
