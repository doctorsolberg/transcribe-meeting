from transcribe_meeting.core import write_readable_changes
from transcribe_meeting.model import Utterance
from transcribe_meeting.readable import classify_changes
from transcribe_meeting.util import atomic_json


def _u(uid="u1", raw="", alt=None, readable=None, start=0.0, extra_variants=None):
    variants = dict(extra_variants or {})
    if alt is not None:
        variants["model_b"] = alt
    return Utterance(
        utterance_id=uid,
        source_file="a.m4a",
        source_start=start,
        source_end=start + 1,
        meeting_start=start,
        meeting_end=start + 1,
        speaker_id="S1",
        speaker_name="Игорь",
        raw_text=raw,
        confidence={},
        light_text=raw + ".",
        variants=variants,
        readable_text=readable,
    )


def test_transliteration_is_classified_as_transliteration():
    item = _u(
        raw="через виспер или через дипграм",
        alt="через виспер или через дипграм",
        readable="Через Whisper или через Deepgram.",
    )

    assert sorted(classify_changes(item)) == ["transliteration", "transliteration"]


def test_near_miss_is_classified_as_repair_not_substitution():
    item = _u(
        raw="в биржевском ранскрибаторе встреч", readable="В биржевском транскрибаторе встреч."
    )

    assert classify_changes(item) == ["repair"]


def test_a_cyrillic_swap_is_flagged_for_a_human():
    item = _u(raw="мы очень смеялись", alt="мы очень смеялись", readable="Мы очень хохотали.")

    # The vanished word is the other half of the same swap, not an extra loss.
    assert classify_changes(item) == ["substitution"]


def test_a_word_only_one_model_heard_is_not_a_loss():
    """A single model mishearing is not evidence the word was spoken."""
    item = _u(
        raw="это гигачатовский продукт", alt="это гигачатовский", readable="Это гигачатовский."
    )

    assert "loss" not in classify_changes(item)


def test_a_word_both_models_heard_and_vanished_is_a_loss():
    item = _u(
        raw="это гигачатовский продукт",
        alt="это гигачатовский продукт",
        readable="Это гигачатовский.",
    )

    assert classify_changes(item) == ["loss"]


def test_a_word_only_the_third_model_heard_is_allowed_not_a_substitution():
    """Regression guard: without the union over all variants, a
    Cyrillic word readable kept from c alone (a and b both misheard something
    unrelated) would be misclassified as a substitution — it is neither Latin
    (transliteration) nor close enough to a/b's word to read as a repair
    (0-budget at this length)."""
    item = _u(
        raw="это гигачат продукт",
        alt="это гигачат продукт",
        readable="Это флоу продукт.",
        extra_variants={"model_c": "это флоу продукт"},
    )

    assert classify_changes(item) == []


def test_digitized_number_is_classified_as_number_not_substitution():
    item = _u(raw="приехало сто тридцать человек", readable="Приехало 130 человек.")

    assert classify_changes(item) == ["number"]


def test_digit_already_present_in_source_is_not_a_change():
    item = _u(raw="цена 130 рублей", readable="Цена 130 рублей.")

    assert classify_changes(item) == []


def test_unchanged_utterance_produces_no_rows():
    item = _u(raw="совсем без правок", alt="совсем без правок", readable="Совсем без правок.")

    assert classify_changes(item) == []


def test_csv_puts_risky_rows_first_and_skips_clean_ones(tmp_path):
    items = [
        _u("u1", "через виспер", "через виспер", "Через Whisper.", 0.0),
        _u("u2", "мы очень смеялись", "мы очень смеялись", "Мы очень хохотали.", 60.0),
        _u("u3", "без правок", "без правок", "Без правок.", 120.0),
    ]
    atomic_json(
        tmp_path / "canonical.json",
        {"schema_version": 2, "utterances": [i.to_dict() for i in items]},
    )

    target, totals = write_readable_changes(tmp_path)
    lines = target.read_text(encoding="utf-8").splitlines()

    assert totals == {
        "transliteration": 1,
        "substitution": 1,
        "utterances": 2,
        "needs_attention": 1,
    }
    assert lines[1].startswith("u2,00:01:00,Игорь,substitution,1,yes,")
    assert lines[2].startswith("u1,")
    assert len(lines) == 3


def test_csv_is_written_even_when_nothing_changed(tmp_path):
    """An empty file still means something: the model touched nothing."""
    items = [_u("u1", "без правок", "без правок", "Без правок.")]
    atomic_json(
        tmp_path / "canonical.json",
        {"schema_version": 2, "utterances": [i.to_dict() for i in items]},
    )

    target, totals = write_readable_changes(tmp_path)

    assert target.exists()
    assert totals["utterances"] == 0
    assert target.read_text(encoding="utf-8").splitlines()[0].startswith("utterance_id,")
