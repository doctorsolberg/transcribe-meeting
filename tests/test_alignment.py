from transcribe_meeting.alignment import (
    assign_alt_words,
    assign_words,
    make_utterances,
    mark_overlap,
)
from transcribe_meeting.model import Word


def test_assigns_maximal_overlap_then_nearby_fallback_and_unknown():
    words = [Word(0.1, 0.3, " Привет", 0.9), Word(2.2, 2.3, " мир", 0.8), Word(10, 10.1, " ?", 0.2)]
    result = assign_words(words, [(0, 1, "S1"), (2.4, 3, "S2")])
    assert [word.speaker_id for word in result] == ["S1", "S2", "Speaker ?"]


def test_utterances_break_on_speaker_and_stable_ids():
    words = [
        Word(0, 0.1, " один", speaker_id="S1"),
        Word(0.2, 0.3, " два", speaker_id="S1"),
        Word(0.4, 0.5, " три", speaker_id="S2"),
    ]
    items = make_utterances(
        words,
        source_file="a.m4a",
        source_offset=0,
        speaker_names={"S1": "А", "S2": "Б"},
        prefix="p001",
    )
    assert [item.utterance_id for item in items] == ["p001-u00001", "p001-u00002"]
    assert [item.speaker_name for item in items] == ["А", "Б"]


def test_variants_dict_fills_in_for_every_non_canonical_model():
    """Mirrors core.run_job's per-variant-model loop: each
    non-canonical model is projected in turn, and only contributes a key when it
    actually has words for that utterance — a third model with nothing there leaves
    the dict exactly as the second model left it, key absent rather than empty."""
    utterances = make_utterances(
        [Word(0, 0.4, " один", speaker_id="S1"), Word(3.6, 4.0, " два", speaker_id="S1")],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    utterances[0].source_start, utterances[0].source_end = 1.0, 2.0
    utterances[1].source_start, utterances[1].source_end = 5.0, 6.0
    model_b_words = [
        Word(1.4, 1.6, "b-один", speaker_id="Speaker ?"),
        Word(5.4, 5.6, "b-два", speaker_id="Speaker ?"),
    ]
    # model_c only has words for the first utterance's interval.
    model_c_words = [Word(1.4, 1.6, "c-один", speaker_id="Speaker ?")]

    for model_name, words in (("model_b", model_b_words), ("model_c", model_c_words)):
        assigned = assign_alt_words(words, utterances)
        for utterance in utterances:
            text = assigned.get(utterance.utterance_id)
            if text:
                utterance.variants[model_name] = text

    assert utterances[0].variants == {"model_b": "b-один", "model_c": "c-один"}
    assert utterances[1].variants == {"model_b": "b-два"}


def test_assign_alt_words_center_inside_interval_belongs_there_not_neighbor():
    """A word whose center lands in an interval belongs there even on partial overlap;
    this is what keeps each alt word in exactly one utterance (regression guard for A)."""
    utterances = make_utterances(
        [
            Word(0, 0.4, " один", speaker_id="S1"),
            Word(3.6, 4.0, " шесть", speaker_id="S1"),
        ],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    # These two synthetic utterances span [0, 0.4) and [3.6, 4.0); rebuild with a real
    # gap in between so containment/nearest logic is unambiguous.
    utterances[0].source_start, utterances[0].source_end = 1.0, 2.0
    utterances[1].source_start, utterances[1].source_end = 5.0, 6.0
    alt_words = [
        Word(0.9, 1.3, " в", speaker_id="Speaker ?"),  # center 1.1: inside utterance 0
        Word(1.3, 1.9, " начале", speaker_id="Speaker ?"),  # center 1.6: inside utterance 0
        Word(4.9, 5.3, " в", speaker_id="Speaker ?"),  # center 5.1: inside utterance 1
    ]

    result = assign_alt_words(alt_words, utterances)

    assert result[utterances[0].utterance_id] == "в начале"
    assert result[utterances[1].utterance_id] == "в"


def test_assign_alt_words_attaches_within_tolerance_of_nearest_utterance():
    """A word whose center misses every interval, but is within ALT_ATTACH_TOLERANCE
    (0.5s) of one utterance's boundary, still attaches to it — this is the fix for the
    real-run bug where short utterances lost their alt text entirely."""
    utterances = make_utterances(
        [Word(0, 0.1, " слово", speaker_id="S1")],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    utterances[0].source_start, utterances[0].source_end = 10.0, 10.3  # a 0.3s utterance
    # Center at 10.7: 0.4s past source_end (10.3) — inside the 0.5s tolerance.
    alt_words = [Word(10.6, 10.8, " рядом", speaker_id="Speaker ?")]

    result = assign_alt_words(alt_words, utterances)

    assert result[utterances[0].utterance_id] == "рядом"


def test_assign_alt_words_drops_a_word_farther_than_tolerance():
    utterances = make_utterances(
        [Word(0, 0.1, " слово", speaker_id="S1")],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    utterances[0].source_start, utterances[0].source_end = 10.0, 10.3
    # Center at 11.0: 0.7s past source_end — outside the 0.5s tolerance.
    alt_words = [Word(10.9, 11.1, " далеко", speaker_id="Speaker ?")]

    result = assign_alt_words(alt_words, utterances)

    assert result == {}


def test_assign_alt_words_never_assigns_one_word_to_two_utterances():
    utterances = make_utterances(
        [
            Word(0, 0.1, " раз", speaker_id="S1"),
            Word(5, 5.1, " два", speaker_id="S1"),
        ],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    utterances[0].source_start, utterances[0].source_end = 0.0, 1.0
    utterances[1].source_start, utterances[1].source_end = 1.4, 2.4
    # Sits in the small gap between the two utterances, equidistant-ish; must land in
    # exactly one bucket, never both and never silently duplicated.
    alt_words = [Word(1.15, 1.25, " между", speaker_id="Speaker ?")]

    result = assign_alt_words(alt_words, utterances)

    total_occurrences = sum(1 for text in result.values() if "между" in text)
    assert total_occurrences == 1


def test_assign_alt_words_does_not_alter_utterance_boundaries():
    """The second model only fills in comparison text; it can never add,
    remove, or resize a canonical utterance, no matter how its own words are shaped."""
    utterances = make_utterances(
        [
            Word(0, 0.1, " один", speaker_id="S1"),
            Word(0.2, 0.3, " два", speaker_id="S1"),
            Word(0.4, 0.5, " три", speaker_id="S2"),
        ],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={"S1": "А", "S2": "Б"},
        prefix="p001",
    )
    ids_before = [item.utterance_id for item in utterances]

    # An alt model that disagrees wildly about where words/speakers fall must still
    # never change which utterances exist.
    alt_words = [Word(0.0, 0.5, "совсем другое", speaker_id="S9")]
    assign_alt_words(alt_words, utterances)

    assert [item.utterance_id for item in utterances] == ids_before


def test_mark_overlap_flags_long_intersection_only():
    utterances = make_utterances(
        [Word(0, 1.0, " реплика", speaker_id="S1")],
        source_file="a.m4a",
        source_offset=0,
        speaker_names={},
        prefix="p001",
    )
    utterances[0].source_start, utterances[0].source_end = 10.0, 12.0  # 2s utterance

    # base offset 100: utterance sits at [110, 112) in diarization clock.
    mark_overlap(utterances, [(110.5, 111.5)], {"a.m4a": 100.0})  # 1.0s intersection
    assert utterances[0].overlap is True

    utterances[0].overlap = False
    mark_overlap(utterances, [(111.9, 112.3)], {"a.m4a": 100.0})  # 0.1s intersection
    assert utterances[0].overlap is False


def test_alt_words_keep_model_order_despite_overlapping_timestamps():
    """Whisper emits overlapping word spans; its own sequence is the truthful order."""
    from transcribe_meeting.alignment import assign_alt_words
    from transcribe_meeting.model import Utterance, Word

    utterance = Utterance(
        utterance_id="u1",
        source_file="a.m4a",
        source_start=0.0,
        source_end=10.0,
        meeting_start=0.0,
        meeting_end=10.0,
        speaker_id="S1",
        speaker_name="S1",
        raw_text="попробую потому что она стоит в",
        confidence={},
    )
    # "она" starts earlier than the word the model emitted before it.
    words = [
        Word(start=1.00, end=1.40, text="попробую"),
        Word(start=1.50, end=1.80, text=" потому"),
        Word(start=1.85, end=2.10, text=" что"),
        Word(start=1.60, end=2.40, text=" она"),
        Word(start=2.50, end=2.80, text=" стоит"),
    ]

    result = assign_alt_words(words, [utterance])

    assert result["u1"] == "попробую потому что она стоит"
