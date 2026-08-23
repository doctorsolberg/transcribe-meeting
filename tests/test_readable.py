import json

import pytest

from transcribe_meeting.model import Utterance
from transcribe_meeting.readable import (
    CONCERN_FALLBACK,
    CONCERN_REPAIR,
    READABLE_REASONING_ENV,
    _fuse_batch,
    build_payload,
    readable_reasoning,
    validate_readable,
)
from transcribe_meeting.schemas import ReadablePayload


class _FakeUsage:
    """Stands in for the SDK's usage object, which exposes model_dump()."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeResponse:
    def __init__(self, output_text, usage=None):
        self.output_text = output_text
        self.usage = _FakeUsage(usage) if usage is not None else None


class _FakeResponses:
    """Records every responses.create() call and replays canned answers in order."""

    def __init__(self, outputs):
        self.calls: list[dict] = []
        self._outputs = list(outputs)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._outputs.pop(0)


class _FakeClient:
    def __init__(self, outputs):
        self.responses = _FakeResponses(outputs)


def _valid_payload_json(utterance_id="u1", text="Текст.", concerns=None):
    return json.dumps(
        {
            "items": [
                {
                    "utterance_id": utterance_id,
                    "readable_text": text,
                    "concerns": concerns or [],
                }
            ]
        }
    )


def utterance(identifier="u1", text="В 2026 году сайт https://example.test"):
    return Utterance(identifier, "a", 0, 1, 0, 1, "S1", "S1", text, {})


def _alt_utterance(identifier, raw, alt_raw, model_name="model_b"):
    item = Utterance(identifier, "a", 0, 1, 0, 1, "S1", "S1", raw, {})
    if alt_raw is not None:
        item.variants[model_name] = alt_raw
    return item


def test_readable_rejects_lost_or_reordered_ids():
    """The id check is the one failure a retry of the same batch can fix, so it is
    the only thing left that raises rather than degrading a single utterance."""
    values = [utterance("u1", "Нужно 42"), utterance("u2", "Готово")]
    with pytest.raises(ValueError):
        validate_readable(
            values,
            [
                {"utterance_id": "u2", "readable_text": "Готово"},
                {"utterance_id": "u1", "readable_text": "Нужно 42"},
            ],
            set(),
        )


def test_readable_accepts_preserved_protected_data():
    item = utterance()
    texts, rejected, _ = validate_readable(
        [item],
        [{"utterance_id": "u1", "readable_text": "В 2026 году сайт https://example.test."}],
        set(),
    )

    assert texts == {"u1": "В 2026 году сайт https://example.test."}
    assert rejected == {}


def test_payload_omits_b_when_no_alt_or_texts_match():
    same = _alt_utterance("u1", "текст совпал", "текст совпал")
    missing = _alt_utterance("u2", "только a", None)

    payload = build_payload([same, missing], ["model_b"])

    assert payload == [
        {"id": "u1", "speaker": "S1", "a": "текст совпал"},
        {"id": "u2", "speaker": "S1", "a": "только a"},
    ]


def test_payload_includes_b_when_texts_differ():
    item = _alt_utterance("u1", "версия а", "версия б")

    payload = build_payload([item], ["model_b"])

    assert payload == [{"id": "u1", "speaker": "S1", "a": "версия а", "b": "версия б"}]


def test_payload_includes_c_for_the_third_model_and_omits_it_when_it_matches_a():
    differs = Utterance("u1", "a", 0, 1, 0, 1, "S1", "S1", "версия а", {})
    differs.variants = {"model_b": "версия б", "model_c": "версия в"}
    matches_a = Utterance("u2", "a", 0, 1, 0, 1, "S1", "S1", "версия а", {})
    matches_a.variants = {"model_b": "версия б", "model_c": "версия а"}

    payload = build_payload([differs, matches_a], ["model_b", "model_c"])

    assert payload == [
        {"id": "u1", "speaker": "S1", "a": "версия а", "b": "версия б", "c": "версия в"},
        {"id": "u2", "speaker": "S1", "a": "версия а", "b": "версия б"},
    ]


def test_yo_is_spelling_not_an_invented_word():
    """ASR writes 'все'; restoring 'всё' is readability, exactly what we asked for."""
    item = utterance("u1", "сейчас все записываю")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Сейчас всё записываю."}], set()
    )

    assert texts == {"u1": "Сейчас всё записываю."}
    assert rejected == {}


def test_readable_accepts_a_word_that_only_appears_in_b():
    item = _alt_utterance(
        "u1", "мы очень смеялисьс этой транскрибацие", "мы очень смеялись с этой транскрибацией"
    )
    texts, rejected, _ = validate_readable(
        [item],
        [{"utterance_id": "u1", "readable_text": "Мы очень смеялись с этой транскрибацией."}],
        set(),
    )

    assert texts == {"u1": "Мы очень смеялись с этой транскрибацией."}
    assert rejected == {}


def test_protected_number_present_only_in_b_counts_as_preserved():
    item = _alt_utterance("u1", "цена сто рублей", "цена 100 рублей")
    texts, _, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Цена 100 рублей."}], set()
    )

    assert texts == {"u1": "Цена 100 рублей."}


def test_readable_accepts_a_word_that_only_appears_in_c():
    """A third model's variant must widen the allowed word set exactly like b does."""
    item = Utterance("u1", "a", 0, 1, 0, 1, "S1", "S1", "попробовал виспер", {})
    item.variants = {"model_b": "попробовал виспер", "model_c": "попробовал Whisper"}

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Попробовал Whisper."}], set()
    )

    assert texts == {"u1": "Попробовал Whisper."}
    assert rejected == {}


def test_hyphenating_two_source_words_is_spelling_not_invention():
    """Correct Russian joins what the ASR split: 'что то' -> 'что-то'."""
    item = utterance("u1", "это какой то что то вопрос")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Это какой-то что-то вопрос."}], set()
    )

    assert rejected == {}
    assert texts["u1"] == "Это какой-то что-то вопрос."


def test_joining_two_source_words_without_a_hyphen_is_also_allowed():
    item = utterance("u1", "с супер разных сторон")

    _, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "С суперразных сторон."}], set()
    )

    assert rejected == {}


def test_hyphenating_one_glued_source_word_is_allowed():
    """'петпроекта' -> 'пет-проекта' adds a separator, not a letter."""
    item = utterance("u1", "гермес тоже для петпроекта")

    _, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Гермес тоже для пет-проекта."}], set()
    )

    assert rejected == {}


def test_obvious_one_letter_repair_is_accepted_and_logged():
    """A mechanical near-miss check replaces trusting the model's stated confidence."""
    item = utterance("u1", "стоит в биржевском ранскрибаторе встреч")

    texts, rejected, repairs = validate_readable(
        [item],
        [{"utterance_id": "u1", "readable_text": "Стоит в биржевском транскрибаторе встреч."}],
        set(),
    )

    assert rejected == {}
    assert repairs == {"u1": ["ранскрибаторе → транскрибаторе"]}
    assert texts["u1"] == "Стоит в биржевском транскрибаторе встреч."


def test_a_single_substitution_is_logged_but_allowed():
    """Per-word veto is gone by owner decision; the change must stay visible."""
    item = utterance("u1", "мы очень смеялись")

    texts, rejected, repairs = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Мы очень хохотали."}], set()
    )

    assert rejected == {}
    assert repairs == {"u1": ["+хохотали"]}
    assert texts["u1"] == "Мы очень хохотали."


def test_a_wholesale_rewrite_still_falls_back():
    """Above the proportionality limit it is no longer an edit but a rewrite."""
    item = utterance("u1", "мы очень долго смеялись над этой смешной транскрибацией")
    item.light_text = "мы очень долго смеялись над этой смешной транскрибацией."

    texts, rejected, _ = validate_readable(
        [item],
        [{"utterance_id": "u1", "readable_text": "Участники обсудили качество работы системы."}],
        set(),
    )

    assert "u1" in rejected
    assert texts["u1"] == "мы очень долго смеялись над этой смешной транскрибацией."


def test_transliteration_of_a_known_term_is_allowed_and_logged():
    item = utterance("u1", "попробовал виспер попробовал дипграм и ну такое")

    texts, rejected, repairs = validate_readable(
        [item],
        [
            {
                "utterance_id": "u1",
                "readable_text": "Попробовал Whisper, попробовал Deepgram, и ну такое.",
            }
        ],
        set(),
    )

    assert rejected == {}
    assert sorted(repairs["u1"]) == ["+deepgram", "+whisper"]
    assert "Whisper" in texts["u1"]


def test_short_word_swaps_are_logged_as_substitutions_not_repairs():
    """'не'->'но' is a meaning change; it may pass now, but never silently."""
    item = utterance("u1", "у него и так рак не оно")

    _, _, repairs = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "У него и так рак но оно."}], set()
    )

    assert repairs == {"u1": ["+но"]}


def test_spelled_out_number_is_digitized_without_rejection():
    """Rule 4A: '130' does not appear literally in a source that said it in words."""
    item = utterance("u1", "приехало сто тридцать человек")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Приехало 130 человек."}], set()
    )

    assert rejected == {}
    assert texts["u1"] == "Приехало 130 человек."


def test_digit_token_is_always_accepted_even_without_a_matching_source_numeral():
    """The model may normalize a number from a fused variant or context not literally
    present in this utterance's own text; the digit-token exception is unconditional."""
    item = utterance("u1", "приехало много человек")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Приехало 130 человек."}], set()
    )

    assert rejected == {}
    assert texts["u1"] == "Приехало 130 человек."


def test_a_fabricated_word_is_still_flagged_not_silently_allowed_like_a_digit():
    item = utterance("u1", "мы обсудили проект")

    _, _, repairs = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Мы обсудили мегапроект."}], set()
    )

    assert repairs == {"u1": ["+мегапроект"]}


def test_a_number_heard_only_by_one_variant_no_longer_fails_the_batch():
    """Regression for a real incident on a three-model run: a batch
    of 150 utterances failed whole because one utterance's canonical text spelled a
    number out in words while a variant model heard it as a digit, and the old check
    validated protected values against the union of every variant instead of only
    the canonical text — so a digit only Whisper heard was still "protected" and had
    to survive fusion verbatim. Now the canonical text has no digit to protect, so
    fusion is accepted here exactly like any other utterance, and every other
    utterance in the batch keeps its fused text too instead of all degrading."""
    problem = _alt_utterance(
        "u1", "да сто процентов", "Да, 100 процентов", model_name="groq-whisper-large-v3"
    )
    others = [utterance(f"u{i}", f"реплика номер {i}") for i in range(2, 151)]
    batch = [problem, *others]
    items = [{"utterance_id": "u1", "readable_text": "Да, сто процентов."}] + [
        {"utterance_id": item.utterance_id, "readable_text": f"Реплика номер {i}."}
        for i, item in enumerate(others, start=2)
    ]

    texts, rejected, _ = validate_readable(batch, items, set())

    assert rejected == {}
    assert texts["u1"] == "Да, сто процентов."
    assert all(
        texts[item.utterance_id] == f"Реплика номер {i}." for i, item in enumerate(others, start=2)
    )


def test_a_lost_protected_value_degrades_only_that_utterance():
    """Proof that degradation is per-utterance: the second utterance in the same
    batch keeps its fused text even though the first one falls back."""
    lost_number = utterance("u1", "встреча пройдёт 15 числа")
    normal = utterance("u2", "все согласны")

    texts, rejected, _ = validate_readable(
        [lost_number, normal],
        [
            {"utterance_id": "u1", "readable_text": "Встреча пройдёт в понедельник."},
            {"utterance_id": "u2", "readable_text": "Все согласны."},
        ],
        set(),
    )

    assert rejected["u1"] == ["lost protected value '15'"]
    assert texts["u1"] == "встреча пройдёт 15 числа"
    assert texts["u2"] == "Все согласны."


def test_empty_readable_text_degrades_instead_of_raising():
    item = utterance("u1", "привет всем")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "   "}], set()
    )

    assert rejected["u1"] == ["the model returned empty text"]
    assert texts["u1"] == "привет всем"


def test_readable_reasoning_none_disables_the_field(monkeypatch):
    monkeypatch.setenv(READABLE_REASONING_ENV, "none")

    assert readable_reasoning() is None


def test_readable_reasoning_defaults_to_medium(monkeypatch):
    monkeypatch.delenv(READABLE_REASONING_ENV, raising=False)

    assert readable_reasoning() == {"effort": "medium"}


def test_readable_reasoning_rejects_an_unknown_level(monkeypatch):
    monkeypatch.setenv(READABLE_REASONING_ENV, "bogus")

    with pytest.raises(ValueError):
        readable_reasoning()


def test_fuse_batch_sends_the_documented_request_shape(monkeypatch):
    """The only network contract behind readable quality: model, strict json_schema
    matching ReadablePayload, store=False, and instructions plus build_payload as
    input. Reasoning defaults to medium, so the field must be present here too."""
    monkeypatch.delenv(READABLE_REASONING_ENV, raising=False)
    item = utterance("u1", "текст")
    client = _FakeClient(
        [_FakeResponse(_valid_payload_json(), usage={"input_tokens": 10, "output_tokens": 5})]
    )

    texts, concerns, usage = _fuse_batch(client, "gpt-test", "SYSTEM", [item], set())

    assert texts == {"u1": "Текст."}
    assert concerns == []
    assert usage == {"input_tokens": 10, "output_tokens": 5}
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "gpt-test"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "medium"}
    text_format = request["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] == ReadablePayload.model_json_schema()
    assert request["input"][0] == {"role": "system", "content": "SYSTEM"}
    assert request["input"][1]["role"] == "user"
    assert json.loads(request["input"][1]["content"]) == build_payload([item], ())


def test_fuse_batch_omits_reasoning_when_level_is_none(monkeypatch):
    monkeypatch.setenv(READABLE_REASONING_ENV, "none")
    item = utterance("u1", "текст")
    client = _FakeClient([_FakeResponse(_valid_payload_json())])

    _fuse_batch(client, "gpt-test", "SYSTEM", [item], set())

    assert "reasoning" not in client.responses.calls[0]


def test_fuse_batch_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    monkeypatch.delenv(READABLE_REASONING_ENV, raising=False)
    item = utterance("u1", "текст")
    client = _FakeClient([_FakeResponse("not json"), _FakeResponse(_valid_payload_json())])

    texts, concerns, _usage = _fuse_batch(client, "gpt-test", "SYSTEM", [item], set())

    assert texts == {"u1": "Текст."}
    assert concerns == []
    assert len(client.responses.calls) == 2
    first_system = client.responses.calls[0]["input"][0]["content"]
    second_system = client.responses.calls[1]["input"][0]["content"]
    assert first_system == "SYSTEM"
    assert second_system.startswith("SYSTEM")
    assert second_system != "SYSTEM"


def test_fuse_batch_raises_after_two_invalid_responses(monkeypatch):
    monkeypatch.delenv(READABLE_REASONING_ENV, raising=False)
    item = utterance("u1", "текст")
    client = _FakeClient([_FakeResponse("not json"), _FakeResponse("still not json")])

    with pytest.raises(ValueError):
        _fuse_batch(client, "gpt-test", "SYSTEM", [item], set())

    assert len(client.responses.calls) == 2


def test_fuse_batch_reports_fallback_and_repair_concerns_with_their_prefixes():
    """rejected/repairs from validate_readable must reach the caller as notes with
    the CONCERN_FALLBACK/CONCERN_REPAIR prefixes core.run_readable counts on."""
    protected = utterance("u1", "встреча пройдёт 15 числа")
    repaired = utterance("u2", "усвол материал")
    payload = json.dumps(
        {
            "items": [
                {
                    "utterance_id": "u1",
                    "readable_text": "Встреча пройдёт в понедельник.",
                    "concerns": [],
                },
                {"utterance_id": "u2", "readable_text": "Усвоил материал.", "concerns": []},
            ]
        }
    )
    client = _FakeClient([_FakeResponse(payload)])

    texts, concerns, _usage = _fuse_batch(
        client, "gpt-test", "SYSTEM", [protected, repaired], set()
    )

    assert texts["u1"] == "встреча пройдёт 15 числа"  # fell back to the local text
    assert texts["u2"] == "Усвоил материал."
    fallback_notes = [c for c in concerns if c["utterance_id"] == "u1"]
    repair_notes = [c for c in concerns if c["utterance_id"] == "u2"]
    assert len(fallback_notes) == 1 and fallback_notes[0]["note"].startswith(CONCERN_FALLBACK)
    assert len(repair_notes) == 1 and repair_notes[0]["note"].startswith(CONCERN_REPAIR)


def test_calibrated_model_order_produces_no_warning():
    from transcribe_meeting.readable import CALIBRATED_FUSION_ORDER, uncalibrated_fusion_order

    assert uncalibrated_fusion_order(list(CALIBRATED_FUSION_ORDER)) is None
    assert uncalibrated_fusion_order([*CALIBRATED_FUSION_ORDER, "groq-whisper-large-v3"]) is None
    # One model means no fusion at all, so rule 3 never applies.
    assert uncalibrated_fusion_order(["v3_rnnt"]) is None


def test_reversed_model_order_warns_that_rule_three_is_uncalibrated():
    """Reordering --asr-models silently inverts the prompt's claim about which model
    is better at English terms and which at Russian morphology."""
    from transcribe_meeting.readable import uncalibrated_fusion_order

    warning = uncalibrated_fusion_order(["v3_rnnt", "multilingual_large_ctc"])

    assert warning is not None
    assert "multilingual_large_ctc,v3_rnnt" in warning
