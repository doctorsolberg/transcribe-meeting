import json
from types import SimpleNamespace

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
        self.output = [
            SimpleNamespace(
                type="message", content=[SimpleNamespace(type="output_text", text=output_text)]
            )
        ]
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


def test_protected_link_cannot_be_extended_to_a_different_destination():
    item = utterance("u1", "сайт https://example.test/private-route")
    texts, rejected, _ = validate_readable(
        [item],
        [{"utterance_id": "u1", "readable_text": "Сайт https://example.test/private-route-2."}],
        set(),
    )
    assert texts == {"u1": item.raw_text}
    assert "private-route" not in repr(rejected)


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


@pytest.mark.parametrize(
    ("original", "edited"),
    [
        ("я не согласен", "Я согласен."),
        ("да, согласен", "Да, я не согласен."),
        ("не мы одобрили бюджет", "Мы не одобрили бюджет."),
        ("я не согласен но он согласен", "Я согласен, но он не согласен."),
        ("у него и так рак не оно", "У него и так рак но оно."),
        ("we don’t agree", "We do agree."),
        ("there is no delay", "There is a delay."),
        ("я не подпишу", "Я не-подпишу."),
    ],
)
def test_negation_change_keeps_local_text(original, edited):
    item = utterance("u1", original)
    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": edited}], set()
    )
    assert texts == {"u1": original}
    assert rejected == {"u1": ["changed negation or its immediate context"]}


@pytest.mark.parametrize("reply", [".", "—…?!", "Проверили."])
def test_readable_keeps_speech_when_a_reply_drops_most_words(reply):
    item = utterance("u1", "проверили результаты работы всей команды")
    item.variants = {"other": "проверили итог работы нашей группы"}
    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": reply}], set()
    )
    assert texts == {"u1": item.raw_text}
    assert "u1" in rejected


def test_punctuation_and_number_formatting_near_negation_are_allowed():
    item = utterance("u1", "нет не  пять рублей")
    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Нет, не 5 рублей."}], set()
    )
    assert not rejected
    assert texts == {"u1": "Нет, не 5 рублей."}


def test_spelled_out_number_is_digitized_without_rejection():
    """Rule 4A: '130' does not appear literally in a source that said it in words."""
    item = utterance("u1", "приехало сто тридцать человек")

    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": "Приехало 130 человек."}], set()
    )

    assert rejected == {}
    assert texts["u1"] == "Приехало 130 человек."


@pytest.mark.parametrize(
    ("raw", "edited"),
    [
        ("приехало много человек", "Приехало 130 человек."),
        ("встреча будет 15 числа", "Встреча будет 115 числа."),
        ("отправили 42 заявки", "Отправили 42888 заявки."),
        ("отправили 42 заявки", "Отправили 42 заявки и 888."),
        ("5 плюс 5", "5 плюс."),
        ("5 и 10 заявок", "10 и 5 заявок."),
        ("сто тридцать человек", "Сто сорок человек."),
        ("сто тридцать человек", "30 человек."),
        ("цена -15 рублей", "Цена 15 рублей."),
        ("версия 2.1.3", "Версия 2.13."),
        ("модель GPT-5", "Модель GPT-6."),
        ("срок 3–5 дней", "Срок 3, 5 дней."),
        ("двадцать седьмого числа", "27 числа."),
        ("27-го числа", "27-ой числа."),
        ("one hundred thirty people", "Thirty people."),
        ("бюджет - 15 рублей", "Бюджет 15 рублей."),
        ("удержание 10 %", "Удержание 10."),
        ("бюджет 25 ₽", "Бюджет 25 $."),
        ("аня 5 борис 10", "Борис 5, Аня 10."),
        ("выделили один миллион рублей", "Выделили один миллиард рублей."),
        ("plan two million dollars", "Plan two billion dollars."),
    ],
)
def test_changed_number_falls_back_with_sanitized_diagnostic(raw, edited):
    item = utterance("u1", raw)
    texts, rejected, _ = validate_readable(
        [item], [{"utterance_id": "u1", "readable_text": edited}], set()
    )
    assert texts == {"u1": raw}
    assert rejected == {"u1": ["changed numeric content"]}


@pytest.mark.parametrize(
    ("raw", "edited"),
    [
        ("двадцать седьмого числа", "27-го числа."),
        ("приехали с тридцатью человеками", "Приехали с 30 человеками."),
        ("две тысячи двадцать шестой год", "2026 год."),
        ("one hundred thirty people", "130 people."),
        ("срок 3–5 дней", "Срок 3–5 дней."),
    ],
)
def test_proven_number_formatting_is_preserved(raw, edited):
    texts, rejected, _ = validate_readable(
        [utterance("u1", raw)], [{"utterance_id": "u1", "readable_text": edited}], set()
    )
    assert not rejected
    assert texts == {"u1": edited}


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

    assert rejected["u1"] == ["changed numeric content"]
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

    usage = {}
    result = _fuse_batch(client, "gpt-test", "SYSTEM", [item], set(), usage=usage)

    assert result.texts == {"u1": "Текст."}
    assert result.concerns == []
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

    _fuse_batch(client, "gpt-test", "SYSTEM", [item], set(), usage={})

    assert "reasoning" not in client.responses.calls[0]


def test_fuse_batch_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    monkeypatch.delenv(READABLE_REASONING_ENV, raising=False)
    item = utterance("u1", "текст")
    client = _FakeClient([_FakeResponse("not json"), _FakeResponse(_valid_payload_json())])

    result = _fuse_batch(client, "gpt-test", "SYSTEM", [item], set(), usage={})

    assert result.texts == {"u1": "Текст."}
    assert result.concerns == []
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
        _fuse_batch(client, "gpt-test", "SYSTEM", [item], set(), usage={})

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

    result = _fuse_batch(client, "gpt-test", "SYSTEM", [protected, repaired], set(), usage={})

    assert result.texts["u1"] == "встреча пройдёт 15 числа"  # fell back to the local text
    assert result.texts["u2"] == "Усвоил материал."
    fallback_notes = [c for c in result.concerns if c["utterance_id"] == "u1"]
    repair_notes = [c for c in result.concerns if c["utterance_id"] == "u2"]
    assert len(fallback_notes) == 1 and fallback_notes[0]["note"].startswith(CONCERN_FALLBACK)
    assert len(repair_notes) == 1 and repair_notes[0]["note"].startswith(CONCERN_REPAIR)


def test_validator_counts_repaired_utterances_not_words_or_provider_notes():
    items = [utterance("u1", "усвол материал в ранскрибаторе"), utterance("u2", "все хорошо")]
    reply = ReadablePayload.model_validate(
        {
            "items": [
                {
                    "utterance_id": "u1",
                    "readable_text": "Усвоил материал в транскрибаторе.",
                    "concerns": [],
                },
                {
                    "utterance_id": "u2",
                    "readable_text": "Всё хорошо.",
                    "concerns": [f"{CONCERN_REPAIR}: one", f"{CONCERN_REPAIR}: two"],
                },
            ]
        }
    )
    result = _fuse_batch(
        _FakeClient([_FakeResponse(reply.model_dump_json())]), "m", "prompt", items, set(), usage={}
    )
    assert result.repaired_utterances == 1


def test_novel_word_is_audited_separately_from_a_proven_near_match():
    item = utterance("u1", "мы очень смеялись")
    client = _FakeClient([_FakeResponse(_valid_payload_json(text="Мы очень хохотали."))])
    result = _fuse_batch(client, "m", "prompt", [item], set(), usage={})
    assert result.repaired_utterances == 0
    assert result.concerns == [
        {"utterance_id": "u1", "note": "words outside the variants: +хохотали"}
    ]


@pytest.mark.parametrize(
    ("raw", "edited", "kinds", "removed"),
    [
        ("я помню да", "Я помню.", ["loss"], "да"),
        ("я помню да", "Я помню, помню.", ["loss"], "да"),
        ("мы снова снова здесь", "Мы снова здесь.", ["loss"], "снова"),
        ("шли в лес впереди", "Шли впереди.", ["loss", "loss"], "лес"),
        (
            "мы шли на запад и смеялись",
            "Мы шли и хохотали.",
            ["substitution", "loss", "loss"],
            "запад",
        ),
        ("это сверх сверхсложный тест", "Это сверхсложный тест.", ["loss"], "сверх"),
        ("с супер разных сторон", "С суперразных сторон.", [], None),
        ("сто тридцать человек", "130 человек.", ["number"], None),
    ],
)
def test_removal_audit_tracks_exact_tokens_and_repetitions(raw, edited, kinds, removed):
    from transcribe_meeting.readable import classify_changes

    item = utterance("u1", raw)
    item.readable_text = edited
    assert classify_changes(item) == kinds
    result = _fuse_batch(
        _FakeClient([_FakeResponse(_valid_payload_json(text=edited))]),
        "m",
        "prompt",
        [item],
        set(),
        usage={},
    )
    assert any(c["note"].startswith("removed source words:") for c in result.concerns) == (
        removed is not None
    )
    if removed:
        assert any(
            removed in c["note"]
            for c in result.concerns
            if c["note"].startswith("removed source words:")
        )


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
