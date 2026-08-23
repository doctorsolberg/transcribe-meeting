from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .model import Utterance

# Verbatim: written for a cheap model, with numbered rules and
# concrete divergence examples measured on a real recording.
# Do not rewrite, shorten, or "improve" the wording — for the
# manual-diff evidence backing rule 3's asymmetry.
READABLE_FUSION_INSTRUCTIONS = """Ты редактируешь стенограмму русской деловой встречи. Работай механически по правилам.

ЧТО ТЫ ПОЛУЧАЕШЬ
Массив реплик. У каждой есть id, speaker и текст a. У части реплик есть ещё текст b,
у части — ещё и текст c. a, b и c — это ОДНА И ТА ЖЕ фраза, распознанная разными
программами. Они расходятся только там, где одна из программ ошиблась.

ЧТО ТЫ ДЕЛАЕШЬ
Для каждой реплики верни readable_text — исправленный текст этой же фразы.

ПРАВИЛО 1. Если есть только a — возьми a.
ПРАВИЛО 2. Если есть a и b — иди по фразе слово за словом. Там, где слова совпадают,
бери их как есть. Там, где расходятся, выбери ОДИН из двух вариантов.

ПРАВИЛО 2А. У части реплик есть третий текст c. Это распознавание третьей
программой, у неё другие сильные и слабые стороны: она ХУЖЕ слышит русские
окончания, но ЛУЧШЕ узнаёт английские термины, названия продуктов и аббревиатуры.
Используй c так: если a и b расходятся между собой и оба варианта выглядят
испорченными, посмотри, что в c — особенно если речь об английском термине.
Если a и b согласны между собой, а c отличается — доверяй a и b.

ПРАВИЛО 3. Как выбирать при расхождении:
  - Английские слова, термины, названия программ и аббревиатуры — почти всегда
    правильнее в a. Примеры: «ллмку» (a) вернее «лмку» (b); «дипграм» вернее «дип грам»;
    «стейкхолдеров» вернее «стек холдеров»; «эндиэй» вернее «энд».
  - Русские окончания, согласование и разделение слипшихся слов — почти всегда
    правильнее в b. Примеры: «конфиденциальная» (b) вернее «кониденциальная» (a);
    «смеялись с» вернее «смеялисьс»; «одной стороны» вернее «однойстороы»;
    «усвоил» вернее «усвол».
  - Если правила не помогают, реши по смыслу разговора: выбери тот вариант,
    который осмысленнее в этой фразе и согласуется с темой встречи.
    Если и это не помогает — бери вариант из a.
ПРАВИЛО 4. Расставь заглавные буквы в начале предложений и в именах собственных.
Расставь точки, запятые, вопросительные знаки.
ПРАВИЛО 4А. ЧИСЛА ПИШИ ЦИФРАМИ. Если число произнесено словами, запиши его цифрами:
«сто тридцать человек» → «130 человек»; «двадцать седьмого числа» → «27-го числа»;
«тридцатью человеками» → «30 человеками»; «две тысячи двадцать шестой год» → «2026 год».
Значение менять нельзя — только запись. Если не уверен, какое именно число прозвучало,
оставь словами.
Это не относится к числам внутри устойчивых выражений («в двух словах», «на все сто»),
к «один» и «одна» в роли артикля («один мой знакомый») и к номерам версий и моделей,
которые уже записаны как есть.
ПРАВИЛО 5. Ничего не сокращай, не пересказывай, не переставляй и не дописывай.
Каждое слово результата обязано встречаться в a, в b или в c. Числа, имена, названия,
ссылки и адреса почты переноси символ в символ.
ПРАВИЛО 5А. ЧИНИ ИСПОРЧЕННЫЕ СЛОВА. Если слово искажено во всех вариантах, но ты
ТОЧНО понимаешь по разговору, какое слово имелось в виду, — напиши правильное.
Примеры: «ранскрибаторе» → «транскрибаторе»; «саморизирую» → «саммаризирую»;
«кониденциальная» → «конфиденциальная»; «горазт» → «горазд».
Порог: ты чинишь, только если уверен настолько, что готов поручиться. Малейшее
сомнение — оставь как было. Лучше оставить корявое слово, чем поставить неверное.

ПРАВИЛО 5Б. АНГЛИЙСКИЕ ТЕРМИНЫ ПИШИ ЛАТИНИЦЕЙ. Если слово в любом из вариантов — это
русская запись известного английского термина, названия продукта или компании,
верни его нормальным английским написанием.
Примеры: «дипграм» → «Deepgram»; «виспер» → «Whisper»; «гигаам» → «GigaAM»;
«ллм» → «LLM»; «стейчгейт» → «Stage-Gate»; «эндиэй» → «NDA»; «сиэрэмка» → «CRM»;
«тиэндм» → «T&M»; «пруф оф концепт» → «proof of concept»; «стейкхолдер» →
«стейкхолдер» (это уже прижившееся русское слово, оставь как есть).
Термин должен быть тебе ИЗВЕСТЕН. Не выдумывай английское слово по созвучию:
если не узнаёшь термин — оставь русскую запись без изменений.
Одно и то же название пиши одинаково во всех репликах.

ПРАВИЛО 5В. ЧЕГО НЕЛЬЗЯ НИКОГДА, даже если очень хочется:
  - менять числа, даты, суммы, имена людей, ссылки и адреса почты — только копия;
  - заменять слово другим словом по смыслу («смеялись» → «хохотали» ЗАПРЕЩЕНО);
  - переписывать фразу своими словами, даже если исходная звучит коряво;
  - брать слова из описания встречи: оно объясняет тему, но люди его не произносили;
  - дописывать то, чего не было, и убирать то, что было.
Ты редактор, а не соавтор. Если фраза непонятна — оставь её непонятной и напиши
об этом в concerns.

ПРАВИЛО 5Г. Каждое изменённое слово — и починку, и перевод в латиницу — перечисли
в concerns этой реплики в виде «было → стало».
ПРАВИЛО 6. Не объединяй и не разделяй реплики. Сколько реплик получил — столько верни,
в том же порядке, с теми же id.

ПРИМЕР
Вход:  {"id": "u1", "speaker": "S1", "a": "мы очень смеялисьс этой транскрибацие",
        "b": "мы очень смеялись с этой транскрибацией"}
Выход: {"utterance_id": "u1", "readable_text": "Мы очень смеялись с этой транскрибацией.",
        "concerns": []}

Верни JSON: {"items": [{"utterance_id": ..., "readable_text": ..., "concerns": [...]}]}
В concerns клади короткие пометки, если фраза осталась непонятной. Если всё ясно — [].
"""


@dataclass
class ReadableResult:
    texts: dict[str, str]
    concerns: list[dict[str, Any]]
    usage: dict[str, Any]


def openai_client_options() -> dict[str, str]:
    """Pass an optional OpenAI-compatible base URL without changing OpenAI defaults."""
    options = {"api_key": os.environ["OPENAI_API_KEY"]}
    if base_url := os.environ.get("OPENAI_BASE_URL", "").strip():
        options["base_url"] = base_url
    return options


#: Reasoning effort for the fusion request. "none" omits the field entirely, which is
#: what an OpenAI-compatible endpoint without reasoning support needs (auth
#: set-openai-base-url is a documented feature).
READABLE_REASONING_ENV = "TRANSCRIBE_READABLE_REASONING"
READABLE_REASONING_LEVELS = ("none", "low", "medium", "high")


def readable_reasoning() -> dict[str, str] | None:
    level = os.environ.get(READABLE_REASONING_ENV, "medium").strip().lower() or "medium"
    if level not in READABLE_REASONING_LEVELS:
        raise ValueError(
            f"{READABLE_REASONING_ENV}={level!r} is not supported, allowed values: "
            + ", ".join(READABLE_REASONING_LEVELS)
        )
    if level == "none":
        return None
    return {"effort": level}


_LATIN = re.compile("[a-z]")

# Rule 4A: a digit token ('130', '27-го', '130-й') never appears
# literally in a source that said the number in words ('сто тридцать'), so word
# membership cannot be the test for it. This is deliberately narrow — digits only,
# optionally one trailing case-ending after a hyphen — so it cannot smuggle through
# a fabricated word that merely starts with a digit.
_DIGIT_TOKEN = re.compile(r"^\d+(?:[.,]\d+)?(?:-[а-яё]+)?$")


def _is_digit_token(word: str) -> bool:
    return bool(_DIGIT_TOKEN.match(word))


def _protected(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)?\b|https?://\S+|[\w.+-]+@[\w.-]+", text)


def _normalize_word(token: str) -> str:
    """Strip edge punctuation and case so 'Мы,' matches source token 'мы'.

    ё folds to е: ASR writes 'все', restoring it to 'всё' is spelling, not a new word,
    and rejecting it would punish the model for improving readability as instructed.
    """
    return token.strip(" \t.,!?…:;\"'()«»—-").lower().replace("ё", "е")


def _word_set(text: str) -> set[str]:
    return {word for token in text.split() if (word := _normalize_word(token))}


#: Above this share of words absent from both variants an utterance reads as a
#: rewrite rather than an edit, and falls back to the local text instead.
_MAX_CHANGED_SHARE = 0.4


def _edit_distance(left: str, right: str, limit: int) -> int:
    """Levenshtein distance, giving up as soon as it exceeds *limit*."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _repair_budget(word: str) -> int:
    """How far a repaired word may sit from the source word it fixes.

    Scaled by length on purpose: one letter rewrites a short word into a different
    one ('не'/'но', 'рак'/'так'), so short words get no budget at all, while a long
    mangled term can only be a distortion of the word it nearly matches.
    """
    if len(word) <= 4:
        return 0
    return 1 if len(word) <= 8 else 2


def _repaired_from(word: str, allowed: set[str]) -> str | None:
    """Return the source word this one plainly repairs, or None.

    A mechanical check replaces the model's own claim of confidence: whatever it
    asserts, a word only passes if it is provably a near-miss of something actually
    spoken. This also blocks vocabulary lifted from the meeting context, which is
    never a near-miss of the audio.
    """
    budget = _repair_budget(word)
    if not budget:
        return None
    best = min(
        (candidate for candidate in allowed if abs(len(candidate) - len(word)) <= budget),
        key=lambda candidate: _edit_distance(word, candidate, budget),
        default=None,
    )
    if best is None or _edit_distance(word, best, budget) > budget:
        return None
    return best


def _is_allowed_word(word: str, allowed: set[str]) -> bool:
    """Accept a word that introduces no content the source did not already have.

    Rule 4 asks for correct spelling, and correct Russian joins tokens the ASR left
    apart: 'что то' becomes 'что-то', 'супер разных' becomes 'суперразных'. Joining
    adjacent source words invents nothing, so it must not read as a fabrication.
    """
    if word in allowed:
        return True
    if _is_digit_token(word):
        # Rule 4A's exception to word membership: a digitized number is a spelling
        # choice, not new content, whether or not the spelled-out words happen to
        # also be present. Letters are never covered by this branch.
        return True
    parts = [part for part in word.split("-") if part]
    if len(parts) > 1:
        if all(part in allowed for part in parts):
            return True
        # The mirror case: hyphenating one glued source token ('петпроекта' ->
        # 'пет-проекта') adds a separator, never a letter.
        if "".join(parts) in allowed:
            return True
    return any(word[:cut] in allowed and word[cut:] in allowed for cut in range(1, len(word)))


#: The model order rule 3 was calibrated on: "a" is the slot whose English terms and
#: abbreviations the prompt trusts, "b" the slot whose Russian morphology it trusts.
#: Those claims come from a manual diff of 180 divergences on a real recording between
#: exactly these two checkpoints — reorder --asr-models and the prompt starts telling
#: the model the opposite of what is true.
CALIBRATED_FUSION_ORDER = ("multilingual_large_ctc", "v3_rnnt")


def uncalibrated_fusion_order(asr_models: Sequence[str]) -> str | None:
    """Return a warning when the model order does not match what the prompt assumes.

    Silent quality loss otherwise: the fusion still runs, still validates and still
    produces plausible text, only with rule 3 pointing at the wrong slot.
    """
    if len(asr_models) < 2 or tuple(asr_models[:2]) == CALIBRATED_FUSION_ORDER:
        return None
    return (
        "readable: the fusion rules are calibrated for the order "
        f"--asr-models {','.join(CALIBRATED_FUSION_ORDER)}, but the first two models are "
        f"{', '.join(asr_models[:2])}. Fusion will still run, but the prompt's hints about "
        "which model is more accurate on anglicisms and which on Russian word endings have "
        "not been verified for this order."
    )


#: Payload keys for non-canonical models, in config["asr_models"] order (minus the
#: canonical, first, model). At most 3 ASR models run in total (asr.validate_asr_models),
#: so at most 2 variant slots ever exist, and the prompt only ever needs to name b/c.
_VARIANT_PAYLOAD_KEYS = "bc"


def build_payload(
    utterances: list[Utterance], variant_models: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Build the fusion payload described in the prompt: id/speaker/a and, only
    when it adds information, b (and c, for a third model). variant_models must be
    config["asr_models"][1:], in order, so "b"/"c" line up with the prompt's rule 3
    (b) and rule 2A (c). Omitting an identical or absent variant saves tokens and
    removes a choice the cheap model would otherwise have to make for nothing.
    """
    payload = []
    for utterance in utterances:
        item: dict[str, Any] = {
            "id": utterance.utterance_id,
            "speaker": utterance.speaker_id,
            "a": utterance.raw_text,
        }
        for key, model_name in zip(_VARIANT_PAYLOAD_KEYS, variant_models, strict=False):
            text = utterance.variants.get(model_name)
            if text and text != utterance.raw_text:
                item[key] = text
        payload.append(item)
    return payload


def _source_union(utterance: Utterance) -> str:
    """Canonical text plus every variant model's text present, space-joined.

    Word-membership checks (validate_readable's allowed set, classify_changes'
    allowed set) must accept a word regardless of which of the up-to-three models
    contributed it — generalizing the old two-field "a plus b" union.
    """
    parts = [utterance.raw_text, *utterance.variants.values()]
    return " ".join(part for part in parts if part)


RISKY_CHANGE_KINDS = frozenset({"substitution", "loss"})

#: Prefixes of the concern notes core.run_readable counts, so neither side has to
#: guess at a substring of a sentence that may be reworded later.
CONCERN_FALLBACK = "kept the local text"
CONCERN_REPAIR = "repaired by closeness to the source word"


def classify_changes(utterance: Utterance) -> list[str]:
    """Classify every word the LLM changed in one utterance, without asking a model.

    An LLM auditing an LLM would share its failure modes, so this is arithmetic over
    two texts: edit distance for a repair, a Latin character for a transliteration,
    set difference for a word that vanished. A Cyrillic term swapped for another
    Cyrillic term lands in "substitution" on purpose — that is the case a human must read.
    """
    result = utterance.readable_text or ""
    if not result:
        return []
    allowed = _word_set(_source_union(utterance))
    produced = [word for token in result.split() if (word := _normalize_word(token))]
    kinds: list[str] = []
    for word in produced:
        if _is_digit_token(word):
            # Reported separately from "substitution" so a reviewer can tell "130" (rule
            # 4A's intended digitization) apart from a genuinely substituted word;
            # a digit already present verbatim in the source is not a change at all.
            if word not in allowed:
                kinds.append("number")
            continue
        if _is_allowed_word(word, allowed):
            continue
        if _repaired_from(word, allowed):
            kinds.append("repair")
        else:
            kinds.append("transliteration" if _LATIN.search(word) else "substitution")
    # Only words every model heard count as lost; one model mishearing is not a loss.
    heard = _word_set(utterance.raw_text)
    for variant_text in utterance.variants.values():
        heard &= _word_set(variant_text)
    folded = result.lower().replace("ё", "е")
    lost = [
        word
        for word in sorted(heard)
        if len(word) >= 4 and word not in produced and word not in folded
    ]
    # Every replacement is physically a loss plus a gain: 'виспер' -> 'Whisper' would
    # otherwise be reported twice, once as a phantom loss. Only the net remainder is
    # a word that genuinely went missing, and netting needs no transliteration table.
    return kinds + ["loss"] * max(0, len(lost) - len(kinds))


def validate_readable(
    utterances: list[Utterance], items: list[dict[str, Any]], protected_terms: set[str]
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    """Validate one batch's fused texts, degrading a bad utterance instead of the batch.

    Losing an id, gaining one, or seeing them reordered still raises: that is the one
    failure a retry of the same batch can plausibly fix. Everything else — an empty
    text, a protected value gone missing, too many invented words — used to raise the
    same way, which meant one number heard by a single model out of up to three could
    drop every other utterance in the batch to its local text. Each of those is now a
    per-utterance fallback instead.

    Protected values (digits, URLs, emails, glossary terms) are checked against the
    canonical utterance (`raw_text`, payload field "a") only, not the union of every
    variant: a number only Whisper heard is real content the model may legitimately
    prefer over `a`, per rule 3, and requiring it verbatim in the output punished the
    model for doing exactly what the prompt asks. Word membership stays checked
    against the union of all variants — a word the model kept from b or c, instead of
    a, is an edit, not a fabrication, and rejecting it would contradict rule 3 too.
    """
    expected = [u.utterance_id for u in utterances]
    ids = [item.get("utterance_id") for item in items]
    if ids != expected or len(ids) != len(set(ids)):
        raise ValueError(
            "The LLM returned missing, extra, duplicated, or reordered utterance_id values."
        )
    output: dict[str, str] = {}
    rejected: dict[str, list[str]] = {}
    repairs: dict[str, list[str]] = {}
    for utterance, item in zip(utterances, items, strict=True):
        text = str(item.get("readable_text", "")).strip()
        if not text:
            rejected[utterance.utterance_id] = ["the model returned empty text"]
            output[utterance.utterance_id] = utterance.light_text or utterance.raw_text
            continue
        protected_values = _protected(utterance.raw_text) + [
            term for term in protected_terms if term in utterance.raw_text
        ]
        missing = [value for value in protected_values if value not in text]
        if missing:
            rejected[utterance.utterance_id] = [
                f"lost protected value {value!r}" for value in missing
            ]
            output[utterance.utterance_id] = utterance.light_text or utterance.raw_text
            continue
        # Word membership degrades per utterance instead of raising: one over-eager
        # word must not discard a whole meeting's fused text. The offending utterance
        # falls back to its mechanically cleaned local text, which is never invented.
        # The owner chose readability over a per-word veto: transliteration and
        # confident repairs are allowed, governed by the prompt. What stays is a
        # proportionality limit, which separates editing from rewriting, and a full
        # log, so every liberty the model takes remains reviewable afterwards.
        allowed_words = _word_set(_source_union(utterance))
        tokens = [word for token in text.split() if (word := _normalize_word(token))]
        changed: list[str] = []
        for word in tokens:
            if _is_allowed_word(word, allowed_words):
                continue
            source_word = _repaired_from(word, allowed_words)
            changed.append(f"{source_word} → {word}" if source_word else f"+{word}")
        if len(changed) > max(2, round(len(tokens) * _MAX_CHANGED_SHARE)):
            # Not an edit any more: too much of the utterance is not in either source.
            rejected[utterance.utterance_id] = [f"words outside the variants: {', '.join(changed)}"]
            output[utterance.utterance_id] = utterance.light_text or utterance.raw_text
            continue
        if changed:
            repairs[utterance.utterance_id] = changed
        output[utterance.utterance_id] = text
    return output, rejected, repairs


#: Utterances per LLM request. A whole two-hour meeting in one call risks hitting the
#: output limit and truncating, which fails the id check and wastes the entire pass.
READABLE_BATCH_SIZE = 150


def _fuse_batch(
    client: Any,
    model: str,
    instructions: str,
    batch: list[Utterance],
    protected_terms: set[str],
    variant_models: Sequence[str] = (),
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, Any]]:
    """Fuse one batch, retrying once. Raises ValueError if it still does not validate."""
    import json

    from .schemas import ReadablePayload

    retry_note = ""
    last_error: Exception | None = None
    reasoning = readable_reasoning()
    for attempt in range(2):
        request_kwargs: dict[str, Any] = {
            "model": model,
            "store": False,
            "input": [
                {"role": "system", "content": instructions + retry_note},
                {
                    "role": "user",
                    "content": json.dumps(build_payload(batch, variant_models), ensure_ascii=False),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "readable_transcript",
                    "strict": True,
                    "schema": ReadablePayload.model_json_schema(),
                }
            },
        }
        if reasoning is not None:
            request_kwargs["reasoning"] = reasoning
        response = client.responses.create(**request_kwargs)
        try:
            parsed = ReadablePayload.model_validate_json(response.output_text)
            texts, rejected, repairs = validate_readable(
                batch, [item.model_dump() for item in parsed.items], protected_terms
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt:
                break
            retry_note = (
                f" The previous response failed local validation: {exc}. "
                "Fix only the format and keep all the constraints."
            )
            continue
        concerns = [
            {"utterance_id": item.utterance_id, "note": concern}
            for item in parsed.items
            for concern in item.concerns
        ]
        concerns.extend(
            {"utterance_id": utterance_id, "note": f"{CONCERN_REPAIR}: " + ", ".join(fixes)}
            for utterance_id, fixes in repairs.items()
        )
        concerns.extend(
            {"utterance_id": uid, "note": f"{CONCERN_FALLBACK}: {'; '.join(reasons)}"}
            for uid, reasons in rejected.items()
        )
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return texts, concerns, usage
    raise ValueError(str(last_error))


def clean_readable(
    utterances: list[Utterance],
    model: str,
    protected_terms: set[str],
    context_prompt: str = "",
    status: Callable[[str], None] | None = None,
    variant_models: Sequence[str] = (),
) -> ReadableResult:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "readable expects OPENAI_API_KEY in the environment; raw and light are already "
            "ready locally."
        )
    from openai import OpenAI  # lazy; no audio is provided to this client

    client = OpenAI(**openai_client_options())
    instructions = READABLE_FUSION_INSTRUCTIONS
    if context_prompt.strip():
        # Rule 3 now invites the model to use this context, so the boundary has to be
        # explicit right here: it may break a tie, never supply a word of its own.
        instructions += (
            "\nКОНТЕКСТ ВСТРЕЧИ. Помогает выбрать между вариантами, но не даёт права взять "
            "слово, которого нет ни в одном из них, и не является источником фактов для "
            f"транскрипта: {context_prompt.strip()}"
        )
    batches = [
        utterances[start : start + READABLE_BATCH_SIZE]
        for start in range(0, len(utterances), READABLE_BATCH_SIZE)
    ]
    texts: dict[str, str] = {}
    concerns: list[dict[str, str]] = []
    usage: dict[str, Any] = {}
    failed = 0
    for number, batch in enumerate(batches, 1):
        if status:
            status(f"readable: batch {number}/{len(batches)} ({len(batch)} utterances)")
        try:
            batch_texts, batch_concerns, batch_usage = _fuse_batch(
                client, model, instructions, batch, protected_terms, variant_models
            )
        except ValueError as exc:
            # One bad batch must not discard the rest: its utterances keep the local
            # text, exactly as a rejected word does inside a batch.
            failed += 1
            texts.update({item.utterance_id: item.light_text or item.raw_text for item in batch})
            concerns.append(
                {
                    "utterance_id": batch[0].utterance_id,
                    "note": f"batch {number} failed validation, kept the local text: {exc}",
                }
            )
            if status:
                status(f"readable: batch {number} failed validation, kept the local text")
            continue
        texts.update(batch_texts)
        concerns.extend(batch_concerns)
        for key, value in batch_usage.items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    if failed == len(batches):
        # Nothing was gained anywhere: report it as a failure, not a silent no-op.
        raise RuntimeError("readable failed validation in every batch; raw/light are unchanged.")
    return ReadableResult(texts, concerns, usage)
