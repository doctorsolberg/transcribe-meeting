from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from ipaddress import ip_address
from typing import Any, TypedDict
from urllib.parse import urlsplit

from .model import Utterance
from .readable_transport import EndpointError, UnusableReply, request_readable
from .schemas import ReadablePayload
from .secrets import OPENAI_ENV_CONFLICT

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
    # None distinguishes older callers constructing a result without metrics.
    accepted_utterances: int | None = None
    fallback_utterances: int | None = None
    api: str = "responses"
    response_format: str = "json_schema"
    provider: str = "openai"
    repaired_utterances: int | None = None
    endpoint_failure: str | None = None
    pending_utterance_ids: list[str] | None = None


class _OpenAIClientOptions(TypedDict):
    api_key: str
    base_url: str


DEFAULT_READABLE_BASE_URL = "https://api.openai.com/v1"
_AUTHORITY = re.compile(r"(?:\[[0-9a-fA-F:.]+\]|[^:\[\]]+)(?::[0-9]+)?\Z")
_DNS_LABEL = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\Z")


class ReadableConfigError(RuntimeError, ValueError):
    """A local configuration error with no input values in its diagnostic."""


@dataclass(frozen=True)
class ReadableReadiness:
    ready: bool
    host: str | None
    diagnostic: str | None = None


def _endpoint(value: str) -> tuple[str, str, bool]:
    """Return the URL, display authority, and whether the host is a literal loopback."""
    invalid = "OPENAI_BASE_URL must be an HTTP(S) API URL without credentials, query, or fragment."
    url = value.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url) or any(
        ch in url for ch in "\\?#"
    ):
        raise ReadableConfigError(invalid)
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme not in ("http", "https")
            or not host
            or "@" in parsed.netloc
            or not _AUTHORITY.fullmatch(parsed.netloc)
            or port == 0
        ):
            raise ValueError
        try:
            address = ip_address(host)
        except ValueError:
            address = None
            ascii_host = host.encode("idna").decode("ascii")
            domain = ascii_host.removesuffix(".")
            if len(domain) > 253 or not all(_DNS_LABEL.fullmatch(p) for p in domain.split(".")):
                raise ValueError from None
    except ValueError:
        raise ReadableConfigError(invalid) from None
    loopback = bool(address and address.is_loopback and not getattr(address, "ipv4_mapped", None))
    if parsed.scheme == "http" and not loopback:
        raise ReadableConfigError("OPENAI_BASE_URL requires HTTPS except for literal loopback IPs.")
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return url, authority, loopback


def validate_readable_base_url(value: str) -> str:
    """Validate an explicit endpoint for client configuration or the auth setter."""
    return _endpoint(value)[0]


def _configured_endpoint() -> tuple[str, str, bool]:
    value = os.environ.get("OPENAI_BASE_URL", "")
    if not value.strip() and os.environ.get(OPENAI_ENV_CONFLICT) == "endpoint":
        raise ReadableConfigError(
            "OPENAI_BASE_URL from .env cannot be combined with an OPENAI_API_KEY from the "
            "environment for another origin. Set both in the environment or unset the shell key "
            "and configure them in the same .env."
        )
    return _endpoint(value if value.strip() else DEFAULT_READABLE_BASE_URL)


def _configured_key(loopback: bool) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and not loopback:
        if os.environ.get(OPENAI_ENV_CONFLICT) == "key":
            raise ReadableConfigError(
                "OPENAI_API_KEY from .env is configured for another origin than OPENAI_BASE_URL "
                "in the environment. Set both in the environment or configure both in .env."
            )
        raise ReadableConfigError(
            "OPENAI_API_KEY is required for a non-loopback readable endpoint."
        )
    if any(not 33 <= ord(character) <= 126 for character in key):
        raise ReadableConfigError("OPENAI_API_KEY must contain only visible ASCII characters.")
    return key


def openai_client_options() -> _OpenAIClientOptions:
    """Pass an explicit endpoint so the SDK cannot re-read a blank environment value."""
    url, _, loopback = _configured_endpoint()
    return _OpenAIClientOptions(api_key=_configured_key(loopback), base_url=url)


def readable_readiness() -> ReadableReadiness:
    """Inspect effective settings without creating an API client or exposing secrets."""
    host = None
    try:
        _, host, loopback = _configured_endpoint()
        _configured_key(loopback)
        _ReadableSettings.from_env()
    except ReadableConfigError as exc:
        return ReadableReadiness(False, host, str(exc))
    return ReadableReadiness(True, host)


#: Reasoning effort for the fusion request. "none" omits the field entirely, which is
#: what an OpenAI-compatible endpoint without reasoning support needs (auth
#: set-openai-base-url is a documented feature).
READABLE_REASONING_ENV = "TRANSCRIBE_READABLE_REASONING"
READABLE_REASONING_LEVELS = ("none", "low", "medium", "high")


def readable_reasoning(api: str = "responses") -> dict[str, str] | None:
    default = "medium" if api == "responses" else "none"
    level = _env_choice(READABLE_REASONING_ENV, default, READABLE_REASONING_LEVELS)
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


_LINK_OR_EMAIL = r"https?://[^\s<>]+|[\w.+-]+@[\w.-]+"
_SEMANTIC_WORD = re.compile(
    rf"{_LINK_OR_EMAIL}|(?:[+−-]\s*|[$€£₽]\s*)*(?=\d)\d+"
    r"(?:(?<=\d)[.,:/–−-]\d+|[-’']\w+|\w+)*(?:\s*[%‰$€£₽])?|\w+(?:[-’']\w+)*",
    re.IGNORECASE,
)
_PLAIN_NUMBER = re.compile(r"(?P<value>[0-9]+)(?:-(?P<ending>[а-я]+)|(?P<en>st|nd|rd|th))?\Z")
_NEGATIONS = frozenset(
    [
        "не",
        "ни",
        "нет",
        "неть",
        "нельзя",
        "без",
        "никак",
        "нигде",
        "ничто",
        "ничего",
        "никто",
        "никого",
        "никогда",
        "никакой",
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "without",
        "cannot",
    ]
)

# A conservative grammar, not a general text-to-number converter. Unknown/ambiguous
# phrases are kept verbatim. The tuple marks English/Russian, place and ordinal ending.
_RU_NUMBER_FORMS = {
    0: "ноль нуль ноля нуля",
    1: "один одна одно одни одного одной одним одну",
    2: "два две двух двум двумя",
    3: "три трех трем тремя",
    4: "четыре четырех четырем четырьмя",
    5: "пять пяти пятью",
    6: "шесть шести шестью",
    7: "семь семи семью",
    8: "восемь восьми восемью восьмью",
    9: "девять девяти девятью",
    10: "десять десяти десятью",
    11: "одиннадцать одиннадцати одиннадцатью",
    12: "двенадцать двенадцати двенадцатью",
    13: "тринадцать тринадцати тринадцатью",
    14: "четырнадцать четырнадцати четырнадцатью",
    15: "пятнадцать пятнадцати пятнадцатью",
    16: "шестнадцать шестнадцати шестнадцатью",
    17: "семнадцать семнадцати семнадцатью",
    18: "восемнадцать восемнадцати восемнадцатью",
    19: "девятнадцать девятнадцати девятнадцатью",
    20: "двадцать двадцати двадцатью",
    30: "тридцать тридцати тридцатью",
    40: "сорок сорока",
    50: "пятьдесят пятидесяти пятьюдесятью",
    60: "шестьдесят шестидесяти шестьюдесятью",
    70: "семьдесят семидесяти семьюдесятью",
    80: "восемьдесят восьмидесяти восемьюдесятью",
    90: "девяносто девяноста",
    100: "сто ста",
    200: "двести двухсот",
    300: "триста трехсот",
    400: "четыреста четырехсот",
    500: "пятьсот пятисот",
    600: "шестьсот шестисот",
    700: "семьсот семисот",
    800: "восемьсот восьмисот",
    900: "девятьсот девятисот",
    1000: "тысяча тысячи тысяч тысячу тысячей",
}
_ORDINAL_STEMS = {
    "перв": 1,
    "втор": 2,
    "четверт": 4,
    "пят": 5,
    "шест": 6,
    "седьм": 7,
    "восьм": 8,
    "девят": 9,
    "десят": 10,
    "одиннадцат": 11,
    "двенадцат": 12,
    "тринадцат": 13,
    "четырнадцат": 14,
    "пятнадцат": 15,
    "шестнадцат": 16,
    "семнадцат": 17,
    "восемнадцат": 18,
    "девятнадцат": 19,
    "двадцат": 20,
    "тридцат": 30,
    "сороков": 40,
    "пятидесят": 50,
    "шестидесят": 60,
    "семидесят": 70,
    "восьмидесят": 80,
    "девяност": 90,
    "сот": 100,
}
_ORDINAL_ENDINGS = {
    "ый": "й",
    "ой": "й",
    "ая": "я",
    "ое": "е",
    "ые": "е",
    "ого": "го",
    "ому": "му",
    "ым": "м",
    "ом": "м",
    "ую": "ю",
    "ых": "х",
    "ыми": "ми",
}
_RU_CARDINALS = {word: value for value, forms in _RU_NUMBER_FORMS.items() for word in forms.split()}
_RU_ORDINALS = {
    stem + ending: (value, suffix)
    for stem, value in _ORDINAL_STEMS.items()
    for ending, suffix in _ORDINAL_ENDINGS.items()
} | {"третий": (3, "й"), "третьего": (3, "го"), "третья": (3, "я")}
_EN_CARDINALS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
        ],
        [*range(20), *range(20, 100, 10), 100, 1000],
        strict=True,
    )
)


def _protected_links(text: str) -> list[str]:
    return [link.rstrip(".,!?;:»”)") for link in re.findall(_LINK_OR_EMAIL, text, re.IGNORECASE)]


_AMBIGUOUS_NUMBER_WORDS = frozenset(
    [
        "миллион",
        "миллиона",
        "миллионов",
        "миллиону",
        "миллионом",
        "миллионе",
        "миллионы",
        "миллиард",
        "миллиарда",
        "миллиардов",
        "миллиарду",
        "миллиардом",
        "миллиарде",
        "миллиарды",
        "триллион",
        "триллиона",
        "триллионов",
        "триллионы",
        "полтора",
        "полторы",
        "полтораста",
        "половина",
        "million",
        "millions",
        "billion",
        "billions",
        "trillion",
        "trillions",
        "half",
        "quarter",
    ]
)


def _number_word(word: str) -> tuple[str, int, str] | None:
    if word in _RU_CARDINALS:
        return "ru", _RU_CARDINALS[word], ""
    if word in _RU_ORDINALS:
        value, ending = _RU_ORDINALS[word]
        return "ru", value, ending
    if word in _EN_CARDINALS:
        return "en", _EN_CARDINALS[word], ""
    if word in _AMBIGUOUS_NUMBER_WORDS:
        return "", -1, ""
    return None


def _spelled_integer(parts: list[tuple[str, int, str]]) -> tuple[int, str] | None:
    lang = parts[0][0]
    if not lang or any(part[0] != lang for part in parts) or any(part[2] for part in parts[:-1]):
        return None
    values = [part[1] for part in parts]
    total = 0
    if 1000 in values:
        index = values.index(1000)
        if index > 1 or (index and not 1 <= values[0] <= 9):
            return None
        total = (values[0] if index else 1) * 1000
        values = values[index + 1 :]
    if lang == "en" and len(values) >= 2 and 1 <= values[0] <= 9 and values[1] == 100:
        total += values[0] * 100
        values = values[2:]
    elif values and 100 <= values[0] <= 900:
        if lang == "en":
            return None
        total += values.pop(0)
    if len(values) == 2 and 20 <= values[0] <= 90 and 1 <= values[1] <= 9:
        total += sum(values)
    elif len(values) == 1 and values[0] < 100 and (values[0] or not total):
        total += values[0]
    elif values:
        return None
    return total, parts[-1][2]


@dataclass(frozen=True)
class _NumberSpan:
    tokens: tuple[str, ...]
    value: tuple[int, str] | None = None
    # Only spelled values can be reformatted; numeric identifiers, signs, suffixes,
    # currency and range punctuation must be preserved literally.
    spelled: bool = False


def _semantic_tokens(text: str) -> list[str | _NumberSpan]:
    tokens: list[str | _NumberSpan] = []
    matches = list(_SEMANTIC_WORD.finditer(text.lower().replace("ё", "е").replace("’", "'")))
    index = 0
    while index < len(matches):
        match = matches[index]
        word = match[0]
        if re.match(_LINK_OR_EMAIL, word, flags=re.IGNORECASE):
            tokens.append(word)
        elif any(character.isnumeric() for character in word):
            number = _PLAIN_NUMBER.fullmatch(word)
            value = None
            if number and len(number["value"]) <= 4:
                value = int(number["value"]), number["ending"] or number["en"] or ""
            tokens.append(_NumberSpan((word,), value))
        elif all(_number_word(part) for part in word.split("-")):
            words = [word]
            while index + 1 < len(matches):
                next_ = matches[index + 1]
                gap = text[matches[index].end() : next_.start()]
                if not gap.isspace() or not all(_number_word(p) for p in next_[0].split("-")):
                    break
                index += 1
                words.append(next_[0])
            parts = [part for w in words for p in w.split("-") if (part := _number_word(p))]
            value = _spelled_integer(parts)
            # Preserve the prompt's explicit exceptions and ambiguous singular uses.
            if index + 1 < len(matches) and matches[index + 1][0] in {"словах", "мой", "моя"}:
                value = None
            if len(words) == 1 and words[0] in {"один", "одна", "одно", "one"}:
                value = None
            if (
                value
                and value[1] == "й"
                and index + 1 < len(matches)
                and matches[index + 1][0] == "год"
            ):
                value = value[0], ""
            tokens.append(_NumberSpan(tuple(words), value, spelled=True))
        else:
            tokens.append(word)
        index += 1
    return tokens


def _matching_number(source: _NumberSpan, output: _NumberSpan) -> bool:
    return source.tokens == output.tokens or (
        source.spelled
        and not output.spelled
        and source.value is not None
        and source.value == output.value
    )


def _numeric_context(tokens: list[str | _NumberSpan]) -> list[tuple[Any, ...]]:
    # Keep values attached to the same neighboring words (e.g. quantities and units).
    return [
        tuple(part if isinstance(part, str) else None for part in tokens[max(0, i - 1) : i + 2])
        for i, token in enumerate(tokens)
        if isinstance(token, _NumberSpan)
    ]


def _negation_context(tokens: list[str | _NumberSpan]) -> list[tuple[Any, ...]]:
    def comparable(token: str | _NumberSpan) -> Any:
        if isinstance(token, _NumberSpan):
            return token.value if token.value is not None else token.tokens
        return token

    return [
        (
            *(comparable(t) for t in tokens[max(0, i - 2) : i]),
            None,
            *(comparable(t) for t in tokens[i : i + 3]),
        )
        for i, token in enumerate(tokens)
        if isinstance(token, str)
        and any(part in _NEGATIONS or part.endswith("n't") for part in token.split("-"))
    ]


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
CONCERN_REMOVAL = "removed source words"


def _audit_tokens(text: str) -> list[str]:
    return [word for token in text.split() if (word := _normalize_word(token))]


def _removed_words(utterance: Utterance, text: str) -> list[str]:
    """Audit short and repeated words; a substring does not account for a token."""
    source = _audit_tokens(utterance.raw_text)
    produced = _audit_tokens(text)
    allowed = _word_set(_source_union(utterance))
    heard = Counter(source)
    for variant in utterance.variants.values():
        if variant.strip():
            heard &= Counter(_audit_tokens(variant))
    # Alignment distinguishes a replacement from otherwise hidden deletions.
    alignment = SequenceMatcher(None, source, produced, autojunk=False)
    removed: list[str] = []
    for kind, a, b, c, d in alignment.get_opcodes():
        if kind not in ("replace", "delete"):
            continue
        prior, after = source[a:b], produced[c:d]
        if after and "".join(prior).replace("-", "") == "".join(after).replace("-", ""):
            continue  # a separator-only join or split
        if after and _is_digit_token(after[0]):
            before_number = _semantic_tokens(" ".join(prior))
            after_number = _semantic_tokens(after[0])
            if (
                len(before_number) == len(after_number) == 1
                and isinstance(x := before_number[0], _NumberSpan)
                and isinstance(y := after_number[0], _NumberSpan)
                and _matching_number(x, y)
            ):
                continue
        if len(prior) == len(after) == 1 and not _is_allowed_word(after[0], allowed):
            # An actual novel replacement is classified separately; an extra
            # instance of an existing word must not hide an unrelated deletion.
            continue
        removed.extend(prior)
    # Do not claim a loss if only one model heard it, or if the agreed count
    # of the word is still present elsewhere in the output.
    missing = heard - Counter(produced)
    lost = []
    for word in removed:
        if missing[word] > 0:
            lost.append(word)
            missing[word] -= 1
    return lost


def classify_changes(utterance: Utterance) -> list[str]:
    """Classify every word the LLM changed in one utterance, without asking a model.

    Use local edit distance, token alignment and counts; never ask the same model
    to verify its own changes. Cyrillic substitutions stay flagged for review.
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
    return kinds + ["loss"] * len(_removed_words(utterance, result))


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

    Numbers and negation must agree with the canonical utterance. A number can
    change form only when the local grammar proves a lossless conversion. Word
    membership for other edits continues to use the union of all variants.
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
        source = _semantic_tokens(utterance.raw_text)
        edited = _semantic_tokens(text)
        numbers = [t for t in source if isinstance(t, _NumberSpan)]
        edited_numbers = [t for t in edited if isinstance(t, _NumberSpan)]
        reason = None
        if (
            len(numbers) != len(edited_numbers)
            or _numeric_context(source) != _numeric_context(edited)
            or not all(_matching_number(a, b) for a, b in zip(numbers, edited_numbers, strict=True))
        ):
            reason = "changed numeric content"
        elif _negation_context(source) != _negation_context(edited):
            reason = "changed negation or its immediate context"
        if not reason and _protected_links(utterance.raw_text) != _protected_links(text):
            reason = "changed a protected link or email address"
        if not reason and any(
            term in utterance.raw_text and term not in text for term in protected_terms
        ):
            reason = "lost a protected glossary term"
        if not reason and source:
            if not any(ch.isalnum() for ch in text):
                reason = "the model returned no spoken text"
            elif len(edited) * 2 < len(source) and len(source) - len(edited) >= 3:
                # Compare semantic tokens so normal number formatting is not a loss.
                reason = "the model dropped most of the source words"
        if reason:
            rejected[utterance.utterance_id] = [reason]
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
            # Digit tokens have already passed the ordered, exact semantic check.
            if _is_digit_token(word) or _is_allowed_word(word, allowed_words):
                continue
            source_word = _repaired_from(word, allowed_words)
            changed.append(f"{source_word} → {word}" if source_word else f"+{word}")
        if len(changed) > max(2, round(len(tokens) * _MAX_CHANGED_SHARE)):
            # Not an edit any more: too much of the utterance is not in either source.
            rejected[utterance.utterance_id] = ["too many words outside the ASR variants"]
            output[utterance.utterance_id] = utterance.light_text or utterance.raw_text
            continue
        if changed:
            repairs[utterance.utterance_id] = changed
        output[utterance.utterance_id] = text
    return output, rejected, repairs


#: Utterances per LLM request. A whole two-hour meeting in one call risks hitting the
#: output limit and truncating, which fails the id check and wastes the entire pass.
READABLE_BATCH_SIZE = 150
READABLE_API_ENV = "TRANSCRIBE_READABLE_API"
READABLE_RESPONSE_FORMAT_ENV = "TRANSCRIBE_READABLE_RESPONSE_FORMAT"
READABLE_BATCH_SIZE_ENV = "TRANSCRIBE_READABLE_BATCH_SIZE"
READABLE_REQUEST_TIMEOUT = 120.0


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.environ.get(name, "").strip().lower() or default
    if value not in allowed:
        raise ReadableConfigError(f"{name} must be one of: {', '.join(allowed)}")
    return value


@dataclass(frozen=True)
class _ReadableSettings:
    api: str
    response_format: str
    reasoning: dict[str, str] | None
    batch_size: int

    @classmethod
    def from_env(cls) -> _ReadableSettings:
        api = _env_choice(READABLE_API_ENV, "responses", ("responses", "chat"))
        response_format = _env_choice(
            READABLE_RESPONSE_FORMAT_ENV, "json_schema", ("json_schema", "json_object")
        )
        raw_size = os.environ.get(READABLE_BATCH_SIZE_ENV, "").strip()
        try:
            batch_size = int(raw_size) if raw_size else READABLE_BATCH_SIZE
        except ValueError:
            batch_size = 0
        if batch_size < 1:
            raise ReadableConfigError(f"{READABLE_BATCH_SIZE_ENV} must be a positive integer")
        return cls(api, response_format, readable_reasoning(api), batch_size)


@dataclass
class _BatchResult:
    texts: dict[str, str]
    concerns: list[dict[str, str]]
    fallback_utterances: int
    repaired_utterances: int = 0


def _add_usage(usage: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        usage[key] = usage.get(key, 0) + value


def _fuse_batch(
    client: Any,
    model: str,
    instructions: str,
    batch: list[Utterance],
    protected_terms: set[str],
    variant_models: Sequence[str] = (),
    *,
    usage: dict[str, int],
    settings: _ReadableSettings | None = None,
) -> _BatchResult:
    """Fuse one batch, retrying JSON schema or utterance ID validation once."""
    retry_note = ""
    settings = settings or _ReadableSettings.from_env()
    payload = json.dumps(build_payload(batch, variant_models), ensure_ascii=False)
    for attempt in range(2):
        try:
            output, counts = request_readable(
                client,
                settings.api,
                settings.response_format,
                model,
                instructions + retry_note,
                payload,
                settings.reasoning,
            )
        except UnusableReply as exc:
            _add_usage(usage, exc.usage or {})
            raise
        _add_usage(usage, counts)
        try:
            parsed = ReadablePayload.model_validate_json(output)
            texts, rejected, repairs = validate_readable(
                batch, [item.model_dump() for item in parsed.items], protected_terms
            )
        except ValueError:
            if attempt:
                break
            retry_note = (
                "\nThe previous response failed local validation. Return valid JSON matching "
                "the schema, with each original utterance ID exactly once and in input order."
            )
            continue
        concerns = [
            {"utterance_id": item.utterance_id, "note": "the model flagged uncertainty"}
            for item in parsed.items
            if any(concern.strip() for concern in item.concerns)
        ]
        repaired_count = 0
        for uid, changes in repairs.items():
            additions = [change for change in changes if change.startswith("+")]
            corrections = [change for change in changes if not change.startswith("+")]
            if corrections:
                repaired_count += 1
                concerns.append(
                    {"utterance_id": uid, "note": f"{CONCERN_REPAIR}: " + ", ".join(corrections)}
                )
            if additions:
                concerns.append(
                    {
                        "utterance_id": uid,
                        "note": "words outside the variants: " + ", ".join(additions),
                    }
                )
        concerns.extend(
            {"utterance_id": uid, "note": f"{CONCERN_FALLBACK}: {'; '.join(reasons)}"}
            for uid, reasons in rejected.items()
        )
        for utterance in batch:
            uid = utterance.utterance_id
            if uid not in rejected and (removed := _removed_words(utterance, texts[uid])):
                concerns.append(
                    {"utterance_id": uid, "note": f"{CONCERN_REMOVAL}: {', '.join(removed)}"}
                )
        return _BatchResult(texts, concerns, len(rejected), repaired_count)
    raise UnusableReply("the response failed local JSON schema or utterance ID validation")


def clean_readable(
    utterances: list[Utterance],
    model: str,
    protected_terms: set[str],
    context_prompt: str = "",
    status: Callable[[str], None] | None = None,
    variant_models: Sequence[str] = (),
) -> ReadableResult:
    if not utterances:
        raise ReadableConfigError("readable has no utterances to clean; no request was sent.")
    options = openai_client_options()
    settings = _ReadableSettings.from_env()

    instructions = READABLE_FUSION_INSTRUCTIONS
    if context_prompt.strip():
        # Rule 3 now invites the model to use this context, so the boundary has to be
        # explicit right here: it may break a tie, never supply a word of its own.
        instructions += (
            "\nКОНТЕКСТ ВСТРЕЧИ. Помогает выбрать между вариантами, но не даёт права взять "
            "слово, которого нет ни в одном из них, и не является источником фактов для "
            f"транскрипта: {context_prompt.strip()}"
        )
    from httpx import InvalidURL
    from openai import DefaultHttpxClient, OpenAI

    custom_endpoint = options["base_url"].rstrip("/") != DEFAULT_READABLE_BASE_URL

    def remove_provider_headers(request: Any) -> None:
        request.headers.pop("OpenAI-Organization", None)
        request.headers.pop("OpenAI-Project", None)

    hooks = {"request": [remove_provider_headers]} if custom_endpoint else None
    with DefaultHttpxClient(
        follow_redirects=False, trust_env=False, event_hooks=hooks
    ) as http_client:
        try:
            # No automatic transport retries: the server may have processed/charged
            # a request even when the client gets a timeout or sees a gateway error.
            client = OpenAI(
                **options,
                http_client=http_client,
                timeout=READABLE_REQUEST_TIMEOUT,
                max_retries=0,
            )
        except InvalidURL:
            raise ReadableConfigError("OPENAI_BASE_URL is not a valid API URL.") from None
        result = _clean_batches(
            client,
            utterances,
            model,
            protected_terms,
            instructions,
            status,
            variant_models,
            settings,
        )
    if custom_endpoint:
        result.provider = "openai-compatible"
    return result


def _clean_batches(
    client: Any,
    utterances: list[Utterance],
    model: str,
    protected_terms: set[str],
    instructions: str,
    status: Callable[[str], None] | None,
    variant_models: Sequence[str],
    settings: _ReadableSettings,
) -> ReadableResult:
    batch_size = settings.batch_size
    batches = [
        utterances[start : start + batch_size] for start in range(0, len(utterances), batch_size)
    ]
    texts: dict[str, str] = {}
    concerns: list[dict[str, str]] = []
    usage: dict[str, int] = {}
    accepted = 0
    repaired = 0
    endpoint_failure = None
    pending_ids: list[str] = []
    validated_batches = 0
    last_failure = ""
    for number, batch in enumerate(batches, 1):
        if status:
            status(f"readable: batch {number}/{len(batches)} ({len(batch)} utterances)")
        try:
            result = _fuse_batch(
                client,
                model,
                instructions,
                batch,
                protected_terms,
                variant_models,
                usage=usage,
                settings=settings,
            )
        except EndpointError as exc:
            if not accepted:
                raise
            # An endpoint failure is not a malformed batch. Keep prior edits, but do
            # not resend more transcript text to an unreachable/refusing endpoint.
            remaining = utterances[(number - 1) * batch_size :]
            endpoint_failure = str(exc)
            pending_ids.extend(item.utterance_id for item in remaining)
            texts.update(
                {item.utterance_id: item.light_text or item.raw_text for item in remaining}
            )
            concerns.append(
                {
                    "utterance_id": batch[0].utterance_id,
                    "note": f"{exc}; kept the remaining local text",
                }
            )
            if status:
                status(f"{exc}; kept the remaining local text")
            break
        except UnusableReply as exc:
            # One bad batch must not discard the rest: its utterances keep the local
            # text, exactly as a rejected word does inside a batch.
            last_failure = str(exc)
            pending_ids.extend(item.utterance_id for item in batch)
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
        texts.update(result.texts)
        concerns.extend(result.concerns)
        validated_batches += 1
        accepted += len(batch) - result.fallback_utterances
        repaired += result.repaired_utterances
    if not accepted:
        if not validated_batches:
            raise UnusableReply(
                f"readable failed validation in every batch ({last_failure}); raw/light are unchanged."
            )
        raise UnusableReply(
            "readable: no utterances passed local content validation; raw/light are unchanged."
        )
    return ReadableResult(
        texts,
        concerns,
        usage,
        accepted_utterances=accepted,
        fallback_utterances=len(utterances) - accepted,
        api=settings.api,
        response_format=settings.response_format,
        repaired_utterances=repaired,
        endpoint_failure=endpoint_failure,
        pending_utterance_ids=pending_ids or None,
    )
