import json
import sys
import types
import wave
from pathlib import Path

import pytest

from transcribe_meeting import asr
from transcribe_meeting.util import read_json


def _wav(path, *, seconds: int = 1, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * rate * seconds)


def test_groq_transcription_uses_openai_compatible_endpoint(tmp_path, monkeypatch):
    audio = tmp_path / "part.wav"
    output = tmp_path / "part.json"
    _wav(audio)
    seen = {}

    class Response:
        def model_dump(self):
            return {
                "text": "тест",
                "segments": [],
                "words": [{"word": "тест", "start": 0, "end": 1}],
            }

    class Transcriptions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return Response()

    class OpenAI:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.audio = types.SimpleNamespace(transcriptions=Transcriptions())

    from openai import DefaultHttpxClient

    fake_openai = types.SimpleNamespace(
        OpenAI=OpenAI, APIStatusError=RuntimeError, DefaultHttpxClient=DefaultHttpxClient
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    asr._transcribe_groq(audio, output, "ru", "Тема: тест", limiter=None)

    assert seen["client"]["base_url"] == "https://api.groq.com/openai/v1"
    assert seen["model"] == "whisper-large-v3"
    assert seen["timestamp_granularities"] == ["segment", "word"]
    assert read_json(output)["run_parameters"]["provider"] == "groq"


def test_groq_does_not_forward_upload_or_openai_headers(tmp_path, monkeypatch):
    import httpx

    audio = tmp_path / "part.wav"
    _wav(audio)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "test-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "test-project")
    requests = []

    def redirect(_client, request):
        requests.append(request)
        if request.url.host == "api.groq.com":
            return httpx.Response(
                307, headers={"Location": "https://other.test/upload"}, request=request
            )
        return httpx.Response(401, request=request)

    monkeypatch.setattr(httpx.Client, "_send_single_request", redirect)

    with pytest.raises(RuntimeError, match="HTTP"):
        asr._request_groq(audio, "ru", "private context", None)

    assert [request.url.host for request in requests] == ["api.groq.com"]
    assert "OpenAI-Organization" not in requests[0].headers
    assert "OpenAI-Project" not in requests[0].headers


def test_groq_prompt_keeps_beginning_and_latest_context():
    prompt = "A" * 500 + "B" * 500
    clipped = asr._groq_prompt(prompt)

    assert clipped is not None
    assert len(clipped) <= asr.GROQ_PROMPT_MAX_CHARACTERS
    assert clipped.startswith("A")
    assert clipped.endswith("B")


def test_groq_provider_caps_chunk_duration_to_ten_minutes(tmp_path, monkeypatch):
    audio = tmp_path / "part.wav"
    _wav(audio, seconds=601, rate=1)
    output = tmp_path / "asr" / "001.json"
    requests = []

    def fake_groq(chunk, *_args):
        requests.append(asr._wav_duration(chunk))
        return {"text": "", "segments": [], "words": []}

    monkeypatch.setattr(asr, "_request_groq", fake_groq)
    asr.transcribe_part(
        audio,
        output,
        None,
        "ru",
        "",
        provider="groq",
        chunk_minutes=30,
    )

    assert len(requests) == 2
    assert max(requests) <= asr.GROQ_FREE_MAX_CHUNK_SECONDS


def test_groq_transcribe_part_validates_a_real_response_and_tolerates_its_overlap(
    tmp_path, monkeypatch
):
    """The groq branch used to return whatever the API sent without validating it at
    all; this real, overlap-containing fixture must now pass (allow_overlap=True is
    correct for Whisper), proving the added call did not become a strict one instead."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "groq_words.json").read_text(encoding="utf-8")
    )
    audio = tmp_path / "part.wav"
    _wav(audio, seconds=20)
    output = tmp_path / "asr" / "001.json"

    def fake_groq(chunk, *_args):
        return {"text": "", "segments": [], "words": fixture["words"]}

    monkeypatch.setattr(asr, "_request_groq", fake_groq)

    combined = asr.transcribe_part(audio, output, None, "ru", "", provider="groq")

    assert len(combined["words"]) == len(fixture["words"])


def test_groq_transcribe_part_rejects_a_word_outside_chunk_bounds(tmp_path, monkeypatch):
    """A word starting after the chunk ends must be rejected, not passed through:
    before this fix the groq branch never called validate_word_timestamps at all."""
    audio = tmp_path / "part.wav"
    _wav(audio, seconds=5, rate=1)
    output = tmp_path / "asr" / "001.json"

    def fake_groq(chunk, *_args):
        return {
            "text": "",
            "segments": [],
            "words": [{"word": "край", "start": 4.0, "end": 50.0, "confidence": 0.9}],
        }

    monkeypatch.setattr(asr, "_request_groq", fake_groq)

    with pytest.raises(ValueError):
        asr.transcribe_part(audio, output, None, "ru", "", provider="groq")


class _Status429(Exception):
    """Stands in for openai.APIStatusError, which needs a real httpx response to build."""

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.status_code = 429
        self.message = "rate limited"
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = types.SimpleNamespace(headers=headers)


def _groq_client_raising(sequence):
    """Client whose transcription call replays `sequence`: exceptions or a result."""
    calls = {"n": 0}

    def create(**_kwargs):
        item = sequence[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=types.SimpleNamespace(create=create))
    )
    return client, calls


def _install_groq(monkeypatch, client):
    from openai import DefaultHttpxClient

    from transcribe_meeting import asr

    monkeypatch.setenv("GROQ_API_KEY", "k")
    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **_: client, APIStatusError=_Status429, DefaultHttpxClient=DefaultHttpxClient
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return asr


def test_groq_retries_a_429_and_honours_retry_after(tmp_path, monkeypatch):
    """One 429 in the middle of a two-hour meeting used to kill the whole ASR stage."""
    audio = tmp_path / "chunk.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    ok = types.SimpleNamespace(
        model_dump=lambda: {
            "text": "ок",
            "segments": [],
            "words": [{"word": "ок", "start": 0, "end": 1}],
        }
    )
    client, calls = _groq_client_raising([_Status429(retry_after="4"), ok])
    asr = _install_groq(monkeypatch, client)
    slept = []
    monkeypatch.setattr(asr.time, "sleep", lambda s: slept.append(s))

    result = asr._transcribe_groq(audio, tmp_path / "out.json", "ru", "", limiter=None)

    assert result["text"] == "ок"
    assert calls["n"] == 2
    assert slept == [4.0], "the server named 4s — we wait exactly that, not our own estimate"


def test_groq_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    audio = tmp_path / "chunk.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    from transcribe_meeting import asr as asr_module

    attempts = asr_module.GROQ_RATE_LIMIT_RETRIES + 1
    client, calls = _groq_client_raising([_Status429() for _ in range(attempts)])
    asr = _install_groq(monkeypatch, client)
    monkeypatch.setattr(asr.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        asr._transcribe_groq(audio, tmp_path / "out.json", "ru", "", limiter=None)

    assert calls["n"] == attempts


def test_groq_does_not_retry_a_non_rate_limit_error(tmp_path, monkeypatch):
    """A bad key must fail immediately, not after three pointless waits."""
    audio = tmp_path / "chunk.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    bad = _Status429()
    bad.status_code = 401
    bad.message = "invalid api key"
    client, calls = _groq_client_raising([bad])
    asr = _install_groq(monkeypatch, client)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        asr._transcribe_groq(audio, tmp_path / "out.json", "ru", "", limiter=None)

    assert calls["n"] == 1
