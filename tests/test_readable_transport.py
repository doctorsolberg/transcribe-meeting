import json
import logging

import httpx
import pytest
from openai import OpenAI

from transcribe_meeting import readable as R
from transcribe_meeting.model import Utterance
from transcribe_meeting.schemas import ReadablePayload


@pytest.fixture(autouse=True)
def readable_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/v1")
    for name in ("API", "RESPONSE_FORMAT", "BATCH_SIZE", "REASONING"):
        monkeypatch.delenv(f"TRANSCRIBE_READABLE_{name}", raising=False)


def utterance(identifier="u1", text="тест 42"):
    return Utterance(identifier, "a", 0, 1, 0, 1, "S1", "S1", text, {})


def item(identifier="u1", text="Тест 42."):
    return {"utterance_id": identifier, "readable_text": text, "concerns": []}


def payload(*items):
    return json.dumps({"items": items}, ensure_ascii=False)


def chat(content=None, *, finish_reason="stop", refusal=None, index=0):
    return {
        "choices": [
            {
                "index": index,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def responses(content=None, *, status="completed", refusal=None):
    parts = [
        {"type": "refusal", "refusal": refusal}
        if refusal is not None
        else {"type": "output_text", "text": content, "annotations": []}
    ]
    return {
        "status": status,
        "output": [{"type": "message", "role": "assistant", "content": parts}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


def install_stub(monkeypatch, *replies):
    calls = []
    options = []
    replies = iter(replies)

    def send(request):
        calls.append((request.url.path, json.loads(request.content)))
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "private request text"}})
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def client(**kwargs):
        options.append(kwargs)
        if "http_client" not in kwargs:
            kwargs["http_client"] = transport_client(follow_redirects=True)
        return OpenAI(**kwargs)

    def transport_client(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(send), **kwargs)

    monkeypatch.setattr("openai.OpenAI", client)
    monkeypatch.setattr("openai.DefaultHttpxClient", transport_client)
    return calls, options


def test_chat_uses_chat_completions_with_strict_schema(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", "chat")
    calls, _ = install_stub(monkeypatch, chat(payload(item())))

    result = R.clean_readable([utterance()], "model-id", set())

    assert result.texts == {"u1": "Тест 42."}
    assert result.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert len(calls) == 1
    path, request = calls[0]
    assert path == "/v1/chat/completions"
    assert request["model"] == "model-id"
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "readable_transcript",
            "strict": True,
            "schema": ReadablePayload.model_json_schema(),
        },
    }
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert json.loads(request["messages"][1]["content"]) == [
        {"id": "u1", "speaker": "S1", "a": "тест 42"}
    ]
    assert set(request) == {"model", "messages", "response_format"}


def test_sdk_and_http_logging_do_not_emit_request_text_or_endpoint_path(monkeypatch, caplog):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", "chat")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/v1/private-route")
    install_stub(monkeypatch, 403)
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(RuntimeError, match="HTTP 403"),
    ):
        R.clean_readable([utterance(text="not-for-the-logs")], "model-id", set())
    assert "not-for-the-logs" not in caplog.text
    assert "private-route" not in caplog.text
    assert "test-key" not in caplog.text


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_json_object_requests_embed_the_same_schema_in_the_instructions(monkeypatch, api):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", api)
    monkeypatch.setenv("TRANSCRIBE_READABLE_RESPONSE_FORMAT", "json_object")
    reply = chat if api == "chat" else responses
    calls, _ = install_stub(monkeypatch, reply(payload(item())))

    assert R.clean_readable([utterance()], "m", set()).texts == {"u1": "Тест 42."}

    _, request = calls[0]
    if api == "chat":
        assert request["response_format"] == {"type": "json_object"}
        instructions = request["messages"][0]["content"]
    else:
        assert request["text"]["format"] == {"type": "json_object"}
        assert request["store"] is False
        instructions = request["input"][0]["content"]
    assert json.dumps(ReadablePayload.model_json_schema(), ensure_ascii=False) in instructions


def test_batch_size_is_positive_and_resolved_only_once(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_READABLE_BATCH_SIZE", "1")
    calls, _ = install_stub(monkeypatch, responses(payload(item())), responses(payload(item("u2"))))

    def status(message):
        if "batch 2/2" in message:
            monkeypatch.setenv("TRANSCRIBE_READABLE_API", "chat")
            monkeypatch.setenv("TRANSCRIBE_READABLE_BATCH_SIZE", "0")
            monkeypatch.setenv("TRANSCRIBE_READABLE_RESPONSE_FORMAT", "json_object")
            monkeypatch.setenv("TRANSCRIBE_READABLE_REASONING", "none")

    R.clean_readable([utterance(), utterance("u2")], "m", set(), status=status)

    assert [path for path, _ in calls] == ["/v1/responses", "/v1/responses"]
    assert all(request["text"]["format"]["type"] == "json_schema" for _, request in calls)
    assert all(request["reasoning"] == {"effort": "medium"} for _, request in calls)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("BATCH_SIZE", "0"),
        ("BATCH_SIZE", "2.5"),
        ("API", "private-garbage"),
        ("RESPONSE_FORMAT", "private-garbage"),
        ("REASONING", "private-garbage"),
    ],
)
def test_invalid_settings_fail_before_constructing_a_client(monkeypatch, setting, value):
    env_name = f"TRANSCRIBE_READABLE_{setting}"
    monkeypatch.setenv(env_name, value)
    _, options = install_stub(monkeypatch)

    with pytest.raises(R.ReadableConfigError, match=env_name) as error:
        R.clean_readable([utterance()], "m", set())

    assert not options
    assert "private-garbage" not in str(error.value)


@pytest.mark.parametrize("level", ["none", "low"])
def test_chat_only_sends_explicit_supported_reasoning(monkeypatch, level):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", "chat")
    monkeypatch.setenv("TRANSCRIBE_READABLE_REASONING", level)
    calls, _ = install_stub(monkeypatch, chat(payload(item())))

    R.clean_readable([utterance()], "m", set())

    request = calls[0][1]
    if level == "none":
        assert "reasoning_effort" not in request
    else:
        assert request["reasoning_effort"] == level


@pytest.mark.parametrize(
    ("reply", "diagnostic"),
    [
        (401, "HTTP 401"),
        (403, "HTTP 403"),
        (429, "HTTP 429"),
        (500, "HTTP 500"),
        (httpx.ReadTimeout("private request text"), "timed out"),
        (httpx.ConnectError("private request text"), "could not reach"),
    ],
)
def test_endpoint_failures_are_bounded_and_sanitized(monkeypatch, reply, diagnostic):
    monkeypatch.setenv("TRANSCRIBE_READABLE_BATCH_SIZE", "1")
    # Leave a spare reply for the RED case without SDK retry limits.
    calls, options = install_stub(monkeypatch, reply, reply, reply)

    with pytest.raises(RuntimeError, match=diagnostic) as error:
        R.clean_readable([utterance(), utterance("u2")], "m", set())

    assert "private request text" not in str(error.value)
    assert "test-key" not in str(error.value)
    assert len(calls) == 1
    assert options[0]["max_retries"] == 0
    assert 0 < options[0]["timeout"] <= 120


@pytest.mark.parametrize(
    ("api", "reply", "diagnostic"),
    [
        ("chat", {"choices": []}, "no choices"),
        ("chat", chat(" "), "empty"),
        ("chat", chat(payload(item()), refusal="private request text"), "refused"),
        ("chat", chat(payload(item()), finish_reason="length"), "truncated"),
        ("responses", responses(" "), "empty"),
        ("responses", responses(refusal="private request text"), "refused"),
        ("responses", responses(payload(item()), status="incomplete"), "incomplete"),
        ("responses", {"output": None}, "empty"),
    ],
)
def test_bad_reply_envelopes_are_not_accepted(monkeypatch, api, reply, diagnostic):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", api)
    calls, _ = install_stub(monkeypatch, reply, reply)

    with pytest.raises(RuntimeError, match=diagnostic) as error:
        R.clean_readable([utterance()], "m", set())

    assert len(calls) == 1
    assert "private request text" not in str(error.value)


@pytest.mark.parametrize(
    ("api", "reply"),
    [("chat", chat), ("responses", responses)],
)
@pytest.mark.parametrize(
    "invalid_items",
    [
        [dict(item(), readable_text={"private reply text": True})],
        [item(), item()],
        [item("private reply text")],
        [item(), item("unexpected-id")],
    ],
)
def test_schema_and_id_retry_never_echo_untrusted_output(monkeypatch, api, reply, invalid_items):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", api)
    monkeypatch.setenv("TRANSCRIBE_READABLE_RESPONSE_FORMAT", "json_object")
    invalid_reply = reply(payload(*invalid_items))
    calls, _ = install_stub(monkeypatch, invalid_reply, invalid_reply)
    seen = []

    with pytest.raises(RuntimeError, match="failed validation") as error:
        R.clean_readable([utterance()], "m", set(), status=seen.append)

    request_key = "messages" if api == "chat" else "input"
    assert len(calls) == 2
    instructions = [request[request_key][0]["content"] for _, request in calls]
    assert instructions[0] != instructions[1]
    assert "private reply text" not in instructions[1]
    assert "private reply text" not in str(error.value)
    assert all("private reply text" not in line for line in seen)


def test_batches_count_each_local_fallback_without_trusting_provider_notes(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", "chat")
    monkeypatch.setenv("TRANSCRIBE_READABLE_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("TRANSCRIBE_READABLE_BATCH_SIZE", "2")
    calls, _ = install_stub(
        monkeypatch,
        chat(payload(item(), item("u2", "Нет числа."))),
        chat("invalid JSON"),
        chat("still invalid JSON"),
        chat(payload(dict(item("u5"), concerns=["kept the local text: a provider note"]))),
    )
    utterances = [utterance(f"u{i}") for i in range(1, 6)]
    for original in utterances:
        original.light_text = "Локально 42."
    utterances[4].light_text = "Тест 42."

    result = R.clean_readable(utterances, "m", set())

    assert len(calls) == 4
    assert result.api == "chat"
    assert result.response_format == "json_object"
    assert result.accepted_utterances == 2  # An unchanged but valid utterance also counts.
    assert result.fallback_utterances == 3
    assert list(result.texts.values()) == ["Тест 42.", *["Локально 42."] * 3, "Тест 42."]
    assert result.usage == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
    assert result.repaired_utterances == 0
    assert {"utterance_id": "u5", "note": "the model flagged uncertainty"} in result.concerns
    assert all("a provider note" not in entry["note"] for entry in result.concerns)


def test_schema_valid_response_with_every_utterance_rejected_is_not_accepted(monkeypatch):
    calls, _ = install_stub(monkeypatch, responses(payload(item(text="Нет числа."))))

    with pytest.raises(R.UnusableReply, match="no utterances passed local content validation"):
        R.clean_readable([utterance()], "m", set())

    assert len(calls) == 1


def test_empty_input_is_not_accepted_and_never_constructs_a_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    calls, options = install_stub(monkeypatch)

    with pytest.raises(R.ReadableConfigError, match="no utterances to clean"):
        R.clean_readable([], "m", set())

    assert not calls and not options


def test_endpoint_failure_keeps_prior_accepted_batch_without_sending_more(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_READABLE_BATCH_SIZE", "1")
    calls, _ = install_stub(monkeypatch, responses(payload(item())), 429)

    result = R.clean_readable([utterance(), utterance("u2"), utterance("u3")], "m", set())

    assert len(calls) == 2
    assert result.texts == {"u1": "Тест 42.", "u2": "тест 42", "u3": "тест 42"}
    assert result.accepted_utterances == 1
    assert result.fallback_utterances == 2
    assert result.pending_utterance_ids == ["u2", "u3"]
    assert result.endpoint_failure == "readable request failed with HTTP 429"
    assert any("HTTP 429" in concern["note"] for concern in result.concerns)
    assert all("private request text" not in concern["note"] for concern in result.concerns)


def test_invalid_http_json_is_reported_without_echoing_provider_body(monkeypatch):
    calls, _ = install_stub(
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"private malformed body",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid API response") as error:
        R.clean_readable([utterance()], "m", set())

    assert "private malformed body" not in str(error.value)
    assert len(calls) == 1


def test_provenance_and_usage_do_not_expose_provider_urls_or_extra_usage_fields(monkeypatch):
    reply = responses(payload(item()))
    reply["usage"]["private response text"] = 578
    install_stub(monkeypatch, reply)

    result = R.clean_readable([utterance()], "m", set())

    assert result.provider == "openai-compatible"
    assert "provider.test" not in repr(result)
    assert "password" not in repr(result)
    assert result.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_invalid_base_url_does_not_echo_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://[user:private-pass/path")
    calls, _ = install_stub(monkeypatch)

    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL") as error:
        R.clean_readable([utterance()], "m", set())

    assert "private-pass" not in str(error.value)
    assert not calls


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("location", ["http://redirect.test/elsewhere", "/other-route"])
def test_all_redirects_are_rejected_without_sending_a_second_request(monkeypatch, code, location):
    calls, options = install_stub(
        monkeypatch,
        httpx.Response(code, headers={"location": location}),
        responses(payload(item())),
    )
    with pytest.raises(R.EndpointError, match=f"HTTP {code}") as error:
        R.clean_readable([utterance()], "m", set())
    assert len(calls) == 1
    assert not options[0]["http_client"].follow_redirects
    assert "elsewhere" not in str(error.value)


def test_explicit_keyless_loopback_never_uses_a_proxy_or_inherited_auth(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.test:8888")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_ORG_ID", "private-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "private-project")
    calls = []

    def send(request):
        calls.append(request)
        return httpx.Response(200, json=responses(payload(item())))

    def transport_client(**kwargs):
        assert kwargs["trust_env"] is False
        return httpx.Client(transport=httpx.MockTransport(send), **kwargs)

    monkeypatch.setattr("openai.DefaultHttpxClient", transport_client)
    assert R.clean_readable([utterance()], "m", set()).texts == {"u1": "Тест 42."}
    assert len(calls) == 1
    assert calls[0].url.host == "127.0.0.1"
    assert "Authorization" not in calls[0].headers
    assert "OpenAI-Organization" not in calls[0].headers
    assert "OpenAI-Project" not in calls[0].headers


@pytest.mark.parametrize("code", [404, 405])
def test_responses_route_failure_suggests_explicit_chat_without_retry(monkeypatch, code):
    calls, _ = install_stub(monkeypatch, code)
    with pytest.raises(R.EndpointError, match="TRANSCRIBE_READABLE_API=chat"):
        R.clean_readable([utterance()], "m", set())
    assert len(calls) == 1
