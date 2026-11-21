"""Request and response shapes for OpenAI-compatible text endpoints."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from threading import Lock
from typing import Any

from .schemas import ReadablePayload

_LOG_LOCK = Lock()


@contextmanager
def _mute_sdk_log():
    # SDK debug records include full payloads/headers; HTTP logs include the URL
    # or response headers. Publish only our own sanitized diagnostic instead.
    loggers = [
        logging.getLogger(name)
        for name in (
            "openai._base_client",
            "httpx",
            "httpcore.connection",
            "httpcore.http11",
            "httpcore.http2",
            "httpcore.proxy",
            "httpcore.socks",
            "httpcore.connection_pool",
        )
    ]
    with _LOG_LOCK:
        previous = [logger.disabled for logger in loggers]
        for logger in loggers:
            logger.disabled = True
        try:
            yield
        finally:
            for logger, disabled in zip(loggers, previous, strict=True):
                logger.disabled = disabled


class UnusableReply(RuntimeError, ValueError):
    """A sanitized error when the endpoint's output does not pass local validation."""

    usage: dict[str, int] | None = None


class EndpointError(RuntimeError):
    """A sanitized endpoint failure; stop sending further batches on this attempt."""


def _chat_text(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise UnusableReply("the endpoint returned no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if getattr(message, "refusal", None):
        raise UnusableReply("the model refused the request")
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise UnusableReply("the model response was truncated")
    if finish_reason == "content_filter":
        raise UnusableReply("the model response was filtered")
    if finish_reason not in (None, "stop"):
        raise UnusableReply("the model response did not finish with text")
    return getattr(message, "content", None)


def _responses_text(response: Any) -> str:
    status = getattr(response, "status", None)
    if status == "incomplete" or getattr(response, "incomplete_details", None):
        raise UnusableReply("the model response was incomplete")
    if status not in (None, "completed") or getattr(response, "error", None):
        raise UnusableReply("the model response did not complete")
    parts: list[str] = []
    messages = getattr(response, "output", None)
    for message in messages if isinstance(messages, list) else ():
        if getattr(message, "type", None) != "message":
            continue
        if getattr(message, "status", None) in ("incomplete", "in_progress"):
            raise UnusableReply("the model response was incomplete")
        contents = getattr(message, "content", None)
        for content in contents if isinstance(contents, list) else ():
            kind = getattr(content, "type", None)
            if kind == "refusal":
                raise UnusableReply("the model refused the request")
            text = getattr(content, "text", None)
            if kind == "output_text" and isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def request_readable(
    client: Any,
    api: str,
    response_format: str,
    model: str,
    instructions: str,
    payload: str,
    reasoning: dict[str, str] | None,
) -> tuple[str, dict[str, int]]:
    from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

    schema = ReadablePayload.model_json_schema()
    format_options: dict[str, Any] = {"type": response_format}
    if response_format == "json_object":
        instructions += "\nReturn exactly one JSON object matching this schema:\n" + json.dumps(
            schema, ensure_ascii=False
        )
    else:
        named_schema = {"name": "readable_transcript", "strict": True, "schema": schema}
        if api == "chat":
            format_options["json_schema"] = named_schema
        else:
            format_options.update(named_schema)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": payload},
    ]

    options: dict[str, Any] = {"model": model}
    try:
        with _mute_sdk_log():
            if api == "chat":
                if reasoning is not None:
                    options["reasoning_effort"] = reasoning["effort"]
                response = client.chat.completions.create(
                    **options, messages=messages, response_format=format_options
                )
            else:
                if reasoning is not None:
                    options["reasoning"] = reasoning
                response = client.responses.create(
                    **options, store=False, input=messages, text={"format": format_options}
                )
    except APITimeoutError:
        raise EndpointError("readable request timed out") from None
    except APIConnectionError:
        raise EndpointError("readable could not reach the endpoint") from None
    except APIStatusError as exc:
        status_code = exc.status_code
        diagnostic = f"readable request failed with HTTP {status_code}"
        if 300 <= status_code < 400:
            diagnostic += "; redirects are not followed"
        elif api == "responses" and status_code in (404, 405):
            diagnostic += "; if this endpoint supports only Chat Completions, set "
            diagnostic += "TRANSCRIBE_READABLE_API=chat"
        raise EndpointError(diagnostic) from None
    except (APIError, json.JSONDecodeError, UnicodeDecodeError):
        raise EndpointError("readable endpoint returned an invalid API response") from None

    dump_usage = getattr(getattr(response, "usage", None), "model_dump", None)
    usage = dump_usage() if dump_usage is not None else {}
    token_fields = (
        ("prompt_tokens", "completion_tokens", "total_tokens")
        if api == "chat"
        else ("input_tokens", "output_tokens", "total_tokens")
    )
    counts = {
        field: value
        for name, field in zip(
            token_fields, ("input_tokens", "output_tokens", "total_tokens"), strict=True
        )
        if type(value := usage.get(name)) is int and value >= 0
    }
    try:
        output = _chat_text(response) if api == "chat" else _responses_text(response)
        if not isinstance(output, str) or not output.strip():
            raise UnusableReply("the endpoint returned empty text")
    except UnusableReply as exc:
        exc.usage = counts
        raise
    return output, counts
