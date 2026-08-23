"""Strict boundaries for persisted data and text-only LLM responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReadableItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    utterance_id: str
    readable_text: str
    # Strict JSON Schema providers require every property to be required. The
    # model must therefore return [] explicitly when it has no concerns.
    concerns: list[str]


class ReadablePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ReadableItem]
