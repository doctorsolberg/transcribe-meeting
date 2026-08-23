from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .util import atomic_json, read_json

GIGAAM_COMMIT = "7447938d791c4f3e643386ee22c33777004293a5"
GROQ_MODEL = "whisper-large-v3"
# The registry entry whose name goes in config["asr_models"]/asr/<model>/... for the
# Groq leg. Kept distinct from GROQ_MODEL (the literal Whisper model id the API call
# uses) so a mixed local+groq run never confuses "which registry model" with "which
# upstream Whisper checkpoint".
GROQ_ASR_MODEL_NAME = "groq-whisper-large-v3"
DEEPGRAM_API_MODEL = "nova-3"
# Same "registry name" vs "literal upstream id" split as GROQ_ASR_MODEL_NAME/GROQ_MODEL.
DEEPGRAM_ASR_MODEL_NAME = "deepgram-nova-3"
# Env var each network ASR provider needs, keyed by ASR_MODELS[name]["provider"] —
# shared by cli.py's degrade-without-key messages and result.py's dropped-model warning
# so neither hardcodes "GROQ_API_KEY" for a model that was actually dropped for lack
# of DEEPGRAM_API_KEY (or vice versa).
ASR_API_KEY_ENV = {"groq": "GROQ_API_KEY", "deepgram": "DEEPGRAM_API_KEY"}
# MD5s copied from the installed package's own _MODEL_HASHES (already upstream-verified);
# see .venv/lib/python3.12/site-packages/gigaam/__init__.py. Each entry's "provider"
# is what run_job branches on per model instead of a single global
# --asr-provider flag, so a job can mix local GigaAM models with the Groq model.
# "scope" tells run_job how the model is invoked: "part" loops the
# model once per source file (local/groq, as always); "meeting" (deepgram-nova-3
# only, so far) sends the whole concatenated work/meeting.wav in a single request,
# because that is the only way its speaker labels stay consistent across parts.
ASR_MODELS: dict[str, dict[str, Any]] = {
    "multilingual_large_ctc": {
        "provider": "local",
        "scope": "part",
        "checkpoint": "multilingual_large_ctc.ckpt",
        "md5": "79a9adde50dd7f35bbf70927cb6557d0",
    },
    "v3_rnnt": {
        "provider": "local",
        "scope": "part",
        "checkpoint": "v3_rnnt.ckpt",
        "md5": "0fd2c9a1ff66abd8d32a3a07f7592815",
    },
    "v3_ctc": {
        "provider": "local",
        "scope": "part",
        "checkpoint": "v3_ctc.ckpt",
        "md5": "73413e7be9c6a5935827bfab5c0dd678",
    },
    "v2_rnnt": {
        "provider": "local",
        "scope": "part",
        "checkpoint": "v2_rnnt.ckpt",
        "md5": "547460139acfebd842323f59ed54ab54",
    },
    "v2_ctc": {
        "provider": "local",
        "scope": "part",
        "checkpoint": "v2_ctc.ckpt",
        "md5": "e00f59cb5d39624fb30d1786044795bf",
    },
    GROQ_ASR_MODEL_NAME: {"provider": "groq", "scope": "part", "api_model": GROQ_MODEL},
    DEEPGRAM_ASR_MODEL_NAME: {
        "provider": "deepgram",
        "scope": "meeting",
        "api_model": DEEPGRAM_API_MODEL,
    },
}
# e2e models (v3_e2e_*) are deliberately excluded: they need a separate tokenizer file
# that `setup` does not download. Do not add them here without also fixing setup.
DEFAULT_ASR_MODELS = ("multilingual_large_ctc", "v3_rnnt")
VAD_REPOSITORY = "pyannote/segmentation-3.0"
ASR_PROVIDERS = ("local", "groq")
ASR_DEVICES = ("auto", "mps", "cpu")
ASR_OUTPUT_SCHEMA_VERSION = 3
LOCAL_VAD_TARGET_SECONDS = 22.0
LOCAL_VAD_ABSOLUTE_SECONDS = 23.5  # was 24.0: leaves room for the context padding below
LOCAL_VAD_MIN_SECONDS = 15.0
#: Extra audio decoded on each side of a chunk. 42% of chunks on a real recording end
#: exactly at the target, i.e. mid-speech; without context the word on the seam is cut in
#: half and recognised twice as garbage. 23.5 + 2*0.5 = 24.5 s stays under GigaAM's 25 s limit.
LOCAL_VAD_CONTEXT_SECONDS = 0.5

GROQ_FREE_MAX_UPLOAD_BYTES = 24_000_000
GROQ_FREE_MAX_CHUNK_SECONDS = 10 * 60
GROQ_FREE_REQUESTS_PER_MINUTE = 20
GROQ_FREE_AUDIO_SECONDS_PER_HOUR = 7_200
GROQ_FREE_AUDIO_SECONDS_PER_DAY = 28_800
GROQ_PROMPT_MAX_CHARACTERS = 600
#: How many times a 429 is retried before the chunk (and the stage) fails, and how long
#: to wait when the response carries no Retry-After header.
GROQ_RATE_LIMIT_RETRIES = 3
GROQ_RATE_LIMIT_BACKOFF_SECONDS = (10.0, 30.0, 90.0)

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
# Documented Deepgram limit is on the request body, not on audio duration (a real
# 218 MB/113.8-minute request finished in 79s well inside it) — chunking would
# solve a problem this route does not have. 2 GiB, matching "API limit" in the plan.
DEEPGRAM_MAX_UPLOAD_BYTES = 2 * 1024**3
#: Deepgram's documented processing timeout is 10 minutes; without an explicit
#: timeout a stalled connection would hang the run forever instead of failing it.
DEEPGRAM_REQUEST_TIMEOUT_SECONDS = 900
#: Shortest span a word may occupy once normalized; see deepgram_words_for_source.
MIN_WORD_SECONDS = 0.01


class LocalASRBackend(Protocol):
    device: str

    def transcribe(self, audio: Path) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _finite_time(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("ASR returned a NaN or infinite timestamp.")
    return result


def validate_word_timestamps(
    words: list[dict[str, Any]],
    duration: float,
    *,
    tolerance: float = 0.05,
    allow_overlap: bool = False,
) -> None:
    """Reject invalid upstream timings instead of manufacturing substitute timings.

    ``allow_overlap`` relaxes only the ordering rule, never the bounds checks. Backends
    that align to acoustic frames (GigaAM) cannot legitimately emit overlapping spans, so
    a violation there means something broke. Backends that estimate each word's span
    independently (Deepgram, Whisper) routinely do: on a real two-hour recording Deepgram
    overlapped 290 of 16460 spans by a median 0.67 s while only 6 words moved backwards in
    start time. Nothing downstream depends on non-overlap — words are assigned to
    utterances by their centre and rendered in the model's own order — so rejecting those
    responses would discard a correct transcript over a harmless property.
    """
    previous_end = -tolerance
    for index, word in enumerate(words, 1):
        text = str(word.get("word", word.get("text", ""))).strip()
        start = _finite_time(word["start"])
        end = _finite_time(word["end"])
        if not text:
            raise ValueError(f"ASR returned an empty word #{index}.")
        if start < -tolerance or end > duration + tolerance or start >= end:
            raise ValueError(
                f"ASR returned a timestamp outside the chunk bounds for word {text!r}."
            )
        if not allow_overlap and start + tolerance < previous_end:
            raise ValueError("ASR returned out-of-order word timestamps.")
        previous_end = max(previous_end, end)


def gigaam_result_to_asr(result: Any, duration: float) -> dict[str, Any]:
    """Normalize GigaAM's dataclass result into our root-level word contract."""
    raw_words = getattr(result, "words", None)
    if raw_words is None:
        raise RuntimeError(
            "GigaAM did not return word timestamps; continuing with VAD timestamps is not possible."
        )
    words = [
        {
            "word": str(item.text),
            "start": float(item.start),
            "end": float(item.end),
            "probability": None,
        }
        for item in raw_words
    ]
    validate_word_timestamps(words, duration)
    return {"text": str(getattr(result, "text", "")), "segments": [], "words": words}


def validate_asr_models(models: list[str]) -> None:
    """Registry/count/duplicate/provider checks shared by CLI parsing for `run` and
    `setup`. Legacy --asr-provider compatibility (mismatch against an explicit
    --asr-models) is a separate, caller-side check in cli.py, because it also needs
    the requested provider flag, which this function does not take.
    """
    if not models:
        raise ValueError("Specify at least one ASR model.")
    if len(models) > 3:
        raise ValueError("No more than three ASR models are supported at once.")
    if len(set(models)) != len(models):
        raise ValueError("ASR models must not repeat.")
    unknown = [name for name in models if name not in ASR_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown ASR models: {', '.join(unknown)}. Available: {', '.join(ASR_MODELS)}."
        )
    # A second groq model would just repeat the same API and model as the first, so
    # it can never add information a third slot is meant to add.
    groq_count = sum(1 for name in models if ASR_MODELS[name]["provider"] == "groq")
    if groq_count > 1:
        raise ValueError("No more than one groq model is supported at once.")
    # Same reasoning as groq, plus a structural one: a second deepgram-nova-3 would
    # share the same meeting.json checkpoint and so would literally
    # be the exact same run, not a second opinion.
    deepgram_count = sum(1 for name in models if ASR_MODELS[name]["provider"] == "deepgram")
    if deepgram_count > 1:
        raise ValueError("No more than one deepgram model is supported at once.")


def _degrade_provider_without_key(
    models: list[str], provider: str, *, has_key: bool
) -> tuple[list[str], list[str]]:
    """Drop every model of one network provider when its key is missing, and report
    what was dropped. Mirrors readable's own "degrade with a warning, don't hard-fail
    the whole run over one missing key" pattern (see cli.run's readable handling)
    instead of refusing the entire job because of a network model nobody can
    currently reach.
    """
    if has_key:
        return list(models), []
    dropped = [name for name in models if ASR_MODELS[name]["provider"] == provider]
    kept = [name for name in models if name not in dropped]
    return kept, dropped


def degrade_groq_without_key(models: list[str], *, has_key: bool) -> tuple[list[str], list[str]]:
    return _degrade_provider_without_key(models, "groq", has_key=has_key)


def degrade_deepgram_without_key(
    models: list[str], *, has_key: bool
) -> tuple[list[str], list[str]]:
    return _degrade_provider_without_key(models, "deepgram", has_key=has_key)


def effective_asr_device(requested: str, *, mps_available: bool | None = None) -> str:
    if requested not in ASR_DEVICES:
        raise ValueError("--asr-device: auto, mps, or cpu")
    if mps_available is None:
        try:
            import torch

            mps_available = bool(torch.backends.mps.is_available())
        except ImportError:
            mps_available = False
    if requested == "mps" and not mps_available:
        raise RuntimeError("Requested --asr-device mps, but PyTorch MPS is unavailable.")
    return "mps" if requested == "mps" or (requested == "auto" and mps_available) else "cpu"


@dataclass
class GigaAMBackend:
    """One GigaAM instance for one ASR stage; VAD is deliberately separate.

    Only one backend may be loaded at a time: multilingual_large_ctc peaked at
    3.1 GB RSS in a measured run, and two of those plus the OS do not fit
    comfortably in 16 GB;
    callers are responsible for closing one model before constructing the next.
    """

    model_cache: Path
    device: str
    model_name: str
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            if ASR_MODELS.get(self.model_name, {}).get("provider") != "local":
                raise ValueError(f"Unknown local GigaAM model: {self.model_name}")
            info = ASR_MODELS[self.model_name]
            checkpoint = self.model_cache / info["checkpoint"]
            if not checkpoint.is_file():
                # This should already be unreachable through `run` (its preflight —
                # cli._check_route_ready — rejects a missing checkpoint before
                # create_job even runs) — kept as defence in depth for any other
                # caller of this backend, so it still names the exact fix instead of
                # a generic "setup" that may not even download this model.
                raise RuntimeError(
                    f"GigaAM checkpoint {self.model_name} not found in models/gigaam. Run: "
                    f"transcribe-meeting setup --asr-models {self.model_name}"
                )
            digest = hashlib.md5()
            with checkpoint.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != info["md5"]:
                raise RuntimeError(
                    f"GigaAM checkpoint {self.model_name} failed the upstream MD5 check. "
                    "Do not use it; it is safe to delete only this file and re-run setup."
                )
            try:
                import gigaam
            except ImportError as exc:
                raise RuntimeError("GigaAM is missing: run `uv sync --group dev`.") from exc
            # Passing the project cache prevents GigaAM from creating the only copy under ~/.cache.
            self._model = gigaam.load_model(
                self.model_name,
                device=self.device,
                download_root=str(self.model_cache),
            )
        return self._model

    def transcribe(self, audio: Path) -> dict[str, Any]:
        result = self._load().transcribe(str(audio), word_timestamps=True)
        return gigaam_result_to_asr(result, _wav_duration(audio))

    def close(self) -> None:
        self._model = None
        gc.collect()
        if self.device == "mps":
            try:
                import torch

                torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass


def _wav_duration(audio: Path) -> float:
    with wave.open(str(audio), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _write_wav_slice(source: Path, target: Path, start: float, end: float) -> None:
    """Copy exact PCM frames without modifying the source working WAV."""
    with wave.open(str(source), "rb") as reader:
        frames_per_second = reader.getframerate()
        first = max(0, round(start * frames_per_second))
        count = max(0, round((end - start) * frames_per_second))
        reader.setpos(min(first, reader.getnframes()))
        data = reader.readframes(count)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(data)


def split_vad_regions(
    regions: list[tuple[float, float]],
    duration: float,
    *,
    target_seconds: float = LOCAL_VAD_TARGET_SECONDS,
    absolute_seconds: float = LOCAL_VAD_ABSOLUTE_SECONDS,
    min_seconds: float = LOCAL_VAD_MIN_SECONDS,
) -> list[tuple[float, float]]:
    """Deterministically merge speech activity and hard-split every oversized region."""
    if target_seconds <= 0 or absolute_seconds < target_seconds:
        raise ValueError("Invalid VAD chunk limits.")
    normalized: list[tuple[float, float]] = []
    for start, end in sorted(regions):
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end > start:
            normalized.append((start, end))
    chunks: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    for start, end in normalized:
        while end - start > absolute_seconds:
            if current:
                chunks.append(current)
                current = None
            chunks.append((start, start + target_seconds))
            start += target_seconds
        if current is None:
            current = (start, end)
            continue
        combined = end - current[0]
        if combined <= target_seconds or (
            current[1] - current[0] < min_seconds and combined <= absolute_seconds
        ):
            current = (current[0], end)
        else:
            chunks.append(current)
            current = (start, end)
    if current:
        chunks.append(current)
    strict_chunks: list[tuple[float, float]] = []
    for start, end in chunks:
        while end - start > absolute_seconds:
            strict_chunks.append((start, start + target_seconds))
            start += target_seconds
        strict_chunks.append((start, end))
    # Round only at persistence boundary; preserve no more than the strict public limit.
    return [(round(start, 6), round(end, 6)) for start, end in strict_chunks if end > start]


def detect_speech_regions(audio: Path, model_path: Path) -> list[tuple[float, float]]:
    """Run only segmentation-3.0 VAD locally on CPU, never Community-1."""
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if not model_path.exists():
        raise RuntimeError("Local VAD model segmentation-3.0 not found. Run setup.")
    try:
        import torch
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is missing: run `uv sync --group dev`.") from exc
    model = Model.from_pretrained(str(model_path))
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    pipeline.to(torch.device("cpu"))
    annotation = pipeline(str(audio))
    return [(float(turn.start), float(turn.end)) for turn in annotation.get_timeline().support()]


def local_vad_parameters() -> dict[str, float]:
    """The VAD/chunking parameters recorded in a chunk fingerprint, so changing any of
    them invalidates an unfinished part's checkpoints instead of mixing two chunkings."""
    return {
        "min_duration_on": 0.0,
        "min_duration_off": 0.0,
        "target": LOCAL_VAD_TARGET_SECONDS,
        "absolute": LOCAL_VAD_ABSOLUTE_SECONDS,
        "context": LOCAL_VAD_CONTEXT_SECONDS,
    }


#: Keys of a local ASR fingerprint that identify the model rather than the audio or
#: the chunking. The VAD plan is independent of all of them.
_MODEL_FINGERPRINT_KEYS = ("gigaam_model", "gigaam_model_hash", "gigaam_commit", "device")


def vad_plan_fingerprint(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Strip the model identity out of a local ASR fingerprint.

    What the plan actually depends on is the source audio, the VAD checkpoint and the
    chunking parameters — all of which stay. Dropping the model keys is what lets two
    models in one job share a single segmentation pass.
    """
    return {key: value for key, value in fingerprint.items() if key not in _MODEL_FINGERPRINT_KEYS}


def _local_chunk_fingerprint(
    fingerprint: dict[str, Any], start: float, end: float
) -> dict[str, Any]:
    return {
        **fingerprint,
        "start": start,
        "end": end,
        "context": LOCAL_VAD_CONTEXT_SECONDS,
        "schema": ASR_OUTPUT_SCHEMA_VERSION,
    }


def _local_part(
    audio: Path,
    output: Path,
    *,
    backend: LocalASRBackend,
    vad_model_path: Path,
    fingerprint: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    duration = _wav_duration(audio)
    # One plan per source file, not per ASR model: which GigaAM checkpoint will read
    # the chunks does not change where the speech is. Keeping it under asr/<model>/
    # keyed by the full fingerprint made every extra model re-run segmentation-3.0
    # over the whole part to arrive at a byte-identical plan.
    vad_path = output.parent.parent / "vad" / f"{output.stem}.json"
    vad_fingerprint = vad_plan_fingerprint(fingerprint)
    if vad_path.exists() and read_json(vad_path).get("fingerprint") == vad_fingerprint:
        plan = read_json(vad_path).get("chunks", [])
    else:
        regions = detect_speech_regions(audio, vad_model_path)
        plan = [
            {"index": index, "start": start, "end": end}
            for index, (start, end) in enumerate(split_vad_regions(regions, duration), 1)
        ]
        atomic_json(
            vad_path,
            {
                "fingerprint": vad_fingerprint,
                "regions": regions,
                "chunks": plan,
                "duration": duration,
            },
        )
    combined: dict[str, Any] = {"text": "", "segments": [], "words": [], "chunks": []}
    chunk_root = output.parent / "chunks" / output.stem
    for item in plan:
        index, start, end = int(item["index"]), float(item["start"]), float(item["end"])
        # Decode a bit of the neighbouring chunk on each side so a word sitting on the
        # seam is heard whole, not cut in half twice. absolute + 2*context stays below
        # GigaAM's 25 s longform limit (see LOCAL_VAD_CONTEXT_SECONDS above).
        slice_start = max(0.0, start - LOCAL_VAD_CONTEXT_SECONDS)
        slice_end = min(duration, end + LOCAL_VAD_CONTEXT_SECONDS)
        wav_path = chunk_root / f"{index:03d}.wav"
        json_path = chunk_root / f"{index:03d}.json"
        chunk_fp = _local_chunk_fingerprint(fingerprint, start, end)
        checkpoint = read_json(json_path) if json_path.exists() else None
        if not checkpoint or checkpoint.get("checkpoint_fingerprint") != chunk_fp:
            _write_wav_slice(audio, wav_path, slice_start, slice_end)
            value = backend.transcribe(wav_path)
            validate_word_timestamps(value.get("words", []), slice_end - slice_start)
            checkpoint = {**value, "checkpoint_fingerprint": chunk_fp}
            atomic_json(json_path, checkpoint)
        # Keep only the words whose centre falls in this chunk's nominal window: the
        # padding on each side was also decoded by the neighbouring chunk, so without
        # this a seam word would be duplicated, and clipping to the window is what
        # keeps every kept word inside [start, end) for the final duration check below.
        lower, upper = start - slice_start, end - slice_start
        kept = []
        for word in checkpoint.get("words", []):
            center = (float(word["start"]) + float(word["end"])) / 2
            if not (lower <= center < upper):
                continue  # the word belongs to the neighbouring chunk, which also decoded it
            kept.append(
                {
                    **word,
                    "start": max(float(word["start"]), lower),
                    "end": min(float(word["end"]), upper),
                }
            )
        value = _shift_timestamps(
            {"text": " ".join(word["word"] for word in kept), "segments": [], "words": kept},
            slice_start,
        )
        combined["text"] += (" " if combined["text"] and value["text"] else "") + value["text"]
        combined["segments"].extend(value["segments"])
        combined["words"].extend(value["words"])
        combined["chunks"].append(
            {
                "index": index,
                "start": start,
                "end": end,
                "slice_start": slice_start,
                "slice_end": slice_end,
                "path": str(json_path),
            }
        )
    validate_word_timestamps(combined["words"], duration)
    combined["run_parameters"] = {
        "provider": "local",
        "backend": "gigaam",
        "model": model_name,
        "device": backend.device,
        "word_timestamps": True,
        "output_schema_version": ASR_OUTPUT_SCHEMA_VERSION,
    }
    atomic_json(output, combined)
    return combined


@dataclass
class GroqFreeTierLimiter:
    status: Callable[[str], None] | None = None
    requests: list[float] = field(default_factory=list)
    audio: list[tuple[float, float]] = field(default_factory=list)

    def before_upload(self, duration: float) -> None:
        while True:
            now = time.monotonic()
            self.requests = [stamp for stamp in self.requests if now - stamp < 60]
            self.audio = [(stamp, seconds) for stamp, seconds in self.audio if now - stamp < 3600]
            request_wait = (
                self.requests[0] + 60 - now
                if len(self.requests) >= GROQ_FREE_REQUESTS_PER_MINUTE
                else 0
            )
            used = sum(seconds for _, seconds in self.audio)
            audio_wait = (
                self.audio[0][0] + 3600 - now
                if used + duration > GROQ_FREE_AUDIO_SECONDS_PER_HOUR and self.audio
                else 0
            )
            wait = max(request_wait, audio_wait, 0)
            if not wait:
                self.requests.append(now)
                self.audio.append((now, duration))
                return
            if self.status:
                self.status(f"Groq: rate limit; waiting {int(wait) + 1} s")
            time.sleep(wait + 0.1)


def _retry_after_seconds(exc: Any) -> float | None:
    """Read Retry-After off a 429, preferring the server's own number to our guess."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    try:
        return max(0.0, float(header)) if header else None
    except (TypeError, ValueError):
        return None


def _groq_prompt(prompt: str) -> str | None:
    if len(prompt) <= GROQ_PROMPT_MAX_CHARACTERS:
        return prompt or None
    half = (GROQ_PROMPT_MAX_CHARACTERS - 5) // 2
    return f"{prompt[:half]} … {prompt[-half:]}"


def _transcribe_groq(
    audio: Path, output: Path, language: str, prompt: str, *, limiter: GroqFreeTierLimiter | None
) -> dict[str, Any]:
    token = os.environ.get("GROQ_API_KEY")
    if not token:
        raise RuntimeError("For --asr-provider groq, set GROQ_API_KEY in the environment.")
    if audio.stat().st_size > GROQ_FREE_MAX_UPLOAD_BYTES:
        raise RuntimeError("Groq free accepts up to 25 MB per upload; reduce --chunk-minutes.")
    if limiter:
        limiter.before_upload(_wav_duration(audio))
    try:
        from openai import APIStatusError, OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is missing; run uv sync.") from exc
    client = OpenAI(api_key=token, base_url="https://api.groq.com/openai/v1")
    status = limiter.status if limiter else None
    # The limiter models the documented free-tier ceilings, but an account can be
    # throttled for reasons it cannot see (a stricter tier, a shared key, a burst
    # elsewhere). Before this, one 429 in the middle of a two-hour meeting raised and
    # took the whole ASR stage down; every earlier chunk survives as a checkpoint, but
    # the run still had to be restarted by hand.
    for attempt in range(GROQ_RATE_LIMIT_RETRIES + 1):
        try:
            with audio.open("rb") as handle:
                response = client.audio.transcriptions.create(
                    file=(audio.name, handle, "audio/wav"),
                    model=GROQ_MODEL,
                    language=None if language == "auto" else language,
                    prompt=_groq_prompt(prompt),
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                    temperature=0,
                )
            break
        except APIStatusError as exc:
            if exc.status_code != 429 or attempt == GROQ_RATE_LIMIT_RETRIES:
                raise RuntimeError(
                    f"Groq ASR returned HTTP {exc.status_code}: {exc.message}"
                ) from exc
            wait = _retry_after_seconds(exc) or GROQ_RATE_LIMIT_BACKOFF_SECONDS[attempt]
            if status:
                status(
                    f"Groq: HTTP 429, retrying in {wait:.0f} s "
                    f"(attempt {attempt + 2} of {GROQ_RATE_LIMIT_RETRIES + 1})"
                )
            time.sleep(wait)
    result = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    result["run_parameters"] = {
        "provider": "groq",
        "model": GROQ_MODEL,
        "word_timestamps": True,
        "language": language,
        "output_schema_version": ASR_OUTPUT_SCHEMA_VERSION,
    }
    atomic_json(output, result)
    return result


def _shift_timestamps(data: dict[str, Any], offset: float) -> dict[str, Any]:
    for segment in data.get("segments", []):
        for key in ("start", "end"):
            if key in segment:
                segment[key] = float(segment[key]) + offset
        for word in segment.get("words", []):
            for key in ("start", "end"):
                if key in word:
                    word[key] = float(word[key]) + offset
    for word in data.get("words", []):
        for key in ("start", "end"):
            if key in word:
                word[key] = float(word[key]) + offset
    return data


def _deepgram_language_param(language: str) -> str:
    # nova-3's "multi" mode is code-switching across ~10 languages including Russian
    # — the reason this route exists at all, unlike local/groq's
    # "auto" which just means "detect one language". Mapped here, not by the caller,
    # so every entry point (ASR stage, diarization-only stage) maps it identically.
    return "multi" if language == "auto" else language


def _deepgram_request(audio_bytes: bytes, api_key: str, language: str) -> dict[str, Any]:
    """The network boundary, isolated so tests can monkeypatch it without touching
    urllib or performing a real HTTP request."""
    params = {
        "model": DEEPGRAM_API_MODEL,
        "language": _deepgram_language_param(language),
        # Diarization is enabled BY specifying diarize_model; passing "diarize" as well is a
        # 400 ("diarize_model cannot be used together with diarize or diarize_version"),
        # verified against the live API. Do not "make it explicit" — it is exclusive.
        "diarize_model": "latest",
        "smart_format": "true",
        "punctuate": "true",
    }
    url = f"{DEEPGRAM_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        data=audio_bytes,
        method="POST",
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DEEPGRAM_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Deepgram ASR returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Deepgram ASR is unavailable: {exc.reason}") from exc


def transcribe_deepgram(audio: Path, api_key: str, language: str, target: Path) -> dict[str, Any]:
    """One Deepgram request serves both ASR and diarization: the raw
    response is written to `target` and doubles as the checkpoint for both stages —
    whichever of them runs second reads it back from disk instead of paying for and
    waiting on a second request. `target` is normally
    `<job>/asr/deepgram-nova-3/meeting.json`, shared by name regardless of whether the
    ASR stage actually requested deepgram-nova-3.
    """
    if target.exists():
        return read_json(target)
    size = audio.stat().st_size
    if size > DEEPGRAM_MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"The concatenated track {size / 1024**3:.2f} GB exceeds the Deepgram limit of "
            f"{DEEPGRAM_MAX_UPLOAD_BYTES / 1024**3:.0f} GB per request."
        )
    data = _deepgram_request(audio.read_bytes(), api_key, language)
    atomic_json(target, data)
    return data


def deepgram_words(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Words exactly as Deepgram ordered them — never re-sorted by timestamp (plan
    section 4; the same fix already applied for Whisper's out-of-order words)."""
    try:
        return data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError("The Deepgram response has no words at the expected path.") from exc


def deepgram_words_for_source(
    all_words: list[dict[str, Any]], base: float, duration: float
) -> list[dict[str, Any]]:
    """Slice one source's words out of the meeting-wide response and shift them into
    its local clock. `all_words` is in diarization-clock time because
    the request ran on the concatenated work/meeting.wav; `base` is that source's
    `diarization_offset`, the same value word-to-speaker assignment already uses to
    undo the synthetic inter-part gap (timeline.build_sources/GAP_SECONDS). Order is
    preserved, not re-sorted — see deepgram_words. `end` is clamped to `duration` so
    upstream timing jitter across the exact source boundary can never fail
    validate_word_timestamps.
    """
    result = []
    for word in all_words:
        start = float(word["start"])
        if base <= start < base + duration:
            result.append(
                {
                    # smart_format writes case, punctuation and digitized numbers only
                    # into punctuated_word — README recommends this route specifically
                    # for that formatting, so plain "word" is only a fallback.
                    "word": word.get("punctuated_word") or word.get("word", ""),
                    "start": start - base,
                    # Deepgram occasionally emits a zero-length span (one word in 16460
                    # on a real recording, confidence 0.99). Widening it to the minimum
                    # keeps a genuine word that dropping would lose and that a weaker
                    # validator would let through for every backend.
                    "end": min(
                        max(float(word["end"]) - base, start - base + MIN_WORD_SECONDS),
                        duration,
                    ),
                    "probability": word.get("confidence"),
                }
            )
    return result


def transcribe_part(
    audio: Path,
    output: Path,
    model_path: Path | None,
    language: str,
    prompt: str,
    *,
    provider: str = "local",
    chunk_minutes: int = 0,
    ffmpeg_threads: int = 1,
    groq_limiter: GroqFreeTierLimiter | None = None,
    local_backend: LocalASRBackend | None = None,
    vad_model_path: Path | None = None,
    local_fingerprint: dict[str, Any] | None = None,
    model_name: str = "",
) -> dict[str, Any]:
    if provider not in ASR_PROVIDERS:
        raise ValueError(f"Unknown ASR provider: {provider}")
    if provider == "local":
        if local_backend is None or vad_model_path is None or local_fingerprint is None:
            raise RuntimeError("Local GigaAM ASR requires VAD, a backend, and a fingerprint.")
        return _local_part(
            audio,
            output,
            backend=local_backend,
            vad_model_path=vad_model_path,
            fingerprint=local_fingerprint,
            model_name=model_name,
        )
    duration = _wav_duration(audio)
    chunk_minutes = min(chunk_minutes or 10, GROQ_FREE_MAX_CHUNK_SECONDS // 60)
    cache = output.parent / "chunks" / output.stem
    combined: dict[str, Any] = {"segments": [], "words": [], "text": "", "chunks": []}
    for index, start in enumerate(range(0, int(duration) + 1, chunk_minutes * 60), 1):
        length = min(chunk_minutes * 60, duration - start)
        if length <= 0:
            continue
        chunk, result_path = cache / f"{index:03d}.wav", cache / f"{index:03d}.json"
        if not result_path.exists():
            chunk.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-threads",
                    str(ffmpeg_threads),
                    "-ss",
                    str(start),
                    "-t",
                    str(length),
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(chunk),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Failed to prepare ASR chunk {index}: {completed.stderr.strip()}"
                )
            _transcribe_groq(chunk, result_path, language, prompt, limiter=groq_limiter)
        value = _shift_timestamps(read_json(result_path), float(start))
        combined["segments"].extend(value.get("segments", []))
        combined["words"].extend(value.get("words", []))
        combined["text"] += value.get("text", "")
        combined["chunks"].append(
            {"index": index, "start": start, "end": start + length, "path": str(result_path)}
        )
    combined["run_parameters"] = {
        "provider": "groq",
        "model": GROQ_MODEL,
        "word_timestamps": True,
        "output_schema_version": ASR_OUTPUT_SCHEMA_VERSION,
        "language": language,
        "chunk_minutes": chunk_minutes,
    }
    # Whisper estimates each word's span independently, same as Deepgram, so
    # overlapping spans are normal and allowed; the bounds checks (empty word,
    # out-of-range timestamp, start >= end) still apply. Unlike the local path
    # (validated inside _local_part) and Deepgram (validated in core.py), nothing
    # upstream of this return validated the groq response at all.
    validate_word_timestamps(combined["words"], duration, allow_overlap=True)
    atomic_json(output, combined)
    return combined


def words_from_asr(data: dict[str, Any]) -> list[dict[str, Any]]:
    root_words = data.get("words", [])
    if root_words:
        result = []
        for word in root_words:
            text = str(word.get("word", word.get("text", "")))
            if result and text and not text[0].isspace() and not result[-1]["text"].endswith(" "):
                text = " " + text
            result.append(
                {
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "text": text,
                    "probability": word.get("probability"),
                }
            )
        return result
    return [
        {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": word.get("word", ""),
            "probability": word.get("probability"),
        }
        for segment in data.get("segments", [])
        for word in segment.get("words", [])
    ]
