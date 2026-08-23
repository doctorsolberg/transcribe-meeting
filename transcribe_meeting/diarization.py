from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .asr import deepgram_words, transcribe_deepgram
from .util import atomic_json, atomic_text

DIARIZATION_PROVIDERS = ("pyannote", "foxnose", "deepgram")
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"
# Not a real PyPI/HF id: a stable label for manifests and models.lock.json.
FOXNOSE_REPO = "diarize/wespeaker-voxceleb-resnet34-LM"
# Ditto: a stable label, not a real HF id — Deepgram is an API, not a downloaded model.
DEEPGRAM_REPO = "deepgram/nova-3"
WESPEAKER_MODEL_URL = (
    "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/"
    "voxceleb_resnet34_LM.onnx"
)
WESPEAKER_LANG = "en"
# Only overlap spans longer than this are worth marking in the transcript; shorter
# ones are ordinary turn-taking, not simultaneous speech.
OVERLAP_MIN_SECONDS = 0.5
# The same gap alignment.make_utterances hardcodes for merging
# consecutive same-speaker words into one utterance — reused here so a Deepgram
# diarization segment breaks on the same silence a canonical utterance would.
DEEPGRAM_SEGMENT_GAP = 0.8


def _write_rttm(rttm: Path, rows: list[dict[str, Any]]) -> None:
    """Shared writer so pyannote and foxnose produce byte-identical RTTM formatting."""
    atomic_text(
        rttm,
        "".join(
            f"SPEAKER meeting 1 {x['start']:.3f} {x['end'] - x['start']:.3f} "
            f"<NA> <NA> {x['speaker']} <NA> <NA>\n"
            for x in rows
        ),
    )


def diarize(
    audio: Path,
    output: Path,
    rttm: Path,
    model_path: Path,
    speakers: int | None,
    minimum: int,
    maximum: int,
    cpu_threads: int,
    *,
    provider: str,
    wespeaker_model: Path | None = None,
    deepgram_api_key: str | None = None,
    deepgram_language: str | None = None,
    deepgram_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if provider == "pyannote":
        return _diarize_pyannote(
            audio, output, rttm, model_path, speakers, minimum, maximum, cpu_threads
        )
    if provider == "foxnose":
        if wespeaker_model is None:
            raise RuntimeError(
                "--diarization-provider foxnose needs a path to the WeSpeaker model."
            )
        return _diarize_foxnose(
            audio, output, rttm, speakers, minimum, maximum, cpu_threads, wespeaker_model
        )
    if provider == "deepgram":
        # speakers/minimum/maximum/cpu_threads are deliberately unused: Deepgram's API
        # takes no speaker-count parameter and runs on their servers,
        # not ours.
        if not deepgram_api_key:
            raise RuntimeError("--diarization-provider deepgram needs DEEPGRAM_API_KEY.")
        if deepgram_checkpoint is None:
            raise RuntimeError(
                "--diarization-provider deepgram needs a path to the Deepgram checkpoint."
            )
        return _diarize_deepgram(
            audio, output, rttm, deepgram_api_key, deepgram_language or "auto", deepgram_checkpoint
        )
    raise RuntimeError(f"Unknown diarization provider: {provider}")


def _diarize_pyannote(
    audio: Path,
    output: Path,
    rttm: Path,
    model_path: Path,
    speakers: int | None,
    minimum: int,
    maximum: int,
    cpu_threads: int,
) -> dict[str, Any]:
    os.environ.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYANNOTE_METRICS_ENABLED": "0"}
    )
    try:
        import torch  # type: ignore
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is missing: run `uv run transcribe-meeting setup`."
        ) from exc
    if not model_path.exists():
        # Same defence-in-depth note as GigaAMBackend._load: `run` rejects this route
        # in cli._check_route_ready before create_job, so reaching here means some
        # other caller invoked diarization directly.
        raise RuntimeError(
            "Local pyannote model (Community-1) not found. Accept the terms on "
            "Hugging Face, then run: "
            "transcribe-meeting setup --diarization-provider pyannote"
        )
    torch.set_num_threads(cpu_threads)
    # PyTorch only permits this setting before parallel work begins in a process.
    with suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
    pipeline = Pipeline.from_pretrained(str(model_path))
    pipeline.to(torch.device("cpu"))
    kwargs = (
        {"num_speakers": speakers}
        if speakers
        else {"min_speakers": minimum, "max_speakers": maximum}
    )
    result = pipeline(str(audio), **kwargs)
    normal = getattr(result, "speaker_diarization", result)
    exclusive = getattr(result, "exclusive_speaker_diarization", normal)

    def rows(annotation: Any) -> list[dict[str, Any]]:
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]

    payload = {
        "provider": "pyannote",
        "repository": PYANNOTE_REPO,
        "cpu_threads": cpu_threads,
        "supports_overlap": True,
        "regular": rows(normal),
        "exclusive": rows(exclusive),
    }
    atomic_json(output, payload)
    _write_rttm(rttm, payload["regular"])
    return payload


def _deepgram_segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build speaker segments by gluing consecutive same-speaker words: a segment
    breaks on a speaker change or a gap longer than
    DEEPGRAM_SEGMENT_GAP. Words are consumed in response order (see deepgram_words),
    so segments come out already sorted by start.
    """
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for word in words:
        speaker = word.get("speaker")
        if speaker is None:
            continue
        label = f"SPEAKER_{int(speaker):02d}"
        start, end = float(word["start"]), float(word["end"])
        if (
            current is not None
            and current["speaker"] == label
            and start - current["end"] <= DEEPGRAM_SEGMENT_GAP
        ):
            current["end"] = end
        else:
            if current is not None:
                segments.append(current)
            current = {"start": start, "end": end, "speaker": label}
    if current is not None:
        segments.append(current)
    return segments


def _diarize_deepgram(
    audio: Path,
    output: Path,
    rttm: Path,
    api_key: str,
    language: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Derive diarization from the same Deepgram response the ASR stage uses (plan
    sections 3 and 5): `checkpoint` is `transcribe_deepgram`'s own — a call from here
    when no deepgram-nova-3 ASR model ran this job makes the one request that both
    stages then share; a call from here after that model already ran reads it back
    instead of paying twice.
    """
    data = transcribe_deepgram(audio, api_key, language, checkpoint)
    words = deepgram_words(data)
    regular = _deepgram_segments_from_words(words)
    payload = {
        "provider": "deepgram",
        "repository": DEEPGRAM_REPO,
        # Deepgram diarizes on their servers, not ours; 0 makes that explicit rather
        # than recording a thread count that did no diarization work.
        "cpu_threads": 0,
        # One speaker label per word, never more than one active at once — Deepgram's
        # response has no concept of overlapping speakers to subtract.
        "supports_overlap": False,
        "regular": regular,
        "exclusive": regular,
        "num_speakers": len({row["speaker"] for row in regular}),
    }
    atomic_json(output, payload)
    _write_rttm(rttm, payload["regular"])
    return payload


def _filter_overlap_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only spans longer than OVERLAP_MIN_SECONDS.

    Split out from detect_overlap_regions so the filtering rule is testable without a
    real model or audio file.
    """
    return [(start, end) for start, end in regions if end - start > OVERLAP_MIN_SECONDS]


def detect_overlap_regions(
    audio: Path, model_path: Path, cpu_threads: int
) -> list[tuple[float, float]]:
    """Extract simultaneous-speech spans from pyannote/segmentation-3.0's powerset output.

    Reuses the VAD model already required by local ASR (models.lock.json's "vad" entry)
    instead of the gated speaker-diarization-community-1 pipeline: the overlap mark only
    needs overlap spans, not a full diarization. The installed pyannote.audio no longer ships a
    ready-made OverlappedSpeechDetection pipeline class, so this reimplements its logic
    directly on top of Inference: for a powerset model, Inference's default conversion
    already turns each chunk's soft powerset prediction into a hard per-speaker-slot
    multilabel array (pyannote.audio.utils.powerset.Powerset.forward, called from
    Inference.infer). Summing that array across the speaker axis and thresholding at
    >=2 — before overlap-add aggregation runs, exactly like the local VAD's own
    `np.max` pre_aggregation_hook in asr.detect_speech_regions — collapses it to a
    single, permutation-invariant "two or more speakers active" channel that can be
    averaged across sliding windows and then binarized into regions.
    """
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if not model_path.exists():
        raise RuntimeError("Local VAD model segmentation-3.0 not found. Run setup.")
    try:
        import numpy as np
        import torch
        from pyannote.audio import Inference, Model
        from pyannote.audio.utils.signal import Binarize
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is missing: run `uv run transcribe-meeting setup`."
        ) from exc
    torch.set_num_threads(cpu_threads)
    model = Model.from_pretrained(str(model_path))
    inference = Inference(
        model,
        pre_aggregation_hook=lambda scores: (scores.sum(axis=-1, keepdims=True) >= 2).astype(
            np.float32
        ),
    )
    inference.to(torch.device("cpu"))
    scores = inference(str(audio))
    binarize = Binarize(onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0)
    annotation = binarize(scores)
    regions = [(float(turn.start), float(turn.end)) for turn in annotation.get_timeline().support()]
    return _filter_overlap_regions(regions)


@contextmanager
def _local_wespeaker(model_path: Path, cpu_threads: int):
    """diarize hardcodes Speaker(lang="en"), which downloads the ONNX model on first use.

    Pin it to the locally provisioned file and let it use the run's thread budget
    (upstream hardcodes intra_op_num_threads=1). Both patches are restored in
    ``finally`` so the shim never leaks state, even when the wrapped call raises.
    """
    if not model_path.exists():
        raise RuntimeError(
            "Local WeSpeaker model not found. Run `uv run transcribe-meeting setup "
            "--diarization-provider foxnose`."
        )
    import wespeakerruntime as wsr
    from wespeakerruntime import hub as wsr_hub

    original_get_model = wsr_hub.Hub.get_model_by_lang
    original_init = wsr.Speaker.__init__

    def patched_get_model(lang: str) -> str:
        return str(model_path)

    def patched_init(
        self: Any,
        onnx_path: str | None = None,
        lang: str = "chs",
        inter_op_num_threads: int = 1,
        intra_op_num_threads: int | None = None,
    ) -> None:
        original_init(
            self,
            onnx_path=onnx_path,
            lang=lang,
            inter_op_num_threads=inter_op_num_threads,
            intra_op_num_threads=(
                cpu_threads if intra_op_num_threads is None else intra_op_num_threads
            ),
        )

    wsr_hub.Hub.get_model_by_lang = staticmethod(patched_get_model)
    wsr.Speaker.__init__ = patched_init
    try:
        yield
    finally:
        # __get__ unwraps the descriptor to a plain function; re-wrap so instance access
        # (Hub().get_model_by_lang(...)) keeps working exactly as it did before the shim.
        wsr_hub.Hub.get_model_by_lang = staticmethod(original_get_model)
        wsr.Speaker.__init__ = original_init


def _diarize_foxnose(
    audio: Path,
    output: Path,
    rttm: Path,
    speakers: int | None,
    minimum: int,
    maximum: int,
    cpu_threads: int,
    wespeaker_model: Path,
) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing: run `uv sync`.") from exc
    torch.set_num_threads(cpu_threads)
    # PyTorch only permits this setting before parallel work begins in a process.
    with suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2)))
    try:
        from diarize import diarize as foxnose_diarize
    except ImportError as exc:
        raise RuntimeError("The diarize package is missing: run `uv sync`.") from exc
    kwargs = (
        {"num_speakers": speakers}
        if speakers
        else {"min_speakers": minimum, "max_speakers": maximum}
    )
    with _local_wespeaker(wespeaker_model, cpu_threads):
        result = foxnose_diarize(str(audio), **kwargs)
    rows = sorted(
        (
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(segment.speaker),
            }
            for segment in result.segments
        ),
        key=lambda item: item["start"],
    )
    # Diagnostic-only field: serialize only if it is a well-formed pydantic model,
    # never let it break the diarization stage.
    estimation_details = (
        result.estimation_details.model_dump()
        if hasattr(result.estimation_details, "model_dump")
        else None
    )
    payload = {
        "provider": "foxnose",
        "repository": FOXNOSE_REPO,
        "cpu_threads": cpu_threads,
        "supports_overlap": False,
        "regular": rows,
        # foxnose does not model overlap: exclusive == regular by construction.
        "exclusive": rows,
        "num_speakers": result.num_speakers,
        "estimation_details": estimation_details,
    }
    atomic_json(output, payload)
    _write_rttm(rttm, payload["regular"])
    return payload
