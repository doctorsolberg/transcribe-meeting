from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .util import atomic_json


class AudioError(RuntimeError):
    pass


ASR_PREPROCESS_MODES = ("none", "hpf80-preemphasis")
ASR_COMPRESSION_MODES = ("none", "4:1")
PRE_EMPHASIS_COEFFICIENT = 0.97
COMPRESSION_4_TO_1 = {
    "threshold_dbfs": -20,
    "threshold_linear": 0.1,
    "ratio": 4,
    "attack_ms": 20,
    "release_ms": 250,
    "makeup_gain_db": 0,
}


def wav_duration(path: Path) -> float:
    """Return the duration of the decoded audio, not the container's declared duration."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def load_model_waveform(path: Path) -> dict[str, Any]:
    """Read a working PCM WAV as channel-first float32, without torchcodec."""
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # The transpose is a view; the usual mono WAV needs no additional PCM buffer.
    return {"waveform": torch.from_numpy(samples.T), "sample_rate": sample_rate}


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def probe(path: Path, audio_stream: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    result = run_command(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    )
    if result.returncode:
        raise AudioError(f"FFprobe could not read {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    audio = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if len(audio) != 1 and audio_stream is None:
        raise AudioError(f"{path.name}: found {len(audio)} audio streams; specify --audio-stream.")
    if audio_stream is not None:
        try:
            stream = next(item for item in audio if item.get("index") == audio_stream)
        except StopIteration as exc:
            raise AudioError(f"{path.name}: audio stream #{audio_stream} is missing.") from exc
    else:
        stream = audio[0]
    return {
        "path": str(path),
        "format": data.get("format", {}).get("format_name"),
        "duration": float(data.get("format", {}).get("duration", 0)),
        "codec": stream.get("codec_name"),
        "channels": stream.get("channels"),
        "sample_rate": stream.get("sample_rate"),
        "stream_index": stream.get("index"),
        "decode_errors": [],
    }


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise AudioError("ffmpeg and ffprobe are required (e.g.: brew install ffmpeg).")


def measure_loudness(path: Path) -> dict[str, Any]:
    """Measure EBU R128 loudness with a full decode pass.

    The level must be `info`: loudnorm prints its JSON summary there, so a quieter
    ffmpeg silently yields no measurement at all.
    """
    measure = run_command(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    text = measure.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"warning": "measurement unavailable"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"warning": "measurement unavailable"}


def diagnostics(
    paths: list[Path],
    output: Path,
    stream: int | None = None,
    with_loudness: bool = False,
) -> list[dict[str, Any]]:
    """Probe every part; loudness costs a full decode pass, so it is opt-in.

    Measurements are advisory either way: no filter here ever affects decoding.
    """
    check_ffmpeg()
    values = [probe(path, stream) for path in paths]
    for value, path in zip(values, paths, strict=True):
        if with_loudness:
            value["loudness"] = measure_loudness(path)
        value["warnings"] = []
        if value["duration"] <= 0:
            value["warnings"].append("zero duration")
    atomic_json(output, {"parts": values, "processing": "none", "loudness_measured": with_loudness})
    return values


def pre_emphasize_pcm(
    source: Path, target: Path, coefficient: float = PRE_EMPHASIS_COEFFICIENT
) -> int:
    """Apply pre-emphasis in bounded NumPy blocks, preserving the prior sample between blocks."""
    with wave.open(str(source), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise AudioError("Pre-emphasis expects a mono signed-16-bit PCM WAV.")
        params = reader.getparams()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        clipped = 0
        previous = 0.0
        try:
            with wave.open(str(temporary), "wb") as writer:
                writer.setparams(params)
                while frames := reader.readframes(1_000_000):
                    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
                    prior = np.empty_like(samples)
                    prior[0] = previous
                    prior[1:] = samples[:-1]
                    emphasized = np.rint(samples - coefficient * prior)
                    clipped += int(np.count_nonzero((emphasized > 32767) | (emphasized < -32768)))
                    writer.writeframes(np.clip(emphasized, -32768, 32767).astype("<i2").tobytes())
                    previous = float(samples[-1])
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return clipped


def _ffmpeg_pcm(
    input_path: Path,
    output_path: Path,
    audio_filter: str | None = None,
    ffmpeg_threads: int = 1,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-threads",
        str(ffmpeg_threads),
        "-filter_threads",
        str(ffmpeg_threads),
        "-i",
        str(input_path),
    ]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)])
    result = run_command(command)
    if result.returncode:
        raise AudioError(f"FFmpeg could not process the PCM: {result.stderr.strip()}")


def _compress_pcm(source: Path, target: Path, ffmpeg_threads: int) -> None:
    settings = COMPRESSION_4_TO_1
    audio_filter = (
        f"acompressor=threshold={settings['threshold_linear']}:ratio={settings['ratio']}:"
        f"attack={settings['attack_ms']}:release={settings['release_ms']}:makeup=1"
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        _ffmpeg_pcm(source, temporary, audio_filter, ffmpeg_threads)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def decode_pcm(
    source: Path,
    target: Path,
    stream_index: int,
    asr_preprocess: str = "none",
    asr_compression: str = "none",
    ffmpeg_threads: int = 1,
) -> dict[str, Any]:
    if asr_preprocess not in ASR_PREPROCESS_MODES:
        raise AudioError(f"Unknown preprocessing mode: {asr_preprocess}")
    if asr_compression not in ASR_COMPRESSION_MODES:
        raise AudioError(f"Unknown compression mode: {asr_compression}")
    if ffmpeg_threads < 1:
        raise AudioError("The number of FFmpeg threads must be at least 1")
    target.parent.mkdir(parents=True, exist_ok=True)
    decoded = target.with_suffix(".decode.wav")
    pre_emphasized = target.with_suffix(".preemphasis.wav")
    filters = ["highpass=f=80"] if asr_preprocess == "hpf80-preemphasis" else []
    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-threads",
            str(ffmpeg_threads),
            "-filter_threads",
            str(ffmpeg_threads),
            "-i",
            str(source),
            "-map",
            f"0:{stream_index}",
            "-ac",
            "1",
            "-ar",
            "16000",
            *(["-af", ",".join(filters)] if filters else []),
            "-c:a",
            "pcm_s16le",
            str(decoded),
        ]
    )
    if result.returncode:
        raise AudioError(f"Could not decode {source.name}: {result.stderr.strip()}")
    current = decoded
    report: dict[str, Any] = {"mode": asr_preprocess, "filters": filters}
    try:
        if asr_preprocess == "hpf80-preemphasis":
            clipped = pre_emphasize_pcm(current, pre_emphasized)
            decoded.unlink(missing_ok=True)
            current = pre_emphasized
            report.update(
                {
                    "filters": [
                        "highpass=f=80",
                        f"pre_emphasis=y[n]-{PRE_EMPHASIS_COEFFICIENT}*y[n-1]",
                    ],
                    "highpass_hz": 80,
                    "pre_emphasis_coefficient": PRE_EMPHASIS_COEFFICIENT,
                    "pre_emphasis_clipped_samples": clipped,
                }
            )
        if asr_compression == "4:1":
            _compress_pcm(current, target, ffmpeg_threads)
            report["filters"].append(
                "acompressor=threshold=0.1:ratio=4:attack=20:release=250:makeup=1"
            )
            report["compression"] = {"mode": "4:1", **COMPRESSION_4_TO_1}
        else:
            os.replace(current, target)
    finally:
        decoded.unlink(missing_ok=True)
        pre_emphasized.unlink(missing_ok=True)
    return report


def concat_pcm(parts: list[Path], target: Path, ffmpeg_threads: int = 1) -> None:
    """Concatenate PCM WAVs and insert exactly one second silence between files."""
    if not parts:
        raise AudioError("No audio to concatenate.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # concat filter keeps this entirely local and emits a technical-only WAV.
    inputs = [value for part in parts for value in ("-i", str(part))]
    filters: list[str] = []
    labels: list[str] = []
    for index in range(len(parts)):
        labels.append(f"[{index}:a]")
        if index < len(parts) - 1:
            filters.append(f"anullsrc=r=16000:cl=mono:d=1[s{index}]")
            labels.append(f"[s{index}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]")
    result = run_command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-threads",
            str(ffmpeg_threads),
            "-filter_threads",
            str(ffmpeg_threads),
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    if result.returncode:
        raise AudioError(f"Could not assemble the technical track: {result.stderr.strip()}")
