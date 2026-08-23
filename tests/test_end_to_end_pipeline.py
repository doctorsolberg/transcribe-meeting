"""The one test that runs the whole job pipeline end to end.

Every unit test elsewhere in this suite checks one stage in isolation; none of them
would catch a broken handoff between stages (e.g. a fingerprint or file path stage
N writes that stage N+1 reads differently). This test drives core.create_job and
core.run_job exactly like `run` does, but with decode/ASR/diarization/overlap
replaced by fakes — no ffmpeg, no downloaded model, no network call — so it runs in
CI (see .github/workflows/ci.yml) on a fresh checkout with nothing set up.
"""

from __future__ import annotations

import datetime as dt
import wave
from pathlib import Path

from transcribe_meeting import core
from transcribe_meeting.cli import _resolved_run_config
from transcribe_meeting.core import create_job, run_job
from transcribe_meeting.result import write_job_result
from transcribe_meeting.util import atomic_json, read_json


def _write_silence_wav(path: Path, seconds: float = 2.0, rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


def _install_stub_backends(monkeypatch, calls: dict[str, int]) -> None:
    """Replace every stage boundary that would otherwise touch ffmpeg, a downloaded
    model, or the network. Each fake is installed on the `core` module object,
    exactly where run_job's stages resolve these names from (see core.py's
    `from .audio import concat_pcm, decode_pcm` etc.) — the same pattern
    tests/test_working_audio.py already uses for `core.decode_pcm`.
    """

    def fake_decode(_source, target, _stream, *_args, **_kwargs):
        calls["decode"] += 1
        _write_silence_wav(target)
        return {"mode": "none", "filters": []}

    def fake_concat(_parts, target, *_args, **_kwargs):
        calls["concat"] += 1
        _write_silence_wav(target)

    def fake_transcribe_part(_audio, output, *_args, **kwargs):
        calls["asr"] += 1
        data = {
            "text": "привет мир",
            "segments": [],
            "words": [
                {"word": "привет", "start": 0.1, "end": 0.5, "probability": None},
                {"word": "мир", "start": 0.6, "end": 1.0, "probability": None},
            ],
            "run_parameters": {"provider": kwargs.get("provider", "local")},
        }
        atomic_json(output, data)
        return data

    def fake_diarize(_audio, output, _rttm, *_args, **kwargs):
        calls["diarize"] += 1
        payload = {
            "provider": kwargs.get("provider", "foxnose"),
            "repository": "test/stub",
            "cpu_threads": 1,
            "supports_overlap": False,
            "regular": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
            "exclusive": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        }
        atomic_json(output, payload)
        return payload

    def fake_overlap(*_args, **_kwargs):
        calls["overlap"] += 1
        return []

    monkeypatch.setattr(core, "decode_pcm", fake_decode)
    monkeypatch.setattr(core, "concat_pcm", fake_concat)
    monkeypatch.setattr(core, "transcribe_part", fake_transcribe_part)
    monkeypatch.setattr(core, "diarize", fake_diarize)
    monkeypatch.setattr(core, "detect_overlap_regions", fake_overlap)


def _build_config() -> dict:
    config = _resolved_run_config(
        language="ru",
        audio_stream=None,
        speakers=None,
        min_speakers=1,
        max_speakers=8,
        clean=["raw", "light"],  # readable needs OPENAI_API_KEY/network; excluded on purpose
        profile_name="background",
        cpu_threads=None,
        ffmpeg_threads=None,
        chunk_minutes=None,
        interactive_names=False,
        resume=True,
        keep_working_audio=True,  # so the second run_job call can prove stages are reused
        asr_preprocess="none",
        asr_compression="none",
        context_prompt="",
        llm_model="gpt-5.6-terra",
        asr_device="cpu",
        diarization_provider="foxnose",
        asr_models=["v3_rnnt"],
    )
    config["asr_dropped_models"] = []
    return config


def test_run_job_pipeline_end_to_end_on_stub_backends(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    source = tmp_path / "src" / "meeting.wav"
    _write_silence_wav(source)

    calls = {"decode": 0, "concat": 0, "asr": 0, "diarize": 0, "overlap": 0}
    _install_stub_backends(monkeypatch, calls)

    job = create_job([source], "E2E test", _build_config(), [{"duration": 2.0}])
    atomic_json(job / "diagnostics.json", {"parts": [{"path": str(source), "stream_index": 0}]})

    run_job(job, resume=True)

    assert (job / "canonical.json").exists()
    canonical = read_json(job / "canonical.json")
    assert canonical["utterances"], "align stage must produce at least one utterance"
    assert (job / "transcript.raw.v3_rnnt.md").exists()
    assert (job / "transcript.light.v3_rnnt.md").exists()

    result = write_job_result(job)
    assert (job / "result.json").exists()
    assert (job / "summary.md").exists()
    assert result["status"] == "complete"
    assert result["utterances"] == len(canonical["utterances"])

    calls_after_first_run = dict(calls)
    assert calls_after_first_run == {"decode": 1, "concat": 1, "asr": 1, "diarize": 1, "overlap": 1}

    # Stages must be reused, not redone, on a second call with
    # nothing changed — the exact resume guarantee `resume`/a second `run` rely on.
    run_job(job, resume=True)
    assert calls == calls_after_first_run


def test_run_job_removes_chunk_wavs_but_keeps_json_checkpoints_when_not_keeping_audio(
    tmp_path, monkeypatch
):
    """The stub transcribe_part never exercises _local_part's own chunk files, so this
    plants a WAV+JSON pair exactly where a real local ASR run leaves one and checks
    run_job's cleanup step (config["keep_working_audio"] is False) sweeps only the
    184 MB of sliced audio, never the small checkpoint resume reads."""
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    source = tmp_path / "src" / "meeting.wav"
    _write_silence_wav(source)

    calls = {"decode": 0, "concat": 0, "asr": 0, "diarize": 0, "overlap": 0}
    _install_stub_backends(monkeypatch, calls)

    config = _build_config()
    config["keep_working_audio"] = False
    job = create_job([source], "E2E WAV cleanup", config, [{"duration": 2.0}])
    atomic_json(job / "diagnostics.json", {"parts": [{"path": str(source), "stream_index": 0}]})

    chunk_dir = job / "asr" / "v3_rnnt" / "chunks" / "001"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "001.wav").write_bytes(b"RIFF....")
    (chunk_dir / "001.json").write_text("{}", encoding="utf-8")

    run_job(job, resume=True)

    assert not (chunk_dir / "001.wav").exists()
    assert (chunk_dir / "001.json").exists()
    assert not (job / "work").exists()


def test_create_job_appends_a_suffix_on_a_stamp_and_name_collision(tmp_path, monkeypatch):
    """Two runs in the same second with the same --meeting-name must not collide on
    mkdir(exist_ok=False); a fixed clock forces the collision deterministically."""
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    source = tmp_path / "src" / "meeting.wav"
    _write_silence_wav(source)
    config = _build_config()

    class _FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(core.dt, "datetime", _FixedDatetime)

    first = create_job([source], "Same meeting name", config, [{"duration": 2.0}])
    second = create_job([source], "Same meeting name", config, [{"duration": 2.0}])

    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert second.name == first.name + "-2"
