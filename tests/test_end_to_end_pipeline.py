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

import pytest

from transcribe_meeting import asr, core, diarization
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

    def fake_diarize(_audio, output, rttm, *_args, **kwargs):
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
        rttm.write_text("SPEAKER meeting 1 0.000 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
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


@pytest.fixture
def completed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    source = tmp_path / "source.wav"
    _write_silence_wav(source)
    calls = {"decode": 0, "concat": 0, "asr": 0, "diarize": 0, "overlap": 0}
    _install_stub_backends(monkeypatch, calls)
    job = create_job([source], "Checkpoint test", _build_config(), [{"duration": 2.0}])
    atomic_json(job / "diagnostics.json", {"parts": [{"path": str(source), "stream_index": 0}]})
    run_job(job)
    return job, source, calls


def test_preprocessing_change_redecodes_existing_pcm(completed_job):
    job, _, calls = completed_job
    manifest = read_json(job / "manifest.json")
    manifest["config"]["asr_preprocess"] = "hpf80-preemphasis"
    atomic_json(job / "manifest.json", manifest)

    run_job(job)

    assert calls["decode"] == 2
    assert calls["asr"] == 2
    assert calls["diarize"] == 2


def test_redecode_refuses_a_replaced_source(completed_job):
    job, source, calls = completed_job
    source.write_bytes(b"a different recording")
    (job / "work" / "pcm" / "001.wav").unlink()

    with pytest.raises(RuntimeError, match="source audio has changed"):
        run_job(job)

    assert calls["decode"] == 1


def test_changing_one_selected_stream_only_redecodes_that_part(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))
    sources = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for source in sources:
        _write_silence_wav(source)
    calls = {"decode": 0, "concat": 0, "asr": 0, "diarize": 0, "overlap": 0}
    _install_stub_backends(monkeypatch, calls)
    job = create_job(sources, "Per-part test", _build_config(), [{"duration": 2.0}] * 2)
    diagnostics_path = job / "diagnostics.json"
    atomic_json(
        diagnostics_path,
        {"parts": [{"path": str(source), "stream_index": 0} for source in sources]},
    )
    run_job(job)
    diagnostics = read_json(diagnostics_path)
    diagnostics["parts"][1]["stream_index"] = 1
    atomic_json(diagnostics_path, diagnostics)

    run_job(job)

    assert calls == {"decode": 3, "concat": 2, "asr": 3, "diarize": 2, "overlap": 2}


def test_vad_revision_invalidates_completed_local_asr(completed_job, monkeypatch):
    job, _, calls = completed_job
    monkeypatch.setattr(core, "model_lock", lambda: {"models": {"vad": {"revision": "new-rev"}}})

    run_job(job)

    assert calls["decode"] == 1
    assert calls["asr"] == 2


def test_corrupt_asr_part_is_recreated_with_stage_marker_present(completed_job):
    job, _, calls = completed_job
    path = job / "asr" / "v3_rnnt" / "001.json"
    previous = read_json(path)
    atomic_json(path, {})
    run_job(job)
    assert read_json(path) == previous
    assert calls["asr"] == 2


def test_missing_asr_part_is_recreated_with_stage_marker_present(completed_job):
    job, _, calls = completed_job
    part = job / "asr" / "v3_rnnt" / "001.json"
    part.unlink()

    run_job(job)

    assert part.is_file()
    assert calls["asr"] == 2


def test_unrelated_settings_and_new_model_do_not_retranscribe_existing_model(completed_job):
    job, _, calls = completed_job
    manifest = read_json(job / "manifest.json")
    manifest["config"]["llm_model"] = "new-readable-model"
    manifest["config"]["asr_models"].append("v3_ctc")
    atomic_json(job / "manifest.json", manifest)
    first_model = job / "asr" / "v3_rnnt" / "001.json"
    original_mtime = first_model.stat().st_mtime_ns

    run_job(job)

    assert calls["asr"] == 2
    assert first_model.stat().st_mtime_ns == original_mtime
    assert calls["diarize"] == 1
    assert "v3_ctc" in read_json(job / "canonical.json")["utterances"][0]["variants"]


def test_groq_chunks_are_reused_only_for_the_same_request(completed_job, monkeypatch):
    job, _, _ = completed_job
    original_transcribe = core.transcribe_part

    def dispatch(_audio, _output, *_args, **kwargs):
        transcribe = asr.transcribe_part if kwargs["provider"] == "groq" else original_transcribe
        return transcribe(_audio, _output, *_args, **kwargs)

    monkeypatch.setattr(core, "transcribe_part", dispatch)
    monkeypatch.setattr(asr.subprocess, "run", lambda args, **_: _fake_chunk(args[-1]))
    uploads = []

    def fake_groq(_audio, language, prompt, _limiter):
        uploads.append((language, prompt))
        return {"text": language, "words": [{"word": language, "start": 0.1, "end": 0.5}]}

    monkeypatch.setattr(asr, "_request_groq", fake_groq)
    manifest = read_json(job / "manifest.json")
    manifest["config"].update(asr_models=[asr.GROQ_ASR_MODEL_NAME], asr_provider="groq")
    atomic_json(job / "manifest.json", manifest)
    run_job(job)

    (job / "asr" / asr.GROQ_ASR_MODEL_NAME / "001.json").unlink()
    run_job(job)
    assert uploads == [("ru", "")]

    manifest = read_json(job / "manifest.json")
    manifest["config"]["language"] = "en"
    atomic_json(job / "manifest.json", manifest)
    run_job(job)
    assert uploads == [("ru", ""), ("en", "")]
    assert read_json(job / "canonical.json")["utterances"][0]["raw_text"] == "en"

    manifest = read_json(job / "manifest.json")
    manifest["config"]["asr_models"].append("v3_rnnt")
    manifest["config"]["llm_model"] = "another-readable-model"
    atomic_json(job / "manifest.json", manifest)
    monkeypatch.setattr(core, "model_lock", lambda: {"models": {"vad": {"revision": "new"}}})
    groq_part = job / "asr" / asr.GROQ_ASR_MODEL_NAME / "001.json"
    cached_mtime = groq_part.stat().st_mtime_ns
    run_job(job)
    assert groq_part.stat().st_mtime_ns == cached_mtime
    assert uploads == [("ru", ""), ("en", "")]


def _fake_chunk(path):
    from subprocess import CompletedProcess

    _write_silence_wav(Path(path))
    return CompletedProcess([], 0)


def test_deepgram_meeting_cache_is_shared_and_language_scoped(completed_job, monkeypatch):
    job, _, _ = completed_job
    monkeypatch.setenv("DEEPGRAM_API_KEY", "mock-key")
    monkeypatch.setattr(core, "diarize", diarization.diarize)
    uploads = []

    def fake_request(_audio, _key, language):
        uploads.append(language)
        words = [{"word": language, "start": 0.1, "end": 0.5, "speaker": 0}]
        return {"results": {"channels": [{"alternatives": [{"words": words}]}]}}

    monkeypatch.setattr(asr, "_deepgram_request", fake_request)
    manifest = read_json(job / "manifest.json")
    manifest["config"].update(
        asr_models=[asr.DEEPGRAM_ASR_MODEL_NAME],
        asr_provider="deepgram",
        diarization_provider="deepgram",
    )
    atomic_json(job / "manifest.json", manifest)
    run_job(job)
    assert uploads == ["ru"]

    (job / "asr" / asr.DEEPGRAM_ASR_MODEL_NAME / "001.json").unlink()
    (job / "diarization" / "diarization.rttm").unlink()
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    run_job(job)
    assert uploads == ["ru"]

    monkeypatch.setenv("DEEPGRAM_API_KEY", "mock-key")
    manifest = read_json(job / "manifest.json")
    manifest["config"]["language"] = "en"
    atomic_json(job / "manifest.json", manifest)
    run_job(job)
    assert uploads == ["ru", "en"]
    assert read_json(job / "canonical.json")["utterances"][0]["raw_text"] == "en"


def test_deepgram_missing_raw_recovery_refreshes_existing_slices(completed_job, monkeypatch):
    job, _, _ = completed_job
    monkeypatch.setenv("DEEPGRAM_API_KEY", "mock-key")
    monkeypatch.setattr(core, "diarize", diarization.diarize)
    responses = iter(["old", "new"])

    def fake_request(*_args):
        words = [{"word": next(responses), "start": 0.1, "end": 0.5, "speaker": 0}]
        return {"results": {"channels": [{"alternatives": [{"words": words}]}]}}

    monkeypatch.setattr(asr, "_deepgram_request", fake_request)
    manifest = read_json(job / "manifest.json")
    manifest["config"].update(
        asr_models=[asr.DEEPGRAM_ASR_MODEL_NAME],
        asr_provider="deepgram",
        diarization_provider="deepgram",
    )
    atomic_json(job / "manifest.json", manifest)
    run_job(job)
    (job / "asr" / asr.DEEPGRAM_ASR_MODEL_NAME / "meeting.json").unlink()
    (job / "diarization" / "diarization.rttm").unlink()

    run_job(job)

    assert read_json(job / "canonical.json")["utterances"][0]["raw_text"] == "new"


def test_missing_speaker_map_preserves_manual_edits(completed_job):
    job, _, _ = completed_job
    canonical = read_json(job / "canonical.json")
    first = canonical["utterances"][0]
    first["raw_text"] = "ручное исправление"
    first["speaker_name"] = "Анна"
    atomic_json(job / "canonical.json", canonical)
    (job / "speakers.yaml").unlink()

    run_job(job)

    assert read_json(job / "canonical.json") == canonical
    assert "Анна" in (job / "speakers.yaml").read_text(encoding="utf-8")


def test_missing_canonical_is_recreated_with_align_marker_present(completed_job):
    job, _, calls = completed_job
    expected = read_json(job / "canonical.json")
    (job / "canonical.json").unlink()

    run_job(job)

    assert read_json(job / "canonical.json") == expected
    assert calls["asr"] == 1


def test_legacy_part_without_provenance_is_rebuilt(completed_job):
    job, _, calls = completed_job
    part = job / "asr" / "v3_rnnt" / "001.json"
    atomic_json(part, {"text": "stale", "words": []})
    (job / ".stages" / "asr" / "v3_rnnt" / "001.json").unlink()

    run_job(job)

    assert calls["asr"] == 2
    assert read_json(part)["text"] == "привет мир"


def test_resuming_completed_legacy_job_does_not_overwrite_accepted_transcript(completed_job):
    job, source, calls = completed_job
    canonical = read_json(job / "canonical.json")
    canonical["utterances"][0]["raw_text"] = "Manually corrected."
    atomic_json(job / "canonical.json", canonical)
    atomic_json(job / ".stages" / "align.json", {"fingerprint": {"word_timestamp_schema": 3}})
    write_job_result(job)
    source.unlink()

    run_job(job)

    assert read_json(job / "canonical.json") == canonical
    assert calls["asr"] == 1


def test_readable_is_not_claimed_complete_after_alignment_text_changes(completed_job):
    job, _, calls = completed_job
    manifest = read_json(job / "manifest.json")
    manifest["config"]["clean"].append("readable")
    manifest["readable"] = {"status": "complete"}
    manifest["glossary"] = [{"canonical": "здравствуй", "variants": ["привет"]}]
    atomic_json(job / "manifest.json", manifest)
    canonical = read_json(job / "canonical.json")
    canonical["utterances"][0]["readable_text"] = "Привет, мир!"
    atomic_json(job / "canonical.json", canonical)
    (job / "transcript.readable.md").write_text("former readable", encoding="utf-8")

    run_job(job)

    assert read_json(job / "manifest.json").get("readable", {}).get("status") != "complete"
    assert not (job / "transcript.readable.md").exists()
    assert "здравствуй" in read_json(job / "canonical.json")["utterances"][0]["raw_text"]
    assert calls["asr"] == 1


def test_rebuilding_same_alignment_preserves_readable(completed_job):
    job, _, _ = completed_job
    manifest = read_json(job / "manifest.json")
    manifest["config"]["clean"].append("readable")
    manifest["readable"] = {"status": "complete"}
    atomic_json(job / "manifest.json", manifest)
    canonical = read_json(job / "canonical.json")
    canonical["utterances"][0]["readable_text"] = "Привет, мир!"
    atomic_json(job / "canonical.json", canonical)
    (job / ".stages" / "align.json").unlink()

    run_job(job)

    assert read_json(job / "canonical.json")["utterances"][0]["readable_text"] == "Привет, мир!"
    assert (job / "transcript.readable.md").is_file()


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

    run_job(job)

    assert calls == {"decode": 2, "concat": 2, "asr": 1, "diarize": 1, "overlap": 1}


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
