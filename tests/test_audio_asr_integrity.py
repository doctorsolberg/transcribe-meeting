import shutil
import subprocess
import wave
from types import SimpleNamespace
from typing import Any

import pytest

from transcribe_meeting import asr, audio, core, diarization
from transcribe_meeting.cli import _resolved_run_config
from transcribe_meeting.timeline import build_sources, mapping
from transcribe_meeting.util import atomic_json, read_json, sha256


def _wav(path, frames=32_000, sample=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16_000, 0, "NONE", ""))
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * frames)


def _audio_job(job, sources, diagnostics):
    manifest = {
        "sources": mapping(build_sources([(str(path), sha256(path), 5.0) for path in sources])),
        "config": {"asr_preprocess": "none", "asr_compression": "none"},
    }
    atomic_json(job / "manifest.json", manifest)
    atomic_json(job / "diagnostics.json", {"parts": diagnostics})
    return manifest


def _short_audio_video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=16x16:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:duration=1",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
def test_diagnostics_resolve_relative_input_outside_data_root(tmp_path, monkeypatch, legacy):
    source = tmp_path / "invocation" / "media" / "speech.wav"
    _wav(source)
    monkeypatch.chdir(source.parent.parent)
    diagnostic = audio.probe(source.relative_to(source.parent.parent))
    if legacy:
        diagnostic["path"] = "media/speech.wav"
    else:
        assert diagnostic["path"] == str(source)
    monkeypatch.chdir(tmp_path)
    job = tmp_path / "data" / "job"
    monkeypatch.setenv("TRANSCRIBE_HOME", str(job.parent))
    manifest = _audio_job(job, [source], [diagnostic])
    assert core._pcm_fingerprints(job, manifest)[0]["stream"] == 0


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
def test_legacy_positional_match_rejects_unverified_stream(tmp_path):
    source = tmp_path / "origin" / "media" / "speech.wav"
    _wav(source)
    job = tmp_path / "job"
    diagnostics = {**audio.probe(source), "path": "media/speech.wav", "stream_index": 1}
    manifest = _audio_job(job, [source], [diagnostics])
    with pytest.raises(RuntimeError, match="Cannot verify") as error:
        core._pcm_fingerprints(job, manifest)
    assert "stream" in str(error.value.__cause__).lower()


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
def test_legacy_diagnostics_reject_crossed_rows_even_with_identical_streams(tmp_path):
    first = tmp_path / "invocation" / "media" / "first.wav"
    second = first.with_name("second.wav")
    _wav(first)
    _wav(second)
    rows = [
        {**audio.probe(second), "path": "../media/second.wav"},
        {**audio.probe(first), "path": "../media/first.wav"},
    ]
    job = tmp_path / "job"
    manifest = _audio_job(job, [first, second], rows)
    with pytest.raises(RuntimeError, match="Cannot verify"):
        core._pcm_fingerprints(job, manifest)


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
def test_verified_legacy_diagnostics_do_not_need_original_on_later_reuse(tmp_path):
    source = tmp_path / "origin" / "speech.wav"
    _wav(source)
    job = tmp_path / "job"
    diagnostics = {**audio.probe(source), "path": "speech.wav"}
    manifest = _audio_job(job, [source], [diagnostics])
    fingerprint = core._pcm_fingerprints(job, manifest)
    source.unlink()
    assert core._pcm_fingerprints(job, manifest) == fingerprint


@pytest.mark.parametrize("working_file", ["pcm/001.wav", "meeting.wav"])
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_corrupt_working_pcm_rebuilt_despite_same_length(tmp_path, working_file):
    source = tmp_path / "speech.wav"
    _wav(source)
    job = tmp_path / "job"
    manifest = _audio_job(job, [source], [audio.probe(source)])
    before = core._stage_audio(job, manifest, resume=True, ffmpeg_threads=1, decode_workers=1)
    target = job / "work" / working_file
    original = target.read_bytes()
    target.write_bytes(original[:-4] + b"\x11\x11\x11\x11")
    after = core._stage_audio(job, manifest, resume=True, ffmpeg_threads=1, decode_workers=1)
    assert target.read_bytes() == original
    assert before == after


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_decoded_audio_sets_multipart_clocks(tmp_path):
    video, second = tmp_path / "video.mp4", tmp_path / "second.wav"
    _short_audio_video(video)
    _wav(second)
    job = tmp_path / "job"
    manifest = _audio_job(job, [video, second], [audio.probe(video), audio.probe(second)])
    assert manifest["sources"][1]["diarization_offset"] == 6.0
    core._stage_audio(job, manifest, resume=True, ffmpeg_threads=1, decode_workers=1)
    core._refresh_decoded_timeline(job, manifest)
    with wave.open(str(job / "work" / "meeting.wav"), "rb") as handle:
        actual_duration = handle.getnframes() / handle.getframerate()
    second_source = manifest["sources"][1]
    assert second_source["offset"] == pytest.approx(actual_duration - 3.0, abs=1 / 16_000)
    assert second_source["diarization_offset"] == pytest.approx(
        actual_duration - 2.0, abs=1 / 16_000
    )
    assert read_json(job / "manifest.json")["sources"] == manifest["sources"]


@pytest.mark.parametrize("model", ["v3_rnnt", asr.DEEPGRAM_ASR_MODEL_NAME])
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_multipart_video_attributes_second_part_with_decoded_clock(tmp_path, monkeypatch, model):
    video, wav = tmp_path / "video.mp4", tmp_path / "second.wav"
    _short_audio_video(video)
    _wav(wav)
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path / "data"))
    network = model == asr.DEEPGRAM_ASR_MODEL_NAME
    config = _resolved_run_config(
        language="ru",
        audio_stream=None,
        speakers=None,
        min_speakers=1,
        max_speakers=3,
        clean=["raw"],
        profile_name="background",
        asr_device="cpu",
        diarization_provider="deepgram" if network else "foxnose",
        cpu_threads=1,
        ffmpeg_threads=1,
        chunk_minutes=None,
        interactive_names=False,
        resume=True,
        keep_working_audio=True,
        asr_preprocess="none",
        asr_compression="none",
        context_prompt="",
        llm_model="mock",
        asr_models=[model],
    )
    sources = [video, wav]
    probes = [audio.probe(path) for path in sources]
    job = core.create_job(sources, "clock", config, probes)
    atomic_json(job / "diagnostics.json", {"parts": probes})

    # Derive the mock providers' true clock from the meeting audio, independently
    # from the manifest under test. A wrong manifest would shift/drop the words.
    def meeting_offset():
        return audio.wav_duration(job / "work" / "meeting.wav") - 2.0

    def deepgram(*_):
        second = meeting_offset()
        return _deepgram_response(
            [_word("first", speaker=0), _word("second", second + 0.1, second + 0.8, speaker=1)]
        )

    def local(_wav, output, *_args, **_kwargs):
        atomic_json(output, {"words": [_word("first" if output.stem == "001" else "second")]})

    def diarize(_wav, output, rttm, *_args, **_kwargs):
        second = meeting_offset()
        rows = [
            {"start": 0.0, "end": 0.9, "speaker": "SPEAKER_00"},
            {"start": second, "end": second + 0.9, "speaker": "SPEAKER_01"},
        ]
        atomic_json(output, {"provider": "foxnose", "regular": rows, "exclusive": rows})
        diarization._write_rttm(rttm, rows)

    monkeypatch.setattr(core, "detect_overlap_regions", lambda *_: [])
    if network:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-key")
        monkeypatch.setattr(asr, "_deepgram_request", deepgram)
    else:
        monkeypatch.setattr(asr, "effective_asr_device", lambda *_: "cpu")
        monkeypatch.setattr(core, "transcribe_part", local)
        monkeypatch.setattr(core, "diarize", diarize)
    core.run_job(job)
    first, second = read_json(job / "canonical.json")["utterances"]
    assert [
        (first["raw_text"], first["speaker_id"]),
        (second["raw_text"], second["speaker_id"]),
    ] == [
        ("first", "SPEAKER_00"),
        ("second", "SPEAKER_01"),
    ]
    assert second["source_file"] == str(wav)
    assert second["meeting_start"] == pytest.approx(meeting_offset() - 1.0 + 0.1)
    # Neither preserving native word labels nor regenerating a derivative RTTM
    # should send the same validated Deepgram request again.
    if network:
        rttm = job / "diarization" / "diarization.rttm"
        original = rttm.read_bytes()
        rttm.write_bytes(b"X" * len(original))
        monkeypatch.delenv("DEEPGRAM_API_KEY")
        core.run_job(job)
        assert rttm.read_bytes() == original


def _word(text="spoken", start=0.1, end=0.8, **extra):
    return {"word": text, "start": start, "end": end, **extra}


def _deepgram_response(words) -> dict[str, Any]:
    return {"results": {"channels": [{"alternatives": [{"words": words}]}]}}


def test_deepgram_reuses_valid_response_after_marker_interruption(tmp_path, monkeypatch):
    meeting = tmp_path / "work" / "meeting.wav"
    _wav(meeting)
    config = {"language": "ru"}
    fp = {"pcm_sha256": sha256(meeting)}
    response = _deepgram_response([_word(speaker=0)])
    requests = []

    def request(*args):
        requests.append(args)
        return response

    def interrupted(*args):
        raise KeyboardInterrupt

    real_complete = core.complete_stage
    monkeypatch.setattr(asr, "_deepgram_request", request)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-key")
    monkeypatch.setattr(core, "complete_stage", interrupted)
    with pytest.raises(KeyboardInterrupt):
        core._deepgram_checkpoint(tmp_path, config, fp)
    monkeypatch.setattr(core, "complete_stage", real_complete)
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert core._deepgram_checkpoint(tmp_path, config, fp) == response
    assert len(requests) == 1


def test_deepgram_rejects_null_words_before_publishing_checkpoint(tmp_path, monkeypatch):
    meeting = tmp_path / "work" / "meeting.wav"
    _wav(meeting)
    responses = iter([_deepgram_response(None), _deepgram_response([])])
    monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-key")
    monkeypatch.setattr(asr, "_deepgram_request", lambda *_: next(responses))
    config = {"language": "ru"}
    fp = {"pcm_sha256": sha256(meeting)}
    with pytest.raises(RuntimeError, match=r"Deepgram.*words"):
        core._deepgram_checkpoint(tmp_path, config, fp)
    assert not (tmp_path / "asr" / asr.DEEPGRAM_ASR_MODEL_NAME / "meeting.json").exists()
    assert asr.deepgram_words(core._deepgram_checkpoint(tmp_path, config, fp)) == []


def test_deepgram_does_not_treat_text_without_word_timings_as_silence():
    payload = _deepgram_response([])
    payload["results"]["channels"][0]["alternatives"][0]["transcript"] = "spoken"
    with pytest.raises(RuntimeError, match=r"Deepgram.*words"):
        asr.deepgram_words(payload)


def test_deepgram_does_not_cache_out_of_meeting_words_as_silence(tmp_path, monkeypatch):
    meeting = tmp_path / "work" / "meeting.wav"
    _wav(meeting)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        asr, "_deepgram_request", lambda *_: _deepgram_response([_word(start=10, end=11)])
    )
    with pytest.raises(RuntimeError, match=r"Deepgram.*words"):
        core._deepgram_checkpoint(tmp_path, {"language": "ru"}, {"pcm_sha256": sha256(meeting)})
    assert not (tmp_path / "asr" / asr.DEEPGRAM_ASR_MODEL_NAME / "meeting.json").exists()


def test_deepgram_cache_rejects_changed_request_provenance(tmp_path, monkeypatch):
    meeting = tmp_path / "work" / "meeting.wav"
    _wav(meeting)
    responses = iter([_deepgram_response([_word("ru")]), _deepgram_response([_word("en")])])
    monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-key")
    monkeypatch.setattr(asr, "_deepgram_request", lambda *_: next(responses))
    fp = {"pcm_sha256": sha256(meeting)}
    core._deepgram_checkpoint(tmp_path, {"language": "ru"}, fp)
    checkpoint = tmp_path / "asr" / asr.DEEPGRAM_ASR_MODEL_NAME / "meeting.json"
    data = read_json(checkpoint)
    data["_checkpoint"]["parameters"]["language"] = "en"
    atomic_json(checkpoint, data)
    result = core._deepgram_checkpoint(tmp_path, {"language": "en"}, fp)
    assert asr.deepgram_words(result)[0]["word"] == "en"


def test_deepgram_preserves_native_speaker_labels_after_slicing():
    words = [_word(speaker=0), _word("nested", 0.3, 0.4, speaker=1)]
    sliced = asr.deepgram_words_for_source(words, 0.0, 2.0)
    decoded = asr.words_from_asr({"words": sliced}, native_speakers=True)
    assert [word["speaker_id"] for word in decoded] == ["SPEAKER_00", "SPEAKER_01"]


@pytest.mark.parametrize("product", ["chunk", "plan"])
def test_local_checkpoint_corruption_is_not_silence(tmp_path, monkeypatch, product):
    source = tmp_path / "speech.wav"
    _wav(source)
    output = tmp_path / "asr" / "v3_rnnt" / "001.json"
    calls = {"asr": 0, "vad": 0}

    class Local:
        device = "cpu"

        def transcribe(self, _source):
            calls["asr"] += 1
            return {"words": [_word()], "segments": [], "text": "spoken"}

    def detect(*_args):
        calls["vad"] += 1
        return [(0.0, 2.0)]

    monkeypatch.setattr(asr, "detect_speech_regions", detect)
    options = {
        "backend": Local(),
        "vad_model_path": tmp_path,
        "fingerprint": {},
        "model_name": "v3_rnnt",
    }
    assert asr._local_part(source, output, **options)["text"] == "spoken"
    if product == "chunk":
        target = output.parent / "chunks" / "001" / "001.json"
        data = read_json(target)
        data.pop("words")
    else:
        target = output.parent.parent / "vad" / "001.json"
        data = read_json(target)
        data["chunks"] = []
    atomic_json(target, data)
    output.unlink()
    assert asr._local_part(source, output, **options)["text"] == "spoken"
    expected = {"asr": 2, "vad": 1} if product == "chunk" else {"asr": 1, "vad": 2}
    assert calls == expected


def test_local_seam_keeps_word_when_each_copy_moves_into_other_chunk(tmp_path, monkeypatch):
    source = tmp_path / "speech.wav"
    _wav(source, 24 * 16_000)
    words = iter([_word("seam", 21.99, 22.21), _word("seam", 0.28, 0.56)])

    class Backend:
        device = "cpu"

        def transcribe(self, audio):
            return {"words": [next(words)], "text": "seam", "segments": []}

        def close(self):
            pass

    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [(0.0, 24.0)])
    output = tmp_path / "asr" / "model" / "001.json"
    payload = asr._local_part(
        source,
        output,
        backend=Backend(),
        vad_model_path=tmp_path,
        fingerprint={},
        model_name="v3_rnnt",
    )
    assert len(payload["chunks"]) == 2
    assert [word["word"] for word in payload["words"]] == ["seam"]


@pytest.fixture
def groq_responses(monkeypatch):
    import openai

    calls = []

    def install(responses):
        items = iter(responses)

        def transcribe(**options):
            with wave.open(options["file"][1], "rb") as handle:
                calls.append(handle.getnframes())
            return next(items)

        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe))
        )
        monkeypatch.setenv("GROQ_API_KEY", "synthetic-key")
        monkeypatch.setattr(openai, "OpenAI", lambda **_: client)
        return calls

    return install


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_groq_reuses_response_after_interruption_before_part_output(
    tmp_path, monkeypatch, groq_responses
):
    source = tmp_path / "speech.wav"
    output = tmp_path / "asr" / "groq" / "001.json"
    _wav(source)
    calls = groq_responses([{"words": [_word()], "segments": [], "text": "spoken"}])
    original = asr.atomic_json

    def interrupt(path, data):
        if path == output:
            raise KeyboardInterrupt
        return original(path, data)

    def transcribe():
        return asr.transcribe_part(source, output, None, "ru", "", provider="groq")

    with monkeypatch.context() as context:
        context.setattr(asr, "atomic_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            transcribe()
    monkeypatch.delenv("GROQ_API_KEY")
    assert transcribe()["words"] == [_word()]
    assert calls == [32_000]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_groq_corrupt_chunk_is_not_silence(tmp_path, groq_responses):
    source = tmp_path / "speech.wav"
    output = tmp_path / "asr" / "groq" / "001.json"
    _wav(source)
    calls = groq_responses([{"words": [_word()], "segments": [], "text": "spoken"}] * 2)
    assert asr.transcribe_part(source, output, None, "ru", "", provider="groq")["words"]
    chunk = output.parent / "chunks" / "001" / "001.json"
    edited = read_json(chunk)
    del edited["words"]
    atomic_json(chunk, edited)
    assert asr.transcribe_part(source, output, None, "ru", "", provider="groq")["words"]
    assert calls == [32_000] * 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_groq_preserves_part_with_one_sample_tail(tmp_path, groq_responses):
    source = tmp_path / "speech.wav"
    _wav(source, 60 * 16_000 + 1)
    calls = groq_responses([{"words": [], "segments": [], "text": ""}] * 2)
    output = tmp_path / "asr" / "001.json"
    asr.transcribe_part(source, output, None, "ru", "", provider="groq", chunk_minutes=1)
    assert calls == [60 * 16_000, 1]


@pytest.mark.parametrize("text", ["", "spoken"])
@pytest.mark.parametrize("words", [None, []])
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_groq_nullable_words_only_represent_silence_without_speech(
    tmp_path, groq_responses, text, words
):
    source = tmp_path / "speech.wav"
    _wav(source)
    groq_responses([{"words": words, "segments": None, "text": text}])
    output = tmp_path / "asr" / "001.json"
    if text:
        with pytest.raises(ValueError, match="word timestamps"):
            asr.transcribe_part(source, output, None, "ru", "", provider="groq")
        assert not (output.parent / "chunks" / "001" / "001.json").exists()
    else:
        assert asr.transcribe_part(source, output, None, "ru", "", provider="groq")["words"] == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_groq_rejects_invalid_segment_before_checkpointing(tmp_path, groq_responses):
    source = tmp_path / "speech.wav"
    _wav(source)
    groq_responses(
        [{"words": [_word()], "segments": [{"start": None, "end": 1.0}], "text": "speech"}]
    )
    output = tmp_path / "asr" / "001.json"
    with pytest.raises(ValueError, match=r"Groq response.*segment"):
        asr.transcribe_part(source, output, None, "ru", "", provider="groq")
    assert not (output.parent / "chunks" / "001" / "001.json").exists()
