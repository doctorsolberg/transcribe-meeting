import wave

from transcribe_meeting import asr
from transcribe_meeting.util import read_json


def _wav(path, seconds=30, rate=100):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * rate * seconds)


class FakeBackend:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        duration = asr._wav_duration(audio)
        return {
            "text": "тест",
            "segments": [],
            "words": [{"word": "тест", "start": 0, "end": min(1, duration), "probability": None}],
        }

    def close(self):
        pass


def test_vad_chunk_checkpoints_shift_words_and_resume(tmp_path, monkeypatch):
    audio = tmp_path / "part.wav"
    _wav(audio, 50)
    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [(0, 23), (24, 50)])
    backend = FakeBackend()
    output = tmp_path / "asr" / "001.json"
    kwargs = {
        "local_backend": backend,
        "vad_model_path": tmp_path,
        "local_fingerprint": {"source": "a"},
    }

    asr.transcribe_part(audio, output, None, "ru", "", **kwargs)
    data = read_json(output)
    assert [item["start"] for item in data["words"]] == [0.0, 24.0, 46.0]
    assert backend.calls == 3
    # One directory above asr/<model>/: the plan is shared by every local model.
    assert (output.parent.parent / "vad" / "001.json").exists()
    assert (output.parent / "chunks" / "001" / "001.json").exists()

    asr.transcribe_part(audio, output, None, "ru", "", **kwargs)
    assert backend.calls == 3


def test_vad_splitting_never_exceeds_absolute_limit_and_empty_is_valid(tmp_path, monkeypatch):
    chunks = asr.split_vad_regions([(0, 80)], 80)
    assert chunks and max(end - start for start, end in chunks) <= asr.LOCAL_VAD_ABSOLUTE_SECONDS
    audio = tmp_path / "silence.wav"
    _wav(audio, 2)
    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [])
    output = tmp_path / "asr" / "001.json"
    asr.transcribe_part(
        audio,
        output,
        None,
        "ru",
        "",
        local_backend=FakeBackend(),
        vad_model_path=tmp_path,
        local_fingerprint={},
    )
    assert read_json(output)["words"] == []


def test_absolute_plus_context_padding_stays_under_gigaams_longform_limit():
    """Guards the two constants together: gigaam.model.LONGFORM_THRESHOLD is 25 s, and
    transcribe() raises above it, so a future bump to either constant alone must not
    silently push a padded slice past that ceiling."""
    assert asr.LOCAL_VAD_ABSOLUTE_SECONDS + 2 * asr.LOCAL_VAD_CONTEXT_SECONDS < 25.0


def test_padded_slice_length_never_exceeds_24_5_seconds_on_a_long_recording():
    duration = 1000.0
    chunks = asr.split_vad_regions([(0.0, duration)], duration)
    assert chunks
    for start, end in chunks:
        slice_start = max(0.0, start - asr.LOCAL_VAD_CONTEXT_SECONDS)
        slice_end = min(duration, end + asr.LOCAL_VAD_CONTEXT_SECONDS)
        assert slice_end - slice_start <= 24.5


def test_padded_slice_is_wider_than_the_nominal_chunk_window(tmp_path, monkeypatch):
    """A chunk in the middle of the recording is padded on both sides; a chunk that
    touches a file boundary is padded only on the side that is not that boundary."""
    audio = tmp_path / "part.wav"
    _wav(audio, 30)
    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [])
    monkeypatch.setattr(
        asr, "split_vad_regions", lambda *_a, **_k: [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    )
    slices = []
    real_write = asr._write_wav_slice

    def capture(source, target, start, end):
        slices.append((start, end))
        real_write(source, target, start, end)

    monkeypatch.setattr(asr, "_write_wav_slice", capture)
    output = tmp_path / "asr" / "001.json"

    asr.transcribe_part(
        audio,
        output,
        None,
        "ru",
        "",
        local_backend=FakeBackend(),
        vad_model_path=tmp_path,
        local_fingerprint={},
    )

    assert slices == [(0.0, 10.5), (9.5, 20.5), (19.5, 30.0)]
    assert [chunk["slice_start"] for chunk in read_json(output)["chunks"]] == [0.0, 9.5, 19.5]
    assert [chunk["slice_end"] for chunk in read_json(output)["chunks"]] == [10.5, 20.5, 30.0]


class SeamBackend:
    """Models two independent decodes of the same physical instant around a chunk
    seam: chunk 1's padded slice reaches into chunk 2's nominal window and hears the
    word there too, exactly as real GigaAM would on overlapping audio."""

    device = "cpu"

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        if self.calls == 1:
            # Chunk 1 is [0, 10), padded to slice [0, 10.5]; this word's centre
            # (10.3, slice-relative) falls past the nominal end at 10 — padding only.
            words = [{"word": "шов", "start": 10.2, "end": 10.4, "probability": None}]
        else:
            # Chunk 2 is [10, 20), padded to slice [9.5, 20]; the same physical word
            # is slice-relative here, straddling lower=0.5 (nominal start), so it must
            # be clipped to it rather than dropped or kept at its raw, earlier start.
            words = [{"word": "шов", "start": 0.3, "end": 0.9, "probability": None}]
        return {"text": "шов", "segments": [], "words": words}

    def close(self):
        pass


def test_a_word_straddling_the_seam_is_kept_once_and_clipped_to_the_boundary(tmp_path, monkeypatch):
    audio = tmp_path / "part.wav"
    _wav(audio, 20)
    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [])
    monkeypatch.setattr(asr, "split_vad_regions", lambda *_a, **_k: [(0.0, 10.0), (10.0, 20.0)])
    output = tmp_path / "asr" / "001.json"

    asr.transcribe_part(
        audio,
        output,
        None,
        "ru",
        "",
        local_backend=SeamBackend(),
        vad_model_path=tmp_path,
        local_fingerprint={},
    )

    words = read_json(output)["words"]
    # Not duplicated: chunk 1 dropped its copy (centre in its own padding), so only
    # chunk 2's survives. Clipped: its raw start (0.3 slice-relative -> 9.8 absolute)
    # sat before the nominal seam at 10.0, so it is pinned to the seam, not left early.
    assert words == [{"word": "шов", "start": 10.0, "end": 10.4, "probability": None}]


def test_the_vad_plan_is_computed_once_and_reused_by_a_second_model(tmp_path, monkeypatch):
    """Two local models in one job must share one segmentation pass.

    The plan depends on the audio and the chunking, never on which checkpoint reads
    it — proven here by the byte-identical plans two models produced on a real run.
    Keying the cache per model made the second model redo the whole pass for nothing.
    """
    audio = tmp_path / "part.wav"
    _wav(audio, 50)
    passes = []

    def counting_regions(*_args, **_kwargs):
        passes.append(1)
        return [(0, 23), (24, 50)]

    monkeypatch.setattr(asr, "detect_speech_regions", counting_regions)
    for model in ("multilingual_large_ctc", "v3_rnnt"):
        asr.transcribe_part(
            audio,
            tmp_path / "asr" / model / "001.json",
            None,
            "ru",
            "",
            local_backend=FakeBackend(),
            vad_model_path=tmp_path,
            # Same audio, different model identity — exactly what run_job builds.
            local_fingerprint={"source_sha256": "a", "gigaam_model": model, "device": "cpu"},
            model_name=model,
        )

    assert passes == [1], "segmentation must run once for both models, not once each"
    assert (tmp_path / "asr" / "vad" / "001.json").exists()


def test_a_changed_chunking_parameter_still_invalidates_the_shared_plan(tmp_path, monkeypatch):
    """Sharing must not cost invalidation: the plan is still keyed on what it depends on."""
    audio = tmp_path / "part.wav"
    _wav(audio, 50)
    passes = []

    def counting_regions(*_args, **_kwargs):
        passes.append(1)
        return [(0, 23), (24, 50)]

    monkeypatch.setattr(asr, "detect_speech_regions", counting_regions)
    for parameters in ({"target": 22.0}, {"target": 18.0}):
        asr.transcribe_part(
            audio,
            tmp_path / "asr" / "m" / "001.json",
            None,
            "ru",
            "",
            local_backend=FakeBackend(),
            vad_model_path=tmp_path,
            local_fingerprint={"source_sha256": "a", "vad_parameters": parameters},
        )

    assert passes == [1, 1]


def test_local_chunks_are_not_paused_between_each_other(tmp_path, monkeypatch):
    """Local recognition must never sleep between chunks.

    A removed --pause-between-chunks option used to insert 20 s here between every
    ~22 s chunk under --profile background: 3.8 hours of pure sleep on a two-hour
    meeting next to ~18 minutes of actual work. Nothing local is being rate-limited,
    so any sleep on this path is a regression; throttling belongs in the profile's
    thread and decode-worker counts.
    """
    audio = tmp_path / "part.wav"
    _wav(audio, 50)
    monkeypatch.setattr(asr, "detect_speech_regions", lambda *_: [(0, 23), (24, 50)])
    slept = []
    monkeypatch.setattr(asr.time, "sleep", lambda seconds: slept.append(seconds))

    asr.transcribe_part(
        audio,
        tmp_path / "asr" / "m" / "001.json",
        None,
        "ru",
        "",
        local_backend=FakeBackend(),
        vad_model_path=tmp_path,
        local_fingerprint={"source_sha256": "a"},
    )

    assert slept == []
