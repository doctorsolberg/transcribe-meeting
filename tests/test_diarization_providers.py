import sys
import types

import pytest

from transcribe_meeting import diarization


class _FakeSegment:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker


class _FakeEstimationDetails:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeResult:
    def __init__(self, segments, num_speakers, estimation_details=None):
        self.segments = segments
        self.num_speakers = num_speakers
        self.estimation_details = estimation_details


def _install_fake_diarize(
    monkeypatch, captured_kwargs, segments, *, num_speakers=0, estimation_details=None
):
    """Stand in for the whole `diarize` package with a single deterministic call."""

    def fake_diarize(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResult(segments, num_speakers, estimation_details)

    module = types.ModuleType("diarize")
    module.diarize = fake_diarize
    monkeypatch.setitem(sys.modules, "diarize", module)


def _install_fake_wespeakerruntime(monkeypatch):
    """Minimal stand-in for wespeakerruntime.hub.Hub and wespeakerruntime.Speaker."""
    hub_module = types.ModuleType("wespeakerruntime.hub")

    class Hub:
        @staticmethod
        def get_model_by_lang(lang):
            raise AssertionError("real Hub.get_model_by_lang must never run under the shim")

    hub_module.Hub = Hub

    class Speaker:
        def __init__(
            self, onnx_path=None, lang="chs", inter_op_num_threads=1, intra_op_num_threads=1
        ):
            self.onnx_path = onnx_path
            self.lang = lang
            self.inter_op_num_threads = inter_op_num_threads
            self.intra_op_num_threads = intra_op_num_threads

    root_module = types.ModuleType("wespeakerruntime")
    root_module.hub = hub_module
    root_module.Speaker = Speaker
    monkeypatch.setitem(sys.modules, "wespeakerruntime", root_module)
    monkeypatch.setitem(sys.modules, "wespeakerruntime.hub", hub_module)
    return Hub, Speaker


def test_diarize_dispatch_rejects_unknown_provider(tmp_path):
    with pytest.raises(RuntimeError, match="Unknown diarization provider"):
        diarization.diarize(
            tmp_path / "a.wav",
            tmp_path / "o.json",
            tmp_path / "o.rttm",
            tmp_path / "m",
            None,
            1,
            8,
            1,
            provider="bogus",
        )


def test_diarize_dispatch_foxnose_requires_wespeaker_model_path(tmp_path):
    with pytest.raises(RuntimeError, match="WeSpeaker"):
        diarization.diarize(
            tmp_path / "a.wav",
            tmp_path / "o.json",
            tmp_path / "o.rttm",
            tmp_path / "m",
            None,
            1,
            8,
            1,
            provider="foxnose",
        )


def test_diarize_foxnose_missing_model_file_raises_with_setup_hint(tmp_path, monkeypatch):
    _install_fake_wespeakerruntime(monkeypatch)
    _install_fake_diarize(monkeypatch, {}, segments=[])
    missing = tmp_path / "missing.onnx"

    with pytest.raises(RuntimeError, match="setup"):
        diarization._diarize_foxnose(
            tmp_path / "a.wav", tmp_path / "o.json", tmp_path / "o.rttm", None, 1, 8, 1, missing
        )


def test_local_wespeaker_shim_patches_hub_and_thread_default(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    Hub, Speaker = _install_fake_wespeakerruntime(monkeypatch)

    with diarization._local_wespeaker(model_path, cpu_threads=7):
        assert Hub.get_model_by_lang("en") == str(model_path)
        speaker = Speaker(lang="en")
        assert speaker.intra_op_num_threads == 7
        # An explicit caller-provided value must still win over the shim's default.
        explicit = Speaker(lang="en", intra_op_num_threads=2)
        assert explicit.intra_op_num_threads == 2

    # Fully rolled back: the original (raising) Hub lookup and default thread count return.
    with pytest.raises(AssertionError):
        Hub.get_model_by_lang("en")
    assert Speaker(lang="en").intra_op_num_threads == 1
    # Restored as a staticmethod, not a plain function, so instance access keeps working too.
    assert type(Hub.__dict__["get_model_by_lang"]) is staticmethod


def test_local_wespeaker_shim_restores_after_exception(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    Hub, Speaker = _install_fake_wespeakerruntime(monkeypatch)

    with (
        pytest.raises(RuntimeError, match="boom"),
        diarization._local_wespeaker(model_path, cpu_threads=3),
    ):
        raise RuntimeError("boom")

    with pytest.raises(AssertionError):
        Hub.get_model_by_lang("en")
    assert Speaker(lang="en").intra_op_num_threads == 1


def test_local_wespeaker_shim_raises_when_model_missing(tmp_path):
    missing = tmp_path / "missing.onnx"
    with (
        pytest.raises(RuntimeError, match="setup"),
        diarization._local_wespeaker(missing, cpu_threads=1),
    ):
        pass


def test_diarize_foxnose_payload_contract(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_wespeakerruntime(monkeypatch)
    captured: dict = {}
    segments = [_FakeSegment(2.0, 3.5, "SPEAKER_01"), _FakeSegment(0.0, 1.0, "SPEAKER_00")]
    _install_fake_diarize(
        monkeypatch,
        captured,
        segments,
        num_speakers=2,
        estimation_details=_FakeEstimationDetails({"method": "gmm_bic", "best_k": 2}),
    )
    output = tmp_path / "diarization.json"
    rttm = tmp_path / "diarization.rttm"

    payload = diarization._diarize_foxnose(
        tmp_path / "meeting.wav", output, rttm, None, 1, 8, 2, model_path
    )

    assert payload["provider"] == "foxnose"
    assert payload["repository"] == diarization.FOXNOSE_REPO
    assert payload["supports_overlap"] is False
    assert payload["regular"] == payload["exclusive"]
    # Rows must come back sorted by start even though the fake result was unsorted.
    assert [row["start"] for row in payload["regular"]] == [0.0, 2.0]
    assert payload["num_speakers"] == 2
    assert payload["estimation_details"] == {"method": "gmm_bic", "best_k": 2}
    assert output.exists()
    assert rttm.read_text(encoding="utf-8") == (
        "SPEAKER meeting 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER meeting 1 2.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )
    # speakers=None routes to min/max, never num_speakers.
    assert captured == {"min_speakers": 1, "max_speakers": 8}


def test_diarize_foxnose_forwards_num_speakers_when_speakers_given(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _install_fake_wespeakerruntime(monkeypatch)
    captured: dict = {}
    _install_fake_diarize(monkeypatch, captured, segments=[])

    diarization._diarize_foxnose(
        tmp_path / "meeting.wav", tmp_path / "o.json", tmp_path / "o.rttm", 3, 1, 8, 1, model_path
    )

    assert captured == {"num_speakers": 3}


def test_write_rttm_is_the_shared_formatter_for_both_routes(tmp_path):
    rows = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.5, "speaker": "SPEAKER_01"},
    ]
    path_a = tmp_path / "a.rttm"
    path_b = tmp_path / "b.rttm"

    diarization._write_rttm(path_a, rows)
    diarization._write_rttm(path_b, rows)

    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")
    assert path_a.read_text(encoding="utf-8") == (
        "SPEAKER meeting 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER meeting 1 2.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )


def test_diarize_foxnose_on_empty_segments_never_touches_speaker(tmp_path, monkeypatch):
    """Regression guard for the setup() foxnose smoke test.

    Upstream `diarize()` returns right after VAD when there is no speech (e.g. on a
    silent smoke WAV), before it ever reaches extract_embeddings/Speaker. A fake
    `diarize()` with zero segments reproduces that shape: _diarize_foxnose alone must
    not be treated as proof that the WeSpeaker ONNX file and onnxruntime actually load
    (setup() therefore also instantiates Speaker directly through the shim).
    """
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _, Speaker = _install_fake_wespeakerruntime(monkeypatch)
    speaker_calls: list = []
    original_speaker_init = Speaker.__init__

    def tracking_init(self, *args, **kwargs):
        speaker_calls.append((args, kwargs))
        return original_speaker_init(self, *args, **kwargs)

    Speaker.__init__ = tracking_init
    _install_fake_diarize(monkeypatch, {}, segments=[])

    diarization._diarize_foxnose(
        tmp_path / "silence.wav",
        tmp_path / "o.json",
        tmp_path / "o.rttm",
        None,
        1,
        8,
        1,
        model_path,
    )

    assert speaker_calls == []


def test_filter_overlap_regions_keeps_only_spans_longer_than_minimum():
    regions = [(0.0, 0.3), (1.0, 1.6), (5.0, 5.5)]

    filtered = diarization._filter_overlap_regions(regions)

    # 0.3s dropped (< 0.5s), 0.6s kept, exactly-0.5s dropped (strictly longer required).
    assert filtered == [(1.0, 1.6)]


def test_detect_overlap_regions_raises_clearly_when_model_missing(tmp_path):
    missing = tmp_path / "missing-segmentation-model"

    with pytest.raises(RuntimeError, match="VAD model"):
        diarization.detect_overlap_regions(tmp_path / "meeting.wav", missing, 1)


def test_diarize_dispatch_requires_explicit_provider(tmp_path):
    """The CLI owns the default; the dispatcher must never guess a route."""
    with pytest.raises(TypeError, match="provider"):
        diarization.diarize(
            tmp_path / "a.wav",
            tmp_path / "o.json",
            tmp_path / "o.rttm",
            tmp_path / "m",
            None,
            1,
            8,
            1,
        )
