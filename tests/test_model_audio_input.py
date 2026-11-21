import sys
import types
import wave
from types import SimpleNamespace

import pytest
import torch

from transcribe_meeting import audio


@pytest.fixture
def pcm_wav(tmp_path):
    path = tmp_path / "working.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        # Alternating -0.5/+0.5 at exactly representable PCM16 values.
        handle.writeframes(b"\x00\xc0\x00\x40" * 8_000)
    return path


def test_model_waveform_preserves_pcm_samples_and_duration(pcm_wav):
    loaded = audio.load_model_waveform(pcm_wav)
    waveform = loaded["waveform"]
    assert loaded["sample_rate"] == 16_000
    assert waveform.dtype == torch.float32
    assert waveform.shape == (1, 16_000)
    assert waveform.shape[-1] / loaded["sample_rate"] == 1.0
    torch.testing.assert_close(waveform, torch.tensor([[-0.5, 0.5]]).repeat(1, 8_000))


@pytest.mark.parametrize("route", ["vad", "overlap", "community1"])
def test_pyannote_uses_waveform_instead_of_path(route, pcm_wav, tmp_path, monkeypatch):
    from transcribe_meeting import asr, diarization

    turn = SimpleNamespace(start=0.0, end=0.75)
    timeline = SimpleNamespace(support=lambda: [turn])
    annotation = SimpleNamespace(
        get_timeline=lambda: timeline,
        itertracks=lambda yield_label: [(turn, None, "SPEAKER_00")],
    )

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            assert device.type == "cpu"

        def instantiate(self, params):
            pass

        def __call__(self, model_input, **kwargs):
            assert isinstance(model_input, dict), "path input uses the unavailable decoder"
            assert model_input["sample_rate"] == 16_000
            assert model_input["waveform"].shape == (1, 16_000)
            assert model_input["waveform"][0, 0].item() == -0.5
            return annotation

    modules = {
        "pyannote.audio": {
            "Model": SimpleNamespace(from_pretrained=lambda _: object()),
            "Inference": FakePipeline,
            "Pipeline": SimpleNamespace(from_pretrained=lambda _: FakePipeline()),
        },
        "pyannote.audio.pipelines": {"VoiceActivityDetection": FakePipeline},
        "pyannote.audio.utils.signal": {"Binarize": lambda **_: lambda scores: scores},
    }
    for name, members in modules.items():
        module = types.ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYANNOTE_METRICS_ENABLED"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)

    if route == "vad":
        assert asr.detect_speech_regions(pcm_wav, tmp_path) == [(0.0, 0.75)]
    elif route == "overlap":
        checkpoint = tmp_path / "pytorch_model.bin"
        checkpoint.write_bytes(b"checkpoint")
        assert diarization.detect_overlap_regions(pcm_wav, tmp_path, 1) == [(0.0, 0.75)]
    else:
        for name in (
            "config.yaml",
            "segmentation/pytorch_model.bin",
            "embedding/pytorch_model.bin",
            "plda/plda.npz",
            "plda/xvec_transform.npz",
        ):
            file = tmp_path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"checkpoint")
        payload = diarization.diarize(
            pcm_wav,
            tmp_path / "out.json",
            tmp_path / "out.rttm",
            tmp_path,
            1,
            1,
            8,
            1,
            provider="pyannote",
        )
        assert payload["regular"] == [{"start": 0.0, "end": 0.75, "speaker": "SPEAKER_00"}]


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_foxnose_silero_reads_working_pcm_and_restores_reader(fail, pcm_wav, tmp_path, monkeypatch):
    import wespeakerruntime.speaker as speaker

    from transcribe_meeting import diarization

    original_speaker_audio = speaker.torchaudio

    def unavailable_decoder(path, sampling_rate=16_000):
        raise AssertionError("Silero's original reader reached the unavailable decoder")

    def speech_timestamps(waveform, model, *, sampling_rate, **kwargs):
        assert waveform.dtype == torch.float32
        assert waveform.shape == (16_000,)
        assert sampling_rate == 16_000
        torch.testing.assert_close(waveform, torch.tensor([-0.5, 0.5]).repeat(8_000))
        if fail:
            raise RuntimeError("Silero failed after reading")
        return []

    # Keep diarize 0.1.2's actual run_vad; replace only the Silero model and decoder.
    silero = types.ModuleType("silero_vad")
    vars(silero).update(
        read_audio=unavailable_decoder,
        load_silero_vad=object,
        get_speech_timestamps=speech_timestamps,
    )
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not loaded on empty speech spans")
    args = (pcm_wav, tmp_path / "out.json", tmp_path / "out.rttm", None, 1, 8, 1, model)

    if fail:
        with pytest.raises(RuntimeError, match="Silero failed after reading"):
            diarization._diarize_foxnose(*args)
    else:
        assert diarization._diarize_foxnose(*args)["regular"] == []
    assert vars(silero)["read_audio"] is unavailable_decoder
    assert speaker.torchaudio is original_speaker_audio


def test_foxnose_wespeaker_extracts_embeddings_without_torchaudio_load(
    pcm_wav, tmp_path, monkeypatch
):
    import numpy as np
    import torchaudio
    import wespeakerruntime.speaker as speaker

    from transcribe_meeting import diarization

    def unavailable_decoder(*args, **kwargs):
        raise AssertionError("external decoder was called")

    class Session:
        def run(self, *, output_names, input_feed):
            features = input_feed["feats"]
            assert features.ndim == 3 and features.shape[2] == 80
            return [np.ones((1, 256), dtype=np.float32)]

    def speech_timestamps(waveform, model, **kwargs):
        return [{"start": 0.0, "end": 1.0}]

    silero = types.ModuleType("silero_vad")
    vars(silero).update(
        read_audio=unavailable_decoder,
        load_silero_vad=object,
        get_speech_timestamps=speech_timestamps,
    )
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torchaudio, "load", unavailable_decoder)
    monkeypatch.setattr(speaker.ort, "InferenceSession", lambda *args, **kwargs: Session())
    original_speaker_audio = speaker.torchaudio
    model = tmp_path / "model.onnx"
    model.write_bytes(b"checkpoint")

    payload = diarization._diarize_foxnose(
        pcm_wav, tmp_path / "out.json", tmp_path / "out.rttm", 1, 1, 8, 1, model
    )

    assert payload["regular"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert speaker.torchaudio is original_speaker_audio
    assert torchaudio.load is unavailable_decoder


@pytest.mark.parametrize("route", ["community1", "overlap", "foxnose"])
def test_local_diarization_rejects_missing_checkpoint_files(route, pcm_wav, tmp_path):
    from transcribe_meeting import diarization

    # A directory alone is never ready, even if its name looks like a model file.
    model = tmp_path / "model"
    model.mkdir()
    args = (pcm_wav, tmp_path / "out.json", tmp_path / "out.rttm")
    if route == "community1":
        with pytest.raises(RuntimeError, match=r"Community-1.*setup"):
            diarization.diarize(*args, model, 1, 1, 8, 1, provider="pyannote")
    elif route == "overlap":
        with pytest.raises(RuntimeError, match=r"VAD model.*setup"):
            diarization.detect_overlap_regions(pcm_wav, model, 1)
    else:
        with (
            pytest.raises(RuntimeError, match=r"WeSpeaker.*setup"),
            diarization._local_wespeaker(model, 1),
        ):
            pytest.fail("entered WeSpeaker with a directory in place of a checkpoint")
