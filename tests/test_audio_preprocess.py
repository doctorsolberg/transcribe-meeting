import hashlib
import shutil
import wave
from array import array

import pytest

from transcribe_meeting.audio import (
    COMPRESSION_4_TO_1,
    PRE_EMPHASIS_COEFFICIENT,
    decode_pcm,
    pre_emphasize_pcm,
)

# decode_pcm shells out to a real ffmpeg binary (no monkeypatching), so these two
# tests need it on PATH; the rest of the suite stubs ffmpeg entirely.
requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a real ffmpeg")


def write_pcm(path, samples):
    data = array("h", samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(data.tobytes())


def read_pcm(path):
    with wave.open(str(path), "rb") as handle:
        data = array("h")
        data.frombytes(handle.readframes(handle.getnframes()))
        return list(data)


def test_pre_emphasis_is_non_destructive_and_uses_expected_coefficient(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "processed.wav"
    write_pcm(source, [1000, 2000, -30000])

    clipped = pre_emphasize_pcm(source, target)

    assert read_pcm(source) == [1000, 2000, -30000]
    assert read_pcm(target) == [1000, round(2000 - PRE_EMPHASIS_COEFFICIENT * 1000), -31940]
    assert clipped == 0


def test_pre_emphasis_reports_clipping_instead_of_normalizing(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "processed.wav"
    write_pcm(source, [32767, -32768])

    assert pre_emphasize_pcm(source, target) == 1
    assert read_pcm(target) == [32767, -32768]


@requires_ffmpeg
def test_hpf_and_pre_emphasis_decode_to_working_pcm_without_changing_source(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "work" / "processed.wav"
    write_pcm(source, [1000, 2000, -30000, 2000] * 400)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    report = decode_pcm(source, target, 0, "hpf80-preemphasis")

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert target.is_file()
    assert report["highpass_hz"] == 80
    assert report["pre_emphasis_coefficient"] == PRE_EMPHASIS_COEFFICIENT
    assert not target.with_suffix(".decode.wav").exists()
    assert not target.with_suffix(".preemphasis.wav").exists()


@requires_ffmpeg
def test_compression_4_to_1_is_applied_to_a_temporary_pcm_copy(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "work" / "compressed.wav"
    write_pcm(source, [0] * 160 + [20000] * 1600)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    report = decode_pcm(source, target, 0, asr_compression="4:1")

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert report["compression"] == {"mode": "4:1", **COMPRESSION_4_TO_1}
    assert "ratio=4" in report["filters"][-1]
    assert max(read_pcm(target)[500:]) < 20000
