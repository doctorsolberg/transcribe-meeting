from transcribe_meeting import core
from transcribe_meeting.util import atomic_json, read_json, sha256


def test_working_audio_matches_relative_diagnostics_to_absolute_manifest_path(
    tmp_path, monkeypatch
):
    source = tmp_path / "src" / "part.wav"
    source.parent.mkdir()
    source.write_bytes(b"source")
    atomic_json(
        tmp_path / "diagnostics.json",
        {"parts": [{"path": "src/part.wav", "stream_index": 0}]},
    )
    manifest = {
        "sources": [{"path": str(source.resolve()), "sha256": sha256(source)}],
        "config": {"asr_preprocess": "none", "asr_compression": "none"},
    }
    monkeypatch.setenv("TRANSCRIBE_HOME", str(tmp_path))

    def fake_decode(_source, target, stream, *_args):
        assert stream == 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"pcm")
        return {"mode": "none", "filters": []}

    monkeypatch.setattr(core, "decode_pcm", fake_decode)
    assert core._working_audio(tmp_path, manifest, ffmpeg_threads=1, decode_workers=1) == [
        tmp_path / "work" / "pcm" / "001.wav"
    ]
    assert read_json(tmp_path / "diagnostics.json")["asr_and_diarization_preprocessing"][0][
        "source_file"
    ] == str(source.resolve())
