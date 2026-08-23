import wave

from transcribe_meeting import asr, core
from transcribe_meeting.model import Utterance
from transcribe_meeting.util import atomic_json, read_json


def _wav(path, seconds=1):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(100)
        handle.writeframes(b"\0\0" * seconds * 100)


class Backend:
    device = "cpu"

    def transcribe(self, path):
        return {
            "text": "слово",
            "segments": [],
            "words": [
                {
                    "word": "слово",
                    "start": 0,
                    "end": min(0.5, asr._wav_duration(path)),
                    "probability": None,
                }
            ],
        }

    def close(self):
        pass


def test_local_parts_keep_separate_source_clocks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        asr, "detect_speech_regions", lambda audio, _: [(0, asr._wav_duration(audio))]
    )
    backend = Backend()
    for number in (1, 2):
        audio = tmp_path / f"{number}.wav"
        _wav(audio)
        asr.transcribe_part(
            audio,
            tmp_path / "asr" / f"{number:03d}.json",
            None,
            "ru",
            "",
            local_backend=backend,
            vad_model_path=tmp_path,
            local_fingerprint={"part": number},
            model_name="v3_rnnt",
        )
    assert read_json(tmp_path / "asr" / "001.json")["words"][0]["start"] == 0
    assert read_json(tmp_path / "asr" / "002.json")["words"][0]["start"] == 0


def test_missing_probability_still_creates_a_review_row(tmp_path):
    # No ASR backend returns word probabilities, so `prepare_review` must not filter
    # on confidence: a None mean_word_probability must not drop the utterance.
    utterance = Utterance(
        "u",
        "a.wav",
        0,
        1,
        0,
        1,
        "SPEAKER_00",
        "SPEAKER_00",
        "тест",
        {"mean_word_probability": None},
    )
    atomic_json(
        tmp_path / "canonical.json",
        {"schema_version": core.CANONICAL_SCHEMA_VERSION, "utterances": [utterance.to_dict()]},
    )
    review = core.prepare_review(tmp_path)
    lines = review.read_text(encoding="utf-8").splitlines()
    assert (
        lines[0]
        == "utterance_id,timestamp,speaker,text,context,suggestion,replacement,speaker_replacement"
    )
    assert len(lines) == 2
