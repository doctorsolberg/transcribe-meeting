from transcribe_meeting.asr import words_from_asr


def test_groq_root_words_are_used_and_spaced_for_rendering():
    data = {
        "words": [
            {"start": 0.0, "end": 0.2, "word": "Первое"},
            {"start": 0.2, "end": 0.5, "word": "слово"},
        ]
    }

    words = words_from_asr(data)

    assert [word["text"] for word in words] == ["Первое", " слово"]
