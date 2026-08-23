from transcribe_meeting.core import speaker_previews
from transcribe_meeting.model import Word


def test_speaker_previews_choose_a_substantive_contiguous_phrase_with_meeting_time():
    source = {"path": "/meeting/one.m4a", "offset": 120.0}
    words = [
        Word(1, 1.2, " да", speaker_id="S1"),
        Word(2, 2.2, " начнём с плана релиза и ответственных", speaker_id="S2"),
        Word(2.3, 2.5, " на эту неделю", speaker_id="S2"),
        Word(3, 3.2, " хорошо", speaker_id="S1"),
    ]

    previews = speaker_previews([(source, words)])

    assert previews["S2"]["meeting_start"] == 122
    assert previews["S2"]["source_file"] == "/meeting/one.m4a"
    assert previews["S2"]["text"] == "начнём с плана релиза и ответственных на эту неделю"
