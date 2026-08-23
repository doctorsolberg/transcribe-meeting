from transcribe_meeting.timeline import build_sources


def test_three_clocks_exclude_synthetic_gap_from_meeting_time():
    first, second = build_sources([("one.m4a", "a", 60.0), ("two.m4a", "b", 20.0)])
    assert first.offset == 0.0
    assert first.diarization_offset == 0.0
    # The 1-second synthetic gap only exists on the diarization clock: meeting_start
    # for the second source is exactly the first source's duration, not +1.
    assert second.offset == 60.0
    assert second.diarization_offset == 61.0
