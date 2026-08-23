from transcribe_meeting.core import PROFILES


def test_profiles_limit_parallel_decode_jobs_on_16gb_machine():
    assert PROFILES["background"]["decode_workers"] == 1
    assert PROFILES["balanced"]["decode_workers"] == 2
    assert PROFILES["max"]["decode_workers"] == 2
    assert PROFILES["balanced"]["ffmpeg_threads"] == 2


def test_working_bytes_estimate_scales_with_duration_and_local_models():
    """Two tracks always coexist (per-part PCM plus the concatenated one) and each
    local model adds one set of sliced chunk WAVs on top."""
    from transcribe_meeting.core import WORKING_AUDIO_BYTES_PER_SECOND, estimate_working_bytes

    hour = 3600.0
    assert estimate_working_bytes(hour, 0) == int(hour * WORKING_AUDIO_BYTES_PER_SECOND * 2 * 1.25)
    assert estimate_working_bytes(hour, 2) == int(hour * WORKING_AUDIO_BYTES_PER_SECOND * 4 * 1.25)
    # A network-only route needs no scratch space of its own beyond the two tracks.
    assert estimate_working_bytes(hour, 0) < estimate_working_bytes(hour, 1)
