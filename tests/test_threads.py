from transcribe_meeting import core


def test_threads_are_limited_to_available_cores_without_oversubscription(monkeypatch):
    monkeypatch.setattr(core.os, "cpu_count", lambda: 10)

    assert core.effective_threads(16) == 10
    assert core.effective_threads(8, reserve_for_parallel_jobs=2) == 5
