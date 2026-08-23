from transcribe_meeting.model import Utterance
from transcribe_meeting.render import blocks, finish_light_block, render_markdown


def _utterance(index: int, speaker: str, text: str, start: float, source: str = "a.m4a"):
    return Utterance(
        utterance_id=f"p001-u{index:05d}",
        source_file=source,
        source_start=start,
        source_end=start + 2.0,
        meeting_start=start,
        meeting_end=start + 2.0,
        speaker_id=speaker,
        speaker_name=speaker,
        raw_text=text,
        confidence={"mean_word_probability": 0.9},
        light_text=text,
    )


def test_light_mode_merges_consecutive_turns_of_one_speaker():
    items = [
        _utterance(1, "SPEAKER_00", "первая мысль", 0.0),
        _utterance(2, "SPEAKER_00", "её продолжение", 5.0),
        _utterance(3, "SPEAKER_01", "реплика второго", 10.0),
    ]

    grouped = blocks(items, "light")

    assert [len(block) for block in grouped] == [2, 1]


def test_raw_mode_keeps_one_line_per_utterance():
    """raw is the audit trail against canonical.json, so it must not be merged."""
    items = [
        _utterance(1, "SPEAKER_00", "первая мысль", 0.0),
        _utterance(2, "SPEAKER_00", "её продолжение", 5.0),
    ]

    assert [len(block) for block in blocks(items, "raw")] == [1, 1]


def test_grouping_never_crosses_a_source_file_boundary():
    """A merged block could not name a single source file, so the split is mandatory."""
    items = [
        _utterance(1, "SPEAKER_00", "конец первой части", 0.0, source="a.m4a"),
        _utterance(2, "SPEAKER_00", "начало второй", 5.0, source="b.m4a"),
    ]

    assert [len(block) for block in blocks(items, "light")] == [1, 1]


def test_merged_block_spans_from_first_start_to_last_end(tmp_path):
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    items = [
        _utterance(1, "SPEAKER_00", "первая мысль", 0.0),
        _utterance(2, "SPEAKER_00", "её продолжение", 60.0),
        _utterance(3, "SPEAKER_01", "ответ", 120.0),
    ]

    render_markdown(tmp_path, manifest, items, "light")
    body = (tmp_path / "transcript.light.md").read_text(encoding="utf-8")

    assert "[00:00:00–00:01:02] SPEAKER_00: Первая мысль её продолжение." in body
    assert "[00:02:00–00:02:02] SPEAKER_01: Ответ." in body
    # Every mode is a named file of its own; there is no ambiguous transcript.md.
    assert not (tmp_path / "transcript.md").exists()


def test_overlap_mark_appears_in_light_and_readable_but_not_raw(tmp_path):
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    item = _utterance(1, "SPEAKER_00", "текст реплики", 0.0)
    item.overlap = True
    item.readable_text = "текст реплики"

    render_markdown(tmp_path, manifest, [item], "raw", text_attr="raw_text", suffix="model_a")
    render_markdown(tmp_path, manifest, [item], "light", text_attr="light_text", suffix="model_a")
    render_markdown(tmp_path, manifest, [item], "readable", text_attr="readable_text")

    raw_body = (tmp_path / "transcript.raw.model_a.md").read_text(encoding="utf-8")
    light_body = (tmp_path / "transcript.light.model_a.md").read_text(encoding="utf-8")
    readable_body = (tmp_path / "transcript.readable.md").read_text(encoding="utf-8")

    assert "[overlapping speech]" not in raw_body
    assert "[overlapping speech]" in light_body
    assert "[overlapping speech]" in readable_body


def test_overlap_mark_absent_when_utterance_not_flagged(tmp_path):
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    item = _utterance(1, "SPEAKER_00", "текст реплики", 0.0)
    assert item.overlap is False

    render_markdown(tmp_path, manifest, [item], "light", text_attr="light_text", suffix="model_a")

    body = (tmp_path / "transcript.light.model_a.md").read_text(encoding="utf-8")
    assert "[overlapping speech]" not in body


def test_finish_light_block_capitalizes_and_adds_a_single_final_period():
    assert finish_light_block("привет всем") == "Привет всем."
    # Already terminated: no second period is appended.
    assert finish_light_block("уже готово!") == "Уже готово!"
    assert finish_light_block("") == ""


def test_finish_light_block_skips_the_overlap_marker_when_capitalizing():
    """The marker itself must stay lowercase; capitalization applies to the first
    letter of the actual speech that follows it."""
    assert finish_light_block("[overlapping speech] текст реплики") == (
        "[overlapping speech] Текст реплики."
    )


def test_light_block_is_capitalized_with_a_single_trailing_period(tmp_path):
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    items = [_utterance(1, "SPEAKER_00", "первая мысль", 0.0)]

    render_markdown(tmp_path, manifest, items, "light")
    line = next(
        line
        for line in (tmp_path / "transcript.light.md").read_text(encoding="utf-8").splitlines()
        if "SPEAKER_00" in line
    )

    assert line.endswith("Первая мысль.")
    assert not line.endswith("..")


def test_overlap_mark_in_a_light_block_is_not_capitalized_but_speech_after_it_is(tmp_path):
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    item = _utterance(1, "SPEAKER_00", "текст реплики", 0.0)
    item.overlap = True

    render_markdown(tmp_path, manifest, [item], "light", text_attr="light_text", suffix="model_a")
    body = (tmp_path / "transcript.light.model_a.md").read_text(encoding="utf-8")

    assert "[overlapping speech] Текст реплики." in body


def test_raw_mode_stays_verbatim_without_capitalization_or_a_trailing_period(tmp_path):
    """raw is the verbatim ASR snapshot; only light gets finish_light_block."""
    manifest = {"meeting_name": "test", "sources": [{"path": "/x/a.m4a"}]}
    items = [_utterance(1, "SPEAKER_00", "первая мысль", 0.0)]

    render_markdown(tmp_path, manifest, items, "raw", text_attr="raw_text", suffix="model_a")
    body = (tmp_path / "transcript.raw.model_a.md").read_text(encoding="utf-8")

    assert "SPEAKER_00: первая мысль" in body
    assert "Первая" not in body


def test_every_utterance_id_survives_grouping():
    """Cleaning modes must preserve the set and order of utterance_ids."""
    items = [
        _utterance(1, "SPEAKER_00", "a", 0.0),
        _utterance(2, "SPEAKER_00", "b", 5.0),
        _utterance(3, "SPEAKER_01", "c", 10.0),
        _utterance(4, "SPEAKER_00", "d", 15.0),
    ]

    flattened = [item.utterance_id for block in blocks(items, "light") for item in block]

    assert flattened == [item.utterance_id for item in items]
