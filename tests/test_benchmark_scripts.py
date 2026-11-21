"""Exercise the offline tools from the checkout with small synthetic jobs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def make_job(root, name, utterances):
    path = root / name
    path.mkdir()
    (path / "canonical.json").write_text(json.dumps({"utterances": utterances}), encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps({"config": {"asr_models": ["v3_rnnt"], "diarization_provider": "foxnose"}}),
        encoding="utf-8",
    )
    return path


def utterance(start, end, speaker="a", text="Привет", **extra):
    return {
        "meeting_start": start,
        "meeting_end": end,
        "speaker_id": speaker,
        "raw_text": text,
        **extra,
    }


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def test_empty_jobs_report_undefined_metrics(tmp_path):
    a = make_job(tmp_path, "a", [])
    b = make_job(tmp_path, "b", [])
    result = run_script("compare_routes.py", a, b)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rate N/A") == 2
    assert "text match: N/A" in result.stdout
    assert "speaker labeling match: N/A (over 0 sampled seconds of shared speech)" in result.stdout


def test_many_speakers_are_matched_optimally_without_permutations(tmp_path):
    # Matching the most frequent pair (a, x) first would be suboptimal:
    # (a, y) + (b, x) gives eight hits instead of five.
    pairs = [("a", "x")] * 5 + [("a", "y")] * 4 + [("b", "x")] * 4
    pairs.extend((f"s{i}", f"t{i}") for i in range(10))
    pairs.append(("a", "extra"))
    a = make_job(tmp_path, "a", [utterance(i, i + 1, x) for i, (x, _) in enumerate(pairs)])
    b = make_job(tmp_path, "b", [utterance(i, i + 1, y) for i, (_, y) in enumerate(pairs)])
    for jobs in ((a, b), (b, a)):
        result = run_script("compare_routes.py", *jobs)
        assert result.returncode == 0, result.stderr
        assert "speaker labeling match: 75.00% (over 24 sampled seconds" in result.stdout


def test_speaker_assignment_has_a_clear_cli_limit(tmp_path):
    spans = [utterance(i, i + 1, str(i)) for i in range(17)]
    a, b = [make_job(tmp_path, name, spans) for name in ("a", "b")]
    result = run_script("compare_routes.py", a, b)
    assert result.returncode != 0
    assert "at most 16 labels" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, float("nan")), (0, float("inf"))])
def test_invalid_time_bounds_report_an_input_error(tmp_path, start, end):
    bad = make_job(tmp_path, "bad-job", [utterance(start, end)])
    other = make_job(tmp_path, "other", [utterance(0, 1)])
    result = run_script("compare_routes.py", bad, other)
    assert result.returncode != 0
    assert "bad-job" in result.stderr and "invalid utterance time bounds" in result.stderr
    assert "Traceback" not in result.stderr


def test_no_shared_samples_does_not_imply_speaker_mismatch(tmp_path):
    a = make_job(tmp_path, "a", [utterance(0.2, 1.1)])
    b = make_job(tmp_path, "b", [utterance(0.0, 1.0)])
    result = run_script("compare_routes.py", a, b)
    assert result.returncode == 0, result.stderr
    assert "text match: 100.00%" in result.stdout
    assert "speaker labeling match: N/A (over 0 sampled seconds" in result.stdout


def test_unattributed_speech_does_not_count_as_a_speaker_match(tmp_path):
    a = make_job(tmp_path, "a", [utterance(0, 2, "Speaker ?")])
    b = make_job(tmp_path, "b", [utterance(0, 2, "speaker-b")])
    result = run_script("compare_routes.py", a, b)
    assert result.returncode == 0, result.stderr
    assert "speakers 0:" in result.stdout
    assert "speaker labeling match: N/A (over 0 sampled seconds" in result.stdout


def test_compare_requires_at_least_two_jobs():
    result = run_script("compare_routes.py")
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_model_agreement_reports_variant_coverage_without_scoring_missing_text(tmp_path):
    job = make_job(
        tmp_path,
        "local",
        [
            utterance(0, 2, text="Ёлка, дом!", variants={"second": "елка ДОМ", "blank": "…"}),
            utterance(2, 3, text="потом", variants=None),
        ],
    )
    result = run_script("model_agreement.py", f"{job}/")
    assert result.returncode == 0, result.stderr
    assert "local: utterances=2 words=3" in result.stdout
    assert "vs blank: N/A (0 positions; 0/2 utterances covered)" in result.stdout
    assert "vs second: 100.00% (2 positions; 1/2 utterances covered)" in result.stdout


def test_model_agreement_on_silent_job_reports_no_variants(tmp_path):
    job = make_job(tmp_path, "silent", [])
    result = run_script("model_agreement.py", job)
    assert result.returncode == 0, result.stderr
    assert "silent: utterances=0 words=0" in result.stdout
    assert "No ASR variants found" in result.stdout
