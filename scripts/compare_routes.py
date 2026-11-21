"""Compare completed jobs on the same recording offline; see docs/benchmarks.md."""

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from math import ceil, isfinite
from pathlib import Path

MAX_ASSIGNMENT_SPEAKERS = 16
# Serialized UNKNOWN_SPEAKER from canonical.json; not an actual speaker label.
UNKNOWN_SPEAKER = "Speaker ?"


def norm(t):
    return [
        w
        for w in (x.strip(" \t.,!?…:;\"'()«»—-").lower().replace("ё", "е") for x in t.split())
        if w
    ]


def load(job):
    c = json.loads((job / "canonical.json").read_text(encoding="utf-8"))
    m = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    us = c["utterances"]
    for i, u in enumerate(us, 1):
        start, end = u["meeting_start"], u["meeting_end"]
        if not (isfinite(start) and isfinite(end) and 0 <= start <= end):
            raise ValueError(f"invalid utterance time bounds at utterance {i}")
    us = sorted(us, key=lambda u: u["meeting_start"])
    return {
        "name": job.name or str(job),
        "models": m["config"]["asr_models"],
        "diarization": m["config"]["diarization_provider"],
        "utterances": us,
        "text": [w for u in us for w in norm(u["raw_text"])],
        "duration": max((u["meeting_end"] for u in us), default=0.0),
    }


def speaker_track(job):
    """Label at each whole second in [start, end), without extending boundaries."""
    track = [None] * ceil(job["duration"])
    for u in job["utterances"]:
        label = u["speaker_id"]
        for i in range(ceil(u["meeting_start"]), ceil(u["meeting_end"])):
            track[i] = None if label == UNKNOWN_SPEAKER else label
    return track


def diarization_agreement(a, b):
    counts = Counter(
        (x, y) for x, y in zip(a["speaker_track"], b["speaker_track"], strict=False) if x and y
    )
    both = counts.total()
    if not both:
        return 0.0, 0

    left = sorted({x for x, _ in counts})
    right = sorted({y for _, y in counts})
    if len(right) > len(left):
        left, right = right, left
        counts = Counter({(y, x): count for (x, y), count in counts.items()})
    if len(right) > MAX_ASSIGNMENT_SPEAKERS:
        raise ValueError(
            f"speaker matching supports at most {MAX_ASSIGNMENT_SPEAKERS} labels on the "
            "smaller side of the shared speech"
        )

    # Track used labels on the smaller side as a bit mask. Each row may take
    # one unused label or stay unmatched; zero-count matches cannot help.
    bits = {label: 1 << i for i, label in enumerate(right)}
    rows = defaultdict(list)
    for (x, y), count in counts.items():
        rows[x].append((bits[y], count))
    scores = {0: 0}
    for x in left:
        next_scores = scores.copy()
        for bit, count in rows[x]:
            for used, score in scores.items():
                if not used & bit:
                    new = used | bit
                    next_scores[new] = max(next_scores.get(new, 0), score + count)
        scores = next_scores

    return 100.0 * max(scores.values()) / both, both


def compare(jobs):
    print("=== routes ===")
    for j in jobs:
        j["speaker_track"] = speaker_track(j)
        speech = defaultdict(float)
        for u in j["utterances"]:
            label = u["speaker_id"]
            if label and label != UNKNOWN_SPEAKER:
                speech[label] += u["meeting_end"] - u["meeting_start"]
        print(f"{j['name']}")
        print(f"    ASR: {','.join(j['models'])}   diarization: {j['diarization']}")
        duration = j["duration"]
        rate = f"{60 * len(j['text']) / duration:.0f} words/min" if duration else "N/A"
        print(
            f"    utterances {len(j['utterances'])}, words {len(j['text'])}, "
            f"duration {duration / 60:.1f} min, rate {rate}"
        )
        print(
            f"    speakers {len(speech)}: "
            + ", ".join(
                f"{k}={v / 60:.1f}m" for k, v in sorted(speech.items(), key=lambda kv: -kv[1])
            )
        )

    print("\n=== pairwise comparison ===")
    for i in range(len(jobs)):
        for b in jobs[i + 1 :]:
            a = jobs[i]
            if a["text"] or b["text"]:
                ratio = f"{100 * SequenceMatcher(None, a['text'], b['text']).ratio():.2f}%"
            else:
                ratio = "N/A"
            try:
                agree, both = diarization_agreement(a, b)
            except ValueError as exc:
                raise ValueError(f"{a['name']} <-> {b['name']}: {exc}") from exc
            speaker_match = f"{agree:.2f}%" if both else "N/A"
            print(f"{a['name']}  <->  {b['name']}")
            print(f"    text match: {ratio}")
            print(
                f"    speaker labeling match: {speaker_match} "
                f"(over {both} sampled seconds of shared speech)"
            )


def main():
    parser = argparse.ArgumentParser(description="Compare completed jobs on the same recording.")
    parser.add_argument("job", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.job) < 2:
        parser.error("provide at least two job directories")
    jobs = []
    for path in args.job:
        try:
            jobs.append(load(path))
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            parser.error(f"{path}: {exc}")
    try:
        compare(jobs)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
