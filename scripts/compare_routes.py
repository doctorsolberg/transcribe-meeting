"""Compare finished jobs produced by different routes on the same recording.

Checks two things: whether the transcripts say the same thing (word-level similarity
of the whole meeting's text) and whether the diarizations agree on who is speaking
when (per-second labels under the best label correspondence — the same measure the
README uses to report Deepgram/foxnose agreement).

    uv run python scripts/compare_routes.py output/<job-a> output/<job-b> [...]

Word-level similarity against Deepgram is understated by construction: it returns
punctuation, capitalization, and digits instead of words.
"""

import json
import sys
from difflib import SequenceMatcher
from itertools import permutations
from pathlib import Path


def norm(t):
    return [
        w
        for w in (x.strip(" \t.,!?…:;\"'()«»—-").lower().replace("ё", "е") for x in t.split())
        if w
    ]


def load(job):
    c = json.loads(Path(f"{job}/canonical.json").read_text(encoding="utf-8"))
    m = json.loads(Path(f"{job}/manifest.json").read_text(encoding="utf-8"))
    us = sorted(c["utterances"], key=lambda u: u["meeting_start"])
    return {
        "name": job.rstrip("/").split("/")[-1],
        "models": m["config"]["asr_models"],
        "diarization": m["config"]["diarization_provider"],
        "utterances": us,
        "text": [w for u in us for w in norm(u["raw_text"])],
        "duration": max((u["meeting_end"] for u in us), default=0.0),
    }


def speaker_track(job, step=1.0):
    """Label per time step; None where nobody is speaking in that job."""
    end = job["duration"]
    track = [None] * (int(end / step) + 1)
    for u in job["utterances"]:
        for i in range(
            int(u["meeting_start"] / step), min(int(u["meeting_end"] / step) + 1, len(track))
        ):
            track[i] = u["speaker_id"]
    return track


def diarization_agreement(a, b):
    ta, tb = speaker_track(a), speaker_track(b)
    n = min(len(ta), len(tb))
    ta, tb = ta[:n], tb[:n]
    labels_a = sorted({x for x in ta if x})
    labels_b = sorted({x for x in tb if x})
    if not labels_a or not labels_b:
        return 0.0, 0
    best = 0
    wide, narrow = (labels_a, labels_b) if len(labels_a) >= len(labels_b) else (labels_b, labels_a)
    for perm in permutations(wide, len(narrow)):
        mapping = dict(zip(narrow, perm, strict=True))
        if len(labels_a) >= len(labels_b):
            hits = sum(1 for x, y in zip(ta, tb, strict=True) if x and y and x == mapping.get(y))
        else:
            hits = sum(1 for x, y in zip(ta, tb, strict=True) if x and y and mapping.get(x) == y)
        best = max(best, hits)
    both = sum(1 for x, y in zip(ta, tb, strict=True) if x and y)
    return (100.0 * best / both if both else 0.0), both


jobs = [load(p) for p in sys.argv[1:]]
print("=== routes ===")
for j in jobs:
    speech = {}
    for u in j["utterances"]:
        speech[u["speaker_id"]] = (
            speech.get(u["speaker_id"], 0.0) + u["meeting_end"] - u["meeting_start"]
        )
    print(f"{j['name']}")
    print(f"    ASR: {','.join(j['models'])}   diarization: {j['diarization']}")
    rate = len(j["text"]) / (j["duration"] / 60)
    print(
        f"    utterances {len(j['utterances'])}, words {len(j['text'])}, "
        f"duration {j['duration'] / 60:.1f} min, rate {rate:.0f} words/min"
    )
    print(
        f"    speakers {len(speech)}: "
        + ", ".join(f"{k}={v / 60:.1f}m" for k, v in sorted(speech.items(), key=lambda kv: -kv[1]))
    )

print()
print("=== pairwise comparison ===")
for i in range(len(jobs)):
    for k in range(i + 1, len(jobs)):
        a, b = jobs[i], jobs[k]
        ratio = SequenceMatcher(None, a["text"], b["text"]).ratio() * 100
        agree, both = diarization_agreement(a, b)
        print(f"{a['name']}  <->  {b['name']}")
        print(f"    text match: {ratio:.2f}%")
        print(f"    speaker labeling match: {agree:.2f}% (over {both} s of shared speech)")
