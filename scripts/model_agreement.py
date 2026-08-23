"""Agreement between the canonical ASR model and each variant model within one job.

The project has no reference transcript, and no connected backend returns word
probabilities, so the only available quality metric is agreement between two
independent models. This is exactly the metric behind the numbers in the README's
"Measured facts" section; this script lives here so those numbers can be rechecked,
not just read.

    uv run python scripts/model_agreement.py output/<job-id> [output/<other-job>]

Be careful interpreting this: the metric is blind to defects shared by both models.
If both are chunked at the same boundaries, their errors at those boundaries will
match, and agreement will not reveal it.
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path


def norm(text):
    return [
        t
        for t in (w.strip(" \t.,!?…:;\"'()«»—-").lower().replace("ё", "е") for w in text.split())
        if t
    ]


def report(job):
    data = json.loads(Path(f"{job}/canonical.json").read_text(encoding="utf-8"))
    us = data["utterances"]
    models = sorted({m for u in us for m in (u.get("variants") or {})})
    out = {}
    for m in models:
        same = total = 0
        for u in us:
            a, b = norm(u["raw_text"]), norm((u.get("variants") or {}).get(m, ""))
            if not b:
                continue
            match = sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())
            same += match
            total += max(len(a), len(b))
        out[m] = (100.0 * same / total if total else 0.0, total)
    words = sum(len(norm(u["raw_text"])) for u in us)
    return {"utterances": len(us), "canonical_words": words, "agreement": out}


for job in sys.argv[1:]:
    r = report(job)
    print(f"{job.split('/')[-1]}: utterances={r['utterances']} words={r['canonical_words']}")
    for m, (pct, tot) in r["agreement"].items():
        print(f"    vs {m}: {pct:.2f}% (over {tot} positions)")
