"""Compare canonical and variant ASR text offline; see docs/benchmarks.md."""

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def norm(text):
    return [
        t
        for t in (w.strip(" \t.,!?…:;\"'()«»—-").lower().replace("ё", "е") for w in text.split())
        if t
    ]


def report(job):
    data = json.loads((job / "canonical.json").read_text(encoding="utf-8"))
    us = data["utterances"]
    tally = defaultdict(lambda: [0, 0, 0])
    words = 0
    for u in us:
        a = norm(u["raw_text"])
        words += len(a)
        for m, text in (u.get("variants") or {}).items():
            stats = tally[m]
            b = norm(text or "")
            if not b:
                continue
            stats[0] += sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())
            stats[1] += max(len(a), len(b))
            stats[2] += 1
    return {"utterances": len(us), "canonical_words": words, "agreement": tally}


def main():
    parser = argparse.ArgumentParser(description="Compare canonical and variant ASR text.")
    parser.add_argument("job", nargs="+", type=Path)
    for job in parser.parse_args().job:
        try:
            r = report(job)
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"{job}: {exc}")
        utterances = r["utterances"]
        print(f"{job.name or job}: utterances={utterances} words={r['canonical_words']}")
        if not r["agreement"]:
            print("    No ASR variants found")
        for m, (same, total, covered) in sorted(r["agreement"].items()):
            pct = f"{100 * same / total:.2f}%" if total else "N/A"
            print(
                f"    vs {m}: {pct} ({total} positions; {covered}/{utterances} utterances covered)"
            )


if __name__ == "__main__":
    main()
