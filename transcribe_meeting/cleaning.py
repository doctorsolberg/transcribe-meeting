from __future__ import annotations

import re
from collections.abc import Iterable

FILLERS = {"э-э", "э…", "эм", "эм-м", "м-м", "мм", "гм"}
NOISE = {"[шум]", "[музыка]", "(шум)", "(смех)"}


def glossary_replace(text: str, terms: Iterable[dict]) -> str:
    for term in terms:
        for variant in term.get("variants", []):
            text = re.sub(
                rf"(?<!\w){re.escape(variant)}(?!\w)", term["canonical"], text, flags=re.IGNORECASE
            )
    return text


def light_clean(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for token in tokens:
        normalized = token.strip(".,!?…").lower()
        if normalized in FILLERS or token.lower() in NOISE:
            continue
        # Exact adjacent-word repeats only. Do not infer semantic repetitions.
        if kept and normalized and normalized == kept[-1].strip(".,!?…").lower():
            continue
        kept.append(token)
    # Short exact phrase repeats (two to four tokens) are equally mechanical.
    # Work after filler removal so we never remove context-dependent words by meaning.
    changed = True
    while changed:
        changed = False
        for width in range(min(4, len(kept) // 2), 1, -1):
            index = 0
            while index + width * 2 <= len(kept):
                left = [x.strip(".,!?…").lower() for x in kept[index : index + width]]
                right = [x.strip(".,!?…").lower() for x in kept[index + width : index + width * 2]]
                if left == right:
                    del kept[index + width : index + width * 2]
                    changed = True
                else:
                    index += 1
    output = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(kept))
    output = re.sub(r"\s+", " ", output).strip()
    return output
