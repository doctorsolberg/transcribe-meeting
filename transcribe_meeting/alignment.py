from __future__ import annotations

import bisect

from .diarization import OVERLAP_MIN_SECONDS
from .model import Utterance, Word

# A word whose center misses every canonical interval (small timing drift between the
# two ASR models is enough on a sub-second utterance) still attaches to the nearest
# utterance boundary within this many seconds; farther than that, it is genuinely
# outside anything the canonical model heard nearby and gets dropped (see A in
# this is what stopped 5% of a real run's utterances from getting alt text).
ALT_ATTACH_TOLERANCE = 0.5


def assign_alt_words(alt_words: list[Word], utterances: list[Utterance]) -> dict[str, str]:
    """Assign every second-model word to exactly one canonical utterance's alt text.

    One global pass over all words (rather than the old per-utterance containment
    filter) is what lets a very short utterance still receive alt text: a word is
    scored against every utterance's interval, matched by containment first and by
    nearest boundary (within ALT_ATTACH_TOLERANCE) second, and placed in exactly one
    bucket — so no word is ever duplicated across, or permanently lost between, two
    adjacent utterances. Utterance boundaries themselves are never touched here: they
    come only from the first/canonical model.
    """
    if not utterances:
        return {}
    ordered = sorted(utterances, key=lambda item: item.source_start)
    starts = [item.source_start for item in ordered]
    buckets: dict[str, list[Word]] = {item.utterance_id: [] for item in ordered}
    for word in alt_words:
        center = (word.start + word.end) / 2
        # bisect finds the last utterance starting at or before the word's center;
        # checking its immediate neighbours too covers the (rare) case of canonical
        # intervals that touch or slightly overlap due to ASR timing jitter.
        anchor = bisect.bisect_right(starts, center) - 1
        candidates = [ordered[i] for i in (anchor - 1, anchor, anchor + 1) if 0 <= i < len(ordered)]
        containing = [item for item in candidates if item.source_start <= center <= item.source_end]
        target: Utterance | None
        if containing:
            # Narrowest containing interval wins; only relevant when canonical
            # intervals overlap, which keeps the choice deterministic even then.
            target = min(containing, key=lambda item: item.source_end - item.source_start)
        else:
            pool = candidates or ordered

            def boundary_distance(item: Utterance, center: float = center) -> float:
                if center < item.source_start:
                    return item.source_start - center
                return center - item.source_end

            nearest = min(pool, key=boundary_distance)
            target = nearest if boundary_distance(nearest) <= ALT_ATTACH_TOLERANCE else None
        if target is not None:
            buckets[target.utterance_id].append(word)
    result: dict[str, str] = {}
    for utterance_id, words in buckets.items():
        if not words:
            continue
        # Model sequence order, never re-sorted by time: Whisper emits overlapping
        # word spans (131 of 3847 words out of order on a real run), and sorting
        # by imprecise timestamps scrambles a transcription order that was right.
        # Timestamps decide WHICH utterance a word joins, not the order within it.
        text = "".join(item.text for item in words).strip()
        if text:
            result[utterance_id] = text
    return result


def mark_overlap(
    utterances: list[Utterance],
    overlap_regions: list[tuple[float, float]],
    diarization_offsets: dict[str, float],
) -> None:
    """Flag utterances whose diarization-clock interval overlaps a simultaneous-speech
    region by more than OVERLAP_MIN_SECONDS. Mutates in place: Utterance
    already carries a mutable `overlap` field the caller persists via canonical.json.

    `overlap_regions` are in the technical (concatenated) track's clock, exactly as
    written to diarization.json; converting each utterance's own source-local interval
    into that same clock (by adding its source's diarization_offset) is what makes the
    comparison meaningful, mirroring how word-to-speaker assignment already undoes this
    offset in run_job.
    """
    for utterance in utterances:
        base = diarization_offsets.get(utterance.source_file, 0.0)
        start, end = utterance.source_start + base, utterance.source_end + base
        utterance.overlap = any(
            min(end, region_end) - max(start, region_start) > OVERLAP_MIN_SECONDS
            for region_start, region_end in overlap_regions
        )


def overlap_seconds(start: float, end: float, segment: tuple[float, float, str]) -> float:
    return max(0.0, min(end, segment[1]) - max(start, segment[0]))


def assign_words(words: list[Word], exclusive: list[tuple[float, float, str]]) -> list[Word]:
    if not exclusive or not words:
        return words
    # Index turns by start with the maximum end of each subtree. This prunes
    # non-overlapping branches without assuming turns or words are disjoint.
    turns = sorted(enumerate(exclusive), key=lambda turn: turn[1][0])
    max_ends = [0.0] * len(turns)

    def index_ends(lo: int, hi: int) -> float:
        if lo == hi:
            return float("-inf")
        mid = (lo + hi) // 2
        max_ends[mid] = max(turns[mid][1][1], index_ends(lo, mid), index_ends(mid + 1, hi))
        return max_ends[mid]

    def candidates(lo: int, hi: int, start: float, end: float):
        if lo == hi:
            return
        mid = (lo + hi) // 2
        if max_ends[mid] <= start or turns[lo][1][0] >= end:
            return
        idx, item = turns[mid]
        if item[0] < end and item[1] > start:
            yield idx, item
        yield from candidates(lo, mid, start, end)
        yield from candidates(mid + 1, hi, start, end)

    index_ends(0, len(turns))
    # For the fallback, nearest to either boundary; preserve original turn order
    # for ties, including equal timestamps that belong to different speakers.
    boundary_turn: dict[float, int] = {}
    for index, (start, end, _) in enumerate(exclusive):
        boundary_turn.setdefault(start, index)
        boundary_turn.setdefault(end, index)
    boundaries = sorted(boundary_turn)
    for word in words:
        scored = (
            (overlap_seconds(word.start, word.end, item), -index)
            for index, item in candidates(0, len(turns), word.start, word.end)
        )
        score, neg_index = max(scored, default=(0.0, 0))
        if score > 0:
            word.speaker_id = exclusive[-neg_index][2]
            continue
        midpoint = (word.start + word.end) / 2
        position = bisect.bisect_left(boundaries, midpoint)
        distance, index = min(
            (abs(midpoint - point), boundary_turn[point])
            for point in boundaries[max(0, position - 1) : position + 1]
        )
        if distance <= 1.5:
            word.speaker_id = exclusive[index][2]
    return words


def make_utterances(
    words: list[Word],
    *,
    source_file: str,
    source_offset: float,
    speaker_names: dict[str, str],
    prefix: str,
) -> list[Utterance]:
    groups: list[list[Word]] = []
    for word in words:
        if (
            not groups
            or word.speaker_id != groups[-1][-1].speaker_id
            or word.start - groups[-1][-1].end > 0.8
        ):
            groups.append([word])
        else:
            groups[-1].append(word)
    output: list[Utterance] = []
    for index, group in enumerate(groups, 1):
        text = "".join(word.text for word in group).strip()
        probabilities = [word.probability for word in group if word.probability is not None]
        start = min(word.start for word in group)
        end = max(word.end for word in group)
        output.append(
            Utterance(
                utterance_id=f"{prefix}-u{index:05d}",
                source_file=source_file,
                source_start=start,
                source_end=end,
                meeting_start=source_offset + start,
                meeting_end=source_offset + end,
                speaker_id=group[0].speaker_id,
                speaker_name=speaker_names.get(group[0].speaker_id, group[0].speaker_id),
                raw_text=text,
                confidence={
                    "mean_word_probability": sum(probabilities) / len(probabilities)
                    if probabilities
                    else None
                },
            )
        )
    return output
