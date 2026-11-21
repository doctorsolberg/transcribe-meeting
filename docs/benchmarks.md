# Historical benchmarks

These are historical observations on a few Russian recordings, not release guarantees or accuracy scores. The runs have not been repeated for the current version. Agreement measures similarity; without a reference transcript and speaker annotations, it cannot tell whether either system was right.

## Runtime

A 113.7-minute meeting split across two files, measured on an M1 Pro (MPS for local ASR). Times exclude the `readable` pass. Diarization times include the separate local overlap detection pass.

| Route | ASR | Diarization | Total through alignment |
|---|---:|---:|---:|
| Two GigaAM models + Groq, `foxnose` | 17.6 min | 9.6 min | 27.3 min |
| Deepgram ASR + Deepgram diarization | 0.9 min | 5.2 min | 6.2 min |
| `v3_rnnt` + `pyannote` | 12.3 min | 95.2 min | 107.6 min |

On a separate 35-minute recording, `multilingual_large_ctc` and `v3_rnnt` took 161 s and 242 s respectively; diarization took 73 s. On the longer meeting, the earlier scripts reported 92–95% speaker labeling agreement on sampled seconds with speech in both jobs and 79–91% text similarity. The current script uses corrected time boundaries, so its speaker scores need not match those historical values. The Deepgram diarization timing includes a separate local overlap pass; it is not the time for a second Deepgram upload.

## Preprocessing and chunk margins

On one 54-minute recording, agreement between two local models was 92.62% with no preprocessing, 92.24% with compression, 92.17% with high-pass filtering and pre-emphasis, and 92.01% with both. This does not establish how the filters behave on noisy material.

On the roughly two-hour recording, adding context margins to local ASR chunks changed agreement from 89.33% to 89.42%. That does not establish an accuracy gain; both models can make the same mistakes at a shared seam.

## Compare your own runs

Run these standard-library scripts from a checkout. They read existing `canonical.json` files (`compare_routes.py` also reads `manifest.json`) and do not run inference, read the recording, write to the jobs, or use the network. Use the same audio streams and parts in the same order. Compare unedited jobs when measuring model behavior so that manual transcript edits do not affect the results.

```bash
uv run --no-sync python scripts/model_agreement.py output/JOB
uv run --no-sync python scripts/compare_routes.py output/JOB_A output/JOB_B
```

- [Within one job](../scripts/model_agreement.py): for each stored ASR variant, sum the token matches from Python's `SequenceMatcher` across covered utterances and divide by the sum of the longer token count in each pair. Utterances with no usable variant text are excluded and coverage is reported separately; a high score on low coverage is not evidence of strong agreement overall.
- [Across routes](../scripts/compare_routes.py): compare the normalized canonical raw texts over the whole job (`SequenceMatcher` ratio: twice the matching token count divided by the combined token count). The heuristic can depend on job order and underweight common tokens. Both scripts split on whitespace, lowercase, fold `ё` to `е`, and strip common punctuation at word edges. They do not equate `15` with `пятнадцать` or measure word error rate. Empty against nonempty text scores 0%; two empty texts have no meaningful score (`N/A`). Duration and word rate use the end of the last utterance, not the recording duration.
- Speaker labeling uses the canonical utterance labels sampled at whole seconds. Spans include their start but not their end; later utterances in start-time order overwrite an earlier label at the same sample. Short spans can fall between samples. Only samples attributed to a speaker by **both** jobs count; `Speaker ?` is not a speaker label. The comparison does not score missed speech or simultaneous speaker labels. The score is the largest fraction of shared samples explained by a one-to-one speaker-label assignment; it does not identify people or measure diarization error rate (DER). The exact, deterministic bit-mask assignment avoids an extra dependency and is limited to 16 participating labels on the smaller side. No shared attributed speech produces `N/A`, not 0% agreement.
