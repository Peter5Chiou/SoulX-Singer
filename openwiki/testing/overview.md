---
type: Guide
title: Testing Guidance
description: How SoulX-Singer is exercised without a formal test suite — example end-to-end scripts, inline smoke tests, and what to validate when changing code.
tags: [testing, examples, verification]
---

# Testing Guidance

SoulX-Singer has **no formal automated test suite** (no `tests/` directory, no pytest config). Verification is manual and example-driven. This page explains the practical test surface and what to check when you change code.

## Primary test surface: example scripts

The `example/` directory contains the end-to-end scripts that double as the main regression checks:

- `example/preprocess.sh` — runs the full preprocessing pipeline on prompt and target audio (vocal separation, F0, VAD, lyric + note transcription, metadata generation). Exercises most of `preprocess/`.
- `example/infer.sh` — SVS inference (needs metadata already produced). Exercises `cli/inference.py` + `SoulXSinger`.
- `example/infer_svc.sh` — SVC inference (audio + F0 only). Exercises `cli/inference_svc.py` + `SoulXSingerSVC`.

These require the pretrained models in `pretrained_models/` and a CUDA device. Walk each script end-to-end after changes that touch the corresponding pipeline.

## Inline smoke tests

Several modules have `if __name__ == "__main__":` blocks that serve as lightweight smoke checks:

- `soulxsinger/utils/data_processor.py` — runs `DataProcessor.process` and prints the produced keys. Note it references `example/metadata/zh_prompt.json`, but the bundled examples live under `example/audio/` (e.g. `example/audio/zh_prompt.json`), so the smoke block needs a path fix to run as-is.
- `soulxsinger/models/modules/whisper_encoder.py` — encodes random audio and prints the output shape.
- These are useful for quickly sanity-checking a single module without a full inference run.

## What to validate when changing code

The biggest risk areas, tied to the domain concepts in [Domain Concepts](/openwiki/domain/overview.md):

1. **Alignment logic** (`DataProcessor.preprocess`, `mel2note`) — verify token boundaries are correct; misalignment degrades synthesis quality silently.
2. **F0 / pitch handling** (`f0_to_coarse`, pitch shift) — check bin ranges are respected and silence frames map to bin 0; out-of-range shifts break generation.
3. **Segment merging** — for SVS (`cli/inference.py`) make sure per-segment audio never overflows the merged buffer (see the `f177900` fix); for SVC (`soulxsinger_svc.py`) ensure overlap segments merge at the correct sample offsets.
4. **FP16 correctness** — confirm mel stays FP32 and autocast scopes are correct when enabling `--fp16`.
5. **Windows compatibility** — watch frame/sample index arithmetic (the `33ad03f` negative-dimensions fix and the integer-overflow guards in `preprocess/utils.py`).

## Recommended workflow

Because there is no CI test gate, prefer an explicit verification pass:

1. Run the relevant `example/*.sh` script on a short sample.
2. Confirm a `generated.wav` (and `metadata.json` where applicable) is produced with plausible length and no NaN/empty output.
3. For changes to alignment/pitch, listen to the output or compare F0/pitch-range logs; transcription tools print segment info you can inspect.
4. Keep the changes aligned with the latest design in `/openwiki/architecture/overview.md` and note any behavior change in the operations page.

## Backlog

A formal test suite (unit tests for `DataProcessor`, `midi_parser`, g2p, and the segmentation utilities) is not implemented. If added, it should target the pure-Python utilities first since they are deterministic and GPU-free.
