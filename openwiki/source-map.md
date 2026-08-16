---
type: Reference
title: Source Map
description: Navigation map of the SoulX-Singer repository — where models, utilities, preprocessing tools, CLI entrypoints, web UIs, and examples live.
tags: [source-map, navigation, repository]
---

# Source Map

A navigational map of the SoulX-Singer repository. Detailed explanations are in [Architecture Overview](/openwiki/architecture/overview.md) and [Domain Concepts](/openwiki/domain/overview.md).

## Model & inference core (`soulxsinger/`)

| Path | Purpose |
|------|---------|
| `models/soulxsinger.py` | `SoulXSinger` — SVS model (melody/score control) |
| `models/soulxsinger_svc.py` | `SoulXSingerSVC` — SVC model (Whisper + F0) |
| `models/modules/vocoder.py` | `Vocoder` — mel → waveform |
| `models/modules/decoder.py` | `CFMDecoder` wrapper |
| `models/modules/flow_matching.py` | `FlowMatchingTransformer` — diffusion/CFG sampling (adapted from Amphion) |
| `models/modules/llama.py` | `DiffLlama` — Llama-based diffusion estimator with adaptive RMSNorm |
| `models/modules/mel_transform.py` | `MelSpectrogramEncoder` — STFT mel extraction |
| `models/modules/whisper_encoder.py` | Frozen Whisper content encoder (SVC) |
| `models/modules/convnext.py` | `ConvNeXtV2Block` + `GRN` (SVS pre-flow) |
| `config/soulxsinger.yaml` | Model/audio/inference hyper-parameters |
| `utils/data_processor.py` | `DataProcessor` — metadata → model tensors |
| `utils/audio_utils.py` | `load_wav` (load + resample + mono) |
| `utils/pitch_utils.py` | F0↔LF0, mel/MIDI coarse quantization |
| `utils/file_utils.py` | Config loading, JSONL read/write |
| `utils/phoneme/phone_set.json` | Phoneme → index vocabulary |

## Preprocessing toolkit (`preprocess/`)

| Path | Purpose |
|------|---------|
| `pipeline.py` | `PreprocessPipeline` orchestrator |
| `utils.py` | `SegmentMetadata`, `convert_metadata`, `merge_short_segments` |
| `tools/__init__.py` | Stable import surface for the tools |
| `tools/f0_extraction.py` | `F0Extractor` (RMVPE) |
| `tools/vocal_detection.py` | `VocalDetector` (VAD segmentation) |
| `tools/vocal_separation/` | `VocalSeparator` (mel-band-roformer karaoke + dereverb) |
| `tools/lyric_transcription.py` | `LyricTranscriber` (ASR: Paraformer / Parakeet) |
| `tools/note_transcription/` | `NoteTranscriber` (ROSVOT-based) |
| `tools/midi_parser.py` | `MidiParser` — metadata ↔ MIDI conversion |
| `tools/g2p.py` | Grapheme-to-phoneme (Mandarin/Cantonese/English) |
| `tools/midi_editor/` | Standalone Vite/TypeScript MIDI editing app |
| `README.md` | Preprocessing docs (setup, usage, MIDI edit workflow) |

## Runnable surfaces

| Path | Purpose |
|------|---------|
| `cli/inference.py` | SVS inference entrypoint |
| `cli/inference_svc.py` | SVC inference entrypoint |
| `webui.py` | Gradio SVS UI |
| `webui_svc.py` | Gradio SVC UI |

## Examples & supporting

| Path | Purpose |
|------|---------|
| `example/preprocess.sh` | Runs preprocessing on sample audio |
| `example/infer.sh` | Runs SVS inference |
| `example/infer_svc.sh` | Runs SVC inference |
| `example/audio/` | Sample prompt/target audio, F0 arrays, metadata |
| `assets/` | Images, performance radar, technical report PDF |
| `pretrained_models/` | Download target for model weights (not committed) |
| `outputs/` | Default inference output directory |
| `.github/workflows/openwiki-update.yml` | Scheduled OpenWiki doc regeneration |
| `requirements.txt`, `preprocess/requirements.txt` | Python dependencies |

## Shared model-module dependency graph

Both models compose the same downstream modules but differ in their front-end conditioning; see [Architecture Overview](/openwiki/architecture/overview.md) for the flow diagram.
