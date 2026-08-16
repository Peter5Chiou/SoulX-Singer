---
type: Playbook
title: Operations & Runbook
description: Environment setup, model downloads, running the preprocessing/SVS/SVC workflows, FP16 and GPU-memory notes, and known cross-platform gotchas for SoulX-Singer.
tags: [operations, runbook, setup, gpu, fp16, troubleshooting]
---

# Operations & Runbook

Practical guidance for setting up and running SoulX-Singer. The canonical setup steps live in `README.md` and `preprocess/README.md`; this page condenses them and adds operational notes grounded in the source and recent commits.

## Environment setup

1. Create a Python 3.10 conda environment (recommended by the README):
   ```bash
   conda create -n soulxsinger -y python=3.10
   conda activate soulxsinger
   ```
2. Install top-level dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   In mainland China the README suggests the Aliyun PyPI mirror for speed.
3. Preprocessing has its own dependency file (`preprocess/requirements.txt`); install from the `preprocess/` directory when you need the toolkit:
   ```bash
   cd preprocess && pip install -r requirements.txt
   ```
   Note the environment differences: `preprocess/requirements.txt` pins `numpy==2.2.6` and `torch==2.10.0`, while the root `requirements.txt` pins `numpy<2.0.0` and `torch==2.2.0`. The root also pins `setuptools==81.0.0` (commit `79e1f06`).

## Pretrained models

Model weights are **not vendored**; download them with `huggingface_hub`:

```sh
pip install -U huggingface_hub

# SoulX-Singer SVS + SVC models
hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer

# Preprocessing models (vocal separation, dereverb, RMVPE, ASR, ROSVOT)
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess
```

The default model paths assumed by the scripts:
- `pretrained_models/SoulX-Singer/model.pt` — SVS
- `pretrained_models/SoulX-Singer/model-svc.pt` — SVC
- Preprocess tools look under `pretrained_models/SoulX-Singer-Preprocess/` (e.g. `rmvpe/rmvpe.pt`, the `mel-band-roformer-karaoke` and `dereverb_mel_band_roformer` checkpoints, Paraformer/Parakeet models, `rosvot/model.pt` + `rosvot/rwbd/model.pt`).

If the checkpoint is missing, `build_model` raises `FileNotFoundError` with a message pointing at the `--model_path` flag (see `cli/inference.py`, `cli/inference_svc.py`).

## Running the workflows

See [Key Workflows](/openwiki/workflows/overview.md) for the full flow. Quick commands:

```bash
# 1. Preprocess (produces metadata + F0). midi_transcribe=True for SVS, False for SVC.
bash example/preprocess.sh

# 2. SVS inference
bash example/infer.sh

# 3. SVC inference
bash example/infer_svc.sh

# 4. Web UIs
python webui.py        # SVS
python webui_svc.py    # SVC
```

The CLI supports `--control melody|score`, `--auto_shift`, `--pitch_shift`, and `--fp16` (recorded for both `cli/inference.py` and `cli/inference_svc.py`).

## GPU memory & FP16 notes

FP16 inference was added to speed up and save memory on GPU. Important correctness detail is documented in code and align with recent commits (`6bc423f` "add fp16 support", `5c45496` "resolve GPU memory leak in webui_svc"):

- After `model.half()`, the **mel encoder is explicitly cast back to `float32`** (`model.mel.float()`), because mel extraction is done in FP32 to preserve accuracy. Both `build_model` and the `infer` paths assume prompt mel stays FP32.
- FP16 is only applied when running on CUDA (`use_fp16 and pt_wav.is_cuda` / `device.startswith("cuda")`).
- Model call is wrapped in `torch.amp.autocast(device_type="cuda", enabled=...)` (the `_autocast_if` helper).

## Cross-platform gotchas

- **Windows "negative dimensions are not allowed"**: fixed in commit `33ad03f` ("fix: resolve negative dimensions are not allowed on windows") touching `preprocess/tools/vocal_detection.py` and `preprocess/utils.py`. Pay attention to frame-length arithmetic on Windows (sample arrays and segment slicing) when touching preprocessing.
- **Segment merger shape mismatch in SVS inference**: fixed in `f177900` ("fix: avoid shape mismatch when merging segments in svs inference") in `cli/inference.py`. When merging per-segment audio into the final buffer, the code clips `gen_len = min(generated_audio.shape[0], generated_merged.shape[0] - start_sample_idx)` to avoid overflow.
- **Integer overflow safety**: `preprocess/utils.py` deliberately uses Python `int(...)` when computing sample indices `start_ms * sample_rate // 1000` to avoid 32-bit overflow on Windows.

## Notes for future maintainers

- When changing the WebUI or SVC memory behavior, note the dedicated memory-leak fixes (`5c45496` for `webui_svc.py`, `676a4b3` for `webui.py`). The established pattern is to call `gc.collect()` plus `torch.cuda.empty_cache()` after preprocess and after inference in the WebUI `AppState` methods; keep that pattern when adding new WebUI steps.
- `example/preprocess.sh` shows `midi_transcribe=False` → vocal+F0-only output; keep this consistent if the pipeline flags change.
- If you add features, update `README.md`, `preprocess/README.md`, and this wiki; the OpenWiki workflow (`/.github/workflows/openwiki-update.yml`) regenerates docs from source on a schedule.
