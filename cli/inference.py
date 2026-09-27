import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
from collections import OrderedDict
from omegaconf import DictConfig

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor


def _rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio) + 1e-8))


def _representative_rms(audio, sample_rate):
    window = max(1, int(0.5 * sample_rate))
    values = [_rms(audio[i:i + window]) for i in range(0, len(audio), window)]
    active = [value for value in values if value > 1e-4]
    return float(np.median(active)) if active else 0.0


def _match_segment_gain(audio, reference_rms, sample_rate):
    """Match a segment to a fixed reference without normalizing silence."""
    if audio.size == 0 or reference_rms <= 1e-4:
        return audio

    current_rms = _representative_rms(audio, sample_rate)
    if current_rms <= 1e-4:
        return audio

    gain = float(np.clip(reference_rms / current_rms, 0.5, 2.0))
    return audio * gain


def _stabilize_local_loudness(audio, sample_rate):
    """Smooth short-term vocal level drops while leaving rests alone."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    window = max(1, int(0.4 * sample_rate))
    hop = max(1, int(0.1 * sample_rate))
    if audio.size < window:
        return audio

    starts = np.arange(0, max(1, audio.size - window + 1), hop)
    if starts[-1] != max(0, audio.size - window):
        starts = np.append(starts, max(0, audio.size - window))

    rms_values = np.array([_rms(audio[start:start + window]) for start in starts], dtype=np.float32)
    non_silent = rms_values[rms_values > 1e-4]
    if non_silent.size == 0:
        return audio

    rough_target = float(np.percentile(non_silent, 60))
    active_threshold = max(0.008, rough_target * 0.16)
    active = rms_values > active_threshold
    if int(active.sum()) < 3:
        return audio

    target_rms = float(np.percentile(rms_values[active], 55))
    if target_rms <= 1e-4:
        return audio

    gains = np.ones_like(rms_values)
    gains[active] = np.power(target_rms / np.maximum(rms_values[active], 1e-4), 0.85)
    gains = np.clip(gains, 0.55, 2.8)

    smooth_frames = max(1, int(1.0 / 0.1))
    if smooth_frames > 1 and gains.size > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        gains = np.convolve(np.pad(gains, (smooth_frames // 2, smooth_frames - 1 - smooth_frames // 2), mode="edge"), kernel, mode="valid")

    centers = starts + window // 2
    sample_positions = np.arange(audio.size)
    sample_gains = np.interp(sample_positions, centers, gains, left=gains[0], right=gains[-1]).astype(np.float32)
    stabilized = audio * sample_gains

    peak = float(np.max(np.abs(stabilized)))
    if peak > 0.98:
        stabilized *= 0.98 / peak
    return stabilized.astype(np.float32)


def _crossfade_into(output, start, audio, sample_rate, fade_seconds=0.08):
    """Write audio at start while smoothing a segment boundary."""
    end = min(start + len(audio), len(output))
    audio = audio[: max(0, end - start)]
    if len(audio) == 0:
        return

    fade_len = min(
        int(fade_seconds * sample_rate),
        start,
        len(audio),
    )
    if fade_len <= 0:
        output[start:end] = audio
        return

    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    output[start - fade_len:start] = (
        output[start - fade_len:start] * fade_out
        + audio[:fade_len] * fade_in
    )
    output[start:end] = audio


def build_model(
    model_path: str,
    config: DictConfig,
    device: str = "cuda",
    use_fp16: bool = False,
):
    """
    Build the model from the pre-trained model path and model configuration.

    Args:
        model_path (str): Path to the checkpoint file.
        config (DictConfig): Model configuration.
        device (str, optional): Device to use. Defaults to "cuda".
        use_fp16 (bool, optional): If True and device is CUDA, convert model to FP16 after load. Defaults to False.

    Returns:
        SoulXSinger: The initialized model.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Please download the pretrained model and place it at the path, or set --model_path."
        )
    model = SoulXSinger(config).to(device)
    print("Model initialized.")
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {model_path} has no 'state_dict' key. "
            "Expected a checkpoint saved with model.state_dict()."
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    if use_fp16 and ((isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and getattr(device, "type", None) == "cuda")):
        model.half()
        model.mel.float()
        print("Model converted to FP16 (mel kept in FP32).")
    model.eval()
    model.to(device)
    print("Model checkpoint loaded.")

    return model


def process(args, config, model: torch.nn.Module):
    """Run the full inference pipeline given a data_processor and model.
    """
    if args.control not in ("melody", "score"):
        raise ValueError(f"control must be 'melody' or 'score', got: {args.control}")

    print(f"prompt_metadata_path: {args.prompt_metadata_path}")
    print(f"target_metadata_path: {args.target_metadata_path}")

    os.makedirs(args.save_dir, exist_ok=True)
    data_processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=args.phoneset_path,
        device=args.device,
    )

    with open(args.prompt_metadata_path, "r", encoding="utf-8") as f:
        prompt_meta_list = json.load(f)
    if not prompt_meta_list:
        raise ValueError("Prompt metadata is empty. Please run preprocess on prompt audio first.")
    prompt_meta = prompt_meta_list[0]  # load the first segment as the prompt
    with open(args.target_metadata_path, "r", encoding="utf-8") as f:
        target_meta_list = json.load(f)
    infer_prompt_data = data_processor.process(prompt_meta, args.prompt_wav_path)

    assert len(target_meta_list) > 0, "No target segments found in the target metadata."
    generated_len = int(target_meta_list[-1]["time"][1] / 1000 * config.audio.sample_rate)
    generated_merged = np.zeros(generated_len, dtype=np.float32)
    reference_rms = 0.0

    for idx, target_meta in enumerate(
        tqdm(target_meta_list, total=len(target_meta_list), desc="Inferring segments"),
    ):
        start_sample_idx = int(target_meta["time"][0] / 1000 * config.audio.sample_rate)
        end_sample_idx = int(target_meta["time"][1] / 1000 * config.audio.sample_rate)
        infer_target_data = data_processor.process(target_meta, None)

        infer_data = {
            "prompt": infer_prompt_data,
            "target": infer_target_data,
        }

        with torch.no_grad():
            generated_audio = model.infer(
                infer_data,
                auto_shift=args.auto_shift,
                pitch_shift=args.pitch_shift,
                n_steps=getattr(args, "n_steps", config.infer.n_steps),
                cfg=getattr(args, "cfg", config.infer.cfg),
                control=args.control,
                use_fp16=args.use_fp16,
            )

        generated_audio = generated_audio.squeeze().cpu().numpy().astype(np.float32)
        gen_len = min(generated_audio.shape[0], generated_merged.shape[0] - start_sample_idx)
        if gen_len <= 0:
            continue

        generated_audio = generated_audio[:gen_len]
        if reference_rms <= 1e-4:
            reference_rms = _representative_rms(
                generated_audio,
                config.audio.sample_rate,
            )
        generated_audio = _match_segment_gain(
            generated_audio,
            reference_rms,
            config.audio.sample_rate,
        )
        _crossfade_into(
            generated_merged,
            start_sample_idx,
            generated_audio,
            config.audio.sample_rate,
        )

    merged_path = os.path.join(args.save_dir, "generated.wav")
    generated_merged = _stabilize_local_loudness(
        generated_merged,
        config.audio.sample_rate,
    )
    sf.write(merged_path, generated_merged, 24000)
    print(f"Generated audio saved to {merged_path}")


def main(args, config):
    model = build_model(
        model_path=args.model_path,
        config=config,
        device=args.device,
        use_fp16=getattr(args, "use_fp16", False),
    )
    process(args, config, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default='pretrained_models/soulx-singer/model.pt')
    parser.add_argument("--config", type=str, default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument("--prompt_wav_path", type=str, default='example/audio/zh_prompt.wav')
    parser.add_argument("--prompt_metadata_path", type=str, default='example/metadata/zh_prompt.json')
    parser.add_argument("--target_metadata_path", type=str, default='example/metadata/zh_target.json')
    parser.add_argument("--phoneset_path", type=str, default='soulxsinger/utils/phoneme/phone_set.json')
    parser.add_argument("--save_dir", type=str, default='outputs')
    parser.add_argument("--auto_shift", action="store_true")
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument(
        "--control",
        type=str,
        default="melody",
        choices=["melody", "score"],
        help="Control mode: melody or score only",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 inference (faster on GPU)",
    )
    args = parser.parse_args()
    args.use_fp16 = args.fp16

    config = load_config(args.config)
    main(args, config)
