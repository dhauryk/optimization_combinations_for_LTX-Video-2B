
import argparse
import csv
import gc
import itertools
import inspect
import math
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch._dynamo as dynamo
import numpy as np
import torch
from PIL import Image



# Data structures



@dataclass(frozen=True)
class Combo:
    """One optimization combination to run."""

    text_encoder: str           # original | precompute | saved_embeds
    checkpoint: str             # base | distilled | distilled_fp8
    quant: str                  # none | fp8 | int8
    compile: bool               # torch.compile + TF32
    offload: str                # none | group
    vae_tiling: bool            # VAE tiling
    fast_steps: bool            # baseline_steps -> fast_steps
    lowres: bool                # generate lowres, then resize final MP4 to target size
    temporal_subsample: bool    # generate fewer frames and interpolate to target frame count

    @property
    def run_id(self) -> str:
        """Compact stable ID used in filenames and tables."""
        return (
            f"text-{self.text_encoder}__"
            f"ckpt-{self.checkpoint}__"
            f"q-{self.quant}__"
            f"compile-{int(self.compile)}__"
            f"offload-{self.offload}__"
            f"vae-tiling-{int(self.vae_tiling)}__"
            f"fast-steps-{int(self.fast_steps)}__"
            f"lowres-{int(self.lowres)}__"
            f"temporal-sub-{int(self.temporal_subsample)}"
        )

    @property
    def label(self) -> str:
        """Human-readable method label for the result table."""
        parts = []
        if self.text_encoder == "original":
            parts.append("original T5 text encoder")
        elif self.text_encoder == "precompute":
            parts.append("precomputed original T5 embeds + unload T5")
        elif self.text_encoder == "saved_embeds":
            parts.append("saved portrait prompt embeds + unload text encoder")

        if self.checkpoint == "base":
            parts.append("base checkpoint")
        elif self.checkpoint == "distilled":
            parts.append("pretrained distilled checkpoint")
        elif self.checkpoint == "distilled_fp8":
            parts.append("pretrained distilled FP8 checkpoint")

        if self.quant == "none":
            parts.append("BF16/selected dtype weights")
        elif self.quant == "fp8":
            parts.append("FP8 layerwise weight casting")
        elif self.quant == "int8":
            parts.append("torchao INT8 weight-only Linear")

        if self.compile:
            parts.append("torch.compile + TF32")
        if self.offload == "group":
            parts.append("group offload")
        if self.vae_tiling:
            parts.append("VAE tiling")
        if self.fast_steps:
            parts.append("fewer diffusion steps")
        if self.lowres:
            parts.append("lowres generation + resize to target")
        if self.temporal_subsample:
            parts.append("temporal subsampling + interpolation")

        return " + ".join(parts)



# Small utilities



def str_to_bool_mode(value: str) -> bool:
    """Convert CLI mode tokens like 'on/off' to bool."""
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean mode expected, got: {value}")


def get_torch_dtype(name: str) -> torch.dtype:
    """Map a readable dtype name to a torch dtype."""
    name = name.lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")



def normalize_checkpoint_mode(value: str) -> str:
    """Normalize checkpoint mode tokens for internal use."""
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "base": "base",
        "default": "base",
        "distilled": "distilled",
        "distill": "distilled",
        "distilled_fp8": "distilled_fp8",
        "fp8_distilled": "distilled_fp8",
    }
    if value not in aliases:
        raise argparse.ArgumentTypeError(f"Unsupported checkpoint mode: {value}")
    return aliases[value]


def is_distilled_checkpoint(checkpoint: str) -> bool:
    """Whether this combo uses official distilled weights."""
    return checkpoint in {"distilled", "distilled_fp8"}


def is_pretrained_fp8_checkpoint(checkpoint: str) -> bool:
    """Whether this combo uses official pre-quantized FP8 weights."""
    return checkpoint == "distilled_fp8"


def normalize_text_encoder_mode(value: str) -> str:
    """Normalize text-encoder mode aliases for grid construction."""
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "original": "original",
        "default": "original",
        "precompute": "precompute",
        "precomputed": "precompute",

        "saved_embeds": "saved_embeds",
        "saved": "saved_embeds",
        "cached_embeds": "saved_embeds",
        "portrait_embeds": "saved_embeds",
    }
    if value not in aliases:
        raise argparse.ArgumentTypeError(f"Unsupported text encoder mode: {value}")
    return aliases[value]


def parse_timestep_list(value: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated timesteps. Keeps ints as int-compatible floats."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    timesteps: List[float] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        timesteps.append(float(token))
    return timesteps or None


def hf_file_exists(repo_id: str, filename: str) -> bool:
    """Check if an HF model repo contains a file. Best effort; returns False on errors."""
    try:
        from huggingface_hub import list_repo_files
    except Exception:
        return False

    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        return filename in files
    except Exception:
        return False


def hf_download_file(repo_id: str, filename: str, cache_dir: Optional[str], local_files_only: bool) -> str:
    """Download an HF file or return its cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic checkpoint download. "
            "Install it or pass a local --distilled-single-file/--distilled-fp8-single-file."
        ) from exc

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": "model",
        "local_files_only": local_files_only,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return hf_hub_download(**kwargs)




def available_checkpoint_modes(args: argparse.Namespace) -> List[str]:
    """Return requested checkpoint modes that are actually available."""
    requested = [normalize_checkpoint_mode(x) for x in args.checkpoint_modes]
    available: List[str] = []

    for mode in requested:
        if mode == "base":
            available.append(mode)
            continue

        if mode == "distilled":
            if args.distilled_single_file or not args.auto_detect_hf_weights:
                available.append(mode)
                continue
            if hf_file_exists(args.distilled_hf_repo, args.distilled_checkpoint_filename):
                available.append(mode)
            else:
                print(
                    "Skipped checkpoint mode 'distilled': "
                    f"{args.distilled_hf_repo}/{args.distilled_checkpoint_filename} was not found."
                )
            continue

        if mode == "distilled_fp8":
            if args.distilled_fp8_single_file or not args.auto_detect_hf_weights:
                available.append(mode)
                continue
            if hf_file_exists(args.distilled_hf_repo, args.distilled_fp8_checkpoint_filename):
                available.append(mode)
            else:
                print(
                    "Skipped checkpoint mode 'distilled-fp8': "
                    f"{args.distilled_hf_repo}/{args.distilled_fp8_checkpoint_filename} was not found."
                )
            continue

    # Keep order while removing duplicates.
    deduped: List[str] = []
    for mode in available:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def resolve_checkpoint_source(args: argparse.Namespace, combo: Combo) -> Tuple[str, str]:
    """Resolve the concrete source for a checkpoint mode.

    Returns:
        ("pretrained", model_id) or ("single_file", local_or_remote_path)
    """
    if combo.checkpoint == "base":
        if args.single_file:
            return "single_file", args.single_file
        return "pretrained", args.model_id

    if combo.checkpoint == "distilled":
        if args.distilled_single_file:
            return "single_file", args.distilled_single_file
        path = hf_download_file(
            repo_id=args.distilled_hf_repo,
            filename=args.distilled_checkpoint_filename,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.hf_local_files_only,
        )
        return "single_file", path

    if combo.checkpoint == "distilled_fp8":
        if args.distilled_fp8_single_file:
            return "single_file", args.distilled_fp8_single_file
        path = hf_download_file(
            repo_id=args.distilled_hf_repo,
            filename=args.distilled_fp8_checkpoint_filename,
            cache_dir=args.hf_cache_dir,
            local_files_only=args.hf_local_files_only,
        )
        return "single_file", path

    raise ValueError(f"Unsupported checkpoint mode: {combo.checkpoint}")



def cuda_available_or_raise() -> None:
    """Fail early with a clear message if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. LTX-Video benchmarking is expected to run on an NVIDIA GPU.")


def clear_cuda() -> None:
    """Best-effort cleanup between stages inside a worker process."""
    gc.collect()
    dynamo.reset()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        torch.cuda.synchronize()
    time.sleep(15)


def run_nvidia_smi_temperature() -> Optional[int]:
    """Read the current GPU temperature via nvidia-smi, if available."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        first = out.strip().splitlines()[0]
        return int(first)
    except Exception:
        return None


class GpuTempMonitor:
    """Poll GPU temperature during generation and keep the maximum value."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background polling."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[int]:
        """Stop polling and return the observed peak temperature."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak

    def _loop(self) -> None:
        """Polling loop."""
        while not self._stop_event.is_set():
            temp = run_nvidia_smi_temperature()
            if temp is not None:
                self.peak = temp if self.peak is None else max(self.peak, temp)
            time.sleep(self.interval_s)


def collect_cuda_stage_metrics(prefix: str, total_mb: float, temp_peak: Optional[int]) -> Dict[str, Any]:
    """Collect prefixed CUDA memory and temperature metrics for one benchmark stage."""
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    end_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    return {
        f"{prefix} Peak VRAM alloc (MB)": round(peak_alloc_mb, 1),
        f"{prefix} Peak alloc (% total)": round(100.0 * peak_alloc_mb / total_mb, 2),
        f"{prefix} Peak VRAM reserved (MB)": round(peak_reserved_mb, 1),
        f"{prefix} VRAM end alloc (MB)": round(end_alloc_mb, 1),
        f"{prefix} GPU temp peak (C)": temp_peak,
    }



# Optional metrics



def compute_temporal_ssim(frames: Sequence[Image.Image]) -> Optional[float]:
    """Compute mean SSIM between neighboring frames. Returns None if skimage is missing."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None

    if len(frames) < 2:
        return None

    scores: List[float] = []
    for a, b in zip(frames[:-1], frames[1:]):
        arr_a = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
        arr_b = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
        scores.append(float(ssim(arr_a, arr_b, channel_axis=-1, data_range=1.0)))
    return float(np.mean(scores)) if scores else None


def compute_temporal_lpips(frames: Sequence[Image.Image], device: str = "cuda") -> Optional[float]:
    """Compute mean LPIPS between neighboring frames. Returns None if lpips is missing."""
    try:
        import lpips  # type: ignore
    except Exception:
        return None

    if len(frames) < 2:
        return None

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(device)

    scores: List[float] = []
    with torch.no_grad():
        for a, b in zip(frames[:-1], frames[1:]):
            score = loss_fn(pil_to_tensor(a), pil_to_tensor(b))
            scores.append(float(score.item()))
    return float(np.mean(scores)) if scores else None


def compute_clip_text_image_similarity(
    frames: Sequence[Image.Image],
    prompt: str,
    device: str = "cuda",
    max_frames: int = 8,
) -> Optional[float]:
    """Compute optional CLIP text-image similarity for a small sample of frames."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return None

    if not frames:
        return None

    if len(frames) <= max_frames:
        sampled = list(frames)
    else:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int).tolist()
        sampled = [frames[i] for i in idx]

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs = processor(
        text=[prompt] * len(sampled),
        images=sampled,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        sim = (img * txt).sum(dim=-1).mean().item()

    del model
    clear_cuda()
    return float(sim)



# Frame handling



def normalize_frames(raw_frames: Any) -> List[Image.Image]:
    """Convert Diffusers output into a list of PIL RGB frames."""
    # Diffusers LTX usually returns frames as List[PIL.Image] inside .frames[0].
    if isinstance(raw_frames, np.ndarray):
        raw_frames = list(raw_frames)

    frames: List[Image.Image] = []
    for frame in raw_frames:
        if isinstance(frame, Image.Image):
            frames.append(frame.convert("RGB"))
        elif isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr).convert("RGB"))
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)!r}")
    return frames


def ensure_exact_frame_count(frames: List[Image.Image], target_frames: int) -> List[Image.Image]:
    """Trim or pad with the last frame so the exported video has exactly target_frames."""
    if len(frames) == target_frames:
        return frames
    if len(frames) > target_frames:
        return frames[:target_frames]
    if not frames:
        raise RuntimeError("The pipeline returned zero frames.")
    return frames + [frames[-1].copy() for _ in range(target_frames - len(frames))]


def resize_frames(frames: List[Image.Image], width: int, height: int) -> List[Image.Image]:
    """Resize frames to the final output size."""
    return [frame.resize((width, height), Image.Resampling.BICUBIC) for frame in frames]




def ltx_compatible_frame_count_at_or_below(value: int) -> int:
    """Return a frame count compatible with 8k+1, at or below value when possible."""
    if value <= 1:
        return 1
    compatible = ((value - 1) // 8) * 8 + 1
    return max(9, compatible)


def compute_generation_frame_count(args: argparse.Namespace, combo: Combo) -> int:
    """Choose how many frames the diffusion model should generate before postprocessing."""
    if not combo.temporal_subsample:
        return args.num_frames

    if args.temporal_subsample_frames is not None:
        requested = max(1, int(args.temporal_subsample_frames))
        return min(requested, args.num_frames)

    raw = int(math.ceil(args.num_frames / max(1.0, float(args.temporal_subsample_factor))))
    return min(args.num_frames, ltx_compatible_frame_count_at_or_below(raw))


def resample_frames_to_count(frames: List[Image.Image], target_frames: int) -> List[Image.Image]:
    """CPU-side temporal interpolation/duplication to restore the requested frame count."""
    if len(frames) == target_frames:
        return frames
    if not frames:
        raise RuntimeError("The pipeline returned zero frames.")
    if target_frames <= 1:
        return [frames[0].copy()]

    positions = np.linspace(0, len(frames) - 1, target_frames)
    result: List[Image.Image] = []
    for pos in positions:
        lo = int(math.floor(float(pos)))
        hi = int(math.ceil(float(pos)))
        if lo == hi:
            result.append(frames[lo].copy())
        else:
            alpha = float(pos - lo)
            result.append(Image.blend(frames[lo].convert("RGB"), frames[hi].convert("RGB"), alpha))
    return result


def filter_pipeline_kwargs(pipe: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop optional kwargs that are not accepted by the installed Diffusers pipeline."""
    try:
        signature = inspect.signature(pipe.__call__)
    except Exception:
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def get_guidance_values(args: argparse.Namespace, combo: Combo) -> Tuple[float, Optional[float]]:
    """Return the guidance settings used by this combo."""
    guidance_scale = args.distilled_guidance_scale if is_distilled_checkpoint(combo.checkpoint) else args.guidance_scale
    guidance_rescale = args.distilled_guidance_rescale if is_distilled_checkpoint(combo.checkpoint) else args.guidance_rescale
    return float(guidance_scale), guidance_rescale


def unload_text_encoder(pipe: Any) -> None:
    """Remove text encoder/tokenizer/projection modules after prompt embeds are precomputed."""
    for attr in ("text_encoder", "tokenizer", "text_projection"):
        try:
            setattr(pipe, attr, None)
        except Exception:
            pass
    try:
        pipe.register_modules(text_encoder=None, tokenizer=None)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_saved_prompt_conditioning(
    pipe: Any,
    args: argparse.Namespace,
    combo: Combo,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load precomputed positive/negative prompt embeddings from disk and unload text encoder."""
    if not args.saved_prompt_embeds:
        raise RuntimeError(
            "--saved-prompt-embeds is required when text encoder mode is saved_embeds"
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    guidance_scale, _ = get_guidance_values(args, combo)
    do_cfg = guidance_scale > 1.0

    data = torch.load(args.saved_prompt_embeds, map_location="cpu")

    prompt_embeds = data["prompt_embeds"].to(device=device, dtype=dtype)
    prompt_attention_mask = data["prompt_attention_mask"].to(device=device)

    prompt_kwargs: Dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
    }

    if do_cfg:
        prompt_kwargs["negative_prompt_embeds"] = data["negative_prompt_embeds"].to(device=device, dtype=dtype)
        prompt_kwargs["negative_prompt_attention_mask"] = data["negative_prompt_attention_mask"].to(device=device)

    text_info = {
        "Text encoder mode": args.text_encoder_mode,
        "Prompt embeds precomputed": True,
        "Text encoder unloaded": True,
        "Prompt embed dim": int(prompt_embeds.shape[-1]),
        "Prompt tokens": int(prompt_embeds.shape[1]),
        "No text conditioning": False,
        "Saved prompt embeds": str(args.saved_prompt_embeds),
    }

    unload_text_encoder(pipe)
    return prompt_kwargs, text_info

def precompute_prompt_conditioning(
    pipe: Any,
    args: argparse.Namespace,
    combo: Combo,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Precompute prompt embeddings and optionally unload the text encoder before denoising."""
    if args.text_encoder_mode == "saved_embeds":
        return load_saved_prompt_conditioning(pipe, args, combo, dtype)

    if args.text_encoder_mode != "precompute":
        return {}, {
            "Text encoder mode": args.text_encoder_mode,
            "Prompt embeds precomputed": False,
            "Text encoder unloaded": False,
            "No text conditioning": False,
        }

    guidance_scale, _ = get_guidance_values(args, combo)
    do_cfg = guidance_scale > 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.inference_mode():
        encoded = pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
            device=device,
            dtype=dtype,
        )

    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encoded
    prompt_kwargs: Dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
    }
    if do_cfg:
        prompt_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        prompt_kwargs["negative_prompt_attention_mask"] = negative_prompt_attention_mask

    text_info = {
        "Text encoder mode": args.text_encoder_mode,
        "Prompt embeds precomputed": True,
        "Text encoder unloaded": True,
        "Prompt embed dim": int(prompt_embeds.shape[-1]),
        "Prompt tokens": int(prompt_embeds.shape[1]),
    }
    unload_text_encoder(pipe)
    return prompt_kwargs, text_info



# Optimization application



def apply_torchao_int8_weight_only(module: torch.nn.Module, group_size: int = 64) -> None:
    """Apply torchao INT8 weight-only quantization to Linear layers.

    This is best-effort because torchao APIs may differ across versions.
    """
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    config = Int8WeightOnlyConfig(group_size=group_size)

    def filter_fn(mod: torch.nn.Module, _fqn: Optional[str] = None) -> bool:
        return isinstance(mod, torch.nn.Linear)

    try:
        quantize_(module, config, filter_fn=filter_fn)
    except TypeError:
        # Some torchao versions call filter_fn with only one argument.
        quantize_(module, config, filter_fn=lambda mod: isinstance(mod, torch.nn.Linear))


def apply_group_offload_best_effort(pipe: Any) -> None:
    """Apply Diffusers group offloading to transformer, text encoder and VAE."""
    from diffusers.hooks import apply_group_offloading

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")

    # If torch.compile wrapped the transformer, the original module usually lives in _orig_mod.
    transformer_target = getattr(pipe.transformer, "_orig_mod", pipe.transformer)

    if hasattr(transformer_target, "enable_group_offload"):
        transformer_target.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
    else:
        raise RuntimeError("This transformer object does not expose enable_group_offload().")

    def apply_to_component(component: Any, **kwargs: Any) -> None:
        try:
            apply_group_offloading(component, offload_device=offload_device, **kwargs)
        except TypeError:
            apply_group_offloading(component, **kwargs)

    apply_to_component(
        pipe.text_encoder,
        onload_device=onload_device,
        offload_type="block_level",
        num_blocks_per_group=2,
    )
    apply_to_component(
        pipe.vae,
        onload_device=onload_device,
        offload_type="leaf_level",
    )


def load_ltx_pipeline(args: argparse.Namespace, combo: Combo) -> Any:
    """Load LTX pipeline and apply the selected optimization combination."""
    from diffusers import AutoModel, LTXPipeline, LTXImageToVideoPipeline

    dtype = get_torch_dtype(args.dtype)
    PipelineCls = LTXImageToVideoPipeline if args.image_path else LTXPipeline
    checkpoint_source, checkpoint_value = resolve_checkpoint_source(args, combo)

    # FP8 layerwise casting has to be applied while constructing the transformer.
    # This script path supports it for standard Diffusers repos only.
    if combo.quant == "fp8":
        if checkpoint_source != "pretrained":
            raise RuntimeError(
                "FP8 layerwise casting is only enabled for Diffusers folder checkpoints in this script. "
                "Use checkpoint='distilled_fp8' for official pre-quantized FP8 single-file weights."
            )
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")

        transformer = AutoModel.from_pretrained(
            checkpoint_value,
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=dtype,
        )

        pipe = PipelineCls.from_pretrained(
            checkpoint_value,
            transformer=transformer,
            torch_dtype=dtype,
        )
    else:
        if checkpoint_source == "single_file":
            from transformers import AutoTokenizer, T5EncoderModel

            text_encoder = T5EncoderModel.from_pretrained(
                args.model_id,
                subfolder="text_encoder",
                torch_dtype=dtype,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                subfolder="tokenizer",
            )

            pipe = PipelineCls.from_single_file(
                checkpoint_value,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                torch_dtype=dtype,
            )
        else:
            pipe = PipelineCls.from_pretrained(
                checkpoint_value,
                torch_dtype=dtype,
            )


    # INT8 is applied after loading.
    if combo.quant == "int8":
        apply_torchao_int8_weight_only(pipe.transformer, group_size=args.int8_group_size)

    # Memory optimization on VAE decode.
    if combo.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # Hardware/graph optimization.
    if combo.compile:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        pipe.transformer = torch.compile(
            pipe.transformer,
            mode=args.compile_mode,
            fullgraph=False,
        )

    # Device placement. For group offload we should not call pipe.to("cuda") for the whole pipeline.
    if combo.offload == "group":
        apply_group_offload_best_effort(pipe)
    else:
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    return pipe



# Generation and worker



def call_pipeline(
    pipe: Any,
    args: argparse.Namespace,
    combo: Combo,
    steps: int,
    gen_w: int,
    gen_h: int,
    gen_frames: int,
    prompt_conditioning_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Image.Image]:
    """Run one LTX generation and return PIL frames."""
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    guidance_scale, guidance_rescale = get_guidance_values(args, combo)

    prompt_conditioning_kwargs = prompt_conditioning_kwargs or {}
    call_kwargs: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": gen_w,
        "height": gen_h,
        "num_frames": gen_frames,
        "decode_timestep": args.decode_timestep,
        "decode_noise_scale": args.decode_noise_scale,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": "pil",
        "max_sequence_length": args.max_sequence_length,
    }
    call_kwargs.update(prompt_conditioning_kwargs)

    if args.pass_frame_rate:
        call_kwargs["frame_rate"] = args.fps

    if is_distilled_checkpoint(combo.checkpoint) and args.tone_map_compression_ratio is not None:
        call_kwargs["tone_map_compression_ratio"] = args.tone_map_compression_ratio

    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        image = image.resize((gen_w, gen_h), Image.Resampling.BICUBIC)
        call_kwargs["image"] = image

    # Distilled checkpoints have their own recommended timestep schedule.
    distilled_timesteps = parse_timestep_list(args.distilled_timesteps) if is_distilled_checkpoint(combo.checkpoint) else None
    fast_timesteps = parse_timestep_list(args.fast_timesteps) if combo.fast_steps else None

    if distilled_timesteps is not None:
        call_kwargs["timesteps"] = distilled_timesteps
    elif fast_timesteps is not None:
        call_kwargs["timesteps"] = fast_timesteps
    else:
        call_kwargs["num_inference_steps"] = steps

    if guidance_rescale is not None:
        call_kwargs["guidance_rescale"] = guidance_rescale

    call_kwargs = filter_pipeline_kwargs(pipe, call_kwargs)

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output = pipe(**call_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return normalize_frames(output.frames[0])


def run_worker(args: argparse.Namespace) -> int:
    """Worker entry point: run exactly one optimization combo and write result.json."""
    cuda_available_or_raise()
    combo = Combo(**json.loads(args.combo_json))
    # In grid mode the text-encoder choice belongs to Combo, so every worker
    # receives exactly one text-conditioning strategy. Keep args in sync because
    # the loading/precompute helpers read args.text_encoder_mode.
    args.text_encoder_mode = combo.text_encoder
    run_dir = Path(args.run_dir)
    videos_dir = run_dir / "videos"
    logs_dir = run_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    result_path = logs_dir / f"{combo.run_id}.json"
    video_path = videos_dir / f"{combo.run_id}.mp4"

    record: Dict[str, Any] = {
        "ID": combo.run_id,
        "Method": combo.label,
        "Checkpoint": combo.checkpoint,
        "Pretrained distilled": is_distilled_checkpoint(combo.checkpoint),
        "Pretrained FP8 checkpoint": is_pretrained_fp8_checkpoint(combo.checkpoint),
        "Quant": combo.quant,
        "Compile": combo.compile,
        "Offload": combo.offload,
        "VAE tiling": combo.vae_tiling,
        "Fast steps": combo.fast_steps,
        "Lowres": combo.lowres,
        "Temporal subsample": combo.temporal_subsample,
        "Text encoder mode": args.text_encoder_mode,
        "Prompt embeds precomputed": False,
        "Text encoder unloaded": False,
        "No text conditioning": False,
        "Max sequence length": args.max_sequence_length,
        "Model": args.model_id if not args.single_file else args.single_file,
        "Output video": str(video_path),
        "Status": "failed",
        "Error": "",
    }

    pipe = None
    try:
        steps = args.distilled_steps if is_distilled_checkpoint(combo.checkpoint) else (
            args.fast_steps if combo.fast_steps else args.baseline_steps
        )
        gen_w = args.lowres_width if combo.lowres else args.width
        gen_h = args.lowres_height if combo.lowres else args.height
        gen_frames = compute_generation_frame_count(args, combo)

        record.update(
            {
                "Steps": steps,
                "Frames": args.num_frames,
                "Generated frames": gen_frames,
                "Output frames": args.num_frames,
                "Generated width": gen_w,
                "Generated height": gen_h,
                "Output width": args.width,
                "Output height": args.height,
            }
        )

        clear_cuda()
        pipe = load_ltx_pipeline(args, combo)
        clear_cuda()

        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        text_seconds: Optional[float] = None
        text_temp_peak: Optional[int] = None
        torch.cuda.reset_peak_memory_stats()

        if args.text_encoder_mode != "original":
            text_monitor = GpuTempMonitor(interval_s=args.temp_poll_interval)
            text_monitor.start()
            text_t0 = time.perf_counter()
            try:
                prompt_conditioning_kwargs, text_encoder_record = precompute_prompt_conditioning(
                    pipe=pipe,
                    args=args,
                    combo=combo,
                    dtype=get_torch_dtype(args.dtype),
                )
                torch.cuda.synchronize()
                text_seconds = time.perf_counter() - text_t0
            finally:
                text_temp_peak = text_monitor.stop()

            record.update(text_encoder_record)
            record.update({"Text Seconds": round(text_seconds, 4)})
            record.update(collect_cuda_stage_metrics("Text", total_mb, text_temp_peak))
        else:
            prompt_conditioning_kwargs, text_encoder_record = precompute_prompt_conditioning(
                pipe=pipe,
                args=args,
                combo=combo,
                dtype=get_torch_dtype(args.dtype),
            )
            record.update(text_encoder_record)

        torch.cuda.reset_peak_memory_stats()
        diffusion_monitor = GpuTempMonitor(interval_s=args.temp_poll_interval)
        diffusion_monitor.start()

        diffusion_t0 = time.perf_counter()
        try:
            frames = call_pipeline(
                pipe,
                args,
                combo,
                steps=steps,
                gen_w=gen_w,
                gen_h=gen_h,
                gen_frames=gen_frames,
                prompt_conditioning_kwargs=prompt_conditioning_kwargs,
            )
            torch.cuda.synchronize()
            diffusion_seconds = time.perf_counter() - diffusion_t0
        finally:
            diffusion_temp_peak = diffusion_monitor.stop()

        frames = ensure_exact_frame_count(frames, gen_frames)
        if combo.temporal_subsample:
            frames = resample_frames_to_count(frames, args.num_frames)
        else:
            frames = ensure_exact_frame_count(frames, args.num_frames)

        if (gen_w, gen_h) != (args.width, args.height):
            frames = resize_frames(frames, args.width, args.height)

        # Export is intentionally outside generation timing, matching the referenced benchmark style.
        from diffusers.utils import export_to_video

        export_to_video(frames, str(video_path), fps=args.fps)

        total_seconds = diffusion_seconds + (text_seconds or 0.0)

        tssim = compute_temporal_ssim(frames)
        tlpips = compute_temporal_lpips(frames)
        clip_sim = compute_clip_text_image_similarity(frames, args.prompt)

        record.update(
            {
                "Diffusion Seconds": round(diffusion_seconds, 4),
                "Text Seconds": None if text_seconds is None else round(text_seconds, 4),
                "Total Seconds": round(total_seconds, 4),
                "Diffusion s/frame": round(diffusion_seconds / max(1, len(frames)), 4),
                "Total s/frame": round(total_seconds / max(1, len(frames)), 4),
                "CLIP sim": None if clip_sim is None else round(float(clip_sim), 6),
                "tSSIM": None if tssim is None else round(float(tssim), 6),
                "tLPIPS": None if tlpips is None else round(float(tlpips), 6),
                "Status": "ok",
                "Error": "",
            }
        )
        record.update(collect_cuda_stage_metrics("Diffusion", total_mb, diffusion_temp_peak))
        return_code = 0

    except Exception as exc:
        record["Status"] = "failed"
        record["Error"] = f"{type(exc).__name__}: {exc}"
        (logs_dir / f"{combo.run_id}.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return_code = 1

    finally:
        try:
            del pipe
        except Exception:
            pass
        clear_cuda()
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return return_code



# Main orchestration



def build_combos(args: argparse.Namespace) -> List[Combo]:
    """Create the full optimization grid from CLI mode lists."""
    text_encoder_modes = [normalize_text_encoder_mode(x) for x in args.text_encoder_modes]
    checkpoint_modes = available_checkpoint_modes(args)
    quant_modes = args.quant_modes
    compile_modes = [str_to_bool_mode(x) for x in args.compile_modes]
    offload_modes = args.offload_modes
    vae_tiling_modes = [str_to_bool_mode(x) for x in args.vae_tiling_modes]
    fast_steps_modes = [str_to_bool_mode(x) for x in args.fast_steps_modes]
    lowres_modes = [str_to_bool_mode(x) for x in args.lowres_modes]
    temporal_subsample_modes = [str_to_bool_mode(x) for x in args.temporal_subsample_modes]

    combos = [
        Combo(
            text_encoder=text_encoder,
            checkpoint=checkpoint,
            quant=quant,
            compile=compile_flag,
            offload=offload,
            vae_tiling=vae_tiling,
            fast_steps=fast_steps,
            lowres=lowres,
            temporal_subsample=temporal_subsample,
        )
        for text_encoder, checkpoint, quant, compile_flag, offload, vae_tiling, fast_steps, lowres, temporal_subsample
        in itertools.product(
            text_encoder_modes,
            checkpoint_modes,
            quant_modes,
            compile_modes,
            offload_modes,
            vae_tiling_modes,
            fast_steps_modes,
            lowres_modes,
            temporal_subsample_modes,
        )
    ]

    original_count = len(combos)
    filtered: List[Combo] = []
    skip_reasons: Dict[str, int] = {}

    def skip(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for c in combos:
        # torchao INT8 weight-only is currently incompatible with group offload.
        if c.quant == "int8" and c.offload == "group":
            skip("torchao INT8 weight-only + group offload")
            continue

        # Official FP8 checkpoints are already quantized; do not quantize them again.
        if is_pretrained_fp8_checkpoint(c.checkpoint) and c.quant != "none":
            skip("pretrained FP8 checkpoint + extra quantization")
            continue

        # The layerwise FP8 path in this script needs a Diffusers folder with transformer subfolder.
        if c.quant == "fp8" and c.checkpoint != "base":
            skip("FP8 layerwise casting for single-file checkpoint")
            continue
        if c.quant == "fp8" and args.single_file:
            skip("FP8 layerwise casting with --single-file")
            continue

        # Distilled checkpoints are designed for few-step inference, so avoid duplicate slow runs.
        if is_distilled_checkpoint(c.checkpoint) and not c.fast_steps:
            skip("distilled checkpoint without few-step mode")
            continue

        filtered.append(c)

    combos = filtered

    skipped_count = original_count - len(combos)
    if skipped_count:
        print(f"Skipped {skipped_count} unsupported or duplicate combos:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"  - {count}: {reason}")

    if args.only_ids:
        wanted = set(args.only_ids)
        combos = [c for c in combos if c.run_id in wanted]

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    return combos


def _float_or_none(value: Any) -> Optional[float]:
    """Convert a table value to float when possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_result_record_columns(row: Dict[str, Any]) -> None:
    """Populate the new prefixed metric columns for older per-run JSON files."""
    legacy_to_diffusion = {
        "Seconds": "Diffusion Seconds",
        "s/frame": "Diffusion s/frame",
        "Peak VRAM alloc (MB)": "Diffusion Peak VRAM alloc (MB)",
        "Peak alloc (% total)": "Diffusion Peak alloc (% total)",
        "Peak VRAM reserved (MB)": "Diffusion Peak VRAM reserved (MB)",
        "VRAM end alloc (MB)": "Diffusion VRAM end alloc (MB)",
        "GPU temp peak (C)": "Diffusion GPU temp peak (C)",
    }
    for old_name, new_name in legacy_to_diffusion.items():
        if new_name not in row and old_name in row:
            row[new_name] = row[old_name]

    text_seconds = _float_or_none(row.get("Text Seconds"))
    diffusion_seconds = _float_or_none(row.get("Diffusion Seconds"))

    if row.get("Total Seconds") is None and diffusion_seconds is not None:
        row["Total Seconds"] = round(diffusion_seconds + (text_seconds or 0.0), 4)

    total_seconds = _float_or_none(row.get("Total Seconds"))
    output_frames = _float_or_none(row.get("Output frames") or row.get("Frames"))

    if row.get("Total s/frame") is None and total_seconds is not None and output_frames:
        row["Total s/frame"] = round(total_seconds / max(1.0, output_frames), 4)


def write_tables(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Save CSV, Markdown and optional XLSX results tables."""
    if not records:
        return

    for row in records:
        normalize_result_record_columns(row)

    # Compute speedup after all rows are available. Baseline is the first successful no-optimization row.
    baseline_seconds: Optional[float] = None
    for row in records:
        if (
            row.get("Status") == "ok"
            and row.get("Checkpoint") == "base"
            and row.get("Quant") == "none"
            and row.get("Compile") is False
            and row.get("Offload") == "none"
            and row.get("VAE tiling") is False
            and row.get("Fast steps") is False
            and row.get("Lowres") is False
            and row.get("Temporal subsample") is False
            and row.get("Text encoder mode") in {None, "original"}
            and row.get("Total Seconds") is not None
        ):
            baseline_seconds = float(row["Total Seconds"])
            break

    for row in records:
        seconds = row.get("Total Seconds")
        if baseline_seconds and seconds and row.get("Status") == "ok":
            row["Speedup vs baseline"] = f"{baseline_seconds / float(seconds):.2f}x"
        else:
            row["Speedup vs baseline"] = None

    columns = [
        "ID",
        "Method",
        "Checkpoint",
        "Pretrained distilled",
        "Pretrained FP8 checkpoint",
        "Text encoder mode",
        "Prompt embeds precomputed",
        "Text encoder unloaded",
        "Prompt embed dim",
        "Prompt tokens",
        "No text conditioning",
        "Steps",
        "Frames",
        "Generated frames",
        "Output frames",
        "Generated width",
        "Generated height",
        "Output width",
        "Output height",
        "Text Seconds",
        "Diffusion Seconds",
        "Total Seconds",
        "Speedup vs baseline",
        "Diffusion s/frame",
        "Total s/frame",
        "CLIP sim",
        "tSSIM",
        "tLPIPS",
        "Text Peak VRAM alloc (MB)",
        "Text Peak alloc (% total)",
        "Text Peak VRAM reserved (MB)",
        "Text VRAM end alloc (MB)",
        "Text GPU temp peak (C)",
        "Diffusion Peak VRAM alloc (MB)",
        "Diffusion Peak alloc (% total)",
        "Diffusion Peak VRAM reserved (MB)",
        "Diffusion VRAM end alloc (MB)",
        "Diffusion GPU temp peak (C)",
        "Status",
        "Error",
        "Output video",
    ]

    csv_path = run_dir / "results_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
        df.to_markdown(run_dir / "results_metrics.md", index=False)
        df.to_excel(run_dir / "results_metrics.xlsx", index=False)
    except Exception:
        # Pandas/openpyxl/tabulate are optional. CSV is always saved.
        pass


def run_main(args: argparse.Namespace) -> int:
    """Main process: build combos, launch workers, aggregate tables."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)

    combos = build_combos(args)
    config_snapshot = {
        "args": vars(args),
        "combos": [asdict(c) for c in combos],
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Run dir: {run_dir}")
        print(f"Planned runs: {len(combos)}")
        for c in combos:
            print(c.run_id, "=>", c.label)
        return 0

    print(f"Run dir: {run_dir}")
    print(f"Planned runs: {len(combos)}")

    records: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos, start=1):
        result_json = run_dir / "logs" / f"{combo.run_id}.json"
        if args.skip_existing and result_json.exists():
            print(f"[{i}/{len(combos)}] skip existing: {combo.run_id}")
        else:
            print(f"[{i}/{len(combos)}] run: {combo.run_id}")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--combo-json",
                json.dumps(asdict(combo), ensure_ascii=False),
                "--run-dir",
                str(run_dir),
            ] + args_to_worker_cli(args)
            completed = subprocess.run(cmd)
            if completed.returncode != 0:
                print(f"    failed, see logs: {result_json}")

        if result_json.exists():
            records.append(json.loads(result_json.read_text(encoding="utf-8")))
            write_tables(run_dir, records)

    write_tables(run_dir, records)
    print(f"Saved table: {run_dir / 'results_metrics.csv'}")
    print(f"Saved videos: {run_dir / 'videos'}")
    return 0


def args_to_worker_cli(args: argparse.Namespace) -> List[str]:
    """Serialize generation/model args for worker subprocesses."""
    items: List[str] = [
        "--model-id",
        args.model_id,
        "--transformer-subfolder",
        args.transformer_subfolder,
        "--dtype",
        args.dtype,
        "--text-encoder-mode",
        args.text_encoder_mode,
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--saved-prompt-embeds",
        args.saved_prompt_embeds,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--lowres-width",
        str(args.lowres_width),
        "--lowres-height",
        str(args.lowres_height),
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--baseline-steps",
        str(args.baseline_steps),
        "--fast-steps",
        str(args.fast_steps),
        "--distilled-steps",
        str(args.distilled_steps),
        "--seed",
        str(args.seed),
        "--decode-timestep",
        str(args.decode_timestep),
        "--decode-noise-scale",
        str(args.decode_noise_scale),
        "--guidance-scale",
        str(args.guidance_scale),
        "--distilled-guidance-scale",
        str(args.distilled_guidance_scale),
        "--compile-mode",
        args.compile_mode,
        "--int8-group-size",
        str(args.int8_group_size),
        "--temp-poll-interval",
        str(args.temp_poll_interval),
        "--distilled-hf-repo",
        args.distilled_hf_repo,
        "--distilled-checkpoint-filename",
        args.distilled_checkpoint_filename,
        "--distilled-fp8-checkpoint-filename",
        args.distilled_fp8_checkpoint_filename,
        "--temporal-subsample-factor",
        str(args.temporal_subsample_factor),
    ]
    if args.image_path:
        items += ["--image-path", args.image_path]
    if args.single_file:
        items += ["--single-file", args.single_file]
    if args.distilled_single_file:
        items += ["--distilled-single-file", args.distilled_single_file]
    if args.distilled_fp8_single_file:
        items += ["--distilled-fp8-single-file", args.distilled_fp8_single_file]
    if args.hf_cache_dir:
        items += ["--hf-cache-dir", args.hf_cache_dir]
    if args.hf_local_files_only:
        items.append("--hf-local-files-only")
    if not args.auto_detect_hf_weights:
        items.append("--no-auto-detect-hf-weights")
    if args.guidance_rescale is not None:
        items += ["--guidance-rescale", str(args.guidance_rescale)]
    if args.distilled_guidance_rescale is not None:
        items += ["--distilled-guidance-rescale", str(args.distilled_guidance_rescale)]
    if args.tone_map_compression_ratio is not None:
        items += ["--tone-map-compression-ratio", str(args.tone_map_compression_ratio)]
    if args.fast_timesteps:
        items += ["--fast-timesteps", args.fast_timesteps]
    if args.distilled_timesteps:
        items += ["--distilled-timesteps", args.distilled_timesteps]
    if args.temporal_subsample_frames is not None:
        items += ["--temporal-subsample-frames", str(args.temporal_subsample_frames)]
    if args.pass_frame_rate:
        items.append("--pass-frame-rate")
    if args.disable_progress_bar:
        items.append("--disable-progress-bar")
    return items



# CLI



def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run no-finetuning optimization combinations for LTX-Video 2B and save videos/table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hidden worker controls.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--combo-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)

    # Model and output.
    parser.add_argument("--model-id", default="Lightricks/LTX-Video", help="Diffusers model id.")
    parser.add_argument("--single-file", default=None, help="Optional local/HF safetensors or GGUF single-file checkpoint.")
    parser.add_argument(
        "--checkpoint-modes",
        nargs="+",
        default=["base", "distilled", "distilled-fp8"],
        help="Checkpoint axis: base, distilled, distilled-fp8. Distilled modes are added only when weights are available.",
    )
    parser.add_argument("--distilled-hf-repo", default="Lightricks/LTX-Video", help="HF repo with official LTX single-file checkpoints.")
    parser.add_argument("--distilled-checkpoint-filename", default="ltxv-2b-0.9.8-distilled.safetensors")
    parser.add_argument("--distilled-fp8-checkpoint-filename", default="ltxv-2b-0.9.8-distilled-fp8.safetensors")
    parser.add_argument("--distilled-single-file", default=None, help="Optional local distilled checkpoint path/URL.")
    parser.add_argument("--distilled-fp8-single-file", default=None, help="Optional local distilled FP8 checkpoint path/URL.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache dir for automatic downloads.")
    parser.add_argument("--hf-local-files-only", action="store_true", help="Use only already cached HF files.")
    parser.add_argument(
        "--no-auto-detect-hf-weights",
        dest="auto_detect_hf_weights",
        action="store_false",
        help="Do not pre-check HF files; try requested checkpoint modes directly.",
    )
    parser.set_defaults(auto_detect_hf_weights=True)
    parser.add_argument("--transformer-subfolder", default="transformer", help="Transformer subfolder for FP8 layerwise casting.")
    parser.add_argument("--outdir", default="outputs_ltx2b", help="Root output directory.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Pipeline dtype.")
    parser.add_argument(
        "--text-encoder-mode",
        default="original",
        choices=["original", "precompute", "saved_embeds"],
        help=(
            "Single text conditioning mode used by worker subprocesses and for backward compatibility. "
            "For full benchmarks use --text-encoder-modes."
        ),
    )
    parser.add_argument(
        "--text-encoder-modes",
        nargs="+",
        default=["original", "precompute", "saved_embeds"],
        choices=["original", "precompute", "saved_embeds"],
        help=(
            "Text-conditioning grid axis. 'precompute' encodes prompt_embeds once per worker and unloads T5. "
            "'saved_embeds' loads precomputed portrait prompt embeddings from disk and unloads the text encoder."
        ),
    )

    # Generation settings.
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to input image for LTX image-to-video generation.",
    )
    # parser.add_argument("--prompt", default=(
    #     "Powerful engines erupt with bright orange tongues of flame, exhaust gases shimmer behind, Earth rushes past below, clouds and coastlines slide quickly across the frame, the camera smoothly tracks the rocket while Earth rotates rapidly beneath, cinematic realistic motion, smooth video."
    # ))
    # parser.add_argument("--negative-prompt", default="static image, still frame, frozen, no motion, motionless, jitter, blurry, distorted, watermark")
    # parser.add_argument("--prompt", default=(
    #     "Yellow taxis driving through the crosswalk, pedestrians walking in different directions"
    # ))
    # parser.add_argument("--negative-prompt", default="static image, still frame, frozen video, no motion, motionless people, parked cars only, locked camera, slideshow, jitter, warping, distorted faces, duplicated people, melting cars, blurry, low quality, watermark, text artifacts, unnatural movement, flickering, broken perspective")
    # parser.add_argument("--prompt", default=(
    #     "A woman walks from left to right across the street, full body visible, clear stepping motion, arms swinging naturally, background moves with slight camera tracking, realistic video."
    # ))
    # parser.add_argument("--negative-prompt", default="still image, frozen, no motion, jitter, distorted, blurry, low quality")
    # parser.add_argument("--prompt", default=(
    #     "Busy Times Square street, yellow taxis driving forward, police car moving slowly, pedestrians walking, LED billboards flickering, camera slowly pushes forward down the street, realistic smooth dynamic video."
    # ))
    # parser.add_argument("--negative-prompt", default="static image, still frame, frozen video, no motion, motionless people, parked cars only, locked camera, slideshow, jitter, warping, distorted cars, distorted buildings, blurry, low quality, watermark, text artifacts")
    
    # parser.add_argument("--prompt", default=(
    #     "A tulip field at sunset, flowers moving in the wind, clouds racing across the sky, sunlight flashing through the clouds, camera slowly pushes forward, cinematic realistic video."
    # ))
    # parser.add_argument("--negative-prompt", default="static image, still frame, frozen video, no motion, locked camera, jitter, warping, distorted flowers, blurry, low quality, watermark, text artifacts")

    parser.add_argument("--prompt", default=(
        "A close-up portrait of a blonde young woman, strong wind moves her long hair, she blinks and slightly turns her head, soft natural smile, background lights flicker, camera slowly pushes in, realistic smooth video, cinematic motion"
    ))
    parser.add_argument("--negative-prompt", default="static image, still frame, frozen video, no motion, distorted face, warped eyes, deformed hair, unnatural smile, flickering face, blurry, low quality, watermark, text artifacts")

    parser.add_argument("--width", type=int, default=512, help="Final video width.")
    parser.add_argument("--height", type=int, default=512, help="Final video height.")
    parser.add_argument("--lowres-width", type=int, default=384, help="Generation width for lowres combos.")
    parser.add_argument("--lowres-height", type=int, default=384, help="Generation height for lowres combos.")
    parser.add_argument("--num-frames", type=int, default=25, help="Final number of frames. 25 is 8*3+1, suitable for LTX-style temporal compression.")
    parser.add_argument("--max-sequence-length", type=int, default=128, help="Max T5 token length used for prompt encoding.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--baseline-steps", type=int, default=25)
    parser.add_argument("--fast-steps", type=int, default=8)
    parser.add_argument("--distilled-steps", type=int, default=8, help="Inference steps recorded for distilled checkpoints.")
    parser.add_argument(
        "--distilled-timesteps",
        default="",
        help="Optional custom timesteps for official distilled LTX checkpoints. Empty string disables.",
    )
    parser.add_argument("--fast-timesteps", default=None, help="Optional comma-separated custom timesteps for fast/distilled runs, e.g. 1000,993,987,981,975,909,725,0.03")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-timestep", type=float, default=0.05)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--distilled-guidance-scale", type=float, default=3.0)
    parser.add_argument("--distilled-guidance-rescale", type=float, default=0.7, help="Guidance rescale used for distilled checkpoints; use none by passing a negative value.")
    parser.add_argument("--tone-map-compression-ratio", type=float, default=0.6, help="Optional LTX 0.9.8 distilled tone mapping parameter, filtered out if unsupported.")
    parser.add_argument("--pass-frame-rate", action="store_true", help="Pass frame_rate=fps to pipelines that support it.")
    parser.add_argument("--guidance-rescale", type=float, default=None)

    # Grid settings.
    parser.add_argument("--quant-modes", nargs="+", default=["none", "fp8", "int8"], choices=["none", "fp8", "int8"])
    parser.add_argument("--compile-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--offload-modes", nargs="+", default=["none", "group"], choices=["none", "group"])
    parser.add_argument("--vae-tiling-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--fast-steps-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--lowres-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--temporal-subsample-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--temporal-subsample-factor", type=float, default=2.0, help="Approximate frame reduction factor for temporal subsampling.")
    parser.add_argument("--temporal-subsample-frames", type=int, default=None, help="Override generated frame count for temporal subsampling.")
    parser.add_argument("--only-ids", nargs="*", default=None, help="Run only exact combo IDs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug limit: run only first N combos.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Optimization parameters.
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--int8-group-size", type=int, default=64)

    # Metrics.
    parser.add_argument("--temp-poll-interval", type=float, default=0.5)
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--saved-prompt-embeds",
        default="portrait_prompt_embeds.pt",
        help="Path to torch .pt file with saved prompt_embeds, negative_prompt_embeds and attention masks.",
    )

    args = parser.parse_args()

    if args.distilled_guidance_rescale is not None and args.distilled_guidance_rescale < 0:
        args.distilled_guidance_rescale = None
    if args.tone_map_compression_ratio is not None and args.tone_map_compression_ratio < 0:
        args.tone_map_compression_ratio = None

    if args.num_frames != 25:
        print("Warning: user requested num_frames != 25; final video will follow --num-frames.", file=sys.stderr)
    if args.width != 512 or args.height != 512:
        print("Warning: user requested final size != 512x512; final video will follow --width/--height.", file=sys.stderr)

    return args


def main() -> int:
    """Program entry point."""
    args = parse_args()
    if args.worker:
        if not args.combo_json or not args.run_dir:
            raise SystemExit("--worker requires --combo-json and --run-dir")
        return run_worker(args)
    return run_main(args)


if __name__ == "__main__":
    main()
