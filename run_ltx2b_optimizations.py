#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark all no-finetuning optimization combinations for LTX-Video 2B in Diffusers.

What the script does:
1. Builds a grid of optimization combinations.
2. Runs each combination in a separate Python process to avoid CUDA memory/state leakage.
3. Generates a 25-frame 512x512 MP4 for every successful run.
4. Saves a results table similar to the referenced optimization_of_diffusion_models project:
   ID, Method, Steps, Frames, Seconds, Speedup vs baseline, s/frame, CLIP sim,
   tSSIM, tLPIPS, Peak VRAM alloc/reserved, GPU temperature, status, error.

Default suite:
- quantization mode: none / fp8 layerwise weight casting / torchao int8 weight-only
- graph/hardware mode: off / torch.compile + TF32
- offload mode: none / group offload
- VAE tiling: off / on
- fewer steps: off / on
- low-resolution generation with final resize to 512x512: off / on

This is inference-only: no optimizer, no training loop, no gradients, no weight updates.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
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


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Combo:
    """One optimization combination to run."""

    quant: str                 # none | fp8 | int8
    compile: bool              # torch.compile + TF32
    offload: str               # none | group
    vae_tiling: bool           # VAE tiling
    fast_steps: bool           # baseline_steps -> fast_steps
    lowres: bool               # generate lowres, then resize final MP4 to target size

    @property
    def run_id(self) -> str:
        """Compact stable ID used in filenames and tables."""
        return (
            f"q-{self.quant}__"
            f"compile-{int(self.compile)}__"
            f"offload-{self.offload}__"
            f"vae-tiling-{int(self.vae_tiling)}__"
            f"fast-steps-{int(self.fast_steps)}__"
            f"lowres-{int(self.lowres)}"
        )

    @property
    def label(self) -> str:
        """Human-readable method label for the result table."""
        parts = []
        if self.quant == "none":
            parts.append("BF16 baseline weights")
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

        return " + ".join(parts)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Optional metrics
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Frame handling
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Optimization application
# -----------------------------------------------------------------------------


def apply_torchao_int8_weight_only(module: torch.nn.Module, group_size: int = 64) -> None:
    """Apply torchao INT8 weight-only quantization to Linear layers.

    This is best-effort because torchao APIs may differ across versions.
    """
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except Exception as exc:
        raise RuntimeError(
            "torchao is required for quant='int8'. Install it or remove 'int8' from --quant-modes."
        ) from exc

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

    # FP8 layerwise casting has to be applied while constructing the transformer.
    if combo.quant == "fp8":
        if args.single_file:
            raise RuntimeError("FP8 layerwise casting with --single-file is not supported by this script path.")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")

        transformer = AutoModel.from_pretrained(
            args.model_id,
            subfolder=args.transformer_subfolder,
            torch_dtype=dtype,
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=dtype,
        )

        pipe = PipelineCls.from_pretrained(
            args.model_id,
            transformer=transformer,
            torch_dtype=dtype,
        )

    else:
        if args.single_file:
            pipe = PipelineCls.from_single_file(args.single_file, torch_dtype=dtype)
        else:
            pipe = PipelineCls.from_pretrained(args.model_id, torch_dtype=dtype)

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


# -----------------------------------------------------------------------------
# Generation and worker
# -----------------------------------------------------------------------------


def call_pipeline(pipe: Any, args: argparse.Namespace, combo: Combo, steps: int, gen_w: int, gen_h: int) -> List[Image.Image]:
    """Run one LTX generation and return PIL frames."""
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    call_kwargs: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": gen_w,
        "height": gen_h,
        "num_frames": args.num_frames,
        "decode_timestep": args.decode_timestep,
        "decode_noise_scale": args.decode_noise_scale,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": "pil",
    }
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        image = image.resize((gen_w, gen_h), Image.Resampling.BICUBIC)
        call_kwargs["image"] = image
    # Most LTX Diffusers checkpoints accept num_inference_steps. Distilled versions may also accept custom timesteps.
    if combo.fast_steps and args.fast_timesteps:
        call_kwargs["timesteps"] = [float(x) if "." in x else int(x) for x in args.fast_timesteps.split(",")]
    else:
        call_kwargs["num_inference_steps"] = steps

    if args.guidance_rescale is not None:
        call_kwargs["guidance_rescale"] = args.guidance_rescale

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
        "Quant": combo.quant,
        "Compile": combo.compile,
        "Offload": combo.offload,
        "VAE tiling": combo.vae_tiling,
        "Fast steps": combo.fast_steps,
        "Lowres": combo.lowres,
        "Model": args.model_id if not args.single_file else args.single_file,
        "Output video": str(video_path),
        "Status": "failed",
        "Error": "",
    }

    pipe = None
    try:
        steps = args.fast_steps if combo.fast_steps else args.baseline_steps
        gen_w = args.lowres_width if combo.lowres else args.width
        gen_h = args.lowres_height if combo.lowres else args.height

        record.update(
            {
                "Steps": steps,
                "Frames": args.num_frames,
                "Generated width": gen_w,
                "Generated height": gen_h,
                "Output width": args.width,
                "Output height": args.height,
            }
        )

        clear_cuda()
        pipe = load_ltx_pipeline(args, combo)
        clear_cuda()

        torch.cuda.reset_peak_memory_stats()
        monitor = GpuTempMonitor(interval_s=args.temp_poll_interval)
        monitor.start()

        t0 = time.perf_counter()
        frames = call_pipeline(pipe, args, combo, steps=steps, gen_w=gen_w, gen_h=gen_h)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0

        temp_peak = monitor.stop()

        frames = ensure_exact_frame_count(frames, args.num_frames)
        if (gen_w, gen_h) != (args.width, args.height):
            frames = resize_frames(frames, args.width, args.height)

        # Export is intentionally outside generation timing, matching the referenced benchmark style.
        from diffusers.utils import export_to_video

        export_to_video(frames, str(video_path), fps=args.fps)

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        end_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        tssim = compute_temporal_ssim(frames)
        tlpips = compute_temporal_lpips(frames)
        clip_sim = compute_clip_text_image_similarity(frames, args.prompt)

        record.update(
            {
                "Seconds": round(seconds, 4),
                "s/frame": round(seconds / max(1, len(frames)), 4),
                "CLIP sim": None if clip_sim is None else round(float(clip_sim), 6),
                "tSSIM": None if tssim is None else round(float(tssim), 6),
                "tLPIPS": None if tlpips is None else round(float(tlpips), 6),
                "Peak VRAM alloc (MB)": round(peak_alloc_mb, 1),
                "Peak alloc (% total)": round(100.0 * peak_alloc_mb / total_mb, 2),
                "Peak VRAM reserved (MB)": round(peak_reserved_mb, 1),
                "VRAM end alloc (MB)": round(end_alloc_mb, 1),
                "GPU temp peak (C)": temp_peak,
                "Status": "ok",
                "Error": "",
            }
        )
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


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def build_combos(args: argparse.Namespace) -> List[Combo]:
    """Create the full optimization grid from CLI mode lists."""
    quant_modes = args.quant_modes
    compile_modes = [str_to_bool_mode(x) for x in args.compile_modes]
    offload_modes = args.offload_modes
    vae_tiling_modes = [str_to_bool_mode(x) for x in args.vae_tiling_modes]
    fast_steps_modes = [str_to_bool_mode(x) for x in args.fast_steps_modes]
    lowres_modes = [str_to_bool_mode(x) for x in args.lowres_modes]

    combos = [
        Combo(
            quant=quant,
            compile=compile_flag,
            offload=offload,
            vae_tiling=vae_tiling,
            fast_steps=fast_steps,
            lowres=lowres,
        )
        for quant, compile_flag, offload, vae_tiling, fast_steps, lowres in itertools.product(
            quant_modes,
            compile_modes,
            offload_modes,
            vae_tiling_modes,
            fast_steps_modes,
            lowres_modes,
        )
    ]

    if args.only_ids:
        wanted = set(args.only_ids)
        combos = [c for c in combos if c.run_id in wanted]

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    return combos


def write_tables(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Save CSV, Markdown and optional XLSX results tables."""
    if not records:
        return

    # Compute speedup after all rows are available. Baseline is the first successful no-optimization row.
    baseline_seconds: Optional[float] = None
    for row in records:
        if (
            row.get("Status") == "ok"
            and row.get("Quant") == "none"
            and row.get("Compile") is False
            and row.get("Offload") == "none"
            and row.get("VAE tiling") is False
            and row.get("Fast steps") is False
            and row.get("Lowres") is False
        ):
            baseline_seconds = float(row["Seconds"])
            break

    for row in records:
        seconds = row.get("Seconds")
        if baseline_seconds and seconds and row.get("Status") == "ok":
            row["Speedup vs baseline"] = f"{baseline_seconds / float(seconds):.2f}x"
        else:
            row["Speedup vs baseline"] = None

    columns = [
        "ID",
        "Method",
        "Steps",
        "Frames",
        "Generated width",
        "Generated height",
        "Output width",
        "Output height",
        "Seconds",
        "Speedup vs baseline",
        "s/frame",
        "CLIP sim",
        "tSSIM",
        "tLPIPS",
        "Peak VRAM alloc (MB)",
        "Peak alloc (% total)",
        "Peak VRAM reserved (MB)",
        "VRAM end alloc (MB)",
        "GPU temp peak (C)",
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
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
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
        "--seed",
        str(args.seed),
        "--decode-timestep",
        str(args.decode_timestep),
        "--decode-noise-scale",
        str(args.decode_noise_scale),
        "--guidance-scale",
        str(args.guidance_scale),
        "--compile-mode",
        args.compile_mode,
        "--int8-group-size",
        str(args.int8_group_size),
        "--temp-poll-interval",
        str(args.temp_poll_interval),
    ]
    if args.image_path:
        items += ["--image-path", args.image_path]
    if args.single_file:
        items += ["--single-file", args.single_file]
    if args.guidance_rescale is not None:
        items += ["--guidance-rescale", str(args.guidance_rescale)]
    if args.fast_timesteps:
        items += ["--fast-timesteps", args.fast_timesteps]
    if args.compute_ssim:
        items.append("--compute-ssim")
    if args.compute_lpips:
        items.append("--compute-lpips")
    if args.compute_clip:
        items.append("--compute-clip")
    if args.disable_progress_bar:
        items.append("--disable-progress-bar")
    return items


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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
    parser.add_argument("--transformer-subfolder", default="transformer", help="Transformer subfolder for FP8 layerwise casting.")
    parser.add_argument("--outdir", default="outputs_ltx2b", help="Root output directory.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Pipeline dtype.")

    # Generation settings.
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to input image for LTX image-to-video generation.",
    )
    parser.add_argument("--prompt", default=(
        "Powerful engines erupt with bright orange tongues of flame, exhaust gases shimmer behind, Earth rushes past below, clouds and coastlines slide quickly across the frame, the camera smoothly tracks the rocket while Earth rotates rapidly beneath, cinematic realistic motion, smooth video."
    ))
    parser.add_argument("--negative-prompt", default="static image, still frame, frozen, no motion, motionless, jitter, blurry, distorted, watermark")
    # parser.add_argument("--prompt", default=(
    #     "Yellow taxis driving through the crosswalk, pedestrians walking in different directions"
    # ))
    # parser.add_argument("--negative-prompt", default="static image, still frame, frozen video, no motion, motionless people, parked cars only, locked camera, slideshow, jitter, warping, distorted faces, duplicated people, melting cars, blurry, low quality, watermark, text artifacts, unnatural movement, flickering, broken perspective")
    # parser.add_argument("--prompt", default=(
    #     "A woman walks from left to right across the street, full body visible, clear stepping motion, arms swinging naturally, background moves with slight camera tracking, realistic video."
    # ))
    # parser.add_argument("--negative-prompt", default="still image, frozen, no motion, jitter, distorted, blurry, low quality")
    parser.add_argument("--width", type=int, default=512, help="Final video width.")
    parser.add_argument("--height", type=int, default=512, help="Final video height.")
    parser.add_argument("--lowres-width", type=int, default=384, help="Generation width for lowres combos.")
    parser.add_argument("--lowres-height", type=int, default=384, help="Generation height for lowres combos.")
    parser.add_argument("--num-frames", type=int, default=25, help="Final number of frames. 25 is 8*3+1, suitable for LTX-style temporal compression.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--baseline-steps", type=int, default=25)
    parser.add_argument("--fast-steps", type=int, default=8)
    parser.add_argument("--fast-timesteps", default=None, help="Optional comma-separated custom timesteps for fast/distilled runs, e.g. 1000,993,987,981,975,909,725,0.03")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-timestep", type=float, default=0.05)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Use 1.0 for guidance-distilled LTX variants; increase for non-distilled models.")
    parser.add_argument("--guidance-rescale", type=float, default=None)

    # Grid settings.
    parser.add_argument("--quant-modes", nargs="+", default=["none", "fp8", "int8"], choices=["none", "fp8", "int8"])
    parser.add_argument("--compile-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--offload-modes", nargs="+", default=["none", "group"], choices=["none", "group"])
    parser.add_argument("--vae-tiling-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--fast-steps-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--lowres-modes", nargs="+", default=["off", "on"], help="Allowed values: off/on.")
    parser.add_argument("--only-ids", nargs="*", default=None, help="Run only exact combo IDs.")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug limit: run only first N combos.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Optimization parameters.
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--int8-group-size", type=int, default=64)

    # Metrics.
    parser.add_argument("--compute-ssim", action="store_true", help="Compute temporal SSIM if scikit-image is installed.")
    parser.add_argument("--compute-lpips", action="store_true", help="Compute temporal LPIPS if lpips is installed.")
    parser.add_argument("--compute-clip", action="store_true", help="Compute CLIP text-image similarity; downloads CLIP model if missing.")
    parser.add_argument("--temp-poll-interval", type=float, default=0.5)
    parser.add_argument("--disable-progress-bar", action="store_true")

    args = parser.parse_args()

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
