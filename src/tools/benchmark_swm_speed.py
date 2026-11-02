#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWM inference speed on GPU and CPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Measure how fast SWM runs as a simulator: frames per second of one environment, and the
number of parallel environments that each keep `target_fps`, on every device.

1) Encode one recorded scene map to its posterior mean z and read the episode's actions.
2) Per batch of B environments: warm up, then time `repeats` windows of free-running steps
   s_{t+1} = s_t + T_psi(s_t, z, a_t) and o_{t+1} = G(z, s_{t+1}); report the median window.
3) Double B until the per-environment rate falls below `target_fps`, then bisect the boundary.
4) Write every measurement and the device summary to a JSON file.

The scene code is computed once per scene, as in a rollout; observations stay on the device.
GPU runs cast both models to `gpu_dtype` with cuDNN autotuning; CPU runs use fp32 on all
`torch.get_num_threads()` threads. Run on an otherwise idle machine: concurrent jobs lower
every number.

Example:
    python -m src.tools.benchmark_swm_speed --transition checkpoints/transition_dmlab/version_0/last.ckpt \
        --generator checkpoints/generator_dmlab/version_0/last.ckpt --devices cuda cpu --output assets/swm_speed_dmlab.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pyrallis
import torch

from src.configs.dmlab import DMLabEnvConfig
from src.models.dds_vae import DDSVAE
from src.models.generator import Generator
from src.models.transition import TransitionModel
from src.tools.render_models import encode_scene


# ── Config ───────────────────────────────────────────
@dataclass
class SpeedConfig:
    """Checkpoints, recorded inputs, devices, timing, and output file."""

    transition: str                   # train_transition checkpoint
    generator: str                    # train_generator checkpoint on the same scene encoder
    rollout_h5: str = "data/DMLabRolloutDataset.h5"
    episode: int = 0                  # recorded scene map, start state, and action sequence (cycled)
    devices: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    gpu_dtype: str = "bfloat16"       # torch dtype of the GPU runs
    target_fps: float = 30.0          # real-time rate every parallel environment must keep
    warmup_seconds: float = 5.0       # untimed steps before every measurement (kernel selection, caches, clocks)
    measure_seconds: float = 4.0      # length of one timed window
    repeats: int = 3                  # timed windows per measurement; the median is reported, the spread exposes warm-up drift
    output: str = "assets/swm_speed_dmlab.json"


# ── Inputs / Models ──────────────────────────────────
def load_inputs(cfg):
    """Posterior mean z [1, d_z], normalized start state s_0 [1, d_s], and actions a [T, d_a] of the recorded episode (CPU, fp32)."""
    config = DMLabEnvConfig()
    with h5py.File(cfg.rollout_h5, "r") as f:
        episodes = f["episodes"]
        scene_map, raw, actions = episodes["topdown"][cfg.episode], episodes["state"][cfg.episode, :1], episodes["action"][cfg.episode]
    scene_encoder = DDSVAE.from_checkpoint(torch.load(cfg.transition, map_location="cpu")["hyper_parameters"]["scene_encoder"], "cpu")
    z = encode_scene(scene_encoder, scene_map, "cpu")
    return z, torch.from_numpy(config.normalize_state(raw)), torch.from_numpy(config.normalize_action(actions))


# ── Measurement ──────────────────────────────────────
@torch.inference_mode()
def measure(transition, generator, inputs, envs, device, dtype, cfg):
    """Per-environment frames per second of every timed window [repeats] for `envs` parallel free-running environments."""
    z, s, actions = (x.to(device, dtype) for x in inputs)
    z, s, hidden = z.expand(envs, -1), s.expand(envs, -1), None  # one scene for all environments [B, d_z], [B, d_s]
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def run(seconds):
        nonlocal s, hidden
        steps, start = 0, time.perf_counter()
        while True:
            for _ in range(10):  # check the clock every 10 steps
                a = actions[steps % len(actions)].expand(envs, -1)  # [B, d_a]
                s, _, hidden = transition.step(s, z, a, hidden)
                generator(z, s)  # o_{t+1} [B, 3, H, W], kept on the device
                steps += 1
            sync()
            elapsed = time.perf_counter() - start
            if elapsed >= seconds:
                return steps, elapsed

    run(cfg.warmup_seconds)
    return [steps / elapsed for steps, elapsed in (run(cfg.measure_seconds) for _ in range(cfg.repeats))]


def sweep(transition, generator, inputs, device, dtype, cfg):
    """Measurements over B: doubling from 1 until the per-environment rate drops below target, then bisection."""
    runs = {}

    def record(envs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        windows = measure(transition, generator, inputs, envs, device, dtype, cfg)
        env_fps = float(np.median(windows))
        memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else None
        runs[envs] = {"envs": envs, "env_fps": env_fps, "total_fps": envs * env_fps, "window_env_fps": windows, "memory_gib": memory}
        print(f"[INFO] {device} {envs:>6} envs: {env_fps:9.1f} FPS per env, {envs * env_fps:11.1f} frames/s, "
              f"windows {' '.join(f'{w:.1f}' for w in windows)}")
        return env_fps >= cfg.target_fps

    # 1) Doubling until the target breaks
    envs = 1
    while record(envs):
        envs *= 2
    low, high = envs // 2, envs  # low keeps the target (0 if one environment already misses it), high does not

    # 2) Bisection of the last doubling interval
    while high - low > max(1, low // 32):  # ~3% resolution
        middle = (low + high) // 2
        low, high = (middle, high) if record(middle) else (low, middle)
    return {"dtype": str(dtype).removeprefix("torch."), "single_env_fps": runs[1]["env_fps"], "max_envs": low,
            "runs": sorted(runs.values(), key=lambda run: run["envs"])}


# ── Main ─────────────────────────────────────────────
def main(cfg: SpeedConfig):
    """Benchmark SWM on every configured device and write the JSON summary."""
    torch.backends.cudnn.benchmark = True  # fixed shapes per measurement: let cuDNN pick the fastest kernels
    inputs = load_inputs(cfg)
    results = {"config": asdict(cfg), "cpu": next(line for line in open("/proc/cpuinfo") if line.startswith("model name")).split(":")[1].strip(),
               "cpu_threads": torch.get_num_threads(), "devices": {}}
    for device in cfg.devices:
        dtype = getattr(torch, cfg.gpu_dtype) if device == "cuda" else torch.float32
        transition = TransitionModel.from_checkpoint(cfg.transition, device).to(dtype)
        generator = Generator.from_checkpoint(cfg.generator, device).to(dtype)
        results["devices"][device] = sweep(transition, generator, inputs, device, dtype, cfg)
        if device == "cuda":
            results["gpu"] = torch.cuda.get_device_name()

    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    summary = ", ".join(f"{d}: {r['single_env_fps']:.0f} FPS single, {r['max_envs']} envs at {cfg.target_fps:g} FPS" for d, r in results["devices"].items())
    print(f"[INFO] {output}: {summary}")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=SpeedConfig))
