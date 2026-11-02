"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scene-map and rollout dataset configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paths are relative to the repository root. All generators default to base
seed 101.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

from src.configs.dmlab import DMLabEnvConfig

DMLAB_ENV = DMLabEnvConfig()

# ── Config ───────────────────────────────────────────
@dataclass
class DMLabMapDatasetConfig:
    """Generate DMLab scene maps m as seed-named RGB PNGs."""

    num_samples: int = 50_000
    seed: int = 101
    img_width: int = DMLAB_ENV.default_topdown_img_width
    img_height: int = DMLAB_ENV.default_topdown_img_height
    chunk_size: int = 256
    num_processes: int = max(1, mp.cpu_count() - 1)
    output_dir: str = "data/dmlab_topdown_maps"
    runfiles_path: Optional[str] = None  # runtime assets for a custom DMLab build

@dataclass
class DMLabRolloutDatasetConfig:
    """Record DMLab observations, raw states/actions, and one scene map per episode."""

    # ── Episodes ──────────────────────────────────────
    num_episodes: int = 20_000
    steps_per_episode: int = 500

    # ── Environment and output images ──────────────────
    img_width: int = 128
    img_height: int = 128
    topdown_width: int = 128
    topdown_height: int = 128
    out_width: int = 96
    out_height: int = 96

    # ── Environment and policy ─────────────────────────
    fps: int = 30
    step_repeat: int = 2
    action_repeat: int = 25
    policy: str = "balanced"  # Fast sweeping with stochastic trajectory perturbations.
    num_workers: int = max(1, mp.cpu_count() - 2)
    base_seed: int = 101

    # ── Output ─────────────────────────────────────────
    output_h5: str = "data/DMLabRolloutDataset.h5"
    tmp_dir: str = "data/dmlab_tmp"
