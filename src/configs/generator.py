"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation model (generator) configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The scene-code width d_z comes from the DDS-VAE checkpoint, d_s from the environment
transform, and the frame size from the rollout file. Paths are repository-relative.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import Optional

from src.training.callbacks import TelemetryConfig


# ── Config ───────────────────────────────────────────
@dataclass
class GeneratorConfig:
    """Renderer architecture; d_z, d_s, and the frame size come from the data."""

    num_bands: int = 10              # L Fourier bands per state dimension
    fourier_scale: float = 0.1       # band frequencies 2^k * pi * scale
    state_dropout: float = 0.1
    alpha_min: float = 0.05          # epsilon: scene coefficient alpha in [epsilon, 1 - epsilon]
    mlp_width: int = 256             # h_G
    mlp_blocks: int = 5              # residual MLP blocks of the stem
    base_channels: int = 512         # C_0 of the C_0 x H_0 x W_0 decoder input
    base_size: int = 4               # H_0 = W_0


@dataclass
class GeneratorOptimConfig:
    """AdamW with per-step cosine decay on the weighted MSE + LPIPS objective."""

    epochs: int = 29
    batch_size: int = 1024
    lr_start: float = 5e-4
    lr_end: float = 5e-5
    weight_decay: float = 1e-4       # weight matrices and kernels only; biases, norm gains, and alpha are not decayed
    mse_weight: float = 0.8
    lpips_weight: float = 0.2


@dataclass
class GeneratorTrainConfig:
    """Observation-model training on rollout frames with scene codes from a trained DDS-VAE."""

    # ── Data ──────────────────────────────────────────
    scene_encoder: str               # DDS-VAE Stage 2 checkpoint
    env: str = "dmlab"               # state transform: "dmlab"
    rollout_h5: str = "data/DMLabRolloutDataset.h5"
    val_fraction: float = 0.01       # held-out episodes
    num_workers: int = 8             # frame-reading processes per loader

    # ── Model and optimization ────────────────────────
    model: GeneratorConfig = field(default_factory=GeneratorConfig)
    optim: GeneratorOptimConfig = field(default_factory=GeneratorOptimConfig)

    # ── Runtime ───────────────────────────────────────
    experiment_name: str = "generator_dmlab"
    output_dir: str = "checkpoints"
    resume: Optional[str] = None     # "last" or a checkpoint path; settings must match the saved run
    seed: int = 101
    precision: str = "bf16-mixed"
    compile_mode: str = "max-autotune-no-cudagraphs"
    gradient_clip_val: Optional[float] = None  # no clipping: required for the fused AdamW

    # ── Telemetry ─────────────────────────────────────
    telemetry_enabled: bool = True
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # ── Notifications ─────────────────────────────────
    notify: bool = False  # Telegram fit reports; requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
