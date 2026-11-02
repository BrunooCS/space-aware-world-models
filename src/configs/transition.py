"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transition model configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The scene-code width d_z comes from the DDS-VAE checkpoint; state, action, and
increment dimensions and loss weights come from the environment transforms.
Paths are relative to the repository root.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import Optional

from src.training.callbacks import TelemetryConfig


# ── Config ───────────────────────────────────────────
@dataclass
class TransitionConfig:
    """Transition network; d_z, d_s, d_a, and d_Delta come from the data."""

    num_bands: int = 10              # L Fourier bands per state dimension
    fourier_scale: float = 0.1       # band frequencies 2^k * pi * scale
    hidden_dim: int = 384            # h_T: input projection and LSTM width
    num_layers: int = 3


@dataclass
class TransitionOptimConfig:
    """AdamW with per-step cosine decay on teacher-forced state windows."""

    epochs: int = 64
    batch_size: int = 32
    sequence_length: int = 200       # states per window
    stride: int = 20                 # window starts within an episode; validation windows use 4 x stride
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    weight_decay: float = 1e-5


@dataclass
class TransitionTrainConfig:
    """Transition-model training on rollout state sequences with scene codes from a trained DDS-VAE."""

    # ── Data ──────────────────────────────────────────
    scene_encoder: str               # DDS-VAE Stage 2 checkpoint
    env: str = "dmlab"               # state, action, and increment transforms: "dmlab"
    rollout_h5: str = "data/DMLabRolloutDataset.h5"
    val_fraction: float = 0.05       # held-out episodes

    # ── Model and optimization ────────────────────────
    model: TransitionConfig = field(default_factory=TransitionConfig)
    optim: TransitionOptimConfig = field(default_factory=TransitionOptimConfig)

    # ── Runtime ───────────────────────────────────────
    experiment_name: str = "transition_dmlab"
    output_dir: str = "checkpoints"
    resume: Optional[str] = None     # "last" or a checkpoint path; settings must match the saved run
    seed: int = 101
    precision: str = "32-true"       # limit accumulated rollout error; TF32 is also disabled
    compile_mode: str = "max-autotune-no-cudagraphs"
    gradient_clip_val: float = 1.0

    # ── Telemetry ─────────────────────────────────────
    telemetry_enabled: bool = True
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # ── Notifications ─────────────────────────────────
    notify: bool = False  # Telegram fit reports; requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
