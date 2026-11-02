"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DDS-VAE scene encoder configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DMLab uses d_z = 256. Paths are relative to the
repository root.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import Optional

from src.training.callbacks import TelemetryConfig


# ── Config ───────────────────────────────────────────
@dataclass
class DDSVAEConfig:
    """Scene encoder architecture for 128x128x3 maps."""

    latent_dim: int = 256            # d_z
    base_channels: int = 16          # Stage 1 UNet and scaling width
    bottleneck_channels: int = 16    # h in R^{16x16x16}
    vae_channels: int = 64           # mini-VAE widths 64 -> 128 -> 256
    keep_ratio: float = 0.2          # DDS fraction of C*H*W units kept per map
    gate_temperature: float = 1.0
    gate_noise: float = 0.1          # scale of the logistic gate noise
    exploration_prob: float = 0.1    # probability that a map keeps every unit


@dataclass
class Stage1Config:
    """Backbone reconstruction stage."""

    epochs: int = 64
    batch_size: int = 256
    lr_start: float = 1e-2
    lr_end: float = 1e-4
    weight_decay: float = 5e-5
    l1_weight: float = 1e-2


@dataclass
class Stage2Config:
    """Variational bottleneck stage on the frozen backbone."""

    epochs: int = 300
    batch_size: int = 256
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    weight_decay: float = 0.0
    beta: float = 0.5
    beta_start: float = 2.0          # KL weight at step 0, cosine-decayed to beta over beta_decay_epochs
    beta_decay_epochs: int = 50      # 0 keeps beta constant
    free_bits: float = 0.0           # per-dimension KL floor; 0 applies the plain KL
    perceptual_weight: float = 0.0   # fraction of reconstruction loss assigned to features; 0 skips decoder passes


@dataclass
class DDSVAETrainConfig:
    """Two-stage DDS-VAE training on a folder of top-down maps."""

    # ── Data ──────────────────────────────────────────
    map_dir: str = "data/dmlab_topdown_maps"
    val_fraction: float = 0.05
    flip_augment: bool = False       # horizontal and vertical map flips

    # ── Model and stages ──────────────────────────────
    model: DDSVAEConfig = field(default_factory=DDSVAEConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage1_checkpoint: Optional[str] = None  # skip Stage 1 and initialize the backbone from this checkpoint

    # ── Runtime ───────────────────────────────────────
    experiment_name: str = "dds_vae_dmlab"
    output_dir: str = "checkpoints"
    resume: Optional[str] = None     # "last" or a stage checkpoint path; settings must match the saved run
    seed: int = 101
    precision: str = "bf16-mixed"
    compile_mode: str = "max-autotune-no-cudagraphs"
    gradient_clip_val: float = 1.0

    # ── Telemetry ─────────────────────────────────────
    telemetry_enabled: bool = True
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # ── Notifications ─────────────────────────────────
    notify: bool = False  # Telegram fit reports; requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
