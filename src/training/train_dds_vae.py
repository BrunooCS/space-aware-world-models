"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two-stage DDS-VAE scene encoder training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train q_phi on top-down maps and save checkpoints, metrics, and previews under
checkpoints/<experiment_name>/version_k/stage{1,2}/.

1) Split maps into training and validation subsets.
2) Train the Stage 1 backbone with MSE + L1 map reconstruction.
3) Freeze its best weights in eval mode and train the mini-VAE with summed
   bottleneck MSE + beta * KL, where beta decays from 2.0 to 0.5 (cosine) over
   the first 50 epochs; free bits and an upscaling-feature reconstruction term
   are optional.

The best Stage 2 checkpoint (lowest val/loss) contains both stages.
Validation uses the final beta throughout. Resuming restores
weights, optimizer, schedule, and epoch; settings, including both stage epoch
budgets, must match the saved run. `--resume last` selects its latest saved stage.

Examples:
    python -m src.training.train_dds_vae
    python -m src.training.train_dds_vae --stage1_checkpoint <stage1.ckpt>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
from dataclasses import asdict

import lightning as L
import pyrallis
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.configs.dds_vae import DDSVAETrainConfig
from src.datasets.maps import batch_sampler, split_maps
from src.models.dds_vae import DDSVAE, normalize_maps, stage1_loss, stage2_loss
from src.training.utils import cosine_schedule, fit, log_preview, run_directory

# summary.png panels per stage: (title, metric, y-scale); None is the learning rate
PANELS = {1: [("loss", "loss", "log"), ("MSE", "mse", "log"), ("L1", "l1", "log"), ("learning rate", None, "log")],
          2: [("loss", "loss", "log"), ("bottleneck reconstruction", "recon", "log"), ("perceptual", "perceptual", "log"),
              ("KL (free bits)", "kl_free_bits", "log"), ("active dimensions", "active_dims", "linear")]}


# ── Model ────────────────────────────────────────────
class DDSVAEModule(L.LightningModule):
    """Train the backbone in Stage 1 or the mini-VAE on the frozen backbone in Stage 2."""

    def __init__(self, model: dict, optim: dict, stage: int, compile_mode: str):
        super().__init__()
        self.save_hyperparameters()
        self.model = DDSVAE(**model).to(memory_format=torch.channels_last)
        self.stage, self.optim = stage, optim
        self.loss_weights = ({"l1_weight": optim["l1_weight"]} if stage == 1 else
                             {key: optim[key] for key in ("free_bits", "perceptual_weight")})
        loss = stage1_loss if stage == 1 else stage2_loss
        self.loss = torch.compile(loss, mode=compile_mode)
        self.frozen().requires_grad_(False).eval()
        self.preview_maps = None

    def frozen(self):
        """Return the component excluded from this stage's updates."""
        return self.model.bottleneck if self.stage == 1 else self.model.backbone

    def train(self, mode=True):
        super().train(mode)
        self.frozen().eval()  # fixed BN statistics and deterministic DDS gates for Stage 2 targets
        return self

    def on_after_batch_transfer(self, maps, dataloader_idx):
        return normalize_maps(maps)  # uint8 transfer, normalization on device

    def weights(self, training):
        """Return step-specific loss weights; validation uses the final KL weight."""
        if self.stage == 1:
            return self.loss_weights
        optim, decay_steps = self.optim, self.optim["beta_decay_epochs"] * self.trainer.num_training_batches
        progress = min(self.global_step / decay_steps, 1.0) if training and decay_steps else 1.0
        beta = optim["beta"] + (optim["beta_start"] - optim["beta"]) * 0.5 * (1 + math.cos(math.pi * progress))
        return {**self.loss_weights, "beta": torch.tensor(beta, device=self.device)}

    def training_step(self, maps, batch_idx):
        weights = self.weights(training=True)
        loss, metrics = self.loss(self.model, maps, **weights)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(maps))
        self.log_dict({f"train/{key}": value for key, value in metrics.items()}, batch_size=len(maps))
        if self.stage == 2:
            self.log("train/beta", weights["beta"], batch_size=len(maps))
        return loss

    def validation_step(self, maps, batch_idx):
        loss, metrics = self.loss(self.model, maps, **self.weights(training=False))
        self.log("val/loss", loss, prog_bar=True, batch_size=len(maps))
        self.log_dict({f"val/{key}": value for key, value in metrics.items()}, batch_size=len(maps))
        if batch_idx == 0:
            self.preview_maps = maps[:10]

    def on_validation_epoch_end(self):
        """Log held-out maps, DDS selections or prior samples, and reconstructions to TensorBoard and `previews/`."""
        m = self.preview_maps
        if self.stage == 1:
            rows = [m, self.model.backbone.select(m), self.model.backbone(m)[0]]
        else:
            generator = torch.Generator(self.device).manual_seed(0)  # fixed prior samples across epochs
            z = torch.randn(len(m), self.model.latent_dim, generator=generator, device=self.device)
            rows = [m, self.model(m)[0], self.model.decode(z)]
        log_preview(self, rows, f"preview/stage{self.stage}")

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = AdamW(params, lr=self.optim["lr_start"], weight_decay=self.optim["weight_decay"])
        return cosine_schedule(optimizer, self.trainer, self.optim["lr_end"])


# ── Execution / Training ─────────────────────────────
def build_module(cfg, stage):
    stage_cfg = cfg.stage1 if stage == 1 else cfg.stage2
    return DDSVAEModule(asdict(cfg.model), asdict(stage_cfg), stage=stage, compile_mode=cfg.compile_mode)


def fit_stage(cfg, module, train_set, val_set, run_dir, resume=None):
    """Train one stage in `version_k/stage{n}` and return its best validation checkpoint."""
    stage_cfg = cfg.stage1 if module.stage == 1 else cfg.stage2
    # In-memory maps are batched in the main process.
    loader = lambda dataset, train: DataLoader(dataset, sampler=batch_sampler(dataset, stage_cfg.batch_size, train), batch_size=None, pin_memory=True)
    return fit(cfg, module, loader(train_set, True), loader(val_set, False), stage_cfg.epochs, run_dir, PANELS[module.stage],
               sub_dir=f"stage{module.stage}", resume=resume)


# ── Main ─────────────────────────────────────────────
def main(cfg: DDSVAETrainConfig):
    """Train both stages in one run, or resume its saved stage and complete the remaining training."""
    if cfg.resume and cfg.stage1_checkpoint:
        raise ValueError("Use --resume to continue a run, or --stage1_checkpoint to start Stage 2 in a new run.")
    torch.set_float32_matmul_precision("high")
    L.seed_everything(cfg.seed, workers=True)
    run_dir, resume = run_directory(cfg, staged=True)
    train_set, val_set = split_maps(cfg.map_dir, cfg.val_fraction, cfg.seed, cfg.flip_augment)
    print(f"[INFO] {len(train_set):,} train / {len(val_set):,} val maps -> {run_dir}")

    if resume and resume.parent.name == "stage2":
        fit_stage(cfg, build_module(cfg, 2), train_set, val_set, run_dir, resume=resume)
        return

    # 1) Stage 1 backbone
    stage1_checkpoint = cfg.stage1_checkpoint or fit_stage(cfg, build_module(cfg, 1), train_set, val_set, run_dir, resume=resume)

    # 2) Stage 2 mini-VAE on the frozen backbone
    module = build_module(cfg, 2)
    module.model.backbone.load_state_dict(DDSVAE.from_checkpoint(stage1_checkpoint).backbone.state_dict())
    fit_stage(cfg, module, train_set, val_set, run_dir)


if __name__ == "__main__":
    main(pyrallis.parse(config_class=DDSVAETrainConfig))
