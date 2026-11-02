"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation model (generator) training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train G_theta on rollout tuples (z, s_t, o_t) and save checkpoints, metrics, and
previews under checkpoints/<experiment_name>/version_k/.

1) Split by episode and normalize states with the environment transform.
2) Encode each scene map once with DDS-VAE.
3) Train with weighted MSE + LPIPS, drawing one posterior scene code per episode
   at the start of every epoch. Validation and previews use posterior means.

The best checkpoint minimizes val/loss. Resuming restores weights, optimizer,
schedule, and epoch; settings, including --optim.epochs, must
match the saved run. `--resume last` selects the latest version.

Examples:
    python -m src.training.train_generator --scene_encoder checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt
    python -m src.training.train_generator --scene_encoder checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt --resume last
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import asdict

import lightning as L
import numpy as np
import pyrallis
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.functional.image.lpips import _NoTrainLpips

from src.configs import ENV_CONFIGS
from src.configs.generator import GeneratorTrainConfig
from src.datasets.maps import batch_sampler
from src.datasets.rollouts import RolloutFrameDataset, encode_scenes, load_states, read_frames, split_episodes
from src.models.dds_vae import reparameterize
from src.models.generator import Generator, normalize_frames, render_loss
from src.training.utils import cosine_schedule, fit, log_preview, run_directory

# summary.png panels: (title, metric, y-scale); None is the learning rate
PANELS = [("loss", "loss", "log"), ("MSE", "mse", "log"), ("LPIPS", "lpips", "log"), ("scene coefficient α", "alpha", "linear"),
          ("learning rate", None, "log")]


# ── Model ────────────────────────────────────────────
class GeneratorModule(L.LightningModule):
    """Renderer training with one posterior scene code per episode, redrawn every epoch."""

    def __init__(self, model: dict, optim: dict, compile_mode: str, env: str, scene_encoder: str, mu, logvar, preview=None):
        super().__init__()
        self.save_hyperparameters(ignore=["mu", "logvar", "preview"])
        self.model = Generator(**model).to(memory_format=torch.channels_last)
        self.lpips = _NoTrainLpips(net="alex").to(memory_format=torch.channels_last)  # frozen LPIPS v0.1 (AlexNet), always in eval mode
        self.optim, self.loss_weights = optim, {key: optim[key] for key in ("mse_weight", "lpips_weight")}
        self.loss = torch.compile(render_loss, mode=compile_mode)
        self.preview = preview  # fixed held-out (frames, s_t, mu), rendered after every validation epoch
        self.register_buffer("mu", mu, persistent=False)  # posterior means of all episodes [N, d_z]
        self.register_buffer("logvar", logvar, persistent=False)
        self.register_buffer("z", torch.empty_like(mu), persistent=False)  # scene codes of the current epoch [N, d_z]

    def on_train_epoch_start(self):
        self.z.copy_(reparameterize(self.mu, self.logvar))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        frames, s, episodes = batch
        return normalize_frames(frames), s, episodes  # uint8 transfer, normalization on device

    def training_step(self, batch, batch_idx):
        o, s, episodes = batch
        loss, metrics = self.loss(self.model, self.lpips, self.z[episodes], s, o, **self.loss_weights)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(o))
        self.log_dict({f"train/{key}": value for key, value in metrics.items()} | {"train/alpha": self.model.alpha}, batch_size=len(o))
        return loss

    def validation_step(self, batch, batch_idx):
        o, s, episodes = batch
        loss, metrics = self.loss(self.model, self.lpips, self.mu[episodes], s, o, **self.loss_weights)
        self.log("val/loss", loss, prog_bar=True, batch_size=len(o))
        self.log_dict({f"val/{key}": value for key, value in metrics.items()}, batch_size=len(o))

    def on_validation_epoch_end(self):
        """Log held-out frames o_t, renders G(mu, s_t), and renders of the same states under prior scene codes z ~ N(0, I)."""
        frames, s, mu = (tensor.to(self.device) for tensor in self.preview)
        generator = torch.Generator(self.device).manual_seed(0)  # fixed prior samples across epochs
        z = torch.randn(mu.shape, generator=generator, device=self.device)
        log_preview(self, [normalize_frames(frames), self.model(mu, s), self.model(z, s)], "preview")

    def configure_optimizers(self):
        """AdamW with decay on matrices and kernels only, and per-step cosine learning-rate decay."""
        params = list(self.model.parameters())
        groups = [{"params": [param for param in params if param.ndim > 1], "weight_decay": self.optim["weight_decay"]},
                  {"params": [param for param in params if param.ndim <= 1], "weight_decay": 0.0}]
        optimizer = AdamW(groups, lr=self.optim["lr_start"], betas=(0.9, 0.99), fused=True)
        return cosine_schedule(optimizer, self.trainer, self.optim["lr_end"])


# ── Main ─────────────────────────────────────────────
def main(cfg: GeneratorTrainConfig):
    """Encode the episode scenes once, then train the renderer in one versioned run."""
    torch.set_float32_matmul_precision("high")
    L.seed_everything(cfg.seed, workers=True)
    run_dir, resume = run_directory(cfg)

    # 1) Episode split and normalized states s_t [N, T, d_s]
    states = load_states(cfg.rollout_h5, ENV_CONFIGS[cfg.env]())
    train_episodes, val_episodes = split_episodes(len(states), cfg.val_fraction, cfg.seed)

    # 2) Scene posteriors mu, logvar [N, d_z] of every episode map
    mu, logvar = encode_scenes(cfg.rollout_h5, cfg.scene_encoder)
    train_set, val_set = (RolloutFrameDataset(cfg.rollout_h5, episodes, states) for episodes in (train_episodes, val_episodes))

    # 3) Ten held-out preview frames spread over validation episodes and episode time
    steps = states.shape[1]
    episodes, times = val_episodes[np.linspace(0, len(val_episodes) - 1, 10).astype(int)], (2 * np.arange(10) + 1) * steps // 20
    frames = read_frames(cfg.rollout_h5, episodes, times)
    preview = (frames, torch.from_numpy(states[episodes, times]), mu[episodes])

    # 4) Renderer sized by the scene code, state, and frame dimensions
    model = asdict(cfg.model) | {"latent_dim": mu.shape[1], "state_dim": states.shape[2], "image_size": frames.shape[1]}
    module = GeneratorModule(model, asdict(cfg.optim), cfg.compile_mode, cfg.env, cfg.scene_encoder, mu, logvar, preview)
    print(f"[INFO] {len(train_set):,} train / {len(val_set):,} val frames ({len(train_episodes):,} / {len(val_episodes):,} episodes) "
          f"-> {run_dir}")

    # 5) Fit; DataLoader workers read whole uint8 batches from the HDF5 file
    loader = lambda dataset, train: DataLoader(dataset, sampler=batch_sampler(dataset, cfg.optim.batch_size, train), batch_size=None,
                                               num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    fit(cfg, module, loader(train_set, True), loader(val_set, False), cfg.optim.epochs, run_dir, PANELS, resume=resume)


if __name__ == "__main__":
    main(pyrallis.parse(config_class=GeneratorTrainConfig))
