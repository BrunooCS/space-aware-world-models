"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transition model training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train T_psi on rollout state sequences and save checkpoints, metrics, and previews
under checkpoints/<experiment_name>/version_k/.

1) Split by episode and build windows without resets or out-of-range states.
2) Encode scene maps with DDS-VAE and standardize increments using training statistics.
3) Train with teacher forcing and weighted increment MSE, drawing one posterior scene
   code per episode at the start of every epoch. Validation and previews use posterior means.
4) Evaluate teacher-forced and free-running predictions; select the checkpoint with
   the lowest val/rollout_loss.

Uses fp32 with TF32 disabled to limit accumulated rollout error. Resuming restores
weights, optimizer, schedule, and epoch; settings, including --optim.epochs, must
match the saved run. `--resume last` selects the latest version.

Examples:
    python -m src.training.train_transition --scene_encoder checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt
    python -m src.training.train_transition --scene_encoder checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt --resume last
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import asdict

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from matplotlib import patheffects
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.configs import ENV_CONFIGS
from src.configs.transition import TransitionTrainConfig
from src.datasets.maps import batch_sampler
from src.datasets.rollouts import RolloutSequenceDataset, encode_scenes, load_transitions, read_scene_maps, sequence_windows, split_episodes
from src.models.dds_vae import reparameterize
from src.models.transition import TransitionModel, rollout_errors, transition_loss
from src.training.utils import BLUE, INK, SUMMARY_STYLE, cosine_schedule, fit, log_figure, run_directory

# summary.png panels: (title, metric, y-scale); None is the learning rate
PANELS = [("loss (teacher forcing)", "loss", "log"), ("rollout loss", "rollout_loss", "log"), ("position error", "position_error", "log"),
          ("heading error (°)", "heading_error", "log"), ("learning rate", None, "log")]


# ── Model ────────────────────────────────────────────
class TransitionModule(L.LightningModule):
    """Teacher-forced transition training with one posterior scene code per episode, redrawn every epoch."""

    def __init__(self, model: dict, optim: dict, compile_mode: str, env: str, scene_encoder: str, mu, logvar, preview=None):
        super().__init__()
        self.save_hyperparameters(ignore=["mu", "logvar", "preview"])
        self.model = TransitionModel(**model)
        self.optim, self.env_config = optim, ENV_CONFIGS[env]()
        self.register_buffer("loss_weights", torch.tensor(self.env_config.delta_loss_weights), persistent=False)  # [d_Delta]
        self.loss = torch.compile(transition_loss, mode=compile_mode)
        self.preview = preview  # held-out episodes (s_t, a_t, mu, shown, maps, bounds), rolled out after every validation epoch
        self.register_buffer("mu", mu, persistent=False)  # posterior means of all episodes [N, d_z]
        self.register_buffer("logvar", logvar, persistent=False)
        self.register_buffer("z", torch.empty_like(mu), persistent=False)  # scene codes of the current epoch [N, d_z]

    def on_train_epoch_start(self):
        self.z.copy_(reparameterize(self.mu, self.logvar))

    def training_step(self, batch, batch_idx):
        s, a, delta, episodes = batch
        loss = self.loss(self.model, s, self.z[episodes], a, delta, self.loss_weights)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(s))
        return loss

    def validation_step(self, batch, batch_idx):
        s, a, delta, episodes = batch
        mu = self.mu[episodes]
        self.log("val/loss", self.loss(self.model, s, mu, a, delta, self.loss_weights), batch_size=len(s))
        rollout_loss, position, heading = rollout_errors(self.model, s, mu, a, delta, self.loss_weights)
        self.log("val/rollout_loss", rollout_loss, prog_bar=True, batch_size=len(s))
        self.log_dict({"val/position_error": position.mean(), "val/final_position_error": position[:, -1].mean(),
                       "val/heading_error": heading.mean()}, batch_size=len(s))

    def on_validation_epoch_end(self):
        """Roll out the held-out preview episodes from their first state; log trajectories and the position error per step."""
        s, a, mu, shown, maps, bounds = self.preview
        s, a, mu = s.to(self.device), a.to(self.device), mu.to(self.device)
        states, _ = self.model.rollout(s[:, 0], mu, a[:, :-1])  # [E, T, d_s]
        errors = (states[..., :2] - s[..., :2]).norm(dim=-1).cpu().numpy()  # [E, T]
        # Raw positions [8, T, 2] of the drawn episodes, in the coordinates of their map extents
        recorded, predicted = (self.env_config.denormalize_state(x.cpu().numpy()[shown])[..., :2] for x in (s, states))
        log_figure(self, plot_rollouts(maps, bounds, recorded, predicted, errors), "preview")

    def configure_optimizers(self):
        """AdamW on all parameters with per-step cosine decay."""
        optimizer = AdamW(self.model.parameters(), lr=self.optim["lr_start"], weight_decay=self.optim["weight_decay"])
        return cosine_schedule(optimizer, self.trainer, self.optim["lr_end"])


# ── Output / IO ──────────────────────────────────────
def plot_rollouts(maps, bounds, recorded, predicted, errors):
    """Scene maps with recorded (white) and free-running (blue) trajectories from a shared start, and the position error per
    rollout step over all preview episodes (median and interquartile band)."""
    outline = [patheffects.withStroke(linewidth=2.6, foreground=INK)]
    with plt.rc_context(SUMMARY_STYLE):
        fig = plt.figure(figsize=(13, 5.4), layout="constrained")
        grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.8])
        for i, (m, (x_min, y_min, x_max, y_max), true, pred) in enumerate(zip(maps, bounds, recorded, predicted)):
            ax = fig.add_subplot(grid[i // 4, i % 4])
            ax.imshow(m, extent=(x_min, x_max, y_min, y_max), interpolation="nearest")
            ax.plot(*true.T, color="white", lw=1.3, path_effects=outline)
            ax.plot(*pred.T, color=BLUE, lw=1.3, path_effects=outline)
            ax.plot(*true[0], "o", color="white", ms=4, mec=INK, mew=0.8)
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
            ax.set_axis_off()

        # ── Position error per rollout step ───────────
        ax = fig.add_subplot(grid[:, 4])
        q1, median, q3 = np.quantile(errors, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(np.arange(errors.shape[1]), q1, q3, color=BLUE, alpha=0.15, lw=0, label="25-75%")
        ax.plot(median, color=BLUE, lw=1.6, label="median")
        ax.set_title(f"position error, {len(errors)} held-out episodes", color=INK, pad=6)
        ax.set(xlabel="rollout step", ylabel="normalized units")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    return fig


# ── Main ─────────────────────────────────────────────
def main(cfg: TransitionTrainConfig):
    """Encode the episode scenes once, then train the transition model in one versioned run."""
    # Disable TF32 to limit accumulated rollout error, including in cuDNN's LSTM.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    L.seed_everything(cfg.seed, workers=True)
    run_dir, resume = run_directory(cfg)
    env_config = ENV_CONFIGS[cfg.env]()

    # 1) s_t, a_t, Delta s_t, and usable transitions of every episode; episode split
    states, actions, deltas, valid = load_transitions(cfg.rollout_h5, env_config)
    train_episodes, val_episodes = split_episodes(len(states), cfg.val_fraction, cfg.seed)

    # 2) Scene posteriors mu, logvar [N, d_z] of every episode map; reset-free windows of each split
    mu, logvar = encode_scenes(cfg.rollout_h5, cfg.scene_encoder)
    length, stride = cfg.optim.sequence_length, cfg.optim.stride
    train_set = RolloutSequenceDataset(sequence_windows(valid, train_episodes, length, stride), length, states, actions, deltas)
    val_set = RolloutSequenceDataset(sequence_windows(valid, val_episodes, length, 4 * stride), length, states, actions, deltas)

    # 3) Previews: whole held-out episodes without resets, eight of them drawn on their scene maps
    episodes = val_episodes[valid[val_episodes].all(axis=1)]
    shown = np.linspace(0, len(episodes) - 1, 8).astype(int)
    maps, bounds = read_scene_maps(cfg.rollout_h5, episodes[shown], env_config)
    preview = (torch.from_numpy(states[episodes]), torch.from_numpy(actions[episodes]), mu[episodes], shown, maps, bounds)

    # 4) Transition model sized by the data, with the increment statistics of the usable training transitions
    train_deltas = deltas[train_episodes][valid[train_episodes]]  # [M, d_Delta]
    model = asdict(cfg.model) | {"latent_dim": mu.shape[1], "state_dim": states.shape[2], "action_dim": actions.shape[2],
                                 "delta_dim": deltas.shape[2], "delta_mean": train_deltas.mean(0).tolist(), "delta_std": train_deltas.std(0).tolist()}
    module = TransitionModule(model, asdict(cfg.optim), cfg.compile_mode, cfg.env, cfg.scene_encoder, mu, logvar, preview)
    print(f"[INFO] {len(train_set):,} train / {len(val_set):,} val windows ({len(train_episodes):,} / {len(val_episodes):,} episodes, "
          f"{len(episodes)} previewed) -> {run_dir}")

    # 5) Fit; batches of windows are gathered from host memory in the main process
    loader = lambda dataset, train: DataLoader(dataset, sampler=batch_sampler(dataset, cfg.optim.batch_size, train), batch_size=None,
                                               pin_memory=True)
    fit(cfg, module, loader(train_set, True), loader(val_set, False), cfg.optim.epochs, run_dir, PANELS, monitor="val/rollout_loss", resume=resume)


if __name__ == "__main__":
    main(pyrallis.parse(config_class=TransitionTrainConfig))
