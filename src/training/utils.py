"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared single-GPU training runs for SWM components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`fit` writes checkpoints, metrics, previews, and telemetry into each run directory:

    epochNNN.ckpt, last.ckpt, metrics.csv, summary.png, previews/, events.out.tfevents.*,
    hparams.yaml, emissions.csv, flops.json

`epochNNN.ckpt` minimizes the monitored validation metric; `last.ckpt` stores the
latest epoch. Resumed fits append metrics and TensorBoard events, reuse saved
hyperparameters, and write a separate FLOP/throughput summary per continuation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import csv
from dataclasses import asdict
from pathlib import Path
import time

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image

from src.training.callbacks import CarbonEmissionCallback, FlopsProfilerCallback, TelegramCallback

INK, SLATE, MIST, HAIRLINE, BLUE = "#191b21", "#626b7a", "#9ba6b7", "#dfe3ea", "#4d7bf3"
SUMMARY_STYLE = {"figure.facecolor": "#fbfbfb", "axes.facecolor": "#fbfbfb", "savefig.facecolor": "#fbfbfb", "font.family": "sans-serif",
                 "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": SLATE, "xtick.color": SLATE,
                 "ytick.color": SLATE, "axes.edgecolor": HAIRLINE, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0,
                 "ytick.major.size": 0, "axes.grid": True, "axes.grid.axis": "y", "grid.color": HAIRLINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
                 "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8}


# ── Output / IO ──────────────────────────────────────
class _ResumeCSVWriter(ExperimentWriter):
    """Reuse the CSV header without deleting existing metrics or rewriting hyperparameters."""

    def _check_log_dir_exists(self):
        with Path(self.metrics_file_path).open(newline="") as stream:
            self.metrics_keys = next(csv.reader(stream))

    def log_hparams(self, params):
        self.hparams.update(params)


class _ResumeCSVLogger(CSVLogger):
    """Append metrics throughout a resumed fit, including fits interrupted again."""

    @property
    def experiment(self):
        if self._experiment is None:
            self._experiment = _ResumeCSVWriter(self.log_dir)
        return self._experiment


# ── Execution / Training ─────────────────────────────
def next_version(output_dir, experiment_name):
    """Index k of the first unused `version_k` run directory of an experiment."""
    versions = [int(path.name.removeprefix("version_")) for path in (Path(output_dir) / experiment_name).glob("version_*")]
    return max(versions, default=-1) + 1


def run_directory(cfg, staged=False):
    """Return the run directory and resume checkpoint; `last` selects the latest version and DDS-VAE stage."""
    root = Path(cfg.output_dir) / cfg.experiment_name
    if cfg.resume == "last":
        run_dir = max(root.glob("version_*"), key=lambda path: int(path.name.removeprefix("version_")))
        checkpoint = max(run_dir.glob("stage[12]/last.ckpt")) if staged else run_dir / "last.ckpt"
    elif cfg.resume:
        checkpoint = Path(cfg.resume)
    else:
        return root / f"version_{next_version(cfg.output_dir, cfg.experiment_name)}", None

    checkpoint = checkpoint.resolve(strict=True)
    return (checkpoint.parent.parent if staged else checkpoint.parent), checkpoint


def cosine_schedule(optimizer, trainer, lr_end):
    """Lightning optimizer config: per-step cosine decay from the optimizer's learning rate to lr_end over the whole fit."""
    scheduler = CosineAnnealingLR(optimizer, T_max=trainer.estimated_stepping_batches, eta_min=lr_end)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def fit(cfg, module, train_loader, val_loader, epochs, run_dir, panels, sub_dir=None, monitor="val/loss", resume=None):
    """Train on one GPU in `run_dir[/sub_dir]` and return the checkpoint minimizing `monitor`.

    `resume` restores model, optimizer, schedule, and epoch while appending metrics.
    `panels` defines summary.png; cfg.notify enables Telegram fit reports.
    """
    # ── Loggers, checkpoints, and telemetry ───────────
    logger = TensorBoardLogger(str(run_dir.parent.parent), name=run_dir.parent.name, version=run_dir.name, sub_dir=sub_dir)
    log_dir = logger.log_dir
    if not resume:
        Path(log_dir).mkdir(parents=True, exist_ok=False)
        logger.log_hyperparams({"train": asdict(cfg)})
    csv_logger = (_ResumeCSVLogger if resume else CSVLogger)(log_dir, name="", version="")
    checkpoint = ModelCheckpoint(dirpath=log_dir, filename="epoch{epoch:03d}", auto_insert_metric_name=False,
                                 monitor=monitor, save_top_k=1)
    # A separate unmonitored callback saves last.ckpt even when validation does not improve.
    last_checkpoint = ModelCheckpoint(dirpath=log_dir, filename="last", enable_version_counter=False)
    callbacks = [checkpoint, last_checkpoint, LearningRateMonitor("step"), CarbonEmissionCallback(cfg, log_dir),
                 FlopsProfilerCallback(cfg, log_dir, resume=bool(resume))]
    if cfg.notify:
        telegram = TelegramCallback(Path(run_dir.parent.name) / run_dir.name / (sub_dir or ""))
        callbacks.append(telegram)

    # ── Fit ───────────────────────────────────────────
    trainer = L.Trainer(accelerator="gpu", devices=1, precision=cfg.precision, max_epochs=epochs, benchmark=True,
                        gradient_clip_val=cfg.gradient_clip_val, logger=[logger, csv_logger], callbacks=callbacks, log_every_n_steps=10,
                        num_sanity_val_steps=0)
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)
    plot_summary(log_dir, panels, monitor)
    print(f"[INFO] Best checkpoint: {checkpoint.best_model_path}")
    if cfg.notify:
        telegram.report(f"best {monitor} {checkpoint.best_model_score:.4f} ({Path(checkpoint.best_model_path).stem})", Path(log_dir) / "summary.png")
    return checkpoint.best_model_path


# ── Output / IO ──────────────────────────────────────
def _preview_path(module):
    """Keep earlier previews when an interrupted epoch is repeated."""
    directory = Path(module.logger.log_dir) / "previews"
    directory.mkdir(exist_ok=True)
    path = directory / f"epoch{module.current_epoch:03d}.png"
    return path.with_stem(f"{path.stem}-{time.time_ns()}") if path.exists() else path


def log_preview(module, rows, tag):
    """Log image rows [N, 3, H, W] in [-1, 1] as one grid to TensorBoard and to `previews/epochNNN.png`."""
    grid = make_grid(torch.cat(rows).float().add(1).div(2).clamp(0, 1), nrow=len(rows[0]), padding=1)
    module.logger.experiment.add_image(tag, grid, module.global_step)
    save_image(grid, _preview_path(module))


def log_figure(module, fig, tag):
    """Log a matplotlib figure to TensorBoard and to `previews/epochNNN.png`, then close it."""
    fig.savefig(_preview_path(module), dpi=150)
    module.logger.experiment.add_figure(tag, fig, module.global_step)  # renders and closes the figure


def plot_summary(log_dir, panels, monitor="val/loss"):
    """Plot epoch-averaged metrics.csv as summary.png, with panels (title, metric, y-scale).

    Gray denotes training and blue validation; markers identify the epoch minimizing
    `monitor`. Metric None selects learning rates; unlogged metrics are omitted.
    """
    frame = pd.read_csv(Path(log_dir) / "metrics.csv")
    frame["epoch"] = frame["epoch"].bfill()  # learning-rate rows carry only a step; the next row, logged at that step, has its epoch
    epochs = frame.groupby("epoch").mean()  # step rows -> one value per epoch
    best = epochs[monitor].idxmin()  # epoch kept by ModelCheckpoint
    panels = [(title, key, yscale) for title, key, yscale in panels if key is None or f"train/{key}" in epochs or f"val/{key}" in epochs]
    with plt.rc_context(SUMMARY_STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 2.7), layout="constrained")
        for ax, (title, key, yscale) in zip(axes, panels):
            if key is None:  # one column per optimizer param group, all on the same schedule
                curves = [("lr", next(column for column in epochs if column.startswith("lr-")), SLATE)]
            elif f"train/{key}" not in epochs:
                curves = [("val", f"val/{key}", BLUE)]
            elif f"val/{key}" in epochs:
                curves = [("train", f"train/{key}", MIST), ("val", f"val/{key}", BLUE)]
            else:
                curves = [("train", f"train/{key}", SLATE)]
            for name, column, color in curves:
                y = epochs[column].dropna()
                ax.plot(y, color=color, lw=1.6 if name == "val" else 1.2, label=name)
                if name == "val":
                    ax.scatter([best], [y[best]], s=22, color=BLUE, zorder=3)
            ax.set_title(title, color=INK, pad=6)
            ax.set_xlabel("epoch")
            ax.set_yscale(yscale)
        axes[0].legend(loc="upper right", frameon=False, fontsize=8.5)
        fig.savefig(Path(log_dir) / "summary.png", dpi=200)
        plt.close(fig)
