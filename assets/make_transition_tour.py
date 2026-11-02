#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWM transition rollout with wall contacts (animated SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Render a looping SVG comparing a recorded DMLab rollout with free-running transition-model
rollouts under the same actions: trajectories on the scene map, and per-step speed,
heading, and position error.

1) Read a window of recorded states and FastSweepingPolicy actions from the rollout dataset.
2) Roll out T_psi from s_0 with the scene's posterior mean z, and with the z of another episode's map.
3) Mark contacts where a forward step moves less than `contact` of the free speed.

The default episode is held out by the transition split (seed 101, 5% validation). Only
the scene code carries the walls, so the mismatched-z rollout shows how much of the contact
behavior comes from z. The map zooms onto the three trajectories.

Example:
    python assets/make_transition_tour.py --checkpoint checkpoints/transition_dmlab/version_0/last.ckpt \
        --episode 205 --start 100 --output assets/swm_transition_dmlab.svg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pyrallis
import torch

from assets.svg_utils import BLUE, HAIRLINE, INK, MONO, agent, document, image_uri, keyframes, move_css, polyline, text, title
from src.configs.dmlab import DMLabEnvConfig
from src.models.dds_vae import DDSVAE
from src.models.transition import TransitionModel
from src.tools.render_models import encode_scene

RECORDED, OTHER = "#bcc4d1", "#9443d0"  # ground-truth band (slate) and mismatched scene code (violet, the latent-code color)


# ── Config ───────────────────────────────────────────
@dataclass
class TransitionTourConfig:
    """Checkpoint, rollout window, timing, and output file."""

    checkpoint: str               # train_transition checkpoint
    rollout_h5: str = "data/DMLabRolloutDataset.h5"
    episode: int = 205            # held-out rollout episode of the window
    start: int = 100              # first recorded step of the window
    other_episode: int = 185      # episode whose scene map drives the mismatched rollout
    steps: int = 200              # rollout length (training window length)
    contact: float = 0.35         # contact when a forward step moves less than this fraction of the free speed
    output: str = "assets/swm_transition_dmlab.svg"
    fps: int = 15                 # playback rate; 15 replays the recorded steps in real time
    hold_seconds: float = 2.0     # pause on the final frame before the loop restarts
    device: str = "cuda"


# ── Rollouts ─────────────────────────────────────────
def read_window(cfg):
    """Raw states [T + 1, 3], actions [T, 7], and the scene maps [128, 128, 3] of the episode and the mismatched episode."""
    with h5py.File(cfg.rollout_h5, "r") as f:
        episodes = f["episodes"]
        raw = episodes["state"][cfg.episode, cfg.start:cfg.start + cfg.steps + 1]
        actions = episodes["action"][cfg.episode, cfg.start:cfg.start + cfg.steps]
        return raw, actions, episodes["topdown"][cfg.episode], episodes["topdown"][cfg.other_episode]


@torch.inference_mode()
def rollout(model, config, raw_states, actions, z, device):
    """Free-running raw states [T + 1, 3] of T_psi from the first recorded state under the recorded actions."""
    s0 = torch.from_numpy(config.normalize_state(raw_states[:1])).to(device)  # [1, d_s]
    a = torch.from_numpy(config.normalize_action(actions)).to(device)[None]  # [1, T, d_a]
    states, _ = model.rollout(s0, z, a)  # [1, T + 1, d_s]
    return config.denormalize_state(states[0].cpu().numpy())


def contacts(positions, forward, contact):
    """Contact mask [T] of forward steps (forward [T] bool) that move less than `contact` of the free speed (95th-percentile step)."""
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return forward & (speed < contact * np.percentile(speed[forward], 95))


# ── Main ─────────────────────────────────────────────
def main(cfg: TransitionTourConfig):
    """Read a recorded window, roll out the transition model with matched and mismatched scene codes, and write the SVG."""
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # limit accumulated rollout error

    # 1) Recorded window and scene codes
    model = TransitionModel.from_checkpoint(cfg.checkpoint, cfg.device)
    scene_encoder = DDSVAE.from_checkpoint(torch.load(cfg.checkpoint, map_location="cpu")["hyper_parameters"]["scene_encoder"], cfg.device)
    raw, actions, scene_map, other_map = read_window(cfg)
    config = DMLabEnvConfig()

    # 2) Free-running rollouts: matched z and the z of another episode's map
    runs = [raw] + [rollout(model, config, raw, actions, encode_scene(scene_encoder, m, cfg.device), cfg.device) for m in (scene_map, other_map)]
    size = scene_map.shape[1]
    poses = [np.stack(config.world_to_pixel(r[:, 0], r[:, 1], size), axis=1) for r in runs]  # map pixels [T + 1, 2] per run
    headings = [np.degrees(np.unwrap(np.radians(r[:, 2]))) for r in runs]  # [T + 1] deg, counterclockwise
    contact = contacts(poses[0], actions[:, config.action_forward_idx] > 0, cfg.contact)
    speeds = [np.linalg.norm(np.diff(p, axis=0), axis=1) for p in poses]  # map px per step [T]
    errors = [np.linalg.norm(p - poses[0], axis=1) for p in poses[1:]]  # [T + 1]

    # ── Timeline: one frame per step, then a hold ─────
    T = cfg.steps
    hold = round(cfg.hold_seconds * cfg.fps)
    num_frames = T + 1 + hold
    duration = num_frames / cfg.fps
    frame_step = np.r_[np.minimum(np.arange(num_frames), T), 0]  # step shown by frame k; frame F wraps to the start

    # ── Layout ────────────────────────────────────────
    M, margin, gap, top = 300, 32, 48, 84
    plot_w, plot_h, plot_gap = 400, 64, 36
    x_map, x_plot = margin, margin + M + gap
    width, height = x_plot + plot_w + margin, top + M + 58
    corner_low, corner_high = np.concatenate(poses).min(axis=0), np.concatenate(poses).max(axis=0)
    span = min(size, (corner_high - corner_low).max() + 16)  # square crop around all trajectories, 8 px padding
    origin = np.clip((corner_low + corner_high - span) / 2, 0, size - span)
    scale = M / span
    poses = [(p - origin) for p in poses]  # crop coordinates [T + 1, 2]

    # ── CSS: arrows, progressive paths, playhead ──────
    css = [f".path{{animation-duration:{duration:.3f}s;animation-timing-function:linear;animation-iteration-count:infinite}}"]
    body = [text(x_map, 36, "SWM learns wall collisions from the scene code", size=15, fill=INK, extra=' font-weight="bold"'),
            title(x_map, top - 20, "TRAJECTORIES"), title(x_plot, top - 20, "PER-STEP STATE")]
    body.append(f'<g transform="translate({x_map},{top})"><clipPath id="crop"><rect width="{M}" height="{M}" rx="4"/></clipPath><g clip-path="url(#crop)">'
                f'<image href="{image_uri(scene_map, 90)}" x="{-origin[0] * scale:.1f}" y="{-origin[1] * scale:.1f}" width="{size * scale:.1f}" '
                f'height="{size * scale:.1f}" opacity=".4" style="image-rendering:pixelated"/>')
    colors, names, lines = (RECORDED, BLUE, OTHER), ("sim", "swm", "mis"), (8, 1.6, 1.6)
    for i in (0, 2, 1):  # wide ground-truth band below, matched-z prediction on top
        name, color, p, h, line = names[i], colors[i], poses[i], headings[i], lines[i]
        points = p * scale
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        total = length[-1]
        offsets = [(100 * k / num_frames, f"{total - length[frame_step[k]]:.1f}") for k in range(num_frames)] + [(100, f"{total:.1f}")]
        css.append(f".p{name}{{animation-name:p{name}}}" + keyframes(f"p{name}", offsets, "stroke-dashoffset"))
        body.append(polyline(points, color, line, f' class="path p{name}" stroke-dasharray="{total:.1f} {total + 1:.1f}" stroke-dashoffset="{total:.1f}"'))
        closed = points[frame_step]
        css.append(move_css(f"a{name}", closed, duration, -h[frame_step]))
        body.append(agent(f"a{name}", "#7d8799" if i == 0 else color, radius=4, cone=14))

    # Contact markers appear when the recorded agent reaches them
    for t in np.flatnonzero(contact & ~np.r_[False, contact[:-1]]):  # first step of every contact interval
        x, y = poses[0][t + 1] * scale
        appear = 100 * (t + 1) / num_frames
        css.append(f".k{t}{{opacity:0;animation:k{t} {duration:.3f}s step-end infinite}}" + keyframes(f"k{t}", [(0, 0), (appear, 1)], "opacity"))
        body.append(f'<path class="k{t}" d="M{x - 4:.1f} {y - 4:.1f}l8 8m0 -8l-8 8" stroke="{INK}" stroke-width="1.6"/>')
    bar = 10 * scale  # scale bar of 10 map pixels
    body.append(f'</g><rect width="{M}" height="{M}" rx="4" fill="none" stroke="{HAIRLINE}"/>'
                f'<path d="M{M - 12 - bar:.1f} {M - 12}h{bar:.1f}" stroke="{INK}" stroke-width="2"/>'
                + text(M - 12 - bar / 2, M - 18, "10 px", size=10, family=MONO, fill=INK, anchor="middle") + "</g>")

    # ── Plots: speed, heading, position error ─────────
    def plot(y0, label, unit, series, end_labels=False):
        """One time-series panel at y0 with contact bands and (values [T or T + 1], color, width) series on a shared axis."""
        high = max(v.max() for v, *_ in series) * 1.08
        low = min(0.0, min(v.min() for v, *_ in series))
        out = [f'<g transform="translate({x_plot},{y0})">']
        for t in np.flatnonzero(contact):
            out.append(f'<rect x="{t / T * plot_w:.1f}" width="{plot_w / T + .4:.1f}" height="{plot_h}" fill="{HAIRLINE}" opacity=".7"/>')
        out.append(f'<path d="M0 {plot_h}h{plot_w}" stroke="{HAIRLINE}"/>')
        for values, color, line in series:
            t = np.arange(len(values)) + (len(values) == T) * 0.5  # increments sit between their states
            points = np.stack([t / T * plot_w, plot_h - (values - low) / (high - low) * plot_h], 1)
            out.append(polyline(points, color, line))
            if end_labels:  # final value next to the curve end
                out.append(text(plot_w + 6, points[-1, 1] + 3.5, f"{values[-1]:.1f}", size=10, family=MONO, fill=color))
        out.append(text(0, -7, label, size=10, family=MONO) + text(plot_w, -7, unit, size=10, family=MONO, anchor="end"))
        return out + ["</g>"]

    y = top + 14
    widths = (3.5, 1.3, 1.3)  # same band-under-line reading as the map
    order = (0, 2, 1)
    body += plot(y, "speed ‖Δp‖", "map px / step", [(speeds[i], colors[i], widths[i]) for i in order])
    body += plot(y + plot_h + plot_gap, "heading θ", "deg", [(headings[i], colors[i], widths[i]) for i in order])
    body += plot(y + 2 * (plot_h + plot_gap), "position error ‖p̂ − p‖ to recorded", "map px",
                 [(errors[1], OTHER, 1.3), (errors[0], BLUE, 1.3)], end_labels=True)
    plot_bottom = y + 3 * plot_h + 2 * plot_gap
    body.append(text(x_plot, plot_bottom + 16, "0", size=10, family=MONO) + text(x_plot + plot_w, plot_bottom + 16, f"{T} steps", size=10, family=MONO, anchor="end"))
    css.append(move_css("head", np.stack([frame_step / T * plot_w, np.zeros(num_frames + 1)], 1), duration))
    body.append(f'<g transform="translate({x_plot},{y - 4})"><path class="head" d="M0 0V{plot_bottom - y + 4}" stroke="{INK}" stroke-opacity=".55" stroke-width="1"/></g>')

    # ── Legend ────────────────────────────────────────
    ly = top + M + 28
    items = [("recorded (ground truth)", RECORDED, 6), ("SWM · z of this map", BLUE, 1.8), ("SWM · z of another map", OTHER, 1.8)]
    x = x_map
    for label, color, line in items:
        body.append(polyline([(x, ly - 4), (x + 18, ly - 4)], color, line) + text(x + 24, ly, label, size=10, family=MONO, fill=INK))
        x += 40 + 6.1 * len(label)
    body.append(f'<path d="M{x_map + 5} {ly + 12}l8 8m0 -8l-8 8" stroke="{INK}" stroke-width="1.6"/>' + text(x_map + 24, ly + 20, "wall contact", size=10, family=MONO)
                + f'<rect x="{x_map + 120}" y="{ly + 11}" width="10" height="10" fill="{HAIRLINE}"/>' + text(x_map + 136, ly + 20, "contact steps", size=10, family=MONO))

    # ── Write ─────────────────────────────────────────
    output = Path(cfg.output)
    output.write_text(document(width, height, css, body))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB, {duration:.1f} s loop, {int((contact & ~np.r_[False, contact[:-1]]).sum())} contacts, "
          f"final error {errors[0][-1]:.1f} px (other z {errors[1][-1]:.1f} px), mean {errors[0].mean():.1f} px")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=TransitionTourConfig))
