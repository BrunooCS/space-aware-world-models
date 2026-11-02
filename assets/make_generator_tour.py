#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWM observation model along a rollout (animated SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Render a looping SVG of the generator rendering ego views from the scene code and the
agent state: the decoded scene code with the pose and s_t, the simulator observation,
and the render o_hat_t = G(z, s_t) at the same state.

1) Drive FastSweepingPolicy for `steps` steps in maze `seed`, recording o_t and s_t.
2) Encode the scene map to its posterior mean z and render every state in batches.
3) Write ego views as sprite sheets and the pose, trail, and s_t bars as CSS keyframes.

The generator sees only z and s_t; the simulator view is shown for reference.

Example:
    python assets/make_generator_tour.py --checkpoint checkpoints/generator_dmlab/version_0/last.ckpt \
        --seed 104 --output assets/swm_generator_dmlab.svg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrallis
import torch

from assets.svg_utils import (BLUE, HAIRLINE, INK, MONO, agent, document, image_uri, keyframes, move_css, polyline, sprite_css, sprite_frames,
                              sprite_panel, text, title)
from src.datasets.make_rollouts_dmlab import FastSweepingPolicy
from src.envs import make_simulator
from src.models.dds_vae import DDSVAE
from src.models.generator import Generator
from src.tools.render_models import encode_scene
from src.tools.render_utils import to_image

STATE_NAMES = ("x", "y", "sin θ", "cos θ")


# ── Config ───────────────────────────────────────────
@dataclass
class GeneratorTourConfig:
    """Checkpoint, maze, rollout length, timing, and output file."""

    checkpoint: str               # train_generator checkpoint
    seed: int = 104               # DMLab maze seed of the rollout
    steps: int = 180              # recorded steps in the loop
    output: str = "assets/swm_generator_dmlab.svg"
    fps: int = 15                 # playback rate; 15 replays the recorded steps in real time
    panel_size: int = 240
    jpeg_quality: int = 80
    device: str = "cuda"


# ── Rollout ──────────────────────────────────────────
def collect(seed, steps):
    """Observations [T, 96, 96, 3], normalized states s_t [T, 4], map poses [T, 3], and the scene map of one policy rollout."""
    simulator = make_simulator("dmlab")
    state, observation, scene_map = simulator.reset(seed)
    policy = FastSweepingPolicy(scene_map, seed=seed)
    views, states, poses = [], [], []
    for _ in range(steps):
        views.append(observation)
        states.append(simulator.config.normalize_state(state))
        poses.append(simulator.map_pose(state))
        state, observation, done = simulator.step(policy.step(state))
        assert not done, "episode ended inside the rollout"
    simulator.close()
    return np.stack(views), np.stack(states), np.array(poses, dtype=float), scene_map


# ── Main ─────────────────────────────────────────────
@torch.inference_mode()
def main(cfg: GeneratorTourConfig):
    """Record a rollout, render it with the generator, and write the animated SVG."""
    # 1) Rollout and renders o_hat_t = G(z, s_t)
    generator = Generator.from_checkpoint(cfg.checkpoint, cfg.device)
    scene_encoder = DDSVAE.from_checkpoint(torch.load(cfg.checkpoint, map_location="cpu")["hyper_parameters"]["scene_encoder"], cfg.device)
    views, states, poses, scene_map = collect(cfg.seed, cfg.steps)
    z = encode_scene(scene_encoder, scene_map, cfg.device)  # posterior mean [1, d_z]
    decoded = to_image(scene_encoder.decode(z))[0]  # [128, 128, 3]
    s = torch.from_numpy(states).to(cfg.device)
    renders = np.concatenate([to_image(generator(z.expand(len(chunk), -1), chunk)) for chunk in s.split(60)])  # [T, 96, 96, 3]
    mse = ((renders / 255.0 - views / 255.0) ** 2).mean()
    T, duration = len(views), len(views) / cfg.fps

    # ── Layout ────────────────────────────────────────
    P, margin, gap, top = cfg.panel_size, 32, 28, 56
    x_map, x_view, x_render = margin, margin + P + gap, margin + 2 * (P + gap)
    bar_w, bar_h, bar_gap = P - 56, 8, 16
    width, height = x_render + P + margin, top + P + 22 + 4 * bar_gap + 30

    # 2) Pose, trail, and s_t bars per frame
    scale = P / decoded.shape[1]
    position = poses[:, :2] * scale  # [T, 2]
    closed = np.concatenate([position, position[:1]])
    heading = -np.degrees(np.unwrap(poses[:, 2]))  # SVG rotates clockwise with y down
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1))]
    total = length[-1]
    view_frames, view_rows = sprite_frames(views)
    render_frames, render_rows = sprite_frames(renders)
    css = [sprite_css("view", view_rows, views.shape[1], duration), sprite_css("render", render_rows, renders.shape[1], duration),
           move_css("agent", closed, duration, np.r_[heading, heading[:1]]),
           f".trail{{animation:trail {duration:.3f}s linear infinite}}",
           keyframes("trail", [(100 * k / T, f"{total - length[k]:.1f}") for k in range(T)] + [(100, f"{total:.1f}")], "stroke-dashoffset")]
    stride = 3  # bar keyframes every third step; CSS interpolates in between
    for i in range(4):
        values = np.r_[states[::stride, i], states[0, i]]
        stops = [(100 * min(k * stride, T) / T, f"scaleX({v:.3f})") for k, v in enumerate(values)]
        css.append(f".b{i}{{transform-box:fill-box;transform-origin:left center;animation:b{i} {duration:.3f}s linear infinite}}"
                   + keyframes(f"b{i}", stops, "transform"))

    # ── Panels ────────────────────────────────────────
    body = [title(x_map, 36, "SCENE CODE · DECODED"), title(x_view, 36, "SIMULATOR o_t"), title(x_render, 36, "GENERATOR G(z, s_t)")]
    body.append(f'<g transform="translate({x_map},{top})"><image href="{image_uri(decoded, 90)}" width="{P}" height="{P}" opacity=".8"/>')
    body.append(polyline(position, INK, 1.4, f' class="trail" stroke-dasharray="{total:.1f} {total + 1:.1f}" stroke-dashoffset="{total:.1f}"'))
    body.append(agent("agent", INK))
    body.append(f'<rect width="{P}" height="{P}" rx="4" fill="none" stroke="{HAIRLINE}"/></g>')
    body.append(sprite_panel(x_view, top, P, "view", view_frames, cfg.jpeg_quality))
    body.append(sprite_panel(x_render, top, P, "render", render_frames, cfg.jpeg_quality))

    # s_t bars on [-1, 1]: a positive half-width rect scaled by the value, flipped left when negative
    y = top + P + 24
    body.append(text(x_map, y, "s_t", size=10, family=MONO, fill=INK))
    for i, name in enumerate(STATE_NAMES):
        by = y + 10 + i * bar_gap
        cx = x_map + 44 + bar_w / 2
        body.append(text(x_map, by + 7, name, size=10, family=MONO) + f'<rect x="{x_map + 44}" y="{by}" width="{bar_w}" height="{bar_h}" rx="2" fill="{HAIRLINE}" opacity=".6"/>'
                    + f'<rect class="b{i}" x="{cx}" y="{by}" width="{bar_w / 2}" height="{bar_h}" fill="{BLUE}" opacity=".85"/>'
                    + f'<path d="M{cx} {by - 2}v{bar_h + 4}" stroke="{INK}" stroke-width="1"/>')
    body.append(text(x_map + 44, y + 10 + 4 * bar_gap + 8, "−1", size=10, family=MONO) + text(x_map + 44 + bar_w, y + 10 + 4 * bar_gap + 8, "+1", size=10, family=MONO, anchor="end"))

    # ── Legends ───────────────────────────────────────
    body.append(text(x_view, y, f"{views.shape[2]} × {views.shape[1]} RGB · maze seed {cfg.seed}", size=10, family=MONO))
    body.append(text(x_view, y + 16, "reference only; not an input to G", size=10, family=MONO))
    body.append(text(x_render, y, f"inputs: z ({z.shape[1]}-d posterior mean), s_t (4-d)", size=10, family=MONO))
    body.append(text(x_render, y + 16, f"alpha {generator.alpha.item():.3f} · clip MSE {mse:.4f} · {T} frames", size=10, family=MONO))

    # ── Write ─────────────────────────────────────────
    output = Path(cfg.output)
    output.write_text(document(width, height, css, body))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB, {duration:.1f} s loop, MSE {mse:.4f}")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=GeneratorTourConfig))
