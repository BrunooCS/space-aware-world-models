#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fast-sweeping data collection (animated SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Render a looping SVG of FastSweepingPolicy collecting a DMLab rollout: the recorded ego
observation, the geodesic travel-time field to the active waypoint, and the scene map with
the descent route and recorded trail.

1) Run the policy in maze `seed` until it reaches `waypoints` targets, recording every step.
2) Replay the solver sweeps of every new field while the agent waits.
3) Write ego views and fields as sprite sheets and the agent, trail, and routes as CSS keyframes.

One agent step is one recorded transition; `fps` 15 plays the rollout in real time
(30 Hz engine, step repeat 2).

Example:
    python assets/make_fast_sweeping_tour.py --seed 104 --output assets/fast_sweeping_dmlab.svg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pyrallis
from scipy.ndimage import distance_transform_edt

from assets.svg_utils import (HAIRLINE, INK, MONO, agent, document, image_uri, keyframes, move_css, polyline, rgb, sprite_css, sprite_frames, sprite_panel, text, title)
from src.datasets.make_rollouts_dmlab import FastSweepingPolicy
from src.envs import make_simulator
from src.envs.dmlab_autonomous_exploration import compute_path_smooth, replay_sweeps

WALL, UNREACHED = (18, 20, 25), (51, 57, 67)  # same field colors as the interactive fast-sweeping viewer
SWEEP_ROWS = 8  # grid rows per solver block of the replay


# ── Config ───────────────────────────────────────────
@dataclass
class SweepTourConfig:
    """Maze, rollout length, timing, and output file."""

    seed: int = 104               # DMLab maze seed (rollout episodes use seeds >= 101)
    waypoints: int = 3            # stop after this many reached targets
    output: str = "assets/fast_sweeping_dmlab.svg"
    fps: int = 15                 # playback rate; 15 replays the recorded steps in real time
    sweep_frames: int = 18        # frames of the solver replay shown for every new field
    field_size: int = 192         # field sprite resolution
    panel_size: int = 240
    jpeg_quality: int = 80


# ── Field images ─────────────────────────────────────
def viridis(t):
    """Viridis RGB [..., 3] uint8 at t [...] in [0, 1]."""
    t = np.asarray(t)
    return cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0, ::-1].reshape(*t.shape, 3)  # BGR -> RGB


def field_image(field, free, d_max, size):
    """Travel time [H, W] -> RGB [size, size, 3]: Viridis (bright at the waypoint), walls, and cells ahead of the sweep front."""
    reached = cv2.resize(np.isfinite(field).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    rows, columns = distance_transform_edt(~np.isfinite(field), return_distances=False, return_indices=True)  # nearest reached cell
    value = cv2.resize(field[rows, columns].astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)  # [S, S]
    image = viridis(1 - (value / d_max).clip(0, 1))
    image[~reached] = UNREACHED
    image[~free] = WALL
    return image


# ── Rollout ──────────────────────────────────────────
def collect(cfg):
    """Run the policy; per frame record the ego view, field image index, map pose, and route index."""
    simulator = make_simulator("dmlab")
    state, observation, scene_map = simulator.reset(cfg.seed)
    policy = FastSweepingPolicy(scene_map, seed=cfg.seed)
    explorer = policy.explorer
    free = cv2.resize(explorer.binary_map, (cfg.field_size, cfg.field_size), interpolation=cv2.INTER_NEAREST) > 0
    views, fields, poses, indices, routes, targets, dist_map = [], [], [], [], [], [], None
    stops = 1 - (1 - np.arange(1, cfg.sweep_frames + 1) / cfg.sweep_frames) ** 2  # replay progress per frame, quadratic ease-out

    while explorer.stats.targets_reached < cfg.waypoints:
        # ── New field: replay its sweeps while the agent waits ──
        if explorer.dist_map is not dist_map:
            dist_map = explorer.dist_map
            d_max = dist_map[np.isfinite(dist_map)].max()
            row, column = explorer.mapper.world_to_grid(state[0], state[1])
            routes.append(compute_path_smooth((row, column), dist_map)[:, ::-1])  # [N, 2] (row, column) -> (x, y)
            targets.append(explorer.current_target[::-1])
            k = 0
            for field, _, _, progress in replay_sweeps(explorer.binary_map, explorer.current_target, SWEEP_ROWS):
                while k < len(stops) and progress >= stops[k] - 1e-9:  # the replay updates one grid in place
                    fields.append(field_image(field, free, d_max, cfg.field_size))
                    views.append(observation)
                    poses.append(simulator.map_pose(state))
                    indices.append((len(fields) - 1, len(routes) - 1))
                    k += 1
            fields.append(field_image(dist_map, free, d_max, cfg.field_size))  # the policy's converged field

        # ── One recorded transition ────────────────────
        views.append(observation)
        poses.append(simulator.map_pose(state))
        indices.append((len(fields) - 1, len(routes) - 1))
        state, observation, done = simulator.step(policy.step(state))
        assert not done, "episode ended before the waypoint quota"

    simulator.close()
    return (scene_map, np.stack(views), np.stack(fields), np.array(poses, dtype=float), np.array(indices), routes,
            np.array(targets, dtype=float))


# ── Main ─────────────────────────────────────────────
def main(cfg: SweepTourConfig):
    """Collect one rollout and write the animated SVG."""
    # 1) Rollout with sweep replays
    scene_map, views, fields, poses, indices, routes, targets = collect(cfg)
    num_frames, duration = len(views), len(views) / cfg.fps
    views, view_rows = sprite_frames(views)
    field_rows = indices[:, 0]
    scale = cfg.panel_size / scene_map.shape[1]  # panel pixels per map pixel (map = 128 grid)

    # ── Layout ────────────────────────────────────────
    P, margin, gap, top = cfg.panel_size, 32, 28, 56
    x_view, x_field, x_map = margin, margin + P + gap, margin + 2 * (P + gap)
    width, height = x_map + P + margin, top + P + 72

    # 2) Agent, trail, active route and waypoint per frame
    position = poses[:, :2] * scale  # [F, 2] panel pixels
    heading = -np.degrees(np.unwrap(poses[:, 2]))  # SVG rotates clockwise with y down
    closed = np.concatenate([position, position[:1]])
    trail_length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1))]  # arc length drawn by frame k
    total = trail_length[-1]

    route_index = indices[:, 1]
    changes = np.flatnonzero(np.r_[True, route_index[1:] != route_index[:-1]])  # first frame of every field
    css = [sprite_css("view", view_rows, views.shape[1], duration), sprite_css("field", field_rows, fields.shape[1], duration),
           move_css("agent", closed, duration, np.r_[heading, heading[:1]]),
           f".trail{{animation:trail {duration:.3f}s linear infinite}}",
           keyframes("trail", [(100 * k / num_frames, f"{total - trail_length[k]:.1f}") for k in range(num_frames)] + [(100, f"{total:.1f}")], "stroke-dashoffset")]
    for i in range(len(routes)):  # each route and target is visible while its field is active
        start = 100 * changes[i] / num_frames
        end = 100 * (changes[i + 1] if i + 1 < len(routes) else num_frames) / num_frames
        stops = [(0, 0), (start, 1), (end, 0)] if start > 0 else [(0, 1), (end, 0)]
        css.append(f".r{i}{{opacity:0;animation:r{i} {duration:.3f}s step-end infinite}}" + keyframes(f"r{i}", stops, "opacity"))

    # ── Panels ────────────────────────────────────────
    body = [title(x_view, 36, "EGO OBSERVATION"), title(x_field, 36, "GEODESIC TRAVEL TIME"), title(x_map, 36, "SCENE MAP")]
    body.append(sprite_panel(x_view, top, P, "view", views, cfg.jpeg_quality))
    body.append(sprite_panel(x_field, top, P, "field", fields, cfg.jpeg_quality))

    # Field overlay: agent and active waypoint
    body.append(f'<g transform="translate({x_field},{top})">')
    for i, (tx, ty) in enumerate(targets * scale):
        body.append(f'<g class="r{i}"><circle cx="{tx:.1f}" cy="{ty:.1f}" r="5" fill="none" stroke="#fff" stroke-width="3"/>'
                    f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="5" fill="none" stroke="{INK}" stroke-width="1.4"/></g>')
    body.append(agent("agent", INK) + "</g>")

    # Scene map: trail, descent routes, waypoints, agent
    body.append(f'<g transform="translate({x_map},{top})"><image href="{image_uri(scene_map, 90)}" width="{P}" height="{P}" opacity=".55"/>')
    body.append(polyline(position, INK, 1.4, f' class="trail" stroke-dasharray="{total:.1f}" stroke-dashoffset="{total:.1f}"'))
    for i, route in enumerate(routes):
        body.append(f'<g class="r{i}">' + polyline(route * scale, INK, 1.2, ' stroke-dasharray="2 3" stroke-opacity=".7"')
                    + f'<circle cx="{targets[i, 0] * scale:.1f}" cy="{targets[i, 1] * scale:.1f}" r="5" fill="none" stroke="{INK}" stroke-width="1.6"/></g>')
    body.append(agent("agent", INK))
    body.append(f'<rect width="{P}" height="{P}" rx="4" fill="none" stroke="{HAIRLINE}"/></g>')

    # ── Legends ───────────────────────────────────────
    y = top + P + 22
    body.append(text(x_view, y, f"o_t · {views.shape[2]} × {views.shape[1]} RGB", size=10, family=MONO))
    body.append(text(x_view, y + 16, f"maze seed {cfg.seed} · {num_frames} frames", size=10, family=MONO))
    bar = "".join(f'<rect x="{x_field + i * P / 40:.1f}" y="{y - 9}" width="{P / 40 + .5:.1f}" height="7" fill="{rgb(viridis(1 - i / 39))}"/>' for i in range(40))
    body.append(bar + text(x_field, y + 10, "0 at waypoint", size=10, family=MONO) + text(x_field + P, y + 10, "far", size=10, family=MONO, anchor="end"))
    body.append(text(x_field, y + 25, "4 iterations × 4 sweep directions", size=10, family=MONO))
    body.append(polyline([(x_map, y - 5), (x_map + 18, y - 5)], INK, 1.4) + text(x_map + 24, y - 1, "recorded trail", size=10, family=MONO))
    body.append(polyline([(x_map + 120, y - 5), (x_map + 138, y - 5)], INK, 1.2, ' stroke-dasharray="2 3"') + text(x_map + 144, y - 1, "descent", size=10, family=MONO))
    body.append(f'<circle cx="{x_map + 5}" cy="{y + 11}" r="4" fill="none" stroke="{INK}" stroke-width="1.4"/>' + text(x_map + 14, y + 15, "active waypoint", size=10, family=MONO))

    # ── Write ─────────────────────────────────────────
    output = Path(cfg.output)
    output.write_text(document(width, height, css, body))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB, {num_frames} frames, {len(fields)} fields, {duration:.1f} s loop")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=SweepTourConfig))
