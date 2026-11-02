#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fast-sweeping exploration viewer for DMLab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run FastSweepingPolicy in a live maze and display the observation, waypoint distance
field, and descent route with the controller's search window and steering target.

1) Solve a new distance field when the waypoint changes.
2) Replay solver row blocks over sweep_seconds of wall time while the agent moves.
3) Advance scenes after waypoints_per_scene reached targets.

The route uses the policy's final field during replay. Default rollout episodes
use seeds >= 101.

Example:
    python -m src.tools.render_fast_sweeping_dmlab --seed 3

Controls: SPACE pause the agent, F replay the sweeps, R next scene, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pygame
import pyrallis
from scipy.ndimage import distance_transform_edt

from src.datasets.make_rollouts_dmlab import FastSweepingPolicy
from src.envs import make_simulator
from src.envs.dmlab_autonomous_exploration import compute_path_smooth, replay_sweeps
from src.tools.render_utils import (
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, TEXT_LABEL, TEXT_PRIMARY, TRAIL_SHADOW, ViewerWindow,
    draw_agent, draw_trajectory, map_pixels, numpy_to_surface,
)

MAP_WALL, MAP_FREE = (18, 20, 25), (51, 57, 67)
SWEEP_ROWS = 8  # grid rows per solver block of the replay


# ── Config ───────────────────────────────────────────
@dataclass
class FastSweepingViewerConfig:
    """First scene, sweep replay, and window settings."""

    seed: int = 0                 # first scene; R advances to the next seed
    waypoints_per_scene: int = 3  # advance after this many reached waypoints; stuck-target changes do not count
    replay: bool = True           # replay the sweeps at every scene start and new waypoint; F replays on demand
    sweep_seconds: float = 1.0    # replay duration in wall-clock seconds
    map_brightness: float = 0.4   # top-down RGB intensity in the route panel
    panel_size: int = 400
    panel_margin: int = 30
    fps: int = 30


# ── Transform / Core Logic ───────────────────────────
def fill(field):
    """Fill unreached cells of a field [H, W] from the nearest reached cell for interpolation."""
    rows, columns = distance_transform_edt(~np.isfinite(field), return_distances=False, return_indices=True)  # nearest reached cell [2, H, W]
    return field[rows, columns]


def upsample(array, size, interpolation):
    """Resize an [H, W] array to [size, size] with an OpenCV interpolation."""
    return cv2.resize(array, (size, size), interpolation=interpolation)


def distance_image(field, free, d_max, size):
    """Travel time [H, W] -> RGB [size, size, 3], with a fixed Viridis scale and separate walls and unreached cells."""
    reached = upsample(np.isfinite(field).astype(np.uint8), size, cv2.INTER_NEAREST) > 0
    value = upsample(fill(field).astype(np.float32), size, cv2.INTER_LINEAR)  # [S, S] smooth travel time
    intensity = ((1 - value / d_max).clip(0, 1) * 255).astype(np.uint8)
    image = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    image[~reached] = MAP_FREE
    image[~free] = MAP_WALL
    return image


# ── Viewer ───────────────────────────────────────────
class FastSweepingViewer:
    """Simulator, fast-sweeping field, and a detailed view of its descending route."""

    def __init__(self, cfg):
        """Start the DMLab simulator, open the window, and load the first scene."""
        # ── Simulator and world coordinates ────────────
        self.cfg, self.paused = cfg, False
        self.simulator = make_simulator("dmlab")
        world = self.simulator.config
        self.bounds = (world.world_min, world.world_min, world.world_max, world.world_max)  # world extent of the scene map

        # ── Window and field legend ────────────────────
        self.window = ViewerWindow("DMLab fast-sweeping exploration", cfg, footer_height=105)
        self.screen, self.clock = self.window.screen, self.window.clock
        self.font_text = self.window.font_text
        self.panels = self.window.panels
        size, margin = cfg.panel_size, cfg.panel_margin
        self.overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        ramp = cv2.applyColorMap(np.linspace(255, 0, size, dtype=np.uint8)[None], cv2.COLORMAP_VIRIDIS)  # [1, S, 3], near -> far
        self.colorbar = numpy_to_surface(cv2.cvtColor(ramp, cv2.COLOR_BGR2RGB), (size, 6))

        self.legend_rect = pygame.Rect(margin, self.panels[0].bottom + 8, self.screen.get_width() - 2 * margin, 40)
        self.status_rect = pygame.Rect(margin, self.panels[0].bottom + 57, self.legend_rect.width, self.font_text.get_linesize())
        self.window.dirty = [self.panels[0], self.panels[1].inflate(24, 24), self.panels[2], self.legend_rect, self.status_rect]
        self.reset(cfg.seed)

    def reset(self, seed):
        """Start maze `seed` with a fresh policy on its scene map."""
        self.seed = seed
        self.state, self.observation, scene_map = self.simulator.reset(seed)
        self.policy = FastSweepingPolicy(scene_map, seed=seed)
        self.explorer = self.policy.explorer
        self.start_position = np.array(self.simulator.map_pose(self.state)[:2], dtype=float)  # scene start in grid (x, y)
        self.free = upsample(self.explorer.binary_map, self.cfg.panel_size, cv2.INTER_NEAREST) > 0  # [S, S] segmented free space
        image = (scene_map.astype(np.float32) * self.cfg.map_brightness).clip(0, 255).astype(np.uint8)
        self.map_surface = numpy_to_surface(image, self.explorer.binary_map.shape[::-1])
        self.load_field()

    def load_field(self):
        """Adopt the current waypoint field and fix its color scale across replay frames."""
        self.dist_map = self.explorer.dist_map
        self.d_max = self.dist_map[np.isfinite(self.dist_map)].max()
        self.route_start = ()  # invalidate the cached route when the waypoint changes
        self.sweeps = None
        if self.cfg.replay:
            self.start_replay()
        else:
            self.surface = self.field_surface(self.dist_map)

    def start_replay(self):
        """Rebuild the current field from its waypoint with the solver's own sweeps, paced by `advance_sweeps`."""
        self.sweeps = replay_sweeps(self.explorer.binary_map, self.explorer.current_target, SWEEP_ROWS)
        self.sweep_start = time.monotonic()

    def field_surface(self, field):
        """Distance-field image of a travel-time grid [H, W]."""
        size = self.cfg.panel_size
        return numpy_to_surface(distance_image(field, self.free, self.d_max, size), (size, size))

    def handle_events(self):
        """Handle scene changes, sweep replay, agent pause, and window events."""
        for event in self.window.events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset(self.seed + 1)
                elif event.key == pygame.K_f:
                    self.start_replay()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

    def advance_agent(self):
        """Step the policy, advance scenes after the waypoint quota, and restart the same scene at episode end."""
        self.state, self.observation, done = self.simulator.step(self.policy.step(self.state))
        if self.explorer.stats.targets_reached >= self.cfg.waypoints_per_scene:
            self.reset(self.seed + 1)
        elif done:
            self.reset(self.seed)
        elif self.explorer.dist_map is not self.dist_map:  # target changes replace the field object
            self.load_field()

    def advance_sweeps(self):
        """Advance the solver replay to elapsed wall time, skipping intermediate display frames."""
        elapsed = min((time.monotonic() - self.sweep_start) / self.cfg.sweep_seconds, 1.0)
        for field, iteration, direction, progress in self.sweeps:
            if progress >= 1 - (1 - elapsed) ** 2:  # quadratic ease-out
                self.surface = self.field_surface(field)
                self.sweep_state = (iteration, direction)
                return
        self.sweeps = None  # display the policy's final field after replay
        self.surface = self.field_surface(self.dist_map)

    def draw_markers(self, rect, scale, origin=(0, 0)):
        """Mark the scene start in green, active and upcoming waypoints in red, and passed waypoints as gray dots."""
        explorer = self.explorer
        for index, (row, column) in enumerate(explorer.targets):
            center = (rect.x + round((column - origin[0]) * scale), rect.y + round((row - origin[1]) * scale))
            if index == explorer.target_idx:
                pygame.draw.circle(self.screen, TRAIL_SHADOW, center, 10)
                pygame.draw.circle(self.screen, ACCENT_RED, center, 8)
                pygame.draw.circle(self.screen, TEXT_PRIMARY, center, 8, 2)
            elif index < explorer.target_idx:
                pygame.draw.circle(self.screen, TEXT_LABEL, center, 3)
            else:
                pygame.draw.circle(self.screen, ACCENT_RED, center, 6, 2)

        start = (self.start_position - origin) * scale
        center = (rect.x + round(start[0]), rect.y + round(start[1]))
        pygame.draw.circle(self.screen, TRAIL_SHADOW, center, 9)
        pygame.draw.circle(self.screen, ACCENT_GREEN, center, 7)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, center, 7, 1)

    def draw_route(self, rect, position, direction, row, column, trail):
        """Fit the descent route and scene start to the RGB map and show the controller's local steering decision."""
        # ── Trace the converged field and find the local steering target ──
        if (row, column) != self.route_start:
            self.route_start = (row, column)
            self.route_path = compute_path_smooth(self.route_start, self.dist_map)[:, ::-1].copy()  # [N, 2], (row, column) -> (x, y)
        grid_size = self.explorer.binary_map.shape[0]
        radius = self.explorer.controller.search_radius
        r0, r1 = max(0, row - radius), min(grid_size, row + radius + 1)
        c0, c1 = max(0, column - radius), min(grid_size, column + radius + 1)
        window = self.dist_map[r0:r1, c0:c1]
        local_row, local_column = np.unravel_index(window.argmin(), window.shape)  # same row-major minimum as AgentController
        steering = np.array([c0 + local_column, r0 + local_row])

        # ── Frame the route and scene start, with at most 3x magnification ──
        points = np.vstack((self.route_path, self.start_position, position, self.explorer.current_target[::-1], (c0, r0), (c1, r1)))
        lower, upper = points.min(axis=0), points.max(axis=0)
        span = min(grid_size, max(math.ceil(grid_size / 3), math.ceil(np.max(upper - lower)) + 16))
        origin = np.clip(np.floor((lower + upper - span) / 2), 0, grid_size - span).astype(int)
        crop = pygame.Rect(int(origin[0]), int(origin[1]), span, span)
        self.screen.blit(pygame.transform.scale(self.map_surface.subsurface(crop), rect.size), rect.topleft)
        scale = rect.width / span
        agent = (np.asarray(position) - origin) * scale
        target = (steering - origin) * scale

        # ── Draw recorded motion, planned descent, and the local controller ──
        self.screen.set_clip(rect)
        search = pygame.Rect(rect.x + (c0 - origin[0]) * scale, rect.y + (r0 - origin[1]) * scale,
                             (c1 - c0) * scale, (r1 - r0) * scale)
        pygame.draw.rect(self.screen, TEXT_LABEL, search, 1)
        draw_trajectory(self.screen, rect, (trail - origin) * scale, TEXT_LABEL, TRAIL_SHADOW)
        draw_trajectory(self.screen, rect, (self.route_path - origin) * scale, ACCENT_BLUE, TRAIL_SHADOW)
        self.draw_markers(rect, scale, origin)
        pygame.draw.line(self.screen, TEXT_PRIMARY, (rect.x + agent[0], rect.y + agent[1]), (rect.x + target[0], rect.y + target[1]), 1)
        draw_agent(self.screen, self.overlay, rect, agent, direction)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, (round(rect.x + target[0]), round(rect.y + target[1])), 4, 1)
        self.screen.set_clip(None)
        return grid_size / span

    def draw(self):
        """Draw the observation, distance field, detailed route, legends, and status line."""
        # ── Titles and agent view ─────────────────────
        self.window.begin(("SIMULATOR", "GEODESIC DISTANCE", "PLANNED ROUTE"))
        agent_rect, map_rect, route_rect = self.panels
        self.screen.blit(numpy_to_surface(self.observation, agent_rect.size), agent_rect.topleft)

        # ── Distance field and detailed route ─────────
        scale = map_rect.width / self.explorer.binary_map.shape[1]  # panel pixels per grid cell
        px, py, heading = map(float, self.simulator.map_pose(self.state))  # pygame rejects numpy float32 positions
        direction = (math.cos(heading), -math.sin(heading))
        position = (map_rect.x + px * scale, map_rect.y + py * scale)
        row, column = self.explorer.mapper.world_to_grid(self.state[0], self.state[1])
        trail = map_pixels(np.asarray(self.explorer.stats.path_history).reshape(-1, 2), self.bounds,
                           self.explorer.binary_map.shape[0])  # [N, 2] world (x, y) -> grid (x, y)
        self.screen.blit(self.surface, map_rect.topleft)
        draw_trajectory(self.screen, map_rect, trail * scale, TEXT_LABEL, TRAIL_SHADOW)
        self.draw_markers(map_rect, scale)
        pygame.draw.circle(self.screen, TRAIL_SHADOW, position, 5)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, position, 3)
        zoom = self.draw_route(route_rect, (px, py), direction, row, column, trail)

        # ── Panel legends ─────────────────────────────
        y = self.legend_rect.y + 2
        pose = f"x {self.state[0]:.1f}    y {self.state[1]:.1f}    yaw {self.state[2]:.1f} deg"
        self.screen.blit(self.font_text.render(pose, True, TEXT_LABEL), (agent_rect.x, y))
        self.screen.blit(self.colorbar, (map_rect.x, y))
        self.screen.blit(self.font_text.render("0 (waypoint)", True, TEXT_LABEL), (map_rect.x, y + 12))
        far = self.font_text.render(f"{self.d_max:.0f} (distance cost)", True, TEXT_LABEL)
        self.screen.blit(far, (map_rect.right - far.get_width(), y + 12))
        self.screen.blit(self.font_text.render("Blue: descent route    Gray: recorded trail", True, TEXT_LABEL), (route_rect.x, y))
        self.screen.blit(self.font_text.render(f"White ring: local target    zoom {zoom:.1f}x", True, TEXT_LABEL), (route_rect.x, y + 20))

        # ── Status and controls ───────────────────────
        explorer, stats = self.explorer, self.explorer.stats
        if self.sweeps is None:
            status = (f"seed {self.seed}    waypoint {explorer.target_idx + 1}/{len(explorer.targets)}    d(x_t) {explorer.get_current_distance(self.state):.1f}"
                      f"    reached {stats.targets_reached}/{self.cfg.waypoints_per_scene}    stuck {stats.stuck_count}    {self.clock.get_fps():.0f} FPS")
        else:
            iteration, (row_step, column_step) = self.sweep_state
            status = (f"seed {self.seed}    fast sweeping, iteration {iteration + 1}:    rows {'↓' if row_step > 0 else '↑'}"
                      f"    columns {'→' if column_step > 0 else '←'}")
        y = self.status_rect.y
        self.screen.blit(self.font_text.render(("paused    " if self.paused else "") + status, True, TEXT_PRIMARY), (agent_rect.x, y))
        self.screen.blit(self.font_text.render("SPACE pause agent    F replay sweeps    R next scene    ESC quit", True, TEXT_LABEL), (agent_rect.x, y + 26))
        legend = self.font_text.render("Green: scene start    Red: waypoint", True, TEXT_LABEL)
        self.screen.blit(legend, (route_rect.right - legend.get_width(), y + 26))

    def run(self):
        """Step the policy and the replay, and redraw at the configured rate until ESC or window close."""
        while self.window.running:
            self.handle_events()
            if not self.window.running:
                break
            if not self.paused:
                self.advance_agent()
            if self.sweeps is not None:
                self.advance_sweeps()
            self.draw()
            self.window.present()
            self.clock.tick(self.cfg.fps)
        self.simulator.close()
        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main(cfg: FastSweepingViewerConfig):
    """Open the viewer on DMLab maze `cfg.seed`."""
    FastSweepingViewer(cfg).run()


if __name__ == "__main__":
    main(pyrallis.parse(config_class=FastSweepingViewerConfig))
