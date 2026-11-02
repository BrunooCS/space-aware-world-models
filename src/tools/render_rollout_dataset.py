#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HDF5 rollout viewer for DMLab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Opens an interactive pygame window with the agent view, top-down trajectory,
and episode statistics for a recorded dataset.

1. Load one episode from the repository's HDF5 dataset format.
2. Map its states onto the stored top-down map and play the frames.

Assumes ego frames were saved. Trajectories use the configured DMLab world range.
Run from the repository root, for example:
    python -m src.tools.render_rollout_dataset --episode 12

Controls: SPACE pause, ←/→ step, N/P next or previous episode, R reset, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import pygame

from src.configs.dmlab import DMLabEnvConfig
from src.tools.render_utils import (ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, TEXT_LABEL, TEXT_PRIMARY, TEXT_SECONDARY,
                                    ViewerWindow, draw_agent, map_pixels, numpy_to_surface)

# ── Config ───────────────────────────────────────────
@dataclass
class ViewerConfig:
    """Window and playback settings for the dataset viewer."""

    window_width: int = 1250
    window_height: int = 550
    panel_size: int = 400
    panel_margin: int = 30
    fps: int = 30



# ── Viewer ───────────────────────────────────────────
class DatasetViewer:
    """Play back DMLab HDF5 episodes."""

    def __init__(self, dataset_path, episode_idx=0):
        """Load the selected dataset episode and create the playback window."""
        self.dataset_path = Path(dataset_path)
        self.env_config = DMLabEnvConfig()
        self.config = ViewerConfig()
        self.current_episode = episode_idx
        self.step = 0
        self.paused = False
        self.fps_history = []

        # ── Window and playback panels ─────────────────
        self.window = ViewerWindow("DMLab Dataset Viewer", self.config, num_panels=2,
                                   window_size=(self.config.window_width, self.config.window_height))
        self.screen, self.clock = self.window.screen, self.window.clock
        self.window.dirty = [self.screen.get_rect()]  # episode statistics and playback controls span the rest of the window
        self.font_stats = pygame.font.SysFont("Arial", 18)
        self.font_label = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_footer = pygame.font.SysFont("Arial", 12)

        margin = self.config.panel_margin
        panel = self.config.panel_size
        self.ego_rect, self.topdown_rect = self.window.panels
        self.panel_y = self.ego_rect.y
        self.stats_x = margin + 2 * (panel + margin)
        self.heading_overlay = pygame.Surface(self.topdown_rect.size, pygame.SRCALPHA)

        # ── Recorded episode ───────────────────────────
        self._load_dataset_info()
        self._load_episode(episode_idx)

    def _load_dataset_info(self):
        """Read the number of episodes in the HDF5 file."""
        with h5py.File(self.dataset_path, "r") as dataset:
            self.num_episodes = dataset["episodes"]["ego"].shape[0]

    def _load_episode(self, episode_idx):
        """Load episode arrays and prepare its map-relative trajectory."""
        with h5py.File(self.dataset_path, "r") as dataset:
            episodes = dataset["episodes"]
            self.ego_frames = episodes["ego"][episode_idx]
            self.states = episodes["state"][episode_idx]
            self.rewards = episodes["reward"][episode_idx]
            self.actions = episodes["action"][episode_idx]
            self.dones = episodes["done"][episode_idx]
            self.topdown = episodes["topdown"][episode_idx]

            self.world_bounds = (self.env_config.world_min, self.env_config.world_min,
                                 self.env_config.world_max, self.env_config.world_max)

        self.num_steps = self.ego_frames.shape[0]
        # [T, d_s] -> [T, 2] map pixels
        self.trajectory = map_pixels(self.states[:, :2], self.world_bounds,
                                     self.config.panel_size - 1).astype(int).tolist()
        self.topdown_surface = numpy_to_surface(self.topdown, self.topdown_rect.size)
        self.current_episode = episode_idx
        self.step = 0

    def _draw_ego_panel(self):
        """Draw the current first-person frame."""
        frame = numpy_to_surface(self.ego_frames[self.step], self.ego_rect.size)
        self.screen.blit(frame, self.ego_rect.topleft)

    def _draw_topdown_panel(self):
        """Draw the map, trajectory, current position, and heading."""
        self.screen.blit(self.topdown_surface, self.topdown_rect.topleft)

        if self.step > 0:
            points = [(self.topdown_rect.x + x, self.topdown_rect.y + y)
                      for x, y in self.trajectory[:self.step + 1]]
            pygame.draw.lines(self.screen, ACCENT_BLUE, False, points, 3)

        position = self.trajectory[self.step]
        heading = float(self.states[self.step][2])
        angle = math.radians(heading)  # DMLab yaw is in degrees and faces +X at zero
        min_x, min_y, max_x, max_y = self.world_bounds
        direction = (math.cos(angle) * self.topdown_rect.width / (max_x - min_x),
                     -math.sin(angle) * self.topdown_rect.height / (max_y - min_y))
        magnitude = math.hypot(*direction)
        direction = (direction[0] / magnitude, direction[1] / magnitude)
        draw_agent(self.screen, self.heading_overlay, self.topdown_rect, position, direction)

    def _draw_stats_panel(self):
        """Draw episode, position, action, and playback statistics."""
        x, y = self.stats_x, self.panel_y
        state = self.states[self.step]
        action = self.actions[self.step]
        heading = float(state[2])

        self.fps_history.append(self.clock.get_fps())
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        fps = sum(self.fps_history) / len(self.fps_history)

        self._draw_stat_section(x, y, "EPISODE", [
            ("Episode", f"{self.current_episode} / {self.num_episodes - 1}", TEXT_PRIMARY),
            ("Step", f"{self.step} / {self.num_steps - 1}", TEXT_PRIMARY),
            ("Done", f"{bool(self.dones[self.step])}", TEXT_PRIMARY),
        ])
        self._draw_stat_section(x, y + 120, "POSITION", [
            ("X", f"{state[0]:.1f}", TEXT_PRIMARY),
            ("Y", f"{state[1]:.1f}", TEXT_PRIMARY),
            ("Heading", f"{heading:.1f}°", TEXT_PRIMARY),
        ])

        turn_idx, forward_idx = self.env_config.action_turn_idx, self.env_config.action_forward_idx
        action_rows = [(self.env_config.action_raw_names[i].title(), f"{float(action[i]):+.2f}", TEXT_PRIMARY)
                       for i in (turn_idx, forward_idx)]
        self._draw_stat_section(x, y + 240, "ACTION", [
            ("Reward", f"{float(self.rewards[self.step]):+.3f}", TEXT_PRIMARY),
            *action_rows,
        ])

        status = "PAUSED" if self.paused else "PLAYING"
        status_color = ACCENT_RED if self.paused else ACCENT_GREEN
        self._draw_stat_section(x, y + 375, "PLAYBACK", [
            ("Status", status, status_color),
            ("FPS", f"{fps:.1f}", TEXT_PRIMARY),
        ])

    def _draw_stat_section(self, x, y, title, items):
        """Draw a titled group of colored key-value rows."""
        self.screen.blit(self.font_label.render(title, True, TEXT_LABEL), (x, y))
        y_offset = y + 22
        for label, value, color in items:
            self.screen.blit(self.font_stats.render(f"{label}:", True, TEXT_SECONDARY), (x, y_offset))
            self.screen.blit(self.font_stats.render(value, True, color), (x + 100, y_offset))
            y_offset += 26

    def _draw_footer(self):
        """Draw keyboard controls at the bottom of the window."""
        text = self.font_footer.render("ESC Quit   SPACE Pause   ←→ Step   N/P Episode   R Reset",
                                      True, TEXT_LABEL)
        self.screen.blit(text, text.get_rect(midbottom=(self.config.window_width // 2,
                                                       self.config.window_height - 8)))

    def _handle_events(self):
        """Process pygame keyboard events and window close requests."""
        for event in self.window.events():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self.step = min(self.step + 1, self.num_steps - 1)
                elif event.key == pygame.K_LEFT:
                    self.step = max(self.step - 1, 0)
                elif event.key == pygame.K_r:
                    self.step = 0
                elif event.key == pygame.K_n:
                    self._load_episode((self.current_episode + 1) % self.num_episodes)
                elif event.key == pygame.K_p:
                    self._load_episode((self.current_episode - 1) % self.num_episodes)

    def run(self):
        """Play episodes at the configured frame rate until the window closes."""
        print(f"[INFO] Dataset: {self.dataset_path} ({self.num_episodes} episodes)")

        while self.window.running:
            # ── Draw the current frame ─────────────────
            self._handle_events()
            if not self.window.running:
                break
            self.window.begin(("AGENT VIEW", "TOP-DOWN PATH"))
            self._draw_ego_panel()
            self._draw_topdown_panel()
            self._draw_stats_panel()
            self._draw_footer()
            self.window.present()
            self.clock.tick(self.config.fps)

            # ── Advance playback ───────────────────────
            if not self.paused:
                self.step += 1
                if self.step >= self.num_steps:
                    self._load_episode((self.current_episode + 1) % self.num_episodes)

        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main():
    """Parse viewer options and launch the requested dataset."""
    parser = argparse.ArgumentParser(description="View DMLab HDF5 rollout episodes.")
    parser.add_argument("--dataset", type=Path, default=Path("data/DMLabRolloutDataset.h5"), help="Rollout HDF5 file.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to start from.")
    args = parser.parse_args()

    DatasetViewer(args.dataset, args.episode).run()


if __name__ == "__main__":
    main()
