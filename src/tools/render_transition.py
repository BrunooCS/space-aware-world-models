#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interactive SWM viewer: the simulator against the free-running world model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare the simulator with free-running SWM predictions under the same keyboard actions.
Panels show the simulator observation, renders at the simulator and predicted states,
and both trajectories on the scene map.

The transition checkpoint selects the environment; the generator must share its scene
encoder. Each scene uses the map's posterior mean z. Predictions advance on idle frames
and restart from the simulator state on SPACE, map placement, scene changes, or termination.
DMLab backward commands normalize to zero forward motion, so S/↓ moves only the simulator.

Example:
    python -m src.tools.render_transition --checkpoint checkpoints/transition_dmlab/version_0/last.ckpt \
        --generator checkpoints/generator_dmlab/version_0/last.ckpt

Controls: W/↑ forward, S/↓ back, A/← and D/→ turn, left-click map to move, SPACE resync, R next scene, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pygame
import pyrallis
import torch

from src.envs import make_simulator
from src.tools.render_models import encode_scene, load_swm
from src.tools.render_utils import (ACCENT_BLUE, TEXT_LABEL, TEXT_PRIMARY, TRAIL_SHADOW, ViewerWindow, draw_agent,
                                    draw_reference, draw_trajectory, keyboard_axes, numpy_to_surface, panel_to_image, to_image)


# ── Config ───────────────────────────────────────────
@dataclass
class TransitionViewerConfig:
    """Transition and generator checkpoints, first scene, and window settings."""

    checkpoint: str                  # train_transition checkpoint
    generator: str                   # train_generator checkpoint on the same scene encoder
    seed: int = 0                    # first scene; R advances to the next seed
    device: str = "cuda"
    panel_size: int = 288
    panel_margin: int = 24
    trail: int = 300                 # most recent steps drawn on the map
    fps: int = 30


# ── Viewer ───────────────────────────────────────────
class TransitionViewer:
    """Simulator, render-at-state, SWM, and scene-map panels for one transition checkpoint."""

    TITLES = ("SIMULATOR", "RENDER AT SIMULATOR STATE", "SWM (FREE-RUNNING STATE)", "SCENE MAP")

    def __init__(self, cfg):
        """Load SWM, start the checkpoint's simulator, and open the window."""
        # ── Models and simulator ───────────────────────
        self.cfg, self.device = cfg, cfg.device
        env, self.scene_encoder, self.generator, self.transition = load_swm(cfg.checkpoint, cfg.generator, cfg.device)
        self.simulator = make_simulator(env)
        self.config = self.simulator.config

        # ── Window and first scene ─────────────────────
        self.window = ViewerWindow(f"{env} SWM", cfg, num_panels=4)
        self.screen, self.clock = self.window.screen, self.window.clock
        self.font_text = self.window.font_text
        self.panels, self.status_rect = self.window.panels, self.window.status_rect
        self.overlay = pygame.Surface(self.panels[3].size, pygame.SRCALPHA)
        self.map_area = self.panels[3].inflate(24, 24)  # markers may overhang the map panel
        self.window.dirty = [*self.panels[:3], self.map_area, self.status_rect]
        self.reset(cfg.seed)

    def state_tensor(self, raw_state):
        """Normalized state s_t [1, d_s] on the device from a raw simulator state."""
        return torch.from_numpy(self.config.normalize_state(raw_state)).to(self.device)[None]

    @torch.inference_mode()
    def reset(self, seed):
        """Start scene `seed`, encode its map to the posterior mean, and initialize the prediction from the simulator state."""
        self.seed = seed
        self.state, self.observation, self.scene_map = self.simulator.reset(seed)
        self.z = encode_scene(self.scene_encoder, self.scene_map, self.device)  # posterior mean [1, d_z]
        self.map_surface = numpy_to_surface(self.scene_map, self.panels[3].size)
        self.resync()

    def resync(self):
        """Restart the free-running prediction from the simulator state, with an empty LSTM state and empty trails."""
        self.s_hat, self.hidden, self.steps = self.state_tensor(self.state), None, 0
        self.trails = (deque(maxlen=self.cfg.trail), deque(maxlen=self.cfg.trail))
        self.track()

    def place(self, position):
        """Move the simulator to a window pixel on the scene map and restart SWM from its new state."""
        rect = self.panels[3]
        px, py = panel_to_image(position, rect, self.scene_map.shape[:2])
        self.state, self.observation, done = self.simulator.place(px, py)
        if done:
            self.reset(self.seed)
        else:
            self.resync()

    def track(self):
        """Map-panel poses (px, py, heading) of the simulator and SWM states; their positions extend the trails."""
        scale = self.cfg.panel_size / self.scene_map.shape[1]
        raw_hat = self.config.denormalize_state(self.s_hat[0].cpu().numpy())
        self.poses = [(px * scale, py * scale, heading) for px, py, heading in map(self.simulator.map_pose, (self.state, raw_hat))]
        for trail, (px, py, _) in zip(self.trails, self.poses):
            trail.append((px, py))

    def handle_events(self):
        """Handle quitting, scene changes, resyncing, left-click map placement, and window re-exposure."""
        for event in self.window.events():
            key = event.key if event.type == pygame.KEYDOWN else None
            if key == pygame.K_r:
                self.reset(self.seed + 1)
            elif key == pygame.K_SPACE:
                self.resync()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.panels[3].collidepoint(event.pos):
                self.place(event.pos)

    @torch.inference_mode()
    def step(self, action):
        """Apply `action` to the simulator and, normalized, to the free-running SWM state; an episode end restarts the scene."""
        a = torch.from_numpy(self.config.normalize_action(action)).to(self.device)[None]  # [1, d_a]
        self.s_hat, _, self.hidden = self.transition.step(self.s_hat, self.z, a, self.hidden)
        self.state, self.observation, done = self.simulator.step(action)
        if done:
            self.reset(self.seed)
        else:
            self.steps += 1
            self.track()

    @torch.inference_mode()
    def draw(self):
        """Render o_hat_t at the simulator and SWM states in one batch; draw the panels, trajectories, poses, and status line."""
        s = self.state_tensor(self.state)
        renders = to_image(self.generator(self.z.expand(2, -1), torch.cat([s, self.s_hat])))  # [2, H, W, 3]
        self.window.begin(self.TITLES)
        for rect, image in zip(self.panels, (self.observation, *renders)):
            self.screen.blit(numpy_to_surface(image, rect.size), rect.topleft)

        # ── Scene map with both trajectories and poses ─
        rect = self.panels[3]
        self.screen.blit(self.map_surface, rect.topleft)
        self.screen.set_clip(self.map_area)
        draw_trajectory(self.screen, rect, self.trails[0], TEXT_PRIMARY, TRAIL_SHADOW)
        draw_trajectory(self.screen, rect, self.trails[1], ACCENT_BLUE, TEXT_PRIMARY)
        (px, py, heading), (qx, qy, heading_hat) = self.poses
        draw_reference(self.screen, rect, (px, py), (math.cos(heading), -math.sin(heading)))
        draw_agent(self.screen, self.overlay, rect, (qx, qy), (math.cos(heading_hat), -math.sin(heading_hat)))
        self.screen.set_clip(None)

        # ── Status and controls ───────────────────────
        position_error = (s - self.s_hat)[0, :2].norm().item()
        heading_error = math.degrees(abs(math.remainder(heading_hat - heading, math.tau)))
        status = (f"seed {self.seed}    step {self.steps}    position error {position_error:.3f}    heading error {heading_error:.1f}°    "
                  f"{self.clock.get_fps():.0f} FPS")
        y = self.status_rect.y
        self.screen.blit(self.font_text.render(status, True, TEXT_PRIMARY), (self.panels[0].x, y))
        controls = "W/↑ forward    S/↓ back    A/← D/→ turn    Left-click map to move    SPACE resync    R next scene    ESC quit"
        self.screen.blit(self.font_text.render(controls, True, TEXT_LABEL), (self.panels[0].x, y + 26))

    def run(self):
        """Step the simulator and SWM with the held keys and redraw at the configured rate until ESC or window close."""
        while self.window.running:
            self.handle_events()
            if not self.window.running:
                break
            forward, turn = keyboard_axes()
            self.step(self.config.manual_action(forward, turn))
            self.draw()
            self.window.present()
            self.clock.tick(self.cfg.fps)
        self.simulator.close()
        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main(cfg: TransitionViewerConfig):
    """Open the viewer for one transition checkpoint."""
    torch.backends.cudnn.allow_tf32 = False  # limit accumulated rollout error
    TransitionViewer(cfg).run()


if __name__ == "__main__":
    main(pyrallis.parse(config_class=TransitionViewerConfig))
