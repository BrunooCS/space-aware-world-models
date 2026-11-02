#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interactive observation-model viewer for DMLab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare simulator observations with generator renders at the same state and show
the decoded scene map with the agent pose.

1) Load the environment and scene encoder selected by the generator checkpoint.
2) Encode each scene map once to its posterior mean z, then render keyboard-driven states.

Default rollout episodes use seeds >= 101.

Example:
    python -m src.tools.render_generator --checkpoint checkpoints/generator_dmlab/version_0/last.ckpt

Controls: W/↑ forward, S/↓ back, A/← and D/→ turn, left-click map to move, R next scene, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pygame
import pyrallis
import torch

from src.envs import make_simulator
from src.models.dds_vae import DDSVAE
from src.models.generator import Generator
from src.tools.render_models import encode_scene
from src.tools.render_utils import (TEXT_LABEL, TEXT_PRIMARY, ViewerWindow, draw_agent, keyboard_axes, numpy_to_surface,
                                    panel_to_image, to_image)


# ── Config ───────────────────────────────────────────
@dataclass
class GeneratorViewerConfig:
    """Generator checkpoint, first scene, and window settings."""

    checkpoint: str                  # train_generator checkpoint
    seed: int = 0                    # first scene; R advances to the next seed
    device: str = "cuda"
    panel_size: int = 400
    panel_margin: int = 30
    fps: int = 30


# ── Viewer ───────────────────────────────────────────
class GeneratorViewer:
    """Simulator, generator, and scene-map panels for one generator checkpoint."""

    def __init__(self, cfg):
        """Load the renderer and its scene encoder, start the checkpoint's simulator, and open the window."""
        # ── Models and simulator ───────────────────────
        self.cfg, self.device = cfg, cfg.device
        hparams = torch.load(cfg.checkpoint, map_location="cpu")["hyper_parameters"]
        self.generator = Generator.from_checkpoint(cfg.checkpoint, cfg.device)
        self.scene_encoder = DDSVAE.from_checkpoint(hparams["scene_encoder"], cfg.device)
        self.simulator = make_simulator(hparams["env"])

        # ── Window and first scene ─────────────────────
        self.window = ViewerWindow(f"{hparams['env']} observation model", cfg)
        self.screen, self.clock = self.window.screen, self.window.clock
        self.font_text = self.window.font_text
        self.panels, self.status_rect = self.window.panels, self.window.status_rect
        self.overlay = pygame.Surface(self.panels[2].size, pygame.SRCALPHA)
        self.window.dirty = [*self.panels[:2], self.panels[2].inflate(24, 24), self.status_rect]  # the agent dot may overhang the map
        self.reset(cfg.seed)

    @torch.inference_mode()
    def reset(self, seed):
        """Start scene `seed` and encode its map to the posterior mean z."""
        self.seed = seed
        self.state, self.observation, scene_map = self.simulator.reset(seed)
        self.z = encode_scene(self.scene_encoder, scene_map, self.device)  # posterior mean [1, d_z]
        self.map_reconstruction = to_image(self.scene_encoder.decode(self.z))[0]

    def place(self, position):
        """Move the simulator to a window pixel on the decoded scene map and refresh its observation."""
        rect = self.panels[2]
        px, py = panel_to_image(position, rect, self.map_reconstruction.shape[:2])
        self.state, self.observation, done = self.simulator.place(px, py)
        if done:
            self.reset(self.seed)

    def handle_events(self):
        """Handle quitting, scene changes, left-click map placement, and window re-exposure."""
        for event in self.window.events():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset(self.seed + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.panels[2].collidepoint(event.pos):
                self.place(event.pos)

    @torch.inference_mode()
    def draw(self):
        """Render o_hat_t at the current state and draw the panels, the agent pose, and the status line."""
        s = torch.from_numpy(self.simulator.config.normalize_state(self.state)).to(self.device)[None]  # [1, d_s]
        render = to_image(self.generator(self.z, s))[0]
        self.window.begin(("SIMULATOR", "GENERATOR", "SCENE CODE (DDS-VAE DECODE)"))
        for rect, image in zip(self.panels, (self.observation, render, self.map_reconstruction)):
            self.screen.blit(numpy_to_surface(image, rect.size), rect.topleft)

        # ── Agent pose on the map panel ───────────────
        px, py, heading = self.simulator.map_pose(self.state)
        scale = self.panels[2].width / self.map_reconstruction.shape[1]
        draw_agent(self.screen, self.overlay, self.panels[2], (px * scale, py * scale), (math.cos(heading), -math.sin(heading)))

        # ── Status and controls ───────────────────────
        state = np.array2string(s[0].cpu().numpy(), precision=2, suppress_small=True)
        status = f"seed {self.seed}    s_t {state}    alpha {self.generator.alpha.item():.3f}    {self.clock.get_fps():.0f} FPS"
        y = self.status_rect.y
        self.screen.blit(self.font_text.render(status, True, TEXT_PRIMARY), (self.panels[0].x, y))
        controls = "W/↑ forward    S/↓ back    A/← D/→ turn    Left-click map to move    R next scene    ESC quit"
        self.screen.blit(self.font_text.render(controls, True, TEXT_LABEL), (self.panels[0].x, y + 26))

    def run(self):
        """Step the simulator with the held keys and redraw at the configured rate until ESC or window close."""
        while self.window.running:
            self.handle_events()
            if not self.window.running:
                break
            forward, turn = keyboard_axes()
            self.state, self.observation, done = self.simulator.step(self.simulator.config.manual_action(forward, turn))
            if done:
                self.reset(self.seed)
            self.draw()
            self.window.present()
            self.clock.tick(self.cfg.fps)
        self.simulator.close()
        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main(cfg: GeneratorViewerConfig):
    """Open the viewer for one generator checkpoint."""
    GeneratorViewer(cfg).run()


if __name__ == "__main__":
    main(pyrallis.parse(config_class=GeneratorViewerConfig))
