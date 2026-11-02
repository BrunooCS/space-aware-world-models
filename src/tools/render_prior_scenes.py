#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interactive SWM explorer for scenes sampled from the DDS-VAE prior
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) Draw epsilon ~ N(0, I) from a seed and set z = temperature * epsilon.
2) Decode the scene map and render G_theta(z, pi_s(s_hat_t)) from the same code.
3) Navigate with the transition model, or place and orient the agent on the map.

The viewer shows the SWM view, decoded map, and temperature presets in [0.25, 1].
Temperature changes retain epsilon, the agent pose, and recurrent state. Dragging
code bars edits epsilon_i = z_i / temperature without resetting the agent.

By default the sampled code is used directly. Optional `--refinements` decode -> encode
passes replace it with posterior means; the resulting code is no longer a direct
Gaussian draw. Unless paused, transitions advance on idle frames. DMLab backward
commands normalize to zero forward motion.

With `--video_dir`, every scene visit is recorded as an MP4 clip of `video_steps`
frames, after which the viewer moves to the next seed. The map overlay uses the
environment's `world_to_pixel`, exact for DMLab since all maps share one world range.

Examples:
    python -m src.tools.render_prior_scenes --checkpoint checkpoints/transition_dmlab/version_0/last.ckpt \
        --generator checkpoints/generator_dmlab/version_0/last.ckpt --seeds "[22,1,7]" --temperature 0.75

Controls: W/↑ forward, S/↓ no-op, A/← D/→ turn, SPACE pause, . step, G new sample,
N/P seed history, R restore the drawn sample and reset the pose, [/] or wheel temperature, 1-4 presets,
drag bars edit z, left-click map to place, right-click map to face, click a tile or drag the slider to set the temperature, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pygame
import pyrallis
import torch

from src.configs import ENV_CONFIGS
from src.tools.render_models import load_swm
from src.tools.render_utils import (ACCENT_BLUE, TEXT_LABEL, TEXT_PRIMARY, TEXT_SECONDARY, TRAIL_SHADOW, ViewerWindow,
                                    code_strip_value, draw_agent, draw_code_strip, draw_trajectory, keyboard_axes, numpy_to_surface, to_image)


# ── Config ───────────────────────────────────────────
@dataclass
class PriorSceneViewerConfig:
    """Checkpoints, prior sampling, start state, window, and optional recording."""

    checkpoint: str                  # train_transition checkpoint
    generator: str                   # train_generator checkpoint on the same scene encoder
    seeds: list[int] = field(default_factory=lambda: [22, 1])  # prior samples; N/P cycle through them
    temperature: float = 1.0         # z = temperature * eps, eps ~ N(0, I); within [0.25, 1]
    temperature_step: float = 0.05   # increment for brackets and the mouse wheel
    code_range: float = 2.0          # half-height of the code strip, in standard deviations of the prior
    refinements: int = 0             # optional decode -> encode passes; 0 keeps the Gaussian sample
    start_state: list[float] = field(default_factory=lambda: [550.0, 550.0, 0.0])  # raw simulator state; DMLab: [x, y, theta_deg]
    device: str = "cuda"
    panel_size: int = 400
    panel_margin: int = 30
    trail: int = 512                 # most recent steps drawn on the map
    fps: int = 30
    video_dir: Optional[str] = None  # record MP4 clips into this directory
    video_steps: int = 600           # frames per clip


# ── Viewer ───────────────────────────────────────────
class PriorSceneViewer:
    """Interactive prior samples with the SWM view, scene map, temperature grid, and optional clip recording."""

    TEMPERATURES = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, cfg):
        """Load SWM, arrange the views and sampling controls, and prepare the first scene."""
        self.cfg, self.device = cfg, cfg.device
        env, self.scene_encoder, self.generator, self.transition = load_swm(cfg.checkpoint, cfg.generator, cfg.device)
        self.env, self.config = env, ENV_CONFIGS[env]()
        self.seeds, self.temperature = list(cfg.seeds), cfg.temperature
        self.paused, self.advance_once = False, False

        # ── Observation, scene map, and temperature grid ──
        self.window = ViewerWindow(f"{env} SWM in prior scenes", cfg, footer_height=201)
        self.screen, self.clock = self.window.screen, self.window.clock
        self.font_title, self.font_text = self.window.font_title, self.window.font_text
        self.panels = self.window.panels
        size, margin = cfg.panel_size, cfg.panel_margin
        tile = (size - 48) // 2  # the slider takes the bottom of the panel
        self.previews = [pygame.Rect(self.panels[2].x + i % 2 * (tile + 4), self.panels[2].y + i // 2 * (tile + 4), tile, tile)
                         for i in range(len(self.TEMPERATURES))]
        self.slider = pygame.Rect(self.panels[2].x + 10, self.panels[2].bottom - 18, size - 20, 4)
        self.strip = pygame.Rect(margin, self.panels[0].bottom + 40, self.screen.get_width() - 2 * margin, 80)
        self.status_rect = pygame.Rect(margin, self.strip.bottom + 16, self.strip.width - 260, self.font_text.get_linesize())
        self.sample_button = pygame.Rect(self.strip.right - 120, self.status_rect.y - 5, 120, 26)
        self.reset_button = pygame.Rect(self.sample_button.x - 88, self.sample_button.y, 80, 26)
        self.overlay = pygame.Surface((size, size), pygame.SRCALPHA)

        self.window.dirty = [*self.panels, self.strip.inflate(0, 8), self.status_rect]
        self.drag, self.last_dim = None, None

        # ── Scene history and optional recording ───────
        self.writer, self.clips, self.index = None, 0, 0
        if cfg.video_dir:
            Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
        self.load_scene()

    def load_scene(self):
        """Draw one seeded Gaussian direction; edits to epsilon start from this sample."""
        generator = torch.Generator(self.device).manual_seed(self.seeds[self.index])
        self.sample = torch.randn(1, self.scene_encoder.latent_dim, generator=generator, device=self.device)  # [1, d_z]
        self.epsilon = self.sample.clone()
        self.set_code()
        self.reset_pose()

    def refine(self, z):
        """Apply the configured number of decode -> posterior-mean encode passes."""
        for _ in range(self.cfg.refinements):
            z = self.scene_encoder.encode(self.scene_encoder.decode(z))[0]  # posterior mean of the decoded map
        return z

    def set_temperature(self, temperature):
        """Set temperature without resetting pose or recurrent state."""
        self.temperature = round(min(max(temperature, self.TEMPERATURES[0]), self.TEMPERATURES[-1]), 4)
        self.set_code()

    @torch.inference_mode()
    def set_code(self):
        """Decode current and preset temperatures in one batch without resetting the agent."""
        temperatures = torch.tensor((*self.TEMPERATURES, self.temperature), device=self.device)[:, None]  # [K + 1, 1]
        codes = self.refine(temperatures * self.epsilon)  # [K + 1, d_z]: the same epsilon at each temperature, then the current one
        maps = to_image(self.scene_encoder.decode(codes))  # [K + 1, 128, 128, 3]
        self.preview_surfaces = [numpy_to_surface(image, rect.size) for image, rect in zip(maps, self.previews)]
        self.z = codes[-1:]  # [1, d_z], shared by the map, renderer, and dynamics
        self.z_rms = self.z.square().mean().sqrt().item()
        self.bars, self.ticks = self.z[0].cpu().numpy(), (self.temperature * self.sample[0]).cpu().numpy()
        self.scene_map = maps[-1]
        self.map_surface = numpy_to_surface(self.scene_map, self.panels[1].size)
        self.frame_dirty = True

    def drag_to(self, position):
        """Edit z on the strip, or move the temperature knob in steps of `temperature_step`."""
        if self.drag == "strip":
            self.edit_code(position)
        else:
            low, high = self.TEMPERATURES[0], self.TEMPERATURES[-1]
            fraction = np.clip((position[0] - self.slider.x) / self.slider.width, 0, 1)
            temperature = round(low + round(fraction * (high - low) / self.cfg.temperature_step) * self.cfg.temperature_step, 4)
            if temperature != self.temperature:
                self.set_temperature(temperature)

    def edit_code(self, position):
        """Set the epsilon coordinates under the cursor on the strip so that z takes the pointed value."""
        dim, value = code_strip_value(position, self.strip, len(self.bars), self.cfg.code_range)
        last = dim if self.last_dim is None else self.last_dim  # interpolate coordinates skipped between mouse events
        self.epsilon[0, min(last, dim):max(last, dim) + 1] = value / self.temperature
        self.last_dim = dim
        self.set_code()

    @torch.inference_mode()
    def reset_pose(self, position=None):
        """Restart at the configured state, or teleport to a window pixel while preserving heading."""
        # ── Convert a map click through the environment's affine world-to-pixel transform ──
        raw = np.float32(self.cfg.start_state) if position is None else self.raw_state.copy()
        if position is not None:
            rect = self.panels[1]
            origin = np.asarray(self.config.world_to_pixel(0.0, 0.0, rect.width))
            scale = np.asarray(self.config.world_to_pixel(1.0, 1.0, rect.width)) - origin
            raw[:2] = (np.asarray(position) - np.asarray(rect.topleft) - origin) / scale

        # ── Reset recurrent state, history, and the recording segment ──
        self.s_hat = torch.from_numpy(self.config.normalize_state(raw)).to(self.device)[None]  # [1, d_s]
        self.hidden, self.steps, self.trail = None, 0, deque(maxlen=self.cfg.trail)
        self.track()
        if self.cfg.video_dir:
            self.start_clip()

    def track(self):
        """Map-panel pose (px, py, heading) of the SWM state; its position extends the trail."""
        raw = self.raw_state = self.config.denormalize_state(self.s_hat[0].cpu().numpy())
        px, py = self.config.world_to_pixel(raw[0], raw[1], self.scene_map.shape[1])
        scale = self.cfg.panel_size / self.scene_map.shape[1]
        self.pose = (px * scale, py * scale, self.config.map_heading(raw))
        self.trail.append(self.pose[:2])
        self.frame_dirty = True

    @torch.inference_mode()
    def face(self, position):
        """Point the agent toward a window pixel and restart its recurrent state at rest."""
        rect = self.panels[1]
        heading = math.atan2(self.pose[1] - (position[1] - rect.y), position[0] - rect.x - self.pose[0])
        offset = self.config.map_heading(np.zeros(self.config.state_raw_dim, dtype=np.float32))
        # s_hat stores sin(theta), cos(theta) at indices 2:4.
        self.s_hat[0, 2], self.s_hat[0, 3] = math.sin(heading - offset), math.cos(heading - offset)
        self.hidden = None
        self.track()

    def new_scene(self):
        """Append an unused seed to the scene history and load its sample."""
        self.seeds.append(max(self.seeds) + 1)
        self.index = len(self.seeds) - 1
        self.load_scene()

    def switch(self, offset):
        """Move `offset` seeds forward (cyclically) and load that prior scene."""
        self.index = (self.index + offset) % len(self.seeds)
        self.load_scene()

    def click(self, position):
        """Apply a preview tile's temperature, press a button, or place the agent on the scene map."""
        for temperature, rect in zip(self.TEMPERATURES, self.previews):
            if rect.collidepoint(position):
                self.set_temperature(temperature)
        if self.panels[1].collidepoint(position):
            self.reset_pose(position)
        elif self.reset_button.collidepoint(position):
            self.load_scene()
        elif self.sample_button.collidepoint(position):
            self.new_scene()

    def handle_events(self):
        """Handle scene sampling, temperature, code edits, playback, map placement, and window events."""
        for event in self.window.events():
            key = event.key if event.type == pygame.KEYDOWN else None
            if key == pygame.K_r:
                self.load_scene()
            elif key == pygame.K_g:
                self.new_scene()
            elif key in (pygame.K_n, pygame.K_p):
                self.switch(1 if key == pygame.K_n else -1)
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_PERIOD:
                self.paused, self.advance_once = True, True
            elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                self.set_temperature(self.temperature + (1 if key == pygame.K_RIGHTBRACKET else -1) * self.cfg.temperature_step)
            elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                self.set_temperature(self.TEMPERATURES[key - pygame.K_1])
            elif event.type == pygame.MOUSEWHEEL:
                self.set_temperature(self.temperature + event.y * self.cfg.temperature_step)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.drag = "strip" if self.strip.inflate(0, 8).collidepoint(event.pos) else "slider" if self.slider.inflate(16, 30).collidepoint(event.pos) else None
                self.last_dim = None
                if self.drag:
                    self.drag_to(event.pos)
                else:
                    self.click(event.pos)
            elif event.type == pygame.MOUSEMOTION and self.drag:
                self.drag_to(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drag = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.panels[1].collidepoint(event.pos):
                self.face(event.pos)

    @torch.inference_mode()
    def step(self, action):
        """Advance the SWM state with the normalized `action`."""
        a = torch.from_numpy(self.config.normalize_action(action)).to(self.device)[None]  # [1, d_a]
        self.s_hat, _, self.hidden = self.transition.step(self.s_hat, self.z, a, self.hidden)
        self.steps += 1
        self.track()

    # ── Output / IO ──────────────────────────────────
    @torch.inference_mode()
    def draw(self):
        """Draw the SWM view, the scene map with the agent, the temperature grid, and the status line."""
        # ── Render only when the scene or state changes ─
        if self.frame_dirty:
            render = to_image(self.generator(self.z, self.s_hat))[0]
            self.render_surface = numpy_to_surface(render, self.panels[0].size)
            self.frame_dirty = False
        self.window.begin(("SWM VIEW", "SAMPLED SCENE MAP", "TEMPERATURE  (same ε)"))
        self.screen.blit(self.render_surface, self.panels[0].topleft)

        # ── Decoded scene map with the trajectory and pose ─
        rect = self.panels[1]
        self.screen.blit(self.map_surface, rect.topleft)
        self.screen.set_clip(rect)
        draw_trajectory(self.screen, rect, self.trail, ACCENT_BLUE, TRAIL_SHADOW)
        px, py, heading = self.pose
        draw_agent(self.screen, self.overlay, rect, (px, py), (math.cos(heading), -math.sin(heading)))
        self.screen.set_clip(None)

        # ── Temperature grid: the selected tile is outlined in blue ─
        for temperature, tile, surface in zip(self.TEMPERATURES, self.previews, self.preview_surfaces):
            self.screen.blit(surface, tile.topleft)
            selected = math.isclose(self.temperature, temperature)
            label = self.font_text.render(f"T = {temperature:g}", True, TEXT_PRIMARY if selected else TEXT_SECONDARY)
            pygame.draw.rect(self.screen, TRAIL_SHADOW, label.get_rect(topleft=(tile.x + 6, tile.y + 6)).inflate(10, 4))
            self.screen.blit(label, (tile.x + 11, tile.y + 8))
            if selected:
                pygame.draw.rect(self.screen, ACCENT_BLUE, tile, 3)

        # ── Temperature slider: ticks at the tile temperatures ─
        low, high = self.TEMPERATURES[0], self.TEMPERATURES[-1]
        knob_x = self.slider.x + round((self.temperature - low) / (high - low) * self.slider.width)
        pygame.draw.line(self.screen, TEXT_LABEL, self.slider.midleft, self.slider.midright, 2)
        pygame.draw.line(self.screen, ACCENT_BLUE, self.slider.midleft, (knob_x, self.slider.centery), 2)
        for temperature in self.TEMPERATURES:
            x = self.slider.x + round((temperature - low) / (high - low) * self.slider.width)
            pygame.draw.line(self.screen, TEXT_LABEL, (x, self.slider.centery - 6), (x, self.slider.centery + 6), 1)
        pygame.draw.circle(self.screen, TRAIL_SHADOW, (knob_x, self.slider.centery), 10)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, (knob_x, self.slider.centery), 8)
        pygame.draw.circle(self.screen, ACCENT_BLUE, (knob_x, self.slider.centery), 5)
        value = self.font_text.render(f"T = {self.temperature:.2f}", True, TEXT_PRIMARY)
        self.screen.blit(value, (min(max(knob_x - value.get_width() // 2, self.panels[2].x), self.panels[2].right - value.get_width()), self.slider.y - 28))

        # ── Code strip ────────────────────────────────
        self.screen.blit(self.font_title.render(f"SAMPLED CODE z = T × ε  (d_z = {len(self.bars)})", True, TEXT_SECONDARY), (self.strip.x, self.strip.y - 24))
        legend = self.font_text.render("bars: z (brighter = larger)    ticks: drawn sample", True, TEXT_LABEL)
        self.screen.blit(legend, (self.strip.right - legend.get_width(), self.strip.y - 22))
        draw_code_strip(self.screen, self.strip, self.bars, self.ticks, self.cfg.code_range)

        # ── Status and controls ───────────────────────
        status = (f"seed {self.seeds[self.index]}    T {self.temperature:.2f}    z rms {self.z_rms:.2f}    step {self.steps}"
                  f"    x {self.raw_state[0]:.1f}  y {self.raw_state[1]:.1f}    heading {math.degrees(self.pose[2]) % 360:.0f}°"
                  f"    {self.clock.get_fps():.0f} FPS" + ("    PAUSED" if self.paused else "") + ("    REC" if self.writer is not None else ""))
        y, x = self.status_rect.y, self.panels[0].x
        self.screen.blit(self.font_text.render(status, True, TEXT_PRIMARY), (x, y))
        for button, label in ((self.reset_button, "Reset  R"), (self.sample_button, "New sample  G")):
            pygame.draw.rect(self.screen, TEXT_LABEL, button, 1, border_radius=6)
            text = self.font_text.render(label, True, TEXT_PRIMARY)
            self.screen.blit(text, text.get_rect(center=button.center))
        controls = (f"W/↑ forward    S/↓ no-op    A/← D/→ turn    SPACE pause    . step    N/P history    "
                    "Drag bars edit z    Left-click map place    Right-click map face    ESC quit")
        self.screen.blit(self.font_text.render(controls, True, TEXT_LABEL), (x, y + 26))

    # ── Recording ────────────────────────────────────
    def start_clip(self):
        """Close the open clip and start a seed- and temperature-labelled MP4 segment."""
        self.close_clip()
        path = Path(self.cfg.video_dir) / f"seed{self.seeds[self.index]}_T{self.temperature:.2f}_clip{self.clips:03d}.mp4"
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.fps, self.screen.get_size())
        assert self.writer.isOpened(), f"OpenCV cannot write {path}"  # OpenCV reports encoder failures through this flag
        self.clips += 1
        print(f"[INFO] Recording {path}")

    def close_clip(self):
        """Release the active recording segment."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def record(self):
        """Append the window to the clip; a complete clip advances to the next seed."""
        width, height = self.screen.get_size()
        frame = np.frombuffer(pygame.image.tobytes(self.screen, "RGB"), np.uint8).reshape(height, width, 3)
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if self.steps >= self.cfg.video_steps:
            self.switch(1)

    def run(self):
        """Navigate with held keys, respecting pause and single-step controls, until the window closes."""
        while self.window.running:
            self.handle_events()
            if not self.window.running:
                break
            forward, turn = keyboard_axes()
            advance = not self.paused or self.advance_once
            self.advance_once = False
            if advance:
                self.step(self.config.manual_action(forward, turn))
            self.draw()
            self.window.present()
            if self.writer is not None and advance:
                self.record()
            self.clock.tick(self.cfg.fps)
        self.close_clip()
        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main(cfg: PriorSceneViewerConfig):
    """Open the prior scene explorer for one transition checkpoint."""
    if not PriorSceneViewer.TEMPERATURES[0] <= cfg.temperature <= PriorSceneViewer.TEMPERATURES[-1] or cfg.temperature_step <= 0 or cfg.refinements < 0:
        raise ValueError("temperature must lie in [0.25, 1], temperature_step must be positive, and refinements nonnegative")
    torch.backends.cudnn.allow_tf32 = False  # limit accumulated rollout error
    PriorSceneViewer(cfg).run()


if __name__ == "__main__":
    main(pyrallis.parse(config_class=PriorSceneViewerConfig))
