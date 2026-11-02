#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DDS-VAE latent-space explorer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inspect decoded maps while editing scene codes z in R^{d_z}.

1) Encode dataset maps to posterior means and fit their principal axes.
2) Display the reference map, decoded map, latent density, PCA sliders, and code tiles.
3) Decode edits or interpolate between maps during the latent tour.

Plane drags change the first two principal coordinates while preserving the anchor's
remaining components. Sliders use sample standard deviations; tiles edit raw coordinates
and are ordered by posterior-mean variance. Active dimensions satisfy Var[mu] > 0.01.

Example:
    python -m src.tools.render_dds_vae_latent --checkpoint checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt

Controls: drag plane, sliders, or tiles edit z, right-click snap to the nearest map, N/P next or previous map,
S sample the prior N(0, I), SPACE latent tour, R reset z to the anchor, ESC quit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pygame
import pyrallis
import torch
from torchvision.io import ImageReadMode, read_image

from src.models.dds_vae import DDSVAE, normalize_maps
from src.tools.render_utils import (ACCENT_BLUE, BACKGROUND, TEXT_LABEL, TEXT_PRIMARY, TEXT_SECONDARY, TRAIL_SHADOW, ViewerWindow,
                                    numpy_to_surface, to_image)

FRAME = (44, 48, 57)  # panel border
TILE_GROUND, TILE_NEGATIVE, TILE_POSITIVE = (26, 29, 36), TEXT_LABEL, (176, 124, 236)  # code tiles: slate below zero, violet above
PLANE_GROUND, PLANE_INK, PLANE_LABEL = (251, 251, 251), (25, 27, 33), (98, 107, 122)
DENSITY_LEVELS = np.linspace(0.02, 1.0, 18)  # densities below 2% of the peak use the background
_DENSITY_STOPS = np.array([PLANE_GROUND, (219, 227, 251), (143, 168, 242), (22, 75, 230), (59, 47, 168), (148, 67, 208)])
_DENSITY_MIDPOINTS = (np.arange(len(DENSITY_LEVELS) - 1) + 0.5) / (len(DENSITY_LEVELS) - 1)
DENSITY_COLORS = np.stack([np.interp(_DENSITY_MIDPOINTS, np.linspace(0, 1, len(_DENSITY_STOPS)), channel)
                          for channel in _DENSITY_STOPS.T], axis=-1).astype(np.uint8)  # [17, 3] colors at the band midpoints

TILE_COLUMNS = 32
PX_PER_SIGMA = 40     # vertical drag distance that changes a tile by one prior standard deviation
TRAIL_SPACING = 7     # pixels between the dots of the trail
MAX_DIMMING = 0.9    # maximum reference-map overlay opacity
ROW = 28              # slider row height


# ── Config ───────────────────────────────────────────
@dataclass
class LatentViewerConfig:
    """DDS-VAE checkpoint, map sample, and window settings."""

    checkpoint: str              # Stage 2 checkpoint of train_dds_vae
    map_dir: str = "data/dmlab_topdown_maps"
    num_maps: int = 2000         # random maps encoded to draw the code space
    plane_dots: int = 0          # optional maps drawn as dots over the density
    num_axes: int = 10           # principal axes with a slider
    axis_range: float = 3.0      # half-width of a slider, in standard deviations of its axis
    code_range: float = 2.0      # |z_i| at which tile colors saturate, in standard deviations of the prior
    seed: int = 0
    device: str = "cuda"
    tour_seconds: float = 2.0    # duration of one leg of the latent tour
    panel_size: int = 400
    panel_margin: int = 30
    fps: int = 30


# ── Inputs / Data ────────────────────────────────────
def load_sample(map_dir, num_maps, seed):
    """Random sample of `num_maps` PNG maps of `map_dir` as one uint8 tensor [N, 3, H, W]."""
    paths = sorted(Path(map_dir).glob("*.png"))
    picks = torch.randperm(len(paths), generator=torch.Generator().manual_seed(seed))[:num_maps]
    with ThreadPoolExecutor() as pool:  # PNG decoding releases the GIL
        return torch.stack(list(pool.map(lambda i: read_image(str(paths[i]), mode=ImageReadMode.RGB), picks.tolist())))


# ── Transform / Core Logic ───────────────────────────
def density_image(points, size, supersample=2):
    """Render a binned Gaussian KDE of PCA pixels [N, 2] as RGB [size, size, 3]."""
    # ── Estimate density ──────────────────────────────
    big = size * supersample  # antialias contour strokes
    hist = np.zeros((big, big), np.float32)
    np.add.at(hist, ((points[:, 1] * supersample).astype(int).clip(0, big - 1), (points[:, 0] * supersample).astype(int).clip(0, big - 1)), 1)

    # PCA diagonalizes covariance; bandwidth is 0.12 times each axis's sample standard deviation.
    sigma_x, sigma_y = 0.12 * supersample * points.std(axis=0, ddof=1)
    density = cv2.GaussianBlur(hist, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    density /= density.max()  # [S, S] in 0..1

    # ── Draw density bands and contours ────────────────
    bands = np.searchsorted(DENSITY_LEVELS, density, side="right") - 1
    image = DENSITY_COLORS[bands.clip(0, len(DENSITY_COLORS) - 1)].astype(np.float32)  # [S, S, 3]
    image[density < DENSITY_LEVELS[0]] = PLANE_GROUND

    for level in DENSITY_LEVELS[::2]:
        contours, _ = cv2.findContours((density >= level).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        line = np.zeros((big, big), np.uint8)
        cv2.polylines(line, contours, True, 255, 1, cv2.LINE_AA)
        drawn = line > 0
        weight = (line[drawn] / 255 * 0.35)[:, None]
        image[drawn] = image[drawn] * (1 - weight) + np.array(PLANE_INK) * weight
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)


def trail_dots(points, spacing):
    """Polyline [T, 2] -> dots [M, 2] every `spacing` pixels of arc length, newest first."""
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    marks = np.arange(arc[-1], -1, -spacing)
    return np.stack([np.interp(marks, arc, points[:, axis]) for axis in (0, 1)], axis=1)


def tile_colors(values, saturation):
    """Code values [...] -> RGB [..., 3]: slate for negative and violet for positive, through the dark tile ground at zero."""
    t = np.clip(values / saturation, -1, 1)[..., None]
    return (TILE_GROUND + (np.where(t < 0, TILE_NEGATIVE, TILE_POSITIVE) - TILE_GROUND) * np.abs(t)).astype(np.uint8)


def lerp(a, b, t):
    """Color between `a` (t = 0) and `b` (t = 1)."""
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


# ── Viewer ───────────────────────────────────────────
class LatentViewer:
    """DDS-VAE decoder driven by a scene code edited in the principal plane, along principal axes, or per coordinate."""

    def __init__(self, cfg):
        """Encode the map sample, fit the principal axes, and open the window on the first map."""
        self.cfg, self.device = cfg, cfg.device
        self.scene_encoder = DDSVAE.from_checkpoint(cfg.checkpoint, cfg.device)
        self.maps = load_sample(cfg.map_dir, cfg.num_maps, cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        # ── Code space: posterior means and their principal axes ──
        with torch.inference_mode():
            mu = [self.scene_encoder.encode(normalize_maps(batch.to(self.device)))[0].cpu() for batch in self.maps.split(250)]
        self.mu = torch.cat(mu).numpy()  # [N, d_z]
        self.mean = self.mu.mean(axis=0)
        _, singular, components = np.linalg.svd(self.mu - self.mean, full_matrices=False)
        k = cfg.num_axes
        self.components, self.sigma = components[:k], singular[:k] / np.sqrt(len(self.mu) - 1)  # [K, d_z], [K] sample std per axis
        self.share = singular[:k] ** 2 / (singular ** 2).sum()  # variance share per axis
        self.basis = components[:2]  # [2, d_z]
        coords = (self.mu - self.mean) @ self.basis.T  # [N, 2]
        low, high = coords.min(axis=0), coords.max(axis=0)
        self.center, self.span = (low + high) / 2, (high - low).max() * 1.16  # full sample with 8% padding on either side
        squared = (self.mu ** 2).sum(axis=1)
        distances = np.sqrt(np.maximum(squared[:, None] + squared[None] - 2 * self.mu @ self.mu.T, 0))  # [N, N]
        np.fill_diagonal(distances, np.inf)
        self.neighbor_distance = np.median(distances.min(axis=1)) / 2  # distance at which reference-map dimming saturates

        # Sort by spread so low-variance coordinates appear last.
        spread = self.mu.std(axis=0)
        self.order = np.argsort(-spread)  # tile rank -> coordinate
        self.rank = np.argsort(self.order)  # coordinate -> tile rank
        self.spread, self.active = spread, int((self.mu.var(axis=0) > 0.01).sum())

        # ── Window and geometry ───────────────────────
        self.window = ViewerWindow("DDS-VAE latent space", cfg, footer_height=240)
        self.screen, self.clock = self.window.screen, self.window.clock
        self.font_title, self.font_text = self.window.font_title, self.window.font_text
        self.panels = self.window.panels
        size, margin = cfg.panel_size, cfg.panel_margin
        top = self.panels[0].bottom + 50
        columns = (self.panels[0].x, self.panels[1].x)  # slider columns sit under the first two panels
        rows = -(-k // 2)
        self.sliders = [pygame.Rect(columns[i // rows], top + (i % rows) * ROW, size, ROW) for i in range(k)]
        self.tracks = [pygame.Rect(r.x + 50, r.y + 6, size - 50 - 58, 16) for r in self.sliders]
        cell = size // TILE_COLUMNS
        self.cell = cell
        self.tiles = pygame.Rect(self.panels[2].x, top, TILE_COLUMNS * cell, -(-len(self.mean) // TILE_COLUMNS) * cell)
        self.top = top

        # ── Static layers: the code-space chart ────────
        self.points = self.to_pixels(coords)  # [N, 2] dots on the plane panel
        self.plane = numpy_to_surface(density_image(self.points, size), (size, size))
        for x, y in self.points[:cfg.plane_dots]:
            pygame.draw.circle(self.plane, PLANE_LABEL, (round(x), round(y)), 1)
        origin = [round(v) for v in self.to_pixels(np.zeros(2))]  # sample mean
        pygame.draw.line(self.plane, PLANE_LABEL, (origin[0] - 4, origin[1]), (origin[0] + 4, origin[1]))
        pygame.draw.line(self.plane, PLANE_LABEL, (origin[0], origin[1] - 4), (origin[0], origin[1] + 4))
        axis_x = self.font_text.render("PC1", True, PLANE_LABEL)
        self.plane.blit(axis_x, (size - axis_x.get_width() - 8, size - 22))
        self.plane.blit(pygame.transform.rotate(self.font_text.render("PC2", True, PLANE_LABEL), 90), (6, 8))
        ramp = DENSITY_COLORS[np.linspace(0, len(DENSITY_COLORS), 130, endpoint=False).astype(int)][None]  # [1, 130, 3] banded density key
        self.plane.blit(self.font_text.render("Relative density", True, PLANE_LABEL), (10, size - 49))
        self.plane.blit(numpy_to_surface(ramp, (130, 6)), (10, size - 28))
        for value in (0.25, 0.5, 0.75, 1.0):
            label = self.font_text.render(f"{value:g}", True, PLANE_LABEL)
            x = 10 + round(130 * (value - DENSITY_LEVELS[0]) / (DENSITY_LEVELS[-1] - DENSITY_LEVELS[0]))
            self.plane.blit(label, (x - label.get_width() // 2, size - 21))
        self.veil = pygame.Surface((size, size))  # reference-map dimming overlay
        self.veil.fill(BACKGROUND)

        # Repaint dynamic panels and controls; static layers refresh on window exposure.
        self.window.dirty = [*(p.inflate(2, 2) for p in self.panels), *(pygame.Rect(p.x, p.y - 26, p.width, 24) for p in self.panels),
                             pygame.Rect(columns[0], top, 2 * size + margin, rows * ROW), pygame.Rect(self.tiles.x, top, size, self.tiles.height + 60)]
        self.changed, self.hover = True, None
        self.drag, self.tour, self.trail = None, None, deque(maxlen=256)
        self.select(0)

    # ── Geometry ──────────────────────────────────────
    def to_pixels(self, coords):
        """Principal coordinates [..., 2] -> pixels [..., 2] inside the plane panel (y up)."""
        return ((coords - self.center) / self.span * (1, -1) + 0.5) * self.cfg.panel_size

    def to_coords(self, pixels):
        """Pixels inside the plane panel -> principal coordinates."""
        return (np.asarray(pixels) / self.cfg.panel_size - 0.5) * (1, -1) * self.span + self.center

    def project(self, z):
        """Code z [d_z] -> principal coordinates [2]."""
        return self.basis @ (z - self.mean)

    def axes(self, z):
        """Code z [d_z] -> its coordinates on the slider axes [K], in standard deviations of the sample."""
        return self.components @ (z - self.mean) / self.sigma

    def slider_x(self, axis, u):
        """Axis coordinate `u` (standard deviations) -> window x on the track of slider `axis`, clipped to its ends."""
        track = self.tracks[axis]
        return track.x + round((np.clip(u / self.cfg.axis_range, -1, 1) / 2 + 0.5) * track.width)

    def tile_rect(self, rank):
        """Window rectangle of the tile with the given rank."""
        return pygame.Rect(self.tiles.x + rank % TILE_COLUMNS * self.cell, self.tiles.y + rank // TILE_COLUMNS * self.cell, self.cell - 1, self.cell - 1)

    def tile_at(self, position):
        """Rank of the tile under a window pixel, or None outside the grid."""
        col, row = (position[0] - self.tiles.x) // self.cell, (position[1] - self.tiles.y) // self.cell
        rank = row * TILE_COLUMNS + col
        return rank if self.tiles.collidepoint(position) and 0 <= col < TILE_COLUMNS and rank < len(self.mean) else None

    # ── State ─────────────────────────────────────────
    def set_anchor(self, index):
        """Take map `index` as the reference: z is edited relative to its posterior mean."""
        self.index, self.anchor = index, self.mu[index]
        self.anchor_coords, self.anchor_axes = self.project(self.anchor), self.axes(self.anchor)
        self.reference = numpy_to_surface(self.maps[index].permute(1, 2, 0).numpy(), self.panels[0].size)

    @torch.inference_mode()
    def set_code(self, z):
        """Decode z [d_z] and extend its trail on the plane."""
        self.z, self.u = z, self.axes(z)
        image = to_image(self.scene_encoder.decode(torch.from_numpy(z).to(self.device)[None]))[0]  # [128, 128, 3]
        self.reconstruction = numpy_to_surface(image, self.panels[1].size)
        self.veil.set_alpha(round(255 * MAX_DIMMING * min(np.linalg.norm(z - self.anchor) / self.neighbor_distance, 1.0)))
        self.trail.append(self.to_pixels(self.project(z)))
        self.changed = True

    def select(self, index):
        """Snap z to the posterior mean of map `index` with a fresh trail."""
        self.set_anchor(index % len(self.mu))
        self.trail.clear()
        self.set_code(self.anchor.copy())

    # ── Interaction ───────────────────────────────────
    def hit(self, position):
        """Drag descriptor of the control under a window pixel, or None."""
        if self.panels[2].collidepoint(position):
            return ("plane",)
        for axis, rect in enumerate(self.sliders):
            if rect.collidepoint(position):
                return ("axis", axis)
        rank = self.tile_at(position)
        if rank is not None:
            self.hover = rank
            return ("tile", int(self.order[rank]), position[1], float(self.z[self.order[rank]]))  # coordinate, press height, press value

    def drag_to(self, position):
        """Move z in the principal plane, along one principal axis, or set one coordinate from the vertical drag."""
        kind = self.drag[0]
        if kind == "plane":
            pixels = np.clip(np.subtract(position, self.panels[2].topleft), 0, self.cfg.panel_size)
            z = self.anchor + (self.to_coords(pixels) - self.anchor_coords) @ self.basis
        elif kind == "axis":
            axis, track = self.drag[1], self.tracks[self.drag[1]]
            u = np.clip((position[0] - track.x) / track.width * 2 - 1, -1, 1) * self.cfg.axis_range
            z = self.z + (u - self.u[axis]) * self.sigma[axis] * self.components[axis]
        else:
            _, dim, y0, value = self.drag
            z = self.z.copy()
            z[dim] = value + (y0 - position[1]) / PX_PER_SIGMA
        self.set_code(z.astype(np.float32))

    def handle_events(self):
        """Handle latent edits, sampling, tours, map selection, and window events."""
        for event in self.window.events():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.drag = self.hit(event.pos)
                if self.drag:
                    self.tour = None
                    if self.drag[0] == "plane":
                        self.trail.clear()
                    self.drag_to(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.panels[2].collidepoint(event.pos):
                pixels = np.subtract(event.pos, self.panels[2].topleft)
                self.tour = None
                self.select(int(np.linalg.norm(self.points - pixels, axis=1).argmin()))
            elif event.type == pygame.MOUSEMOTION:
                if self.drag:
                    self.drag_to(event.pos)
                elif (rank := self.tile_at(event.pos)) != self.hover:
                    self.hover, self.changed = rank, True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drag = None
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_n, pygame.K_p):
                    self.tour = None
                    self.select(self.index + (1 if event.key == pygame.K_n else -1))
                elif event.key == pygame.K_r:
                    self.tour = None
                    self.select(self.index)
                elif event.key == pygame.K_s:
                    self.tour = None
                    self.trail.clear()
                    self.set_code(self.rng.standard_normal(len(self.z)).astype(np.float32))  # z ~ N(0, I)
                elif event.key == pygame.K_SPACE:
                    self.tour = None if self.tour else (self.z, int(self.rng.integers(len(self.mu))), time.monotonic())

    def advance_tour(self):
        """Move z along the current leg with smoothstep timing; the reference follows the nearer endpoint."""
        start, target, started = self.tour
        t = min((time.monotonic() - started) / self.cfg.tour_seconds, 1.0)
        self.set_code(start + (self.mu[target] - start) * t * t * (3 - 2 * t))
        if t > 0.5 and self.index != target:
            self.set_anchor(target)
        if t == 1.0:
            self.tour = (self.z, int(self.rng.integers(len(self.mu))), time.monotonic())

    # ── Drawing ───────────────────────────────────────
    def put(self, string, position, color=TEXT_LABEL, right=False, font=None):
        """Text at `position` (its top-right corner when `right`), in the body font unless `font` is given."""
        image = (font or self.font_text).render(string, True, color)
        self.screen.blit(image, image.get_rect(topright=position) if right else position)

    def draw_plane(self, rect):
        """Draw latent density, the code trail, the anchor, and the current code."""
        self.screen.blit(self.plane, rect.topleft)
        self.screen.set_clip(rect)
        dots = trail_dots(np.array(self.trail), TRAIL_SPACING)[:40]
        for i, (x, y) in enumerate(dots):  # older dots are fainter
            pygame.draw.circle(self.screen, TRAIL_SHADOW, (rect.x + round(x), rect.y + round(y)), 3)
            pygame.draw.circle(self.screen, lerp(TEXT_PRIMARY, TEXT_LABEL, i / len(dots)), (rect.x + round(x), rect.y + round(y)), 2)
        anchor, live = [(rect.x + round(x), rect.y + round(y)) for x, y in (self.to_pixels(self.anchor_coords), self.trail[-1])]
        pygame.draw.line(self.screen, TRAIL_SHADOW, anchor, live, 3)
        pygame.draw.line(self.screen, TEXT_PRIMARY, anchor, live, 1)
        pygame.draw.circle(self.screen, TRAIL_SHADOW, anchor, 7, 4)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, anchor, 6, 2)
        pygame.draw.circle(self.screen, TRAIL_SHADOW, live, 10)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, live, 8)  # contrast ring on the density field
        pygame.draw.circle(self.screen, ACCENT_BLUE, live, 5)
        self.screen.set_clip(None)

    def draw_axes(self):
        """Draw principal-coordinate sliders with anchor ticks and current values."""
        for axis, (rect, track) in enumerate(zip(self.sliders, self.tracks)):
            y = track.centery
            self.put(f"PC{axis + 1}", (rect.x, y - 8), TEXT_PRIMARY)
            pygame.draw.line(self.screen, TEXT_LABEL, track.midleft, track.midright, 1)
            tick = self.slider_x(axis, self.anchor_axes[axis])
            pygame.draw.line(self.screen, TEXT_PRIMARY, (tick, y - 5), (tick, y + 5), 1)
            x = self.slider_x(axis, self.u[axis])
            pygame.draw.circle(self.screen, ACCENT_BLUE, (x, y), 5)
            pygame.draw.circle(self.screen, TEXT_PRIMARY, (x, y), 5, 1)
            self.put(f"{self.u[axis]:+.2f}σ", (rect.right, y - 8), TEXT_PRIMARY, right=True)

    def draw_tiles(self):
        """Draw latent-coordinate tiles, a color key, and the hovered or dragged value."""
        for rank, color in enumerate(tile_colors(self.z[self.order], self.cfg.code_range)):
            pygame.draw.rect(self.screen, color, self.tile_rect(rank), border_radius=2)
        if self.hover is not None:
            pygame.draw.rect(self.screen, TEXT_PRIMARY, self.tile_rect(self.hover).inflate(2, 2), 1, border_radius=2)

        # Color key: slate (-) through ground (0) to violet (+).
        y = self.tiles.bottom + 12
        key = numpy_to_surface(tile_colors(np.linspace(-1, 1, 100) * self.cfg.code_range, self.cfg.code_range)[None], (100, 8))
        self.put(f"-{self.cfg.code_range:g}", (self.tiles.x, y - 4))
        self.screen.blit(key, (self.tiles.x + 30, y))
        self.put(f"+{self.cfg.code_range:g}", (self.tiles.x + 138, y - 4))
        if self.hover is not None:
            dim = self.order[self.hover]
            self.put(f"dim {dim}   z {self.z[dim]:+.2f}   anchor {self.anchor[dim]:+.2f}   spread {self.spread[dim]:.2f}", (self.tiles.x, y + 22), TEXT_PRIMARY)

    def draw(self):
        """Draw the three panels with their readouts, the principal axes, the tiles, and the controls."""
        self.screen.fill(BACKGROUND)

        # ── Panels ────────────────────────────────────
        reference, decode, plane = self.panels
        distance = np.linalg.norm(self.z - self.anchor)
        for rect, title, readout in ((reference, "REFERENCE MAP", f"map {self.index} / {len(self.mu) - 1}"),
                                     (decode, "DDS-VAE DECODE", f"|z - mu| {distance:.2f}   |z| {np.linalg.norm(self.z):.2f}" + ("   touring" if self.tour else "")),
                                     (plane, "CODE SPACE", f"PC1 x PC2   {self.share[:2].sum():.0%} of variance")):
            self.put(title, (rect.x, rect.y - 22), TEXT_SECONDARY, font=self.font_title)
            self.put(readout, (rect.right, rect.y - 21), TEXT_PRIMARY, right=True)
        self.screen.blit(self.reference, reference.topleft)
        self.screen.blit(self.veil, reference.topleft)
        self.screen.blit(self.reconstruction, decode.topleft)
        self.draw_plane(plane)
        for rect in self.panels:
            pygame.draw.rect(self.screen, FRAME, rect.inflate(2, 2), 1)

        # ── Principal axes, tiles, and controls ───────
        self.put("PRINCIPAL AXES", (self.sliders[0].x, self.top - 26), TEXT_SECONDARY, font=self.font_title)
        self.put("in sample std, tick = anchor", (self.sliders[0].right, self.top - 25), TEXT_LABEL, right=True)
        self.put("SCENE CODE z", (self.tiles.x, self.top - 26), TEXT_SECONDARY, font=self.font_title)
        self.put(f"{len(self.z)} dims by spread, {self.active} active", (self.tiles.right, self.top - 25), TEXT_LABEL, right=True)
        self.draw_axes()
        self.draw_tiles()
        controls = "Drag the plane, a slider, or a tile    Right-click: nearest map    N/P map    S prior sample    SPACE tour    R reset    ESC quit"
        self.put(controls, (self.panels[0].x, self.sliders[0].y + 5 * ROW + 18))

    def run(self):
        """Handle input, advance the tour, and redraw whenever z changes, until ESC or window close."""
        while self.window.running:
            self.handle_events()
            if not self.window.running:
                break
            if self.tour:
                self.advance_tour()
            if self.changed or self.window.expose:
                self.draw()
                self.window.present()
                self.changed = False
            self.clock.tick(self.cfg.fps)
        pygame.quit()


# ── Main ─────────────────────────────────────────────
def main(cfg: LatentViewerConfig):
    """Open the latent-space explorer for one DDS-VAE checkpoint."""
    LatentViewer(cfg).run()


if __name__ == "__main__":
    main(pyrallis.parse(config_class=LatentViewerConfig))
