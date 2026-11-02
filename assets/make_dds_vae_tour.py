#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DDS-VAE latent tour (animated SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Render a looping SVG of the DDS-VAE generating and editing maps: a scene code moves
through the principal plane of the code space, its coordinates light up as tiles,
and the decoder renders the map for every frame.

1) Encode a map sample, fit the principal axes, and draw the code density as vector contours.
2) Build a closed tour: dataset map -> prior samples -> dataset map -> prior sample -> start.
3) Decode one map per frame into a sprite sheet and write all motion as CSS keyframes.

The tour interpolates z along a straight direction with the norm interpolated too (no shrinking
midpoints), smoothstep timing, and a pause on every waypoint.
Frames, dot and tiles share one clock; the decoded maps are the only raster content.
The SVG uses system font stacks, has no scripts or external files, and renders inside <img>.

Example:
    python assets/make_dds_vae_tour.py --checkpoint checkpoints/dds_vae_dmlab/version_0/stage2/last.ckpt --seed 0 --output assets/dds_vae_latent_tour.svg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contourpy
import cv2
import numpy as np
import pyrallis
import torch
from torchvision.io import ImageReadMode, read_image

from assets.svg_utils import HAIRLINE, INK, LABEL, MONO, PAPER, keyframes, ramp, rgb, sprite_uri, text
from src.models.dds_vae import DDSVAE, normalize_maps
from src.tools.render_dds_vae_latent import load_sample

TILE_ZERO, TILE_NEGATIVE, TILE_POSITIVE = (236, 239, 244), (155, 166, 183), (148, 67, 208)  # slate below zero, violet above


# ── Config ───────────────────────────────────────────
@dataclass
class TourConfig:
    """DDS-VAE checkpoint, tour timing, and output file."""

    checkpoint: str               # Stage 2 checkpoint of train_dds_vae
    map_dir: str = "data/dmlab_topdown_maps"
    output: str = "assets/dds_vae_latent_tour.svg"
    num_maps: int = 2000          # random maps encoded to draw the code density
    seed: int = 0                 # seeds the map sample drawn for the density
    map_files: list[str] = field(default_factory=lambda: ["171878987.png", "2035133242.png"])  # dataset waypoints, picked among 2000 maps by reconstruction error and total variation
    prior_seeds: list[int] = field(default_factory=lambda: [291, 207, 1323])  # prior waypoints z ~ N(0, I), picked among 2000 seeds by total variation and re-encoding error
    leg_seconds: float = 3.0      # duration of one waypoint-to-waypoint leg, including its pause
    pause: float = 0.35           # fraction of each leg spent paused on its target
    fps: int = 12                 # map frames per second; paused frames are stored once
    jpeg_quality: int = 80        # sprite sheet quality, the main contributor to file size
    num_tiles: int = 64           # code coordinates shown, ordered by posterior-mean spread
    code_range: float = 2.0       # |z_i| at which tile colors saturate
    tile_updates: int = 5         # tile color keyframes per second (CSS interpolates in between)
    plane_size: int = 340
    device: str = "cuda"


# ── Tour ─────────────────────────────────────────────
def tour_codes(waypoints, frames_per_leg, pause):
    """Closed tour through waypoints [L + 1, d_z] (last = first): codes [L * frames_per_leg, d_z]."""
    codes = []
    for start, target in zip(waypoints[:-1], waypoints[1:]):
        s = np.arange(frames_per_leg) / frames_per_leg  # the first frame of a leg sits on its start, so the loop wraps onto itself
        u = np.minimum(s / (1 - pause), 1.0)
        u = (u * u * (3 - 2 * u))[:, None]
        z = start + (target - start) * u
        codes.append(z * ((1 - u) * np.linalg.norm(start) + u * np.linalg.norm(target)) / np.linalg.norm(z, axis=1, keepdims=True))  # norm follows the endpoints
    return np.concatenate(codes)


# ── Plane: density and coordinates ───────────────────
def density_paths(pixels, size):
    """Stacked filled contour paths of a binned Gaussian KDE of plane pixels [N, 2]: (path d, color) per level, low to high."""
    hist = np.zeros((size, size), np.float32)
    np.add.at(hist, (pixels[:, 1].astype(int).clip(0, size - 1), pixels[:, 0].astype(int).clip(0, size - 1)), 1)
    sigma_x, sigma_y = 0.18 * pixels.std(axis=0, ddof=1)  # PCA diagonalizes the covariance; bandwidth 0.18 sample std per axis
    density = cv2.GaussianBlur(hist, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    density /= density.max()

    levels = np.linspace(0.03, 0.95, 14)
    generator = contourpy.contour_generator(z=density, fill_type=contourpy.FillType.ChunkCombinedCode)
    paths = []
    for i, level in enumerate(levels):
        (points,), (codes,) = generator.filled(level, np.inf)
        d = "".join("Z" if c == 79 else f"{'M' if c == 1 else 'L'}{x:.1f} {y:.1f}" for (x, y), c in zip(points, codes))
        paths.append((d, rgb(ramp(i / (len(levels) - 1)))))
    return paths


# ── Main ─────────────────────────────────────────────
def main(cfg: TourConfig):
    """Encode a map sample, decode the closed latent tour, and write the animated SVG."""
    # 1) Encode maps and fit the principal axes of the code space
    model = DDSVAE.from_checkpoint(cfg.checkpoint, cfg.device)
    maps = load_sample(cfg.map_dir, cfg.num_maps, cfg.seed)
    with torch.inference_mode():
        mu = torch.cat([model.encode(normalize_maps(batch.to(cfg.device)))[0].cpu() for batch in maps.split(250)]).numpy()  # [N, d_z]
    mean = mu.mean(axis=0)
    _, singular, components = np.linalg.svd(mu - mean, full_matrices=False)
    basis = components[:2]  # [2, d_z]
    share = 100 * singular[:2] ** 2 / (singular ** 2).sum()  # variance of the posterior means explained by PC1, PC2

    # 2) Closed tour: dataset map A -> prior, prior -> dataset map B -> prior -> A
    with torch.inference_mode():
        data = [model.encode(normalize_maps(read_image(f"{cfg.map_dir}/{name}", mode=ImageReadMode.RGB)[None].to(cfg.device)))[0][0].cpu().numpy() for name in cfg.map_files]
    prior = [torch.randn(mu.shape[1], generator=torch.Generator().manual_seed(seed)).numpy() for seed in cfg.prior_seeds]
    waypoints = np.stack([data[0], prior[0], prior[1], data[1], prior[2], data[0]])
    frames_per_leg = round(cfg.leg_seconds * cfg.fps)
    codes = tour_codes(waypoints, frames_per_leg, cfg.pause)  # [F, d_z]
    num_frames, duration = len(codes), len(codes) / cfg.fps

    keep = np.r_[True, (np.diff(codes, axis=0) != 0).any(axis=1)]  # paused frames repeat their code exactly
    sprite_index = np.cumsum(keep) - 1  # [F] sprite row shown by every frame
    with torch.inference_mode():
        decoded = torch.cat([model.decode(z.to(cfg.device)).cpu() for z in torch.from_numpy(codes[keep]).float().split(60)])
    frames = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()  # [F, H, W, 3]
    map_size = frames.shape[1]

    # ── Layout ────────────────────────────────────────
    P, margin, gap, top = cfg.plane_size, 32, 64, 56
    columns = 8
    cell, cell_gap = 22, 3
    tile_span = columns * (cell + cell_gap) - cell_gap
    x_plane, x_tiles = margin, margin + P + gap
    x_map = x_tiles + tile_span + gap
    width, height = x_map + P + margin, top + P + 78

    # ── Plane: pixels, density, route ─────────────────
    plane_points = (mu - mean) @ basis.T  # [N, 2]
    route = (codes - mean) @ basis.T  # [F, 2]
    both = np.concatenate([plane_points, route])
    low, high = both.min(axis=0), both.max(axis=0)
    center, span = (low + high) / 2, (high - low).max() * 1.12

    def to_pixels(coords):
        return np.stack([(coords[..., 0] - center[0]) / span * P + P / 2, P / 2 - (coords[..., 1] - center[1]) / span * P], axis=-1)

    pixels, route_px = to_pixels(plane_points), to_pixels(np.concatenate([route, route[:1]]))  # closed route [F + 1, 2]
    origin = to_pixels(np.zeros(2))
    waypoint_px = to_pixels((waypoints - mean) @ basis.T)

    # ── Tiles: most variable coordinates ──────────────
    order = np.argsort(-mu.std(axis=0))[:cfg.num_tiles]
    stride = round(cfg.fps / cfg.tile_updates)
    stops = list(range(0, num_frames, stride)) + [num_frames]
    closed = np.concatenate([codes, codes[:1]])  # frame F is frame 0

    def tile_color(value):
        t = np.clip(value / cfg.code_range, -1, 1)
        target = TILE_NEGATIVE if t < 0 else TILE_POSITIVE
        return rgb(np.array(TILE_ZERO) + (np.array(target) - np.array(TILE_ZERO)) * abs(t))

    # ── CSS ───────────────────────────────────────────
    css = [f"text{{font-kerning:normal}}", f".sheet{{animation:sheet {duration:.3f}s step-end infinite}}",
           keyframes("sheet", [(100 * k / num_frames, f"translateY({-map_size * sprite_index[k]}px)") for k in np.flatnonzero(keep)] + [(100, f"translateY({-map_size * sprite_index[-1]}px)")], "transform"),
           f".dot,.trail{{animation:move {duration:.3f}s linear infinite}}",
           keyframes("move", [(100 * k / num_frames, f"translate({x:.1f}px,{y:.1f}px)") for k, (x, y) in enumerate(route_px)], "transform")]
    for i, coordinate in enumerate(order):
        css.append(f".c{i}{{animation:c{i} {duration:.3f}s linear infinite}}")
        css.append(keyframes(f"c{i}", [(100 * k / num_frames, tile_color(closed[k, coordinate])) for k in stops], "fill"))
    css.append("@media (prefers-reduced-motion:reduce){*{animation:none!important}}")

    # ── SVG ───────────────────────────────────────────
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f"<style>{''.join(css)}</style>", f'<rect width="{width}" height="{height}" rx="14" fill="{PAPER}" stroke="{HAIRLINE}"/>']
    for x, title in ((x_plane, "LATENT SPACE · PCA"), (x_tiles, "SCENE CODE"), (x_map, "DECODED MAP")):
        svg.append(text(x, 36, title, extra=' letter-spacing="1.3" font-weight="bold"'))

    # Plane
    svg.append(f'<g transform="translate({x_plane},{top})"><clipPath id="plane"><rect width="{P}" height="{P}" rx="4"/></clipPath>'
               f'<rect width="{P}" height="{P}" rx="4" fill="#fff"/><g clip-path="url(#plane)">')
    for d, color in density_paths(pixels, P):
        svg.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd" stroke="{INK}" stroke-opacity=".10" stroke-width=".5"/>')
    svg.append(f'<path d="M{origin[0]-5:.1f} {origin[1]:.1f}h10M{origin[0]:.1f} {origin[1]-5:.1f}v10" stroke="{LABEL}" stroke-width="1"/>')
    svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in route_px)}" fill="none" stroke="{INK}" stroke-opacity=".28" stroke-width="1" stroke-dasharray="2 4"/>')
    for x, y in waypoint_px[:-1]:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#fff" stroke="{INK}" stroke-opacity=".6" stroke-width="1"/>')
    for k in range(7, 0, -1):  # trail: delayed copies of the dot, fading out
        svg.append(f'<circle class="trail" r="{4.2 - 0.4 * k:.1f}" fill="{INK}" fill-opacity="{0.45 * (1 - k / 8):.2f}" '
                   f'style="animation-delay:-{duration - 0.06 * k:.3f}s;transform:translate({route_px[0][0]:.1f}px,{route_px[0][1]:.1f}px)"/>')
    svg.append(f'<circle class="dot" r="5.5" fill="{INK}" stroke="#fff" stroke-width="2" style="transform:translate({route_px[0][0]:.1f}px,{route_px[0][1]:.1f}px)"/>')
    svg.append("</g>")
    svg.append(f'<rect width="{P}" height="{P}" rx="4" fill="none" stroke="{HAIRLINE}"/>')
    svg.append(text(P - 8, P - 10, f"PC1 · {share[0]:.1f}% var", anchor="end", family=MONO, size=10))
    svg.append(text(10, 18, f"PC2 · {share[1]:.1f}% var", family=MONO, size=10))
    density_key = "".join(f'<rect x="{P - 100 + 10 * i}" y="{P + 14}" width="10" height="8" fill="{rgb(ramp(i / 9))}"/>' for i in range(10))
    svg.append(f'<circle cx="6" cy="{P + 18}" r="3.2" fill="#fff" stroke="{INK}" stroke-opacity=".6"/>' + text(14, P + 21.5, "waypoint", size=10, family=MONO)
               + f'<circle cx="78" cy="{P + 18}" r="4" fill="{INK}"/>' + text(87, P + 21.5, "current z", size=10, family=MONO) + density_key
               + text(P - 100, P + 36, "low", size=10, family=MONO) + text(P - 50, P + 36, "density", size=10, family=MONO, anchor="middle")
               + text(P, P + 36, "high", size=10, family=MONO, anchor="end") + text(0, P + 36, f"N = {len(mu)} posterior means", size=10, family=MONO))
    svg.append("</g>")

    # Tiles
    svg.append(f'<g transform="translate({x_tiles},{top + (P - tile_span - 58) / 2})">')
    for i in range(cfg.num_tiles):
        r, c = divmod(i, columns)
        svg.append(f'<rect class="c{i}" x="{c * (cell + cell_gap)}" y="{r * (cell + cell_gap)}" width="{cell}" height="{cell}" rx="3" fill="{tile_color(closed[0, order[i]])}"/>')
    bar_y = tile_span + 22
    svg.append(text(0, -12, f"{cfg.num_tiles} of {mu.shape[1]} dims, sorted by std(μ)", size=10, family=MONO))
    svg.append("".join(f'<rect x="{i * tile_span / 40:.1f}" y="{bar_y}" width="{tile_span / 40 + .5:.1f}" height="8" fill="{tile_color((i / 39 * 2 - 1) * cfg.code_range)}"/>' for i in range(40)))
    for frac, label in ((0, f"−{cfg.code_range:g}"), (.5, "0"), (1, f"+{cfg.code_range:g}")):
        svg.append(text(frac * tile_span, bar_y + 21, label, size=10, family=MONO, anchor=("start", "middle", "end")[int(frac * 2)]))
    svg.append(text(tile_span / 2, bar_y + 36, "z_i in prior std", size=10, family=MONO, anchor="middle"))
    svg.append("</g>")

    # Decoded map
    svg.append(f'<g transform="translate({x_map},{top})"><svg width="{P}" height="{P}" viewBox="0 0 {map_size} {map_size}">'
               f'<image class="sheet" href="{sprite_uri(frames, cfg.jpeg_quality)}" width="{map_size}" height="{map_size * len(frames)}" style="image-rendering:auto"/></svg>'
               f'<rect width="{P}" height="{P}" rx="4" fill="none" stroke="{HAIRLINE}"/></g>')
    svg.append(text(x_map + P / 2, top + P + 24, "Prior sample", size=13, fill=INK, anchor="middle"))
    svg.append(text(x_map + P / 2, top + P + 44, f"decoder output, {map_size} × {map_size} RGB", size=10, family=MONO, anchor="middle"))

    # Flow arrows
    for x in (x_plane + P + 14, x_tiles + tile_span + 14):
        svg.append(f'<path d="M{x} {top + P / 2}h{gap - 28}m-5 -4l5 4l-5 4" fill="none" stroke="{LABEL}" stroke-opacity=".6" stroke-width="1.2"/>')
    svg.append("</svg>")

    # ── Write ─────────────────────────────────────────
    output = Path(cfg.output)
    output.write_text("\n".join(svg))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB, {num_frames} frames ({len(frames)} stored), {duration:.1f} s loop")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=TourConfig))
