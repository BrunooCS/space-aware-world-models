"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared palette and primitives of the animated SVG assets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All motion is CSS @keyframes on one clock per file; raster content (maps, ego views)
is a vertical JPEG sprite sheet shifted by step-end keyframes. Files have no scripts or
external resources, so they animate inside <img> on GitHub and personal sites.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import base64
import io

import numpy as np
from PIL import Image

INK, LABEL, HAIRLINE, PAPER, BLUE = "#191b21", "#626b7a", "#dfe3ea", "#fbfbfb", "#164be6"
DENSITY_STOPS = np.array([(255, 255, 255), (219, 227, 251), (143, 168, 242), (22, 75, 230), (59, 47, 168), (148, 67, 208)])
SANS, MONO = "Arial,Helvetica,sans-serif", "Menlo,Consolas,'DejaVu Sans Mono',monospace"


# ── Colors ───────────────────────────────────────────
def rgb(color):
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in color)


def ramp(t, stops=DENSITY_STOPS):
    """Colors [..., 3] float of the piecewise-linear ramp `stops` at t [...] in [0, 1]."""
    grid = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(t, grid, stops[:, ch]) for ch in range(3)], axis=-1)


# ── Animation ────────────────────────────────────────
def keyframes(name, stops, prop):
    """CSS @keyframes `name` from (percent, value) stops of one property."""
    return f"@keyframes {name}{{" + "".join(f"{p:.3f}%{{{prop}:{v}}}" for p, v in stops) + "}"


def sprite_frames(frames):
    """Deduplicate consecutive identical frames [F, H, W, 3]: stored frames [U, H, W, 3] and the sprite row of every frame [F]."""
    keep = np.r_[True, (frames[1:] != frames[:-1]).reshape(len(frames) - 1, -1).any(axis=1)]
    return frames[keep], np.cumsum(keep) - 1


def sprite_css(name, rows, size, duration):
    """Step-end keyframes `name` that show sprite row rows[k] [F] of height `size` during frame k."""
    changes = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    stops = [(100 * k / len(rows), f"translateY({-size * rows[k]}px)") for k in changes] + [(100, f"translateY({-size * rows[-1]}px)")]
    return f".{name}{{animation:{name} {duration:.3f}s step-end infinite}}" + keyframes(name, stops, "transform")


def move_css(name, points, duration, angles=None):
    """Linear keyframes `name` placing an element at points [F + 1, 2] (last = first for a loop), optionally rotated by angles [F + 1] deg."""
    values = [f"translate({x:.1f}px,{y:.1f}px)" + ("" if angles is None else f" rotate({a:.1f}deg)") for (x, y), a in
              zip(points, angles if angles is not None else points)]
    stops = [(100 * k / (len(points) - 1), v) for k, v in enumerate(values)]
    return f".{name}{{animation:{name} {duration:.3f}s linear infinite}}" + keyframes(name, stops, "transform")


# ── Output / IO ──────────────────────────────────────
def sprite_uri(frames, quality):
    """Frames [U, H, W, 3] uint8 -> one vertical JPEG sprite sheet as a data URI."""
    buffer = io.BytesIO()
    Image.fromarray(np.concatenate(frames)).save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def sprite_panel(x, y, size, name, frames, quality, smooth=True):
    """Panel of side `size` at (x, y) that shows the sprite sheet of frames [U, h, w, 3] through class `name`."""
    h, w = frames.shape[1:3]
    rendering = "auto" if smooth else "pixelated"
    return (f'<g transform="translate({x},{y})"><svg width="{size}" height="{size}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<image class="{name}" href="{sprite_uri(frames, quality)}" width="{w}" height="{h * len(frames)}" style="image-rendering:{rendering}"/></svg>'
            f'<rect width="{size}" height="{size}" rx="4" fill="none" stroke="{HAIRLINE}"/></g>')


def image_uri(image, quality):
    """One RGB image [H, W, 3] uint8 as a JPEG data URI."""
    return sprite_uri(image[None], quality)


def text(x, y, string, size=11, fill=LABEL, anchor="start", family=SANS, extra=""):
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{string}</text>'


def title(x, y, string):
    return text(x, y, string, extra=' letter-spacing="1.3" font-weight="bold"')


def agent(cls, color, radius=4.5, cone=18, fov=90):
    """Agent marker of class `cls`: a dot with a translucent view cone of `fov` degrees along +x, posed by move_css."""
    x, y = cone * np.cos(np.radians(fov / 2)), cone * np.sin(np.radians(fov / 2))
    return (f'<g class="{cls}"><path d="M0 0L{x:.1f} {-y:.1f}A{cone} {cone} 0 0 1 {x:.1f} {y:.1f}Z" fill="{color}" fill-opacity=".2" '
            f'stroke="{color}" stroke-opacity=".55" stroke-width=".8"/><circle r="{radius}" fill="{color}" stroke="#fff" stroke-width="1.5"/></g>')


def polyline(points, stroke, width=1.2, extra=""):
    return (f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{extra}/>')


def document(width, height, css, body):
    """Full SVG text: paper card of `width` x `height`, the CSS rules, and the body elements."""
    css = ["text{font-kerning:normal}", *css, "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"]
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
                      f"<style>{''.join(css)}</style>", f'<rect width="{width}" height="{height}" rx="14" fill="{PAPER}" stroke="{HAIRLINE}"/>',
                      *body, "</svg>"])
