#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monochrome sketch of the SWM architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a grayscale SVG of scene memory, state dynamics, fusion, and observation rendering.

1) Draw modules and connectors as vectors, using system font stacks for labels.
2) Animate one state update and save the SVG with a reduced-motion still.

Example:
    python assets/render_architecture_minimal.py --duration 9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


# ── Config ───────────────────────────────────────────
@dataclass
class MinimalArchitectureConfig:
    """SVG destination and loop duration in seconds."""

    output: str = "assets/swm_architecture_minimal.svg"
    duration: float = 9.0


# ── SVG primitives ───────────────────────────────────
INK, EDGE, WIRE, LINE, PLATE = "#171717", "#404040", "#454545", "#8e8e8e", "#fdfdfd"
SANS = "Arial, Helvetica, sans-serif"
SERIF = "'Times New Roman', Times, serif"
SMOOTH = "cubic-bezier(0.45,0,0.2,1)"
W, H = 924, 566


RNG = random.Random(7)  # reproducible stroke jitter


def rough(points, over=3.0, jitter=0.7):
    """Polygon as (fill path, outline): each edge is its own stroke that slightly overshoots its corners."""
    fill = "M" + " L".join(f"{x:g} {y:g}" for x, y in points) + " Z"
    line = ""
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        a, b, bend = RNG.uniform(0, over), RNG.uniform(0, over), RNG.uniform(-0.8, 0.8)
        sx, sy = x0 - ux * a + RNG.uniform(-jitter, jitter), y0 - uy * a + RNG.uniform(-jitter, jitter)
        ex, ey = x1 + ux * b + RNG.uniform(-jitter, jitter), y1 + uy * b + RNG.uniform(-jitter, jitter)
        line += f"M{sx:.1f} {sy:.1f} Q{(sx + ex) / 2 - uy * bend:.1f} {(sy + ey) / 2 + ux * bend:.1f} {ex:.1f} {ey:.1f} "
    return fill, line


def rough_box(x, y, w, h, **kw):
    """Axis-aligned rough rectangle."""
    return rough([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], **kw)


def pill(x, y, w, h):
    """Capsule for the action-policy head."""
    r = h / 2
    d = f"M{x + r} {y} H{x + w - r} A{r} {r} 0 0 1 {x + w - r} {y + h} H{x + r} A{r} {r} 0 0 1 {x + r} {y} Z"
    return d, d


def disc(cx, cy, r):
    """Circle as a (fill, outline) pair."""
    d = f"M{cx - r} {cy} a{r} {r} 0 1 0 {2 * r} 0 a{r} {r} 0 1 0 {-2 * r} 0"
    return d, d


class Svg:
    """SVG drawing buffer with animation times expressed as percentages of the loop."""

    def __init__(self, duration):
        self.duration, self.items, self.css = duration, [], []

    def add(self, shape):
        self.items.append(shape)

    def animate(self, points, ease=SMOOTH, extra=""):
        """Build keyframes from percentage/property pairs on the shared loop."""
        name = f"swm-minimal-k{len(self.css)}"
        points = [(0, points[0][1]), *points, (100, points[-1][1])]
        frames = "".join(f"{t:g}%{{" + ";".join(f"{k}:{v}" for k, v in props.items()) + "}" for t, props in points)
        self.css.append(f"@keyframes {name}{{{frames}}}")
        return f'style="{extra}animation:{name} {self.duration:g}s {ease} infinite"'

    def activity(self, shape, start, end):
        """Animate a module's fill and outline during its active interval."""
        fill, line = shape
        anim = self.animate([(start, {"opacity": 0}), (start + 2.5, {"opacity": 1}), (end, {"opacity": 1}), (end + 5, {"opacity": 0})])
        self.add(f'<g opacity="0" {anim}><path d="{fill}" fill="{INK}" fill-opacity="0.1"/>'
                 f'<path d="{line}" fill="none" stroke="{INK}" stroke-width="1.7"/></g>')

    def module(self, shape, start, end, hatch=False):
        """Draw a module silhouette and its animated activity fill."""
        fill, line = shape
        self.add(f'<path d="{fill}" fill="{PLATE}"/>')
        if hatch:
            self.add(f'<path d="{fill}" fill="url(#swm-minimal-hatch)" opacity="0.3"/>')
        self.add(f'<path d="{line}" fill="none" stroke="{EDGE}" stroke-width="0.9"/>')
        self.activity(shape, start, end)

    def connect(self, path, start, duration=10):
        """Draw an arrow with an animated stroke and packet."""
        self.add(f'<path d="{path}" fill="none" stroke="{WIRE}" stroke-width="1.05" marker-end="url(#swm-minimal-arrow)"/>')
        anim = self.animate([(start, {"opacity": 0, "stroke-dashoffset": 18}), (start + 0.5, {"opacity": 1, "stroke-dashoffset": 18}),
                             (start + duration, {"opacity": 1, "stroke-dashoffset": -100}),
                             (start + duration + 0.5, {"opacity": 0, "stroke-dashoffset": -100})], "linear")
        self.add(f'<path d="{path}" pathLength="100" fill="none" stroke="{INK}" stroke-width="2.8" '
                 f'stroke-dasharray="18 200" opacity="0" {anim}/>')
        visible = self.animate([(start, {"opacity": 0}), (start + 0.5, {"opacity": 1}),
                                (start + duration - 0.5, {"opacity": 1}), (start + duration, {"opacity": 0})], "linear")
        self.add(f'<g class="swm-minimal-packet" opacity="0" {visible}><circle r="4.5" fill="{INK}" stroke="{PLATE}" stroke-width="1.6">'
                 f'<animateMotion path="{path}" dur="{self.duration:g}s" repeatCount="indefinite" calcMode="linear" '
                 f'keyTimes="0;{start / 100:g};{(start + duration) / 100:g};1" keyPoints="0;0;1;1"/></circle></g>')

    def code(self, x, y, start, shades, vertical=False):
        """Draw a hatched code strip with staggered cell animation."""
        lift = "translate(5px,0)" if vertical else "translate(0,-5px)"
        for i, shade in enumerate(shades):
            phase = start + i * 0.9
            anim = self.animate([(phase, {"opacity": 0.4, "transform": "translate(0,0)"}), (phase + 2.5, {"opacity": 1, "transform": lift}),
                                 (phase + 7, {"opacity": 1, "transform": "translate(0,0)"}), (86, {"opacity": 1, "transform": "translate(0,0)"}),
                                 (94, {"opacity": 0.4, "transform": "translate(0,0)"})])
            cx, cy, w, h = (x, y + i * 19, 26, 14) if vertical else (x + i * 18, y, 14, 32)
            fill, line = rough_box(cx, cy, w, h, over=1.0, jitter=0.3)
            self.add(f'<g {anim}><path d="{fill}" fill="{shade}"/><path d="{line}" fill="none" stroke="{EDGE}" stroke-width="0.7"/></g>')

    def text(self, x, y, value, size=13, anchor="middle"):
        """Typeset a short label separately from the sketch linework."""
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{SANS}" font-size="{size}" fill="{INK}">{escape(value)}</text>')

    def symbol(self, x, y, value, subscript="", size=24):
        """Typeset a component symbol with an optional subscript."""
        suffix = f'<tspan baseline-shift="sub" font-size="65%">{escape(subscript)}</tspan>' if subscript else ""
        self.add(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{SERIF}" font-size="{size}" '
                 f'font-style="italic" fill="{INK}">{escape(value)}{suffix}</text>')

    def panel(self, x, y, w, h, title):
        """Draw a border around an input pathway."""
        fill, line = rough_box(x, y, w, h, over=7, jitter=1.0)
        self.add(f'<path d="{fill}" fill="#ffffff" fill-opacity="0.7"/><path d="{line}" fill="none" stroke="{LINE}" stroke-width="0.7"/>')
        self.text(x + 22, y + 32, title, size=18, anchor="start")


# ── Figure ───────────────────────────────────────────
def build(cfg: MinimalArchitectureConfig):
    """Compose the monochrome architecture and its single-update animation."""
    g = Svg(cfg.duration)
    g.panel(20, 20, 510, 202, "Scene memory")
    g.panel(20, 244, 510, 312, "State dynamics")

    # ── Connections and data pulses ───────────────────
    g.connect("M140 126 Q157 125.4 174 126", 3, 6)
    g.connect("M256 126 Q297 126.7 338 126", 11, 6)
    g.connect("M490 126 Q530 125.4 558 126 Q572 126 572 140 Q571.3 205 572 270", 20, 15)
    g.connect("M356 148 Q355.2 228 356 308", 20, 10)
    g.connect("M106 348 Q124 347.4 142 348", 23, 7)
    g.connect("M196 348 Q239 348.7 282 348", 30, 7)
    g.connect("M148 483 Q167 482.4 186 483", 18, 6)
    g.connect("M272 483 Q305 483.6 338 483", 27, 6)
    g.connect("M414 461 Q414.7 425 414 390", 37, 10)
    g.connect("M490 483 Q530 483.7 558 483 Q572 483 572 469 Q571.4 391 572 314", 37, 14)
    g.connect("M594 292 Q604 291.6 612 292", 52, 7)
    g.connect("M648 292 Q662 292.4 672 292", 60, 6)
    g.connect("M748 292 Q758 291.5 766 292", 68, 6)
    g.connect("M316 390 V402 Q316 414 304 414 Q213 414.8 122 414 Q110 414 110 426 V444", 55, 17)

    # ── Scene memory ─────────────────────────────────
    g.module(rough_box(42, 80, 92, 92), 2, 12)
    maze = ("M52 90 H88 V102 H112 V114 H124 M52 102 V150 H76 V138 H64 V126 H88 V114 "
            "M76 90 V114 M100 90 V126 H124 V162 H100 V150 H88 V162 H52 M100 138 H112")
    g.add(f'<path d="{maze}" fill="none" stroke="{EDGE}" stroke-width="0.95"/>')
    scan = g.animate([(2, {"opacity": 0, "transform": "translateX(0)"}), (4, {"opacity": 1, "transform": "translateX(0)"}),
                      (11, {"opacity": 1, "transform": "translateX(70px)"}), (13, {"opacity": 0, "transform": "translateX(70px)"})], "linear")
    g.add(f'<g opacity="0" {scan}><rect x="50" y="88" width="20" height="76" rx="2" fill="{INK}" fill-opacity="0.1"/>'
          f'<path d="M70 88 V164" stroke="{INK}" stroke-width="1.2"/></g>')
    g.text(88, 194, "Map")
    g.module(rough([(180, 82), (250, 104), (250, 148), (180, 170)]), 8, 16, hatch=True)
    g.symbol(213, 135, "q", "φ", size=26)
    shades = ("#eeeeee", "#cfcfcf", "#ababab", "#808080", "#bdbdbd", "#999999", "#dedede", "#707070")
    g.code(344, 110, 16, shades)
    g.symbol(414, 174, "z")

    # ── State and action-conditioned update ───────────
    g.module(rough([(42, 516), (120, 516), (162, 450), (84, 450)], over=2), 70, 88)
    for i in range(1, 7):
        u = v = i / 7
        g.add(f'<path d="M{42 + 78 * u:g} 516 l42 -66 M{42 + 42 * v:g} {516 - 66 * v:g} h78" fill="none" stroke="{LINE}" stroke-width="0.5"/>')
    move = g.animate([(70, {"transform": "translate(0,0) rotate(0deg)", "opacity": 1}), (88, {"transform": "translate(40px,-24px) rotate(-18deg)", "opacity": 1}),
                      (92, {"transform": "translate(40px,-24px) rotate(-18deg)", "opacity": 1}), (95, {"transform": "translate(40px,-24px) rotate(-18deg)", "opacity": 0}),
                      (96, {"transform": "translate(0,0)", "opacity": 0}), (99, {"transform": "translate(0,0)", "opacity": 1})])
    g.add(f'<g transform="translate(86,493)"><g {move}>'
          f'<path d="M0 0 L20 -19 Q29 -9 23 4 Z" fill="{INK}" fill-opacity="0.1" stroke="{LINE}" stroke-width="0.9"/>'
          f'<path d="M11 -9 L-7 -6 L-2 1 L-3 10 Z" fill="{INK}" stroke="#ffffff" stroke-width="0.9"/></g></g>')
    g.text(102, 548, "State")
    g.module(rough([(192, 442), (266, 461), (266, 505), (192, 524)]), 23, 32, hatch=True)
    g.symbol(227, 489, "π", "s", size=25)
    g.code(344, 467, 32, shades[2:] + shades[:2])
    g.text(414, 531, "State code")

    g.module(rough_box(288, 314, 210, 70), 46, 56)
    g.symbol(393, 357, "T", "ψ", size=28)
    g.module(rough_box(42, 316, 58, 64, over=2), 19, 28)
    g.add(f'<path d="M59 358 Q70 357.3 81 358 V364 H59 Z M70 358 V342" fill="none" stroke="{EDGE}" stroke-width="0.85"/>'
          f'<circle cx="70" cy="337" r="5" fill="none" stroke="{INK}" stroke-width="0.85"/>')
    g.text(71, 402, "Action")
    g.module(pill(148, 336, 42, 24), 27, 35)
    g.symbol(169, 352, "π", "a", size=17)

    # ── Fusion and one abstract observation ───────────
    g.module(disc(572, 292, 16), 49, 57)
    g.add(f'<path d="M565 292 H579 M572 285 V299" fill="none" stroke="{INK}" stroke-width="0.9"/>')
    g.text(629, 206, "Fusion")
    g.code(616, 219, 58, shades[1:] + shades[:1], vertical=True)
    g.symbol(629, 402, "f")
    g.module(rough([(676, 245), (742, 204), (742, 380), (676, 339)]), 64, 72, hatch=True)
    g.symbol(710, 300, "D", "θ", size=28)

    g.module(rough_box(770, 228, 128, 128, over=4), 74, 82)
    reveal = g.animate([(68, {"opacity": 0.5, "transform": "scale(0.96)"}), (77, {"opacity": 1, "transform": "scale(1)"}),
                        (88, {"opacity": 1, "transform": "translate(-4px,2px) scale(1.14)"}), (96, {"opacity": 0.5, "transform": "scale(0.96)"})])
    walls = [([(974, 236), (1003, 258), (1003, 322), (974, 348)], "#f1f1f1"),
             ([(1086, 236), (1060, 258), (1060, 322), (1086, 348)], "#ededed"),
             ([(974, 348), (1003, 322), (1060, 322), (1086, 348)], "#f7f7f7")]
    g.add(f'<g transform="translate(-196,0)"><g clip-path="url(#swm-minimal-view)"><g transform="translate(1030,292)"><g {reveal}><g transform="translate(-1030,-292)">')
    for points, fill in walls:
        outline = rough(points, over=0, jitter=0.4)
        g.add(f'<path d="{outline[0]}" fill="{fill}"/><path d="{outline[1]}" fill="none" stroke="{WIRE}" stroke-width="0.65"/>')
    door = rough_box(1003, 258, 57, 64, over=0, jitter=0.4)
    g.add(f'<path d="{door[0]}" fill="{PLATE}"/><path d="{door[1]}" fill="none" stroke="{WIRE}" stroke-width="0.65"/>'
          f'<path d="M1022 322 V282 Q1032 281.4 1042 282 V322" fill="#e5e5e5" stroke="{EDGE}" stroke-width="0.65"/>'
          f'<path d="M974 236 Q1030 235.2 1086 236 M988 348 L1015 322 M1013 348 L1026 322 M1048 348 L1042 322 M1073 348 L1053 322 M994 330 H1068 M983 340 H1078" '
          f'fill="none" stroke="{LINE}" stroke-width="0.55"/>'
          f'<path d="M978 251 L999 267 M978 260 L999 276 M978 269 L999 285 M978 278 L999 294" fill="none" stroke="{LINE}" stroke-width="0.5"/></g></g></g></g></g>')
    g.text(834, 393, "Observation")

    # ── Standalone SVG and reduced-motion still ───────
    reduced = '@media(prefers-reduced-motion:reduce){.swm-minimal *{animation:none!important}.swm-minimal .swm-minimal-packet{display:none}}'
    marker = (f'<marker id="swm-minimal-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" '
              f'orient="auto"><path d="M1 1 L7 4 L1 7" fill="none" stroke="{WIRE}" stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round"/></marker>')
    hatch = (f'<pattern id="swm-minimal-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(-34)">'
             f'<path d="M0 0 V7" fill="none" stroke="{EDGE}" stroke-width="0.45"/></pattern>')
    clip = '<clipPath id="swm-minimal-view"><rect x="970" y="232" width="120" height="120" rx="6"/></clipPath>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            'class="swm-minimal" role="img" aria-labelledby="swm-minimal-title swm-minimal-desc" '
            'style="display:block;max-width:100%;height:auto" stroke-linecap="round" stroke-linejoin="round">\n'
            '<title id="swm-minimal-title">Space-Aware World Model, monochrome sketch</title>\n'
            '<desc id="swm-minimal-desc">Scene and state codes merge into one observation. An action-conditioned transition updates the state. '
            'Moving packets show data flow, and the observation shifts as the state moves.</desc>\n'
            f'<defs>{marker}{hatch}{clip}</defs>\n<style>{"".join(g.css)}{reduced}</style>\n' + "\n".join(g.items) + '\n</svg>\n')


# ── Main ─────────────────────────────────────────────
def main(cfg: MinimalArchitectureConfig):
    """Write the self-contained animated SVG to cfg.output."""
    Path(cfg.output).write_text(build(cfg), encoding="utf-8")
    print(f"[INFO] wrote {cfg.output}")


if __name__ == "__main__":
    defaults = MinimalArchitectureConfig()
    parser = argparse.ArgumentParser(description="Render the monochrome sketch of the SWM architecture.")
    parser.add_argument("--output", default=defaults.output, help="destination SVG path")
    parser.add_argument("--duration", type=float, default=defaults.duration, help="loop duration in seconds")
    main(MinimalArchitectureConfig(**vars(parser.parse_args())))
