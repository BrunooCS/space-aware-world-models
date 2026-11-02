#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWM rollout overview (animated SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw the SWM factorization unrolled over a recorded DMLab rollout: the scene encoder
q_phi maps the top-down map m to one scene code z, the transition model T_psi advances
the explicit state s_t, and the observation model G_theta renders o_hat_t from the
additive fusion of z and pi_s(s_t).

1) Read the scene map from the fast-sweeping tour and the renders and poses from the
   generator tour; both record maze seed 104 under the same policy.
2) Draw the scene pathway, the shared scene-code bus, and four rollout steps
   `stride` steps apart, each with its real G_theta render and its pose on the map.
3) Light one step at a time and reveal its frame, so z stays fixed while s_t advances.

Example:
    python assets/make_rollout_overview.py --stride 40
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import base64
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrallis
from PIL import Image

from assets.svg_utils import BLUE, HAIRLINE, INK, LABEL, MONO, document, image_uri, keyframes, polyline, text, title

INK2, VIOLET = "#353941", "#9443d0"
SERIF = "'Times New Roman',Times,serif"
W, H, TOUR_P = 960, 344, 240  # canvas and panel side of the source tours
BUS_Y, STATE_Y, FRAME_Y, FRAME = 200, 266, 60, 78
MAP_X, MAP_Y, MAP = 28, 138, 124
STEPS_X = [384, 524, 664, 804]  # x of the four shown rollout steps
CODE = [0.85, 0.4, 0.65, 0.3, 0.95, 0.5, 0.75, 0.35]  # fill opacity of the z cells


# ── Config ───────────────────────────────────────────
@dataclass
class RolloutOverviewConfig:
    """Source tours, spacing of the shown steps, loop duration, and output file."""

    generator_tour: str = "assets/swm_generator_dmlab.svg"
    fast_sweeping_tour: str = "assets/fast_sweeping_dmlab.svg"
    stride: int = 40              # recorded steps between consecutive shown states
    duration: float = 6.0
    output: str = "assets/swm_rollout_overview.svg"


# ── Inputs ───────────────────────────────────────────
def keyframe_stops(svg, name):
    """(percent, transform) stops of `@keyframes name` written by svg_utils."""
    block = re.search(rf"@keyframes {name}\{{((?:[^{{}}]*\{{[^}}]*\}})*)\}}", svg).group(1)
    return [(float(p), v) for p, v in re.findall(r"([\d.]+)%\{transform:([^}]*)\}", block)]


def read_tours(cfg):
    """Scene-map URI, renders [K, side, side, 3] and map poses [K, 2] (tour panel px) at steps 0, stride, ..."""
    tour, sweep = Path(cfg.generator_tour).read_text(), Path(cfg.fast_sweeping_tour).read_text()
    scene_map = re.search(r'<image href="(data:image/jpeg;base64,[^"]+)" width="240" height="240"', sweep).group(1)

    # ── Render sprite: row of frame k from the step-end keyframes ──
    uri, side = re.search(r'<image class="render" href="data:image/jpeg;base64,([^"]+)" width="(\d+)"', tour).groups()
    sprite, side = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri))).convert("RGB")), int(side)  # [U * side, side, 3]
    agent = keyframe_stops(tour, "agent")  # one stop per recorded step, plus the loop closure
    T = len(agent) - 1
    steps = [k * cfg.stride for k in range(len(STEPS_X))]
    rows = [int(float(re.search(r"-?[\d.]+", v).group(0)) / -side) for p, v in
            (max((s for s in keyframe_stops(tour, "render") if s[0] <= 100 * k / T + 1e-6), key=lambda s: s[0]) for k in steps)]
    renders = np.stack([sprite[r * side:(r + 1) * side] for r in rows])

    # ── Poses: translate(x, y) of the agent marker per step ──
    poses = np.array([[float(c) for c in re.findall(r"translate\(([\d.-]+)px,([\d.-]+)px\)", v)[0]] for _, v in agent[:T]])
    return scene_map, renders, poses, steps


# ── Primitives ───────────────────────────────────────
def math(x, y, parts, size=14, fill=INK, anchor="middle"):
    """Italic serif expression from (base, subscript) parts, e.g. [("G", "θ")]."""
    spans, shift = [], 0.0
    for base, sub in parts:
        spans.append(f'<tspan dy="{-shift:.1f}">{base}</tspan>')
        shift = size * 0.28 if sub else 0.0
        spans.append(f'<tspan font-size="{size * 0.68:.1f}" dy="{shift:.1f}">{sub}</tspan>' if sub else "")
    return f'<text x="{x}" y="{y}" font-family="{SERIF}" font-style="italic" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{"".join(spans)}</text>'


def arrow(points, color=INK2, marker="ink", width=1.2, extra=""):
    path = "M" + "L".join(f"{x} {y}" for x, y in points)
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#ro-{marker})"{extra}/>'


def markers():
    head = ('<marker id="ro-{}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">'
            '<path d="M1 1L7 4L1 7" fill="none" stroke="{}" stroke-width="1.2"/></marker>')
    return "<defs>" + "".join(head.format(name, color) for name, color in (("ink", INK2), ("blue", BLUE), ("violet", VIOLET))) + "</defs>"


def polygon(points):
    return f'<polygon points="{" ".join(f"{x},{y}" for x, y in points)}" fill="#fff" stroke="{INK2}" stroke-width="1.2" stroke-linejoin="round"/>'


# ── Figure ───────────────────────────────────────────
def scene_pathway(scene_map, poses, steps):
    """Map m with the shown poses, scene encoder q_phi, and the scene-code bus."""
    trail = poses[:steps[-1] + 1]
    body = [f'<g transform="translate({MAP_X},{MAP_Y}) scale({MAP / TOUR_P:.4f})"><image href="{scene_map}" width="{TOUR_P}" height="{TOUR_P}"/>',
            polyline(trail, BLUE, 2.6, ' stroke-opacity=".7"')]
    body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{BLUE}" stroke="#fff" stroke-width="2"/>' for x, y in poses[steps]]
    body += [f'<circle class="on{t}" cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" stroke="{BLUE}" stroke-width="3"/>' for t, (x, y) in enumerate(poses[steps])]
    body += [f'<rect width="{TOUR_P}" height="{TOUR_P}" rx="8" fill="none" stroke="{HAIRLINE}" stroke-width="2"/></g>',
             text(MAP_X + MAP / 2, MAP_Y + MAP + 20, "top-down map", size=10, anchor="middle"),
             math(MAP_X + MAP / 2, MAP_Y + MAP + 36, [("m", "")], size=13, fill=LABEL),
             arrow([(MAP_X + MAP + 4, BUS_Y), (166, BUS_Y)]),
             polygon([(170, BUS_Y - 32), (210, BUS_Y - 14), (210, BUS_Y + 14), (170, BUS_Y + 32)]), math(189, BUS_Y + 5, [("q", "φ")], size=15),
             arrow([(214, BUS_Y), (220, BUS_Y)])]
    body += [f'<rect x="{224 + 10 * k}" y="{BUS_Y - 9}" width="9" height="18" rx="2" fill="{VIOLET}" fill-opacity="{a}"/>' for k, a in enumerate(CODE)]
    body += [math(263, BUS_Y - 18, [("z", "")], size=16, fill=VIOLET), text(263, BUS_Y + 27, "scene code", size=10, anchor="middle"),
             text(263, BUS_Y + 41, "once per episode", size=10, anchor="middle"),
             f'<path d="M308 {BUS_Y}H898" stroke="{VIOLET}" stroke-width="2" stroke-opacity=".85"/>', text(920, BUS_Y + 4, "…", size=14, anchor="middle")]
    return body


def rollout_step(t, cx, step, stride, render):
    """Fusion, renderer, real frame, state, transition, and action of shown step t; returns (static, highlight, frame)."""
    tx, last = cx + 70, t == len(STEPS_X) - 1  # transition model between shown states
    fusion = (f'<circle cx="{cx}" cy="{BUS_Y}" r="9" fill="#fff" stroke="{INK2}" stroke-width="1.2"/>'
              f'<path d="M{cx - 5} {BUS_Y}H{cx + 5}M{cx} {BUS_Y - 5}V{BUS_Y + 5}" stroke="{INK2}" stroke-width="1.2"/>')
    static = [fusion, arrow([(cx, STATE_Y - 16), (cx, BUS_Y + 11)], BLUE, "blue"), f'<path d="M{cx} {BUS_Y - 9}V176" stroke="{INK2}" stroke-width="1.2"/>',
              polygon([(cx - 11, 176), (cx + 11, 176), (cx + 26, 150), (cx - 26, 150)]),
              f'<path d="M{cx} 150V{FRAME_Y + FRAME + 2}" stroke="{INK2}" stroke-width="1.2"/>',
              f'<rect x="{cx - FRAME / 2}" y="{FRAME_Y}" width="{FRAME}" height="{FRAME}" rx="5" fill="#fff" stroke="{HAIRLINE}" stroke-dasharray="3 3"/>',
              math(cx, FRAME_Y - 8, [("ô", str(step))], size=14),
              f'<circle cx="{cx}" cy="{STATE_Y}" r="15" fill="#fff" stroke="{BLUE}" stroke-width="1.5"/>', math(cx - 2, STATE_Y + 4, [("s", str(step))], size=13, fill=BLUE),
              arrow([(cx + 16, STATE_Y), (tx - 25, STATE_Y)], BLUE, "blue"),
              f'<rect x="{tx - 23}" y="{STATE_Y - 13}" width="46" height="26" rx="6" fill="#fff" stroke="{INK2}" stroke-width="1.2"/>',
              math(tx - 2, STATE_Y + 5, [("T", "ψ")], size=14), f'<circle cx="{tx}" cy="{BUS_Y}" r="2.6" fill="{VIOLET}"/>',
              arrow([(tx, BUS_Y + 3), (tx, STATE_Y - 15)], VIOLET, "violet", 1.1, ' stroke-opacity=".75"'),
              arrow([(tx, STATE_Y + 42), (tx, STATE_Y + 15)]), math(tx, STATE_Y + 58, [("a", f"{step}–{step + stride - 1}")], size=14, fill=LABEL)]
    static.append(arrow([(tx + 23, STATE_Y), (cx + 140 - 15, STATE_Y)], BLUE, "blue") if not last else
                  f'<path d="M{tx + 23} {STATE_Y}H898" stroke="{BLUE}" stroke-width="1.2" stroke-dasharray="3 4"/>' + text(920, STATE_Y + 4, "…", size=14, anchor="middle"))
    if t == 0:  # name each module once, at the first step
        static += [math(cx - 30, 168, [("G", "θ")], size=14, anchor="end"), math(cx - 8, 238, [("π", "s")], size=13, fill=LABEL, anchor="end")]
    highlight = (f'<g class="on{t}"><circle cx="{cx}" cy="{STATE_Y}" r="15" fill="{BLUE}"/>{math(cx - 2, STATE_Y + 4, [("s", str(step))], size=13, fill="#fff")}'
                 f'<circle cx="{cx}" cy="{BUS_Y}" r="9" fill="none" stroke="{BLUE}" stroke-width="1.6"/>'
                 f'<rect x="{cx - FRAME / 2 - 2}" y="{FRAME_Y - 2}" width="{FRAME + 4}" height="{FRAME + 4}" rx="6" fill="none" stroke="{BLUE}" stroke-width="1.6"/></g>')
    frame = (f'<g class="fr{t}"><image href="{image_uri(render, 90)}" x="{cx - FRAME / 2}" y="{FRAME_Y}" width="{FRAME}" height="{FRAME}"/>'
             f'<rect x="{cx - FRAME / 2}" y="{FRAME_Y}" width="{FRAME}" height="{FRAME}" rx="2" fill="none" stroke="{HAIRLINE}"/></g>')
    return static, highlight, frame


def step_css(n, duration):
    """Keyframes that light step t during the t-th slot and keep its frame rendered for the rest of the loop."""
    css, slot = [], 100 / n
    for t in range(n):
        on = [(0, int(t == 0))] + ([(slot * t, 1)] if t else []) + ([(slot * (t + 1), 0)] if t < n - 1 else []) + [(100, int(t == n - 1))]
        css += [f".on{t}{{opacity:0;animation:on{t} {duration:.2f}s step-end infinite}}", keyframes(f"on{t}", on, "opacity")]
        if t:  # frame 0 is rendered as soon as the rollout starts
            css += [f".fr{t}{{animation:fr{t} {duration:.2f}s step-end infinite}}", keyframes(f"fr{t}", [(0, 0), (slot * t, 1), (100, 1)], "opacity")]
    return css


# ── Main ─────────────────────────────────────────────
def main(cfg: RolloutOverviewConfig):
    """Write the animated rollout overview SVG."""
    scene_map, renders, poses, steps = read_tours(cfg)
    body = [markers(), title(28, 36, "SPACE-AWARE WORLD MODEL · ROLLOUT"),
            text(W - 28, 36, "z is inferred once per episode; only the explicit state s_t is updated", size=10, family=MONO, anchor="end")]
    body += scene_pathway(scene_map, poses, steps)
    shown = [rollout_step(t, cx, steps[t], cfg.stride, renders[t]) for t, cx in enumerate(STEPS_X)]
    body += [element for static, _, _ in shown for element in static]
    body += [frame for _, _, frame in shown] + [highlight for _, highlight, _ in shown]

    output = Path(cfg.output)
    output.write_text(document(W, H, step_css(len(STEPS_X), cfg.duration), body))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=RolloutOverviewConfig))
