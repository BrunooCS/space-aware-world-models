#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Animated SWM architecture figure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a looping SVG of scene encoding, scene-state fusion, observation rendering,
and recurrent state updates.

1) Compose modules and connectors, then animate three rollout steps with CSS keyframes.
2) Embed raster inputs and optional font subsets, then save the SVG.

Map and frame inputs are embedded as PNG data URIs. `prefers-reduced-motion`
displays the final still.

Example:
    python assets/render_architecture.py --font_dir /path/to/fonts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import base64
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pyrallis
from fontTools import subset
from fontTools.ttLib import TTFont
from PIL import Image

# ── Config ───────────────────────────────────────────
@dataclass
class ArchitectureFigureConfig:
    """Inputs, output, and playback speed of the figure."""

    output: str = "assets/swm_architecture.svg"
    map_path: str = "data/dmlab_topdown_maps/1000002258.png"
    frame_paths: list[str] = field(default_factory=lambda: [f"assets/architecture_frames/frame_{k}.png" for k in range(3)])
    font_dir: str = ""      # Inter, DM Mono, and EB Garamond TTFs; empty uses system fonts
    slowdown: float = 1.4   # playback time per storyboard unit, in seconds


# ── Palette and storyboard ───────────────────────────
PAPER, CARD, PLATE = "#fbfbfb", "#ffffff", "#fafafa"
INK, INK2, MUTED, LINE, SLATE = "#191b21", "#353941", "#626b7a", "#dfe3ea", "#7d8899"
BLUE, BLUE_LINE, BLUE_GRID, BLUE_SOFT = "#164be6", "#577de0", "#ccd8f4", "#edf2ff"
SLATE_FILL, SLATE_LINE = "#f3f5f8", "#d7dce4"
RED, PURPLE, GREEN = "#e0245e", "#8a52e0", "#34a56f"
RED_STOPS = ["#c3093f", "#e3224c", "#f8503f", "#fd7a2c", "#ffa31a"]     # scene code
BLUE_STOPS = ["#0c3a78", "#0e66c8", "#1f93ff", "#62bdff", "#cdeeff"]    # state code
PURPLE_STOPS = ["#e6d6ff", "#b88af0", "#8a52e0", "#5a3cc0", "#2f2070"]  # fused code
ROSE, ROSE_EDGE, BLUE_EDGE, PURPLE_EDGE = "#f2a6b6", "#eba9b8", "#a9c0ee", "#bfaee6"
GRAY = "#d0d5de"                                                        # idle module borders
MINT, MINT_EDGE, MINT_ON, MINT_SWATCH = "#f7fcf9", "#d5eadf", "#e2f4ea", "#6fc39a"  # action: idle, idle border, active, legend
SANS = "Inter, 'Inter Variable', ui-sans-serif, system-ui, sans-serif"
MONO = "'DM Mono', ui-monospace, 'SF Mono', Menlo, monospace"
SERIF = "'EB Garamond', Georgia, 'Times New Roman', serif"
ROLES = {"label": (SANS, 13, INK), "small": (SANS, 11.5, MUTED), "title": (SANS, 17, INK), "heading": (SERIF, 25, INK)}
FACES = [("Inter", "normal", 400, "Inter-Regular.ttf"), ("Inter", "italic", 400, "Inter-Italic.ttf"),
         ("DM Mono", "normal", 400, "DMMono-Regular.ttf"), ("EB Garamond", "normal", 400, "EBGaramond-Regular.ttf"),
         ("EB Garamond", "italic", 400, "EBGaramond-Italic.ttf")]

W, H = 1440, 740
LOOP, END = 14.0, 13.4             # storyboard loop length and the time the episode fades out
STEPS = [2.4, 6.0, 9.6]            # start of each rollout step
EASE = "cubic-bezier(0.22,1,0.36,1)"
N, CELL_W, CELL_H, GAP = 8, 21, 36, 4
FX, FY = 770, 362                  # fusion node
Z_RIGHT, Z_TOP, S_TOP = 440, 141, 548


def mix(a, b, t):
    """Linear blend of two #rrggbb colors."""
    ca, cb = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def ramp(stops, t):
    """Color at position t in [0, 1] along a list of stops."""
    t = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return mix(stops[i], stops[i + 1], t - i)


def data_uri(image):
    """PIL image -> PNG data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ── SVG primitives ───────────────────────────────────
def math_spans(src, size):
    """Typeset math with italic letters, upright operators, and sub/superscripts."""
    toks, i = [], 0
    while i < len(src):
        if src[i] in "_^" and i + 1 < len(src):
            if src[i + 1] == "{":
                j = src.index("}", i + 2)
                toks.append((src[i], src[i + 2:j])); i = j + 1
            else:
                toks.append((src[i], src[i + 1])); i += 2
        else:
            toks.append(("", src[i])); i += 1

    def run(text, sz, dy):
        spans, buf, cur = [], "", None
        for c in text:
            italic = c.isalpha() and c not in "ℝ∈"
            if cur is None or italic == cur:
                buf += c; cur = italic
            else:
                spans.append((buf, cur)); buf, cur = c, italic
        spans.append((buf, cur))
        tags = []
        for k, (b, it) in enumerate(spans):
            attrs = (' font-style="italic"' if it else "") + (f' dy="{dy:.1f}"' if k == 0 and dy else "") + (f' font-size="{sz:.1f}"' if sz != size else "")
            tags.append(f"<tspan{attrs}>{escape(b)}</tspan>")
        return tags

    out, shift = [], 0.0
    for kind, text in toks:
        if kind == "":
            out += run(text, size, -shift); shift = 0.0
        else:
            target = size * (0.24 if kind == "_" else -0.38)
            out += run(text, size * 0.68, target - shift); shift = target
    return "".join(out)


class Svg:
    """Accumulates drawing items, defs, and CSS keyframes for one looping figure."""

    def __init__(self, slowdown):
        self.items, self.defs, self.css, self.markers, self.slowdown = [], [], [], {}, slowdown
        self.defs.append('<filter id="soft" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="6" '
                         'stdDeviation="9" flood-color="#284078" flood-opacity="0.09"/></filter>')

    def add(self, s):
        self.items.append(s)

    # ── Animation ─────────────────────────────────────
    def style(self, pts, ease=EASE, extra=""):
        """Inline `animation` for keyframes pts = [(storyboard time, {css property: value})]."""
        pts = sorted(pts, key=lambda p: p[0])
        if pts[0][0] > 0: pts = [(0, pts[0][1])] + pts
        if pts[-1][0] < LOOP: pts = pts + [(LOOP, pts[-1][1])]
        name = f"k{len(self.css)}"
        body = "".join(f"{100 * t / LOOP:.3f}%{{" + ";".join(f"{k}:{v}" for k, v in pr.items()) + "}" for t, pr in pts)
        self.css.append(f"@keyframes {name}{{{body}}}")
        return f'style="{extra}animation:{name} {LOOP * self.slowdown}s {ease} infinite"'

    @staticmethod
    def window(t0, t1, on, off, fade=0.2):
        return [(t0, off), (t0 + fade, on), (t1, on), (t1 + fade, off)]

    def flow(self, d, color, t0, dur, width=3.4):
        """A short dash that travels along connector path d."""
        pts = [(t0, {"opacity": 0, "stroke-dashoffset": 16}), (t0 + 0.03, {"opacity": 1, "stroke-dashoffset": 16}),
               (t0 + dur, {"opacity": 1, "stroke-dashoffset": -100}), (t0 + dur + 0.05, {"opacity": 0, "stroke-dashoffset": -100})]
        self.add(f'<path d="{d}" pathLength="100" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
                 f'stroke-dasharray="16 300" stroke-dashoffset="16" opacity="0" {self.style(pts, "linear")}/>')

    def glow(self, shape, t0, dur, color=BLUE):
        """Draw an outline overlay during a module's active interval."""
        self.add(f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" opacity="0" '
                 f'{self.style(self.window(t0, t0 + dur, {"opacity": 1}, {"opacity": 0}))}>{shape}</g>')

    # ── Shapes ────────────────────────────────────────
    def rect(self, x, y, w, h, r=0, fill="none", stroke=None, width=1.0, opacity=None, dash=None, shadow=False, anim=""):
        a = f' rx="{r}"' if r else ""
        a += f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        a += f' stroke-dasharray="{dash}"' if dash else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        a += ' filter="url(#soft)"' if shadow else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{a} {anim}/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, width=1.0):
        a = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{a}/>')

    def line(self, x0, y0, x1, y1, color, width=1.0):
        self.add(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')

    def polygon(self, pts, stroke, fill):
        self.add(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')

    def cells(self, x0, y0, step, size, stroke, cell_at):
        """Draw a code strip; cell_at(i) returns (resting fill, keyframes)."""
        for i in range(N):
            fill, pts = cell_at(i)
            self.add(f'<rect x="{x0 + i * step[0]}" y="{y0 + i * step[1]}" width="{size[0]}" height="{size[1]}" rx="4" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="0.8" {self.style(pts)}/>')

    def trapezoid(self, pts, fill_b, stroke, gid, r=6):
        """Return a rounded trapezoid path for reuse in animation overlays."""
        d = ""
        for i, (cx, cy) in enumerate(pts):
            (px, py), (nx, ny) = pts[i - 1], pts[(i + 1) % len(pts)]
            l1, l2 = math.hypot(cx - px, cy - py), math.hypot(nx - cx, ny - cy)
            a, b = (cx - (cx - px) / l1 * r, cy - (cy - py) / l1 * r), (cx + (nx - cx) / l2 * r, cy + (ny - cy) / l2 * r)
            d += f'{"M" if i == 0 else "L"}{a[0]:.1f},{a[1]:.1f} Q{cx},{cy} {b[0]:.1f},{b[1]:.1f} '
        self.defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff"/>'
                         f'<stop offset="1" stop-color="{fill_b}"/></linearGradient>')
        self.add(f'<path d="{d}Z" fill="url(#{gid})" stroke="{stroke}" stroke-width="1.1" stroke-linejoin="round"/>')
        return f'<path d="{d}Z"/>'

    def card(self, x, y, w, h, title):
        self.rect(x, y, w, h, r=18, fill=CARD, stroke=LINE, shadow=True)
        self.text(x + 28, y + 40, title, "heading")

    def module(self, x, y, w, h, r=9):
        self.defs.append(f'<linearGradient id="module" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{CARD}"/>'
                         f'<stop offset="0.55" stop-color="{mix(CARD, SLATE_FILL, 0.65)}"/><stop offset="1" stop-color="{SLATE_FILL}"/></linearGradient>')
        self.rect(x, y, w, h, r=r, fill="url(#module)", stroke=SLATE_LINE, shadow=True)

    def image(self, x, y, w, h, uri, r, cid, pixelated=False, anim=""):
        self.defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath>')
        px = ' image-rendering="pixelated"' if pixelated else ""
        self.add(f'<image href="{uri}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"{px} {anim}/>')

    # ── Text ──────────────────────────────────────────
    def text(self, x, y, s, role="label", anchor="start", family=None, size=None, fill=None):
        fam, sz, col = ROLES[role]
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family or fam}" font-size="{size or sz}" fill="{fill or col}"{a}>{escape(s)}</text>')

    def formula(self, x, y, src, size=16, anchor="start"):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{SERIF}" font-size="{size}" fill="{INK2}"{a}>{math_spans(src, size)}</text>')

    def label_math(self, x, y, label, src, size=15):
        """A name followed by its symbol, centered as one line."""
        self.add(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="13" fill="{INK}" text-anchor="middle">{escape(label)}'
                 f'<tspan dx="7" font-family="{SERIF}" font-size="{size}" fill="{INK2}">{math_spans(src, size)}</tspan></text>')

    # ── Connectors ────────────────────────────────────
    def marker(self, color, size):
        if (color, size) not in self.markers:
            mid = f"mk{len(self.markers)}"
            self.markers[(color, size)] = mid
            self.defs.append(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="{size}" markerHeight="{size}" '
                             f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0.6,0.8 L9.4,5 L0.6,9.2 Z" fill="{color}" '
                             f'stroke="{color}" stroke-width="0.6" stroke-linejoin="round"/></marker>')
        return self.markers[(color, size)]

    def connect(self, pts, color=SLATE, dash=None, width=1.6, head=14, r=10, gap=8):
        """Orthogonal connector with rounded corners and an arrowhead, `gap` px short of both end points
        (module edges); returns its path for flow pulses."""
        (x0, y0), (x1, y1) = pts[0], pts[1]
        n = math.hypot(x1 - x0, y1 - y0)
        pts = [(x0 + (x1 - x0) / n * gap, y0 + (y1 - y0) / n * gap)] + list(pts[1:])
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        n = math.hypot(x1 - x0, y1 - y0)
        pts = list(pts[:-1]) + [(x1 - (x1 - x0) / n * gap, y1 - (y1 - y0) / n * gap)]
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
            l1, l2 = math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by)
            rr = min(r, l1 / 2, l2 / 2)
            d += f" L{bx - (bx - ax) / l1 * rr:.1f},{by - (by - ay) / l1 * rr:.1f} Q{bx},{by} {bx + (cx - bx) / l2 * rr:.1f},{by + (cy - by) / l2 * rr:.1f}"
        d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
        dash = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash} '
                 f'marker-end="url(#{self.marker(color, head)})"/>')
        return d


def embedded_fonts(body, font_dir):
    """Return subset-embedded WOFF @font-face rules for the figure's glyphs."""
    chars = set(" 0123456789.,-") | set("".join(re.findall(r">([^<>]+)<", body)).replace("&amp;", "&"))
    rules = []
    for family, style, weight, filename in FACES:
        if family not in body or (style == "italic" and 'font-style="italic"' not in body):
            continue
        font = TTFont(Path(font_dir) / filename)
        options = subset.Options(); options.flavor, options.name_IDs, options.hinting = "woff", [], False
        subsetter = subset.Subsetter(options); subsetter.populate(unicodes=[ord(c) for c in chars]); subsetter.subset(font)
        buffer = io.BytesIO(); font.flavor = "woff"; font.save(buffer)
        rules.append(f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
                     f"src:url(data:font/woff;base64,{base64.b64encode(buffer.getvalue()).decode()}) format('woff')}}")
    return "".join(rules)


# ── Figure ───────────────────────────────────────────
def build(cfg):
    """Compose the figure and storyboard as SVG text."""
    g = Svg(cfg.slowdown)
    clip = lambda v: min(1.0, max(0.0, v))
    frames = [data_uri(Image.open(p).convert("RGB")) for p in cfg.frame_paths]

    # Code colors: z is fixed, the state and fused codes change tone at every step
    jitter = [[0.10, -0.05, 0.08, -0.10, 0.06, -0.08, 0.10, -0.04], [0.16, -0.12, 0.14, -0.16, 0.12, -0.14, 0.16, -0.10],
              [-0.14, 0.16, -0.12, 0.18, -0.16, 0.12, -0.12, 0.14]]
    zc = [ramp(RED_STOPS, i / (N - 1)) for i in range(N)]
    sc = [[ramp(BLUE_STOPS, i / (N - 1) + jitter[k][i]) for i in range(N)] for k in range(3)]
    fc = [[ramp(PURPLE_STOPS, i / (N - 1) - 0.8 * jitter[k][i]) for i in range(N)] for k in range(3)]

    def stepped(colors, empty, i, t0, t1):
        """Keyframes of cell i: empty, then each step's color (staggered by cell), emptied at END."""
        pts = [(0, {"fill": empty})]
        for k, tau in enumerate(STEPS):
            pts += [(tau + t0 + 0.05 * i, {"fill": empty if k == 0 else colors[k - 1][i]}), (tau + t1 + 0.05 * i, {"fill": colors[k][i]})]
        return pts + [(END, {"fill": colors[2][i]}), (END + 0.4, {"fill": empty})]

    g.card(24, 24, 640, 248, "Persistent scene memory")
    g.card(24, 296, 640, 400, "Explicit state dynamics")

    # ── Scene memory ──────────────────────────────────
    g.image(52, 94, 130, 130, data_uri(Image.open(cfg.map_path).convert("RGB")), 12, "map", pixelated=True)
    scan = [(0.2, {"opacity": 0, "transform": "translateX(0)"}), (0.35, {"opacity": 0.28, "transform": "translateX(0)"}),
            (1.2, {"opacity": 0.28, "transform": "translateX(114px)"}), (1.4, {"opacity": 0, "transform": "translateX(114px)"})]
    g.add(f'<rect x="52" y="94" width="16" height="130" rx="8" fill="{RED}" opacity="0" {g.style(scan, "ease-in-out")}/>')
    g.label_math(117, 252, "Top-down map", "m")
    map_to_encoder = g.connect([(182, 159), (244, 159)])
    encoder = g.trapezoid([(244, 100), (338, 128), (338, 190), (244, 218)], "#fff3f5", GRAY, "encoder")
    g.formula(289, 166, "q_φ(z | m)", 16, "middle")
    g.text(291, 252, "Scene encoder", anchor="middle")
    encoder_to_z = g.connect([(338, 159), (Z_RIGHT, 159)])
    reveal = lambda i: [(1.25 + 0.07 * i, {"opacity": 0.15}), (1.5 + 0.07 * i, {"opacity": 1}), (END, {"opacity": 1}), (END + 0.4, {"opacity": 0.15})]
    g.cells(Z_RIGHT, Z_TOP, (CELL_W + GAP, 0), (CELL_W, CELL_H), ROSE_EDGE, lambda i: (zc[i], reveal(i)))
    g.formula(598, 206, "z ∈ ℝ^{d}", 17, "middle")
    g.text(598, 226, "once per episode", "small", "middle")
    g.text(538, 96, "or sampled from the prior", "small", "middle")
    g.formula(538, 122, "z ~ N(0, I)", 16, "middle")
    g.connect([(538, 128), (538, 140)], ROSE, "3 3", 1.2, 7, gap=0)

    # ── Fusion and observation model ──────────────────
    z_to_fusion = g.connect([(636, 159), (FX, 159), (FX, FY - 18)], ROSE, width=1.8)
    z_to_transition = g.connect([(520, 177), (520, 364)], ROSE, width=1.8)
    g.formula(510, 300, "z", 18, "end")
    g.circle(FX, FY, 18, CARD, INK2, 1.5)
    g.line(FX - 8, FY, FX + 8, FY, INK2, 1.5)
    g.line(FX, FY - 8, FX, FY + 8, INK2, 1.5)
    fusion_out = g.connect([(FX + 18, FY), (966, FY)], width=1.8)
    g.formula(872, FY - 22, "f_t = α z + (1−α) π_s(s_t)", 15, "middle")
    g.text(868, FY + 34, "bounded additive fusion", "small", "middle")
    g.cells(966, FY - 102, (0, 26), (36, 22), "#d9cdf2", lambda i: (fc[2][i], stepped(fc, "#f1ecfa", i, 2.3, 2.6)))
    g.formula(984, 494, "f_t ∈ ℝ^{d}", 17, "middle")
    decoder = g.trapezoid([(1050, FY - 65), (1148, FY - 130), (1148, FY + 130), (1050, FY + 65)], "#f6f1ff", GRAY, "decoder", r=7)
    g.formula(1100, FY + 8, "D_θ", 28, "middle")
    g.text(1100, FY + 34, "decoder", "small", "middle")
    g.label_math(1100, 530, "Observation model", "G_θ", 17)

    frame_x, frame_size = 1192, 200
    frame_y = FY - frame_size // 2
    g.rect(frame_x, frame_y, frame_size, frame_size, r=16, fill=PLATE, stroke=LINE, dash="3 3")
    for k, tau in enumerate(STEPS):
        t_on, t_off = tau + 3.05, (STEPS[k + 1] + 3.05 if k < 2 else END)
        fade = [(t_on, {"opacity": 0}), (t_on + 0.4, {"opacity": 1}), (t_off, {"opacity": 1}), (t_off + 0.4, {"opacity": 0 if k == 2 else 1})]
        g.image(frame_x, frame_y, frame_size, frame_size, frames[k], 16, f"frame{k}", anim=f'opacity="{1 if k == 2 else 0}" {g.style(fade)}')
        g.add(f'<text x="{frame_x + frame_size / 2}" y="520" text-anchor="middle" font-family="{MONO}" font-size="12" fill="{BLUE}" '
              f'opacity="{1 if k == 2 else 0}" {g.style(g.window(t_on, t_off, {"opacity": 1}, {"opacity": 0}))}>step t = {k}</text>')
    g.rect(frame_x + 0.5, frame_y + 0.5, frame_size - 1, frame_size - 1, r=15.5, stroke=INK, opacity=0.08)
    g.label_math(frame_x + frame_size / 2, 494, "Rendered frame", "ô_t", 17)

    # ── Explicit state dynamics ───────────────────────
    plate = np.array([[56, 612], [168, 612], [222, 524], [110, 524]], dtype=float)  # state plane, projected
    point = lambda u, v: plate[0] + u * (plate[1] - plate[0]) + v * (plate[3] - plate[0])
    g.polygon(plate, BLUE_GRID, "#f6f8fe")
    for i in range(1, 7):
        g.line(*point(i / 7, 0), *point(i / 7, 1), BLUE_GRID, 0.6)
        g.line(*point(0, i / 7), *point(1, i / 7), BLUE_GRID, 0.6)
    poses = [point(0.20, 0.24), point(0.38, 0.40), point(0.57, 0.50), point(0.75, 0.68)]
    heading = [math.degrees(math.atan2(*(poses[min(k + 1, 3)] - poses[min(k, 2)])[::-1])) for k in range(4)]
    for k in range(3):
        g.add(f'<circle cx="{poses[k][0]:.1f}" cy="{poses[k][1]:.1f}" r="2.6" fill="{BLUE_LINE}" opacity="0.7" '
              f'{g.style(g.window(STEPS[k] + 3.1, END, {"opacity": 0.7}, {"opacity": 0}))}/>')
    pose = lambda k: f"translate({(poses[k] - poses[0])[0]:.1f}px,{(poses[k] - poses[0])[1]:.1f}px) rotate({heading[k]:.1f}deg)"
    move = [(0, {"transform": pose(0), "opacity": 1})]
    for k in range(3):
        move += [(STEPS[k] + 3.0, {"transform": pose(k)}), (STEPS[k] + 3.45, {"transform": pose(k + 1)})]
    move += [(END, {"opacity": 1}), (END + 0.25, {"opacity": 0}), (END + 0.3, {"opacity": 0, "transform": pose(0)}), (LOOP - 0.01, {"opacity": 1})]
    g.add(f'<g transform="translate({poses[0][0]:.1f},{poses[0][1]:.1f})"><g {g.style(move, "ease-in-out")}>'
          f'<path d="M12,0 L-7,-8 L-3,0 L-7,8 Z" fill="{BLUE}" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></g></g>')
    g.label_math(139, 652, "State", "s_t ∈ ℝ^{k}", 17)
    plate_to_encoder = g.connect([(196, 566), (272, 566)])

    state_encoder = g.trapezoid([(272, 510), (364, 532), (364, 600), (272, 622)], BLUE_SOFT, GRAY, "state-encoder")
    g.formula(317, 575, "π_s", 24, "middle")
    g.text(318, 656, "State encoder", anchor="middle")
    g.text(318, 674, "Fourier features + linear", "small", "middle")
    encoder_to_code = g.connect([(364, 566), (Z_RIGHT, 566)])
    g.cells(Z_RIGHT, S_TOP, (CELL_W + GAP, 0), (CELL_W, CELL_H), BLUE_GRID, lambda i: (sc[2][i], stepped(sc, "#eef1f7", i, 1.1, 1.4)))
    g.formula(538, 618, "π_s(s_t) ∈ ℝ^{d}", 17, "middle")
    g.text(538, 638, "state code", "small", "middle")
    code_to_fusion = g.connect([(636, 566), (FX, 566), (FX, FY + 18)], width=1.8)
    code_to_transition = g.connect([(538, S_TOP), (538, 450)], width=1.8)

    g.module(380, 364, 260, 86)
    g.text(510, 396, "Transition model", "title", "middle")
    g.formula(510, 428, "T_ψ(π_s(s_t), z, π_a(a_t))", 17, "middle")
    idle, lit = {"fill": MINT, "stroke": MINT_EDGE}, {"fill": MINT_ON, "stroke": GREEN}
    active = lambda t0, t1: [(0, idle)] + [pt for tau in STEPS for pt in g.window(tau + t0, tau + t1, lit, idle)]
    g.rect(52, 368, 76, 80, r=16, fill=MINT, stroke=MINT_EDGE, width=1.4, shadow=True, anim=g.style(active(0.3, 1.0)))
    g.rect(74, 414, 32, 11, r=4, stroke=INK2, width=1.3)
    g.line(90, 414, 90, 397, INK2, 1.7)
    g.circle(90, 391, 6.5, stroke=INK2, width=1.5)
    g.formula(90, 441, "a_t", 17, "middle")
    g.rect(190, 393, 44, 26, r=13, fill=MINT, stroke=MINT_EDGE, width=1.4, anim=g.style(active(0.7, 1.5)))
    g.formula(212, 412, "π_a", 16, "middle")
    action_to_chip = g.connect([(128, 406), (190, 406)])
    chip_to_transition = g.connect([(234, 406), (380, 406)])
    update = g.connect([(410, 450), (410, 478), (150, 478), (150, 524)], dash="5 5")
    g.formula(424, 472, "Δs_t", 16)
    g.formula(280, 466, "s_{t+1} = s_t + Δs_t", 16, "middle")

    # ── Rollout column ────────────────────────────────
    g.text(712, 592, "Rollout")
    g.text(712, 612, "z is held fixed, only the state evolves", "small")
    for j, (color, edge, label) in enumerate([(zc[1], ROSE_EDGE, "scene code, inferred once from the map"), (sc[0][3], BLUE_EDGE, "state code, recomputed at every step"),
                                             (fc[0][3], PURPLE_EDGE, "fused code, decoded to the frame"),
                                             (MINT_SWATCH, MINT_EDGE, "action, applied at every step")]):
        g.rect(712, 628 + 24 * j, 20, 14, r=4, fill=color, stroke=edge, width=0.8)
        g.text(742, 640 + 24 * j, label, "small")
    for k, tau in enumerate(STEPS):
        x = 1040 + 126 * k
        g.rect(x, 584, 112, 112, r=12, fill=PLATE, stroke=LINE, dash="3 3")
        appear = [(tau + 3.2, {"opacity": 0}), (tau + 3.6, {"opacity": 1}), (END, {"opacity": 1}), (END + 0.4, {"opacity": 0})]
        g.image(x, 584, 112, 112, frames[k], 12, f"thumb{k}", anim=f'opacity="1" {g.style(appear)}')
        g.rect(x + 0.5, 584.5, 111, 111, r=11.5, stroke=INK, opacity=0.08)
        g.text(x + 56, 718, f"t = {k}", family=MONO, size=12, fill=MUTED, anchor="middle")

    # ── Storyboard events ─────────────────────────────
    plate_outline = f'<path d="M{" L".join(f"{x:.0f},{y:.0f}" for x, y in plate)} Z"/>'
    g.flow(map_to_encoder, RED, 0.25, 0.7); g.glow(encoder, 0.6, 0.9, RED); g.flow(encoder_to_z, RED, 1.0, 0.8)
    for tau in STEPS:
        g.flow(plate_to_encoder, BLUE, tau + 0.15, 0.5); g.glow(state_encoder, tau + 0.5, 0.7); g.flow(encoder_to_code, BLUE, tau + 0.85, 0.5)
        g.flow(code_to_transition, BLUE, tau + 1.5, 0.5); g.flow(code_to_fusion, BLUE, tau + 1.5, 1.2)
        g.flow(z_to_transition, RED, tau + 0.9, 1.0); g.flow(z_to_fusion, RED, tau + 1.3, 1.2)
        g.flow(action_to_chip, GREEN, tau + 0.5, 0.5); g.flow(chip_to_transition, GREEN, tau + 0.95, 0.9)
        g.glow('<rect x="380" y="364" width="260" height="86" rx="9"/>', tau + 2.0, 0.8)
        g.flow(update, BLUE, tau + 2.6, 0.9); g.glow(plate_outline, tau + 3.0, 0.5)
        g.flow(fusion_out, PURPLE, tau + 2.5, 0.5); g.glow(decoder, tau + 3.0, 0.5, PURPLE)

    body = "".join(g.items)
    fonts = embedded_fonts(body, cfg.font_dir) if cfg.font_dir else ""
    reduced = "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" font-kerning="normal">'
            f"<title>Space-Aware World Model architecture</title><style>{fonts}{''.join(g.css)}{reduced}</style>"
            f'<defs>{"".join(g.defs)}</defs><rect width="{W}" height="{H}" fill="{PAPER}"/>{body}</svg>\n')


# ── Main ─────────────────────────────────────────────
def main(cfg: ArchitectureFigureConfig):
    """Write the animated architecture SVG to cfg.output."""
    Path(cfg.output).write_text(build(cfg), encoding="utf-8")
    print(f"[INFO] wrote {cfg.output}")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=ArchitectureFigureConfig))
