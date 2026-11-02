#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SWM inference speed chart (SVG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Draw the results of src/tools/benchmark_swm_speed.py: frames per second per environment
and total frames per second against the number of parallel environments, per device, with
the real-time target and the largest batch that keeps it.

Both axes are logarithmic (base 2 for environments, base 10 for rates).

Example:
    python assets/make_speed_chart.py --results assets/swm_speed_dmlab.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Import from this checkout when launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyrallis

from assets.svg_utils import BLUE, HAIRLINE, INK, LABEL, MONO, document, polyline, text, title

DEVICE_STYLE = {"cuda": ("GPU", BLUE), "cpu": ("CPU", INK)}


# ── Config ───────────────────────────────────────────
@dataclass
class SpeedChartConfig:
    """Benchmark results and output file."""

    results: str = "assets/swm_speed_dmlab.json"
    output: str = "assets/swm_speed_dmlab.svg"


# ── Plot ─────────────────────────────────────────────
def log_plot(x0, y0, w, h, label, unit, series, x_max, y_range, target=None):
    """Log-log panel: (name, color, envs [N], values [N], marker index or None) series over envs 1..x_max."""
    lo, hi = (math.floor(math.log10(y_range[0])), math.ceil(math.log10(y_range[1])))

    def px(envs, values):
        return np.stack([np.log2(envs) / math.log2(x_max) * w, h - (np.log10(values) - lo) / (hi - lo) * h], 1)

    out = [f'<g transform="translate({x0},{y0})">', text(0, -10, label, size=10, family=MONO, fill=INK), text(w, -10, unit, size=10, family=MONO, anchor="end")]
    for e in range(lo, hi + 1):  # decade grid
        y = h - (e - lo) / (hi - lo) * h
        out.append(f'<path d="M0 {y:.1f}h{w}" stroke="{HAIRLINE}"/>' + text(-6, y + 3.5, f"{10 ** e:g}", size=9, family=MONO, anchor="end"))
    for k in range(int(math.log2(x_max)) + 1):  # power-of-two ticks
        x = k / math.log2(x_max) * w
        out.append(f'<path d="M{x:.1f} {h}v4" stroke="{LABEL}"/>' + text(x, h + 15, f"{2 ** k}", size=9, family=MONO, anchor="middle"))
    if target is not None:
        y = px(np.ones(1), np.array([target]))[0, 1]
        out.append(f'<path d="M0 {y:.1f}h{w}" stroke="{LABEL}" stroke-dasharray="4 3"/>' + text(4, y + 12, f"{target:g} FPS target", size=9, family=MONO))
    for name, color, envs, values, marker in series:
        points = px(envs, values)
        out.append(polyline(points, color, 1.6) + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}"/>' for x, y in points))
        if marker is not None:
            x, y = points[marker]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="none" stroke="{color}" stroke-width="1.4"/>'
                       + text(x - 8, y - 8, f"{envs[marker]} envs", size=10, family=MONO, fill=color, anchor="end"))
    return out + [f'<rect width="{w}" height="{h}" fill="none" stroke="{HAIRLINE}"/>', "</g>"]


# ── Main ─────────────────────────────────────────────
def main(cfg: SpeedChartConfig):
    """Read the benchmark JSON and write the static SVG chart."""
    results = json.loads(Path(cfg.results).read_text())
    devices, target = results["devices"], results["config"]["target_fps"]
    data = {d: (np.array([r["envs"] for r in v["runs"]]), np.array([r["env_fps"] for r in v["runs"]]),
                np.array([r["total_fps"] for r in v["runs"]]), v["max_envs"]) for d, v in devices.items()}
    x_max = 2 ** math.ceil(math.log2(max(envs.max() for envs, *_ in data.values())))

    # ── Layout ────────────────────────────────────────
    margin, top, w, h, gap = 56, 64, 330, 220, 80
    width, height = 2 * margin + 2 * w + gap - 24, top + h + 152

    def series(values_index):
        out = []
        for device, (envs, env_fps, total_fps, max_envs) in data.items():
            name, color = DEVICE_STYLE[device]
            values = (env_fps, total_fps)[values_index]
            out.append((name, color, envs, values, int(np.flatnonzero(envs == max_envs)[0]) if max_envs in envs else None))
        return out

    env_range = (min(v[1].min() for v in data.values()), max(v[1].max() for v in data.values()))
    total_range = (min(v[2].min() for v in data.values()), max(v[2].max() for v in data.values()))
    body = [title(margin - 24, 36, "SWM INFERENCE SPEED · DMLAB")]
    body += log_plot(margin, top, w, h, "FPS per environment", "frames / s", series(0), x_max, env_range, target)
    body += log_plot(margin + w + gap, top, w, h, "total throughput", "frames / s", series(1), x_max, total_range)
    for x0 in (margin, margin + w + gap):
        body.append(text(x0 + w / 2, top + h + 32, "parallel environments (batch)", size=10, family=MONO, anchor="middle"))

    # ── Summary and hardware ──────────────────────────
    y = top + h + 72  # below the axis titles
    x = margin - 24
    for device, (envs, env_fps, total_fps, max_envs) in data.items():
        name, color = DEVICE_STYLE[device]
        body.append(f'<circle cx="{x + 4}" cy="{y - 4}" r="4" fill="{color}"/>' + text(x + 14, y, f"{name} · {devices[device]['dtype']}", size=11, fill=INK, extra=' font-weight="bold"'))
        body.append(text(x + 14, y + 17, f"single env {devices[device]['single_env_fps']:,.0f} FPS · {max_envs} envs at ≥ {target:g} FPS", size=10, family=MONO))
        x += w + gap
    hardware = f"{results.get('gpu', '')} · {results['cpu']}, {results['cpu_threads']} threads".lstrip(" ·")
    body.append(text(margin - 24, y + 52, hardware, size=10, family=MONO))
    body.append(text(margin - 24, y + 68, "one step = T_psi transition + G render of a 96 × 96 frame; observations stay on the device", size=10, family=MONO))

    output = Path(cfg.output)
    output.write_text(document(width, height, [], body))
    print(f"[INFO] {output}: {output.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main(pyrallis.parse(config_class=SpeedChartConfig))
