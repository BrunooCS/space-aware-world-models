"""Building blocks shared by the SWM observation and transition models."""

import math

import torch
import torch.nn as nn


# ── Model ────────────────────────────────────────────
class FourierFeatures(nn.Module):
    """State features gamma(s) = [sin(2^k pi c s), cos(2^k pi c s)] for k < L bands and scale c, per state dimension."""

    def __init__(self, num_bands, scale):
        super().__init__()
        self.register_buffer("frequencies", math.pi * scale * 2.0 ** torch.arange(num_bands, dtype=torch.float32), persistent=False)

    def forward(self, s):
        # [..., d_s] -> [..., d_s, L] -> [..., d_s, L, 2] -> [..., 2 L d_s]
        x = s.unsqueeze(-1) * self.frequencies
        return torch.stack([x.sin(), x.cos()], dim=-1).flatten(-3)
