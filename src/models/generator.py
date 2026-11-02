"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Observation model G_theta (generator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deterministic renderer o_hat_t = G_theta(z, pi_s(s_t)) of an egocentric RGB frame
[B, 3, 96, 96] in [-1, 1] from a scene code z [B, d_z] and a normalized state s_t
[B, d_s].

1) Fourier-encode s_t and project it to the scene-code width d_z.
2) Fuse f_t = alpha z + (1 - alpha) pi_s(s_t), with a learned scalar alpha
   bounded to [eps, 1 - eps].
3) Decode f_t through residual MLP and convolutional blocks, resize to the frame
   size, and apply the tanh RGB head.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import FourierFeatures


# ── Blocks ───────────────────────────────────────────
class ResMLP(nn.Module):
    """Residual MLP block x + Linear(SiLU(RMSNorm(x)))."""

    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.RMSNorm(width, eps=1e-6), nn.SiLU(inplace=True), nn.Linear(width, width))

    def forward(self, x):
        return x + self.block(x)


class ResBlock(nn.Module):
    """Pre-activation residual block x + Conv3x3(SiLU(GN(Conv3x3(SiLU(GN(x)))))), with 8 GroupNorm groups."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1),
                                   nn.GroupNorm(8, channels), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


def up_block(in_ch, out_ch):
    """Learned 2x upsampling: 4x4 stride-2 transposed conv -> 3x3 conv -> SiLU."""
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.SiLU(inplace=True))


class ConvDecoder(nn.Module):
    """Spatial features [B, C0, H0, W0] -> RGB frame [B, 3, S, S] in [-1, 1]; each of the four upsamplings halves the channels."""

    def __init__(self, channels, image_size):
        super().__init__()
        widths = [channels // 2**i for i in range(5)]  # [C0, C0/2, C0/4, C0/8, C0/16]
        self.image_size = image_size
        self.upsampling = nn.Sequential(*(nn.Sequential(ResBlock(a), up_block(a, b)) for a, b in zip(widths[:-1], widths[1:])), ResBlock(widths[-1]))
        self.head = nn.Sequential(nn.Conv2d(widths[-1], 32, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(32, 3, 1), nn.Tanh())

    def forward(self, x):
        x = self.upsampling(x)  # [B, C0, 4, 4] -> [B, C0/16, 64, 64]
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)  # -> [B, C0/16, S, S]
        return self.head(x)  # -> [B, 3, S, S]


# ── Model ────────────────────────────────────────────
class Generator(nn.Module):
    """Observation model G_theta(z, pi_s(s_t)) = D_theta(f_t) with bounded additive scene-state fusion."""

    def __init__(self, latent_dim, state_dim, image_size, num_bands, fourier_scale, state_dropout, alpha_min, mlp_width, mlp_blocks,
                 base_channels, base_size):
        super().__init__()
        self.alpha_min, self.base_shape = alpha_min, (base_channels, base_size, base_size)
        # pi_s: [B, d_s] -> [B, 2 L d_s] -> [B, d_z]
        self.state_encoder = nn.Sequential(FourierFeatures(num_bands, fourier_scale), nn.Linear(2 * num_bands * state_dim, latent_dim),
                                           nn.LayerNorm(latent_dim), nn.Dropout(state_dropout))
        self.alpha_logit = nn.Parameter(torch.zeros(()))  # alpha = 0.5 at initialization
        self.fusion_norm = nn.LayerNorm(latent_dim)
        # [B, d_z] -> [B, h_G] -> [B, C0 * H0 * W0]
        self.stem = nn.Sequential(nn.Linear(latent_dim, mlp_width), nn.RMSNorm(mlp_width, eps=1e-6), nn.SiLU(inplace=True),
                                  *(ResMLP(mlp_width) for _ in range(mlp_blocks)), nn.Linear(mlp_width, base_channels * base_size**2))
        self.decoder = ConvDecoder(base_channels, image_size)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a generator training checkpoint on `device` in evaluation mode."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["hyper_parameters"]["model"])
        # Exclude the checkpoint's frozen LPIPS network.
        model.load_state_dict({key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")})
        return model.to(device).eval()

    @property
    def alpha(self):
        """Scene coefficient alpha = eps + (1 - 2 eps) sigmoid(a) in [eps, 1 - eps]; the state coefficient is 1 - alpha."""
        return self.alpha_min + (1 - 2 * self.alpha_min) * torch.sigmoid(self.alpha_logit)

    def fuse(self, z, s):
        """Fusion vector f_t = alpha z + (1 - alpha) pi_s(s_t) [B, d_z], before the decoder's LayerNorm."""
        alpha = self.alpha
        return alpha * z + (1 - alpha) * self.state_encoder(s)

    def decode(self, f):
        """Render frames [B, 3, S, S] in [-1, 1] from fusion vectors f [B, d_z]."""
        # [B, d_z] -> [B, C0 * H0 * W0] -> [B, C0, H0, W0]
        x = self.stem(self.fusion_norm(f)).view(-1, *self.base_shape)
        return self.decoder(x)

    def forward(self, z, s):
        """Render o_hat_t [B, 3, S, S] in [-1, 1] from scene codes z [B, d_z] and normalized states s_t [B, d_s]."""
        return self.decode(self.fuse(z, s))


def normalize_frames(frames):
    """uint8 frames [B, H, W, 3] -> float [B, 3, H, W] in [-1, 1]; the permuted view is already channels-last for cuDNN."""
    return frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0


# ── Loss / Objective ─────────────────────────────────
def render_loss(model, lpips, z, s, o, mse_weight, lpips_weight):
    """L_render = mse_weight * MSE + lpips_weight * LPIPS between o_hat_t = G(z, s_t) and o_t, both [B, 3, H, W] in [-1, 1]."""
    o_hat = model(z, s).float()
    mse = F.mse_loss(o_hat, o)
    perceptual = lpips(o_hat, o).mean()  # per-frame distances [B, 1, 1, 1]
    return mse_weight * mse + lpips_weight * perceptual, {"mse": mse, "lpips": perceptual}
