"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DDS-VAE scene encoder q_phi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two-stage variational encoder that maps a top-down map m [B, 3, 128, 128] in
[-1, 1] to a scene code z in R^{d_z}.

1) Stage 1 backbone: selection UNet -> DDS gates -> downscaling to the bottleneck
   h [B, 16, 16, 16]; upscaling -> refinement UNet reconstructs m.
2) Stage 2 mini-VAE: h -> (mu, log sigma^2) in R^{d_z} -> z -> h_hat, trained
   on the frozen, eval-mode Stage 1 backbone with the summed MSE on h, optionally
   mixed with a perceptual error of h_hat in the frozen upscaling features.

A Stage 2 checkpoint contains both stages.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVE_KL = 0.05  # nats; activity threshold for batch-mean KL, independent of free bits


# ── Blocks ───────────────────────────────────────────
def conv_bn(in_ch, out_ch):
    """3x3 conv -> BatchNorm -> SiLU, used by the UNets and scaling paths."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))


def conv_relu(in_ch, out_ch, kernel=3, stride=1):
    """Conv -> ReLU; kernel=stride=2 gives the learned 2x downsampling."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=(kernel - 1) // 2 if stride == 1 else 0), nn.ReLU(inplace=True))


def up_relu(in_ch, out_ch):
    """Learned 2x upsampling with a stride-2 transposed conv -> ReLU."""
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Four-level UNet with max-pool encoder and concatenated skip connections."""

    def __init__(self, in_ch, out_ch, base_ch):
        super().__init__()
        widths = [base_ch * 2**i for i in range(5)]  # [C, 2C, 4C, 8C, 16C]
        self.stem = conv_bn(in_ch, widths[0])
        self.down = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), conv_bn(a, b)) for a, b in zip(widths[:-1], widths[1:]))
        self.up = nn.ModuleList(nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths[:-1], widths[1:]))
        self.fuse = nn.ModuleList(conv_bn(2 * a, a) for a in widths[:-1])
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, x):
        # [B, in, H, W] -> skips at H, H/2, H/4, H/8; bottom at H/16 with 16C channels
        skips = [self.stem(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up, fuse in zip(reversed(self.up), reversed(self.fuse)):
            x = fuse(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)  # [B, out, H, W]


class DDS(nn.Module):
    """Dynamic data selection: hard-concrete gates plus a per-map top-k mask over all C*H*W input units."""

    def __init__(self, keep_ratio, temperature, gate_noise, exploration_prob, eps=1e-6):
        super().__init__()
        self.keep_ratio, self.temperature, self.gate_noise = keep_ratio, temperature, gate_noise
        self.exploration_prob, self.eps = exploration_prob, eps

    def forward(self, logits):
        # [B, C, H, W] -> [B, C*H*W]; gates and top-k ranking in fp32 under bf16 autocast
        u = logits.flatten(1).float() + 1.0
        if self.training:  # logistic noise of the concrete relaxation
            noise = torch.rand_like(u).clamp(self.eps, 1 - self.eps)
            u = u + self.gate_noise * (noise.log() - (-noise).log1p())
        gate = torch.sigmoid(u / self.temperature)
        mask = torch.zeros_like(gate).scatter_(1, gate.topk(int(u.shape[1] * self.keep_ratio), dim=1).indices, 1.0)
        if self.training:  # exploration: a random subset of maps keeps every unit
            mask = mask.maximum((torch.rand_like(gate[:, :1]) < self.exploration_prob).float())
        return (mask * gate).view_as(logits)  # [B, C, H, W], gated selection weights in [0, 1]


# ── Model ────────────────────────────────────────────
class SceneBackbone(nn.Module):
    """Stage 1: deterministic spatial compression m -> h and reconstruction h -> m_hat."""

    def __init__(self, base_channels, bottleneck_channels, keep_ratio, gate_temperature, gate_noise, exploration_prob):
        super().__init__()
        c = base_channels
        self.selector = UNet(3, 3, c)
        self.dds = DDS(keep_ratio, gate_temperature, gate_noise, exploration_prob)
        # [B, 3, 128, 128] -> [B, 16, 16, 16]
        self.downscaling = nn.Sequential(conv_bn(3, c), conv_relu(c, c, 2, 2), conv_bn(c, 2 * c), conv_relu(2 * c, 2 * c, 2, 2),
                                         conv_bn(2 * c, 4 * c), conv_relu(4 * c, 4 * c, 2, 2), nn.Conv2d(4 * c, bottleneck_channels, 1), nn.Tanh())
        # [B, 16, 16, 16] -> [B, 3, 128, 128]
        self.upscaling = nn.Sequential(conv_bn(bottleneck_channels, 2 * c), up_relu(2 * c, 2 * c), conv_bn(2 * c, 4 * c), up_relu(4 * c, 4 * c),
                                       conv_bn(4 * c, 4 * c), up_relu(4 * c, 4 * c), nn.Conv2d(4 * c, 3, 1), nn.Tanh())
        self.refiner = UNet(3, 3, c)

    def select(self, m):
        """DDS-gated map fed to the downscaling path, [B, 3, H, W]."""
        return m * self.dds(self.selector(m))

    def compress(self, m):
        return self.downscaling(self.select(m))  # [B, 3, 128, 128] -> h [B, 16, 16, 16]

    def render(self, h):
        return torch.tanh(self.refiner(self.upscaling(h)))  # h [B, 16, 16, 16] -> m_hat [B, 3, 128, 128]

    def features(self, h):
        """Return six upscaling activations and the tanh map for h [B, 16, 16, 16]."""
        *blocks, to_map, squash = self.upscaling
        maps = []
        for block in blocks:
            h = block(h)
            maps.append(h)
        return maps + [squash(to_map(h))]

    def forward(self, m):
        h = self.compress(m)
        return self.render(h), h


class MiniVAE(nn.Module):
    """Stage 2 bottleneck: 16x16x16 -> 8x8x128 -> 4x4x256 -> (mu, log sigma^2), and its mirrored decoder."""

    def __init__(self, bottleneck_channels, vae_channels, latent_dim):
        super().__init__()
        c, flat = vae_channels, 4 * vae_channels * 4 * 4
        self.encoder = nn.Sequential(conv_relu(bottleneck_channels, c), conv_relu(c, 2 * c, 2, 2), conv_relu(2 * c, 2 * c),
                                     conv_relu(2 * c, 4 * c, 2, 2), conv_relu(4 * c, 4 * c), nn.Flatten())
        self.posterior = nn.Linear(flat, 2 * latent_dim)  # joint (mu, log sigma^2) head
        self.decoder = nn.Sequential(nn.Linear(latent_dim, flat), nn.Unflatten(1, (4 * c, 4, 4)), up_relu(4 * c, 4 * c),
                                     conv_relu(4 * c, 2 * c), up_relu(2 * c, 2 * c), conv_relu(2 * c, c),
                                     nn.Conv2d(c, bottleneck_channels, 3, padding=1), nn.Tanh())

    def encode(self, h):
        return self.posterior(self.encoder(h)).chunk(2, dim=-1)  # h [B, 16, 16, 16] -> mu, logvar [B, d_z]

    def decode(self, z):
        return self.decoder(z)  # z [B, d_z] -> h_hat [B, 16, 16, 16]

    def forward(self, h):
        mu, logvar = self.encode(h)
        return self.decode(reparameterize(mu, logvar)), mu, logvar


class DDSVAE(nn.Module):
    """Scene encoder q_phi(z | m) with the Stage 1 backbone and Stage 2 mini-VAE."""

    def __init__(self, latent_dim, base_channels, bottleneck_channels, vae_channels, keep_ratio, gate_temperature, gate_noise, exploration_prob):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = SceneBackbone(base_channels, bottleneck_channels, keep_ratio, gate_temperature, gate_noise, exploration_prob)
        self.bottleneck = MiniVAE(bottleneck_channels, vae_channels, latent_dim)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a DDS-VAE stage checkpoint on `device` in evaluation mode."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["hyper_parameters"]["model"])
        model.load_state_dict({key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()})
        return model.to(device).eval()

    def encode(self, m):
        """Posterior parameters mu, log sigma^2 [B, d_z] for maps m [B, 3, 128, 128] in [-1, 1]."""
        return self.bottleneck.encode(self.backbone.compress(m))

    def decode(self, z):
        """Render maps [B, 3, 128, 128] in [-1, 1] from scene codes z [B, d_z]."""
        return self.backbone.render(self.bottleneck.decode(z))

    def forward(self, m, temperature=0.0):
        """Reconstruct maps through z = mu + temperature * sigma * eps; returns (m_hat, z)."""
        mu, logvar = self.encode(m)
        z = reparameterize(mu, logvar, temperature)
        return self.decode(z), z


def reparameterize(mu, logvar, temperature=1.0):
    return mu + temperature * torch.randn_like(mu) * (0.5 * logvar).exp()


def normalize_maps(maps):
    """uint8 maps [B, 3, H, W] -> float [-1, 1] in channels-last layout for cuDNN."""
    return (maps.float() / 127.5 - 1.0).contiguous(memory_format=torch.channels_last)


# ── Loss / Objective ─────────────────────────────────
def gaussian_kl(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) per sample and dimension, [B, d_z]."""
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)


def stage1_loss(model, m, l1_weight):
    """Stage 1 map reconstruction: MSE + l1_weight * L1 on m [B, 3, H, W]."""
    m_hat, _ = model.backbone(m)
    m_hat = m_hat.float()
    mse, l1 = F.mse_loss(m_hat, m), F.l1_loss(m_hat, m)
    return mse + l1_weight * l1, {"mse": mse, "l1": l1}


def perceptual_loss(backbone, h, h_hat):
    """Energy-normalized error of frozen upscaling features, scaled to the bottleneck MSE.

    Divide each layer's MSE by its mean squared target activation, average over
    layers, and multiply by the batch-mean squared norm of h.
    """
    # Reduce each sample before the batch to avoid strided channels-last reductions.
    with torch.no_grad():  # detach target features; predicted features retain gradients to h_hat
        targets = backbone.features(h)
        energies = [target.float().square().mean(dim=(1, 2, 3)).mean() for target in targets]
    errors = [(pred.float() - target.float()).square().mean(dim=(1, 2, 3)).mean() / energy
              for pred, target, energy in zip(backbone.features(h_hat), targets, energies)]
    return torch.stack(errors).mean() * h.square().sum(dim=(1, 2, 3)).mean()


def stage2_loss(model, m, beta, free_bits, perceptual_weight):
    """Stage 2 objective: (1 - w) * summed bottleneck MSE + w * perceptual + beta * free-bits KL, with w = perceptual_weight.

    `beta` is a 0-dim tensor so a scheduled KL weight does not recompile the compiled objective.
    """
    with torch.no_grad():
        h = model.backbone.compress(m).float()  # [B, 16, 16, 16], frozen Stage 1 target

    # ── Reconstruct h through the stochastic code ─────
    h_hat, mu, logvar = model.bottleneck(h)
    mu, logvar = mu.float(), logvar.float()
    recon = (h_hat.float() - h).square().sum(dim=(1, 2, 3)).mean()
    reconstruction, metrics = recon, {"recon": recon}
    if perceptual_weight:
        perceptual = perceptual_loss(model.backbone, h, h_hat)
        reconstruction = (1 - perceptual_weight) * recon + perceptual_weight * perceptual
        metrics["perceptual"] = perceptual

    # ── Free bits on the batch-averaged KL per dimension ──
    kl_dims = gaussian_kl(mu, logvar).mean(dim=0)  # [d_z]
    kl = kl_dims.clamp(min=free_bits).sum()
    metrics.update(kl=kl_dims.sum(), kl_free_bits=kl, active_dims=(kl_dims > ACTIVE_KL).sum().float(), sigma=(0.5 * logvar).exp().mean())
    return reconstruction + beta * kl, metrics
