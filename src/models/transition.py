"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transition model T_psi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recurrent model of action-conditioned state increments
Delta s_t = T_psi(pi_s(s_t), pi_z(z), pi_a(a_t)), applied as s_{t+1} = s_t + Delta s_t, with the
persistent scene code z [B, d_z] as context.

The LSTM receives concatenated state features, projected scene codes, and action features.
States use [x, y, sin(theta), cos(theta), *rest]; increments use
[Delta x, Delta y, Delta theta, *Delta rest]; DMLab states have no rest coordinates.
The head predicts increments standardized with stored training statistics.
`step` de-normalizes them and applies heading increments in angle space before
converting back to sine-cosine features.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn

from src.models.layers import FourierFeatures


# ── Model ────────────────────────────────────────────
class TransitionModel(nn.Module):
    """Transition model T_psi over normalized states s_t, with the scene code z as context."""

    def __init__(self, latent_dim, state_dim, action_dim, delta_dim, num_bands, fourier_scale, hidden_dim, num_layers,
                 delta_mean, delta_std):
        super().__init__()
        width = latent_dim  # fusion width d_f = d_z
        # pi_s: [B, T, d_s] -> [B, T, 2 L d_s] -> [B, T, d_f]
        self.state_encoder = nn.Sequential(FourierFeatures(num_bands, fourier_scale), nn.Linear(2 * num_bands * state_dim, width),
                                           nn.RMSNorm(width, eps=1e-6))
        # pi_z: [B, d_z] -> [B, d_f]
        self.scene_projection = nn.Sequential(nn.Linear(latent_dim, width), nn.RMSNorm(width, eps=1e-6), nn.SiLU(inplace=True),
                                              nn.Linear(width, width))
        # pi_a: [B, T, d_a] -> [B, T, d_f / 2]
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, width // 2), nn.LayerNorm(width // 2), nn.SiLU(inplace=True))
        self.input_projection = nn.Linear(2 * width + width // 2, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.RMSNorm(hidden_dim // 2, eps=1e-6), nn.SiLU(inplace=True),
                                  nn.Linear(hidden_dim // 2, delta_dim))
        # Increment statistics of the training transitions: the head predicts (Delta s_t - mean) / std
        self.register_buffer("delta_mean", torch.tensor(delta_mean))
        self.register_buffer("delta_std", torch.tensor(delta_std))

        # ── Initialization ────────────────────────────
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight_ih"):
                nn.init.xavier_uniform_(param)
            elif name.startswith("weight_hh"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a transition training checkpoint on `device` in evaluation mode."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
        model = cls(**checkpoint["hyper_parameters"]["model"])
        model.load_state_dict(state_dict)
        return model.to(device).eval()

    def standardize(self, delta):
        """Increments Delta s_t [..., d_Delta] on the head's scale, (Delta s_t - mean) / std."""
        return (delta - self.delta_mean) / self.delta_std

    def forward(self, s, z, a, hidden=None):
        """Standardized increments [B, T, d_Delta] and the LSTM state (h, c) for states s [B, T, d_s], scene codes z [B, d_z], and
        actions a [B, T, d_a]; `hidden` continues a sequence."""
        z = self.scene_projection(z)[:, None].expand(-1, s.shape[1], -1)  # one scene code per sequence: [B, d_z] -> [B, T, d_f]
        x = self.input_projection(torch.cat([self.state_encoder(s), z, self.action_encoder(a)], dim=-1))  # [B, T, h_T]
        h, hidden = self.lstm(x, hidden)  # output [B, T, h_T]; h and c in hidden each [num_layers, B, h_T]
        return self.head(h), hidden

    def step(self, s, z, a, hidden=None):
        """One transition s_{t+1} = s_t + Delta s_t for s_t [B, d_s], z [B, d_z], and a_t [B, d_a]; returns s_{t+1}, the standardized
        increment [B, d_Delta], and the LSTM state."""
        delta, hidden = self(s[:, None], z, a[:, None], hidden)
        delta = delta[:, 0]
        return apply_increment(s, delta * self.delta_std + self.delta_mean), delta, hidden

    def rollout(self, s, z, actions):
        """Free-running states [B, T + 1, d_s] from s_0 [B, d_s] under actions [B, T, d_a], and their standardized increments
        [B, T, d_Delta]."""
        states, deltas, hidden = [s], [], None
        for a in actions.unbind(1):
            s, delta, hidden = self.step(s, z, a, hidden)
            states.append(s)
            deltas.append(delta)
        return torch.stack(states, dim=1), torch.stack(deltas, dim=1)


def apply_increment(s, delta):
    """Apply Delta s_t [..., d_Delta] to s_t [..., d_s], updating heading in angle space."""
    theta = torch.atan2(s[..., 2:3], s[..., 3:4]) + delta[..., 2:3]  # [..., 1]
    return torch.cat([s[..., :2] + delta[..., :2], theta.sin(), theta.cos(), s[..., 4:] + delta[..., 3:]], dim=-1)


# ── Loss / Objective ─────────────────────────────────
def transition_loss(model, s, z, a, delta, weights):
    """Teacher-forced objective: weighted MSE between predicted and target standardized increments.

    s [B, T, d_s] and a [B, T, d_a] are a window's recorded states and actions, delta [B, T - 1, d_Delta] its increments, and
    weights [d_Delta] the per-dimension loss weights; every step t < T - 1 reads the recorded s_t and a_t.
    """
    pred, _ = model(s[:, :-1], z, a[:, :-1])  # [B, T - 1, d_Delta]
    return ((pred - model.standardize(delta)).square() * weights).mean()


def rollout_errors(model, s, z, a, delta, weights):
    """Free-running rollout of each window from its first recorded state under the recorded actions.

    Returns the weighted MSE of the standardized increments predicted along the rollout, and the per-step position error
    [B, T] (Euclidean, normalized units) and absolute heading error [B, T] (degrees) against the recorded states s [B, T, d_s].
    """
    states, pred = model.rollout(s[:, 0], z, a[:, :-1])  # [B, T, d_s], [B, T - 1, d_Delta]
    loss = ((pred - model.standardize(delta)).square() * weights).mean()
    position = (states[..., :2] - s[..., :2]).norm(dim=-1)
    sin, cos = states[..., 2], states[..., 3]  # heading difference from the sine-cosine pairs, wrapped to [-pi, pi]
    heading = torch.atan2(sin * s[..., 3] - cos * s[..., 2], cos * s[..., 3] + sin * s[..., 2]).abs().rad2deg()
    return loss, position, heading
