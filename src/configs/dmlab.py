"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DMLab state, action, and transition-target transforms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The scene map m is a top-down RGB image; o_t is a first-person RGB observation.
Stored simulator states are [x, y, theta_deg]. Model coordinates are:
    s_t       = [x_n, y_n, sin(theta), cos(theta)]                 (d_s = 4)
    a_t       = [turn_n, forward]                                 (d_a = 2)
    Delta s_t = [Delta x_n, Delta y_n, Delta theta]                (d_Delta = 3)

Positions use fixed affine scaling, with normalized Y following image rows.
Heading increments are wrapped radians. All scene maps span the same world range.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass
import numpy as np


# ── Config ───────────────────────────────────────────
@dataclass
class DMLabEnvConfig:
    """DMLab coordinate scales and the s_t, a_t, and Delta s_t interfaces."""

    # ── World Coordinates ──────────────────────────────────────────────────
    world_min:   float = 50.0
    world_max:   float = 1250.0
    pos_center:  float = 650.0
    pos_half:    float = 600.0

    # ── State & Action Dimensions ──────────────────────────────────────────
    turn_min:          float = -90.0
    turn_max:          float = 90.0
    state_raw_dim:     int = 3      # (x, y, theta_deg)
    state_dim:         int = 4      # d_s: (x_n, y_n, sin(theta), cos(theta))
    state_delta_dim:   int = 3      # d_Delta: (Delta x_n, Delta y_n, Delta theta)
    action_raw_dim:    int = 7      # DMLab raw action
    action_dim:        int = 2      # d_a: (turn_n, forward)
    action_turn_idx:   int = 0
    action_forward_idx: int = 3
    state_raw_names: tuple = ("x", "y", "theta (deg)")
    action_raw_names: tuple = ("turn", "look", "strafe", "forward", "fire", "jump", "crouch")

    # ── Image Sizes ────────────────────────────────────────────────────────
    default_render_width:       int = 396
    default_render_height:      int = 240
    default_ego_img_width:      int = 96
    default_ego_img_height:     int = 96
    default_topdown_img_width:  int = 128
    default_topdown_img_height: int = 128
    default_topdown_grid_size:  int = 128

    # ── Normalization Bounds ───────────────────────────────────────────────
    norm_clip_min: float = -1.0
    norm_clip_max: float = 1.0

    turn_speed: int = 20

    # ── Transition targets ────────────────────────────
    # Weights follow (Delta x_n, Delta y_n, Delta theta).
    delta_loss_weights: tuple = (1.0, 1.0, 1.0)

    # ── State and action transforms ───────────────────
    def normalize_state(self, raw_state):
        """Map simulator coordinates [..., 3] to normalized s_t [..., 4]."""
        x, y, theta_deg = raw_state[..., 0], raw_state[..., 1], raw_state[..., 2]
        y_inv = self.world_max - y
        x_norm = (x - self.pos_center) / self.pos_half
        y_norm = (y_inv - self.pos_center) / self.pos_half
        theta = np.deg2rad(theta_deg)
        state = np.stack([x_norm, y_norm, np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)  # [..., 4]
        return np.clip(state, self.norm_clip_min, self.norm_clip_max)

    def compute_state_deltas(self, raw_states, states):
        """Return Delta s_t with wrapped radian heading increments."""
        # [..., T, 3] raw states and [..., T, 4] normalized states -> [..., T-1, 3] targets.
        delta_xy = states[..., 1:, :2] - states[..., :-1, :2]
        delta_deg = raw_states[..., 1:, 2] - raw_states[..., :-1, 2]
        delta_deg = (delta_deg + 180) % 360 - 180
        delta_theta = np.deg2rad(delta_deg)
        return np.concatenate([delta_xy, delta_theta[..., None]], axis=-1).astype(np.float32)  # [..., T-1, 3]

    def valid_state_mask(self, raw_states):
        """Per-step mask of positions that normalize inside [-1, 1] without clipping."""
        x_norm = (raw_states[..., 0] - self.pos_center) / self.pos_half
        y_norm = (self.world_max - raw_states[..., 1] - self.pos_center) / self.pos_half
        return (np.abs(x_norm) <= 1) & (np.abs(y_norm) <= 1)

    def normalize_action(self, raw_action):
        """Map simulator actions [..., 7] to navigation actions a_t [..., 2]."""
        turn    = raw_action[..., self.action_turn_idx].astype(np.float32)
        forward = raw_action[..., self.action_forward_idx].astype(np.float32)
        turn_norm = 2.0 * (turn - self.turn_min) / (self.turn_max - self.turn_min) - 1.0
        turn_norm = np.clip(turn_norm, self.norm_clip_min, self.norm_clip_max)
        forward   = np.clip(forward, 0.0, 1.0)
        return np.stack([turn_norm, forward], axis=-1).astype(np.float32)

    def manual_action(self, forward, turn):
        """Map (forward, turn) in {-1, 0, 1} to a raw action [7]; +turn rotates right."""
        action = np.zeros(self.action_raw_dim, dtype=np.intc)
        action[self.action_forward_idx] = forward
        action[self.action_turn_idx] = turn * self.turn_speed
        return action

    def denormalize_xy(self, x_norm, y_norm):
        """Map normalized positions back to DMLab world coordinates."""
        x = x_norm * self.pos_half + self.pos_center
        y_inv = y_norm * self.pos_half + self.pos_center
        y = self.world_max - y_inv
        return x, y

    def denormalize_state(self, state):
        """Recover simulator coordinates [..., 3], with principal heading in degrees."""
        x_norm, y_norm = state[..., 0], state[..., 1]
        sin_theta, cos_theta = state[..., 2], state[..., 3]
        x, y = self.denormalize_xy(x_norm, y_norm)
        theta_deg = np.rad2deg(np.arctan2(sin_theta, cos_theta))
        return np.stack([x, y, theta_deg], axis=-1).astype(np.float32)  # [..., 4] -> [..., 3]

    def world_to_pixel(self, x, y, map_size=None):
        """Convert world coordinates to pixel coordinates on top-down map."""
        map_size = self.default_topdown_grid_size if map_size is None else map_size
        scale = map_size / (self.world_max - self.world_min)
        px = (x - self.world_min) * scale
        py = (self.world_max - y) * scale              # Y-inverted
        return px, py

    def map_bounds(self, episodes):
        """Scene-map extents (x_min, y_min, x_max, y_max) [N, 4] of the rollout `episodes` group, in stored-state coordinates."""
        return np.tile(np.float32([self.world_min, self.world_min, self.world_max, self.world_max]), (len(episodes["state"]), 1))

    def map_heading(self, raw_state):
        """Heading in radians, counterclockwise from the scene map's +x axis."""
        return np.deg2rad(raw_state[..., 2])
