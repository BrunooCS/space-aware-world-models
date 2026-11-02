"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rollout episodes for the observation and transition models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads the HDF5 datasets written by `make_rollouts_*`: one scene map m per episode
(`episodes/topdown`, [N, 128, 128, 3]), egocentric frames o_t (`episodes/ego`,
[N, T, 96, 96, 3], one compressed chunk per frame), raw simulator states and actions
(`episodes/state`, [N, T, d_raw]; `episodes/action`), and reset flags (`episodes/done`).

Splits operate at episode level. Scene posteriors are computed once; normalized
states and transitions stay in host memory. Frames are read in worker processes.
Transition windows exclude resets and states outside the normalization range.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.dds_vae import DDSVAE, normalize_maps


# ── Inputs / Data ────────────────────────────────────
def split_episodes(num_episodes, val_fraction, seed):
    """Seeded episode-level split into sorted (train, validation) episode indices."""
    episodes = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(seed)).numpy()
    num_val = int(num_episodes * val_fraction)
    return np.sort(episodes[num_val:]), np.sort(episodes[:num_val])


def load_states(path, env_config):
    """Normalized states s_t [N, T, d_s] (float32) of every episode, from the stored raw simulator states."""
    with h5py.File(path, "r") as f:
        return env_config.normalize_state(f["episodes/state"][...])


@torch.no_grad()
def encode_scenes(path, checkpoint, device="cuda"):
    """DDS-VAE posterior parameters (mu, log sigma^2), each [N, d_z] on the CPU, of every episode's scene map m."""
    encoder = DDSVAE.from_checkpoint(checkpoint, device)
    with h5py.File(path, "r") as f:
        maps = f["episodes/topdown"]  # [N, 128, 128, 3] uint8
        # [B, H, W, 3] -> [B, 3, H, W] channels-last view -> (mu, logvar) -> [B, 2 d_z]
        posteriors = [torch.cat(encoder.encode(normalize_maps(torch.from_numpy(maps[i:i + 256]).to(device).permute(0, 3, 1, 2))), dim=1).cpu()
                      for i in range(0, len(maps), 256)]
    return torch.cat(posteriors).chunk(2, dim=1)


def load_transitions(path, env_config):
    """Normalized states s_t [N, T, d_s], actions a_t [N, T, d_a], increments Delta s_t [N, T - 1, d_Delta], and the usable-transition
    mask [N, T - 1] of every episode: a transition is usable without a simulator reset and with both states inside the coordinate range."""
    with h5py.File(path, "r") as f:
        raw_states, raw_actions, done = (f[f"episodes/{key}"][...] for key in ("state", "action", "done"))
    states, inside = env_config.normalize_state(raw_states), env_config.valid_state_mask(raw_states)
    valid = ~done[:, :-1] & inside[:, :-1] & inside[:, 1:]  # done[t]: the simulator reset after step t, so s_{t+1} starts anew
    return states, env_config.normalize_action(raw_actions), env_config.compute_state_deltas(raw_states, states), valid


def sequence_windows(valid, episodes, length, stride):
    """(episode, start) rows [W, 2] of the `length`-state windows of `episodes`, every `stride` steps, with only usable transitions."""
    starts = np.arange(0, valid.shape[1] + 2 - length, stride)
    # Unchanged prefix counts identify windows with no invalid transitions.
    invalid = np.pad(np.cumsum(~valid[episodes], axis=1), ((0, 0), (1, 0)))  # [E, T]
    rows, columns = np.nonzero(invalid[:, starts + length - 1] == invalid[:, starts])
    return np.stack([episodes[rows], starts[columns]], axis=1)


def read_scene_maps(path, episodes, env_config):
    """uint8 scene maps m [B, H, W, 3] of `episodes` and their extents (x_min, y_min, x_max, y_max) [B, 4] in stored-state coordinates."""
    with h5py.File(path, "r") as f:
        return np.stack([f["episodes/topdown"][e] for e in episodes]), env_config.map_bounds(f["episodes"])[episodes]


def read_frames(path, episodes, steps):
    """uint8 frames o_t [B, H, W, 3] at (episode, step) index pairs; each frame is its own HDF5 chunk."""
    with h5py.File(path, "r") as f:
        frames = f["episodes/ego"]
        return torch.from_numpy(np.stack([frames[e, t] for e, t in zip(episodes, steps)]))


class RolloutFrameDataset(Dataset):
    """Frames of `episodes` with their states s_t and episode indices; an item is a whole batch of frame positions.

    Positions are episode-major. Each batch opens its own HDF5 handle;
    frames remain uint8 until device transfer.
    """

    def __init__(self, path, episodes, states):
        self.path, self.episodes, self.states = path, episodes, states
        self.steps_per_episode = states.shape[1]

    def __len__(self):
        return len(self.episodes) * self.steps_per_episode

    def __getitem__(self, positions):
        positions = np.sort(positions)  # read chunks in file order
        episodes, steps = self.episodes[positions // self.steps_per_episode], positions % self.steps_per_episode
        # frames uint8 [B, H, W, 3], s_t [B, d_s], episode indices [B] into the scene posteriors
        return read_frames(self.path, episodes, steps), torch.from_numpy(self.states[episodes, steps]), torch.from_numpy(episodes)


class RolloutSequenceDataset(Dataset):
    """`length`-state windows with their actions, increments, and episode indices; an item is a whole batch of windows.

    `windows` holds (episode, start) rows. All arrays stay in host memory, so batches are gathered in the main process.
    """

    def __init__(self, windows, length, states, actions, deltas):
        self.episodes, self.starts = torch.from_numpy(windows).unbind(1)
        self.offsets = torch.arange(length)
        self.states, self.actions, self.deltas = torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(deltas)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episodes = self.episodes[index]
        rows, steps = episodes[:, None], self.starts[index][:, None] + self.offsets  # [B, 1], [B, T]
        # s_t [B, T, d_s], a_t [B, T, d_a], Delta s_t [B, T - 1, d_Delta], episode indices [B]
        return self.states[rows, steps], self.actions[rows, steps], self.deltas[rows, steps[:, :-1]], episodes
