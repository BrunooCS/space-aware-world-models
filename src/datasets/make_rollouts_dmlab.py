"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate a sharded DMLab rollout HDF5 dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Store RGB observations o_t, raw states/actions, rewards, done flags, and one
scene map m per episode. Workers write resumable shards before merging.
Raw states use [x, y, theta_deg]; the environment config constructs s_t and a_t.
Environment and policy randomness use base_seed + episode_id.

1) Split episode IDs across workers and generate rollouts.
2) Save each rollout as an HDF5 shard.
3) Merge completed shards into the configured dataset.

Run with: python -m src.datasets.make_rollouts_dmlab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path

import cv2
import h5py
import numpy as np
import pyrallis

from src.configs.datasets import DMLabRolloutDatasetConfig
from src.configs.dmlab import DMLabEnvConfig
from src.datasets.generation import generate_rollouts
from src.envs.dm_lab_maze import make_random_maze_env
from src.envs.dmlab_autonomous_exploration import CoordinateMapper, ExplorationManager, segment_from_numpy

ENV_CONFIG = DMLabEnvConfig()


# ── Fast Sweeping Policy ──────────────────────────────
class FastSweepingPolicy:
    """Geodesic exploration with distance-map updates on target changes or stalled motion."""

    def __init__(self, topdown_rgb, grid_size=ENV_CONFIG.default_topdown_grid_size, verbose=False, seed=None):
        """Create a geodesic explorer for the supplied maze map."""
        self.verbose = verbose
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

        self.mapper = CoordinateMapper(grid_size=grid_size)
        self.binary_map, _ = segment_from_numpy(topdown_rgb)

        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=26.0, stuck_window=40, stuck_threshold=3.0,
                                           verbose=verbose, rng=self.rng)

    def step(self, state):
        """Get action from state (x, y, yaw)."""
        world_pos = (float(state[0]), float(state[1]))
        world_rot = (0.0, float(state[2]), 0.0)
        action, _ = self.explorer.step(world_pos, world_rot)
        return action.astype(np.int16)

    def reset(self, topdown_rgb):
        """Reset for new maze."""
        self.binary_map, _ = segment_from_numpy(topdown_rgb)
        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=26.0, stuck_window=40, stuck_threshold=3.0,
                                           verbose=self.verbose, rng=self.rng)


# ── Slow Fast Sweeping Policy ─────────────────────────
class SlowFastSweepingPolicy:
    """Geodesic exploration with smoothed turning and reduced forward speed."""

    def __init__(self, topdown_rgb, grid_size=ENV_CONFIG.default_topdown_grid_size, verbose=False, seed=None):
        """Create a smoothed geodesic explorer for the supplied maze map."""
        self.verbose = verbose
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

        self.mapper = CoordinateMapper(grid_size=grid_size)
        self.binary_map, _ = segment_from_numpy(topdown_rgb)

        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=26.0, stuck_window=40, stuck_threshold=3.0,
                                           verbose=verbose, rng=self.rng)
        self.prev_turn = 0.0

    def step(self, state):
        """Get smoothed action from state (x, y, yaw)."""
        world_pos = (float(state[0]), float(state[1]))
        world_rot = (0.0, float(state[2]), 0.0)
        action, _ = self.explorer.step(world_pos, world_rot)

        # ── Smooth and limit the turn ───────────────────
        target_turn = float(action[ENV_CONFIG.action_turn_idx])
        self.prev_turn = 0.85 * self.prev_turn + 0.15 * target_turn
        current_turn = np.clip(self.prev_turn * 0.4, -15.0, 15.0)

        # ── Stop forward motion through sharp turns ─────
        if abs(current_turn) > 10.0:
            action[ENV_CONFIG.action_forward_idx] = 0
        action = action.astype(np.float32)
        action[ENV_CONFIG.action_turn_idx] = current_turn
        return action.astype(np.int16)

    def reset(self, topdown_rgb):
        """Reset for new maze."""
        self.binary_map, _ = segment_from_numpy(topdown_rgb)
        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=26.0, stuck_window=40, stuck_threshold=3.0,
                                           verbose=self.verbose, rng=self.rng)
        self.prev_turn = 0.0


# ── Balanced Exploration Policy ───────────────────────
class BalancedExplorationPolicy:
    """Geodesic navigation with slalom, heading drift, random actions, and in-place turns.

    Narrow corridors suppress look-around turns and reduce stochastic perturbations.
    """

    def __init__(self, topdown_rgb, grid_size=ENV_CONFIG.default_topdown_grid_size, verbose=False, seed=None):
        """Create a map explorer with seeded trajectory perturbations."""
        self.verbose = verbose
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.mapper = CoordinateMapper(grid_size=grid_size)

        self.binary_map, _ = segment_from_numpy(topdown_rgb)
        self.corridor_free_thresh = 0.35
        self.corridor_radius = 5

        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=28.0, stuck_window=40, stuck_threshold=3.0,
                                           verbose=verbose, rng=self.rng)

        # ── Slalom and heading drift ────────────────────
        self.step_count = 0
        self.sine_phase = self.rng.uniform(0, 2 * np.pi)
        self.sine_freq  = self.rng.uniform(0.01, 0.04)
        self.sine_amp   = self.rng.uniform(8.0, 18.0)

        self.drift_target = 0.0
        self.drift_current = 0.0
        self.drift_timer = self.rng.integers(80, 160)

        # ── Rotation and random-action modes ────────────
        self.look_mode_steps = 0
        self.look_direction = 0
        self.look_prob = 0.015

        self.random_mode_steps    = 0
        self.current_random_action = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.int16)

    def _local_free_ratio(self, world_pos):
        """Compute free space ratio in local patch (corridor detection)."""
        row, col = self.mapper.world_to_grid(world_pos[0], world_pos[1])
        r0 = max(0, row - self.corridor_radius)
        r1 = min(self.binary_map.shape[0], row + self.corridor_radius + 1)
        c0 = max(0, col - self.corridor_radius)
        c1 = min(self.binary_map.shape[1], col + self.corridor_radius + 1)
        patch = self.binary_map[r0:r1, c0:c1]
        return float(np.count_nonzero(patch)) / patch.size

    def step(self, state):
        """Return a balanced exploration action from the current DMLab state."""
        self.step_count += 1
        world_pos = (float(state[0]), float(state[1]))
        world_rot = (0.0, float(state[2]), 0.0)
        free_ratio = self._local_free_ratio(world_pos)
        in_corridor = free_ratio < self.corridor_free_thresh

        # ── Start look-around mode outside narrow corridors ─
        effective_look_prob = 0.0 if in_corridor else self.look_prob
        if self.look_mode_steps <= 0 and self.random_mode_steps <= 0 and self.rng.random() < effective_look_prob:
            self.look_mode_steps = self.rng.integers(10, 21)
            self.look_direction  = int(self.rng.choice([-1, 1]) * self.rng.integers(60, 91))
            self.explorer.position_buffer.clear()

        # ── Continue active look-around and random modes ───
        if self.look_mode_steps > 0:
            self.look_mode_steps -= 1
            look_action = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.int16)
            look_action[ENV_CONFIG.action_turn_idx] = self.look_direction
            return look_action

        random_prob = 0.008 if in_corridor else 0.03
        if self.random_mode_steps <= 0 and self.rng.random() < random_prob:
            self.random_mode_steps = self.rng.integers(10, 40)
            self.explorer.position_buffer.clear()

        if self.random_mode_steps > 0:
            self.random_mode_steps -= 1

            if self.rng.random() < 0.3:
                self.current_random_action = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.int16)
                if self.rng.random() < 0.8:
                    self.current_random_action[ENV_CONFIG.action_forward_idx] = 1
                self.current_random_action[ENV_CONFIG.action_turn_idx] = int(
                    self.rng.choice([-60, -30, 0, 30, 60]))

            return self.current_random_action

        # ── Follow the geodesic route with smooth drift ────
        action, _ = self.explorer.step(world_pos, world_rot)
        action = action.astype(np.float32)

        if action[ENV_CONFIG.action_forward_idx] > 0:
            sine_scale = 0.4 if in_corridor else 1.0
            sine_drift = np.sin(self.step_count * self.sine_freq + self.sine_phase) * self.sine_amp * sine_scale
            action[ENV_CONFIG.action_turn_idx] += sine_drift

            # Slowly varying heading offset.
            if self.drift_timer <= 0:
                self.drift_target = self.rng.uniform(-25.0, 25.0)
                self.drift_timer  = self.rng.integers(80, 160)
            else:
                self.drift_timer -= 1
            self.drift_current = 0.9 * self.drift_current + 0.1 * self.drift_target

            drift_scale = 0.5 if in_corridor else 1.0
            action[ENV_CONFIG.action_turn_idx] += self.drift_current * drift_scale

        if abs(action[ENV_CONFIG.action_turn_idx]) > 40:
            action[ENV_CONFIG.action_turn_idx] *= 0.8

        action[ENV_CONFIG.action_turn_idx] = np.clip(action[ENV_CONFIG.action_turn_idx],
                                                     ENV_CONFIG.turn_min, ENV_CONFIG.turn_max)
        return action.astype(np.int16)

    def reset(self, topdown_rgb):
        """Reset the map route and trajectory modes for a new maze."""
        self.binary_map, _ = segment_from_numpy(topdown_rgb)
        self.explorer = ExplorationManager(self.binary_map, self.mapper, reach_threshold=28.0, stuck_window=50, stuck_threshold=3.0,
                                           verbose=self.verbose, rng=self.rng)
        self.step_count = 0
        self.sine_phase = self.rng.uniform(0, 2 * np.pi)
        self.sine_freq  = self.rng.uniform(0.01, 0.04)
        self.sine_amp   = self.rng.uniform(8.0, 18.0)
        self.drift_target  = 0.0
        self.drift_current = 0.0
        self.drift_timer   = self.rng.integers(80, 160)
        self.look_mode_steps = 0
        self.random_mode_steps = 0


# ── Random Policy ─────────────────────────────────────
def random_policy(rng, num_steps, repeat_len):
    """Generate random action schedule with repeated segments."""
    num_segments = (num_steps + repeat_len - 1) // repeat_len
    actions = []

    for _ in range(num_segments):
        yaw = rng.choice([-17, 17]) if rng.random() < 0.65 else 0
        forward = 0 if yaw != 0 else 1

        act = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.int16)
        act[ENV_CONFIG.action_turn_idx] = int(yaw)
        act[ENV_CONFIG.action_forward_idx] = int(forward)
        actions.extend([act] * repeat_len)

    return np.stack(actions[:num_steps], axis=0)


# ── Episode Rollout ───────────────────────────────────
def rollout_to_episode_h5(env, episode_h5_path, episode_id, cfg):
    """Record one scene map m and its observation and raw state/action sequences."""
    seed = int(cfg.base_seed + episode_id)
    rng = np.random.default_rng(seed)
    height, width = cfg.img_height, cfg.img_width
    out_height, out_width = cfg.out_height, cfg.out_width
    td_height, td_width = cfg.topdown_height, cfg.topdown_width

    # 1) Environment reset
    state, obs, info = env.reset(seed=seed)
    rgb = np.asarray(obs, dtype=np.uint8)
    if (height, width) != (out_height, out_width):
        rgb = cv2.resize(rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)
    topdown_rgb = np.asarray(info["topdown_rgb"], dtype=np.uint8)

    # 2) Initialize policy
    if cfg.policy == "fast_sweeping":
        policy = FastSweepingPolicy(topdown_rgb, grid_size=td_height, verbose=False, seed=seed)
        precomputed_actions = None
    elif cfg.policy == "slow_fast_sweeping":
        policy = SlowFastSweepingPolicy(topdown_rgb, grid_size=td_height, verbose=False, seed=seed)
        precomputed_actions = None
    elif cfg.policy == "balanced":
        policy = BalancedExplorationPolicy(topdown_rgb, grid_size=td_height, verbose=False, seed=seed)
        precomputed_actions = None
    elif cfg.policy == "random":
        precomputed_actions = random_policy(rng, cfg.steps_per_episode, repeat_len=cfg.action_repeat)
        policy = None
    else:
        raise ValueError(f"Unknown DMLab policy: {cfg.policy}")

    # ── Create shard datasets ─────────────────────────
    episode_h5_path = Path(episode_h5_path)
    episode_h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(episode_h5_path, "w") as f:
        steps = cfg.steps_per_episode
        img_shape = (steps, out_height, out_width, 3)
        td_shape = (td_height, td_width, 3)
        ego_ds = f.create_dataset("ego", shape=img_shape, dtype=np.uint8, compression="gzip", compression_opts=4, chunks=(1, out_height, out_width, 3))
        state_ds = f.create_dataset("state", (steps, ENV_CONFIG.state_raw_dim), dtype=np.float32)
        reward_ds = f.create_dataset("reward", (steps,), dtype=np.float16)
        action_ds = f.create_dataset("action", (steps, ENV_CONFIG.action_raw_dim), dtype=np.int16)
        done_ds = f.create_dataset("done", (steps,), dtype=np.bool_)
        topdown_ds = f.create_dataset("topdown", shape=td_shape, dtype=np.uint8, compression="lzf", chunks=td_shape)

        f.attrs["episode_id"] = int(episode_id)
        f.attrs["image_shape"] = np.array([out_height, out_width, 3], dtype=np.int64)
        f.attrs["topdown_shape"] = np.array(td_shape, dtype=np.int64)
        f.attrs["policy_type"] = cfg.policy
        f.create_dataset("seed", data=np.array(seed, dtype=np.int64))
        topdown_ds[...] = topdown_rgb

        # ── Record observations and transitions ─────────
        for step in range(steps):
            ego_ds[step, ...] = rgb
            state_ds[step, :] = np.asarray(state, dtype=np.float32)

            action = policy.step(state) if policy is not None else precomputed_actions[step]
            action_ds[step, :] = action
            action_cmd = action.astype(np.intc)

            next_state, next_obs, reward, terminated, truncated, new_info = env.step(action_cmd)
            reward_ds[step] = np.float16(reward)
            done_flag = bool(terminated or truncated)
            done_ds[step] = done_flag

            if done_flag:
                state, obs, info = env.reset(seed=seed)
                topdown_rgb = np.asarray(info["topdown_rgb"], dtype=np.uint8)
                if policy is not None:
                    policy.reset(topdown_rgb)
                rgb = np.asarray(obs, dtype=np.uint8)
            else:
                state, obs = next_state, next_obs
                rgb = np.asarray(obs, dtype=np.uint8)

            if (height, width) != (out_height, out_width):
                rgb = cv2.resize(rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)

        f.attrs["completed"] = True


# ── Main ──────────────────────────────────────────────
def make_env(cfg):
    """One DMLab maze environment rendered at the configured size."""
    return make_random_maze_env(width=cfg.img_width, height=cfg.img_height, fps=cfg.fps, step_repeat=cfg.step_repeat,
                                topdown_width=cfg.topdown_width, topdown_height=cfg.topdown_height)


def main():
    """Generate missing rollouts and merge their shards."""
    cfg = pyrallis.parse(config_class=DMLabRolloutDatasetConfig)
    attrs = {"image_shape": [cfg.out_height, cfg.out_width, 3], "topdown_shape": [cfg.topdown_height, cfg.topdown_width, 3]}
    generate_rollouts(cfg, make_env, rollout_to_episode_h5, attrs)


if __name__ == "__main__":
    main()
