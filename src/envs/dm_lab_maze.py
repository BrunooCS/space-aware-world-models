"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DeepMind Lab random maze wrapper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expose RGB observations o_t, scene maps m, and raw [x, y, theta_deg] state.
Reset accepts a maze seed. A terminal step resets the environment and returns
the next maze's initial observation. `DMLabSimulator`
serves episodes in the rollout dataset's conventions to interactive tools.

Requires importable deepmind_lab; runfiles_path optionally selects a custom
build's runtime assets.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import deepmind_lab
import numpy as np
from PIL import Image

from src.configs.datasets import DMLabRolloutDatasetConfig
from src.configs.dmlab import DMLabEnvConfig

ENV_CONFIG = DMLabEnvConfig()

OBSERVATIONS = (
    "RGB_INTERLEAVED",
    "DEBUG.CAMERA_INTERLEAVED.TOP_DOWN",
    "DEBUG.POS.TRANS",
    "DEBUG.POS.ROT",
    "VEL.TRANS",
)

# ── Config ───────────────────────────────────────────
@dataclass
class MazeConfig:
    """Maze geometry, rendering, and optional runtime asset location."""

    width: int = ENV_CONFIG.default_render_width
    height: int = ENV_CONFIG.default_render_height
    fps: int = 30
    step_repeat: int = 1
    maze_height: int = 13
    maze_width: int = 13
    maze_max_rooms: int = 4
    room_min_size: int = 3
    room_max_size: int = 5
    room_spawn_count: int = 3
    pickup_count: int = 4
    weapon_count: int = 2
    topdown_width: int = ENV_CONFIG.default_topdown_img_width
    topdown_height: int = ENV_CONFIG.default_topdown_img_height
    runfiles_path: str | None = None
    interactive: bool = False  # enable map placement in simulator viewers

    def to_dm_config(self):
        """Return DMLab's string-valued engine and maze settings."""
        values = {
            "width": self.width, "height": self.height, "fps": self.fps,
            "mazeGenerationParams.height": self.maze_height,
            "mazeGenerationParams.width": self.maze_width,
            "mazeGenerationParams.maxRooms": self.maze_max_rooms,
            "mazeGenerationParams.roomMinSize": self.room_min_size,
            "mazeGenerationParams.roomMaxSize": self.room_max_size,
            "mazeGenerationParams.roomSpawnCount": self.room_spawn_count,
            "pickupParams.pickupCount": self.pickup_count,
            "pickupParams.weaponCount": self.weapon_count,
        }
        return {key: str(value) for key, value in values.items()}

# ── Model ────────────────────────────────────────────
class DMLabRandomMazeEnv:
    """DMLab maze with explicit reset, cached top-down views, and auto-reset."""

    def __init__(self, config=None):
        """Create the native environment; call reset before stepping."""
        self._cfg = MazeConfig() if config is None else config
        if self._cfg.runfiles_path is not None:
            deepmind_lab.set_runfiles_path(self._cfg.runfiles_path)
        level, dm_config = "contributed/dmlab30/random_maze_safe", self._cfg.to_dm_config()
        if self._cfg.interactive:
            level = "interactive_maze"
            dm_config["levelDirectory"] = str(Path(__file__).with_name("dmlab_levels"))
        self._env = deepmind_lab.Lab(level, list(OBSERVATIONS), config=dm_config)

    def _read_observation(self, refresh_topdown):
        """Read required observations and refresh the map when a maze changes."""
        observations = self._env.observations()
        pos, rot = observations["DEBUG.POS.TRANS"], observations["DEBUG.POS.ROT"]
        state = np.asarray([pos[0], pos[1], rot[1]], dtype=np.float32)  # [3]: x, y, theta_deg
        rgb = np.asarray(observations["RGB_INTERLEAVED"], dtype=np.uint8)  # [H, W, 3]
        if refresh_topdown:
            self._cached_topdown = self._resize_topdown(observations["DEBUG.CAMERA_INTERLEAVED.TOP_DOWN"])
        info = {"topdown_rgb": self._cached_topdown, "raw_observations": observations}
        return state, rgb, info

    def reset(self, seed=None):
        """Return raw state [3], observation o_t [H, W, 3], and scene-map info."""
        self._env.reset(seed=seed)
        return self._read_observation(refresh_topdown=True)

    def step(self, action):
        """Apply a seven-integer action, automatically resetting at episode end."""
        reward = self._env.step(np.asarray(action, dtype=np.intc), num_steps=self._cfg.step_repeat)
        terminated = not self._env.is_running()
        if terminated:
            self._env.reset()
        state, rgb, info = self._read_observation(refresh_topdown=terminated)
        return state, rgb, float(reward), terminated, False, info

    def close(self):
        """Release the native environment resources."""
        self._env.close()

    @property
    def config(self):
        """Return the environment settings."""
        return self._cfg

    def _resize_topdown(self, frame):
        """Resize an RGB frame with nearest-neighbor interpolation."""
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))  # [H, W, 3]
        return np.asarray(image.resize((self._cfg.topdown_width, self._cfg.topdown_height),
                                       resample=Image.Resampling.NEAREST))

# ── Inputs / Data ────────────────────────────────────
def make_random_maze_env(width=ENV_CONFIG.default_render_width, height=ENV_CONFIG.default_render_height,
                         fps=30, step_repeat=1,
                         topdown_width=ENV_CONFIG.default_topdown_img_width,
                         topdown_height=ENV_CONFIG.default_topdown_img_height, runfiles_path=None, interactive=False):
    """Create a maze wrapper with configured RGB and scene-map dimensions."""
    cfg = MazeConfig(width=width, height=height, fps=fps, step_repeat=step_repeat,
                     topdown_width=topdown_width, topdown_height=topdown_height, runfiles_path=runfiles_path, interactive=interactive)
    return DMLabRandomMazeEnv(cfg)

# ── Simulator ────────────────────────────────────────
class DMLabSimulator:
    """Maze episodes in the rollout dataset's conventions: raw state [x, y, theta_deg], o_t [96, 96, 3], and m [128, 128, 3]."""

    config = ENV_CONFIG

    def __init__(self):
        rollouts = DMLabRolloutDatasetConfig()  # render size, frame rate, and step repeat of the recorded episodes
        self.frame_size = (rollouts.out_width, rollouts.out_height)
        self.env = make_random_maze_env(width=rollouts.img_width, height=rollouts.img_height, fps=rollouts.fps, step_repeat=rollouts.step_repeat,
                                        topdown_width=rollouts.topdown_width, topdown_height=rollouts.topdown_height, interactive=True)

    def observe(self, rgb):
        return cv2.resize(rgb, self.frame_size, interpolation=cv2.INTER_AREA)  # rendered [128, 128, 3] -> recorded [96, 96, 3]

    def reset(self, seed):
        """Load maze `seed`; return the raw state, observation o_t, and scene map m."""
        state, rgb, info = self.env.reset(seed=seed)
        return state, self.observe(rgb), info["topdown_rgb"]

    def step(self, action):
        """Apply a seven-integer action; return the raw state, observation o_t, and whether the maze episode ended."""
        state, rgb, _, done, _, _ = self.env.step(action)
        return state, self.observe(rgb), done

    def map_pose(self, state):
        """Agent pixel position (px, py) on the scene map and heading in radians, counterclockwise from the map's +x axis."""
        px, py = self.config.world_to_pixel(state[0], state[1], self.env.config.topdown_width)
        return px, py, self.config.map_heading(state)

    def place(self, px, py):
        """Teleport to a scene-map pixel, preserving yaw; return the raw state, observation, and episode-end flag."""
        span = self.config.world_max - self.config.world_min
        x = self.config.world_min + px / self.env.config.topdown_width * span
        y = self.config.world_max - py / self.env.config.topdown_height * span
        observations = self.env._env.observations()
        z, yaw = observations["DEBUG.POS.TRANS"][2], observations["DEBUG.POS.ROT"][1]
        self.env._env.write_property("teleport", f"{x} {y} {z} {yaw}")
        return self.step(np.zeros(7, dtype=np.intc))  # the engine processes queued console commands on a frame advance

    def close(self):
        self.env.close()
