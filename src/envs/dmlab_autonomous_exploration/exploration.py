"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-waypoint DMLab exploration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cycle through reachable maze targets, recompute geodesic routes when targets
change, and switch targets when progress stalls.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .agent_controller import AgentController
from .coordinate_mapper import CoordinateMapper
from .distance_map import compute_distance_map
from .segmentation import find_nearest_walkable

# ── Exploration Targets / State ───────────────────────
@dataclass
class ExplorationStats:
    """Track steps, target visits, distance, and stuck events."""

    total_steps: int = 0
    targets_reached: int = 0
    total_distance: float = 0.0
    stuck_count: int = 0
    path_history: list = field(default_factory=list)
    visited_targets: list = field(default_factory=list)


def get_exploration_targets(binary_map, margin=15):
    """Find walkable cells near the center and four inset map corners."""
    height, width = binary_map.shape
    target_points = (
        (height // 2, width // 2),
        (margin, width - margin),
        (height - margin, margin),
        (margin, margin),
        (height - margin, width - margin),
    )
    targets = [find_nearest_walkable(binary_map, point) for point in target_points]
    return [target for target in targets if target is not None]


# ── Exploration Manager ───────────────────────────────
class ExplorationManager:
    """Navigate through map targets while detecting stalled motion."""

    def __init__(self, binary_map, mapper, targets=None, reach_threshold=15.0,
                 stuck_window=35, stuck_threshold=5.0, verbose=True, rng=None):
        """Set the walkable map, target sequence, and stuck thresholds."""
        self.binary_map = binary_map
        self.mapper = mapper
        self.reach_threshold = reach_threshold
        self.stuck_window = stuck_window
        self.stuck_threshold = stuck_threshold
        self.verbose = verbose
        self.rng = np.random.default_rng() if rng is None else rng
        self.stats = ExplorationStats()
        self.position_buffer = deque(maxlen=stuck_window)
        self.base_targets = get_exploration_targets(binary_map) if targets is None else targets
        self.targets = list(self.base_targets)
        self.target_idx = 0
        self.cycle_count = 0
        self.current_target = self.targets[self.target_idx]
        self.dist_map, self.obstacle_mask = compute_distance_map(binary_map, self.current_target)
        self.controller = AgentController(self.dist_map, mapper, rng=self.rng)

        if self.verbose:
            print(f"[INFO] Exploration targets: {self.targets}")

    def _check_stuck(self, world_pos):
        """Return whether the position window shows too little displacement."""
        self.position_buffer.append(world_pos)
        if len(self.position_buffer) < self.stuck_window:
            return False
        displacement = np.linalg.norm(np.asarray(self.position_buffer[-1]) - self.position_buffer[0])
        return displacement < self.stuck_threshold

    def _advance_target(self):
        """Move to the next target and shuffle after each complete cycle."""
        self.target_idx += 1
        if self.target_idx == len(self.targets):
            self.target_idx = 0
            self.cycle_count += 1
            self.rng.shuffle(self.targets)
            if self.verbose:
                print(f"[INFO] Completed exploration cycle {self.cycle_count}")
        self.current_target = self.targets[self.target_idx]

    def _replan(self):
        """Recompute the geodesic field and update the controller."""
        self.dist_map, self.obstacle_mask = compute_distance_map(self.binary_map, self.current_target)
        self.controller.update_distance_map(self.dist_map)

    def _force_waypoint_change(self):
        """Advance the target and reset the stuck window after stalled motion."""
        self.stats.stuck_count += 1
        self._advance_target()
        self._replan()
        self.position_buffer.clear()
        if self.verbose:
            print(f"[INFO] Stuck; switched to exploration target {self.target_idx}")

    def step(self, world_pos, world_rot):
        """Return a DMLab action and whether its exploration target changed."""
        target_changed = False

        # 1) Recover by switching targets if the position window is stalled.
        if self._check_stuck(world_pos):
            self._force_waypoint_change()
            target_changed = True

        # 2) Advance after reaching the current target.
        row, column = self.mapper.world_to_grid(world_pos[0], world_pos[1])
        if self.dist_map[row, column] < self.reach_threshold:
            self.stats.visited_targets.append(self.current_target)
            self.stats.targets_reached += 1
            self._advance_target()
            self._replan()
            self.position_buffer.clear()
            target_changed = True

        # 3) Control motion and update trajectory statistics.
        action, _ = self.controller.get_action(world_pos, world_rot)
        if self.stats.path_history:
            self.stats.total_distance += np.linalg.norm(np.asarray(world_pos) - self.stats.path_history[-1])
        self.stats.total_steps += 1
        self.stats.path_history.append(world_pos)
        return action, target_changed

    def get_current_distance(self, world_pos):
        """Return the geodesic distance from a world position to the target."""
        row, column = self.mapper.world_to_grid(world_pos[0], world_pos[1])
        return self.dist_map[row, column]

    def get_grid_path_history(self):
        """Convert recorded world positions to grid coordinates."""
        return [self.mapper.world_to_grid(*position) for position in self.stats.path_history]
