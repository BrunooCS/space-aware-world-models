"""Convert a geodesic distance map into DMLab turn and forward actions."""

import math

import numpy as np

from src.configs.dmlab import DMLabEnvConfig

ENV_CONFIG = DMLabEnvConfig()

# ── Model / Control ───────────────────────────────────
class AgentController:
    """Follow a geodesic distance gradient with DMLab turn and move actions."""

    def __init__(self, dist_map, mapper, turn_speed=ENV_CONFIG.turn_speed, move_speed=1,
                 search_radius=5, rng=None):
        """Configure the map, coordinate transform, and steering limits."""
        self.dist_map = dist_map
        self.mapper = mapper
        self.turn_speed = turn_speed
        self.move_speed = move_speed
        self.search_radius = search_radius
        self.rng = np.random.default_rng() if rng is None else rng

    def get_action(self, world_pos, world_rot):
        """Return a seven-value action and the local world-space target."""
        # ── Find the lowest-distance grid cell nearby ──
        row, column = self.mapper.world_to_grid(world_pos[0], world_pos[1])
        height, width = self.dist_map.shape
        if self.dist_map[row, column] > 1e8:
            action = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.intc)
            action[ENV_CONFIG.action_turn_idx] = self.rng.choice([-60, 60])
            action[ENV_CONFIG.action_forward_idx] = 1
            return action, world_pos

        target_distance = np.inf
        target_row, target_column = row, column
        for row_offset in range(-self.search_radius, self.search_radius + 1):
            for column_offset in range(-self.search_radius, self.search_radius + 1):
                candidate_row, candidate_column = row + row_offset, column + column_offset
                if 0 <= candidate_row < height and 0 <= candidate_column < width:
                    distance = self.dist_map[candidate_row, candidate_column]
                    if distance < target_distance and distance < 1e8:
                        target_distance = distance
                        target_row, target_column = candidate_row, candidate_column

        if target_distance == np.inf:
            return np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.intc), world_pos

        # ── Convert the target direction to a steering action ──
        target_x, target_y = self.mapper.grid_to_world(target_row, target_column)
        desired_yaw = math.degrees(math.atan2(target_y - world_pos[1], target_x - world_pos[0]))
        yaw_difference = (desired_yaw - world_rot[1] + 180) % 360 - 180
        action = np.zeros(ENV_CONFIG.action_raw_dim, dtype=np.intc)

        if abs(yaw_difference) > 15:
            action[ENV_CONFIG.action_turn_idx] = -self.turn_speed if yaw_difference > 0 else self.turn_speed
        else:
            action[ENV_CONFIG.action_forward_idx] = self.move_speed
            if abs(yaw_difference) > 2:
                action[ENV_CONFIG.action_turn_idx] = -5 if yaw_difference > 0 else 5
        return action, (target_x, target_y)

    def update_distance_map(self, dist_map):
        """Replace the map after the exploration target changes."""
        self.dist_map = dist_map
