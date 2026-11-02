"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Geodesic distance maps with Fast Marching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Alternative distance-map implementation using scikit-fmm. Paths avoid walls
by increasing travel cost near obstacles.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, map_coordinates

# ── Distance Map / Fast Marching ──────────────────────
def compute_distance_map(binary_map: np.ndarray, target_point: tuple[int, int],
                         wall_penalty: float = 5.0, penalty_radius: int = 8):
    """Return geodesic distance [H, W] and a boolean wall mask."""
    obstacle_mask = binary_map == 0

    # ── Initialize the target and wall-aware speed field ──
    level_set = np.ones(binary_map.shape, dtype=np.float64)
    level_set[target_point] = -1
    masked_level_set = np.ma.MaskedArray(level_set, mask=obstacle_mask)
    wall_distance = distance_transform_edt(binary_map)
    proximity = np.exp(-wall_distance / penalty_radius)
    speed = 1.0 / (1.0 + wall_penalty * proximity)
    speed[obstacle_mask] = 0.0

    # ── Propagate travel time from the target ───────────
    distance = skfmm.travel_time(masked_level_set, speed, dx=1.0)
    return np.asarray(distance.filled(np.inf), dtype=np.float32), obstacle_mask


# ── Path Extraction ───────────────────────────────────
def compute_path_smooth(start_pos: tuple[int, int], dist_map: np.ndarray,
                        step_size: float = 1.0, max_steps: int = 2000):
    """Trace a smooth descending path from a grid cell to the target."""
    path = [start_pos]
    position = np.array(start_pos, dtype=np.float64)
    height, width = dist_map.shape
    finite_dist = np.where(np.isinf(dist_map), 1e6, dist_map)
    grad_row, grad_column = np.gradient(finite_dist)

    for _ in range(max_steps):
        row, column = map(int, position)
        if not (0 <= row < height and 0 <= column < width) or np.isinf(dist_map[row, column]):
            break
        if dist_map[row, column] < 1.5:
            path.append(tuple(position))
            break

        point = position.reshape(2, 1)
        row_gradient = map_coordinates(grad_row, point, order=1, mode="nearest")[0]
        column_gradient = map_coordinates(grad_column, point, order=1, mode="nearest")[0]
        norm = np.hypot(row_gradient, column_gradient)
        if norm < 1e-6:
            break

        position[0] -= row_gradient / norm * step_size
        position[1] -= column_gradient / norm * step_size
        position[0] = np.clip(position[0], 0, height - 1)
        position[1] = np.clip(position[1], 0, width - 1)
        path.append(tuple(position))
    return np.asarray(path)


def get_border_start_points(binary_map: np.ndarray, step: int = 5, max_scan_dist: int = 30):
    """Find unique walkable cells by scanning inward from each map border."""
    height, width = binary_map.shape
    valid_starts = []

    def scan_inward(row, column, row_step, column_step):
        for offset in range(max_scan_dist):
            candidate = row + row_step * offset, column + column_step * offset
            r, c = candidate
            if 0 <= r < height and 0 <= c < width and binary_map[r, c] > 0:
                return candidate
        return None

    for column in range(0, width, step):
        top = scan_inward(0, column, 1, 0)
        bottom = scan_inward(height - 1, column, -1, 0)
        if top is not None:
            valid_starts.append(top)
        if bottom is not None:
            valid_starts.append(bottom)
    for row in range(0, height, step):
        left = scan_inward(row, 0, 0, 1)
        right = scan_inward(row, width - 1, 0, -1)
        if left is not None:
            valid_starts.append(left)
        if right is not None:
            valid_starts.append(right)
    return list(dict.fromkeys(valid_starts))
