"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Geodesic distance maps with the Fast Sweeping Method
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Solve the eikonal equation over a binary maze map, penalizing routes near
walls. Numba compiles the grid sweeps used by repeated target updates, and
`replay_sweeps` re-runs them in row blocks to visualize the propagation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt, map_coordinates

# ── Distance Map / Fast Sweeping ──────────────────────
@njit(cache=True)
def _solve_eikonal_2d(a, b, speed):
    """Solve one local Godunov update for the eikonal equation."""
    if speed <= 0:
        return np.inf

    cost = 1.0 / speed
    if abs(a - b) >= cost:
        return min(a, b) + cost

    discriminant = 2.0 * cost * cost - (a - b) ** 2
    if discriminant < 0:
        return min(a, b) + cost
    return (a + b + np.sqrt(discriminant)) / 2.0


@njit(cache=True)
def _sweep(distance, speed, row_range, column_range):
    """Update one directional pass over the [H, W] grid."""
    row_start, row_stop, row_step = row_range
    column_start, column_stop, column_step = column_range
    height, width = distance.shape

    for row in range(row_start, row_stop, row_step):
        for column in range(column_start, column_stop, column_step):
            if speed[row, column] <= 0:
                continue
            vertical = min(distance[row - 1, column] if row > 0 else np.inf,
                           distance[row + 1, column] if row < height - 1 else np.inf)
            horizontal = min(distance[row, column - 1] if column > 0 else np.inf,
                             distance[row, column + 1] if column < width - 1 else np.inf)
            candidate = _solve_eikonal_2d(vertical, horizontal, speed[row, column])
            if candidate < distance[row, column]:
                distance[row, column] = candidate


@njit(cache=True)
def _fast_sweeping_2d(distance, speed, max_iter=4):
    """Sweep the grid in four directions until the configured pass limit."""
    height, width = distance.shape
    for _ in range(max_iter):
        _sweep(distance, speed, (0, height, 1), (0, width, 1))
        _sweep(distance, speed, (0, height, 1), (width - 1, -1, -1))
        _sweep(distance, speed, (height - 1, -1, -1), (0, width, 1))
        _sweep(distance, speed, (height - 1, -1, -1), (width - 1, -1, -1))


def _initial_fields(binary_map, target_point, wall_penalty, penalty_radius):
    """Return the travel time (zero at the target, infinite elsewhere) and the wall-aware speed, both [H, W]."""
    distance = np.full(binary_map.shape, np.inf, dtype=np.float64)
    distance[target_point] = 0.0
    wall_distance = distance_transform_edt(binary_map)
    proximity = np.exp(-wall_distance / penalty_radius)
    speed = 1.0 / (1.0 + wall_penalty * proximity)
    speed[binary_map == 0] = 0.0
    return distance, speed


def compute_distance_map(binary_map, target_point, wall_penalty=5.0, penalty_radius=8, max_iter=4):
    """Return geodesic distance [H, W] and a boolean obstacle mask."""
    obstacle_mask = binary_map == 0
    distance, speed = _initial_fields(binary_map, target_point, wall_penalty, penalty_radius)

    # ── Propagate travel time from the target ───────────
    _fast_sweeping_2d(distance, speed, max_iter)
    distance[obstacle_mask] = np.inf
    return distance.astype(np.float32), obstacle_mask


def replay_sweeps(binary_map, target_point, rows_per_step, wall_penalty=5.0, penalty_radius=8, max_iter=4):
    """Replay the solver, yielding after each `rows_per_step` row block.

    Yields (distance, iteration, (row_step, column_step), progress). The distance
    grid [H, W] updates in place and is infinite ahead of the front; progress is
    the fraction of all blocks completed.
    """
    distance, speed = _initial_fields(binary_map, target_point, wall_penalty, penalty_radius)
    height, width = distance.shape
    total, done = max_iter * 4 * -(-height // rows_per_step), 0
    for iteration in range(max_iter):
        for row_step, column_step in ((1, 1), (1, -1), (-1, 1), (-1, -1)):  # order of `_fast_sweeping_2d`
            rows = range(height) if row_step > 0 else range(height - 1, -1, -1)
            columns = (0, width, 1) if column_step > 0 else (width - 1, -1, -1)
            for start in range(0, height, rows_per_step):
                block = rows[start:start + rows_per_step]
                _sweep(distance, speed, (block[0], block[-1] + row_step, row_step), columns)
                done += 1
                yield distance, iteration, (row_step, column_step), done / total


# ── Path Extraction ───────────────────────────────────
def compute_path_smooth(start_pos, dist_map, step_size=1.0, max_steps=2000):
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


def get_border_start_points(binary_map, step=5, max_scan_dist=30):
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
