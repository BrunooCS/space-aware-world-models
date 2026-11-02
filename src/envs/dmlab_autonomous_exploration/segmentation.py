"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Segment walkable DMLab maze regions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify RGB pixels by their distance from the maze wall color, crop map
borders, and erode the walkable mask away from walls.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import deque

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

# ── Segmentation ──────────────────────────────────────
def denormalize(tensor):
    """Convert a tensor from [-1, 1] to [0, 1]."""
    return tensor * 0.5 + 0.5


def _segment_rgb(image, crop_pixels, bg_color_hex, tolerance, wall_margin):
    """Create a binary walkable mask from an RGB uint8 image."""
    image = np.asarray(image, dtype=np.uint8).copy()
    color_hex = bg_color_hex.removeprefix("#")
    background_rgb = np.array([int(color_hex[i:i + 2], 16) for i in (0, 2, 4)])
    background_distance = np.linalg.norm(image - background_rgb, axis=2)
    is_background = background_distance < tolerance

    # Map boundaries are outside the walkable area.
    if crop_pixels > 0:
        is_background[:crop_pixels, :] = True
        is_background[-crop_pixels:, :] = True
        is_background[:, :crop_pixels] = True
        is_background[:, -crop_pixels:] = True

    binary_map = (~is_background).astype(np.uint8)
    if wall_margin > 0:
        binary_map = (distance_transform_edt(binary_map) >= wall_margin).astype(np.uint8)
    return binary_map, image


def segment_image(image_tensor, crop_pixels=5, bg_color_hex="#57a5ff", tolerance=10, wall_margin=3):
    """Return a walkable mask and RGB image from a normalized [C, H, W] tensor."""
    image = denormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return _segment_rgb(image, crop_pixels, bg_color_hex, tolerance, wall_margin)


def segment_from_numpy(image, crop_pixels=5, bg_color_hex="#57a5ff", tolerance=10, wall_margin=3):
    """Return a walkable mask and RGB image from an [H, W, 3] uint8 array."""
    return _segment_rgb(np.asarray(image, dtype=np.uint8), crop_pixels, bg_color_hex, tolerance, wall_margin)


# ── Walkable Cell Search ──────────────────────────────
def find_nearest_walkable(binary_map, start_point):
    """Find the nearest walkable (row, column) cell with a breadth-first search."""
    height, width = binary_map.shape
    row, column = start_point
    if 0 <= row < height and 0 <= column < width and binary_map[row, column] > 0:
        return row, column

    visited = np.zeros_like(binary_map, dtype=bool)
    visited[row, column] = True
    queue = deque([(row, column)])
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while queue:
        current_row, current_column = queue.popleft()
        if binary_map[current_row, current_column] > 0:
            return current_row, current_column
        for row_step, column_step in directions:
            next_row = current_row + row_step
            next_column = current_column + column_step
            if (0 <= next_row < height and 0 <= next_column < width
                    and not visited[next_row, next_column]):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return None
