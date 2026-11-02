"""DMLab maze segmentation, geodesic routing, and exploration utilities."""

from .segmentation import denormalize, find_nearest_walkable, segment_from_numpy, segment_image
from .coordinate_mapper import CoordinateMapper
from .distance_map import compute_distance_map, compute_path_smooth, get_border_start_points, replay_sweeps
from .agent_controller import AgentController
from .exploration import ExplorationManager, ExplorationStats, get_exploration_targets

__all__ = [
    "denormalize",
    "segment_image",
    "segment_from_numpy",
    "find_nearest_walkable",
    "CoordinateMapper",
    "compute_distance_map",
    "compute_path_smooth",
    "get_border_start_points",
    "replay_sweeps",
    "AgentController",
    "ExplorationStats",
    "ExplorationManager",
    "get_exploration_targets",
]
