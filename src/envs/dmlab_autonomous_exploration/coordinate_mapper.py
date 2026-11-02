"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DMLab world and top-down grid coordinate mapping
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Map world (x, y) coordinates to image (row, column) indices and back. The
image row axis is inverted relative to DMLab's positive Y axis.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from src.configs.dmlab import DMLabEnvConfig

ENV_CONFIG = DMLabEnvConfig()
WORLD_MIN = ENV_CONFIG.world_min
WORLD_MAX = ENV_CONFIG.world_max
DEFAULT_GRID_SIZE = ENV_CONFIG.default_topdown_grid_size

# ── Transform / Core Logic ────────────────────────────
class CoordinateMapper:
    """Map between DMLab world coordinates and a square image grid."""

    def __init__(self, world_min=WORLD_MIN, world_max=WORLD_MAX, grid_size=DEFAULT_GRID_SIZE):
        """Configure the world-coordinate range and grid resolution."""
        self.config = DMLabEnvConfig(world_min=world_min, world_max=world_max,
                                     default_topdown_grid_size=grid_size)
        self.world_min = world_min
        self.world_max = world_max
        self.grid_size = grid_size
        self.scale = grid_size / (world_max - world_min)

    def world_to_grid(self, world_x, world_y):
        """Return clamped (row, column) indices for a world position."""
        pixel_x, pixel_y = self.config.world_to_pixel(world_x, world_y, self.grid_size)
        column, row = int(pixel_x), int(pixel_y)
        return (max(0, min(self.grid_size - 1, row)),
                max(0, min(self.grid_size - 1, column)))

    def grid_to_world(self, row, column):
        """Return the world (x, y) position at a grid cell."""
        world_x = column / self.scale + self.world_min
        world_y = self.world_min + (self.grid_size - row) / self.scale
        return world_x, world_y

    def world_to_grid_float(self, world_x, world_y):
        """Return fractional (row, column) coordinates for path interpolation."""
        pixel_x, pixel_y = self.config.world_to_pixel(world_x, world_y, self.grid_size)
        return pixel_y, pixel_x
