"""
core/occupancy_grid.py
═══════════════════════════════════════════════════════════════════
TASK 1 — 2D Occupancy Grid

Represents the robot's probabilistic map of the world as a 2-D
integer matrix stored in a numpy int16 array for speed.

Cell values (matching PDF diagram):
  GRID_UNKNOWN  (-1)  : Not yet observed by LiDAR
  GRID_FREE     ( 0)  : Confirmed passable space
  GRID_OBSTACLE (100) : Confirmed wall / barrier

Architecture (from PDF slide "Translating Space into Discrete Mathematics"):
  Matrix tensor: [3, rows, cols]
    layer 0 → occupancy  (-1 / 0 / 100)
    layer 1 → hit  count (obstacle confirmations)
    layer 2 → miss count (free-space confirmations)

Coordinate system:
  World origin (0,0) → bottom-left of grid
  world (x, y) metres → grid (col, row)
  Grid row 0 is the TOP of the image (y = height_m)
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Tuple
from config.settings import Settings


class OccupancyGrid:
    """
    2-D occupancy grid with vectorised read/write helpers.

    Parameters
    ----------
    width_m  : physical width  of the environment in metres
    height_m : physical height of the environment in metres
    resolution : metres per cell (e.g. 0.1 → each cell = 10 cm²)
    """

    def __init__(self, width_m: float, height_m: float, resolution: float):
        self.resolution = float(resolution)
        self.width_m    = float(width_m)
        self.height_m   = float(height_m)

        self.cols = max(1, int(np.ceil(width_m  / resolution)))
        self.rows = max(1, int(np.ceil(height_m / resolution)))

        # ── 3-layer tensor ────────────────────────────────────────────────────
        self._tensor = np.full((3, self.rows, self.cols), fill_value=-1, dtype=np.int16)
        self._tensor[1] = 0   # hit  counts
        self._tensor[2] = 0   # miss counts

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        """Occupancy layer as (rows × cols) int16 array."""
        return self._tensor[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    # ── Coordinate Conversion ─────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world metres (x, y) → grid (row, col).
        Row 0 = top of grid (y = height_m).
        """
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        col = int(np.clip(col, 0, self.cols - 1))
        row = int(np.clip(row, 0, self.rows - 1))
        return (row, col)

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Return world-coordinate centre of grid cell (row, col)."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return (x, y)

    def world_to_cell_batch(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised conversion for arrays of world coords."""
        cols = np.clip((xs / self.resolution).astype(int), 0, self.cols - 1)
        rows = np.clip((ys / self.resolution).astype(int), 0, self.rows - 1)
        return rows, cols

    # ── Cell Updates ──────────────────────────────────────────────────────────

    def mark_obstacle(self, row: int, col: int) -> None:
        """Mark a cell as confirmed obstacle."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._tensor[0, row, col] = 100
            self._tensor[1, row, col] = min(self._tensor[1, row, col] + 1, 32767)

    def mark_free(self, row: int, col: int) -> None:
        """Mark a cell as confirmed free (cannot override confirmed obstacle)."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self._tensor[0, row, col] != 100:
                self._tensor[0, row, col] = 0
                self._tensor[2, row, col] = min(self._tensor[2, row, col] + 1, 32767)

    def mark_free_batch(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Vectorised free-cell marking for a ray of cells."""
        valid = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        r, c = rows[valid], cols[valid]
        not_obs = self._tensor[0, r, c] != 100
        self._tensor[0, r[not_obs], c[not_obs]] = 0

    def seed_from_world(self, obstacle_bool_map: np.ndarray) -> None:
        """
        Pre-populate from a ground-truth boolean map (True = obstacle).
        Used to give the planner an initial known map.
        """
        r_max = min(obstacle_bool_map.shape[0], self.rows)
        c_max = min(obstacle_bool_map.shape[1], self.cols)
        self._tensor[0, :r_max, :c_max] = np.where(
            obstacle_bool_map[:r_max, :c_max], 100, 0
        ).astype(np.int16)

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_free(self, row: int, col: int) -> bool:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(self._tensor[0, row, col]) == 0
        return False

    def is_obstacle(self, row: int, col: int) -> bool:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(self._tensor[0, row, col]) == 100
        return True   # out-of-bounds treated as obstacle

    def value(self, row: int, col: int) -> int:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(self._tensor[0, row, col])
        return 100

    def coverage_percent(self) -> float:
        """Percentage of cells that have been observed (not UNKNOWN)."""
        known = int(np.sum(self._tensor[0] != -1))
        return 100.0 * known / (self.rows * self.cols)

    def copy_data(self) -> np.ndarray:
        return self._tensor[0].copy()

    def __repr__(self) -> str:
        return (
            f"OccupancyGrid({self.rows}×{self.cols} cells, "
            f"res={self.resolution}m, "
            f"coverage={self.coverage_percent():.1f}%)"
        )
