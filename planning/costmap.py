"""
planning/costmap.py
═══════════════════════════════════════════════════════════════════
CostMap — Inflation Layer

Implements the "Inflated Costmap" described in PDF slide
"The Margin Constraint: Reconciling Math with Physical Reality".

The Inflation Layer artificially increases the traversal cost of
cells adjacent to obstacles, so A* automatically routes through
the safe centre of corridors rather than scraping walls.

Implementation: scipy.ndimage.distance_transform_edt
  • Euclidean Distance Transform — extremely fast even on large grids.
  • Cost decays linearly from INFLATION_COST_MAX at the obstacle
    boundary to 0 at INFLATION_RADIUS metres away.

Dual-costmap architecture (PDF slide "Global vs Local Costmaps"):
  global_costmap  → used by A* for long-range planning
  local_costmap   → rolling window, updated every step from live scan
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from core.occupancy_grid import OccupancyGrid
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class CostMap:
    """Global costmap with inflation layer."""

    def __init__(self, grid: OccupancyGrid, cfg: Settings):
        self.grid   = grid
        self.cfg    = cfg
        self._cost  = np.zeros(grid.shape, dtype=np.float32)
        self._dirty = True

    # ── Inflate ───────────────────────────────────────────────────────────────

    def inflate(self) -> None:
        """
        Recompute inflation from current occupancy data.
        Call after each SLAM update (or lazily on first .cost() access).
        """
        obs_mask = (self.grid.data == 100)

        if not np.any(obs_mask):
            self._cost[:] = 0.0
            self._dirty   = False
            return

        # Euclidean distance (cells) to nearest obstacle
        dist_cells = distance_transform_edt(~obs_mask)
        dist_m     = dist_cells * self.grid.resolution

        radius   = self.cfg.INFLATION_RADIUS
        max_cost = float(self.cfg.INFLATION_COST_MAX)

        # Linear decay
        inflated = np.where(
            dist_m < radius,
            max_cost * (1.0 - dist_m / radius),
            0.0,
        ).astype(np.float32)

        # Obstacle cells stay at 100 (impassable)
        inflated[obs_mask] = 100.0
        # Unknown cells also impassable during planning
        inflated[self.grid.data == -1] = 100.0

        self._cost  = inflated
        self._dirty = False
        log.debug("CostMap inflated")

    # ── Access ────────────────────────────────────────────────────────────────

    def cost(self, row: int, col: int) -> float:
        if self._dirty:
            self.inflate()
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            return float(self._cost[row, col])
        return 100.0

    @property
    def data(self) -> np.ndarray:
        if self._dirty:
            self.inflate()
        return self._cost

    def mark_dirty(self) -> None:
        self._dirty = True
