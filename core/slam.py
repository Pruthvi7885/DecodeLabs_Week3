"""
core/slam.py
═══════════════════════════════════════════════════════════════════
SLAM Engine — Simultaneous Localisation and Mapping

Implements the **occupancy-grid inverse sensor model** using
Bresenham line rasterisation so that every LiDAR beam correctly
marks the free corridor and the hit endpoint.

PDF concepts implemented:
  • "Feed 1: Simulated LiDAR (LaserScan arrays)"
  • "The SLAM Transformation Tree: map → odom → base_link"
  • 2D occupancy tensor with -1/0/100 cell values

Algorithm per beam
──────────────────
  1. Compute absolute endpoint (world x,y) from range + angle.
  2. Walk a Bresenham line from robot cell to endpoint cell.
  3. Mark all traversed cells (except last) as FREE.
  4. If range < max_range → mark endpoint as OBSTACLE.
     Otherwise → mark endpoint as FREE (no wall found).
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from core.occupancy_grid import OccupancyGrid
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Bresenham integer line ────────────────────────────────────────────────────

def _bresenham(r0: int, c0: int, r1: int, c1: int):
    """
    Yield (row, col) integer cells along the line from (r0,c0) to (r1,c1).
    Pure Python — called once per beam per step; fast enough at 180 beams.
    """
    dr = abs(r1 - r0);  sr = 1 if r1 > r0 else -1
    dc = abs(c1 - c0);  sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = err * 2
        if e2 > -dc:
            err -= dc;  r += sr
        if e2 <  dr:
            err += dr;  c += sc


class SLAMEngine:
    """
    Greedy occupancy-grid SLAM.

    Processes each LiDAR scan at the robot's (EKF-fused) pose and
    incrementally builds the 2-D occupancy grid.
    """

    def __init__(self, grid: OccupancyGrid, cfg: Settings):
        self.grid = grid
        self.cfg  = cfg
        self._n   = 0   # update counter

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, pose: np.ndarray, scan: dict) -> None:
        """
        Integrate one LiDAR scan into the occupancy grid.

        Parameters
        ----------
        pose : [x, y, theta]  EKF-fused robot pose in world metres
        scan : {'ranges': ndarray, 'angles': ndarray}
               angles are relative to robot heading (from LiDAR)
        """
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        ranges  = scan["ranges"]                      # (N,)
        angles  = scan["angles"] + theta              # absolute world angles (N,)
        max_r   = self.cfg.LIDAR_MAX_RANGE
        noise   = self.cfg.LIDAR_NOISE_STD * 2

        origin_row, origin_col = self.grid.world_to_cell(x, y)

        for i in range(len(ranges)):
            r = float(ranges[i])
            a = float(angles[i])

            ex = x + r * np.cos(a)
            ey = y + r * np.sin(a)
            end_row, end_col = self.grid.world_to_cell(ex, ey)

            # Walk ray — mark free cells
            cells = list(_bresenham(origin_row, origin_col, end_row, end_col))
            for rr, cc in cells[:-1]:
                self.grid.mark_free(rr, cc)

            # Mark endpoint
            if r < max_r - noise:
                self.grid.mark_obstacle(end_row, end_col)
            else:
                self.grid.mark_free(end_row, end_col)

        self._n += 1
        if self._n % 200 == 0:
            log.debug(
                f"SLAM #{self._n} | coverage {self.grid.coverage_percent():.1f}%"
            )
