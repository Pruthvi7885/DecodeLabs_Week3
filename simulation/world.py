"""
simulation/world.py
═══════════════════════════════════════════════════════════════════
MazeWorld — Physical Simulation Environment

Provides the "Physical World (Gazebo Simulation)" layer from the PDF.
Three maze layouts are included (default / complex / open).
Dynamic obstacles (boxes) bounce along corridors to test Task 3.

Grid encoding:
  '#' = wall
  'S' = robot start
  'G' = goal
  ' ' = free space
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Tuple
from config.settings import Settings


# ── Maze layouts 20×20 ────────────────────────────────────────────────────────

MAZE_LAYOUTS = {
    "default": [
        "####################",
        "#S                 #",
        "#  ###  #  ## ###  #",
        "#  #    #     #    #",
        "#  # ####### ##    #",
        "#    #    #   #    #",
        "#### # ##   # #    #",
        "#    # #  # # #####",
        "#  ###   ## #      #",
        "#    # ##    # ### #",
        "## # #    ## #   # #",
        "#  # ## ##   # # # #",
        "#  #  #   #  # #   #",
        "#     # # ## #   # #",
        "## ## #   #  ##### #",
        "#     ## ##        #",
        "#  ## #    ## ###  #",
        "#   # ## ##   #    #",
        "#   #       #     G#",
        "####################",
    ],
    "complex": [
        "####################",
        "#S  ##   ##   #    #",
        "# ## ## # # ##  ## #",
        "#     #   #  ## #  #",
        "## ## ## ##  #   # #",
        "#   #   #  ## ## # #",
        "#  ## # # #    # # #",
        "## #  # #  ## ## # #",
        "#   ## ##  #    #  #",
        "# ##    ## ## ##   #",
        "#   ### #    # ##  #",
        "## #   # ## #   #  #",
        "#  ## #   # ## # ##",
        "#    # ## #   # #  #",
        "## # #  # ## ## #  #",
        "#   ## #   #    ## #",
        "#  # # ## ## # #   #",
        "## # #   #   # ## ##",
        "#   ## ## ## #    G#",
        "####################",
    ],
    "open": [
        "####################",
        "#S                 #",
        "#                  #",
        "#    ####          #",
        "#    #             #",
        "#    #    ###      #",
        "#         ###      #",
        "#               #  #",
        "#               #  #",
        "#    ##         #  #",
        "#    ##            #",
        "#          #####   #",
        "#                  #",
        "#       #          #",
        "#       #      ##  #",
        "#       #      ##  #",
        "#                  #",
        "#                  #",
        "#                 G#",
        "####################",
    ],
}


class DynamicObstacle:
    """Bouncing dynamic obstacle (box) for Task 3 testing."""

    def __init__(self, x: float, y: float, vx: float, vy: float, r: float = 0.12):
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.radius      = r

    def step(self, dt: float, w: float, h: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        if not (self.radius < self.x < w - self.radius): self.vx *= -1
        if not (self.radius < self.y < h - self.radius): self.vy *= -1


class MazeWorld:
    """
    Holds the static obstacle map and dynamic obstacles.
    Provides vectorised obstacle query used by SimulatedLiDAR.
    """

    def __init__(self, layout: str = "default", cfg: Settings = None):
        self.cfg        = cfg or Settings()
        self.resolution = self.cfg.GRID_RESOLUTION

        rows_str = MAZE_LAYOUTS.get(layout, MAZE_LAYOUTS["default"])
        self.grid_rows = len(rows_str)
        self.grid_cols = max(len(r) for r in rows_str)
        self.height_m  = self.grid_rows * self.resolution
        self.width_m   = self.grid_cols * self.resolution

        # Boolean obstacle map
        self._obs = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        self.start_pose = np.array([0.15, 0.15, 0.0])
        self.goal_cell  = (self.grid_rows - 2, self.grid_cols - 2)

        for r, row_s in enumerate(rows_str):
            for c, ch in enumerate(row_s):
                if ch == "#":
                    self._obs[r, c] = True
                elif ch == "S":
                    self.start_pose = np.array([
                        (c + 0.5) * self.resolution,
                        (r + 0.5) * self.resolution,
                        0.0,
                    ])
                elif ch == "G":
                    self.goal_cell = (r, c)

        # Dynamic obstacles — placed in open corridors
        self.dynamics: list = [
            DynamicObstacle(x=1.0, y=0.8,  vx=0.12, vy=0.0),
            DynamicObstacle(x=0.8, y=1.4,  vx=0.0,  vy=0.09),
            DynamicObstacle(x=1.5, y=1.0,  vx=0.07, vy=0.06),
        ]

    # ── Obstacle Query ────────────────────────────────────────────────────────

    def is_obstacle_world(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Vectorised: returns bool array for arrays of (x, y) world coords.
        Checks both static map and dynamic obstacles.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))

        col = np.floor(x / self.resolution).astype(int)
        row = np.floor(y / self.resolution).astype(int)

        oob = (row < 0) | (row >= self.grid_rows) | (col < 0) | (col >= self.grid_cols)
        row = np.clip(row, 0, self.grid_rows - 1)
        col = np.clip(col, 0, self.grid_cols - 1)

        hit = self._obs[row, col] | oob

        for d in self.dynamics:
            dx = x - d.x;  dy = y - d.y
            hit |= (dx*dx + dy*dy) < d.radius ** 2

        return hit

    def step(self, dt: float) -> None:
        for d in self.dynamics:
            d.step(dt, self.width_m, self.height_m)

    @property
    def obstacle_map(self) -> np.ndarray:
        """Boolean static obstacle map."""
        return self._obs.copy()
