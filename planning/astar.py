"""
planning/astar.py
═══════════════════════════════════════════════════════════════════
TASK 2 — A* (A-Star) Pathfinding Algorithm

Implements the algorithm described in PDF slides:
  "Anatomy of the A* Equation"          f(n) = g(n) + h(n)
  "Implementing A*: The Priority Queue Logic"
  "Guiding the Algorithm: Choosing the Right Heuristic"
  "Why A*? A Pathfinding Diagnostic Matrix"

Design choices (from PDF):
  • Manhattan distance heuristic — admissible on 4-connected grid,
    perfectly matches the robot's discrete grid-based reality.
    (PDF: "Manhattan Distance Advantage: perfectly matches the
     robot's discrete grid-based reality.")
  • heapq priority queue  — O(log n) push/pop
  • numpy boolean closed-set — O(1) membership test
  • Cost penalises cells near obstacles via inflated costmap
  • Path reconstruction via parent dict

f(n) = g(n) + h(n)
  g(n) = exact accumulated cost from start to n
  h(n) = Manhattan distance from n to goal
  f(n) = total priority (always expand lowest f first)
═══════════════════════════════════════════════════════════════════
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict
from planning.costmap import CostMap
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)
Cell = Tuple[int, int]


class AStarPlanner:
    """
    Grid-based A* planner operating on an inflated costmap.

    Supports 4-connected movement (up/down/left/right).
    Diagonal can be enabled via cfg.ASTAR_ALLOW_DIAGONAL.
    """

    _MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _MOVES_8 = _MOVES_4 + [(-1,-1), (-1,1), (1,-1), (1,1)]

    def __init__(self, costmap: CostMap, cfg: Settings):
        self.costmap = costmap
        self.cfg     = cfg
        self._moves  = self._MOVES_8 if cfg.ASTAR_ALLOW_DIAGONAL else self._MOVES_4
        self._n_plans = 0

    # ── Heuristic ─────────────────────────────────────────────────────────────

    @staticmethod
    def heuristic(a: Cell, b: Cell) -> float:
        """
        Manhattan distance — admissible for 4-connected grid.
        |Δrow| + |Δcol|
        """
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    # ── Core A* Algorithm ─────────────────────────────────────────────────────

    def plan(self, start: Cell, goal: Cell) -> List[Cell]:
        """
        Find the lowest-cost path from start to goal.

        Parameters
        ----------
        start : (row, col)
        goal  : (row, col)

        Returns
        -------
        List[Cell] from start-exclusive to goal-inclusive.
        Empty list if no path found.
        """
        self._n_plans += 1
        grid  = self.costmap.grid
        rows, cols = grid.shape

        # ── Sanitise start / goal ─────────────────────────────────────────────
        if grid.is_obstacle(*start):
            start = self._nearest_free(start)
            if start is None:
                log.warning("A*: no free cell near start")
                return []
        if grid.is_obstacle(*goal):
            goal = self._nearest_free(goal)
            if goal is None:
                log.warning("A*: no free cell near goal")
                return []

        # ── Initialise data structures ────────────────────────────────────────
        # g_cost[r,c] = best known cost to reach cell (r,c)
        g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
        g_cost[start] = 0.0

        # Closed set — boolean mask, O(1) lookup
        closed = np.zeros((rows, cols), dtype=bool)

        # Parent pointers for path reconstruction
        parent: Dict[Cell, Optional[Cell]] = {start: None}

        # Priority queue: (f_value, tie_breaker, cell)
        h0 = self.heuristic(start, goal)
        heap: List = [(h0, 0, start)]
        counter = 0   # tie-breaker to avoid comparing tuples

        iters   = 0
        max_it  = self.cfg.ASTAR_MAX_ITERS

        # ── Main Loop ─────────────────────────────────────────────────────────
        while heap and iters < max_it:
            iters += 1
            f, _, current = heapq.heappop(heap)

            if closed[current]:
                continue
            closed[current] = True

            # ── Goal reached ──────────────────────────────────────────────────
            if current == goal:
                path = self._reconstruct(parent, goal)
                log.info(
                    f"A* plan #{self._n_plans}: {len(path)} cells, "
                    f"{iters} iters, g={g_cost[goal]:.1f}"
                )
                return path

            # ── Expand neighbours ─────────────────────────────────────────────
            cr, cc = current
            for dr, dc in self._moves:
                nr, nc = cr + dr, cc + dc
                nb = (nr, nc)

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if closed[nb]:
                    continue

                cell_cost = self.costmap.cost(nr, nc)
                if cell_cost >= 100.0:        # impassable
                    continue

                # Diagonal costs √2, cardinal costs 1
                step = 1.414 if (dr != 0 and dc != 0) else 1.0
                # Penalise proximity to walls via inflated cost
                step += cell_cost * 0.02

                tg = float(g_cost[current]) + step
                if tg < g_cost[nb]:
                    g_cost[nb] = tg
                    parent[nb] = current
                    h  = self.heuristic(nb, goal)
                    counter += 1
                    heapq.heappush(heap, (tg + h, counter, nb))

        log.warning(f"A* plan #{self._n_plans}: no path after {iters} iters")
        return []

    # ── Path Reconstruction ───────────────────────────────────────────────────

    def _reconstruct(
        self, parent: Dict[Cell, Optional[Cell]], goal: Cell
    ) -> List[Cell]:
        path = []
        node: Optional[Cell] = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path[1:]   # exclude start cell

    # ── Nearest Free Cell ─────────────────────────────────────────────────────

    def _nearest_free(self, cell: Cell, radius: int = 8) -> Optional[Cell]:
        """BFS to nearest non-obstacle cell within radius cells."""
        from collections import deque
        grid = self.costmap.grid
        rows, cols = grid.shape
        seen = {cell}
        q = deque([cell])
        while q:
            r, c = q.popleft()
            if not grid.is_obstacle(r, c):
                return (r, c)
            for dr, dc in self._MOVES_4:
                nr, nc = r + dr, c + dc
                nb = (nr, nc)
                if (
                    nb not in seen and
                    0 <= nr < rows and 0 <= nc < cols and
                    abs(nr - cell[0]) <= radius and
                    abs(nc - cell[1]) <= radius
                ):
                    seen.add(nb)
                    q.append(nb)
        return None
