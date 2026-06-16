"""
core/obstacle_avoidance.py — Task 3: Dynamic Obstacle Avoidance

True dynamic obstacle detection:
  A cell is a DYNAMIC OBSTACLE if:
    - The world simulation says it's an obstacle (world.is_obstacle_world = True)
    - BUT the static occupancy grid says it's NOT a mapped wall (grid.is_obstacle = False)
  This precisely identifies moving boxes/pedestrians vs known walls.
"""
import numpy as np
from config.settings import Settings
from utils.logger    import get_logger

log = get_logger(__name__)


class ObstacleAvoider:
    MIN_RANGE = 0.15   # ignore beams shorter than this (self-hit filter)

    def __init__(self, cfg: Settings):
        self.cfg            = cfg
        self._stop_events   = 0
        self._replan_events = 0
        self._world         = None   # set by main after construction

    def set_world(self, world):
        """Call once from main to give avoider access to world for dynamic queries."""
        self._world = world

    def check(self, pose, scan, grid):
        """
        Returns (danger: bool, reason: str) where reason ∈ {'','WARN','STOP'}.
        Only fires for truly dynamic (unmapped) obstacles.
        """
        if self._world is None:
            return False, ""

        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        ranges   = scan["ranges"]
        angles   = scan["angles"] + th

        valid = (ranges < self.cfg.LIDAR_MAX_RANGE - 0.05) & (ranges > self.MIN_RANGE)
        if not np.any(valid):
            return False, ""

        dynamic_dists = []
        idx = np.where(valid)[0]
        for i in idx:
            r = float(ranges[i]); a = float(angles[i])
            ex = x + r * np.cos(a); ey = y + r * np.sin(a)
            # Dynamic = world says obstacle BUT grid doesn't know about it
            w_obs = bool(self._world.is_obstacle_world(
                np.array([ex]), np.array([ey]))[0])
            row, col = grid.world_to_cell(ex, ey)
            g_obs = grid.is_obstacle(row, col)
            if w_obs and not g_obs:
                dynamic_dists.append(r)

        if not dynamic_dists:
            return False, ""

        min_d = min(dynamic_dists)
        if min_d < self.cfg.SAFE_DISTANCE:
            self._stop_events += 1
            log.debug(f"STOP  dynamic obstacle at {min_d:.3f}m")
            return True, "STOP"
        if min_d < self.cfg.WARN_DISTANCE:
            self._replan_events += 1
            log.debug(f"WARN  dynamic obstacle at {min_d:.3f}m")
            return True, "WARN"
        return False, ""

    def sector_analysis(self, scan):
        r = scan["ranges"]; n = len(r); t = n//3
        return {"LEFT":  float(np.min(r[:t])),
                "FRONT": float(np.min(r[t:2*t])),
                "RIGHT": float(np.min(r[2*t:]))}

    @property
    def stats(self):
        return {"stop_events": self._stop_events,
                "replan_events": self._replan_events}
