"""
sensors/lidar.py — Simulated LiDAR (LaserScan)
Vectorised ray-casting. Beams within MIN_RANGE are ignored (self-filter).
"""
import numpy as np
from simulation.world import MazeWorld
from config.settings  import Settings


class SimulatedLiDAR:
    MIN_RANGE = 0.12   # metres — beams shorter than this are self-hits

    def __init__(self, world: MazeWorld, cfg: Settings):
        self.world      = world
        self.cfg        = cfg
        self._step      = cfg.GRID_RESOLUTION * 0.5
        self._max_range = cfg.LIDAR_MAX_RANGE
        self._noise     = cfg.LIDAR_NOISE_STD
        self._angles    = np.linspace(-cfg.LIDAR_FOV/2, cfg.LIDAR_FOV/2,
                                      cfg.LIDAR_NUM_BEAMS, dtype=np.float32)

    def scan(self, pose: np.ndarray) -> dict:
        x0, y0, th = float(pose[0]), float(pose[1]), float(pose[2])
        abs_a = (self._angles + th).astype(np.float64)
        ranges = self._cast(x0, y0, np.cos(abs_a), np.sin(abs_a))
        # Noise
        ranges += np.random.normal(0, self._noise, len(ranges))
        # Clamp and apply min-range filter (self-hit suppression)
        ranges = np.clip(ranges, self.MIN_RANGE, self._max_range).astype(np.float32)
        return {"ranges": ranges, "angles": self._angles.copy()}

    def _cast(self, x0, y0, cos_a, sin_a) -> np.ndarray:
        n      = len(cos_a)
        ranges = np.full(n, self._max_range, dtype=np.float64)
        active = np.ones(n, dtype=bool)
        dist   = np.full(n, self._step, dtype=np.float64)  # start 1 step out
        step   = self._step
        max_r  = self._max_range
        while np.any(active):
            done = dist >= max_r
            active[done] = False
            idx = np.where(active)[0]
            if idx.size == 0: break
            ex = x0 + dist[idx] * cos_a[idx]
            ey = y0 + dist[idx] * sin_a[idx]
            hit = self.world.is_obstacle_world(ex, ey)
            hit_idx = idx[hit]
            if hit_idx.size > 0:
                ranges[hit_idx] = np.maximum(dist[hit_idx] - step, self.MIN_RANGE)
                active[hit_idx] = False
            dist[active] += step
        return ranges
