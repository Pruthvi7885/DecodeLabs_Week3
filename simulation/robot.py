"""
simulation/robot.py
Robot — Differential Drive with stable pure-pursuit controller.
"""
import numpy as np
from config.settings import Settings


class Robot:
    def __init__(self, pose: np.ndarray, cfg: Settings):
        self.pose     = np.array(pose, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.cfg      = cfg

    def move_toward(self, target_xy: tuple, dt: float) -> None:
        x, y, th = self.pose
        tx, ty   = target_xy
        dx, dy   = tx - x, ty - y
        dist     = np.hypot(dx, dy)
        if dist < 0.02:
            self.velocity[:] = 0.0
            return
        desired  = np.arctan2(dy, dx)
        err      = self._wrap(desired - th)
        # Reduce speed when heading error is large (no oscillation)
        speed    = self.cfg.ROBOT_MAX_SPEED * np.clip(np.cos(err), 0.0, 1.0)
        omega    = float(np.clip(3.0 * err, -self.cfg.ROBOT_MAX_OMEGA, self.cfg.ROBOT_MAX_OMEGA))
        self.velocity = np.array([float(speed), omega])
        self._integrate(dt)

    def stop(self):
        self.velocity[:] = 0.0

    def _integrate(self, dt: float):
        v, w = self.velocity
        x, y, th = self.pose
        self.pose[0] += v * np.cos(th) * dt
        self.pose[1] += v * np.sin(th) * dt
        self.pose[2]  = self._wrap(th + w * dt)

    @staticmethod
    def _wrap(a):
        return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)

    def __repr__(self):
        x, y, t = self.pose
        return f"Robot(x={x:.2f}, y={y:.2f}, θ={np.degrees(t):.1f}°)"
