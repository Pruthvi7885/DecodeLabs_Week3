"""
sensors/odometry.py
═══════════════════════════════════════════════════════════════════
Wheel Odometry — Feed 2 (PDF: "Raw encoder tick counts")

Simulates differential-drive encoder-based dead-reckoning with
realistic drift/slip noise.

PDF insight: "Odometry provides a continuous but drifting local
frame. We cannot rely on wheel encoders alone."

The accumulated pose drifts over time due to:
  • Wheel slip (microscopic losses of traction)
  • Angular error accumulation (small errors compound)

This noisy estimate is fed into the EKF for fusion.
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from config.settings import Settings


class WheelOdometry:
    """
    Encoder-based dead-reckoning with proportional noise.
    Integrates [v_linear, omega] commands using Euler integration.
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._x  = 0.0
        self._y  = 0.0
        self._th = 0.0

    def update(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Parameters
        ----------
        velocity : [v (m/s), omega (rad/s)]
        dt       : timestep seconds

        Returns
        -------
        np.ndarray [x, y, theta]  — cumulative noisy odometry pose
        """
        v, w = float(velocity[0]), float(velocity[1])

        # Proportional noise — larger motion = larger error
        v_n = v + np.random.normal(0.0, self.cfg.ODOM_NOISE_LINEAR  * (abs(v)  + 1e-4))
        w_n = w + np.random.normal(0.0, self.cfg.ODOM_NOISE_ANGULAR * (abs(w)  + 1e-4))

        self._th  = (self._th + w_n * dt + np.pi) % (2 * np.pi) - np.pi
        self._x  += v_n * np.cos(self._th) * dt
        self._y  += v_n * np.sin(self._th) * dt

        return np.array([self._x, self._y, self._th], dtype=np.float64)

    def reset(self) -> None:
        self._x = self._y = self._th = 0.0
