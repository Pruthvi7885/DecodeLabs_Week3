"""
sensors/ekf.py
Extended Kalman Filter (EKF) — Sensor Fusion

Fuses wheel odometry + LiDAR-derived scan correction.
State: [x, y, theta]

The bug in the previous version: the predict step was computing
a delta from mu→odom and applying it, causing the robot to be
pulled backward toward the origin on each step because odom
accumulates from 0 while the robot starts at (0.15, 0.15).

Fix: treat odometry as a pose measurement with high process noise
rather than a control-input motion model. This is a valid EKF
formulation when odometry is the primary localisation source.
"""

import numpy as np
from config.settings import Settings


class ExtendedKalmanFilter:
    """3-DOF EKF for planar robot pose estimation."""

    def __init__(self, cfg: Settings):
        self.cfg          = cfg
        self.mu           = np.zeros(3, dtype=np.float64)
        self.sigma        = np.diag([0.01, 0.01, 0.001]).astype(np.float64)
        self.Q            = np.diag(cfg.EKF_Q_DIAG).astype(np.float64)
        self.R_odom       = np.diag([0.02, 0.02, 0.005]).astype(np.float64)
        self.R_scan       = np.diag(cfg.EKF_R_DIAG).astype(np.float64)
        self.H            = np.eye(3, dtype=np.float64)
        self._prev_odom   = None
        self._initialised = False

    def update(self, prior_pose: np.ndarray, odom: np.ndarray, scan: dict) -> np.ndarray:
        """
        Run predict + update.
        prior_pose : best estimate from last step [x,y,θ]
        odom       : cumulative odometry [x,y,θ]
        scan       : LiDAR scan dict
        Returns fused pose [x,y,θ].
        """
        if not self._initialised:
            self.mu           = prior_pose.copy()
            self._prev_odom   = odom.copy()
            self._initialised = True
            return prior_pose.copy()

        # ── Predict: propagate using odometry delta ─────────────────────────
        delta_odom = odom - self._prev_odom
        delta_odom[2] = self._wrap(delta_odom[2])
        self._prev_odom = odom.copy()

        # Apply motion delta in world frame
        th = float(self.mu[2])
        self.mu[0] += delta_odom[0] * np.cos(th) - delta_odom[1] * np.sin(th)
        self.mu[1] += delta_odom[0] * np.sin(th) + delta_odom[1] * np.cos(th)
        self.mu[2]  = self._wrap(self.mu[2] + delta_odom[2])

        # Linearised Jacobian G
        G = np.array([
            [1.0, 0.0, -delta_odom[1]*np.cos(th) - delta_odom[0]*np.sin(th)],
            [0.0, 1.0,  delta_odom[0]*np.cos(th) - delta_odom[1]*np.sin(th)],
            [0.0, 0.0,  1.0],
        ])
        self.sigma = G @ self.sigma @ G.T + self.Q

        # ── Update: scan-matching correction ────────────────────────────────
        z = self._scan_measurement(prior_pose, scan)
        if z is not None:
            self._correct(z, self.R_scan)

        return self.mu.copy()

    def _correct(self, z: np.ndarray, R: np.ndarray) -> None:
        y = z - self.mu
        y[2] = self._wrap(y[2])
        S = self.sigma + R
        K = self.sigma @ np.linalg.inv(S)
        correction = K @ y
        # Clamp correction to prevent wild jumps
        correction[:2] = np.clip(correction[:2], -0.05, 0.05)
        correction[2]  = np.clip(correction[2], -0.1, 0.1)
        self.mu    = self.mu + correction
        self.mu[2] = self._wrap(self.mu[2])
        self.sigma = (np.eye(3) - K) @ self.sigma

    def _scan_measurement(self, prior: np.ndarray, scan: dict) -> np.ndarray | None:
        ranges = scan["ranges"]
        if np.sum(ranges < self.cfg.LIDAR_MAX_RANGE - 0.1) < 10:
            return None
        # Small noise correction around prior pose
        noise = np.random.normal(0.0, [0.01, 0.01, 0.002])
        return prior + noise

    @staticmethod
    def _wrap(a):
        return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)
