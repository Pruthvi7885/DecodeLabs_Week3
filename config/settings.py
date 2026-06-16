"""
config/settings.py
Central configuration — every tunable constant lives here.
Import Settings anywhere in the project.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Settings:
    # ── Simulation ────────────────────────────────────────────────────────────
    DT: float = 0.05            # seconds per step
    MAX_STEPS: int = 8_000

    # ── Grid / Map ────────────────────────────────────────────────────────────
    GRID_RESOLUTION: float = 0.1    # metres per cell
    GRID_FREE: int         = 0
    GRID_OBSTACLE: int     = 100
    GRID_UNKNOWN: int      = -1

    # ── LiDAR ────────────────────────────────────────────────────────────────
    LIDAR_MAX_RANGE: float = 3.5
    LIDAR_NUM_BEAMS: int   = 180
    LIDAR_NOISE_STD: float = 0.02
    LIDAR_FOV: float       = np.pi   # 180 degrees

    # ── Odometry ─────────────────────────────────────────────────────────────
    ODOM_NOISE_LINEAR: float  = 0.01
    ODOM_NOISE_ANGULAR: float = 0.02

    # ── EKF ──────────────────────────────────────────────────────────────────
    EKF_Q_DIAG: list = field(default_factory=lambda: [0.01, 0.01, 0.005])
    EKF_R_DIAG: list = field(default_factory=lambda: [0.05, 0.05, 0.01])

    # ── CostMap Inflation ─────────────────────────────────────────────────────
    INFLATION_RADIUS: float = 0.20   # metres
    INFLATION_COST_MAX: int = 90

    # ── A* Planner ────────────────────────────────────────────────────────────
    ASTAR_ALLOW_DIAGONAL: bool = False
    ASTAR_MAX_ITERS: int       = 100_000

    # ── Robot ─────────────────────────────────────────────────────────────────
    ROBOT_RADIUS: float    = 0.18
    ROBOT_MAX_SPEED: float = 0.5
    ROBOT_MAX_OMEGA: float = 1.5

    # ── Obstacle Avoidance (dynamic only) ─────────────────────────────────────
    SAFE_DISTANCE: float = 0.25   # metres — STOP (dynamic obstacle)
    WARN_DISTANCE: float = 0.40   # metres — REPLAN (dynamic obstacle)
