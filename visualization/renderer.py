"""
visualization/renderer.py
═══════════════════════════════════════════════════════════════════
Real-Time Renderer

Renders 3 panels in a matplotlib figure:
  Panel 1 — Occupancy Grid (SLAM)
  Panel 2 — A* Path + Costmap Inflation
  Panel 3 — Navigation Stats + Trajectory

Performance: redraws every RENDER_EVERY steps to keep the
simulation fast even with matplotlib overhead.
═══════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from typing import List, Tuple

from simulation.world import MazeWorld
from core.occupancy_grid import OccupancyGrid
from config.settings import Settings

_DARK = "#0d1117"
_CYAN = "#00e5ff"
_GREEN= "#39ff14"
_GOLD = "#ffd700"
_RED  = "#ff4500"
_ORANGE="#ff8c00"


def _occ_to_rgb(occ: np.ndarray) -> np.ndarray:
    """Convert occupancy int16 array → RGB float array for imshow."""
    h, w  = occ.shape
    img   = np.zeros((h, w, 3), dtype=np.float32)
    img[occ == -1] = [0.14, 0.14, 0.20]   # unknown  — deep blue-grey
    img[occ ==  0] = [0.82, 0.87, 0.93]   # free     — soft white-blue
    img[occ == 100]= [0.07, 0.07, 0.09]   # obstacle — near-black
    return img


class Renderer:
    """Matplotlib-based live renderer."""

    RENDER_EVERY = 8   # draw every N steps

    def __init__(
        self,
        world:    MazeWorld,
        grid:     OccupancyGrid,
        costmap,                  # CostMap
        cfg:      Settings,
        headless: bool = False,
    ):
        self.world    = world
        self.grid     = grid
        self.costmap  = costmap
        self.cfg      = cfg
        self.headless = headless
        self._step    = 0
        self._traj_x: List[float] = []
        self._traj_y: List[float] = []
        self._replan_pts: List[Tuple] = []

        if headless:
            matplotlib.use("Agg")

        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 7))
        self.fig.patch.set_facecolor(_DARK)
        for ax in self.axes:
            ax.set_facecolor(_DARK)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a2a")
            ax.tick_params(colors="#555")

        plt.tight_layout(pad=2.0)
        os.makedirs("outputs", exist_ok=True)

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(
        self,
        robot_pose: np.ndarray,
        path:       List[Tuple],
        scan:       dict,
        step:       int,
        is_replan:  bool = False,
        goal_reached: bool = False,
        stats:      dict  = None,
    ) -> None:
        self._step += 1
        res = self.cfg.GRID_RESOLUTION

        # Accumulate trajectory
        self._traj_x.append(robot_pose[0] / res)
        self._traj_y.append(robot_pose[1] / res)
        if is_replan:
            self._replan_pts.append((robot_pose[0]/res, robot_pose[1]/res))

        if self._step % self.RENDER_EVERY != 0 and not goal_reached:
            return

        rx = robot_pose[0] / res
        ry = robot_pose[1] / res
        gr, gc = self.world.goal_cell

        # ── Panel 1: Occupancy Grid ───────────────────────────────────────────
        ax1 = self.axes[0]
        ax1.cla()
        ax1.set_facecolor(_DARK)
        ax1.set_title("Task 1 – 2D Occupancy Grid (SLAM)", color="white", fontsize=10, fontweight="bold")

        occ_rgb = _occ_to_rgb(self.grid.data)
        ax1.imshow(occ_rgb, origin="upper", interpolation="nearest")

        # Trajectory
        if len(self._traj_x) > 1:
            ax1.plot(self._traj_x, self._traj_y, color=_CYAN, lw=1.2, alpha=0.7)

        # Start / Goal markers
        sx, sy = robot_pose[0]/res, robot_pose[1]/res
        ax1.plot(self._traj_x[0], self._traj_y[0], "o", color=_GREEN, ms=8, zorder=6)
        ax1.plot(gc, gr, "*", color=_GOLD, ms=14, zorder=6)

        # Robot arrow
        dxr = np.cos(robot_pose[2]) * 1.8
        dyr = np.sin(robot_pose[2]) * 1.8
        ax1.annotate("", xy=(rx+dxr, ry+dyr), xytext=(rx, ry),
                     arrowprops=dict(arrowstyle="->", color=_CYAN, lw=2))

        # Replan markers
        for (rpx, rpy) in self._replan_pts[-20:]:
            ax1.plot(rpx, rpy, "x", color=_RED, ms=5, mew=1.5)

        # Legend
        patches = [
            mpatches.Patch(fc=[0.82,0.87,0.93], label="Free"),
            mpatches.Patch(fc=[0.07,0.07,0.09], ec="gray", label="Obstacle"),
            mpatches.Patch(fc=[0.14,0.14,0.20], label="Unknown"),
        ]
        ax1.legend(handles=patches, fontsize=7, facecolor="#1a1a2e",
                   labelcolor="white", loc="upper right")
        ax1.text(0.02, 0.02, f"Coverage: {self.grid.coverage_percent():.0f}%",
                 transform=ax1.transAxes, color=_CYAN, fontsize=9)
        ax1.set_xlim(0, self.grid.cols); ax1.set_ylim(self.grid.rows, 0)

        # ── Panel 2: Costmap + A* Path ────────────────────────────────────────
        ax2 = self.axes[1]
        ax2.cla()
        ax2.set_facecolor(_DARK)
        ax2.set_title("Task 2 – A* Path + Costmap Inflation", color="white", fontsize=10, fontweight="bold")

        ax2.imshow(self.costmap.data, cmap="plasma", origin="upper",
                   vmin=0, vmax=100, alpha=0.8, interpolation="nearest")

        # A* path
        if path:
            prows = [c[0] for c in path]
            pcols = [c[1] for c in path]
            ax2.plot(pcols, prows, color=_GREEN, lw=2.5, alpha=0.9, label=f"A* ({len(path)} cells)")

        ax2.plot(self._traj_x[0], self._traj_y[0], "o", color=_GREEN, ms=8, zorder=6)
        ax2.plot(gc, gr, "*", color=_GOLD, ms=14, zorder=6)
        ax2.plot(rx, ry, "o", color=_CYAN, ms=9, zorder=7, label="Robot")

        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
        ax2.text(0.02, 0.02, f"Steps: {step}", transform=ax2.transAxes,
                 color=_GREEN, fontsize=9)
        ax2.set_xlim(0, self.grid.cols); ax2.set_ylim(self.grid.rows, 0)

        # ── Panel 3: Navigation Summary ───────────────────────────────────────
        ax3 = self.axes[2]
        ax3.cla()
        ax3.set_facecolor(_DARK)
        ax3.set_title("Task 3 – Obstacle Avoidance + Navigation Stats",
                      color="white", fontsize=10, fontweight="bold")

        ax3.imshow(occ_rgb, origin="upper", interpolation="nearest", alpha=0.5)

        # Full trajectory
        if len(self._traj_x) > 1:
            ax3.plot(self._traj_x, self._traj_y, color=_CYAN, lw=2.0)

        ax3.plot(self._traj_x[0], self._traj_y[0], "o", color=_GREEN, ms=10, zorder=6)
        ax3.plot(gc, gr, "*", color=_GOLD, ms=14, zorder=6)

        # Dynamic obstacles
        for d in self.world.dynamics:
            circ = plt.Circle((d.x/res, d.y/res), d.radius/res,
                              color=_RED, alpha=0.7, zorder=8)
            ax3.add_patch(circ)

        # Stats text
        if stats:
            txt = (
                f"Status:        {'REACHED ✓' if goal_reached else 'Navigating...'}\n"
                f"Steps:         {step}\n"
                f"Stop events:   {stats.get('stop_events', 0)}\n"
                f"Replan events: {stats.get('replan_events', 0)}\n"
                f"Coverage:      {self.grid.coverage_percent():.0f}%"
            )
            ax3.text(0.02, 0.02, txt, transform=ax3.transAxes,
                     color=_CYAN, fontsize=8.5, fontfamily="monospace",
                     va="bottom",
                     bbox=dict(fc="#0a0a18", alpha=0.85, boxstyle="round,pad=0.4"))

        ax3.set_xlim(0, self.grid.cols); ax3.set_ylim(self.grid.rows, 0)

        self.fig.suptitle(
            "Project 3: AMR Navigation  |  DecodeLabs Robotics & Automation 2026",
            color="white", fontsize=12, fontweight="bold",
        )
        plt.tight_layout(pad=1.5)
        self.fig.canvas.draw_idle()
        if not self.headless:
            plt.pause(0.001)

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, path: str = "outputs/final_map.png") -> None:
        self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK)
        print(f"[Renderer] Saved → {path}")
