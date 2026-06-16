"""
main.py – AMR Navigation  |  Project 3  |  DecodeLabs 2026
=============================================================
Correct loop order (critical):
  1. CONTROL  – move_toward() sets velocity for this step
  2. INTEGRATE – robot._integrate() already done inside move_toward
  3. SENSE    – lidar.scan() at new pose
  4. LOCALISE – odom uses velocity just commanded; EKF fuses
  5. MAP      – SLAM update
  6. PLAN     – A* replan if needed
  7. AVOID    – dynamic obstacle check
  8. GOAL     – proximity check
"""
import sys, time, argparse, numpy as np

from config.settings         import Settings
from simulation.world        import MazeWorld
from simulation.robot        import Robot
from core.occupancy_grid     import OccupancyGrid
from core.slam               import SLAMEngine
from core.obstacle_avoidance import ObstacleAvoider
from planning.costmap        import CostMap
from planning.astar          import AStarPlanner
from sensors.lidar           import SimulatedLiDAR
from sensors.odometry        import WheelOdometry
from sensors.ekf             import ExtendedKalmanFilter
from visualization.renderer  import Renderer
from utils.logger            import get_logger

log = get_logger(__name__)
REACH_M = 0.12   # waypoint reached when robot within this radius


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maze",     default="default")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--goal",     default=None)
    p.add_argument("--speed",    type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Settings()
    log.info("AMR Navigation  |  DecodeLabs Project 3  |  2026")

    world  = MazeWorld(layout=args.maze, cfg=cfg)
    if args.goal:
        r,c = map(int, args.goal.split(",")); world.goal_cell=(r,c)
    goal_cell = world.goal_cell
    log.info(f"Maze={args.maze}  Start={world.start_pose[:2].round(3)}  Goal={goal_cell}")

    robot   = Robot(world.start_pose.copy(), cfg)
    lidar   = SimulatedLiDAR(world, cfg)
    odom    = WheelOdometry(cfg)
    ekf     = ExtendedKalmanFilter(cfg)
    grid    = OccupancyGrid(world.width_m, world.height_m, cfg.GRID_RESOLUTION)
    grid.seed_from_world(world.obstacle_map)
    slam    = SLAMEngine(grid, cfg)
    costmap = CostMap(grid, cfg)
    planner = AStarPlanner(costmap, cfg)
    avoider = ObstacleAvoider(cfg)
    renderer= Renderer(world, grid, costmap, cfg, headless=args.headless)

    path          = []
    replan_needed = True
    goal_reached  = False
    dt            = cfg.DT / max(args.speed, 0.1)
    step          = 0
    t0            = time.perf_counter()
    is_replan     = False

    while not goal_reached and step < cfg.MAX_STEPS:
        step   += 1
        is_replan = False

        # ── 1. PLAN ─────────────────────────────────────────────────
        if replan_needed:
            costmap.inflate()
            sc   = grid.world_to_cell(robot.pose[0], robot.pose[1])
            path = planner.plan(sc, goal_cell)
            replan_needed = False
            is_replan     = True
            if not path:
                log.warning(f"[s{step}] No path — replanning next step")
                robot.stop()
                # tiny advance to escape stuck start
                robot.pose[0] += 0.01
                replan_needed = True
                continue

        # ── 2. AVOID dynamic obstacles ───────────────────────────────
        scan_pre = lidar.scan(robot.pose)
        danger, reason = avoider.check(robot.pose, scan_pre, grid)
        if danger:
            robot.stop()
            replan_needed = True
            is_replan     = True
            log.debug(f"[s{step}] {reason}")
            continue

        # ── 3. CONTROL – follow path waypoint ───────────────────────
        if path:
            target = grid.cell_to_world(*path[0])
            robot.move_toward(target, dt)
            dx = robot.pose[0] - target[0]
            dy = robot.pose[1] - target[1]
            if (dx*dx + dy*dy) < REACH_M**2:
                path.pop(0)
                if not path:
                    replan_needed = True

        # ── 4. SENSE ─────────────────────────────────────────────────
        scan = lidar.scan(robot.pose)

        # ── 5. LOCALISE ──────────────────────────────────────────────
        raw_odom   = odom.update(robot.velocity, dt)
        robot.pose = ekf.update(robot.pose, raw_odom, scan)

        # ── 6. MAP ───────────────────────────────────────────────────
        slam.update(robot.pose, scan)
        world.step(dt)

        # ── 7. GOAL CHECK ────────────────────────────────────────────
        gx, gy = grid.cell_to_world(*goal_cell)
        if np.hypot(robot.pose[0]-gx, robot.pose[1]-gy) < cfg.GRID_RESOLUTION * 1.5:
            goal_reached = True
            log.info(f"✅ GOAL REACHED  step={step}  t={time.perf_counter()-t0:.2f}s")

        # ── 8. RENDER ────────────────────────────────────────────────
        renderer.draw(robot.pose, path, scan, step, is_replan, goal_reached, avoider.stats)

        time.sleep(max(0.0, dt * 0.2))

    renderer.draw(robot.pose, path, lidar.scan(robot.pose), step, False, goal_reached, avoider.stats)
    renderer.save("outputs/final_map.png")

    elapsed = time.perf_counter() - t0
    log.info(f"Done: steps={step}  t={elapsed:.2f}s  "
             f"coverage={grid.coverage_percent():.0f}%  {avoider.stats}")
    return 0 if goal_reached else 1


if __name__ == "__main__":
    sys.exit(main())
