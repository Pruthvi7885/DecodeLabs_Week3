"""
tests/test_all.py
═══════════════════════════════════════════════════════════════════
Complete test suite for AMR Navigation — Project 3.

Run with:  python -m pytest tests/ -v
       or: python tests/test_all.py  (no pytest needed)
═══════════════════════════════════════════════════════════════════
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Minimal test framework (works without pytest)
# ══════════════════════════════════════════════════════════════════

_RESULTS = []

def _test(name, fn):
    try:
        fn()
        _RESULTS.append(("PASS", name))
        print(f"  ✅  {name}")
    except Exception as e:
        _RESULTS.append(("FAIL", name, str(e)))
        print(f"  ❌  {name}  →  {e}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "Expected True")

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(msg or f"{a!r} != {b!r}")

def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(msg or f"|{a} - {b}| > {tol}")


# ══════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════

from config.settings import Settings
from core.occupancy_grid import OccupancyGrid
from core.slam import SLAMEngine
from core.obstacle_avoidance import ObstacleAvoider
from planning.costmap import CostMap
from planning.astar import AStarPlanner
from sensors.lidar import SimulatedLiDAR
from sensors.odometry import WheelOdometry
from sensors.ekf import ExtendedKalmanFilter
from simulation.world import MazeWorld
from simulation.robot import Robot


def _cfg():
    return Settings()


def _open_grid(cfg):
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    g._tensor[0, :, :] = 0   # all free
    return g


# ══════════════════════════════════════════════════════════════════
# TASK 1 — Occupancy Grid
# ══════════════════════════════════════════════════════════════════

print("\n── TASK 1: Occupancy Grid ──────────────────────────────────")

def t_initial_unknown():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    assert_equal(g.data[5, 5], -1)
_test("Initial cells are UNKNOWN (-1)", t_initial_unknown)

def t_mark_obstacle():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    g.mark_obstacle(5, 5)
    assert_true(g.is_obstacle(5, 5))
    assert_equal(g.data[5, 5], 100)
_test("mark_obstacle sets cell to 100", t_mark_obstacle)

def t_mark_free():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    g.mark_free(3, 3)
    assert_true(g.is_free(3, 3))
    assert_equal(g.data[3, 3], 0)
_test("mark_free sets cell to 0", t_mark_free)

def t_obstacle_not_overwritten():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    g.mark_obstacle(7, 7)
    g.mark_free(7, 7)
    assert_true(g.is_obstacle(7, 7), "Obstacle should survive free update")
_test("Obstacle cell cannot be overwritten by free", t_obstacle_not_overwritten)

def t_world_to_cell():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    r, c = g.world_to_cell(0.15, 0.15)
    assert_equal(r, 1); assert_equal(c, 1)
_test("world_to_cell correct conversion", t_world_to_cell)

def t_cell_to_world_centre():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    x, y = g.cell_to_world(0, 0)
    assert_close(x, cfg.GRID_RESOLUTION / 2)
    assert_close(y, cfg.GRID_RESOLUTION / 2)
_test("cell_to_world returns cell centre", t_cell_to_world_centre)

def t_out_of_bounds_is_obstacle():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    assert_true(g.is_obstacle(-1, -1))
    assert_true(g.is_obstacle(9999, 9999))
_test("Out-of-bounds treated as obstacle", t_out_of_bounds_is_obstacle)

def t_coverage_increases():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    before = g.coverage_percent()
    g.mark_free(0, 0); g.mark_obstacle(1, 1)
    assert_true(g.coverage_percent() > before)
_test("Coverage increases when cells are marked", t_coverage_increases)

def t_seed_from_world():
    cfg = _cfg()
    g = OccupancyGrid(2.0, 2.0, cfg.GRID_RESOLUTION)
    bmap = np.zeros((g.rows, g.cols), dtype=bool)
    bmap[5, 5] = True
    g.seed_from_world(bmap)
    assert_true(g.is_obstacle(5, 5))
    assert_true(g.is_free(0, 0))
_test("seed_from_world populates grid correctly", t_seed_from_world)


# ══════════════════════════════════════════════════════════════════
# SLAM Engine
# ══════════════════════════════════════════════════════════════════

print("\n── SLAM Engine ─────────────────────────────────────────────")

def t_slam_increases_coverage():
    cfg = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    grid  = OccupancyGrid(world.width_m, world.height_m, cfg.GRID_RESOLUTION)
    slam  = SLAMEngine(grid=grid, cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    scan  = lidar.scan(np.array([0.5, 0.5, 0.0]))
    slam.update(np.array([0.5, 0.5, 0.0]), scan)
    assert_true(grid.coverage_percent() > 0)
_test("Single SLAM update increases coverage", t_slam_increases_coverage)

def t_slam_multi_angle():
    cfg = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    grid  = OccupancyGrid(world.width_m, world.height_m, cfg.GRID_RESOLUTION)
    slam  = SLAMEngine(grid=grid, cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    for angle in [0, np.pi/2, np.pi, -np.pi/2]:
        pose = np.array([0.5, 0.5, angle])
        slam.update(pose, lidar.scan(pose))
    assert_true(grid.coverage_percent() > 5.0, f"Only {grid.coverage_percent():.1f}% covered")
_test("Multi-angle SLAM covers > 5% of map", t_slam_multi_angle)

def t_slam_marks_obstacles():
    cfg = _cfg()
    world = MazeWorld(layout="default", cfg=cfg)
    grid  = OccupancyGrid(world.width_m, world.height_m, cfg.GRID_RESOLUTION)
    slam  = SLAMEngine(grid=grid, cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
        pose = np.array([0.5, 0.5, a])
        slam.update(pose, lidar.scan(pose))
    obs_count = int(np.sum(grid.data == 100))
    assert_true(obs_count > 0, "No obstacles mapped")
_test("SLAM marks obstacle cells", t_slam_marks_obstacles)


# ══════════════════════════════════════════════════════════════════
# TASK 2 — CostMap
# ══════════════════════════════════════════════════════════════════

print("\n── TASK 2a: CostMap Inflation ──────────────────────────────")

def t_inflation_adjacent():
    cfg = _cfg()
    g = _open_grid(cfg)
    g.mark_obstacle(10, 10)
    cm = CostMap(g, cfg)
    cm.inflate()
    assert_true(cm.cost(10, 11) > 0, "Adjacent cell should be inflated")
    assert_true(cm.cost(9,  10) > 0, "Adjacent cell should be inflated")
_test("Inflation raises cost of cells near obstacle", t_inflation_adjacent)

def t_inflation_obstacle_100():
    cfg = _cfg()
    g = _open_grid(cfg)
    g.mark_obstacle(5, 5)
    cm = CostMap(g, cfg)
    cm.inflate()
    assert_close(cm.cost(5, 5), 100.0)
_test("Obstacle cell keeps cost = 100 after inflation", t_inflation_obstacle_100)

def t_inflation_decays_with_distance():
    cfg = _cfg()
    g = _open_grid(cfg)
    g.mark_obstacle(10, 10)
    cm = CostMap(g, cfg)
    cm.inflate()
    c1 = cm.cost(10, 11)  # 1 cell away
    c2 = cm.cost(10, 12)  # 2 cells away
    assert_true(c1 >= c2, f"Cost should decay: {c1:.1f} vs {c2:.1f}")
_test("Inflation cost decays with distance", t_inflation_decays_with_distance)


# ══════════════════════════════════════════════════════════════════
# TASK 2 — A* Planner
# ══════════════════════════════════════════════════════════════════

print("\n── TASK 2b: A* Pathfinding ─────────────────────────────────")

def t_astar_open_grid():
    cfg = _cfg()
    g   = _open_grid(cfg)
    cm  = CostMap(g, cfg)
    pl  = AStarPlanner(cm, cfg)
    path = pl.plan((0, 0), (15, 15))
    assert_true(len(path) > 0)
    assert_equal(path[-1], (15, 15))
_test("A* finds path in open grid", t_astar_open_grid)

def t_astar_adjacent():
    cfg = _cfg()
    g   = _open_grid(cfg)
    cm  = CostMap(g, cfg)
    pl  = AStarPlanner(cm, cfg)
    path = pl.plan((5, 5), (5, 6))
    assert_equal(len(path), 1)
    assert_equal(path[0], (5, 6))
_test("A* returns 1-cell path to adjacent cell", t_astar_adjacent)

def t_astar_blocked():
    cfg = _cfg()
    g   = _open_grid(cfg)
    g._tensor[0, 5, :] = 100   # wall across full row 5
    cm  = CostMap(g, cfg)
    pl  = AStarPlanner(cm, cfg)
    path = pl.plan((0, 5), (10, 5))
    assert_equal(path, [])
_test("A* returns empty path when route is fully blocked", t_astar_blocked)

def t_astar_manhattan():
    assert_equal(int(AStarPlanner.heuristic((0, 0), (3, 4))), 7)
    assert_equal(int(AStarPlanner.heuristic((5, 5), (5, 5))), 0)
_test("Manhattan heuristic correct values", t_astar_manhattan)

def t_astar_avoids_obstacle():
    cfg = _cfg()
    g   = _open_grid(cfg)
    g.mark_obstacle(5, 8)
    cm  = CostMap(g, cfg)
    pl  = AStarPlanner(cm, cfg)
    path = pl.plan((0, 0), (10, 10))
    hit = any(cell == (5, 8) for cell in path)
    assert_true(not hit, "Path should not pass through obstacle cell")
_test("A* path avoids obstacle cells", t_astar_avoids_obstacle)

def t_astar_path_is_connected():
    cfg = _cfg()
    g   = _open_grid(cfg)
    cm  = CostMap(g, cfg)
    pl  = AStarPlanner(cm, cfg)
    path = pl.plan((0, 0), (10, 10))
    assert_true(len(path) > 1)
    for i in range(len(path)-1):
        r0,c0 = path[i]; r1,c1 = path[i+1]
        step = abs(r1-r0) + abs(c1-c0)
        assert_true(step <= 2, f"Disconnected path at index {i}")
_test("A* path cells are connected (no teleport)", t_astar_path_is_connected)


# ══════════════════════════════════════════════════════════════════
# TASK 3 — Obstacle Avoidance
# ══════════════════════════════════════════════════════════════════

print("\n── TASK 3: Obstacle Avoidance ──────────────────────────────")

def _make_scan(cfg, dist):
    return {
        "ranges": np.full(cfg.LIDAR_NUM_BEAMS, dist, dtype=np.float32),
        "angles": np.linspace(-np.pi/2, np.pi/2, cfg.LIDAR_NUM_BEAMS, dtype=np.float32),
    }

def t_av_safe_maxrange():
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    world.dynamics = []
    g   = _open_grid(cfg)
    av  = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, cfg.LIDAR_MAX_RANGE)
    d, _ = av.check(np.array([0.5,0.5,0.0]), scan, g)
    assert_true(not d)
_test("No danger when all beams at max range", t_av_safe_maxrange)

def t_av_stop():
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    from simulation.world import DynamicObstacle
    # Place obstacle directly ahead at 0.20m (front beams angle~0, endpoint x=0.70)
    world.dynamics = [DynamicObstacle(x=0.70, y=0.50, vx=0.0, vy=0.0, r=0.15)]
    g = _open_grid(cfg)
    av = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, 0.20)
    d, reason = av.check(np.array([0.5,0.5,0.0]), scan, g)
    assert_true(d and reason == "STOP", f"Expected STOP got danger={d} reason={reason!r}")
_test("STOP triggered by very close dynamic obstacle", t_av_stop)

def t_av_warn():
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    from simulation.world import DynamicObstacle
    world.dynamics = [DynamicObstacle(x=0.85, y=0.5, vx=0.0, vy=0.0, r=0.10)]
    g = _open_grid(cfg)
    av = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, 0.38)
    d, reason = av.check(np.array([0.5,0.5,0.0]), scan, g)
    assert_true(d and reason == "WARN", f"Expected WARN, got danger={d} reason={reason!r}")
_test("WARN triggered by medium-range dynamic obstacle", t_av_warn)

def t_av_static_wall_ignored():
    """Known static walls should NOT trigger avoidance."""
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    world.dynamics = []   # no dynamic obstacles at all
    g   = _open_grid(cfg)
    # Mark every cell as obstacle (fully known map)
    g._tensor[0,:,:] = 100
    av  = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, 0.20)
    # All beams hit known walls + no dynamic obstacles => no danger
    d, _ = av.check(np.array([0.5,0.5,0.0]), scan, g)
    assert_true(not d, f"No dynamic obs => no danger, got {d}")
_test("Static known walls do NOT trigger avoidance", t_av_static_wall_ignored)

def t_av_sectors():
    cfg = _cfg()
    world = MazeWorld(layout="open", cfg=cfg); world.dynamics=[]
    av  = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, 1.0)
    s = av.sector_analysis(scan)
    assert_equal(set(s.keys()), {"LEFT", "FRONT", "RIGHT"})
_test("sector_analysis returns LEFT/FRONT/RIGHT keys", t_av_sectors)

def t_av_stats_increment():
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    from simulation.world import DynamicObstacle
    world.dynamics = [DynamicObstacle(x=0.55, y=0.5, vx=0.0, vy=0.0, r=0.12)]
    g = _open_grid(cfg)
    av = ObstacleAvoider(cfg); av.set_world(world)
    scan = _make_scan(cfg, 0.20)
    av.check(np.array([0.5,0.5,0.0]), scan, g)
    av.check(np.array([0.5,0.5,0.0]), scan, g)
    assert_equal(av.stats["stop_events"], 2)
_test("Stop event counter increments correctly", t_av_stats_increment)


# ══════════════════════════════════════════════════════════════════
# Sensors
# ══════════════════════════════════════════════════════════════════

print("\n── Sensors ─────────────────────────────────────────────────")

def t_lidar_shape():
    cfg = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    scan  = lidar.scan(np.array([0.5, 0.5, 0.0]))
    assert_equal(scan["ranges"].shape, (cfg.LIDAR_NUM_BEAMS,))
    assert_equal(scan["angles"].shape, (cfg.LIDAR_NUM_BEAMS,))
_test("LiDAR scan output shape correct", t_lidar_shape)

def t_lidar_max_range():
    cfg = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    scan  = lidar.scan(np.array([0.5, 0.5, 0.0]))
    assert_true(np.all(scan["ranges"] <= cfg.LIDAR_MAX_RANGE + 0.15))
_test("LiDAR ranges never exceed max range + noise", t_lidar_max_range)

def t_lidar_detects_wall():
    cfg = _cfg()
    world = MazeWorld(layout="default", cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    scan  = lidar.scan(np.array([0.5, 0.5, 0.0]))
    short = np.any(scan["ranges"] < cfg.LIDAR_MAX_RANGE * 0.9)
    assert_true(short, "Expected some beams to hit walls")
_test("LiDAR detects walls (short beams in maze)", t_lidar_detects_wall)

def t_odom_drift():
    cfg = _cfg()
    od  = WheelOdometry(cfg=cfg)
    for _ in range(100):
        p = od.update(np.array([0.3, 0.1]), dt=0.05)
    assert_true(p[0] != 0.0 or p[1] != 0.0)
_test("Odometry accumulates movement", t_odom_drift)

def t_ekf_finite():
    cfg   = _cfg()
    world = MazeWorld(layout="open", cfg=cfg)
    lidar = SimulatedLiDAR(world=world, cfg=cfg)
    od    = WheelOdometry(cfg=cfg)
    ekf   = ExtendedKalmanFilter(cfg=cfg)
    pose  = np.array([0.5, 0.5, 0.0])
    for _ in range(20):
        scan = lidar.scan(pose)
        raw  = od.update(np.array([0.2, 0.0]), dt=0.05)
        pose = ekf.update(prior_pose=pose, odom=raw, scan=scan)
    assert_true(np.all(np.isfinite(pose)), "EKF produced non-finite pose")
_test("EKF produces finite pose after 20 steps", t_ekf_finite)


# ══════════════════════════════════════════════════════════════════
# Robot
# ══════════════════════════════════════════════════════════════════

print("\n── Robot ───────────────────────────────────────────────────")

def t_robot_moves():
    cfg   = _cfg()
    robot = Robot(np.array([0.0, 0.0, 0.0]), cfg)
    for _ in range(30):
        robot.move_toward((1.0, 0.0), dt=0.05)
    assert_true(robot.pose[0] > 0.3, f"Robot didn't move: x={robot.pose[0]:.3f}")
_test("Robot moves toward target", t_robot_moves)

def t_robot_stop():
    cfg   = _cfg()
    robot = Robot(np.array([0.5, 0.5, 0.0]), cfg)
    robot.velocity = np.array([0.5, 0.2])
    robot.stop()
    assert_true(np.allclose(robot.velocity, 0.0))
_test("Robot stops on command", t_robot_stop)


# ══════════════════════════════════════════════════════════════════
# Integration — full pipeline
# ══════════════════════════════════════════════════════════════════

print("\n── Integration Test ────────────────────────────────────────")

def t_full_pipeline():
    cfg     = Settings()
    world   = MazeWorld(layout="open", cfg=cfg)
    grid    = OccupancyGrid(world.width_m, world.height_m, cfg.GRID_RESOLUTION)
    grid.seed_from_world(world.obstacle_map)
    slam    = SLAMEngine(grid=grid, cfg=cfg)
    lidar   = SimulatedLiDAR(world=world, cfg=cfg)
    od      = WheelOdometry(cfg=cfg)
    ekf     = ExtendedKalmanFilter(cfg=cfg)
    costmap = CostMap(grid=grid, cfg=cfg)
    planner = AStarPlanner(costmap=costmap, cfg=cfg)
    world_i = MazeWorld(layout='open', cfg=cfg)
    avoider = ObstacleAvoider(cfg=cfg); avoider.set_world(world_i)
    robot   = Robot(world.start_pose.copy(), cfg=cfg)

    for _ in range(30):
        scan = lidar.scan(robot.pose)
        raw  = od.update(np.array([0.1, 0.0]), dt=cfg.DT)
        robot.pose = ekf.update(prior_pose=robot.pose, odom=raw, scan=scan)
        slam.update(pose=robot.pose, scan=scan)

    costmap.inflate()
    sc   = grid.world_to_cell(robot.pose[0], robot.pose[1])
    path = planner.plan(sc, world.goal_cell)
    scan = lidar.scan(robot.pose)
    d, _ = avoider.check(robot.pose, scan, grid)

    assert_true(isinstance(path, list))
    assert_true(len(path) > 0, "No path found in open maze")
    assert_true(isinstance(d, bool))
_test("Full sense→SLAM→plan→avoid pipeline runs without crash", t_full_pipeline)


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

print()
passed = sum(1 for r in _RESULTS if r[0] == "PASS")
failed = sum(1 for r in _RESULTS if r[0] == "FAIL")
total  = len(_RESULTS)
print(f"{'='*55}")
print(f"  Results: {passed}/{total} passed  |  {failed} failed")
print(f"{'='*55}")
if failed > 0:
    print("\nFailed tests:")
    for r in _RESULTS:
        if r[0] == "FAIL":
            print(f"  ❌  {r[1]}: {r[2]}")
    sys.exit(1)
sys.exit(0)
