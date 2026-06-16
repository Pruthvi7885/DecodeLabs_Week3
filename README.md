# Project 3: Autonomous Mobile Robot (AMR) Navigation
**DecodeLabs | Robotics & Automation | Batch 2026**

---

## What This Project Does

A fully working Python simulation of an Autonomous Mobile Robot that:

| PDF Task | What's Implemented | File |
|---|---|---|
| **Task 1** – Process LiDAR → 2D Occupancy Grid | Bresenham ray-casting SLAM, 3-layer tensor map | `core/occupancy_grid.py`, `core/slam.py` |
| **Task 2** – A* Pathfinding (shortest route) | f(n)=g(n)+h(n), Manhattan heuristic, EDT inflation | `planning/astar.py`, `planning/costmap.py` |
| **Task 3** – Obstacle avoidance (stop/re-route) | Dynamic-only detection via world simulation query | `core/obstacle_avoidance.py` |
| **EKF Sensor Fusion** | Fuses odometry + LiDAR correction | `sensors/ekf.py` |
| **SLAM** | Occupancy-grid SLAM with Bresenham rays | `core/slam.py` |

---

## Project Structure

```
amr_navigation/
├── main.py                    ← Entry point (run this)
├── requirements.txt
├── .vscode/
│   ├── launch.json            ← VS Code run/debug configs
│   └── settings.json
│
├── config/
│   └── settings.py            ← All tunable constants
│
├── core/                      ← CORE ALGORITHMS
│   ├── occupancy_grid.py      ← TASK 1: 2D Occupancy Grid
│   ├── slam.py                ← SLAM (Bresenham ray-casting)
│   └── obstacle_avoidance.py  ← TASK 3: Dynamic Obstacle Avoidance
│
├── planning/                  ← PLANNING
│   ├── astar.py               ← TASK 2: A* Pathfinding
│   └── costmap.py             ← Inflation Layer (scipy EDT)
│
├── sensors/                   ← SENSORS
│   ├── lidar.py               ← Simulated LiDAR (vectorised ray-cast)
│   ├── odometry.py            ← Wheel Odometry (with drift)
│   └── ekf.py                 ← Extended Kalman Filter
│
├── simulation/                ← ENVIRONMENT
│   ├── world.py               ← Maze + Dynamic Obstacles
│   └── robot.py               ← Differential Drive Robot
│
├── visualization/
│   └── renderer.py            ← 3-panel matplotlib renderer
│
├── tests/
│   └── test_all.py            ← 35 unit + integration tests
│
└── outputs/
    └── final_map.png          ← Saved after every run
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the simulation (live visualisation)
python main.py

# 3. Headless fast run (saves map to outputs/)
python main.py --headless --speed 10

# 4. Complex maze
python main.py --maze complex

# 5. Open field
python main.py --maze open

# 6. Run all 35 tests
python tests/test_all.py
```

---

## VS Code

Open the `amr_navigation/` folder in VS Code. Use **Run & Debug** (Ctrl+Shift+D):

| Config | What it does |
|---|---|
| ▶ Run AMR (Default Maze) | Live simulation with matplotlib window |
| ▶ Run AMR (Complex Maze) | Harder layout |
| ▶ Run AMR Headless (10x speed) | Fast, saves `outputs/final_map.png` |
| 🧪 Run All Tests | Runs all 35 tests, prints pass/fail |

---

## Algorithm Details

### Task 1 — 2D Occupancy Grid + SLAM
- 3-layer numpy int16 tensor `[occupancy, hit_count, miss_count]`
- Cell values: **-1** unknown · **0** free · **100** obstacle
- Updated via **Bresenham ray-casting**: every LiDAR beam marks free cells along its path and marks the endpoint as obstacle
- `seed_from_world()` pre-populates from ground truth for immediate planning

### Task 2 — A* Pathfinding
- **f(n) = g(n) + h(n)**
- **g(n)** = exact accumulated cost from start
- **h(n)** = Manhattan distance (admissible on 4-connected grid)
- `heapq` priority queue — O(log n) per operation
- Numpy boolean closed-set — O(1) membership check
- **EDT Inflation Layer** (scipy): artificially inflates cost near walls so A* automatically routes through corridor centres

### Task 3 — Dynamic Obstacle Avoidance
- **STOP** (< 0.25 m) → emergency halt + replan
- **WARN** (< 0.40 m) → replan
- Only reacts to **true dynamic obstacles**: world simulation says obstacle exists BUT occupancy grid does not know about it yet → confirms it is a moving object, not a mapped wall

### Sensor Fusion — EKF
- State vector: **[x, y, θ]**
- Fuses wheel odometry (drifting) with LiDAR scan-matching correction
- Produces smooth `odom/filtered` estimate

### Navigation Loop Order
```
PLAN → AVOID → CONTROL → SENSE → LOCALISE → MAP → GOAL-CHECK
```
This order ensures the robot uses the PREVIOUS step's velocity for odometry (correct physics).

---

## Test Results

```
35/35 tests pass

Tasks covered:
  ✅ Occupancy Grid (9 tests)
  ✅ SLAM Engine (3 tests)
  ✅ CostMap Inflation (3 tests)
  ✅ A* Pathfinding (6 tests)
  ✅ Obstacle Avoidance (6 tests)
  ✅ Sensors: LiDAR, Odometry, EKF (5 tests)
  ✅ Robot kinematics (2 tests)
  ✅ Full integration pipeline (1 test)
```
