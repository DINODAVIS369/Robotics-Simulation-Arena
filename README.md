# 2D Robot Simulation

This project is a small Python simulation for learning robot motion, collision handling, LiDAR sensing, and occupancy-grid mapping in a 2D environment.

The simulation uses Matplotlib to visualize a robot moving around a bounded map with walls and obstacles. The code is organized as a simple robotics playground rather than a full ROS stack or autonomous navigation framework.

## Features

- Differential-drive robot motion model
- Wall and obstacle collision detection
- Keyboard control with WASD keys
- LiDAR ray casting and point-cloud updates
- LiDAR-based occupancy-grid mapping
- Vectorized occupancy-grid ray updates to reduce CPU work during scans
- Noisy wheel-odometry pose estimation
- Correlative LiDAR scan matching
- Keyframe loop closure with pose-graph optimization
- Selectable A*, Dijkstra, Greedy Best-First, and Theta* grid planning
- Selectable RPP, Pure Pursuit, Stanley, PID, DWB, MPPI, MPC, and neural imitation path following
- Robot-radius obstacle inflation with additional clearance for cell resolution and curved tracking
- Behavior-tree navigation flow for goal check, planning, following, and recovery
- Live planned-route overlay and goal marker in the world view
- Real-time visualization in a Matplotlib window

## Project structure

- `main.py` — entry point for the simulation
- `Environment/Env_V_1.py` — world setup and collision logic
- `Robot/differential_drive.py` — robot kinematics for left/right wheel motion
- `Robot/circle_robot.py` — circular robot representation and drawing
- `Controller/keyboard_controller.py` — keyboard input handling
- `Controller/autonomous_controller.py` — autonomous goal-seeking wheel control
- `Controller/path_followers.py` — interchangeable path-following controllers
- `Controller/behavior_tree.py` — sequence/selector/condition/action nodes
- `Controller/grid_planner.py` — A* path planning on the occupancy grid
- `Engine/SimulationEngine.py` — update loop for motion, LiDAR, and map refresh
- `Robot/odometry.py` — noisy wheel-encoder pose estimate
- `Mapping/scan_matcher.py` — local scan-to-occupancy-grid alignment
- `Mapping/scan_geometry.py` — odometry-based scan deskew transforms
- `Mapping/loop_closure.py` — scan-keyframe revisit detection and correction
- `Mapping/pose_graph.py` — Gauss-Newton pose-graph optimization
- `Sensor/lidar.py` — LiDAR measurement logic and ray tracing
- `Mapping/occupancy_grid_map.py` — occupancy grid map generation
- `Geometry/world_geometry.py` — world geometry and collision primitives
- `test/test.py` — small test area / example usage

## Requirements

This project uses Python 3 and the following packages:

- `matplotlib`
- `numpy`

If they are not already installed in your environment, install them with:

```bash
python3 -m pip install matplotlib numpy
```

## Run the simulation

From the project root:

```bash
python3 main.py
```

The simulation window opens and starts the robot in the environment.

This project uses a single entry point in `main.py`. It starts the robot in autonomous mode with a goal at `(3, 9)`, builds a map from LiDAR scans, and replans a route through the occupancy grid.

Example flow inside `main.py`:

```python
from Environment.Env_V_1 import Environment
from Mapping.occupancy_grid_map import OccupancyGridMap
from Robot.differential_drive import DifferentialDriveRobot
from Controller.autonomous_controller import AutonomousController
from Sensor.lidar import Lidar
from Engine.SimulationEngine import SimulationEngine


def main():
    env = Environment()
    robot = DifferentialDriveRobot(x=1, y=1, theta=0)
    env.robot_radius = robot.radius

    occupancy_map = OccupancyGridMap(
        width=env.width,
        height=env.height,
        resolution=0.1,
    )
    occupancy_map.create_visualization()

    lidar = Lidar(
        geometry=env.geometry,
        rotation_hz=2.0,
        max_range=5.0,
        scan_resolution=360,
    )
    controller = AutonomousController(
        robot=robot,
        occupancy_grid_map=occupancy_map,
        goal=(3.0, 9.0),
    )

    engine = SimulationEngine(
        env=env,
        robot=robot,
        controller=controller,
        lidar=lidar,
        occupancy_grid_map=occupancy_map,
        dt=0.05,
    )

    engine.run()


if __name__ == "__main__":
    main()
```

## Controls

The robot starts in autonomous mode. Press `G` to switch between autonomous and manual control. In manual mode, use:

- `1` — A* (default)
- `2` — Dijkstra
- `3` — Greedy Best-First
- `4` — Theta*
- `F1` — Regulated Pure Pursuit (default)
- `F2` — Pure Pursuit
- `F3` — Stanley
- `F4` — PID
- `F5` — DWB (Dynamic Window approach)
- `F6` — MPPI (Model Predictive Path Integral)
- `F7` — MPC (Model Predictive Control)
- `F8` — neural imitation follower (small NumPy network trained to copy RPP)
- `G` — switch autonomous/manual mode
- `W` — move forward
- `S` — move backward
- `A` — rotate left
- `D` — rotate right
- `Space` — stop

Number keys `1`-`4` select the path planner; `F1`-`F8` select the path follower. Selections can be changed while driving autonomously. The current planner and follower are shown in the simulation title, and the route is replanned/redrawn after planner selection. The behavior tree checks for goal completion, ensures a route exists, follows it, and recovers from repeated collision-blocked movement: it backs away from the obstacle, turns in place so the rotating LiDAR can scan, and replans using the updated map. If the start remains blocked or no route is known, it repeats recovery with a longer retreat up to the configured attempt limit, then stops safely and waits for another scan. The robot cannot drive through walls or obstacles because each attempted motion is checked against the world geometry before updating the pose. The default autonomous goal is `(3, 9)`; change the `goal` argument in `main.py` to select another destination.

In the environment window, the A* route is drawn as an orange dashed line and
the goal is shown as a gold star. The route updates when the planner replans.

## Python implementations of ROS 2 navigation algorithms

This project does **not** install or run ROS 2 or Nav2. It implements comparable
algorithm families directly in Python:

- A*, Dijkstra, Greedy Best-First, and Theta* occupancy-grid planning
- Regulated Pure Pursuit, Pure Pursuit, Stanley, PID, DWB, MPPI, MPC, and an
  imitation-trained neural path follower
- A lightweight Python behavior tree for sequencing navigation tasks
- LiDAR scan matching and keyframe pose-graph loop closure, following concepts
  used by pose-graph SLAM systems such as SLAM Toolbox

These are independent educational Python implementations, not the ROS packages'
source code or bit-for-bit equivalents. The project has no ROS nodes, topics,
TF tree, lifecycle management, plugin interfaces, or ROS message transport.

## Neural follower study

The F8 controller is a small NumPy neural network trained to imitate the RPP
controller; it is an experiment in supervised learning, not a replacement for
SLAM or the path planner. To retrain the included model using the active system
Python and installed libraries, run:

```bash
uv run --no-project --python "$(command -v python3)" train_neural_follower.py
```

Training prints training and validation loss and saves the weights in
`Controller/neural_follower_weights.json`. See the neural follower section in
`LEARNING_GUIDE.md` for the inputs, outputs, and learning math.

## How it works

1. The environment creates a 2D map with walls and obstacles.
2. The robot is represented as a circular body with a heading angle.
3. The keyboard controller sends wheel velocities to the robot.
4. The simulation engine updates movement, then refreshes the LiDAR scan.
5. The occupancy grid stores free and occupied cells based on completed LiDAR scans, deskewed with estimated odometry.
6. Noisy wheel odometry predicts the pose; scan matching adjusts that estimate against the existing occupancy map.
7. The map integrates LiDAR rays at the estimated pose, not the simulator's hidden physical pose.
8. Loop closure checks for revisited keyframes and adds a constraint to the pose graph; Gauss-Newton optimization adjusts the trajectory before rebuilding the map.
9. The selected A*, Dijkstra, Greedy Best-First, or Theta* planner searches the map, with a higher cost for unexplored cells and obstacle inflation for the robot's radius.
10. The selected path follower converts the route and estimated pose into wheel commands. The F8 neural follower is a small NumPy network trained to imitate RPP.
11. A behavior tree sequences goal checking, path availability, route following, and collision recovery. Repeatedly blocked movement or a blocked start cell triggers a retreat, LiDAR scan turn, and route replanning.

This is a self-contained educational 2D SLAM and navigation implementation. It uses noisy wheel odometry, odometry-based scan deskewing, occupancy-grid scan matching, keyframe revisit detection, and nonlinear pose-graph optimization. It is not a production-grade SLAM system: loop closure and scan matching are simplified, and it lacks robust data association, sensor calibration, and the extensive recovery behaviors of a mature navigation stack.

The environment's true pose is shown only as a pose-error metric in the
simulation window; localization, mapping, loop closure, and navigation use
odometry and LiDAR estimates instead.

## Learn how it works

See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for a step-by-step explanation of
the architecture and the math behind differential-drive motion, LiDAR,
occupancy grids, A* planning, and route following, plus experiments to try.

## Notes

- The repository is intentionally lightweight and does not use ROS, Gazebo, or a game engine.
- The code is designed to be easy to read and extend for experiments with robot control, navigation, or SLAM concepts.
