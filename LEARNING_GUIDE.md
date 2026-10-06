# Learning Guide: Robot Navigation, Mapping, and the Math Behind It

This guide explains how the simulator is assembled and how its main equations
turn sensor measurements into wheel commands. Read the matching source files
while following the sections; changing small parameters and observing the
result is the best way to learn.

## 1. The program's data flow

Start at `main.py`. It creates and connects the simulator's objects:

1. `Environment` creates walls and checks collisions.
2. `DifferentialDriveRobot` stores the robot pose and wheel dimensions.
3. `OccupancyGridMap` stores what the LiDAR has observed.
4. `Lidar` measures distances from the robot to rectangles in the world.
5. `AutonomousController` plans a route and turns that route into wheel speeds.
6. `SimulationEngine` repeatedly updates motion, sensing, mapping, and control.

The simulation loop is a feedback loop:

```text
current robot pose -> LiDAR scan -> occupancy map -> path planner
        ^                                            |
        +----------- wheel commands <----------------+
```

In this project, the environment applies the wheel commands, and the resulting
pose is used by the next LiDAR update.

## 2. Coordinate systems and robot pose

The robot pose is `(x, y, theta)`:

- `x`, `y`: position in meters.
- `theta`: heading in radians, measured counterclockwise from the positive
  x-axis.
- `theta = 0` points right; `theta = pi / 2` points up.

Angles are in radians because Python's `sin`, `cos`, and `atan2` use radians.
One full turn is `2*pi` radians, or 360 degrees.

## 3. Differential-drive motion

The robot has a left and a right wheel. Each wheel command in this project is
angular velocity, in radians per second:

- `omega_l`: left wheel angular velocity.
- `omega_r`: right wheel angular velocity.
- `r`: wheel radius in meters.
- `b`: distance between the wheels (wheel base), in meters.

Convert wheel angular velocities to wheel-edge linear velocities:

```text
v_l = r * omega_l
v_r = r * omega_r
```

The robot's forward and turning velocities are:

```text
v     = (v_r + v_l) / 2
omega = (v_r - v_l) / b
```

For a small time step `dt`, the simulator updates its pose using:

```text
x_new     = x + v*cos(theta)*dt
y_new     = y + v*sin(theta)*dt
theta_new = theta + omega*dt
```

This is a simple Euler integration. It is easy to understand and works well
with a small time step, though more accurate integration can be used for larger
steps or fast turns.

Useful checks:

- Equal wheel speeds give `omega = 0`, so the robot drives straight.
- Equal and opposite wheel speeds give `v = 0`, so it spins in place.
- If the right wheel is faster, `omega > 0` and the robot turns left.

The environment tests the proposed center position against obstacles inflated
by the robot's radius. It rejects translation through a wall but still lets the
robot rotate in place.

Source: `Robot/differential_drive.py` and `Environment/Env_V_1.py`.

## 4. LiDAR ray geometry

Each LiDAR reading has an angle relative to the robot and a measured distance.
To turn it into a world coordinate, add the robot heading:

```text
world_angle = theta + lidar_angle
hit_x = robot_x + distance*cos(world_angle)
hit_y = robot_y + distance*sin(world_angle)
```

The LiDAR checks rays against the world's rectangles and returns the nearest
intersection, up to `max_range`. A ray that reaches its maximum range without
an intersection is considered a no-hit reading.

The sensor performs a rotating scan; a completed scan is a collection of
measurements rather than one instantaneous snapshot. Each measurement retains
the wheel-odometry pose estimate at the time it was taken. This supports scan
deskewing without exposing the simulator's true pose to the SLAM algorithm.

Source: `Sensor/lidar.py`.

## 5. Occupancy-grid mapping

The map divides the world into square cells. The cell size is the map
`resolution`; the current `main.py` uses 0.1 m per cell. Each grid value means:

```text
-1 = unknown
 0 = observed free
 1 = observed occupied
```

Convert a world coordinate to a cell with integer division:

```text
grid_x = floor(x / resolution)
grid_y = floor(y / resolution)
```

For each LiDAR ray, cells between the robot and the endpoint are marked free.
If the ray hit an obstacle, its endpoint cell is marked occupied. The map
keeps occupied cells from being overwritten as free.

This is occupancy mapping, one component of SLAM. The simulator's physical
pose is used only to simulate collision and generate LiDAR readings. The
mapping and navigation code receive wheel commands and LiDAR ranges, not the
physical pose.

Source: `Mapping/occupancy_grid_map.py`.

## 6. Noisy odometry and scan matching

Wheel odometry estimates movement by integrating wheel velocities. Because
real encoders are noisy, `NoisyOdometry` adds a small seeded random error to
each wheel reading. This makes the estimate drift over time even though the
simulated robot's physical motion remains deterministic.

Each beam has an odometry pose recorded when it was captured. Since the
scanner rotates over time, points are first deskewed into the first beam's
estimated robot frame. This corrects much of the distortion caused by the
robot moving during a scan. For a deskewed scan point `(px, py)`, a candidate
robot pose transforms it into map coordinates:

```text
map_x = x + px*cos(theta) - py*sin(theta)
map_y = y + px*sin(theta) + py*cos(theta)
```

`ScanMatcher` tests nearby candidate x, y, and heading values around the
odometry estimate. Candidate endpoints that land on occupied map cells receive
a better score than endpoints landing in observed-free cells. The best
supported candidate corrects the odometry pose; if the map has too little
evidence, odometry is kept.

This is a simple correlative scan-to-grid matcher. It is intentionally
understandable, but it can fail in repetitive or featureless spaces and is not
a substitute for a probabilistic estimator or a production scan matcher.

Source: `Robot/odometry.py` and `Mapping/scan_matcher.py`.

## 7. Loop closure

The system retains a keyframe scan when the estimated robot has moved far
enough. When the estimated position later returns near an old keyframe, the
loop-closure module compares the current scan's transformed points with that
keyframe's point cloud. A sufficiently close match is treated as a revisit.

The detected revisit adds a relative-pose constraint between the current scan
and its older keyframe. The `PoseGraph` also stores wheel-odometry constraints
between consecutive scans. A Gauss-Newton least-squares optimizer adjusts the
poses while anchoring the first pose to remove global-coordinate ambiguity.
After optimization, saved scans are replayed into the occupancy map using
corrected poses. This demonstrates the main loop-closure idea: a revisit
provides evidence that accumulated odometry drift should be reduced.

This is an educational pose-graph optimizer, not a reimplementation of the
complete SLAM Toolbox/OpenKarto stack. It omits several production techniques,
including robust feature association and a sparse nonlinear solver.

Source: `Mapping/loop_closure.py` and `Engine/SimulationEngine.py`.

## 8. Grid path-planning algorithms

`Controller/grid_planner.py` can search neighboring grid cells using four
algorithms. All use the same obstacle inflation, unknown-cell costs, and
no-corner-cutting rules:

```text
priority(cell) = g(cell) + h(cell)    # A*
priority(cell) = g(cell)              # Dijkstra
priority(cell) = h(cell)              # Greedy Best-First
```

- `g`: cost of the path found so far.
- `h`: estimated remaining distance to the goal.
- **A\*** balances known path cost and estimated remaining cost. With the
  current octile heuristic, it finds an optimal grid path for the defined
  movement and cell costs.
- **Dijkstra** expands the lowest-cost known route first; it finds optimal
  paths but usually explores more cells because it has no goal heuristic.
- **Greedy Best-First** chooses cells that appear closest to the goal. It can
  be quick, but does not guarantee the lowest-cost path.
- **Theta\*** extends A* by allowing a cell to connect to a visible ancestor,
  producing shorter any-angle routes rather than only grid-edge routes.

For A*, Dijkstra, and Greedy, movement cost `g` is the step length multiplied
by the average unknown/free-cell penalty. Theta* uses a line-of-sight segment
cost and checks that the segment does not pass through blocked cells.

The planner allows horizontal, vertical, and diagonal steps. A straight step
costs 1 cell; a diagonal costs `sqrt(2)` cells. The octile heuristic used by
the code is:

```text
h = max(dx, dy) + (sqrt(2) - 1)*min(dx, dy)
```

Occupied cells are blocked. Obstacles are expanded by the robot's radius so a
route leaves enough room for the circular robot. Unknown cells can be crossed,
but have an extra cost so a route through already-observed free space is
preferred when available. Diagonal moves are rejected if either neighboring
side cell is blocked, preventing corner-cutting through walls.

The controller replans when a new LiDAR scan has been added to the map.
Press `1`, `2`, `3`, or `4` to select A*, Dijkstra, Greedy Best-First, or
Theta* while the program is running. The title bar shows the selected planner.

Source: `Controller/grid_planner.py`.

## 9. Comparing path followers and local controllers

The path planner answers “which route should the robot take?” A path follower
or local controller answers “which wheel velocities should it use now?”
Press `F1`-`F7` to select one of these Python implementations:

- **Regulated Pure Pursuit (F1):** geometric lookahead steering with speed
  reduced for curvature and distance to the goal.
- **Pure Pursuit (F2):** the basic geometric lookahead method, without RPP's
  curvature speed limit.
- **Stanley (F3):** combines path-heading error with cross-track error.
- **PID (F4):** applies proportional, integral, and derivative steering to
  heading and cross-track error.
- **DWB (F5):** samples short constant-velocity commands, simulates each
  trajectory, and scores progress, path distance, and collisions.
- **MPPI (F6):** samples noisy control sequences and uses cost-weighted
  averaging to choose the next velocity.
- **MPC (F7):** searches a short horizon of candidate control sequences and
  applies the first command from the best predicted sequence.

The DWB, MPPI, and MPC implementations here are intentionally compact
educational versions; they do not reproduce every Nav2 critic, optimizer, or
parameter.

## 10. Following the route with wheel commands

The controller applies the Regulated Pure Pursuit idea: choose a lookahead
point on the planned path, compute the curvature needed to reach it, then
reduce speed for sharp curvature and as the robot approaches the goal.

Transform the lookahead point into the robot frame:

```text
x_local =  cos(theta)*dx + sin(theta)*dy
y_local = -sin(theta)*dx + cos(theta)*dy
curvature = 2*y_local / (x_local^2 + y_local^2)
```

Pure Pursuit requests angular speed `omega = v*curvature`. Regulated Pure
Pursuit also limits forward speed based on curvature and goal distance:

```text
curvature_speed = sqrt(max_lateral_acceleration / abs(curvature))
goal_speed = sqrt(2 * max_lateral_acceleration * distance_to_goal)
v = min(max_speed, curvature_speed, goal_speed)
```

The forward speed is reduced while the robot is turning sharply. Given desired
forward velocity `v` and angular velocity `omega`, convert back to wheel
velocities:

```text
v_l = v - omega*b/2
v_r = v + omega*b/2
omega_l = v_l/r
omega_r = v_r/r
```

Those wheel angular velocities are passed to the existing motion model. The
controller stops when it reaches the final waypoint. Press `G` to switch
between autonomous mode and WASD manual control.

This is a compact Regulated Pure Pursuit-style controller; it does not yet
include every Nav2 option such as costmap-based obstacle speed regulation,
rotate-to-heading, or collision projection.

Source: `Controller/autonomous_controller.py`.

The world visualization draws the complete A* route as an orange dashed line
and marks the goal with a gold star. The route line is updated when the
controller calculates a new path.

## 11. How to build a similar system yourself

Build and check one layer at a time:

1. **World:** represent walls as rectangles and draw them. Check point or
   circle collision against each rectangle.
2. **Robot:** store `(x, y, theta)`, wheel radius, and wheel base. Test straight
   motion and spinning in place before adding a sensor.
3. **Sensor:** implement a ray/rectangle intersection and verify known cases:
   a ray pointing at a wall, away from it, and parallel to it.
4. **Map:** convert a few known world coordinates to cells. Trace a ray and
   check that only cells before the endpoint become free.
5. **Planner:** first plan across an empty grid; then add a wall and check that
   the path goes around it; finally test blocked or out-of-map goals.
6. **Controller:** follow a single waypoint first. Add turning, speed limits,
   and waypoint progression after that works.
7. **Engine:** run the pieces in a consistent order and check the updated map
   and controller output on each simulation step.

Keep small tests for math-heavy behavior. For example, a robot facing right
with both wheel velocities equal should increase `x`, leave `y` unchanged, and
keep the same heading.

## 12. Behavior-tree navigation

The top-level autonomous controller is organized as a behavior tree rather
than a single long chain of conditions. A **Sequence** succeeds only when each
child succeeds; a **Selector** tries children until one succeeds or remains
running. Nodes return `SUCCESS`, `FAILURE`, or `RUNNING`.

The navigation tree has three branches:

1. If the goal is reached, stop the wheels and report success.
2. If a route is available, run the selected follower. Following returns
   `RUNNING` because navigation continues over many simulation updates.
3. If no route is available, stop and wait for new map data before planning
   again.

This is a small Python behavior-tree implementation inspired by the task
sequencing style used in Nav2. It does not use Nav2's BT Navigator, XML trees,
plugins, ROS actions, or recovery plugins.

Source: `Controller/behavior_tree.py` and
`Controller/autonomous_controller.py`.

## 13. Experiments to try

- Change the goal passed in `main.py`; ensure it is in a reachable free part of
  the map.
- Change `resolution` in `main.py`. Finer cells give more detail but increase
  search and mapping work.
- Change `unknown_cost` in `GridPlanner`. A higher value favors known free
  routes; a lower value makes the planner more willing to explore.
- Change `linear_speed` and `angular_speed`. Observe how speed and turning
  affect waypoint tracking.
- Increase `wheel_noise` in `NoisyOdometry` to see drift increase, then compare
  how often scan matching corrects it.
- Follow a route that revisits an earlier area and watch the loop-closure
  count and map correction.
- Try the same planner route with each `F1`-`F7` follower. Compare corner
  cutting, smoothness, collision behavior, and time-to-goal.
- Temporarily set a wheel radius or wheel base incorrectly. Notice the effect
  on motion and why consistent units matter.

When an experiment behaves unexpectedly, inspect one value at a time: the
current pose, selected waypoint, heading error, wheel commands, and the
occupancy values around the robot.
