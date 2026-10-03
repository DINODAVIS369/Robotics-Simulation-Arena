# 2D Robot Simulation

This project is a small Python simulation for learning robot motion, collision handling, LiDAR sensing, and occupancy-grid mapping in a 2D environment.

The simulation uses Matplotlib to visualize a robot moving around a bounded map with walls and obstacles. The code is organized as a simple robotics playground rather than a full ROS stack or autonomous navigation framework.

## Features

- Differential-drive robot motion model
- Wall and obstacle collision detection
- Keyboard control with WASD keys
- LiDAR ray casting and point-cloud updates
- Occupancy grid mapping from sensor data
- Real-time visualization in a Matplotlib window

## Project structure

- `main.py` — entry point for the simulation
- `Environment/Env_V_1.py` — world setup and collision logic
- `Robot/differential_drive.py` — robot kinematics for left/right wheel motion
- `Robot/circle_robot.py` — circular robot representation and drawing
- `Controller/keyboard_controller.py` — keyboard input handling
- `Engine/SimulationEngine.py` — update loop for motion, LiDAR, and map refresh
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

This project uses a single entry point in `main.py`. That file creates the environment, robot, LiDAR, occupancy grid, and simulation engine in the correct order so the app starts correctly.

Example flow inside `main.py`:

```python
from Environment.Env_V_1 import Environment
from Mapping.occupancy_grid_map import OccupancyGridMap
from Robot.differential_drive import DifferentialDriveRobot
from Controller.keyboard_controller import KeyboardController
from Sensor.lidar import Lidar
from Engine.SimulationEngine import SimulationEngine


def main():
    env = Environment()
    robot = DifferentialDriveRobot(x=1, y=1, theta=0)
    env.robot_radius = robot.radius

    controller = KeyboardController(robot)
    lidar = Lidar(
        geometry=env.geometry,
        rotation_hz=2.0,
        max_range=5.0,
        scan_resolution=360,
    )

    occupancy_map = OccupancyGridMap(
        width=env.width,
        height=env.height,
        resolution=0.1,
    )
    occupancy_map.create_visualization()

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

Use the Matplotlib window and press:

- `W` / `Up` — move forward
- `S` / `Down` — move backward
- `A` / `Left` — rotate left
- `D` / `Right` — rotate right
- `Space` — stop

The robot cannot drive through walls or obstacles because each attempted motion is checked against the world geometry before updating the pose.

## How it works

1. The environment creates a 2D map with walls and obstacles.
2. The robot is represented as a circular body with a heading angle.
3. The keyboard controller sends wheel velocities to the robot.
4. The simulation engine updates movement, then refreshes the LiDAR scan.
5. The occupancy grid stores free and occupied cells based on the LiDAR results.

This project is mainly intended as a teaching example for robot simulation, sensing, and basic mapping concepts.

## Notes

- The repository is intentionally lightweight and does not use ROS, Gazebo, or a game engine.
- The code is designed to be easy to read and extend for experiments with robot control, navigation, or SLAM concepts.
