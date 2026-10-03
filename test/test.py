
from Environment.Env_V_1 import Environment
from Mapping.occupancy_grid_map import OccupancyGridMap
from Robot.differential_drive import DifferentialDriveRobot
from Controller.keyboard_controller import KeyboardController
from Sensor.lidar import Lidar
from Engine.SimulationEngine import SimulationEngine


# Create the environment.
env = Environment()
1
# Create the robot.
robot = DifferentialDriveRobot(
    x=1,
    y=1,
    theta=0
)


# Give the environment the robot's radius
# for collision detection.
env.robot_radius = robot.radius


# Create the keyboard controller.
controller = KeyboardController(robot)


# Create the LiDAR.
lidar = Lidar(
    geometry=env.geometry,
    rotation_hz=2.0,
    max_range=5.0,
    scan_resolution=360
)

occupancy_map = OccupancyGridMap(
    width=env.width,
    height=env.height,
    resolution=0.1
)

occupancy_map.create_visualization()


# Create the simulation engine.
engine = SimulationEngine(
    env=env,
    robot=robot,
    controller=controller,
    lidar=lidar,
    occupancy_grid_map=occupancy_map,
    dt=0.05
)


# Start the simulation.
# 
engine.run()
