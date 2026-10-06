from Environment.Env_V_1 import Environment
from Mapping.occupancy_grid_map import OccupancyGridMap
from Robot.differential_drive import DifferentialDriveRobot
from Controller.autonomous_controller import AutonomousController
from Sensor.lidar import Lidar
from Engine.SimulationEngine import SimulationEngine


def main():
    # Create the environment.
    env = Environment()

    # Create the robot.
    robot = DifferentialDriveRobot(
        x=1,
        y=1,
        theta=0,
    )

    # Give the environment the robot's radius for collision detection.
    env.robot_radius = robot.radius

    occupancy_map = OccupancyGridMap(
        width=env.width,
        height=env.height,
        resolution=0.1,
    )
    occupancy_map.create_visualization()

    # Create the LiDAR and map-based autonomous controller.
    lidar = Lidar(
        geometry=env.geometry,
        rotation_hz=2.0,
        max_range=5.0,
        scan_resolution=360,
    )
    controller = AutonomousController(
        robot=robot,
        occupancy_grid_map=occupancy_map,
        goal=(5.0, 5.0),
    )

    # Create the simulation engine.
    engine = SimulationEngine(
        env=env,
        robot=robot,
        controller=controller,
        lidar=lidar,
        occupancy_grid_map=occupancy_map,
        dt=0.05,
    )

    # Start the simulation.
    engine.run()


if __name__ == "__main__":
    main()
