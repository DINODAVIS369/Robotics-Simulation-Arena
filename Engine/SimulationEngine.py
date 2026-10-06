import math

import matplotlib.pyplot as plt

from Mapping.loop_closure import LoopClosure
from Mapping.pose_graph import PoseGraph, relative_pose
from Mapping.scan_geometry import pose_in_corrected_frame
from Mapping.scan_matcher import ScanMatcher
from Robot.odometry import NoisyOdometry


class SimulationEngine:

    def __init__(
        self,
        env,
        robot,
        controller,
        lidar,
        occupancy_grid_map,
        dt=0.005
    ):
        self.env = env
        self.robot = robot
        self.controller = controller
        self.lidar = lidar
        self.occupancy_grid_map = occupancy_grid_map
        self.dt = dt
        self.last_mapped_scan = 0
        self.odometry = NoisyOdometry(robot)
        self.scan_matcher = ScanMatcher(occupancy_grid_map)
        self.loop_closure = LoopClosure()
        self.scan_history = []
        self.scan_poses = []
        self.scan_odometry_poses = []
        self.pose_graph = PoseGraph()
        self.loop_closure_count = 0
        self.path_line, = self.env.ax.plot(
            [],
            [],
            color="orange",
            linestyle="--",
            linewidth=2.5,
            label="Planned path",
            zorder=5,
        )
        if hasattr(self.controller, "goal"):
            goal_x, goal_y = self.controller.goal
            self.env.ax.plot(
                [goal_x],
                [goal_y],
                marker="*",
                color="gold",
                markeredgecolor="black",
                markersize=14,
                linestyle="None",
                label="Goal",
                zorder=6,
            )
            self.env.ax.legend(loc="upper right")

        self.env.fig.canvas.mpl_connect(
            "key_press_event",
            self.controller.key_press
        )

        self.env.fig.canvas.mpl_connect(
            "key_release_event",
            self.controller.key_release
        )

    def update(self):

        # 1. Move the robot.
        left_command = self.controller.omega_l
        right_command = self.controller.omega_r
        movement_blocked = self.env.update_motion(
            self.robot,
            left_command,
            right_command,
            self.dt
        )
        odometry_pose = self.odometry.update(
            left_command,
            right_command,
            self.dt,
        )

        # 2. Update the LiDAR.
        # LiDAR needs the robot pose to perform its measurement.
        self.lidar.update(
            self.robot,
            self.dt,
            odometry_pose=odometry_pose,
        )

        # 3. Update robot visualization.
        self.robot.draw(self.env.ax)

        # 4. Update LiDAR visualization.
        self.lidar.draw(
            self.env.ax,
            self.robot
        )
        # 5. Get the latest completed LiDAR scan.
        scan = self.lidar.get_last_scan_data()

        if scan and self.lidar.scan_number != self.last_mapped_scan:
            scan_reference_pose = next(
                (
                    measurement.get("odometry_pose")
                    for measurement in scan
                    if measurement.get("odometry_pose") is not None
                ),
                odometry_pose,
            )
            estimated_pose, match_score = self.scan_matcher.match(
                scan,
                scan_reference_pose,
            )
            self.scan_history.append(scan)
            self.scan_poses.append(estimated_pose)
            current_node = self.pose_graph.add_pose(estimated_pose)
            self.scan_odometry_poses.append(scan_reference_pose)
            if current_node > 0:
                odometry_measurement = relative_pose(
                    self.scan_odometry_poses[-2],
                    scan_reference_pose,
                )
                self.pose_graph.add_constraint(
                    current_node - 1,
                    current_node,
                    odometry_measurement,
                    translation_weight=30.0,
                    rotation_weight=20.0,
                )

            closure = self.loop_closure.add_scan(
                self.lidar.scan_number,
                scan,
                estimated_pose,
            )
            if closure is not None:
                anchor_index = closure["anchor_scan"] - 1
                closure_measurement = relative_pose(
                    self.scan_poses[anchor_index],
                    closure["pose"],
                )
                self.pose_graph.add_constraint(
                    anchor_index,
                    current_node,
                    closure_measurement,
                    translation_weight=80.0,
                    rotation_weight=40.0,
                    loop_closure=True,
                )
                self.scan_poses = self.pose_graph.optimize()
                self.loop_closure.update_keyframe_poses(self.scan_poses)
                self.loop_closure_count += 1
                self._rebuild_map()

            corrected_current_pose = pose_in_corrected_frame(
                scan_reference_pose,
                self.scan_poses[-1],
                odometry_pose,
            )
            self.odometry.set_pose(corrected_current_pose)
            self.occupancy_grid_map.update_from_scan(
                scan,
                pose=self.scan_poses[-1],
            )
            self.occupancy_grid_map.update_visualization()
            self.last_mapped_scan = self.lidar.scan_number
            if hasattr(self.controller, "scan_number"):
                self.controller.scan_number = self.lidar.scan_number
            if hasattr(self.controller, "match_score"):
                self.controller.match_score = match_score

        if hasattr(self.controller, "update"):
            if hasattr(self.controller, "estimated_pose"):
                self.controller.update(
                    self.dt,
                    pose=self.odometry.pose,
                    blocked=movement_blocked,
                )
            else:
                self.controller.update(self.dt)
        self._update_path_visualization()

        if hasattr(self.controller, "status"):
            position_error = math.hypot(
                self.robot.x - self.odometry.x,
                self.robot.y - self.odometry.y,
            )
            self.env.ax.set_title(
                f"SLAM {self.controller.status} | "
                f"planner={self.controller.planner.algorithm_name} | "
                f"follower={self.controller.follower_name} | "
                f"estimated pose=({self.odometry.x:.1f}, "
                f"{self.odometry.y:.1f}) | "
                f"pose error={position_error:.2f} m | "
                f"scan match={getattr(self.controller, 'match_score', 0.0):.2f} | "
                f"loop closures={self.loop_closure_count} "
                f"(goal: {self.controller.goal[0]:.1f}, "
                f"{self.controller.goal[1]:.1f}; 1-4 planner, F1-F7 follower, "
                "G toggles manual/autonomous)"
            )

    def _update_path_visualization(self):
        path = getattr(self.controller, "path", ())
        if path:
            self.path_line.set_data(
                [point[0] for point in path],
                [point[1] for point in path],
            )
        else:
            self.path_line.set_data([], [])

    def _rebuild_map(self):
        self.occupancy_grid_map.clear()
        for scan, pose in zip(self.scan_history, self.scan_poses):
            self.occupancy_grid_map.update_from_scan(scan, pose=pose)

    def run(self):

        # Give Matplotlib time to create the window.
        plt.show(block=False)
        plt.pause(0.1)

        # Simulation loop.
        while plt.fignum_exists(self.env.fig.number):

            self.update()

            plt.pause(self.dt)