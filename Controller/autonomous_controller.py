import math

from .behavior_tree import Action, Condition, NodeStatus, Selector, Sequence
from .grid_planner import GridPlanner
from .keyboard_controller import KeyboardController
from .path_followers import FOLLOWERS


class AutonomousController(KeyboardController):
    """Follow occupancy-grid A* routes with regulated pure pursuit."""

    def __init__(
        self,
        robot,
        occupancy_grid_map,
        goal=(3.0, 9.0),
        linear_speed=0.35,
        angular_speed=1.5,
        lookahead_distance=0.55,
        min_lookahead_distance=0.3,
        max_lookahead_distance=0.8,
        max_lateral_acceleration=0.6,
    ):
        super().__init__(robot)
        self.map = occupancy_grid_map
        self.goal = goal
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.lookahead_distance = lookahead_distance
        self.min_lookahead_distance = min_lookahead_distance
        self.max_lookahead_distance = max_lookahead_distance
        self.max_lateral_acceleration = max_lateral_acceleration
        self.autonomous = True
        self.planner = GridPlanner(occupancy_grid_map, robot.radius)
        self.followers = {
            name: follower(
                occupancy_grid_map,
                robot.radius,
                max_speed=linear_speed,
                max_angular_speed=angular_speed,
                lookahead=lookahead_distance,
            )
            for name, follower in FOLLOWERS.items()
        }
        self.follower_key = "rpp"
        self.path = []
        self.path_index = 0
        self.scan_number = 0
        self.last_plan_scan = -1
        self.status = "Planning route"
        self.estimated_pose = (robot.x, robot.y, robot.theta)
        self.match_score = 0.0
        self.control_dt = 0.05
        self.behavior_tree = Selector(
            Sequence(
                Condition(self._goal_reached),
                Action(self._stop_at_goal),
            ),
            Sequence(
                Action(self._ensure_path),
                Action(self._follow_path),
            ),
            Action(self._recover_from_no_path),
        )

    def key_press(self, event):
        if event.key == "g":
            self.autonomous = not self.autonomous
            self.omega_l = 0
            self.omega_r = 0
            self.status = "Autonomous" if self.autonomous else "Manual"
            return

        planner_keys = {
            "1": "astar",
            "2": "dijkstra",
            "3": "greedy",
            "4": "theta",
        }
        if event.key in planner_keys:
            self.planner.set_algorithm(planner_keys[event.key])
            self.last_plan_scan = -1
            self.path = []
            self.path_index = 0
            self.omega_l = 0
            self.omega_r = 0
            self.status = f"Planner: {self.planner.algorithm_name}"
            return

        follower_keys = {
            f"f{index}": name
            for index, name in enumerate(FOLLOWERS, start=1)
        }
        if event.key in follower_keys:
            self.set_follower(follower_keys[event.key])
            return

        if not self.autonomous:
            super().key_press(event)

    def key_release(self, event):
        if not self.autonomous:
            super().key_release(event)

    def update(self, dt, pose=None):
        if not self.autonomous:
            return

        if pose is not None:
            self.estimated_pose = pose
        self.control_dt = dt
        self.behavior_tree.tick()

    def _goal_reached(self):
        return math.hypot(
            self.goal[0] - self.estimated_pose[0],
            self.goal[1] - self.estimated_pose[1],
        ) < 0.2

    def _stop_at_goal(self):
        self.omega_l = 0
        self.omega_r = 0
        self.status = "Goal reached"
        return NodeStatus.SUCCESS

    def _ensure_path(self):
        self._update_path()
        if not self.path:
            return NodeStatus.FAILURE
        return NodeStatus.SUCCESS

    def _recover_from_no_path(self):
        self.omega_l = 0
        self.omega_r = 0
        self.status = "No route; waiting for map update"
        return NodeStatus.RUNNING

    def _follow_path(self):
        if not self.path:
            return NodeStatus.FAILURE

        waypoint_tolerance = max(self.map.resolution * 0.75, 0.05)
        while self.path_index < len(self.path) - 1:
            waypoint = self.path[self.path_index]
            if math.hypot(
                waypoint[0] - self.estimated_pose[0],
                waypoint[1] - self.estimated_pose[1],
            ) > waypoint_tolerance:
                break
            self.path_index += 1

        linear_velocity, angular_velocity = self.followers[
            self.follower_key
        ].compute(
            self.estimated_pose,
            self.path,
            self.path_index,
            self.goal,
            self.control_dt,
        )

        left_velocity = linear_velocity - angular_velocity * self.robot.wheel_base / 2
        right_velocity = linear_velocity + angular_velocity * self.robot.wheel_base / 2
        self.omega_l = left_velocity / self.robot.wheel_radius
        self.omega_r = right_velocity / self.robot.wheel_radius
        self.status = "Navigating to goal"
        return NodeStatus.RUNNING

    def _update_path(self):
        scan_number = getattr(self, "scan_number", 0)
        if self.last_plan_scan == scan_number:
            return

        self.path = self.planner.plan(
            self.estimated_pose[:2],
            self.goal,
        )
        self.path_index = 0
        self.last_plan_scan = scan_number

    @property
    def follower_name(self):
        return FOLLOWERS[self.follower_key].name

    def set_follower(self, name):
        if name not in self.followers:
            raise ValueError(
                f"Unknown path follower {name!r}; choose from "
                f"{', '.join(self.followers)}"
            )
        current = self.followers[self.follower_key]
        if hasattr(current, "reset"):
            current.reset()
        self.follower_key = name
        selected = self.followers[name]
        if hasattr(selected, "reset"):
            selected.reset()
        self.omega_l = 0
        self.omega_r = 0
        self.status = f"Follower: {self.follower_name}"
