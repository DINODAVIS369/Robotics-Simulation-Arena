import math
import unittest
from types import SimpleNamespace

from Controller.autonomous_controller import AutonomousController
from Controller.behavior_tree import Action, Condition, NodeStatus, Selector, Sequence
from Controller.grid_planner import GridPlanner
from Controller.neural_follower import (
    generate_imitation_data,
    load_network,
    predict_network,
)
from Controller.path_followers import FOLLOWERS
from Mapping.loop_closure import LoopClosure
from Mapping.occupancy_grid_map import OccupancyGridMap
from Mapping.pose_graph import PoseGraph, relative_pose
from Mapping.scan_geometry import local_scan_points, pose_in_corrected_frame
from Mapping.scan_matcher import ScanMatcher
from Environment.Env_V_1 import Environment
from Geometry.world_geometry import WorldGeometry
from Robot.differential_drive import DifferentialDriveRobot
from Robot.odometry import NoisyOdometry


class GridPlannerTests(unittest.TestCase):
    def test_plans_around_an_occupied_cell(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        occupancy_map.grid[2, 2] = 1
        planner = GridPlanner(occupancy_map, robot_radius=0)

        path = planner.plan((0.5, 2.5), (4.5, 2.5))

        self.assertTrue(path)
        self.assertNotIn((2, 2), [
            occupancy_map.world_to_grid(x, y) for x, y in path
        ])
        self.assertEqual(
            occupancy_map.world_to_grid(*path[-1]),
            occupancy_map.world_to_grid(4.5, 2.5),
        )

    def test_robot_radius_inflation_accounts_for_occupied_cell_area(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=0.1)
        occupancy_map.grid[20, 20] = 1
        planner = GridPlanner(occupancy_map, robot_radius=0.4)

        path = planner.plan((1.0, 2.05), (4.0, 2.05))

        self.assertTrue(path)
        clearance = 0.4 + 0.1 * math.sqrt(2) / 2 + 0.15
        for x, y in path:
            self.assertGreater(
                math.hypot(x - 2.05, y - 2.05),
                clearance,
            )

    def test_returns_empty_when_map_blocks_all_routes(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        occupancy_map.grid[2, :] = 1
        planner = GridPlanner(occupancy_map, robot_radius=0)

        self.assertEqual(planner.plan((2.5, 0.5), (2.5, 4.5)), [])

    def test_rejects_goals_outside_the_map(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        planner = GridPlanner(occupancy_map, robot_radius=0)

        self.assertEqual(planner.plan((0.5, 0.5), (6.0, 6.0)), [])

    def test_all_planners_find_valid_routes_around_obstacles(self):
        for algorithm in ("astar", "dijkstra", "greedy", "theta"):
            with self.subTest(algorithm=algorithm):
                occupancy_map = OccupancyGridMap(width=8, height=8, resolution=1)
                occupancy_map.grid[:, :] = 0
                occupancy_map.grid[1:7, 3] = 1
                occupancy_map.grid[5, 3] = 0
                planner = GridPlanner(
                    occupancy_map,
                    robot_radius=0,
                    algorithm=algorithm,
                )

                path = planner.plan((1.5, 1.5), (6.5, 6.5))

                self.assertTrue(path)
                self.assertEqual(
                    occupancy_map.world_to_grid(*path[0]),
                    occupancy_map.world_to_grid(1.5, 1.5),
                )
                self.assertEqual(
                    occupancy_map.world_to_grid(*path[-1]),
                    occupancy_map.world_to_grid(6.5, 6.5),
                )
                for x, y in path:
                    cell_x, cell_y = occupancy_map.world_to_grid(x, y)
                    self.assertNotEqual(occupancy_map.grid[cell_y, cell_x], 1)
                blocked = planner._blocked_cells()
                cells = [occupancy_map.world_to_grid(*point) for point in path]
                for start, end in zip(cells, cells[1:]):
                    self.assertTrue(
                        planner._line_is_clear(
                            start,
                            end,
                            lambda cell: (
                                0 <= cell[0] < occupancy_map.grid_width
                                and 0 <= cell[1] < occupancy_map.grid_height
                                and not blocked[cell[1], cell[0]]
                            ),
                        )
                    )

    def test_dijkstra_and_astar_return_equal_costs(self):
        occupancy_map = OccupancyGridMap(width=6, height=6, resolution=1)
        occupancy_map.grid[:, :] = 0
        occupancy_map.grid[1:5, 3] = 1
        occupancy_map.grid[4, 3] = 0
        planner = GridPlanner(occupancy_map, robot_radius=0)

        astar_path = planner.plan((0.5, 0.5), (5.5, 5.5), algorithm="astar")
        dijkstra_path = planner.plan(
            (0.5, 0.5),
            (5.5, 5.5),
            algorithm="dijkstra",
        )

        def path_cost(path):
            cells = [occupancy_map.world_to_grid(*point) for point in path]
            return sum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(cells, cells[1:])
            )

        self.assertAlmostEqual(path_cost(astar_path), path_cost(dijkstra_path))

    def test_theta_star_shortcuts_open_grid_path(self):
        occupancy_map = OccupancyGridMap(width=10, height=10, resolution=1)
        occupancy_map.grid[:, :] = 0
        planner = GridPlanner(occupancy_map, robot_radius=0)

        astar_path = planner.plan((0.5, 0.5), (9.5, 7.5), algorithm="astar")
        theta_path = planner.plan((0.5, 0.5), (9.5, 7.5), algorithm="theta")

        self.assertLess(len(theta_path), len(astar_path))

    def test_planner_rejects_unknown_algorithm(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        with self.assertRaises(ValueError):
            GridPlanner(occupancy_map, robot_radius=0, algorithm="not-a-planner")


class AutonomousControllerTests(unittest.TestCase):
    def test_follower_collision_check_uses_occupied_cell_clearance_margin(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=0.1)
        occupancy_map.grid[20, 20] = 1
        follower = FOLLOWERS["rpp"](occupancy_map, robot_radius=0.4)

        self.assertTrue(follower._point_blocked(2.55, 2.05))

    def test_default_curved_follower_clears_a_wall_corner(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=0.1)
        occupancy_map.grid[20, 20] = 1
        robot = DifferentialDriveRobot(x=1.0, y=2.05, theta=0.0)
        goal = (4.0, 2.05)
        controller = AutonomousController(robot, occupancy_map, goal=goal)
        geometry = WorldGeometry()
        geometry.add_rectangle(2.0, 2.0, 0.1, 0.1)
        blocked = False
        contacts = 0

        for _ in range(800):
            controller.update(
                0.05,
                pose=(robot.x, robot.y, robot.theta),
                blocked=blocked,
            )
            new_x, new_y, new_theta = robot.calculate_motion(
                controller.omega_l,
                controller.omega_r,
                0.05,
            )
            blocked = geometry.circle_collision(
                new_x,
                new_y,
                robot.radius,
            )
            if blocked:
                contacts += 1
            else:
                robot.x = new_x
                robot.y = new_y
            robot.theta = new_theta
            if robot.x > 3.0 and robot.y < 2.05:
                break

        self.assertEqual(contacts, 0)
        self.assertGreater(robot.x, 3.0)
        self.assertLess(robot.y, 2.05)

    def test_generates_wheel_commands_toward_goal(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0)
        controller = AutonomousController(
            robot,
            occupancy_map,
            goal=(4.5, 0.5),
        )

        controller.update(0.05)

        self.assertGreater(controller.omega_l, 0)
        self.assertGreater(controller.omega_r, 0)
        self.assertEqual(controller.status, "Navigating to goal")

    def test_pure_pursuit_turns_toward_a_curved_path(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=1.5, y=1.5, theta=0)
        controller = AutonomousController(
            robot,
            occupancy_map,
            goal=(3.5, 3.5),
            lookahead_distance=0.5,
        )
        controller.path = [(1.5, 1.5), (2.0, 2.0), (3.5, 3.5)]
        controller.last_plan_scan = controller.scan_number

        controller.update(0.05)

        self.assertGreater(controller.omega_r, controller.omega_l)
        self.assertGreater(controller.omega_l, 0)
        self.assertGreater(controller.omega_r, 0)

    def test_replans_when_a_new_scan_is_mapped(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0)
        controller = AutonomousController(robot, occupancy_map, goal=(4.5, 4.5))

        controller.update(0.05)
        controller.scan_number = 2
        controller.update(0.05)

        self.assertEqual(controller.last_plan_scan, 2)

    def test_keyboard_selects_planner_and_replans(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0)
        controller = AutonomousController(robot, occupancy_map, goal=(4.5, 4.5))

        class KeyEvent:
            key = "4"

        controller.key_press(KeyEvent())

        self.assertEqual(controller.planner.algorithm, "theta")
        self.assertEqual(controller.path, [])
        self.assertEqual(controller.status, "Planner: Theta*")

    def test_every_follower_returns_bounded_finite_commands(self):
        for follower_name in FOLLOWERS:
            with self.subTest(follower=follower_name):
                occupancy_map = OccupancyGridMap(width=10, height=10, resolution=0.1)
                robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
                controller = AutonomousController(
                    robot,
                    occupancy_map,
                    goal=(4.0, 4.0),
                )
                controller.path = [
                    (1.0, 1.0),
                    (1.5, 1.5),
                    (2.0, 2.5),
                    (3.0, 3.5),
                    (4.0, 4.0),
                ]
                controller.last_plan_scan = controller.scan_number
                controller.set_follower(follower_name)
                controller.update(0.05)

                self.assertTrue(math.isfinite(controller.omega_l))
                self.assertTrue(math.isfinite(controller.omega_r))
                self.assertLessEqual(abs(controller.omega_l), 40.0)
                self.assertLessEqual(abs(controller.omega_r), 40.0)

    def test_keyboard_selects_each_path_follower(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0)
        controller = AutonomousController(robot, occupancy_map, goal=(4.5, 4.5))

        class KeyEvent:
            key = "f8"

        controller.key_press(KeyEvent())

        self.assertEqual(controller.follower_key, "neural")
        self.assertEqual(controller.follower_name, "Neural Imitation Follower")

    def test_neural_follower_improves_on_unseen_teacher_examples(self):
        features, labels = generate_imitation_data(
            sample_count=512,
            seed=103,
        )
        predictions = predict_network(load_network(), features)
        validation_mse = float(((predictions - labels) ** 2).mean())

        self.assertLess(validation_mse, 0.08)

    def test_behavior_tree_runs_goal_stop_branch(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=1.5, y=1.5, theta=0)
        controller = AutonomousController(robot, occupancy_map, goal=(1.5, 1.5))

        controller.update(0.05)

        self.assertEqual(controller.status, "Goal reached")
        self.assertEqual(controller.omega_l, 0)
        self.assertEqual(controller.omega_r, 0)

    def test_behavior_tree_stops_and_waits_when_no_route_exists(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        occupancy_map.grid[2, :] = 1
        robot = DifferentialDriveRobot(x=2.5, y=0.5, theta=0)
        controller = AutonomousController(robot, occupancy_map, goal=(2.5, 4.5))
        controller.omega_l = 4
        controller.omega_r = 4

        controller.update(0.05)

        self.assertEqual(controller.status, "No route; backing away to search")
        self.assertEqual(controller.omega_l, 0)
        self.assertEqual(controller.omega_r, 0)
        self.assertTrue(controller.recovery_active)

    def test_behavior_tree_rotates_to_scan_then_replans_after_collision(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=1)
        robot = DifferentialDriveRobot(x=0.5, y=0.5, theta=0)
        controller = AutonomousController(
            robot,
            occupancy_map,
            goal=(4.5, 4.5),
            recovery_turn_rate=1.0,
            recovery_turn_angle=0.4,
            blocked_ticks_to_recover=2,
        )
        controller.update(0.05, blocked=True)
        controller.update(0.05, blocked=True)

        self.assertTrue(controller.recovery_active)
        self.assertLess(controller.omega_l, 0)
        self.assertGreater(controller.omega_r, 0)
        self.assertIn("Recovery", controller.status)

        for _ in range(50):
            controller.update(0.05, blocked=False)

        self.assertFalse(controller.recovery_active)
        self.assertEqual(controller.recovery_attempts, 1)
        self.assertEqual(controller.last_plan_scan, controller.scan_number)
        self.assertTrue(controller.path)

    def test_no_route_recovery_backs_out_of_blocked_start_then_finds_path(self):
        occupancy_map = OccupancyGridMap(width=5, height=5, resolution=0.1)
        occupancy_map.grid[:, :] = 0
        occupancy_map.grid[:, 20] = 1
        robot = DifferentialDriveRobot(x=1.7, y=2.5, theta=0.0)
        controller = AutonomousController(
            robot,
            occupancy_map,
            goal=(0.5, 2.5),
            recovery_reverse_speed=0.2,
            recovery_reverse_distance=0.25,
            recovery_turn_rate=1.0,
            recovery_turn_angle=0.1,
            clearance_margin=0.0,
        )

        controller.update(0.05)
        self.assertTrue(controller.recovery_active)

        for _ in range(30):
            controller.update(0.05)
            if controller.omega_l == controller.omega_r:
                robot.x += (
                    robot.wheel_radius * controller.omega_l * 0.05
                    * math.cos(robot.theta)
                )
                robot.y += (
                    robot.wheel_radius * controller.omega_l * 0.05
                    * math.sin(robot.theta)
                )
            else:
                _, _, robot.theta = robot.calculate_motion(
                    controller.omega_l,
                    controller.omega_r,
                    0.05,
                )
            controller.estimated_pose = (robot.x, robot.y, robot.theta)
            if controller.path:
                break

        self.assertLess(robot.x, 1.5)
        self.assertTrue(controller.path)
        self.assertEqual(controller.status, "Navigating to goal")

    def test_environment_reports_collision_blocking_motion(self):
        env = Environment()
        robot = DifferentialDriveRobot(x=1.59, y=3.0, theta=0.0)
        env.robot_radius = robot.radius

        blocked = env.update_motion(robot, 10.0, 10.0, 0.1)

        self.assertTrue(blocked)
        self.assertAlmostEqual(robot.x, 1.59)
        self.assertAlmostEqual(robot.y, 3.0)
        self.assertAlmostEqual(robot.theta, 0.0)


class SlamTests(unittest.TestCase):
    def test_noisy_odometry_tracks_wheel_commands_without_true_pose_access(self):
        robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
        odometry = NoisyOdometry(robot, wheel_noise=0.01, seed=2)

        pose = odometry.update(10.0, 10.0, 0.1)

        self.assertGreater(pose[0], 1.04)
        self.assertLess(pose[0], 1.06)
        self.assertAlmostEqual(pose[1], 1.0, places=3)
        self.assertLess(abs(pose[2]), 0.01)

    def test_scan_matching_improves_pose_against_grid(self):
        occupancy_map = OccupancyGridMap(width=8, height=8, resolution=0.1)
        known_pose = (3.0, 3.0, 0.2)
        scan = []
        for index in range(360):
            angle = index * 2 * math.pi / 360
            hit = index % 8 == 0
            distance = 1.5
            scan.append({"angle": angle, "distance": distance, "hit": hit})
            if hit:
                world_angle = known_pose[2] + angle
                occupancy_map.mark_occupied(
                    known_pose[0] + distance * math.cos(world_angle),
                    known_pose[1] + distance * math.sin(world_angle),
                )

        matcher = ScanMatcher(occupancy_map)
        initial_pose = (3.15, 3.0, 0.2)
        matched_pose, score = matcher.match(scan, initial_pose)

        self.assertGreater(score, 0.05)
        self.assertLess(
            math.hypot(matched_pose[0] - known_pose[0], matched_pose[1] - known_pose[1]),
            math.hypot(initial_pose[0] - known_pose[0], initial_pose[1] - known_pose[1]),
        )

    def test_map_integrates_scan_at_estimated_pose(self):
        occupancy_map = OccupancyGridMap(width=4, height=4, resolution=0.1)
        scan = [{"angle": 0.0, "distance": 1.0, "hit": True}]

        occupancy_map.update_from_scan(scan, pose=(1.0, 1.0, 0.0))

        self.assertEqual(occupancy_map.grid[10, 20], 1)
        with self.assertRaises(ValueError):
            occupancy_map.update_from_scan(scan)

    def test_ray_mapping_vectorizes_free_cells_without_erasing_obstacles(self):
        occupancy_map = OccupancyGridMap(width=2, height=1, resolution=0.1)
        occupancy_map.grid[2, 5] = 1

        occupancy_map.update_ray(
            robot_x=0.2,
            robot_y=0.25,
            robot_theta=0.0,
            angle=0.0,
            distance=1.0,
            hit=True,
        )

        self.assertEqual(occupancy_map.grid[2, 3], 0)
        self.assertEqual(occupancy_map.grid[2, 5], 1)
        self.assertEqual(occupancy_map.grid[2, 11], 1)

    def test_scan_deskew_uses_odometry_not_hidden_simulator_pose(self):
        scan = [
            {
                "angle": 0.0,
                "distance": 1.0,
                "hit": True,
                "odometry_pose": (1.0, 1.0, 0.0),
            },
            {
                "angle": 0.0,
                "distance": 1.0,
                "hit": True,
                "odometry_pose": (1.0, 1.0, math.pi / 2),
            },
        ]

        points = local_scan_points(scan, stride=1)
        corrected = pose_in_corrected_frame(
            (1.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.0, 1.0, math.pi / 2),
        )

        self.assertAlmostEqual(points[0, 0], 1.0)
        self.assertAlmostEqual(points[1, 0], 0.0, places=7)
        self.assertAlmostEqual(points[1, 1], 1.0)
        self.assertAlmostEqual(corrected[0], 2.0)
        self.assertAlmostEqual(corrected[1], 2.0)
        self.assertAlmostEqual(corrected[2], math.pi / 2)

    def test_loop_closure_detects_a_revisited_keyframe(self):
        scan = [
            {
                "angle": index * 2 * math.pi / 360,
                "distance": 1.5,
                "hit": index % 4 == 0,
            }
            for index in range(360)
        ]
        detector = LoopClosure(closure_cooldown=10)
        detector.add_scan(1, scan, (2.0, 2.0, 0.0))

        closure = detector.add_scan(20, scan, (2.1, 2.0, 0.0))

        self.assertIsNotNone(closure)
        self.assertEqual(closure["anchor_scan"], 1)
        self.assertLess(closure["error"], 0.01)

    def test_pose_graph_loop_constraint_corrects_drift(self):
        graph = PoseGraph()
        graph.add_pose((0.0, 0.0, 0.0))
        graph.add_pose((1.1, 0.0, 0.0))
        graph.add_pose((2.2, 0.0, 0.0))
        graph.add_constraint(0, 1, (1.0, 0.0, 0.0))
        graph.add_constraint(1, 2, (1.0, 0.0, 0.0))
        graph.add_constraint(
            0,
            2,
            (1.9, 0.0, 0.0),
            translation_weight=100.0,
            rotation_weight=100.0,
            loop_closure=True,
        )

        poses = graph.optimize()

        self.assertEqual(poses[0], (0.0, 0.0, 0.0))
        self.assertLess(abs(poses[2][0] - 1.9), 0.02)
        self.assertLess(poses[1][0], 1.1)


class BehaviorTreeTests(unittest.TestCase):
    def test_sequence_and_selector_propagate_statuses(self):
        calls = []
        tree = Selector(
            Sequence(
                Condition(lambda: False),
                Action(lambda: calls.append("unreachable")),
            ),
            Action(lambda: calls.append("fallback") or NodeStatus.RUNNING),
        )

        self.assertEqual(tree.tick(), NodeStatus.RUNNING)
        self.assertEqual(calls, ["fallback"])


if __name__ == "__main__":
    unittest.main()
