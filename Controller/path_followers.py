import math
import random

import numpy as np

from .neural_follower import NeuralFollower


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _nearest_path_point(pose, path, start_index):
    index = min(start_index, len(path) - 1)
    distance = math.inf
    for candidate in range(start_index, len(path)):
        candidate_distance = math.hypot(
            path[candidate][0] - pose[0],
            path[candidate][1] - pose[1],
        )
        if candidate_distance < distance:
            index = candidate
            distance = candidate_distance
    return index, distance


def _lookahead_point(pose, path, start_index, lookahead):
    for index in range(start_index, len(path)):
        if math.hypot(path[index][0] - pose[0], path[index][1] - pose[1]) >= lookahead:
            return path[index]
    return path[-1]


class PathFollower:
    name = "Follower"

    def __init__(
        self,
        occupancy_grid_map,
        robot_radius,
        max_speed=0.35,
        max_angular_speed=1.5,
        lookahead=0.55,
        clearance_margin=0.15,
    ):
        self.map = occupancy_grid_map
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.lookahead = lookahead
        if clearance_margin < 0:
            raise ValueError("clearance_margin must be non-negative")
        self.clearance_margin = clearance_margin

    def compute(self, pose, path, path_index, goal, dt):
        raise NotImplementedError

    def _speed_for_curvature(self, curvature):
        if abs(curvature) < 1e-9:
            return self.max_speed
        return min(self.max_speed, math.sqrt(0.6 / abs(curvature)))

    def _point_blocked(self, x, y):
        cell = self.map.world_to_grid(x, y)
        if cell is None:
            return True
        safe_radius = (
            self.robot_radius
            + self.map.resolution * math.sqrt(2) / 2
            + self.clearance_margin
        )
        radius_cells = math.ceil(safe_radius / self.map.resolution)
        center_x, center_y = cell
        for gy in range(center_y - radius_cells, center_y + radius_cells + 1):
            for gx in range(center_x - radius_cells, center_x + radius_cells + 1):
                if not (0 <= gx < self.map.grid_width and 0 <= gy < self.map.grid_height):
                    return True
                dx = (gx - center_x) * self.map.resolution
                dy = (gy - center_y) * self.map.resolution
                if math.hypot(dx, dy) <= safe_radius and self.map.grid[gy, gx] == 1:
                    return True
        return False

    def _trajectory_cost(self, start, controls, dt, path, goal):
        x, y, theta = start
        cost = 0.0
        for velocity, angular in controls:
            theta = _wrap(theta + angular * dt)
            x += velocity * math.cos(theta) * dt
            y += velocity * math.sin(theta) * dt
            if self._point_blocked(x, y):
                return math.inf
            _, path_distance = _nearest_path_point((x, y, theta), path, 0)
            goal_distance = math.hypot(goal[0] - x, goal[1] - y)
            cost += 0.08 * path_distance + 0.12 * goal_distance
        cost -= 0.12 * controls[0][0]
        return cost


class RegulatedPurePursuit(PathFollower):
    name = "Regulated Pure Pursuit"

    def compute(self, pose, path, path_index, goal, dt):
        target = _lookahead_point(pose, path, path_index, self.lookahead)
        local_x, local_y = _to_robot_frame(pose, target)
        curvature = 2.0 * local_y / max(local_x**2 + local_y**2, 1e-9)
        goal_distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        velocity = min(
            self._speed_for_curvature(curvature),
            math.sqrt(1.2 * goal_distance),
        )
        return velocity, _clamp(velocity * curvature, self.max_angular_speed)


class PurePursuit(PathFollower):
    name = "Pure Pursuit"

    def compute(self, pose, path, path_index, goal, dt):
        target = _lookahead_point(pose, path, path_index, self.lookahead)
        local_x, local_y = _to_robot_frame(pose, target)
        curvature = 2.0 * local_y / max(local_x**2 + local_y**2, 1e-9)
        velocity = min(self.max_speed, math.sqrt(1.2 * math.hypot(
            goal[0] - pose[0], goal[1] - pose[1]
        )))
        return velocity, _clamp(velocity * curvature, self.max_angular_speed)


class StanleyFollower(PathFollower):
    name = "Stanley"

    def compute(self, pose, path, path_index, goal, dt):
        index, cross_track = _nearest_path_point(pose, path, path_index)
        target_index = min(index + 1, len(path) - 1)
        segment = path[target_index][0] - path[index][0], path[target_index][1] - path[index][1]
        path_heading = math.atan2(segment[1], segment[0])
        dx, dy = pose[0] - path[index][0], pose[1] - path[index][1]
        signed_error = math.sin(path_heading) * dx - math.cos(path_heading) * dy
        heading_error = _wrap(path_heading - pose[2])
        steering = heading_error + math.atan2(1.2 * signed_error, 0.15 + self.max_speed)
        velocity = min(self.max_speed, math.sqrt(1.2 * math.hypot(
            goal[0] - pose[0], goal[1] - pose[1]
        )))
        return velocity, _clamp(2.0 * steering, self.max_angular_speed)


class PIDFollower(PathFollower):
    name = "PID"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integral = 0.0
        self.previous_error = None

    def compute(self, pose, path, path_index, goal, dt):
        index, cross_track = _nearest_path_point(pose, path, path_index)
        target_index = min(index + 1, len(path) - 1)
        segment = path[target_index][0] - path[index][0], path[target_index][1] - path[index][1]
        path_heading = math.atan2(segment[1], segment[0])
        dx, dy = pose[0] - path[index][0], pose[1] - path[index][1]
        signed_error = math.sin(path_heading) * dx - math.cos(path_heading) * dy
        error = _wrap(path_heading - pose[2]) - math.atan2(signed_error, 0.5)
        self.integral = _clamp_value(self.integral + error * dt, 1.0)
        derivative = (
            0.0
            if self.previous_error is None or dt <= 0
            else _wrap(error - self.previous_error) / dt
        )
        self.previous_error = error
        angular = 2.0 * error + 0.05 * self.integral + 0.08 * derivative
        velocity = min(self.max_speed, math.sqrt(1.2 * math.hypot(
            goal[0] - pose[0], goal[1] - pose[1]
        )))
        if abs(error) > 1.0:
            velocity *= 0.35
        return velocity, _clamp(angular, self.max_angular_speed)

    def reset(self):
        self.integral = 0.0
        self.previous_error = None


class DWBFollower(PathFollower):
    name = "DWB"

    def compute(self, pose, path, path_index, goal, dt):
        horizon_steps = 8
        sim_dt = 0.15
        best_control = (0.0, 0.0)
        best_cost = math.inf
        for velocity in (0.0, self.max_speed * 0.5, self.max_speed):
            for angular in np.linspace(-self.max_angular_speed, self.max_angular_speed, 9):
                control = (velocity, float(angular))
                cost = self._trajectory_cost(
                    pose,
                    [control] * horizon_steps,
                    sim_dt,
                    path[path_index:],
                    goal,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_control = control
        return best_control


class MPPIFollower(PathFollower):
    name = "MPPI"

    def __init__(self, *args, samples=48, horizon=8, seed=11, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = samples
        self.horizon = horizon
        self.random = random.Random(seed)
        self.previous_controls = []

    def compute(self, pose, path, path_index, goal, dt):
        if not path[path_index:]:
            return 0.0, 0.0
        nominal = self._nominal_control(pose, path, path_index, goal)
        candidates = []
        costs = []
        for _ in range(self.samples):
            controls = []
            for step in range(self.horizon):
                decay = 1.0 / (1.0 + step * 0.15)
                velocity = _clamp_value(
                    nominal[0] + self.random.gauss(0, 0.12) * decay,
                    self.max_speed,
                )
                angular = _clamp_value(
                    nominal[1] + self.random.gauss(0, 0.5) * decay,
                    self.max_angular_speed,
                )
                controls.append((velocity, angular))
            candidates.append(controls)
            costs.append(
                self._trajectory_cost(pose, controls, 0.15, path[path_index:], goal)
            )

        finite = [index for index, cost in enumerate(costs) if math.isfinite(cost)]
        if not finite:
            return 0.0, 0.0
        minimum = min(costs[index] for index in finite)
        weights = {
            index: math.exp(-min((costs[index] - minimum) / 0.2, 50.0))
            for index in finite
        }
        total = sum(weights.values())
        return tuple(
            sum(weights[index] * candidates[index][0][axis] for index in finite) / total
            for axis in (0, 1)
        )

    def _nominal_control(self, pose, path, path_index, goal):
        target = _lookahead_point(pose, path, path_index, self.lookahead)
        local_x, local_y = _to_robot_frame(pose, target)
        curvature = 2.0 * local_y / max(local_x**2 + local_y**2, 1e-9)
        return self.max_speed, _clamp(self.max_speed * curvature, self.max_angular_speed)


class MPCFollower(PathFollower):
    name = "MPC"

    def __init__(self, *args, horizon=6, beam_width=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.beam_width = beam_width

    def compute(self, pose, path, path_index, goal, dt):
        actions = [
            (velocity, angular)
            for velocity in (0.0, self.max_speed * 0.5, self.max_speed)
            for angular in (-self.max_angular_speed, 0.0, self.max_angular_speed)
        ]
        beam = [(0.0, pose, ())]
        sim_dt = 0.2
        for _ in range(self.horizon):
            candidates = []
            for cumulative_cost, state, sequence in beam:
                for action in actions:
                    next_state = _integrate(state, action, sim_dt)
                    if self._point_blocked(next_state[0], next_state[1]):
                        continue
                    _, path_distance = _nearest_path_point(
                        next_state,
                        path[path_index:],
                        0,
                    )
                    goal_distance = math.hypot(
                        goal[0] - next_state[0],
                        goal[1] - next_state[1],
                    )
                    cost = cumulative_cost + 0.12 * path_distance + 0.15 * goal_distance
                    candidates.append((cost, next_state, sequence + (action,)))
            if not candidates:
                return 0.0, 0.0
            beam = sorted(candidates, key=lambda candidate: candidate[0])[
                : self.beam_width
            ]
        return beam[0][2][0]


FOLLOWERS = {
    "rpp": RegulatedPurePursuit,
    "pure_pursuit": PurePursuit,
    "stanley": StanleyFollower,
    "pid": PIDFollower,
    "dwb": DWBFollower,
    "mppi": MPPIFollower,
    "mpc": MPCFollower,
}

FOLLOWERS["neural"] = NeuralFollower


def _to_robot_frame(pose, target):
    dx = target[0] - pose[0]
    dy = target[1] - pose[1]
    cosine = math.cos(pose[2])
    sine = math.sin(pose[2])
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _integrate(pose, control, dt):
    velocity, angular = control
    theta = _wrap(pose[2] + angular * dt)
    return (
        pose[0] + velocity * math.cos(theta) * dt,
        pose[1] + velocity * math.sin(theta) * dt,
        theta,
    )


def _clamp(value, limit):
    return max(-limit, min(limit, value))


def _clamp_value(value, limit):
    return max(-limit, min(limit, value))
