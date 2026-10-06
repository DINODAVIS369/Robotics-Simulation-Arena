import math

import numpy as np

from Robot.odometry import wrap_angle


def relative_pose(source, target):
    """Express target's pose in source's coordinate frame."""
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    cosine = math.cos(source[2])
    sine = math.sin(source[2])
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        wrap_angle(target[2] - source[2]),
    )


class PoseGraph:
    """Small Gauss-Newton 2D pose graph with the first pose fixed as its anchor."""

    def __init__(self):
        self.poses = []
        self.constraints = []

    def add_pose(self, pose):
        self.poses.append(tuple(pose))
        return len(self.poses) - 1

    def add_constraint(
        self,
        source_index,
        target_index,
        measurement,
        translation_weight=20.0,
        rotation_weight=10.0,
        loop_closure=False,
    ):
        if not (
            0 <= source_index < len(self.poses)
            and 0 <= target_index < len(self.poses)
            and source_index != target_index
        ):
            raise ValueError("Pose graph constraint references invalid pose indices")
        if translation_weight <= 0 or rotation_weight <= 0:
            raise ValueError("Pose graph constraint weights must be positive")
        self.constraints.append(
            (
                source_index,
                target_index,
                tuple(measurement),
                np.asarray(
                    (translation_weight, translation_weight, rotation_weight),
                    dtype=float,
                ),
                loop_closure,
            )
        )

    def optimize(self, iterations=6, damping=1e-4):
        if len(self.poses) < 2 or not self.constraints:
            return list(self.poses)

        poses = [list(pose) for pose in self.poses]
        variable_count = 3 * (len(poses) - 1)

        for _ in range(iterations):
            hessian = np.zeros((variable_count, variable_count), dtype=float)
            gradient = np.zeros(variable_count, dtype=float)

            for source, target, measurement, information, is_loop in self.constraints:
                residual = self._residual(poses[source], poses[target], measurement)
                normalized_error = math.sqrt(float(np.sum(information * residual**2)))
                robust_scale = (
                    min(1.0, 2.0 / normalized_error)
                    if is_loop and normalized_error > 0
                    else 1.0
                )
                weighted_information = information * robust_scale
                jacobians = []

                for node_index in (source, target):
                    if node_index == 0:
                        jacobians.append(None)
                        continue

                    jacobian = np.zeros((3, 3), dtype=float)
                    for coordinate in range(3):
                        perturbed = poses[node_index].copy()
                        epsilon = 1e-6
                        perturbed[coordinate] += epsilon
                        if coordinate == 2:
                            perturbed[coordinate] = wrap_angle(
                                perturbed[coordinate]
                            )

                        if node_index == source:
                            changed = self._residual(
                                perturbed,
                                poses[target],
                                measurement,
                            )
                        else:
                            changed = self._residual(
                                poses[source],
                                perturbed,
                                measurement,
                            )
                        jacobian[:, coordinate] = (
                            changed - residual
                        ) / epsilon
                    jacobians.append(jacobian)

                variable_indices = []
                for node_index in (source, target):
                    if node_index == 0:
                        variable_indices.append(None)
                    else:
                        start = 3 * (node_index - 1)
                        variable_indices.append(slice(start, start + 3))

                for left_jacobian, left_slice in zip(jacobians, variable_indices):
                    if left_jacobian is None:
                        continue
                    gradient[left_slice] += (
                        (left_jacobian.T * weighted_information) @ residual
                    )
                    for right_jacobian, right_slice in zip(
                        jacobians,
                        variable_indices,
                    ):
                        if right_jacobian is None:
                            continue
                        hessian[left_slice, right_slice] += (
                            left_jacobian.T
                            @ (weighted_information[:, None] * right_jacobian)
                        )

            try:
                update = np.linalg.solve(
                    hessian + damping * np.eye(variable_count),
                    -gradient,
                )
            except np.linalg.LinAlgError as error:
                raise RuntimeError("Pose graph optimization failed to solve") from error

            if not np.all(np.isfinite(update)):
                raise RuntimeError("Pose graph optimization produced a non-finite update")

            largest_update = 0.0
            for node_index in range(1, len(poses)):
                start = 3 * (node_index - 1)
                delta_x, delta_y, delta_theta = update[start : start + 3]
                translation_norm = math.hypot(delta_x, delta_y)
                if translation_norm > 0.25:
                    scale = 0.25 / translation_norm
                    delta_x *= scale
                    delta_y *= scale
                delta_theta = max(-0.15, min(0.15, delta_theta))
                largest_update = max(
                    largest_update,
                    abs(delta_x),
                    abs(delta_y),
                    abs(delta_theta),
                )
                poses[node_index] = [
                    poses[node_index][0] + delta_x,
                    poses[node_index][1] + delta_y,
                    wrap_angle(poses[node_index][2] + delta_theta),
                ]

            if largest_update < 1e-5:
                break

        self.poses = [tuple(pose) for pose in poses]
        return list(self.poses)

    @staticmethod
    def _residual(source, target, measurement):
        predicted = relative_pose(source, target)
        return np.asarray(
            (
                predicted[0] - measurement[0],
                predicted[1] - measurement[1],
                wrap_angle(predicted[2] - measurement[2]),
            ),
            dtype=float,
        )
