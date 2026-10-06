import math

import numpy as np

from Robot.odometry import wrap_angle
from Mapping.scan_geometry import local_scan_points


class ScanMatcher:
    """Small correlative scan matcher for aligning LiDAR returns to a grid map."""

    def __init__(
        self,
        occupancy_grid_map,
        translation_window=0.3,
        translation_step=None,
        angle_window=0.2,
        angle_step=0.05,
    ):
        self.map = occupancy_grid_map
        self.translation_window = translation_window
        self.translation_step = translation_step or occupancy_grid_map.resolution / 2
        self.angle_window = angle_window
        self.angle_step = angle_step

    @staticmethod
    def local_points(scan, stride=4):
        return local_scan_points(scan, stride)

    def match(self, scan, initial_pose):
        """Return (pose, score); retain odometry when map evidence is insufficient."""
        points = self.local_points(scan)
        if len(points) < 12 or np.count_nonzero(self.map.grid == 1) < 12:
            return initial_pose, 0.0

        x_values = self._candidates(
            initial_pose[0], self.translation_window, self.translation_step
        )
        y_values = self._candidates(
            initial_pose[1], self.translation_window, self.translation_step
        )
        heading_values = self._candidates(
            initial_pose[2], self.angle_window, self.angle_step, angular=True
        )

        best_pose = initial_pose
        best_score = self._score(points, *initial_pose)
        prior_score = best_score
        for heading in heading_values:
            cosine = math.cos(heading)
            sine = math.sin(heading)
            rotated_x = points[:, 0] * cosine - points[:, 1] * sine
            rotated_y = points[:, 0] * sine + points[:, 1] * cosine
            for x in x_values:
                for y in y_values:
                    score = self._score_xy(rotated_x, rotated_y, x, y)
                    if score > best_score:
                        best_score = score
                        best_pose = (x, y, heading)

        if best_score < 0.05 or best_score <= prior_score:
            return initial_pose, max(0.0, prior_score)
        return best_pose, best_score

    def _candidates(self, center, window, step, angular=False):
        count = int(math.floor(window / step))
        values = [center + offset * step for offset in range(-count, count + 1)]
        if center not in values:
            values.append(center)
        if angular:
            return [wrap_angle(value) for value in values]
        return values

    def _score(self, points, x, y, theta):
        cosine = math.cos(theta)
        sine = math.sin(theta)
        rotated_x = points[:, 0] * cosine - points[:, 1] * sine
        rotated_y = points[:, 0] * sine + points[:, 1] * cosine
        return self._score_xy(rotated_x, rotated_y, x, y)

    def _score_xy(self, rotated_x, rotated_y, x, y):
        grid_x = np.floor((x + rotated_x) / self.map.resolution).astype(int)
        grid_y = np.floor((y + rotated_y) / self.map.resolution).astype(int)
        inside = (
            (grid_x >= 0)
            & (grid_x < self.map.grid_width)
            & (grid_y >= 0)
            & (grid_y < self.map.grid_height)
        )
        scores = np.full(len(grid_x), -0.25, dtype=float)
        values = np.zeros(len(grid_x), dtype=float)
        values[inside] = self.map.grid[grid_y[inside], grid_x[inside]]
        scores[inside] = np.where(
            values[inside] == 1,
            1.0,
            np.where(values[inside] == 0, -0.25, 0.0),
        )
        return float(np.mean(scores))
