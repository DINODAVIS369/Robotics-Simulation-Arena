import math

import numpy as np

from Mapping.scan_geometry import local_scan_points
from Robot.odometry import wrap_angle


class LoopClosure:
    """Detect revisited scan keyframes and distribute a pose correction."""

    def __init__(
        self,
        minimum_separation=12,
        revisit_radius=1.0,
        translation_window=0.45,
        translation_step=0.1,
        angle_window=0.35,
        angle_step=0.1,
        max_match_error=0.12,
        closure_cooldown=30,
    ):
        self.keyframes = []
        self.minimum_separation = minimum_separation
        self.revisit_radius = revisit_radius
        self.translation_window = translation_window
        self.translation_step = translation_step
        self.angle_window = angle_window
        self.angle_step = angle_step
        self.max_match_error = max_match_error
        self.closure_cooldown = closure_cooldown
        self.last_closure_scan = -closure_cooldown

    def add_scan(self, scan_number, scan, pose):
        points = self._local_points(scan)
        if len(points) < 12:
            return None

        can_close_loop = (
            scan_number - self.last_closure_scan >= self.closure_cooldown
        )
        for frame in self.keyframes:
            if not can_close_loop:
                break
            if scan_number - frame["scan_number"] < self.minimum_separation:
                continue
            if math.hypot(pose[0] - frame["pose"][0], pose[1] - frame["pose"][1]) > self.revisit_radius:
                continue

            matched_pose, error = self._align(
                points,
                frame["world_points"],
                pose,
            )
            if error <= self.max_match_error:
                self.last_closure_scan = scan_number
                return {
                    "scan_number": scan_number,
                    "anchor_scan": frame["scan_number"],
                    "pose": matched_pose,
                    "error": error,
                }

        if not self.keyframes or math.hypot(
            pose[0] - self.keyframes[-1]["pose"][0],
            pose[1] - self.keyframes[-1]["pose"][1],
        ) >= 0.5:
            world_points = self._transform(points, pose)
            self.keyframes.append(
                {
                    "scan_number": scan_number,
                    "pose": pose,
                    "local_points": points,
                    "world_points": world_points,
                }
            )
        return None

    def update_keyframe_poses(self, scan_poses):
        for frame in self.keyframes:
            index = frame["scan_number"] - 1
            if index >= len(scan_poses):
                continue
            frame["pose"] = scan_poses[index]
            frame["world_points"] = self._transform(
                frame["local_points"],
                frame["pose"],
            )

    @staticmethod
    def _local_points(scan):
        return local_scan_points(scan)

    @staticmethod
    def _transform(points, pose):
        cosine = math.cos(pose[2])
        sine = math.sin(pose[2])
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return points @ rotation.T + np.asarray(pose[:2])

    def _align(self, local_points, reference_points, initial_pose):
        search_window = max(self.translation_window, self.revisit_radius)
        count = int(search_window / self.translation_step)
        offsets = [index * self.translation_step for index in range(-count, count + 1)]
        angle_count = int(self.angle_window / self.angle_step)
        angle_offsets = [
            index * self.angle_step for index in range(-angle_count, angle_count + 1)
        ]
        sample = local_points[::2]
        reference = reference_points[::2]
        best_pose = initial_pose
        best_error = math.inf

        for delta_heading in angle_offsets:
            heading = wrap_angle(initial_pose[2] + delta_heading)
            cosine = math.cos(heading)
            sine = math.sin(heading)
            rotation = np.asarray(((cosine, -sine), (sine, cosine)))
            rotated = sample @ rotation.T
            for dx in offsets:
                for dy in offsets:
                    translated = rotated + np.asarray(
                        (initial_pose[0] + dx, initial_pose[1] + dy)
                    )
                    differences = translated[:, None, :] - reference[None, :, :]
                    distances = np.sqrt(np.sum(differences * differences, axis=2))
                    forward_error = np.mean(np.min(distances, axis=1))
                    reverse_error = np.mean(np.min(distances, axis=0))
                    error = float((forward_error + reverse_error) / 2.0)
                    if error < best_error:
                        best_error = error
                        best_pose = (
                            initial_pose[0] + dx,
                            initial_pose[1] + dy,
                            heading,
                        )

        return best_pose, best_error
