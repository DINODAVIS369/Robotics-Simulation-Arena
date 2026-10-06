import math

import numpy as np

from Robot.odometry import wrap_angle


def pose_in_corrected_frame(reference_pose, corrected_pose, pose):
    """Transform an odometry pose by the scan match's rigid pose correction."""
    heading_delta = wrap_angle(corrected_pose[2] - reference_pose[2])
    cosine = math.cos(heading_delta)
    sine = math.sin(heading_delta)
    relative_x = pose[0] - reference_pose[0]
    relative_y = pose[1] - reference_pose[1]
    return (
        corrected_pose[0] + cosine * relative_x - sine * relative_y,
        corrected_pose[1] + sine * relative_x + cosine * relative_y,
        wrap_angle(corrected_pose[2] + wrap_angle(pose[2] - reference_pose[2])),
    )


def local_scan_points(scan, stride=4):
    """Return hit points in the scan's first estimated robot frame."""
    reference_pose = next(
        (
            measurement.get("odometry_pose")
            for measurement in scan
            if measurement.get("odometry_pose") is not None
        ),
        None,
    )
    points = []
    for measurement in scan[::stride]:
        if not measurement["hit"]:
            continue

        angle = measurement["angle"]
        distance = measurement["distance"]
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)
        beam_pose = measurement.get("odometry_pose")
        if reference_pose is not None and beam_pose is not None:
            world_angle = beam_pose[2]
            world_x = beam_pose[0] + (
                local_x * math.cos(world_angle) - local_y * math.sin(world_angle)
            )
            world_y = beam_pose[1] + (
                local_x * math.sin(world_angle) + local_y * math.cos(world_angle)
            )
            relative_x = world_x - reference_pose[0]
            relative_y = world_y - reference_pose[1]
            reference_cosine = math.cos(reference_pose[2])
            reference_sine = math.sin(reference_pose[2])
            local_x = reference_cosine * relative_x + reference_sine * relative_y
            local_y = -reference_sine * relative_x + reference_cosine * relative_y
        points.append((local_x, local_y))

    return np.asarray(points, dtype=float)
