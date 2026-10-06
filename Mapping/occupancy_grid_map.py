import math

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from Mapping.scan_geometry import pose_in_corrected_frame


class OccupancyGridMap:

    def __init__(
        self,
        width=10.0,
        height=10.0,
        resolution=0.1
    ):

        # =====================================================
        # MAP SETTINGS
        # =====================================================

        self.width = width
        self.height = height
        self.resolution = resolution

        # Number of cells in X and Y direction.

        self.grid_width = int(
            width / resolution
        )

        self.grid_height = int(
            height / resolution
        )

        # =====================================================
        # OCCUPANCY GRID
        # =====================================================

        # Cell values:
        #
        # -1 = Unknown
        #  0 = Free
        #  1 = Occupied
        #

        self.grid = np.full(
            (
                self.grid_height,
                self.grid_width
            ),
            -1,
            dtype=np.int8
        )

        # =====================================================
        # VISUALIZATION
        # =====================================================

        self.fig = None
        self.ax = None
        self.image = None

    # =========================================================
    # WORLD POSITION -> GRID CELL
    # =========================================================

    def world_to_grid(
        self,
        x,
        y
    ):

        grid_x = int(
            x / self.resolution
        )

        grid_y = int(
            y / self.resolution
        )

        # Check whether the position
        # is inside the map.

        if (
            0 <= grid_x < self.grid_width
            and
            0 <= grid_y < self.grid_height
        ):

            return grid_x, grid_y

        return None

    # =========================================================
    # GRID CELL -> WORLD POSITION
    # =========================================================

    def grid_to_world(
        self,
        grid_x,
        grid_y
    ):

        # Return the centre of the cell.

        x = (
            grid_x + 0.5
        ) * self.resolution

        y = (
            grid_y + 0.5
        ) * self.resolution

        return x, y

    # =========================================================
    # MARK CELL AS FREE
    # =========================================================

    def mark_free(
        self,
        x,
        y
    ):

        grid_position = self.world_to_grid(
            x,
            y
        )

        if grid_position is None:
            return

        grid_x, grid_y = grid_position

        # Do not overwrite an occupied cell.

        if self.grid[
            grid_y,
            grid_x
        ] != 1:

            self.grid[
                grid_y,
                grid_x
            ] = 0

    # =========================================================
    # MARK CELL AS OCCUPIED
    # =========================================================

    def mark_occupied(
        self,
        x,
        y
    ):

        grid_position = self.world_to_grid(
            x,
            y
        )

        if grid_position is None:
            return

        grid_x, grid_y = grid_position

        self.grid[
            grid_y,
            grid_x
        ] = 1

    # =========================================================
    # UPDATE ONE LIDAR RAY
    # =========================================================

    def update_ray(
        self,
        robot_x,
        robot_y,
        robot_theta,
        angle,
        distance,
        hit
    ):

        # =====================================================
        # WORLD ANGLE
        # =====================================================

        world_angle = (
            robot_theta
            + angle
        )

        # =====================================================
        # RAY DIRECTION
        # =====================================================

        direction_x = math.cos(
            world_angle
        )

        direction_y = math.sin(
            world_angle
        )

        # =====================================================
        # RAY SAMPLING STEP
        # =====================================================

        # Use a step smaller than the grid cell.
        #
        # Example:
        #
        # resolution = 0.1 m
        # step       = 0.05 m
        #

        step = (
            self.resolution / 2
        )

        current_distance = 0.0

        # =====================================================
        # TRACE RAY THROUGH THE ENVIRONMENT
        # =====================================================

        while current_distance < distance:

            x = (
                robot_x
                + direction_x
                * current_distance
            )

            y = (
                robot_y
                + direction_y
                * current_distance
            )

            # Mark everything between the
            # robot and obstacle as free.

            self.mark_free(
                x,
                y
            )

            current_distance += step

        # =====================================================
        # OBSTACLE DETECTED
        # =====================================================

        if hit:

            hit_x = (
                robot_x
                + direction_x
                * distance
            )

            hit_y = (
                robot_y
                + direction_y
                * distance
            )

            # The LiDAR endpoint is an obstacle.

            self.mark_occupied(
                hit_x,
                hit_y
            )

    # =========================================================
    # UPDATE MAP FROM LIDAR SCAN
    # =========================================================

    def update_from_scan(
        self,
        scan_data,
        pose=None,
    ):

        reference_pose = next(
            (
                item.get("odometry_pose")
                for item in scan_data
                if item.get("odometry_pose") is not None
            ),
            pose,
        )

        # Process every LiDAR measurement.

        for measurement in scan_data:

            # =================================================
            # LiDAR DATA
            # =================================================

            angle = measurement[
                "angle"
            ]

            distance = measurement[
                "distance"
            ]

            hit = measurement[
                "hit"
            ]

            # =================================================
            # ROBOT POSE AT MEASUREMENT TIME
            # =================================================

            beam_pose = measurement.get("odometry_pose")
            if pose is None:
                try:
                    beam_pose = (
                        measurement["robot_x"],
                        measurement["robot_y"],
                        measurement["robot_theta"],
                    )
                except KeyError as error:
                    raise ValueError(
                        "A robot pose is required to map scans without pose metadata"
                    ) from error
            elif beam_pose is not None:
                beam_pose = pose_in_corrected_frame(
                    reference_pose,
                    pose,
                    beam_pose,
                )
            else:
                beam_pose = pose

            robot_x, robot_y, robot_theta = beam_pose

            # =================================================
            # UPDATE THIS RAY
            # =================================================

            self.update_ray(
                robot_x,
                robot_y,
                robot_theta,
                angle,
                distance,
                hit
            )

    def clear(self):
        self.grid.fill(-1)

    # =========================================================
    # CREATE MAP VISUALIZATION
    # =========================================================

    def create_visualization(
        self
    ):

        self.fig, self.ax = (
            plt.subplots()
        )

        # =====================================================
        # MAP COLORS
        # =====================================================

        # -1 -> Gray  -> Unknown
        #  0 -> White -> Free
        #  1 -> Black -> Occupied

        occupancy_colors = (
            ListedColormap(
                [
                    "gray",
                    "white",
                    "black"
                ]
            )
        )

        # =====================================================
        # DISCRETE COLOR NORMALIZATION
        # =====================================================

        occupancy_norm = BoundaryNorm(
            [
                -1.5,
                -0.5,
                0.5,
                1.5
            ],
            occupancy_colors.N
        )

        # =====================================================
        # DISPLAY GRID
        # =====================================================

        self.image = self.ax.imshow(
            self.grid,

            origin="lower",

            extent=[
                0,
                self.width,
                0,
                self.height
            ],

            cmap=occupancy_colors,

            norm=occupancy_norm,

            interpolation="nearest"
        )

        # =====================================================
        # AXIS SETTINGS
        # =====================================================

        self.ax.set_xlim(
            0,
            self.width
        )

        self.ax.set_ylim(
            0,
            self.height
        )

        self.ax.set_aspect(
            "equal"
        )

        self.ax.set_xlabel(
            "X position (m)"
        )

        self.ax.set_ylabel(
            "Y position (m)"
        )

        self.ax.set_title(
            "Occupancy Grid Map"
        )

    # =========================================================
    # UPDATE MAP DISPLAY
    # =========================================================

    def update_visualization(
        self
    ):

        # If visualization has not been
        # created yet, do nothing.

        if self.image is None:
            return

        # Update the image using
        # the latest grid.

        self.image.set_data(
            self.grid
        )

        # Ask Matplotlib to redraw.

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()