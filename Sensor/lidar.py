import math
from matplotlib.lines import Line2D


class Lidar:

    def __init__(
        self,
        geometry,
        rotation_hz=0.5,
        max_range=5.0,
        scan_resolution=360,
        laser_color="red",
        laser_width=1.5,
        laser_alpha=0.8,
        point_color="lime",
        point_size=10
    ):

        # -----------------------------------------------------
        # WORLD GEOMETRY
        # -----------------------------------------------------

        self.geometry = geometry

        # -----------------------------------------------------
        # LIDAR CONFIGURATION
        # -----------------------------------------------------

        self.rotation_hz = rotation_hz
        self.max_range = max_range
        self.scan_resolution = scan_resolution

        # Angle between two LiDAR measurements.
        #
        # 360 measurements -> 1 degree
        # 720 measurements -> 0.5 degree
        #
        self.angle_increment = (
            2 * math.pi / self.scan_resolution
        )

        # -----------------------------------------------------
        # VISUALIZATION CONFIGURATION
        # -----------------------------------------------------

        self.laser_color = laser_color
        self.laser_width = laser_width
        self.laser_alpha = laser_alpha

        self.point_color = point_color
        self.point_size = point_size

        # -----------------------------------------------------
        # PHYSICAL LIDAR ANGLE
        # -----------------------------------------------------

        # This is the actual rotating beam angle.
        self.angle = 0.0

        # -----------------------------------------------------
        # NEXT MEASUREMENT ANGLE
        # -----------------------------------------------------

        # LiDAR does not need to measure at every simulation
        # update. It measures whenever the rotating beam
        # reaches the next sampling angle.
        self.next_measurement_angle = 0.0

        # -----------------------------------------------------
        # CURRENT MEASUREMENT
        # -----------------------------------------------------

        self.current_distance = max_range

        self.current_hit = False

        self.hit_point = None

        # Angle at which the current measurement was taken.
        self.current_measurement_angle = 0.0

        # -----------------------------------------------------
        # SCAN DATA
        # -----------------------------------------------------

        self.scan_data = []

        # Point cloud for the current scan.
        self.scan_points = []

        # -----------------------------------------------------
        # COMPLETED SCAN
        # -----------------------------------------------------

        # Keep the last completed point cloud visible.
        self.last_scan_points = []

        self.last_scan_data = []

        # -----------------------------------------------------
        # MATPLOTLIB OBJECTS
        # -----------------------------------------------------

        self.laser_line = None

        self.point_cloud = None

        # -----------------------------------------------------
        # SCAN COUNTER
        # -----------------------------------------------------

        self.scan_number = 0

    # =========================================================
    # MEASURE
    # =========================================================

    def measure(
        self,
        robot,
        measurement_angle
    ):

        # -----------------------------------------------------
        # BEAM ANGLE
        # -----------------------------------------------------

        laser_angle = (
            robot.theta
            + measurement_angle
        )

        # -----------------------------------------------------
        # LIDAR POSITION
        # -----------------------------------------------------

        start_x = robot.x
        start_y = robot.y

        # -----------------------------------------------------
        # BEAM DIRECTION
        # -----------------------------------------------------

        direction_x = math.cos(
            laser_angle
        )

        direction_y = math.sin(
            laser_angle
        )

        # -----------------------------------------------------
        # DEFAULT MEASUREMENT
        # -----------------------------------------------------

        nearest_distance = self.max_range

        nearest_point = None

        # -----------------------------------------------------
        # CHECK ALL WORLD OBJECTS
        # -----------------------------------------------------

        for (
            x,
            y,
            width,
            height
        ) in self.geometry.rectangles:

            distance = (
                self.ray_rectangle_intersection(
                    start_x,
                    start_y,
                    direction_x,
                    direction_y,
                    x,
                    y,
                    width,
                    height
                )
            )

            # -------------------------------------------------
            # KEEP THE CLOSEST OBJECT
            # -------------------------------------------------

            if (
                distance is not None
                and distance < nearest_distance
            ):

                nearest_distance = distance

                nearest_point = (
                    start_x
                    + direction_x * distance,

                    start_y
                    + direction_y * distance
                )

        # -----------------------------------------------------
        # STORE CURRENT MEASUREMENT
        # -----------------------------------------------------

        self.current_distance = (
            nearest_distance
        )

        self.current_measurement_angle = (
            measurement_angle
        )

        # -----------------------------------------------------
        # OBSTACLE DETECTED
        # -----------------------------------------------------

        if nearest_point is not None:

            self.current_hit = True

            self.hit_point = nearest_point

        # -----------------------------------------------------
        # NO OBSTACLE
        # -----------------------------------------------------

        else:

            self.current_hit = False

            self.hit_point = None

    # =========================================================
    # RAY / RECTANGLE INTERSECTION
    # =========================================================

    def ray_rectangle_intersection(
        self,
        ray_x,
        ray_y,
        direction_x,
        direction_y,
        rect_x,
        rect_y,
        rect_width,
        rect_height
    ):

        # -----------------------------------------------------
        # RECTANGLE LIMITS
        # -----------------------------------------------------

        left = rect_x

        right = (
            rect_x
            + rect_width
        )

        bottom = rect_y

        top = (
            rect_y
            + rect_height
        )

        distances = []

        # =====================================================
        # LEFT / RIGHT SIDES
        # =====================================================

        if direction_x != 0:

            # -------------------------------------------------
            # LEFT SIDE
            # -------------------------------------------------

            t = (
                (left - ray_x)
                / direction_x
            )

            if t >= 0:

                hit_y = (
                    ray_y
                    + t * direction_y
                )

                if (
                    bottom
                    <= hit_y
                    <= top
                ):

                    distances.append(t)

            # -------------------------------------------------
            # RIGHT SIDE
            # -------------------------------------------------

            t = (
                (right - ray_x)
                / direction_x
            )

            if t >= 0:

                hit_y = (
                    ray_y
                    + t * direction_y
                )

                if (
                    bottom
                    <= hit_y
                    <= top
                ):

                    distances.append(t)

        # =====================================================
        # BOTTOM / TOP SIDES
        # =====================================================

        if direction_y != 0:

            # -------------------------------------------------
            # BOTTOM SIDE
            # -------------------------------------------------

            t = (
                (bottom - ray_y)
                / direction_y
            )

            if t >= 0:

                hit_x = (
                    ray_x
                    + t * direction_x
                )

                if (
                    left
                    <= hit_x
                    <= right
                ):

                    distances.append(t)

            # -------------------------------------------------
            # TOP SIDE
            # -------------------------------------------------

            t = (
                (top - ray_y)
                / direction_y
            )

            if t >= 0:

                hit_x = (
                    ray_x
                    + t * direction_x
                )

                if (
                    left
                    <= hit_x
                    <= right
                ):

                    distances.append(t)

        # -----------------------------------------------------
        # NO INTERSECTION
        # -----------------------------------------------------

        if not distances:

            return None

        # -----------------------------------------------------
        # CLOSEST INTERSECTION
        # -----------------------------------------------------

        distance = min(distances)

        # -----------------------------------------------------
        # OUTSIDE MAX RANGE
        # -----------------------------------------------------

        if distance > self.max_range:

            return None

        return distance

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        robot,
        dt
    ):

        # -----------------------------------------------------
        # 1. CALCULATE PHYSICAL ROTATION
        # -----------------------------------------------------

        angular_velocity = (
            2
            * math.pi
            * self.rotation_hz
        )

        angle_change = (
            angular_velocity
            * dt
        )

        # -----------------------------------------------------
        # 2. SAVE PREVIOUS ANGLE
        # -----------------------------------------------------

        previous_angle = self.angle

        # -----------------------------------------------------
        # 3. ROTATE PHYSICAL BEAM
        # -----------------------------------------------------

        self.angle += angle_change

        # -----------------------------------------------------
        # 4. CHECK WHETHER BEAM PASSED 360°
        # -----------------------------------------------------

        completed_revolution = (
            self.angle >= 2 * math.pi
        )

        # -----------------------------------------------------
        # 5. MEASUREMENT SAMPLING
        # -----------------------------------------------------

        while (
            self.next_measurement_angle
            <= self.angle
        ):

            measurement_angle = (
                self.next_measurement_angle
            )

            # -------------------------------------------------
            # PERFORM ONE LIDAR MEASUREMENT
            # -------------------------------------------------

            self.measure(
                robot,
                measurement_angle
            )

            # -------------------------------------------------
            # STORE RAW SCAN DATA
            # -------------------------------------------------

            self.scan_data.append(
                {
                    "angle": measurement_angle,

                    "distance":
                        self.current_distance,

                    "hit":
                        self.current_hit,

                    "point":
                        self.hit_point,

                    "robot_x": robot.x,
                    "robot_y": robot.y,
                    "robot_theta": robot.theta
                }
            )

            # -------------------------------------------------
            # STORE POINT
            # -------------------------------------------------

            if self.current_hit:

                self.scan_points.append(
                    self.hit_point
                )

            # -------------------------------------------------
            # NEXT MEASUREMENT
            # -------------------------------------------------

            self.next_measurement_angle += (
                self.angle_increment
            )

        # -----------------------------------------------------
        # 6. COMPLETE SCAN
        # -----------------------------------------------------

        if completed_revolution:

            self.scan_number += 1

            # -------------------------------------------------
            # SAVE COMPLETED SCAN
            # -------------------------------------------------

            self.last_scan_data = (
                self.scan_data.copy()
            )

            self.last_scan_points = (
                self.scan_points.copy()
            )

            # -------------------------------------------------
            # PRINT INFORMATION
            # -------------------------------------------------

            print(
                f"LiDAR scan "
                f"{self.scan_number} complete: "
                f"{len(self.scan_data)} measurements, "
                f"{len(self.scan_points)} points"
            )

            # -------------------------------------------------
            # START NEW SCAN
            # -------------------------------------------------

            self.scan_data = []

            self.scan_points = []

            self.next_measurement_angle = 0.0

        # -----------------------------------------------------
        # 7. KEEP PHYSICAL ANGLE IN 0 → 2π
        # -----------------------------------------------------

        self.angle %= (
            2 * math.pi
        )

    # =========================================================
    # DRAW
    # =========================================================

    def draw(
        self,
        ax,
        robot
    ):

        # -----------------------------------------------------
        # CURRENT PHYSICAL BEAM ANGLE
        # -----------------------------------------------------

        laser_angle = (
            robot.theta
            + self.angle
        )

        # -----------------------------------------------------
        # LIDAR POSITION
        # -----------------------------------------------------

        start_x = robot.x

        start_y = robot.y

        # -----------------------------------------------------
        # BEAM END POINT
        # -----------------------------------------------------

        # IMPORTANT:
        #
        # The visible beam uses the SAME distance calculated
        # by measure().
        #
        end_x = (
            start_x
            + self.current_distance
            * math.cos(laser_angle)
        )

        end_y = (
            start_y
            + self.current_distance
            * math.sin(laser_angle)
        )

        # =====================================================
        # DRAW CURRENT LASER BEAM
        # =====================================================

        if self.laser_line is None:

            self.laser_line = Line2D(
                [start_x, end_x],
                [start_y, end_y],

                color=self.laser_color,

                linewidth=self.laser_width,

                alpha=self.laser_alpha
            )

            ax.add_line(
                self.laser_line
            )

        else:

            self.laser_line.set_data(
                [start_x, end_x],
                [start_y, end_y]
            )

        # =====================================================
        # CREATE POINT CLOUD
        # =====================================================

        if self.point_cloud is None:

            self.point_cloud = ax.scatter(
                [],
                [],
                s=self.point_size,
                c=self.point_color
            )

        # =====================================================
        # DISPLAY POINT CLOUD
        # =====================================================

        # Show the currently accumulating scan if available.
        #
        # Otherwise show the previous completed scan.
        #
        if self.scan_points:

            points_to_display = (
                self.scan_points
            )

        else:

            points_to_display = (
                self.last_scan_points
            )

        # -----------------------------------------------------
        # UPDATE POINT CLOUD
        # -----------------------------------------------------

        if points_to_display:

            x_points = [
                point[0]
                for point in points_to_display
            ]

            y_points = [
                point[1]
                for point in points_to_display
            ]

            self.point_cloud.set_offsets(
                list(
                    zip(
                        x_points,
                        y_points
                    )
                )
            )

        else:

            # Empty 2-column array.
            self.point_cloud.set_offsets(
                [
                    [
                        float("nan"),
                        float("nan")
                    ]
                ]
            )

    # =========================================================
    # GET CURRENT SCAN
    # =========================================================

    def get_scan_data(self):

        return self.scan_data

    # =========================================================
    # GET COMPLETED SCAN
    # =========================================================

    def get_last_scan_data(self):

        return self.last_scan_data

    # =========================================================
    # GET POINT CLOUD
    # =========================================================

    def get_point_cloud(self):

        if self.scan_points:

            return self.scan_points

        return self.last_scan_points