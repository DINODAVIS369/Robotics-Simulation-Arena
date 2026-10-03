import math

from .circle_robot import CircleRobot


class DifferentialDriveRobot(CircleRobot):

    def __init__(
        self,
        x=1,
        y=1,
        theta=0,
        wheel_radius=0.05,
        wheel_base=0.3
    ):
        # Call CircleRobot's constructor.
        # This gives us:
        #   self.x      -> robot X position
        #   self.y      -> robot Y position
        #   self.theta  -> robot heading
        #   self.radius -> circular robot radius
        super().__init__(x, y, theta)

        # Radius of each wheel in meters.
        #
        # We need this because the wheel's angular velocity
        # (rad/s) must be converted into linear velocity (m/s).
        self.wheel_radius = wheel_radius

        # Distance between the left and right wheels in meters.
        #
        # We need this to calculate how quickly the robot rotates.
        self.wheel_base = wheel_base

    def calculate_motion(self, omega_l, omega_r, dt):

        v_l = self.wheel_radius * omega_l
        v_r = self.wheel_radius * omega_r

        v = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.wheel_base

        x_velocity = v * math.cos(self.theta)
        y_velocity = v * math.sin(self.theta)

        # Calculate proposed position
        new_x = self.x + x_velocity * dt
        new_y = self.y + y_velocity * dt
        new_theta = self.theta + omega * dt

        return new_x, new_y, new_theta