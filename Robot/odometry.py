import math
import random


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class NoisyOdometry:
    """Estimate pose by integrating noisy wheel encoder velocities."""

    def __init__(
        self,
        robot,
        wheel_noise=0.02,
        seed=7,
    ):
        self.x = robot.x
        self.y = robot.y
        self.theta = robot.theta
        self.wheel_radius = robot.wheel_radius
        self.wheel_base = robot.wheel_base
        self.wheel_noise = wheel_noise
        self.random = random.Random(seed)

    @property
    def pose(self):
        return self.x, self.y, self.theta

    def set_pose(self, pose):
        self.x, self.y, self.theta = pose
        self.theta = wrap_angle(self.theta)

    def update(self, omega_l, omega_r, dt):
        noisy_left = omega_l * (
            1.0 + self.random.gauss(0.0, self.wheel_noise)
        )
        noisy_right = omega_r * (
            1.0 + self.random.gauss(0.0, self.wheel_noise)
        )

        left_velocity = self.wheel_radius * noisy_left
        right_velocity = self.wheel_radius * noisy_right
        velocity = (left_velocity + right_velocity) / 2.0
        angular_velocity = (
            right_velocity - left_velocity
        ) / self.wheel_base

        midpoint_heading = self.theta + angular_velocity * dt / 2.0
        self.x += velocity * math.cos(midpoint_heading) * dt
        self.y += velocity * math.sin(midpoint_heading) * dt
        self.theta = wrap_angle(self.theta + angular_velocity * dt)
        return self.pose
