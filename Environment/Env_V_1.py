
import matplotlib.pyplot as plt

from Geometry.world_geometry import WorldGeometry


class Environment:

    def __init__(self, width=10, height=10):

        self.width = width
        self.height = height

        # Create the physical world geometry.
        self.geometry = WorldGeometry()

        self.create_world()

        # Create the visualization.
        self.fig, self.ax = plt.subplots()

        self.visualization()
        self.geometry.draw(self.ax)

        self.ax.set_title("2D Autonomous Robot Environment")

    def create_world(self):

        # Boundary walls.
        self.geometry.add_rectangle(
            0, 0, self.width, 0.1
        )

        self.geometry.add_rectangle(
            0, 0, 0.1, self.height
        )

        self.geometry.add_rectangle(
            self.width - 0.1, 0, 0.1, self.height
        )

        self.geometry.add_rectangle(
            0, self.height - 0.1, self.width, 0.1
        )

        # Internal walls.
        self.geometry.add_rectangle(2, 2, 0.1, 6)
        self.geometry.add_rectangle(2, 2, 6, 0.1)
        self.geometry.add_rectangle(8, 4, 0.1, 4)
        self.geometry.add_rectangle(4, 8, 6, 0.1)
        self.geometry.add_rectangle(4, 4, 4, 0.1)
        self.geometry.add_rectangle(4, 4, 0.1, 2)
        self.geometry.add_rectangle(4, 6, 2, 0.1)

    def visualization(self):

        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.ax.set_xlabel("X axis (m)")
        self.ax.set_ylabel("Y axis (m)")

        self.ax.set_title("Autonomous Robot Environment")

    def collision(self, robot_x, robot_y):

        return self.geometry.circle_collision(
            robot_x,
            robot_y,
            self.robot_radius
        )

    def update_motion(self, robot, omega_l, omega_r, dt):

        new_x, new_y, new_theta = robot.calculate_motion(
            omega_l,
            omega_r,
            dt
        )

        if not self.collision(new_x, new_y):

            robot.x = new_x
            robot.y = new_y

        # Update heading even if translation is blocked.
        robot.theta = new_theta

    def show(self):
        plt.show()