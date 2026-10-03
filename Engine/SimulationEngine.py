import matplotlib.pyplot as plt


class SimulationEngine:

    def __init__(
        self,
        env,
        robot,
        controller,
        lidar,
        occupancy_grid_map,
        dt=0.005
    ):
        self.env = env
        self.robot = robot
        self.controller = controller
        self.lidar = lidar
        self.occupancy_grid_map = occupancy_grid_map
        self.dt = dt

        self.env.fig.canvas.mpl_connect(
            "key_press_event",
            self.controller.key_press
        )

        self.env.fig.canvas.mpl_connect(
            "key_release_event",
            self.controller.key_release
        )

    def update(self):

        # 1. Move the robot.
        self.env.update_motion(
            self.robot,
            self.controller.omega_l,
            self.controller.omega_r,
            self.dt
        )

        # 2. Update the LiDAR.
        # LiDAR needs the robot pose to perform its measurement.
        self.lidar.update(
            self.robot,
            self.dt
        )

        # 3. Update robot visualization.
        self.robot.draw(self.env.ax)

        # 4. Update LiDAR visualization.
        self.lidar.draw(
            self.env.ax,
            self.robot
        )


         # 5. Get the latest completed LiDAR scan
        scan = self.lidar.get_last_scan_data()

    # 6. Update occupancy grid
        if scan:

         self.occupancy_grid_map.update_from_scan(
        
            scan
        )

        self.occupancy_grid_map.update_visualization()

    def run(self):

        # Give Matplotlib time to create the window.
        plt.show(block=False)
        plt.pause(0.1)

        # Simulation loop.
        while plt.fignum_exists(self.env.fig.number):

            self.update()

            plt.pause(self.dt)