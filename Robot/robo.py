class Robot:

    def __init__(
        self,
        x=1,
        y=1,
        theta=0
    ):

        # Robot position in the world
        self.x = x
        self.y = y

        # Robot direction in radians
        # theta = 0 means the robot faces right
        self.theta = theta

        # Matplotlib drawing references
        self.body_patch = None
        self.heading_patch = None