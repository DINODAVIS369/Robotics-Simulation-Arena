
class KeyboardController:

    def __init__(self, robot):
        self.robot = robot

        # Wheel angular velocities
        self.forward_speed = 10
        self.turn_speed = 10

        # Current wheel commands
        self.omega_l = 0
        self.omega_r = 0

    def key_press(self, event):

        if event.key == "w":
            # Move forward
            self.omega_l = self.forward_speed
            self.omega_r = self.forward_speed

        elif event.key == "s":
            # Move backward
            self.omega_l = -self.forward_speed
            self.omega_r = -self.forward_speed

        elif event.key == "a":
            # Rotate left
            self.omega_l = -self.turn_speed
            self.omega_r = self.turn_speed

        elif event.key == "d":
            # Rotate right
            self.omega_l = self.turn_speed
            self.omega_r = -self.turn_speed

        elif event.key == " ":
            # Stop
            self.omega_l = 0
            self.omega_r = 0

    def key_release(self, event):

        # Stop when movement key is released
        if event.key in ["w", "s", "a", "d"]:
            self.omega_l = 0
            self.omega_r = 0

    def update(self, dt):

        self.robot.update_position(
            self.omega_l,
            self.omega_r,
            dt
        )

