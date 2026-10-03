import math
from .robo import Robot
from matplotlib.patches import Circle,RegularPolygon


class CircleRobot(Robot):

  def __init__(self, x=1, y=1, theta=0):
    super().__init__(x, y, theta)
    

    self.radius = 0.4  # Set the radius of the robot
 
  def draw(self, ax):
        marker_radius = self.radius * 0.4
        # -----------------------------------------
        # FIRST DRAW
        # -----------------------------------------

        if self.body_patch is None:

            self.body_patch = Circle(
                (self.x, self.y),
                self.radius,
                color="blue"
            )

            ax.add_patch(self.body_patch)

            self.head_x = self.x + self.radius * math.cos(self.theta)
            self.head_y = self.y + self.radius * math.sin(self.theta)
            
            self.heading_marker = RegularPolygon(
                (self.head_x, self.head_y),
                numVertices=3,
                radius=marker_radius,
                orientation=self.theta-math.pi / 2,
                color="red"
            )

            ax.add_patch(self.heading_marker)


        # -----------------------------------------
        # UPDATE DRAWING
        # -----------------------------------------

        else:

            # Move the existing body
            self.body_patch.center = (
                self.x,
                self.y,
            
            )
        

            import math
from .robo import Robot
from matplotlib.patches import Circle,RegularPolygon


class CircleRobot(Robot):

  def __init__(self, x=1, y=1, theta=0):
    super().__init__(x, y, theta)
    

    self.radius = 0.4  # Set the radius of the robot
 
  def draw(self, ax):
        marker_radius = self.radius * 0.4
        # -----------------------------------------
        # FIRST DRAW
        # -----------------------------------------

        if self.body_patch is None:

            self.body_patch = Circle(
                (self.x, self.y),
                self.radius,
                color="blue"
            )

            ax.add_patch(self.body_patch)

            self.head_x = self.x + self.radius * math.cos(self.theta)
            self.head_y = self.y + self.radius * math.sin(self.theta)

            self.heading_marker = RegularPolygon(
                (self.head_x, self.head_y),
                numVertices=3,
                radius=marker_radius,
                orientation=self.theta-math.pi / 2,
                color="red"
            )

            ax.add_patch(self.heading_marker)


        # -----------------------------------------
        # UPDATE DRAWING
        # -----------------------------------------

        else:

            # Move the existing body
            self.body_patch.center = (
                self.x,
                self.y,
            
            )
        

            
            
            self.head_x = self.x + self.radius * math.cos(self.theta)
            self.head_y = self.y + self.radius * math.sin(self.theta)
            


            # Rotate heading triangle
            self.heading_marker.orientation = self.theta-math.pi / 2


            
            self.heading_marker.xy =(
                self.head_x,
                self.head_y
            )



            