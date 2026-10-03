
from matplotlib.patches import Rectangle


class WorldGeometry:

    def __init__(self):
        self.rectangles = []

    def add_rectangle(self, x, y, width, height):
        rectangle = (x, y, width, height)
        self.rectangles.append(rectangle)

    def draw(self, ax):
        for x, y, width, height in self.rectangles:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    color="black"
                )
            )

    def circle_collision(self, center_x, center_y, radius):

        for x, y, width, height in self.rectangles:

            closest_x = max(x, min(center_x, x + width))
            closest_y = max(y, min(center_y, y + height))

            distance_x = center_x - closest_x
            distance_y = center_y - closest_y

            distance = (distance_x**2 + distance_y**2)**0.5

            if distance <= radius:
                return True

        return False