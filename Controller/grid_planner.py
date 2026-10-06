import heapq
import math

import numpy as np


class GridPlanner:
    """Grid search using A*, Dijkstra, Greedy Best-First, or Theta*."""

    ALGORITHMS = {
        "astar": "A*",
        "dijkstra": "Dijkstra",
        "greedy": "Greedy Best-First",
        "theta": "Theta*",
    }

    def __init__(
        self,
        occupancy_grid_map,
        robot_radius,
        unknown_cost=2.0,
        algorithm="astar",
    ):
        self.map = occupancy_grid_map
        self.robot_radius = robot_radius
        self.unknown_cost = unknown_cost
        self.algorithm = self._validate_algorithm(algorithm)

    @property
    def algorithm_name(self):
        return self.ALGORITHMS[self.algorithm]

    def set_algorithm(self, algorithm):
        self.algorithm = self._validate_algorithm(algorithm)

    @classmethod
    def _validate_algorithm(cls, algorithm):
        normalized = algorithm.lower().replace("*", "star").replace("-", "_").replace(" ", "_")
        aliases = {
            "a_star": "astar",
            "astar": "astar",
            "dijkstra": "dijkstra",
            "greedy": "greedy",
            "greedy_best_first": "greedy",
            "thetastar": "theta",
            "theta_star": "theta",
            "theta": "theta",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unknown planner {algorithm!r}; choose one of "
                f"{', '.join(cls.ALGORITHMS.values())}"
            )
        return aliases[normalized]

    def _blocked_cells(self):
        occupied = self.map.grid == 1
        inflation_cells = math.ceil(
            self.robot_radius / self.map.resolution
        )
        blocked = occupied.copy()

        for offset_y in range(-inflation_cells, inflation_cells + 1):
            for offset_x in range(-inflation_cells, inflation_cells + 1):
                if math.hypot(offset_x, offset_y) * self.map.resolution > self.robot_radius:
                    continue

                source_y_start = max(0, -offset_y)
                source_y_end = min(self.map.grid_height, self.map.grid_height - offset_y)
                source_x_start = max(0, -offset_x)
                source_x_end = min(self.map.grid_width, self.map.grid_width - offset_x)

                target_y_start = max(0, offset_y)
                target_y_end = target_y_start + source_y_end - source_y_start
                target_x_start = max(0, offset_x)
                target_x_end = target_x_start + source_x_end - source_x_start

                blocked[target_y_start:target_y_end, target_x_start:target_x_end] |= (
                    occupied[source_y_start:source_y_end, source_x_start:source_x_end]
                )

        return blocked

    def plan(self, start, goal, algorithm=None):
        """Return a list of (x, y) world-space cell centers, or [] if unreachable."""
        selected_algorithm = (
            self.algorithm if algorithm is None else self._validate_algorithm(algorithm)
        )
        start_cell = self.map.world_to_grid(*start)
        goal_cell = self.map.world_to_grid(*goal)
        if start_cell is None or goal_cell is None:
            return []

        blocked = self._blocked_cells()

        def valid(cell):
            x, y = cell
            return (
                0 <= x < self.map.grid_width
                and 0 <= y < self.map.grid_height
                and not blocked[y, x]
            )

        if not valid(start_cell) or not valid(goal_cell):
            return []

        def heuristic(cell):
            dx = abs(cell[0] - goal_cell[0])
            dy = abs(cell[1] - goal_cell[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

        frontier = [(self._priority(0.0, heuristic(start_cell), selected_algorithm), 0.0, start_cell)]
        came_from = {}
        costs = {start_cell: 0.0}
        directions = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        )

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal_cell:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                if selected_algorithm == "theta":
                    cells = self._smooth_theta_cells(cells, valid)
                return [self.map.grid_to_world(x, y) for x, y in cells]

            if current_cost > costs[current]:
                continue

            for dx, dy, step_cost in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not valid(neighbor):
                    continue

                if dx and dy and (
                    not valid((current[0] + dx, current[1]))
                    or not valid((current[0], current[1] + dy))
                ):
                    continue

                edge_cost = self._edge_cost(current, neighbor, step_cost)
                parent = current
                parent_cost = costs[current]
                if selected_algorithm == "theta" and current in came_from:
                    candidate_parent = came_from[current]
                    if self._line_is_clear(candidate_parent, neighbor, valid):
                        distance = math.hypot(
                            neighbor[0] - candidate_parent[0],
                            neighbor[1] - candidate_parent[1],
                        )
                        edge_cost = self._line_cost(candidate_parent, neighbor, distance)
                        parent = candidate_parent
                        parent_cost = costs[candidate_parent]

                next_cost = parent_cost + edge_cost
                if next_cost >= costs.get(neighbor, math.inf):
                    continue

                costs[neighbor] = next_cost
                came_from[neighbor] = parent
                priority = self._priority(
                    next_cost,
                    heuristic(neighbor),
                    selected_algorithm,
                )
                heapq.heappush(
                    frontier,
                    (priority, next_cost, neighbor),
                )

        return []

    def _priority(self, path_cost, heuristic, algorithm):
        if algorithm == "dijkstra":
            return path_cost
        if algorithm == "greedy":
            return heuristic
        return path_cost + heuristic

    def _cell_penalty(self, cell):
        value = self.map.grid[cell[1], cell[0]]
        return self.unknown_cost if value == -1 else 1.0

    def _edge_cost(self, start, end, distance):
        return distance * (
            self._cell_penalty(start) + self._cell_penalty(end)
        ) / 2.0

    def _line_cost(self, start, end, distance):
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        penalties = [
            self._cell_penalty(
                (
                    round(start[0] + (end[0] - start[0]) * index / steps),
                    round(start[1] + (end[1] - start[1]) * index / steps),
                )
            )
            for index in range(steps + 1)
        ]
        return distance * sum(penalties) / len(penalties)

    @staticmethod
    def _line_is_clear(start, end, valid):
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        previous = start
        for index in range(1, steps + 1):
            cell = (
                round(start[0] + (end[0] - start[0]) * index / steps),
                round(start[1] + (end[1] - start[1]) * index / steps),
            )
            if not valid(cell):
                return False
            if cell[0] != previous[0] and cell[1] != previous[1]:
                if not valid((cell[0], previous[1])) or not valid(
                    (previous[0], cell[1])
                ):
                    return False
            previous = cell
        return True

    def _smooth_theta_cells(self, cells, valid):
        if len(cells) < 3:
            return cells
        smoothed = [cells[0]]
        anchor_index = 0
        while anchor_index < len(cells) - 1:
            next_index = len(cells) - 1
            while next_index > anchor_index + 1 and not self._line_is_clear(
                cells[anchor_index],
                cells[next_index],
                valid,
            ):
                next_index -= 1
            smoothed.append(cells[next_index])
            anchor_index = next_index
        return smoothed
