"""@package JPS
@brief Contains implementation class of Jump Point Search Algorithm along with methods to display it on a map

@detail JPS is the one planner in this repository that does NOT reuse
        Map.graphify()'s cost_matrix/node_array abstraction. That abstraction
        collapses the map into a generic weighted graph, which is exactly what
        JPS's speed-up throws away: JPS prunes symmetric paths using the grid's
        2D structure (straight-line "jumping" over uninteresting cells via
        forced-neighbour detection), so it operates directly on the raw pixel
        grid (self.map.array) instead.
"""

from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Optional
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# 8 directions: 4 straight, 4 diagonal
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]

class JPSAlgorithm(PathPlanningAlgorithm):
    """
    @breif Initialises JPS object
    @param map Map(Coarse) object
    @map_path Path to load a map(Coarse) from
    """
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        super().__init__(map, map_path)
        if self.map.map_type == "Fine":
            print("[ERROR] JPS algorithm can be applied to a Coarse Map only(with lesser number of nodes)")
            exit()

    def _walkable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.map.size[0] or y >= self.map.size[1]:
            return False
        return self.map.array[y][x] == 255

    def _jump(self, x: int, y: int, dx: int, dy: int, goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        @brief Recursively steps in direction (dx, dy) from (x, y) until it hits
               the goal, a forced neighbour (a jump point), a dead end, or leaves
               the walkable grid.
        """
        nx, ny = x + dx, y + dy
        if not self._walkable(nx, ny):
            return None
        if (nx, ny) == goal:
            return (nx, ny)

        if dx != 0 and dy != 0:
            # Diagonal move: check for forced neighbours on either side.
            # A neighbour is "forced" when the cell directly behind it (relative
            # to travel direction) is blocked but the cell itself is open --
            # that open cell could only be reached this fast via (nx, ny), so
            # (nx, ny) has to be recorded as a jump point.
            if (self._walkable(nx - dx, ny + dy) and not self._walkable(nx - dx, ny)) or \
               (self._walkable(nx + dx, ny - dy) and not self._walkable(nx, ny - dy)):
                return (nx, ny)
            # Diagonal jump also probes the two straight components
            if self._jump(nx, ny, dx, 0, goal) is not None or self._jump(nx, ny, 0, dy, goal) is not None:
                return (nx, ny)
        else:
            # Straight move: check for forced neighbours perpendicular to travel
            if dx != 0:
                if (self._walkable(nx + dx, ny + 1) and not self._walkable(nx, ny + 1)) or \
                   (self._walkable(nx + dx, ny - 1) and not self._walkable(nx, ny - 1)):
                    return (nx, ny)
            else:
                if (self._walkable(nx + 1, ny + dy) and not self._walkable(nx + 1, ny)) or \
                   (self._walkable(nx - 1, ny + dy) and not self._walkable(nx - 1, ny)):
                    return (nx, ny)

        return self._jump(nx, ny, dx, dy, goal)

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[float, List[Tuple[int, int]]]:
        """
        @brief Runs Jump Point Search
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return shortest_distance, shortest_path (list of (x, y) pixel coordinates,
                unlike the other planners which return node_array indices --
                JPS has no node_array to index into)
        """
        source = tuple(source_point)
        target = tuple(target_point)
        print(f'Source: {source} -> Target: {target}')

        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("JPS Algorithm", live_fig, live_ax, False)
            live_ax.scatter(source[0], source[1], c="green")
            live_ax.scatter(target[0], target[1], c="yellow")
            jump_markers = live_ax.scatter([], [], c='#f4c095', label="Jump points expanded", zorder=99)
            final_path, = live_ax.plot([], [], color='green', label='Path', zorder=101)
            live_ax.legend()

        def heuristic(node):
            return np.linalg.norm(np.array(node) - np.array(target))

        g_cost = {source: 0}
        via = {source: None}
        open_queue = [[heuristic(source), source]]
        closed = set()
        found = False

        while open_queue:
            _, current = hq.heappop(open_queue)
            if current in closed:
                continue
            closed.add(current)

            if visual:
                current_offsets = list(jump_markers.get_offsets())
                current_offsets.append(list(current))
                jump_markers.set_offsets(current_offsets)
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.)

            if current == target:
                found = True
                break

            cx, cy = current
            for dx, dy in DIRECTIONS:
                jump_point = self._jump(cx, cy, dx, dy, target)
                if jump_point is None or jump_point in closed:
                    continue
                step_cost = np.linalg.norm(np.array(jump_point) - np.array(current))
                new_g = g_cost[current] + step_cost
                if jump_point not in g_cost or new_g < g_cost[jump_point]:
                    g_cost[jump_point] = new_g
                    via[jump_point] = current
                    hq.heappush(open_queue, [new_g + heuristic(jump_point), jump_point])

        if not found:
            print("[INFO]: No path possible between the source and target.")
            exit()

        # Jump points are the only recorded waypoints; expand each straight-line
        # segment between consecutive jump points into a full pixel path so the
        # output is directly comparable (same format) to the other planners.
        waypoints = [target]
        node = via[target]
        while node is not None:
            waypoints.append(node)
            node = via[node]
        waypoints = waypoints[::-1]

        shortest_path = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            ax_, ay_ = a
            bx_, by_ = b
            steps = max(abs(bx_ - ax_), abs(by_ - ay_))
            sx = (bx_ - ax_) // steps if steps else 0
            sy = (by_ - ay_) // steps if steps else 0
            x, y = ax_, ay_
            for _ in range(steps):
                x, y = x + sx, y + sy
                shortest_path.append((x, y))

        shortest_distance = g_cost[target]

        if visual:
            for (x1, y1) in shortest_path:
                x_data, y_data = list(final_path.get_data())
                x_data = list(x_data)
                y_data = list(y_data)
                x_data.append(x1)
                y_data.append(y1)
                final_path.set_data((x_data, y_data))
                live_fig.suptitle(f'JPS Algorithm Final Path [Distance: {shortest_distance:.2f}, Jump points: {len(waypoints)}]')
                live_ax.legend()
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.03)
            sleep(2)
            plt.close(live_fig)
            plt.ioff()
        return shortest_distance, shortest_path

if __name__ == "__main__":
    PPA = JPSAlgorithm(map_path="Path-Planning-Algorithms/Maps/demo_maps/30x10_B.png")
    PPA.operate(True)
