"""@package GreedyBestFirst
@brief Contains implementation class of Greedy Best-First Search Algorithm along with methods to display it on a map
"""

from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Any
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class GreedyBestFirstAlgorithm(PathPlanningAlgorithm):
    """
    @breif Initialises Greedy Best-First object
    @param map Map(Coarse) object
    @map_path Path to load a map(Coarse) from
    """
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        super().__init__(map, map_path)
        if self.map.map_type == "Fine":
            print("[ERROR] Greedy Best-First algorithm can be applied to a Coarse Map only(with lesser number of nodes)")
            exit()

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs Greedy Best-First Search
        @detail Identical priority-queue machinery to A*, with one change: the
                queue is ordered purely by heuristic distance-to-target, with no
                accumulated path cost term. That makes it fast and directionally
                greedy -- and, unlike A*, NOT guaranteed optimal. The returned
                path_cost is the true accumulated edge cost of the path found,
                so a comparison against A*/Dijkstra on the same map/pair shows
                exactly how much optimality was traded for speed.
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return path_cost, path
        """
        def list_search(list_: List[List], index: int, search_item: Any):
            for ind, List_ in enumerate(list_):
                if List_[index] == search_item:
                    return ind
            return -1

        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        source_node = node_array.index(tuple(source_point))
        target_node = node_array.index(tuple(target_point))
        print(f'Source: {source_node} -> Target: {target_node}')

        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("Greedy Best-First Algorithm", live_fig, live_ax, False)
            live_ax.scatter(source_point[0], source_point[1], c="green")
            live_ax.scatter(target_point[0], target_point[1], c="yellow")
            for point in node_array:
                live_ax.scatter(point[0], point[1], c = 'black', alpha=0.2)
            just_poped_marker = live_ax.scatter(source_point[0], source_point[1], c = '#1d7874', zorder = 100, label = "Current Node")
            checked_markers = live_ax.scatter([], [], c = '#679289', label = "Visited", zorder = 99)
            in_queue_markers = live_ax.scatter([], [], c = '#f4c095', label = "In Queue")
            final_path, = live_ax.plot([], [], color = 'green', label='Path', zorder = 101)
            live_ax.legend()

        visited = [False] * len(node_array)
        via_node = [-1] * len(node_array)
        accumulated_cost = [np.inf] * len(node_array)
        accumulated_cost[source_node] = 0

        heuristic = lambda node: np.linalg.norm(np.array(node_array[node]) - np.array(target_point))

        check_queue = [[heuristic(source_node), source_node]]

        just_poped = None
        while check_queue:
            _, just_poped = hq.heappop(check_queue)
            if visited[just_poped]:
                continue
            visited[just_poped] = True

            if visual:
                just_poped_marker.set_offsets(node_array[just_poped])
                if just_poped != source_node:
                    current_offsets = list(checked_markers.get_offsets())
                    current_offsets.append(list(node_array[just_poped]))
                    checked_markers.set_offsets(current_offsets)
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.)

            if just_poped == target_node:
                break

            for node, cost in enumerate(cost_matrix[just_poped]):
                if cost == np.inf or node == just_poped:
                    continue
                if visited[node]:
                    continue
                if accumulated_cost[node] == np.inf:
                    accumulated_cost[node] = accumulated_cost[just_poped] + cost
                    via_node[node] = just_poped
                    hq.heappush(check_queue, [heuristic(node), node])
                    if visual:
                        current_offsets = list(in_queue_markers.get_offsets())
                        current_offsets.append(list(node_array[node]))
                        in_queue_markers.set_offsets(current_offsets)

        if just_poped != target_node:
            print("[INFO]: No path possible between the source and target.")
            exit()

        shortest_path = [target_node]
        prev_node = via_node[target_node]
        while prev_node != -1:
            shortest_path.append(prev_node)
            prev_node = via_node[prev_node]
        shortest_path = shortest_path[::-1]
        path_cost = accumulated_cost[target_node]

        if visual:
            for node1 in shortest_path:
                x1, y1 = node_array[node1]
                x_data, y_data = list(final_path.get_data())
                x_data = list(x_data)
                y_data = list(y_data)
                x_data.append(x1)
                y_data.append(y1)
                final_path.set_data((x_data, y_data))
                live_fig.suptitle(f'Greedy Best-First Final Path [Cost: {path_cost}]')
                live_ax.legend()
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.1)
            sleep(2)
            plt.close(live_fig)
            plt.ioff()
        return path_cost, shortest_path

    def visualise_graph(self):
        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        fig, ax = plt.subplots()
        self.map.show("Graph on Map", fig, ax, False)
        for node1 in range(len(node_array)):
            for node2, cost in enumerate(cost_matrix[node1]):
                if node1 == node2:
                    continue
                if cost == np.inf:
                    continue
                elif cost == 1:
                    x1, y1 = node_array[node1]
                    x2, y2 = node_array[node2]
                    ax.plot([x1, x2], [y1, y2], 'y-')
                elif cost == 2**0.5:
                    x1, y1 = node_array[node1]
                    x2, y2 = node_array[node2]
                    ax.plot([x1, x2], [y1, y2], 'r-')
        plt.show()

if __name__ == "__main__":
    PPA = GreedyBestFirstAlgorithm(map_path="Path-Planning-Algorithms/Maps/demo_maps/30x10_B.png")
    PPA.visualise_graph()
    PPA.operate(True)
