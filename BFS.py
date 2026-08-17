"""@package BFS
@brief Contains implementation class of Breadth-First Search Algorithm along with methods to display it on a map
"""

from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Any
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class BFSAlgorithm(PathPlanningAlgorithm):
    """
    @breif Initialises BFS object
    @param map Map(Coarse) object
    @map_path Path to load a map(Coarse) from
    """
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        super().__init__(map, map_path)
        if self.map.map_type == "Fine":
            print("[ERROR] BFS algorithm can be applied to a Coarse Map only(with lesser number of nodes)")
            exit()

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs the BFS Algorithm
        @detail BFS explores the graph in layers (unweighted hop count), which is
                what "layers" means for an unweighted grid: it guarantees the
                shortest path in *number of edges*, not in accumulated edge cost
                (contrast with Dijkstra/A*, which minimise cost on a weighted graph).
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return shortest_hop_count, shortest_path
        """
        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        source_node = node_array.index(tuple(source_point))
        target_node = node_array.index(tuple(target_point))
        print(f'Source: {source_node} -> Target: {target_node}')

        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("BFS Algorithm", live_fig, live_ax, False)
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
        hop_count = [np.inf] * len(node_array)

        frontier = deque([source_node])
        visited[source_node] = True
        hop_count[source_node] = 0

        found = source_node == target_node
        while frontier and not found:
            current = frontier.popleft()
            if visual:
                just_poped_marker.set_offsets(node_array[current])
                if current != source_node:
                    current_offsets = list(checked_markers.get_offsets())
                    current_offsets.append(list(node_array[current]))
                    checked_markers.set_offsets(current_offsets)
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.)

            for neighbour, cost in enumerate(cost_matrix[current]):
                if cost == np.inf or neighbour == current:
                    continue
                if visited[neighbour]:
                    continue
                visited[neighbour] = True
                via_node[neighbour] = current
                hop_count[neighbour] = hop_count[current] + 1
                frontier.append(neighbour)
                if visual:
                    current_offsets = list(in_queue_markers.get_offsets())
                    current_offsets.append(list(node_array[neighbour]))
                    in_queue_markers.set_offsets(current_offsets)
                if neighbour == target_node:
                    found = True
                    break

        if not found and not visited[target_node]:
            print("[INFO]: No path possible between the source and target.")
            exit()

        shortest_path = [target_node]
        prev_node = via_node[target_node]
        while prev_node != -1:
            shortest_path.append(prev_node)
            prev_node = via_node[prev_node]
        shortest_path = shortest_path[::-1]
        shortest_hop_count = hop_count[target_node]

        if visual:
            for node1 in shortest_path:
                x1, y1 = node_array[node1]
                x_data, y_data = list(final_path.get_data())
                x_data = list(x_data)
                y_data = list(y_data)
                x_data.append(x1)
                y_data.append(y1)
                final_path.set_data((x_data, y_data))
                live_fig.suptitle(f'BFS Algorithm Final Path [Hops: {shortest_hop_count}]')
                live_ax.legend()
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.1)
            sleep(2)
            plt.close(live_fig)
            plt.ioff()
        return shortest_hop_count, shortest_path

    def visualise_graph(self):
        """
        To verify the graph formed from graphiphy method
        """
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
    PPA = BFSAlgorithm(map_path="Path-Planning-Algorithms/Maps/demo_maps/30x10_B.png")
    PPA.visualise_graph()
    PPA.operate(True)
