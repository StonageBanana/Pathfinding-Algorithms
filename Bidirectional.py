"""@package Bidirectional
@brief Contains implementation class of Bidirectional Dijkstra Search Algorithm along with methods to display it on a map
"""

from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class BidirectionalAlgorithm(PathPlanningAlgorithm):
    """
    @breif Initialises Bidirectional Search object
    @param map Map(Coarse) object
    @map_path Path to load a map(Coarse) from
    """
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        super().__init__(map, map_path)
        if self.map.map_type == "Fine":
            print("[ERROR] Bidirectional algorithm can be applied to a Coarse Map only(with lesser number of nodes)")
            exit()

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs Bidirectional Dijkstra Search
        @detail Two Dijkstra searches run at once -- one growing outward from
                the source, one growing outward from the target -- alternating
                which side expands next node. Every time a node relaxed on one
                side is already settled on the other, that gives a candidate
                complete path length (mu); the search stops once neither
                frontier can possibly beat the best candidate found so far
                (standard bidirectional-Dijkstra stopping rule: pop_f + pop_b >= mu).
                This still returns the true shortest path (like Dijkstra), it
                just gets there by meeting in the middle instead of flooding
                outward from a single side.
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return shortest_distance, shortest_path
        """
        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        source_node = node_array.index(tuple(source_point))
        target_node = node_array.index(tuple(target_point))
        print(f'Source: {source_node} -> Target: {target_node}')

        n = len(node_array)
        INF = np.inf

        dist_f = [INF] * n
        dist_b = [INF] * n
        via_f = [-1] * n
        via_b = [-1] * n
        closed_f = [False] * n
        closed_b = [False] * n

        dist_f[source_node] = 0
        dist_b[target_node] = 0
        queue_f = [[0, source_node]]
        queue_b = [[0, target_node]]

        mu = INF
        meeting_node = -1

        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("Bidirectional Algorithm", live_fig, live_ax, False)
            live_ax.scatter(source_point[0], source_point[1], c="green")
            live_ax.scatter(target_point[0], target_point[1], c="yellow")
            for point in node_array:
                live_ax.scatter(point[0], point[1], c = 'black', alpha=0.2)
            fwd_markers = live_ax.scatter([], [], c = '#679289', label = "Visited (forward)", zorder = 99)
            bwd_markers = live_ax.scatter([], [], c = '#a4508b', label = "Visited (backward)", zorder = 99)
            final_path, = live_ax.plot([], [], color = 'green', label='Path', zorder = 101)
            live_ax.legend()

        def expand(queue, dist, via, closed, other_dist, other_closed):
            nonlocal mu, meeting_node
            while queue:
                d, u = hq.heappop(queue)
                if closed[u]:
                    continue
                closed[u] = True
                if visual:
                    marker = fwd_markers if dist is dist_f else bwd_markers
                    current_offsets = list(marker.get_offsets())
                    current_offsets.append(list(node_array[u]))
                    marker.set_offsets(current_offsets)
                    live_fig.canvas.draw()
                    plt.pause(0.000001)
                if other_closed[u] and dist[u] + other_dist[u] < mu:
                    mu = dist[u] + other_dist[u]
                    meeting_node = u
                for v, cost in enumerate(cost_matrix[u]):
                    if cost == INF or v == u or closed[v]:
                        continue
                    if dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        via[v] = u
                        hq.heappush(queue, [dist[v], v])
                        if other_closed[v] and dist[v] + other_dist[v] < mu:
                            mu = dist[v] + other_dist[v]
                            meeting_node = v
                return  # expanded exactly one node, hand control back to alternator

        while queue_f and queue_b:
            top_f = queue_f[0][0]
            top_b = queue_b[0][0]
            if top_f + top_b >= mu:
                break
            if top_f <= top_b:
                expand(queue_f, dist_f, via_f, closed_f, dist_b, closed_b)
            else:
                expand(queue_b, dist_b, via_b, closed_b, dist_f, closed_f)

        if meeting_node == -1:
            print("[INFO]: No path possible between the source and target.")
            exit()

        # Stitch: source -> meeting_node via via_f, then meeting_node -> target via via_b
        forward_half = [meeting_node]
        prev_node = via_f[meeting_node]
        while prev_node != -1:
            forward_half.append(prev_node)
            prev_node = via_f[prev_node]
        forward_half = forward_half[::-1]

        backward_half = []
        next_node = via_b[meeting_node]
        while next_node != -1:
            backward_half.append(next_node)
            next_node = via_b[next_node]

        shortest_path = forward_half + backward_half
        shortest_distance = mu

        if visual:
            for node1 in shortest_path:
                x1, y1 = node_array[node1]
                x_data, y_data = list(final_path.get_data())
                x_data = list(x_data)
                y_data = list(y_data)
                x_data.append(x1)
                y_data.append(y1)
                final_path.set_data((x_data, y_data))
                live_fig.suptitle(f'Bidirectional Algorithm Final Path [Distance: {shortest_distance}]')
                live_ax.legend()
                live_fig.canvas.draw()
                plt.pause(0.000001)
                sleep(0.1)
            sleep(2)
            plt.close(live_fig)
            plt.ioff()
        return shortest_distance, shortest_path

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
    PPA = BidirectionalAlgorithm(map_path="Path-Planning-Algorithms/Maps/demo_maps/30x10_B.png")
    PPA.visualise_graph()
    PPA.operate(True)
