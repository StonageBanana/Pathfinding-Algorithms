import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List

class SarsaPathPlanner(PathPlanningAlgorithm):
    def __init__(self, map: Map = None, map_path: str = None, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, episodes: int = 1000) -> None:
        """
        Initializes the SARSA Path Planner
        """
        super().__init__(map, map_path)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = None

    def initialize_q_table(self):
        """Initialize the Q-table for the grid environment."""
        self.q_table = np.zeros((*self.map.size, 4))  # Four possible actions: up, down, left, right

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)  # Random action
        return np.argmax(self.q_table[state])  # Best action based on Q-values

    def take_action(self, state, action):
        """Take an action and return the next state and reward."""
        x, y = state
        if action == 0:  # Up
            next_state = (x, max(y - 1, 0))
        elif action == 1:  # Down
            next_state = (x, min(y + 1, self.map.size[1] - 1))
        elif action == 2:  # Left
            next_state = (max(x - 1, 0), y)
        elif action == 3:  # Right
            next_state = (min(x + 1, self.map.size[0] - 1), y)

        if self.map.array[next_state[1], next_state[0]] == 0:  # Obstacle
            reward = -1
            next_state = state  # Stay in place
        elif next_state == self.target_state:
            reward = 10  # Goal reward
        else:
            reward = -0.1  # Step penalty

        return next_state, reward

    def extract_path(self, source_point: Tuple[int, int], target_point: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Extract the optimal path from the Q-table and calculate the total distance.
        """
        path = [source_point]
        state = (int(source_point[0]), int(source_point[1]))
        total_distance = 0.0

        while state != self.target_state:
            action = np.argmax(self.q_table[state])
            next_state, _ = self.take_action(state, action)

            # Calculate Euclidean distance between consecutive points
            distance = np.linalg.norm(np.array(next_state) - np.array(state))
            total_distance += distance

            path.append(next_state)
            state = next_state

        return path, total_distance

    def train(self, source_point: Tuple[int, int], target_point: Tuple[int, int]):
        """
        Train the SARSA agent with an early stopping mechanism.
        """
        self.target_state = (int(target_point[0]), int(target_point[1]))
        self.initialize_q_table()

        last_paths = []
        for episode in range(self.episodes):
            state = (int(source_point[0]), int(source_point[1]))
            action = self.choose_action(state)
            total_reward = 0
            episode_path = []

            while state != self.target_state:
                next_state, reward = self.take_action(state, action)
                next_action = self.choose_action(next_state)

                # SARSA Q-value update
                self.q_table[state + (action,)] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state + (next_action,)] - self.q_table[state + (action,)]
                )

                state = next_state
                action = next_action
                episode_path.append(state)
                total_reward += reward

            # Store the path for early stopping
            last_paths.append(tuple(episode_path))
            if len(last_paths) > 3:
                last_paths.pop(0)

            # Check if the last three paths are identical
            if len(last_paths) == 3 and last_paths[0] == last_paths[1] == last_paths[2]:
                print(f"Training stopped early at episode {episode + 1} due to path repetition.")
                break

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}")

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> List[Tuple[int, int]]:
        """
        Runs the SARSA Path Planner and optionally visualizes the path.
        """
        self.train(source_point, target_point)
        path, shortest_distance = self.extract_path(source_point, target_point)

        print(f"Shortest Distance: {shortest_distance:.2f}")

        if visual:
            fig, ax = plt.subplots()
            self.map.show("SARSA Path", fig, ax, False)
            plt.scatter(source_point[0], source_point[1], c='green', label="Source")
            plt.scatter(target_point[0], target_point[1], c='yellow', label="Target")

            for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                ax.plot([x1, x2], [y1, y2], c='blue')

            # Add heading with path distance
            fig.suptitle(f"Policy Gradient\nShortest Distance: {shortest_distance:.2f}", fontsize=14, fontweight='bold')

            plt.legend()
            plt.show()

        return path


if __name__ == "__main__":
    planner = SarsaPathPlanner(map_path="Path-Planning-Algorithms/Maps/demo_maps/30x10_B.png", alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000)
    planner.operate(visual=True)
