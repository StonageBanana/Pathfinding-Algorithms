import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNPathPlanner(PathPlanningAlgorithm):
    def __init__(self, map: Map = None, map_path: str = None, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995, lr: float = 0.001, batch_size: int = 64, memory_size: int = 10000,
                 episodes: int = 1000, target_update: int = 10) -> None:
        """
        Initializes the DQN Path Planner
        """
        super().__init__(map, map_path)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.episodes = episodes
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=memory_size)

        input_dim = 2  # State is the (x, y) coordinates
        output_dim = 4  # Actions: up, down, left, right

        self.policy_net = DQNetwork(input_dim, output_dim).to(self.device)
        self.target_net = DQNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(range(4))  # Random action
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

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

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Sample from memory and train the policy network."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, source_point: Tuple[int, int], target_point: Tuple[int, int]):
        """Train the DQN agent."""
        self.target_state = (int(target_point[0]), int(target_point[1]))

        for episode in range(self.episodes):
            state = (int(source_point[0]), int(source_point[1]))
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                done = next_state == self.target_state
                self.store_transition(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                self.replay()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}")

    def extract_path(self, source_point: Tuple[int, int], target_point: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Extract the optimal path using the trained policy network.
        """
        path = [source_point]
        state = (int(source_point[0]), int(source_point[1]))
        total_distance = 0.0

        while state != self.target_state:
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action = torch.argmax(self.policy_net(state_tensor)).item()
            next_state, _ = self.take_action(state, action)

            distance = np.linalg.norm(np.array(next_state) - np.array(state))
            total_distance += distance

            path.append(next_state)
            state = next_state

        return path, total_distance

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> List[Tuple[int, int]]:
        """Run the DQN Path Planner and optionally visualize the path."""
        self.train(source_point, target_point)
        path, shortest_distance = self.extract_path(source_point, target_point)

        print(f"Shortest Distance: {shortest_distance:.2f}")

        if visual:
            fig, ax = plt.subplots()
            self.map.show("DQN Path", fig, ax, False)
            plt.scatter(source_point[0], source_point[1], c='green', label="Source")
            plt.scatter(target_point[0], target_point[1], c='yellow', label="Target")

            for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                ax.plot([x1, x2], [y1, y2], c='blue')

            fig.suptitle(f"DQN Path Planning\nShortest Distance: {shortest_distance:.2f}", fontsize=14, fontweight='bold')
            plt.legend()
            plt.show()

        return path


if __name__ == "__main__":
    planner = DQNPathPlanner(map_path='Maps/demo_maps/30x10_B.png', episodes=500)
    planner.operate(visual=True)
