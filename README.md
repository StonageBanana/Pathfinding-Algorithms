# Pathfinding: Classical Search vs. Reinforcement Learning vs. Sampling-Based Planners

**One interface, twelve planners, one interactive grid — so you can watch the difference instead of reading about it.**

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-ee4c2c)
![Pygame](https://img.shields.io/badge/Pygame-visualization-green)

---

## What this is

Three families of algorithms solve the same problem — get from A to B through obstacles — with completely different assumptions:

- **Classical search** knows the map and guarantees optimality
- **Reinforcement learning** knows nothing and has to learn the map by falling over
- **Sampling-based planners** don't try to be optimal, they try to be fast in high-dimensional space

This repository implements all three behind a **single abstract interface**, then renders them on the same interactive grid so their behaviour is directly comparable. Watching Dijkstra flood outward while A\* drives straight at the goal, or watching a DQN spend thousands of episodes rediscovering what BFS knew immediately, makes the tradeoffs obvious in a way a complexity table doesn't.

---

## Architecture

Every planner subclasses one abstract base class:

```python
class PathPlanningAlgorithm(ABC):
    @abstractmethod
    def run(self) -> None: ...
    @abstractmethod
    def extract_path(self) -> List[Tuple[int, int]]: ...
```

That constraint is the design point. A tabular Q-learner, a priority-queue graph search, and a randomised tree planner have nothing in common internally — forcing them behind `run()` / `extract_path()` means the visualizer and benchmarking harness treat them identically, and adding a new planner requires touching exactly one new file.

Code is fully type-hinted with Doxygen-style docstrings throughout.

---

## Implemented planners

**Classical search** — A\* (Manhattan-distance heuristic), Dijkstra's, BFS, DFS, Greedy Best-First, Bidirectional Search, Jump Point Search

**Reinforcement learning** — Q-Learning, SARSA, Deep Q-Network

**Sampling-based** — RRT, RRT\*

### The DQN is a real one

Not a wrapper around a library call. `DQN.py` implements from scratch:

- Experience replay buffer (`collections.deque`) to break sample correlation
- A **separate target network**, periodically synced — without it the Bellman targets chase the network that's producing them and training diverges
- ε-greedy exploration with decay schedule
- MSE loss on the Bellman residual
- Automatic CUDA/CPU device handling

State is `(x, y)` into a 128–128 MLP. That's deliberately minimal — the goal was a correct, readable DQN sitting next to tabular SARSA and Q-Learning for comparison, not a competitive agent.

### What the comparison shows

- **A\* vs Dijkstra** — same guarantee, dramatically different node expansion once a heuristic is admissible
- **Greedy Best-First** — fast, and confidently wrong on concave obstacles
- **Q-Learning vs SARSA** — off-policy vs on-policy divergence shows up clearly near hazards; SARSA takes the safer path
- **RRT vs RRT\*** — RRT finds *a* path fast; RRT\* rewires toward optimality and you can watch it happen
- **RL vs classical** — the RL agents eventually match paths that A\* found instantly, which is exactly the point: they got there without ever being told the map

---

## Interactive visualization

Real-time Pygame grid:

- Click to place start, goal, and obstacles
- Adjustable animation speed to step through expansion
- Random maze generation for repeatable comparisons
- Live rendering of the frontier / explored set / final path

---

## Layout

```
A_star.py          Dijkstras.py       QLearning.py
SARSA.py           DQN.py             RRT.py
RRT_star.py        Algorithms/        Maps/
Output/
```

Flat module layout — each planner is a self-contained, independently runnable file.

---

## Running it

```bash
pip install pygame numpy torch matplotlib
python A_star.py
```

---

## Context

Built to consolidate coursework in **search, graph algorithms, and reinforcement learning** into a single comparable framework rather than a folder of disconnected assignments.

**Author:** Simhadri Mohana Kushal · [LinkedIn](https://www.linkedin.com/in/mohana-kuhsal-simhadri-177205200/) · [GitHub](https://github.com/StonageBanana)
