# Pathfinding: Classical Search vs. Reinforcement Learning vs. Sampling-Based Planners

**One interface, twelve planners, one interactive grid — so you can watch the difference instead of reading about it.**

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-ee4c2c)
![Live visualizer](https://img.shields.io/badge/live--visualizer-HTML%2FJS-orange)

---

## What this is

Three families of algorithms solve the same problem — get from A to B through obstacles — with completely different assumptions:

- **Classical search** knows the map and guarantees optimality
- **Reinforcement learning** knows nothing and has to learn the map by falling over
- **Sampling-based planners** don't try to be optimal, they try to be fast in high-dimensional space

This repository implements all three behind a **common interface**, then renders them on the same interactive grid so their behaviour is directly comparable. Watching Dijkstra flood outward while A\* drives straight at the goal, or watching a DQN spend thousands of episodes rediscovering what BFS knew immediately, makes the tradeoffs obvious in a way a complexity table doesn't.

---

## Architecture

Every planner subclasses one base class:

```python
class PathPlanningAlgorithm:
    def __init__(self, map: Map = None, map_path: str = None) -> None: ...

    def operate(self, visual: bool = False) -> List[Tuple[int, int]]:
        """Gets source/target by mouse click, then calls run()."""

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int],
            visual: bool = False) -> Tuple[int, List[int]]:
        """To be defined in the subclasses."""
```

That constraint is the design point. A tabular Q-learner, a priority-queue graph search, and a randomised tree planner have nothing in common internally — forcing them behind `run()` means the visualizer and benchmarking harness treat them identically, and adding a new planner requires touching exactly one new file.

Code is fully type-hinted with Doxygen-style docstrings throughout.

---

## Implemented planners

**Classical search** — A\* (Manhattan-distance heuristic), Dijkstra's, BFS, DFS, Greedy Best-First, Bidirectional Search, Jump Point Search

**Reinforcement learning** — Q-Learning, SARSA, Deep Q-Network

**Sampling-based** — RRT, RRT\*

That's 7 + 3 + 2 = 12 planners. All seven classical-search planners share `Map.graphify()`'s cost_matrix/node_array graph abstraction except Jump Point Search, which operates directly on the raw pixel grid instead (JPS's speed-up comes from exploiting 2D grid structure, which the abstracted graph throws away). BFS, Bidirectional Search, and JPS were verified against an independently-computed shortest path across dozens of random maps before being added, and are optimal in hops (BFS) or cost (Bidirectional, JPS); Greedy Best-First and DFS are intentionally non-optimal by design and documented as such in their own docstrings.

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
- **A\*/Dijkstra vs Bidirectional vs JPS** — three different ways to guarantee the same optimal cost; the visited-node counts (and, on the live visualizer, the shape of the search) differ sharply even though the answer doesn't
- **BFS vs the weighted planners** — BFS minimises edge *count*, not edge *cost*, so on a map with diagonal moves it can return a different (worse-cost) path than A\*/Dijkstra even though both are "correct" by their own definition
- **Greedy Best-First** — fast, and confidently wrong on concave obstacles
- **DFS** — valid but not remotely optimal; included as the baseline "wrong" search to make the others legible by contrast
- **Q-Learning vs SARSA** — off-policy vs on-policy divergence shows up clearly near hazards; SARSA takes the safer path
- **RRT vs RRT\*** — RRT finds *a* path fast; RRT\* rewires toward optimality and you can watch it happen
- **RL vs classical** — the RL agents eventually match paths that A\* found instantly, which is exactly the point: they got there without ever being told the map

---

## Interactive visualization

Two ways to watch a search run live:

- **In Python** — every planner's `run(..., visual=True)` renders the frontier expansion and final path with matplotlib in real time (see `if __name__ == "__main__"` in any planner file).
- **`live_visualizer.html`** — a standalone, dependency-free browser page (open it directly, no install). Pick an algorithm from the dropdown, generate a random maze or draw walls by right-click-dragging, place source/target by clicking, and hit Run to watch the search animate on an HTML canvas with live node-expansion, a speed slider, and a final cost/timing readout. Implements BFS, DFS, Greedy Best-First, A\*, Dijkstra, Bidirectional, and JPS in JavaScript (the classical-search family; the RL and sampling-based planners are Python-only for now).

---

## Layout

```
A_star.py           Dijkstras.py        BFS.py
DFS.py               GreedyBestFirst.py  Bidirectional.py
JPS.py                QLearning.py        SARSA.py
DQN.py                 RRT.py              RRT_star.py
live_visualizer.html    Algorithms/         Maps/
Output/
```

Flat module layout — each planner is a self-contained, independently runnable file.

---

## Running it

```bash
pip install numpy torch matplotlib opencv-python scipy
python A_star.py
```

Or, for the live browser visualizer, just open `live_visualizer.html` — no install required.

---

## Context

Built to consolidate coursework in **search, graph algorithms, and reinforcement learning** into a single comparable framework rather than a folder of disconnected assignments.

**Author:** Simhadri Mohana Kushal · [LinkedIn](https://www.linkedin.com/in/mohana-kuhsal-simhadri-177205200/) · [GitHub](https://github.com/StonageBanana)
