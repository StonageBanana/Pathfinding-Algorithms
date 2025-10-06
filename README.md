# Pathfinding Algorithms Visualization

A comprehensive Python implementation and visualization of various pathfinding algorithms using Pygame. This project demonstrates how different algorithms find paths between points on a grid, with customizable obstacles and real-time visualization.

![Pathfinding Algorithms](https://img.shields.io/badge/Pathfinding-Algorithms-blue)
![Python](https://img.shields.io/badge/Python-3.6%2B-green)
![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-orange)

## 🚀 Features

- **Multiple Algorithms**: Implementations of 7 different pathfinding algorithms
- **Interactive Grid**: Click to place start/end points and obstacles
- **Real-time Visualization**: Watch algorithms explore the grid step by step
- **Performance Metrics**: Compare algorithm efficiency and path length
- **Customizable Settings**: Adjust grid size, animation speed, and more

## 📋 Implemented Algorithms

| Algorithm | Guaranteed Shortest Path | Weighted | Use Case |
|-----------|-------------------------|----------|----------|
| **A*** Search | ✅ | ✅ | Optimal pathfinding with heuristics |
| **Dijkstra's Algorithm** | ✅ | ✅ | General purpose shortest path |
| **Breadth-First Search (BFS)** | ✅ | ❌ | Unweighted graphs, few obstacles |
| **Depth-First Search (DFS)** | ❌ | ❌ | Maze exploration, not optimal |
| **Greedy Best-First Search** | ❌ | ❌ | Fast but suboptimal |
| **Bidirectional Search** | ✅ | ❌ | Faster search from both ends |
| **Jump Point Search** | ✅ | ✅ | Optimized for uniform-cost grids |

## 🛠️ Installation

### Prerequisites
- Python 3.6 or higher
- Pygame library

### Setup
1. Clone the repository:
```
git clone https://github.com/StonageBanana/Pathfinding-Algorithms.git
cd Pathfinding-Algorithms
```

2. Install required dependencies:
```
pip install pygame
```

## 🎮 How to Use
1. Run the application:
```
python main.py
```

2. Controls:
Left Click: Place start point (green), end point (red), or obstacles (black)
Right Click: Remove nodes
Spacebar: Start the selected algorithm
C: Clear the current path and reset
R: Generate random maze
1-7: Select different algorithms
+/−: Adjust visualization speed

3. Algorithm Selection:
Press number keys 1-7 to switch between algorithms
Watch the visualization to understand how each algorithm works

## 📊 Algorithm Details
### A* Search
* Type: Informed search
* Heuristic: Manhattan distance
* Best for: Most pathfinding scenarios
* Complexity: O(b^d) with good heuristic

### Dijkstra's Algorithm
* Type: Uninformed search
* Approach: Explores all directions equally
* Best for: Weighted graphs
* Complexity: O((V+E) log V)

### Breadth-First Search (BFS)
* Type: Uninformed search
* Approach: Explores neighbors first
* Best for: Unweighted graphs
* Complexity: O(V+E)

### Depth-First Search (DFS)
* Type: Uninformed search
* Approach: Explores depth first
* Note: Doesn't guarantee shortest path
* Complexity: O(V+E)

### Greedy Best-First Search
* Type: Informed search
* Approach: Always moves toward goal
* Note: Can get stuck in local minima
* Complexity: O(b^m)

### Bidirectional Search
* Type: Uninformed search
* Approach: Searches from both start and end
* Best for: Large grids
* Complexity: O(b^(d/2))

### Jump Point Search
* Type: Optimized A*
* Approach: Skips symmetric paths
* Best for: Uniform cost grids
* Complexity: O(b^d) but faster in practice

## 🎨 Color Scheme
* Green: Start node
* Red: End node
* Black: Obstacle/Wall
* Blue: Visited nodes
* Light Blue: Nodes in frontier
* Yellow: Final path
* White: Empty space

## 📁 Project Structure
```text
Pathfinding-Algorithms/
├── main.py              # Main application file
├── algorithms/          # Algorithm implementations
│   ├── a_star.py
│   ├── dijkstra.py
│   ├── bfs.py
│   ├── dfs.py
│   ├── greedy_bfs.py
│   ├── bidirectional.py
│   └── jump_point.py
├── components/          # UI and grid components
│   ├── grid.py
│   └── button.py
└── utils/              # Utility functions
    ├── constants.py
    └── helpers.py
```

## 🔧 Customization
You can easily modify the project by:
1. Changing grid size: Edit GRID_WIDTH and GRID_HEIGHT in constants
2. Adding new algorithms: Create new files in the algorithms/ directory
3. Modifying colors: Update color constants in the configuration
4. Adjusting heuristics: Modify heuristic functions in A* and Greedy BFS

## 🤝 Contributing
Contributions are welcome! Feel free to:
* Add new pathfinding algorithms
* Improve the visualization
* Fix bugs or optimize performance
* Add new features or customization options
