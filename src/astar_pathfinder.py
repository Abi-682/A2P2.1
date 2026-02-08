"""
A* search pathfinder for warehouse environment.
Uses priority queue ordered by f(n) = g(n) + h(n).
h(n) is the Manhattan distance heuristic.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Tuple
from warehouse_env import WarehouseEnv


class AStarPathfinder:
    """A* search pathfinder with Manhattan distance heuristic."""
    
    def __init__(self, grid: List[str], start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        """
        Initialize A* pathfinder.
        
        Args:
            grid: The warehouse grid
            start_pos: Starting position
            goal_pos: Goal position
        """
        self.grid = grid
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.height = len(grid)
        self.width = len(grid[0])
        
        # Statistics
        self.nodes_expanded = 0
        self.frontier_size = 0
        self.computation_time = 0.0
        self.path_length = 0
        
    def is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is a wall."""
        r, c = pos
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return True
        return self.grid[r][c] == "#"
    
    def heuristic(self, pos: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic.
        h(n) = |x_n - x_goal| + |y_n - y_goal|
        """
        r, c = pos
        goal_r, goal_c = self.goal_pos
        return abs(r - goal_r) + abs(c - goal_c)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors (Manhattan distance moves)."""
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N, E, S, W
            nr, nc = r + dr, c + dc
            if not self.is_wall((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors
    
    def search(self) -> Tuple[List[Tuple[int, int]] | None, Dict]:
        """
        Perform A* search.
        
        Returns:
            Tuple of (path, statistics_dict)
            path: List of positions from start to goal, or None if no path exists
            statistics_dict: Dict with search statistics
        """
        start_time = time.time()
        
        # Priority queue: (f_score, counter, node)
        # Counter is used to break ties in priority queue
        counter = 0
        frontier = [(self.heuristic(self.start_pos), counter, self.start_pos)]
        explored = set()
        parent = {self.start_pos: None}
        g_score = {self.start_pos: 0}
        
        path = None
        
        while frontier:
            self.frontier_size = len(frontier)
            f_score, _, current = heapq.heappop(frontier)
            
            if current in explored:
                continue
                
            explored.add(current)
            self.nodes_expanded += 1
            
            # Goal test
            if current == self.goal_pos:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                self.path_length = len(path)
                break
            
            # Expand neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in explored:
                    tentative_g = g_score[current] + 1  # Each step costs 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = current
                        h_score = self.heuristic(neighbor)
                        f_score_neighbor = tentative_g + h_score
                        counter += 1
                        heapq.heappush(frontier, (f_score_neighbor, counter, neighbor))
        
        self.computation_time = time.time() - start_time
        
        stats = {
            "nodes_expanded": self.nodes_expanded,
            "path_length": self.path_length,
            "computation_time": self.computation_time,
            "frontier_size": self.frontier_size,
            "path_found": path is not None,
        }
        
        return path, stats


def find_path_astar(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]] | None, Dict]:
    """
    Find a path using A* search.
    
    Args:
        grid: The warehouse grid
        start: Starting position
        goal: Goal position
    
    Returns:
        Tuple of (path, statistics_dict)
    """
    pathfinder = AStarPathfinder(grid, start, goal)
    return pathfinder.search()
