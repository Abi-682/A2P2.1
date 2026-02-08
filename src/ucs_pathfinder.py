"""
Uniform Cost Search (UCS) pathfinder for warehouse environment.
Uses a priority queue ordered by path cost g(n) (number of steps).
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Tuple
from warehouse_env import WarehouseEnv


class UCSPathfinder:
    """Uniform Cost Search pathfinder."""
    
    def __init__(self, grid: List[str], start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        """
        Initialize UCS pathfinder.
        
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
        Perform Uniform Cost Search.
        
        Returns:
            Tuple of (path, statistics_dict)
            path: List of positions from start to goal, or None if no path exists
            statistics_dict: Dict with search statistics
        """
        start_time = time.time()
        
        # Priority queue: (cost, node)
        frontier = [(0, self.start_pos)]
        explored = set()
        parent = {self.start_pos: None}
        cost_so_far = {self.start_pos: 0}
        
        path = None
        
        while frontier:
            self.frontier_size = len(frontier)
            current_cost, current = heapq.heappop(frontier)
            
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
                    new_cost = cost_so_far[current] + 1  # Each step costs 1
                    
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        parent[neighbor] = current
                        heapq.heappush(frontier, (new_cost, neighbor))
        
        self.computation_time = time.time() - start_time
        
        stats = {
            "nodes_expanded": self.nodes_expanded,
            "path_length": self.path_length,
            "computation_time": self.computation_time,
            "frontier_size": self.frontier_size,
            "path_found": path is not None,
        }
        
        return path, stats


def find_path_ucs(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]] | None, Dict]:
    """
    Find a path using Uniform Cost Search.
    
    Args:
        grid: The warehouse grid
        start: Starting position
        goal: Goal position
    
    Returns:
        Tuple of (path, statistics_dict)
    """
    pathfinder = UCSPathfinder(grid, start, goal)
    return pathfinder.search()
