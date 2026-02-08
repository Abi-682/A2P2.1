"""Compare UCS vs A* on randomized warehouse configurations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ucs_pathfinder import SearchResult as UcsResult, ucs_path
from astar_pathfinder import SearchResult as AstarResult, astar_path
from warehouse_env import WarehouseEnv


@dataclass
class TrialStats:
    path_len: int
    nodes_expanded: int
    frontier_max: int
    time_sec: float


SearchResult = UcsResult | AstarResult


def _run_segment(
    grid: List[str],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    algo: str,
) -> SearchResult:
    """Execute either UCS or A* on a path from start to goal."""
    if algo == "ucs":
        return ucs_path(grid, start, goal)
    if algo == "astar":
        return astar_path(grid, start, goal)
    raise ValueError(f"Unknown algo: {algo}")


def _aggregate(results: List[SearchResult]) -> TrialStats:
    """Sum path lengths, node expansions, and timings across search segments."""
    path_len = sum(max(0, len(r.path) - 1) for r in results)
    nodes_expanded = sum(r.nodes_expanded for r in results)
    frontier_max = max((r.frontier_max for r in results), default=0)
    time_sec = sum(r.time_sec for r in results)
    return TrialStats(path_len, nodes_expanded, frontier_max, time_sec)


def run_trials(num_trials: int = 10) -> Dict[str, List[TrialStats]]:
    """Execute UCS and A* on 10 random warehouse layouts, measuring performance on start→pickup→dropoff paths."""
    env = WarehouseEnv()
    stats: Dict[str, List[TrialStats]] = {"ucs": [], "astar": []}

    for _ in range(num_trials):
        obs = env.reset(randomize=True)
        start = obs["robot_pos"]
        pickup = obs["pickup_pos"]
        dropoff = obs["dropoff_pos"]
        if pickup is None or dropoff is None:
            continue

        for algo in ["ucs", "astar"]:
            seg1 = _run_segment(env.grid, start, pickup, algo)
            seg2 = _run_segment(env.grid, pickup, dropoff, algo)
            stats[algo].append(_aggregate([seg1, seg2]))

    return stats


def summary_table(stats: Dict[str, List[TrialStats]]) -> None:
    """Display mean path length, node expansions, frontier size, and wall-clock time for UCS vs A* across all trials."""
    def arr(algo: str, attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in stats[algo]])

    rows = []
    for algo in ["ucs", "astar"]:
        mean_time_ms = arr(algo, "time_sec").mean() * 1000.0
        rows.append(
            [
                algo.upper(),
                arr(algo, "path_len").mean(),
                arr(algo, "nodes_expanded").mean(),
                arr(algo, "frontier_max").mean(),
                mean_time_ms,
            ]
        )

    headers = ["Algorithm", "Mean Path Len", "Mean Nodes", "Mean Frontier", "Mean Time (ms)"]
    print("\nSummary (10 trials):")
    print("{:<10} {:>14} {:>12} {:>14} {:>14}".format(*headers))
    for row in rows:
        print(
            "{:<10} {:>14.2f} {:>12.2f} {:>14.2f} {:>14.2f}".format(
                row[0], row[1], row[2], row[3], row[4]
            )
        )


def plot_nodes(stats: Dict[str, List[TrialStats]]) -> None:
    """Compare mean node expansions between UCS and A*."""
    means = [
        np.mean([s.nodes_expanded for s in stats["ucs"]]),
        np.mean([s.nodes_expanded for s in stats["astar"]]),
    ]
    labels = ["UCS", "A*"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, means, color=["#4e79a7", "#f28e2b"])
    plt.ylabel("Mean Nodes Expanded")
    plt.title("UCS vs A* (10 Random Trials)")
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the Full Comparison
    stats = run_trials(num_trials=10)

    # Display summary statistics
    summary_table(stats)

    # Visualize node expansion comparison
    plot_nodes(stats)
