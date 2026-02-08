"""
Comparison script for UCS and A* search algorithms on warehouse pathfinding tasks.
Tests both algorithms on 10 randomized warehouse configurations.
Generates statistics and visualizations.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Tuple

# Add src directory to path
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path)

from warehouse_env import WarehouseEnv
from ucs_pathfinder import find_path_ucs
from astar_pathfinder import find_path_astar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def run_comparison_trial(trial_num: int) -> Dict:
    """
    Run a single trial comparing UCS and A* on randomized warehouse config.
    
    Args:
        trial_num: Trial number (for logging)
    
    Returns:
        Dict with results for this trial
    """
    # Create environment and randomize
    env = WarehouseEnv()
    obs = env.reset(randomize=True)
    
    start_pos = obs["robot_pos"]
    pickup_pos = obs["pickup_pos"]
    dropoff_pos = obs["dropoff_pos"]
    
    # Skip this trial if we couldn't get valid positions
    if not pickup_pos or not dropoff_pos:
        print(f"Trial {trial_num}: Invalid positions, skipping")
        return None
    
    print(f"\nTrial {trial_num}:")
    print(f"  Start: {start_pos}, Pickup: {pickup_pos}, Dropoff: {dropoff_pos}")
    
    # ===== UCS: Start to Pickup =====
    ucs_path_1, ucs_stats_1 = find_path_ucs(env.grid, start_pos, pickup_pos)
    print(f"  UCS (start->pickup): path_len={ucs_stats_1['path_length']}, "
          f"nodes_exp={ucs_stats_1['nodes_expanded']}, "
          f"time={ucs_stats_1['computation_time']:.6f}s")
    
    # ===== A*: Start to Pickup =====
    astar_path_1, astar_stats_1 = find_path_astar(env.grid, start_pos, pickup_pos)
    print(f"  A*  (start->pickup): path_len={astar_stats_1['path_length']}, "
          f"nodes_exp={astar_stats_1['nodes_expanded']}, "
          f"time={astar_stats_1['computation_time']:.6f}s")
    
    # Check optimality for segment 1
    optimality_1 = "✓" if ucs_stats_1['path_length'] == astar_stats_1['path_length'] else "✗"
    print(f"  Optimality (segment 1): {optimality_1}")
    
    # ===== UCS: Pickup to Dropoff =====
    ucs_path_2, ucs_stats_2 = find_path_ucs(env.grid, pickup_pos, dropoff_pos)
    print(f"  UCS (pickup->dropoff): path_len={ucs_stats_2['path_length']}, "
          f"nodes_exp={ucs_stats_2['nodes_expanded']}, "
          f"time={ucs_stats_2['computation_time']:.6f}s")
    
    # ===== A*: Pickup to Dropoff =====
    astar_path_2, astar_stats_2 = find_path_astar(env.grid, pickup_pos, dropoff_pos)
    print(f"  A*  (pickup->dropoff): path_len={astar_stats_2['path_length']}, "
          f"nodes_exp={astar_stats_2['nodes_expanded']}, "
          f"time={astar_stats_2['computation_time']:.6f}s")
    
    # Check optimality for segment 2
    optimality_2 = "✓" if ucs_stats_2['path_length'] == astar_stats_2['path_length'] else "✗"
    print(f"  Optimality (segment 2): {optimality_2}")
    
    return {
        "trial": trial_num,
        "start_pos": start_pos,
        "pickup_pos": pickup_pos,
        "dropoff_pos": dropoff_pos,
        
        # Segment 1: start -> pickup
        "ucs_path_len_1": ucs_stats_1['path_length'],
        "ucs_nodes_exp_1": ucs_stats_1['nodes_expanded'],
        "ucs_time_1": ucs_stats_1['computation_time'],
        
        "astar_path_len_1": astar_stats_1['path_length'],
        "astar_nodes_exp_1": astar_stats_1['nodes_expanded'],
        "astar_time_1": astar_stats_1['computation_time'],
        
        "optimal_1": ucs_stats_1['path_length'] == astar_stats_1['path_length'],
        
        # Segment 2: pickup -> dropoff
        "ucs_path_len_2": ucs_stats_2['path_length'],
        "ucs_nodes_exp_2": ucs_stats_2['nodes_expanded'],
        "ucs_time_2": ucs_stats_2['computation_time'],
        
        "astar_path_len_2": astar_stats_2['path_length'],
        "astar_nodes_exp_2": astar_stats_2['nodes_expanded'],
        "astar_time_2": astar_stats_2['computation_time'],
        
        "optimal_2": ucs_stats_2['path_length'] == astar_stats_2['path_length'],
        
        # Totals
        "ucs_total_nodes": ucs_stats_1['nodes_expanded'] + ucs_stats_2['nodes_expanded'],
        "astar_total_nodes": astar_stats_1['nodes_expanded'] + astar_stats_2['nodes_expanded'],
        "ucs_total_time": ucs_stats_1['computation_time'] + ucs_stats_2['computation_time'],
        "astar_total_time": astar_stats_1['computation_time'] + astar_stats_2['computation_time'],
    }


def main():
    """Main comparison runner."""
    print("=" * 70)
    print("UCS vs A* Pathfinding Comparison on Randomized Warehouse Configs")
    print("=" * 70)
    
    # Run 10 trials
    num_trials = 10
    results = []
    
    for i in range(1, num_trials + 1):
        result = run_comparison_trial(i)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results!")
        return
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Overall statistics
    print(f"\nTotal trials: {len(df)}")
    print(f"All paths optimal (both segments): {(df['optimal_1'] & df['optimal_2']).sum()}/{len(df)}")
    
    # Nodes expanded statistics
    print("\n--- Nodes Expanded ---")
    print(f"UCS (mean):   {df['ucs_total_nodes'].mean():.2f} ± {df['ucs_total_nodes'].std():.2f}")
    print(f"A*  (mean):   {df['astar_total_nodes'].mean():.2f} ± {df['astar_total_nodes'].std():.2f}")
    print(f"Reduction:    {((df['ucs_total_nodes'] - df['astar_total_nodes']) / df['ucs_total_nodes'] * 100).mean():.1f}%")
    
    # Computation time statistics
    print("\n--- Computation Time (seconds) ---")
    print(f"UCS (mean):   {df['ucs_total_time'].mean()*1000:.4f} ± {df['ucs_total_time'].std()*1000:.4f} ms")
    print(f"A*  (mean):   {df['astar_total_time'].mean()*1000:.4f} ± {df['astar_total_time'].std()*1000:.4f} ms")
    print(f"Speedup:      {(df['ucs_total_time'] / df['astar_total_time']).mean():.2f}x")
    
    # Path length verification
    print("\n--- Path Length (optimality check) ---")
    print(f"UCS path lengths agree with A*: {(df['ucs_path_len_1'] == df['astar_path_len_1']).sum() + (df['ucs_path_len_2'] == df['astar_path_len_2']).sum()}/{len(df)*2}")
    
    # Create visualizations
    create_visualizations(df)
    
    # Create summary table
    create_summary_table(df)
    
    print("\n✓ Comparison complete!")


def create_visualizations(df: pd.DataFrame):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('UCS vs A* Search Comparison', fontsize=16, fontweight='bold')
    
    # 1. Nodes Expanded (bar chart)
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['ucs_total_nodes'], width, label='UCS', alpha=0.8)
    ax.bar(x + width/2, df['astar_total_nodes'], width, label='A*', alpha=0.8)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Nodes Expanded')
    ax.set_title('Nodes Expanded per Trial')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(1, len(df)+1)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Mean nodes expanded (comparison)
    ax = axes[0, 1]
    means = [df['ucs_total_nodes'].mean(), df['astar_total_nodes'].mean()]
    stds = [df['ucs_total_nodes'].std(), df['astar_total_nodes'].std()]
    algorithms = ['UCS', 'A*']
    ax.bar(algorithms, means, yerr=stds, capsize=10, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Mean Nodes Expanded')
    ax.set_title('Average Nodes Expanded (with std dev)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 5, f'{mean:.1f}', ha='center', fontweight='bold')
    
    # 3. Computation time
    ax = axes[1, 0]
    ax.plot(range(1, len(df)+1), df['ucs_total_time']*1000, 'o-', label='UCS', linewidth=2, markersize=8)
    ax.plot(range(1, len(df)+1), df['astar_total_time']*1000, 's-', label='A*', linewidth=2, markersize=8)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Computation Time (ms)')
    ax.set_title('Computation Time per Trial')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Efficiency ratio (UCS nodes / A* nodes)
    ax = axes[1, 1]
    efficiency_ratio = df['ucs_total_nodes'] / (df['astar_total_nodes'] + 1)  # +1 to avoid division by zero
    ax.bar(range(1, len(df)+1), efficiency_ratio, alpha=0.8, color='#2ca02c')
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Equal Efficiency')
    ax.set_xlabel('Trial')
    ax.set_ylabel('UCS Nodes / A* Nodes')
    ax.set_title('Efficiency Ratio (higher = UCS less efficient)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(1, len(df)+1))
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_dir, 'ucs_vs_astar_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()


def create_summary_table(df: pd.DataFrame):
    """Create summary statistics table."""
    summary_data = {
        'Metric': [
            'Mean Nodes Expanded',
            'Std Dev Nodes Expanded',
            'Min Nodes Expanded',
            'Max Nodes Expanded',
            'Mean Computation Time (ms)',
            'Std Dev Time (ms)',
            'Path Optimality (both segments)',
            'Mean Path Length (seg 1)',
            'Mean Path Length (seg 2)',
        ],
        'UCS': [
            f"{df['ucs_total_nodes'].mean():.1f}",
            f"{df['ucs_total_nodes'].std():.1f}",
            f"{df['ucs_total_nodes'].min():.0f}",
            f"{df['ucs_total_nodes'].max():.0f}",
            f"{df['ucs_total_time'].mean()*1000:.4f}",
            f"{df['ucs_total_time'].std()*1000:.4f}",
            f"{(df['optimal_1'] & df['optimal_2']).sum()}/{len(df)}",
            f"{df['ucs_path_len_1'].mean():.1f}",
            f"{df['ucs_path_len_2'].mean():.1f}",
        ],
        'A*': [
            f"{df['astar_total_nodes'].mean():.1f}",
            f"{df['astar_total_nodes'].std():.1f}",
            f"{df['astar_total_nodes'].min():.0f}",
            f"{df['astar_total_nodes'].max():.0f}",
            f"{df['astar_total_time'].mean()*1000:.4f}",
            f"{df['astar_total_time'].std()*1000:.4f}",
            f"{(df['optimal_1'] & df['optimal_2']).sum()}/{len(df)}",
            f"{df['astar_path_len_1'].mean():.1f}",
            f"{df['astar_path_len_2'].mean():.1f}",
        ],
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved table: {summary_path}")
    
    # Also save detailed results
    df_save = df[[
        'trial', 
        'ucs_path_len_1', 'astar_path_len_1',
        'ucs_path_len_2', 'astar_path_len_2',
        'ucs_nodes_exp_1', 'astar_nodes_exp_1',
        'ucs_nodes_exp_2', 'astar_nodes_exp_2',
        'ucs_total_nodes', 'astar_total_nodes',
        'ucs_total_time', 'astar_total_time',
    ]]
    detailed_path = os.path.join(output_dir, 'detailed_results.csv')
    df_save.to_csv(detailed_path, index=False)
    print(f"✓ Saved detailed results: {detailed_path}")
    
    # Additional insights
    print("\n" + "=" * 70)
    print("INSIGHTS")
    print("=" * 70)
    
    avg_reduction = ((df['ucs_total_nodes'] - df['astar_total_nodes']) / df['ucs_total_nodes'] * 100).mean()
    print(f"\n✓ A* reduces nodes expanded by ~{avg_reduction:.1f}% on average")
    
    avg_speedup = (df['ucs_total_time'] / df['astar_total_time']).mean()
    print(f"✓ A* is ~{avg_speedup:.2f}x faster than UCS on average")
    
    all_optimal = (df['optimal_1'] & df['optimal_2']).all()
    if all_optimal:
        print(f"✓ Both algorithms find optimal paths in all {len(df)} trials")
    else:
        print(f"✓ Path lengths match in {(df['optimal_1'] & df['optimal_2']).sum()}/{len(df)} trials")


if __name__ == '__main__':
    main()
