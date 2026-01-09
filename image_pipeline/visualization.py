"""
CST435: Parallel and Cloud Computing - Assignment 2
Visualization Module

This module generates performance comparison graphs:
- Execution time bar chart
- Speedup line graph
- Efficiency line graph
- Combined summary plot
"""

import os

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for servers
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from image_pipeline.analysis import calculate_metrics


def generate_performance_graphs(all_times, sequential_time, worker_counts, output_dir="performance_graphs"):
    """
    Generate professional performance comparison graphs.
    
    Creates:
    1. Bar chart comparing execution times
    2. Line graph showing speedup vs worker count
    3. Line graph showing efficiency vs worker count
    4. Combined summary plot
    """
    if not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib not installed. Skipping graph generation.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    data = _prepare_graph_data(all_times, sequential_time, worker_counts)
    
    # Set colors
    colors = {'mp': '#2ecc71', 'tp': '#3498db', 'seq': '#e74c3c'}
    
    # Generate individual graphs
    _generate_execution_time_chart(data, sequential_time, worker_counts, colors, output_dir)
    _generate_speedup_graph(data, worker_counts, colors, output_dir)
    _generate_efficiency_graph(data, worker_counts, colors, output_dir)
    _generate_summary_plot(data, sequential_time, worker_counts, colors, output_dir)
    
    print(f"\n📊 Performance graphs saved to {output_dir}/")
    print(f"   • execution_time_comparison.png  (Bar Chart)")
    print(f"   • speedup_comparison.png         (Line Graph)")
    print(f"   • efficiency_comparison.png      (Line Graph)")
    print(f"   • performance_summary.png        (Combined Summary)")


def _prepare_graph_data(all_times, sequential_time, worker_counts):
    """Prepare data for graphing."""
    data = {
        'mp_times': [],
        'tp_times': [],
        'mp_speedups': [],
        'tp_speedups': [],
        'mp_efficiencies': [],
        'tp_efficiencies': []
    }
    
    for workers in worker_counts:
        mp_key = f'multiprocessing_{workers}_workers'
        tp_key = f'threadpool_{workers}_workers'
        
        if mp_key in all_times:
            mp_time = all_times[mp_key]
            data['mp_times'].append(mp_time)
            speedup, efficiency = calculate_metrics(sequential_time, mp_time, workers)
            data['mp_speedups'].append(speedup)
            data['mp_efficiencies'].append(efficiency)
        
        if tp_key in all_times:
            tp_time = all_times[tp_key]
            data['tp_times'].append(tp_time)
            speedup, efficiency = calculate_metrics(sequential_time, tp_time, workers)
            data['tp_speedups'].append(speedup)
            data['tp_efficiencies'].append(efficiency)
    
    return data


def _generate_execution_time_chart(data, sequential_time, worker_counts, colors, output_dir):
    """Generate execution time bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(worker_counts))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], data['mp_times'], width, 
                   label='multiprocessing.Pool', color=colors['mp'], edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], data['tp_times'], width, 
                   label='ThreadPoolExecutor', color=colors['tp'], edgecolor='black')
    
    ax.axhline(y=sequential_time, color=colors['seq'], linestyle='--', linewidth=2, 
               label=f'Sequential Baseline ({sequential_time:.2f}s)')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Execution Time Comparison: multiprocessing.Pool vs ThreadPoolExecutor', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(worker_counts)
    ax.legend(loc='upper right')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'), dpi=150)
    plt.close()


def _generate_speedup_graph(data, worker_counts, colors, output_dir):
    """Generate speedup line graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(worker_counts, data['mp_speedups'], 'o-', color=colors['mp'], 
            linewidth=2, markersize=10, label='multiprocessing.Pool')
    ax.plot(worker_counts, data['tp_speedups'], 's-', color=colors['tp'], 
            linewidth=2, markersize=10, label='ThreadPoolExecutor')
    ax.plot(worker_counts, worker_counts, '--', color='gray', linewidth=1, 
            label='Ideal Linear Speedup')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Speedup (T_sequential / T_parallel)', fontsize=12)
    ax.set_title('Speedup Analysis: Scaling Performance with Worker Count', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(worker_counts)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mp_s, tp_s) in enumerate(zip(data['mp_speedups'], data['tp_speedups'])):
        ax.annotate(f'{mp_s:.2f}x', xy=(worker_counts[i], mp_s), 
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.annotate(f'{tp_s:.2f}x', xy=(worker_counts[i], tp_s), 
                    xytext=(5, -15), textcoords="offset points", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=150)
    plt.close()


def _generate_efficiency_graph(data, worker_counts, colors, output_dir):
    """Generate efficiency line graph."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(worker_counts, data['mp_efficiencies'], 'o-', color=colors['mp'], 
            linewidth=2, markersize=10, label='multiprocessing.Pool')
    ax.plot(worker_counts, data['tp_efficiencies'], 's-', color=colors['tp'], 
            linewidth=2, markersize=10, label='ThreadPoolExecutor')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, 
               label='Ideal Efficiency (100%)')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency: Utilization of Worker Resources', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(worker_counts)
    ax.set_ylim(0, max(max(data['mp_efficiencies']), max(data['tp_efficiencies'])) * 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mp_e, tp_e) in enumerate(zip(data['mp_efficiencies'], data['tp_efficiencies'])):
        ax.annotate(f'{mp_e:.1f}%', xy=(worker_counts[i], mp_e), 
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.annotate(f'{tp_e:.1f}%', xy=(worker_counts[i], tp_e), 
                    xytext=(5, -15), textcoords="offset points", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=150)
    plt.close()


def _generate_summary_plot(data, sequential_time, worker_counts, colors, output_dir):
    """Generate combined summary plot (2x2 subplot)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CST435 Assignment 2: Performance Analysis Summary', fontsize=16, fontweight='bold')
    
    x = range(len(worker_counts))
    width = 0.35
    
    # Subplot 1: Execution Time Bar Chart
    ax1 = axes[0, 0]
    ax1.bar([i - width/2 for i in x], data['mp_times'], width, 
            label='multiprocessing.Pool', color=colors['mp'])
    ax1.bar([i + width/2 for i in x], data['tp_times'], width, 
            label='ThreadPoolExecutor', color=colors['tp'])
    ax1.axhline(y=sequential_time, color=colors['seq'], linestyle='--', label='Sequential')
    ax1.set_xlabel('Workers')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(worker_counts)
    ax1.legend(fontsize=8)
    
    # Subplot 2: Speedup Line Graph
    ax2 = axes[0, 1]
    ax2.plot(worker_counts, data['mp_speedups'], 'o-', color=colors['mp'], label='multiprocessing.Pool')
    ax2.plot(worker_counts, data['tp_speedups'], 's-', color=colors['tp'], label='ThreadPoolExecutor')
    ax2.plot(worker_counts, worker_counts, '--', color='gray', label='Ideal')
    ax2.set_xlabel('Workers')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Workers')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Efficiency Line Graph
    ax3 = axes[1, 0]
    ax3.plot(worker_counts, data['mp_efficiencies'], 'o-', color=colors['mp'], label='multiprocessing.Pool')
    ax3.plot(worker_counts, data['tp_efficiencies'], 's-', color=colors['tp'], label='ThreadPoolExecutor')
    ax3.axhline(y=100, color='gray', linestyle='--', label='Ideal (100%)')
    ax3.set_xlabel('Workers')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Efficiency vs Workers')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_mp_idx = data['mp_times'].index(min(data['mp_times']))
    best_tp_idx = data['tp_times'].index(min(data['tp_times']))
    
    summary_text = f"""
    Performance Summary
    ═══════════════════════════════════════
    
    Sequential Baseline: {sequential_time:.4f}s
    
    multiprocessing.Pool:
      • Best: {worker_counts[best_mp_idx]} workers → {min(data['mp_times']):.4f}s
      • Speedup: {data['mp_speedups'][best_mp_idx]:.2f}x
      • Efficiency: {data['mp_efficiencies'][best_mp_idx]:.1f}%
    
    ThreadPoolExecutor:
      • Best: {worker_counts[best_tp_idx]} workers → {min(data['tp_times']):.4f}s
      • Speedup: {data['tp_speedups'][best_tp_idx]:.2f}x
      • Efficiency: {data['tp_efficiencies'][best_tp_idx]:.1f}%
    
    Key Insight:
      Process-based parallelism (multiprocessing)
      bypasses Python's GIL for TRUE parallelism,
      while thread-based parallelism is constrained.
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=150)
    plt.close()
