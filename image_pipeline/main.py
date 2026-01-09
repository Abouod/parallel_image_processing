"""
CST435: Parallel and Cloud Computing - Assignment 2
Concurrent Image Processing Pipeline

This module compares two parallel processing paradigms in Python:
1. multiprocessing.Pool (process-based parallelism - bypasses GIL)
2. concurrent.futures.ThreadPoolExecutor (thread-based parallelism - limited by GIL)

For CPU-bound tasks like image processing, multiprocessing is expected to outperform
threading due to Python's Global Interpreter Lock (GIL).
"""

import os
import time
import argparse
import threading
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from image_pipeline.filters import apply_filters
from image_pipeline.utils import download_food101_subset, load_images, save_image

# Try to import psutil for CPU core monitoring (like Lab 2)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. CPU core tracking will be limited.")
    print("Install with: pip install psutil")

# Try to import matplotlib for performance graphs
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for servers without display
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Graphs will not be generated.")
    print("Install with: pip install matplotlib")

# ================= Configuration Parameters =================
DATASET_PATH = "food101_subset"
OUTPUT_PATH_MULTIPROCESSING = "output_multiprocessing"
OUTPUT_PATH_THREADPOOL = "output_threadpool"
NUM_IMAGES = 100  # Number of images to process for demonstration
WORKER_COUNTS = [2, 4]  # Different worker counts to test for performance analysis


def get_cpu_core_id():
    """Get the current CPU core ID (similar to Lab 2)."""
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            return p.cpu_num()
        except:
            return "N/A"
    return "N/A"


def get_process_info():
    """Get PID and Thread ID for tracking (similar to Lab 2)."""
    pid = os.getpid()
    tid = threading.get_ident()
    return pid, tid


def process_image_for_multiprocessing(image_data):
    """
    Process a single image using multiprocessing.Pool.
    Returns processed image along with execution metadata for analysis.
    """
    image, filename = image_data
    start_time = time.time()
    
    # Get execution context information
    pid, tid = get_process_info()
    core_id = get_cpu_core_id()
    
    # Apply all 5 filters
    processed_image = apply_filters(image)
    
    duration = time.time() - start_time
    
    return {
        'image': processed_image,
        'filename': filename,
        'pid': pid,
        'tid': tid,
        'core_id': core_id,
        'duration': duration
    }


def process_image_for_threadpool(image_data):
    """
    Process a single image using ThreadPoolExecutor.
    Returns processed image along with execution metadata for analysis.
    """
    image, filename = image_data
    start_time = time.time()
    
    # Get execution context information
    pid, tid = get_process_info()
    core_id = get_cpu_core_id()
    
    # Apply all 5 filters
    processed_image = apply_filters(image)
    
    duration = time.time() - start_time
    
    return {
        'image': processed_image,
        'filename': filename,
        'pid': pid,
        'tid': tid,
        'core_id': core_id,
        'duration': duration
    }


def run_sequential(images_data):
    """
    Sequential baseline processing (single-threaded, single-process).
    Used as baseline for calculating speedup metrics.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL PROCESSING (Baseline)")
    print("="*60)
    
    start_time = time.time()
    results = []
    
    for image_data in images_data:
        image, filename = image_data
        processed_image = apply_filters(image)
        results.append({'image': processed_image, 'filename': filename})
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Sequential processing completed in {total_time:.4f} seconds")
    print(f"Images processed: {len(results)}")
    
    return results, total_time


def run_multiprocessing_pool(images_data, num_workers, verbose=True):
    """
    Paradigm 1: multiprocessing.Pool (Process-based parallelism)
    
    Synchronization & Load Balancing:
    - Pool context manager handles process lifecycle (creation/cleanup)
    - Using optimized chunksize for better load distribution
    - Each process has its own memory space (no shared state = no locks needed)
    
    Characteristics:
    - Creates separate Python interpreter processes
    - Each process has its own memory space
    - Bypasses Python's GIL for true parallelism
    - Best for CPU-bound tasks like image processing
    """
    if verbose:
        print("\n" + "="*60)
        print(f"PARADIGM 1: multiprocessing.Pool ({num_workers} workers)")
        print("="*60)
        print("Characteristics:")
        print("  - Process-based parallelism (separate memory spaces)")
        print("  - Bypasses Python's Global Interpreter Lock (GIL)")
        print("  - Each worker is a separate Python interpreter")
        print("  - Best for CPU-bound tasks")
        print("-"*60)
    
    start_time = time.time()
    
    # Optimized chunksize for better load balancing
    # Smaller chunks = better distribution when task times vary
    # Formula: chunksize = max(1, len(data) // (num_workers * 4))
    chunksize = max(1, len(images_data) // (num_workers * 4))
    
    with Pool(processes=num_workers) as pool:
        # pool.map with optimized chunksize distributes work evenly
        # Context manager ensures proper cleanup of worker processes
        results = pool.map(process_image_for_multiprocessing, images_data, chunksize=chunksize)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        # Display execution details (similar to Lab 2 output)
        print(f"\n{'Image':<15} {'PID':<10} {'Core ID':<10} {'Time (s)':<12}")
        print("-"*50)
        
        # Count PIDs from ALL results (not just displayed ones)
        unique_pids = set(res['pid'] for res in results)
        unique_cores = set(res['core_id'] for res in results if res['core_id'] != "N/A")
        
        for res in results[:10]:  # Show first 10 for brevity
            print(f"{res['filename']:<15} {res['pid']:<10} {res['core_id']:<10} {res['duration']:.4f}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more images")
        
        print("-"*50)
        print(f"Unique PIDs observed: {len(unique_pids)} → Confirms SEPARATE PROCESSES")
        print(f"All PIDs: {sorted(unique_pids)}")
        if unique_cores:
            print(f"CPU Cores utilized: {unique_cores} → True parallel execution")
        print(f"\nTotal time: {total_time:.4f} seconds")
    
    return results, total_time


def run_threadpool_executor(images_data, num_workers, verbose=True):
    """
    Paradigm 2: concurrent.futures.ThreadPoolExecutor (Thread-based parallelism)
    
    Synchronization & Load Balancing:
    - ThreadPoolExecutor context manager handles thread lifecycle
    - executor.map() automatically distributes work across threads
    - Shared memory space - but our tasks are independent (no shared state)
    - GIL provides implicit synchronization for Python objects
    
    Characteristics:
    - Uses threads within a single process
    - Shared memory space between threads
    - Limited by Python's GIL for CPU-bound tasks
    - Better suited for I/O-bound tasks
    """
    if verbose:
        print("\n" + "="*60)
        print(f"PARADIGM 2: concurrent.futures.ThreadPoolExecutor ({num_workers} workers)")
        print("="*60)
        print("Characteristics:")
        print("  - Thread-based parallelism (shared memory)")
        print("  - LIMITED by Python's Global Interpreter Lock (GIL)")
        print("  - All threads share the same Python interpreter")
        print("  - Better for I/O-bound tasks, not ideal for CPU-bound")
        print("-"*60)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image_for_threadpool, images_data))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        # Display execution details (similar to Lab 2 output)
        print(f"\n{'Image':<15} {'PID':<10} {'TID':<20} {'Core ID':<10} {'Time (s)':<12}")
        print("-"*70)
        
        # Count PIDs/TIDs from ALL results (not just displayed ones)
        unique_pids = set(res['pid'] for res in results)
        unique_tids = set(res['tid'] for res in results)
        unique_cores = set(res['core_id'] for res in results if res['core_id'] != "N/A")
        
        for res in results[:10]:  # Show first 10 for brevity
            tid_short = str(res['tid'])[-8:]  # Last 8 digits for readability
            print(f"{res['filename']:<15} {res['pid']:<10} ...{tid_short:<16} {res['core_id']:<10} {res['duration']:.4f}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more images")
        
        print("-"*70)
        print(f"Unique PIDs observed: {len(unique_pids)} → Confirms SINGLE PROCESS (all threads share PID)")
        print(f"Unique Thread IDs: {len(unique_tids)} → Multiple threads within same process")
        if unique_cores:
            print(f"CPU Cores observed: {unique_cores} → May switch cores but GIL limits parallelism")
        print(f"\nTotal time: {total_time:.4f} seconds")
    
    return results, total_time


def calculate_metrics(sequential_time, parallel_time, num_workers):
    """Calculate speedup and efficiency metrics."""
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        efficiency = (speedup / num_workers) * 100
    else:
        speedup = 0
        efficiency = 0
    return speedup, efficiency


def print_performance_comparison(all_times, sequential_time):
    """Print comprehensive performance comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(f"{'Paradigm':<45} {'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*80)
    
    # Sequential baseline
    print(f"{'Sequential (Baseline)':<45} {'1':<10} {sequential_time:<12.4f} {'1.00x':<10} {'100.00%':<12}")
    
    # Parallel paradigms
    for key, time_val in sorted(all_times.items()):
        if 'multiprocessing' in key:
            workers = int(key.split('_')[-2])
            speedup, efficiency = calculate_metrics(sequential_time, time_val, workers)
            print(f"{'multiprocessing.Pool':<45} {workers:<10} {time_val:<12.4f} {speedup:<10.2f}x {efficiency:<12.2f}%")
        elif 'threadpool' in key:
            workers = int(key.split('_')[-2])
            speedup, efficiency = calculate_metrics(sequential_time, time_val, workers)
            print(f"{'concurrent.futures.ThreadPoolExecutor':<45} {workers:<10} {time_val:<12.4f} {speedup:<10.2f}x {efficiency:<12.2f}%")
    
    print("="*80)


def print_analysis_summary(all_times, sequential_time):
    """Print detailed analysis and observations."""
    print("\n" + "="*80)
    print("ANALYSIS AND OBSERVATIONS")
    print("="*80)
    
    # Find best performer for each paradigm
    mp_times = {k: v for k, v in all_times.items() if 'multiprocessing' in k}
    tp_times = {k: v for k, v in all_times.items() if 'threadpool' in k}
    
    if mp_times:
        best_mp = min(mp_times.items(), key=lambda x: x[1])
        workers = int(best_mp[0].split('_')[-2])
        speedup, efficiency = calculate_metrics(sequential_time, best_mp[1], workers)
        print(f"\n1. MULTIPROCESSING.POOL (Best: {workers} workers)")
        print(f"   - Time: {best_mp[1]:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}%")
        print("   - Uses separate processes → Bypasses GIL → TRUE parallelism")
        print("   - Each worker has its own memory space and Python interpreter")
    
    if tp_times:
        best_tp = min(tp_times.items(), key=lambda x: x[1])
        workers = int(best_tp[0].split('_')[-2])
        speedup, efficiency = calculate_metrics(sequential_time, best_tp[1], workers)
        print(f"\n2. THREADPOOLEXECUTOR (Best: {workers} workers)")
        print(f"   - Time: {best_tp[1]:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}%")
        print("   - Uses threads → LIMITED by GIL → Constrained parallelism for CPU-bound tasks")
        print("   - All threads share the same memory space and Python interpreter")
    
    print("\n3. KEY FINDINGS:")
    print("   - Image processing is CPU-BOUND (requires significant computation)")
    print("   - Python's GIL allows only ONE thread to execute Python bytecode at a time")
    print("   - multiprocessing.Pool BYPASSES GIL by using separate processes")
    print("   - ThreadPoolExecutor is LIMITED by GIL for CPU-bound tasks")
    print("   - For I/O-bound tasks (file reading, network), ThreadPoolExecutor would excel")
    
    if mp_times and tp_times:
        best_mp_time = min(mp_times.values())
        best_tp_time = min(tp_times.values())
        if best_mp_time < best_tp_time:
            improvement = ((best_tp_time - best_mp_time) / best_tp_time) * 100
            print(f"\n4. CONCLUSION:")
            print(f"   multiprocessing.Pool is {improvement:.1f}% faster than ThreadPoolExecutor")
            print("   → Confirms that process-based parallelism is superior for CPU-bound tasks")
        else:
            print(f"\n4. CONCLUSION:")
            print("   Results may vary based on system load and image complexity")
    
    print("\n" + "="*80)


def generate_performance_graphs(all_times, sequential_time, worker_counts, output_dir="performance_graphs"):
    """
    Generate professional performance comparison graphs.
    
    Creates:
    1. Bar chart comparing execution times
    2. Line graph showing speedup vs worker count
    3. Line graph showing efficiency vs worker count
    """
    if not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib not installed. Skipping graph generation.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    mp_times = []
    tp_times = []
    mp_speedups = []
    tp_speedups = []
    mp_efficiencies = []
    tp_efficiencies = []
    
    for workers in worker_counts:
        mp_key = f'multiprocessing_{workers}_workers'
        tp_key = f'threadpool_{workers}_workers'
        
        if mp_key in all_times:
            mp_time = all_times[mp_key]
            mp_times.append(mp_time)
            speedup, efficiency = calculate_metrics(sequential_time, mp_time, workers)
            mp_speedups.append(speedup)
            mp_efficiencies.append(efficiency)
        
        if tp_key in all_times:
            tp_time = all_times[tp_key]
            tp_times.append(tp_time)
            speedup, efficiency = calculate_metrics(sequential_time, tp_time, workers)
            tp_speedups.append(speedup)
            tp_efficiencies.append(efficiency)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
    colors = {'mp': '#2ecc71', 'tp': '#3498db', 'seq': '#e74c3c'}
    
    # ==================== GRAPH 1: Execution Time Bar Chart ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(worker_counts))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], mp_times, width, label='multiprocessing.Pool', color=colors['mp'], edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], tp_times, width, label='ThreadPoolExecutor', color=colors['tp'], edgecolor='black')
    
    # Add sequential baseline line
    ax.axhline(y=sequential_time, color=colors['seq'], linestyle='--', linewidth=2, label=f'Sequential Baseline ({sequential_time:.2f}s)')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Execution Time Comparison: multiprocessing.Pool vs ThreadPoolExecutor', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(worker_counts)
    ax.legend(loc='upper right')
    
    # Add value labels on bars
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
    
    # ==================== GRAPH 2: Speedup Line Graph ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(worker_counts, mp_speedups, 'o-', color=colors['mp'], linewidth=2, markersize=10, label='multiprocessing.Pool')
    ax.plot(worker_counts, tp_speedups, 's-', color=colors['tp'], linewidth=2, markersize=10, label='ThreadPoolExecutor')
    ax.plot(worker_counts, worker_counts, '--', color='gray', linewidth=1, label='Ideal Linear Speedup')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Speedup (T_sequential / T_parallel)', fontsize=12)
    ax.set_title('Speedup Analysis: Scaling Performance with Worker Count', fontsize=14, fontweight='bold')
    ax.set_xticks(worker_counts)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mp_s, tp_s) in enumerate(zip(mp_speedups, tp_speedups)):
        ax.annotate(f'{mp_s:.2f}x', xy=(worker_counts[i], mp_s), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.annotate(f'{tp_s:.2f}x', xy=(worker_counts[i], tp_s), xytext=(5, -15), textcoords="offset points", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=150)
    plt.close()
    
    # ==================== GRAPH 3: Efficiency Line Graph ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(worker_counts, mp_efficiencies, 'o-', color=colors['mp'], linewidth=2, markersize=10, label='multiprocessing.Pool')
    ax.plot(worker_counts, tp_efficiencies, 's-', color=colors['tp'], linewidth=2, markersize=10, label='ThreadPoolExecutor')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Ideal Efficiency (100%)')
    
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Parallel Efficiency: Utilization of Worker Resources', fontsize=14, fontweight='bold')
    ax.set_xticks(worker_counts)
    ax.set_ylim(0, max(max(mp_efficiencies), max(tp_efficiencies)) * 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mp_e, tp_e) in enumerate(zip(mp_efficiencies, tp_efficiencies)):
        ax.annotate(f'{mp_e:.1f}%', xy=(worker_counts[i], mp_e), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.annotate(f'{tp_e:.1f}%', xy=(worker_counts[i], tp_e), xytext=(5, -15), textcoords="offset points", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=150)
    plt.close()
    
    # ==================== GRAPH 4: Combined Summary (2x2 subplot) ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CST435 Assignment 2: Performance Analysis Summary', fontsize=16, fontweight='bold')
    
    # Subplot 1: Execution Time Bar Chart
    ax1 = axes[0, 0]
    x = range(len(worker_counts))
    width = 0.35
    ax1.bar([i - width/2 for i in x], mp_times, width, label='multiprocessing.Pool', color=colors['mp'])
    ax1.bar([i + width/2 for i in x], tp_times, width, label='ThreadPoolExecutor', color=colors['tp'])
    ax1.axhline(y=sequential_time, color=colors['seq'], linestyle='--', label='Sequential')
    ax1.set_xlabel('Workers')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(worker_counts)
    ax1.legend(fontsize=8)
    
    # Subplot 2: Speedup Line Graph
    ax2 = axes[0, 1]
    ax2.plot(worker_counts, mp_speedups, 'o-', color=colors['mp'], label='multiprocessing.Pool')
    ax2.plot(worker_counts, tp_speedups, 's-', color=colors['tp'], label='ThreadPoolExecutor')
    ax2.plot(worker_counts, worker_counts, '--', color='gray', label='Ideal')
    ax2.set_xlabel('Workers')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Workers')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Efficiency Line Graph
    ax3 = axes[1, 0]
    ax3.plot(worker_counts, mp_efficiencies, 'o-', color=colors['mp'], label='multiprocessing.Pool')
    ax3.plot(worker_counts, tp_efficiencies, 's-', color=colors['tp'], label='ThreadPoolExecutor')
    ax3.axhline(y=100, color='gray', linestyle='--', label='Ideal (100%)')
    ax3.set_xlabel('Workers')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Efficiency vs Workers')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_mp_idx = mp_times.index(min(mp_times))
    best_tp_idx = tp_times.index(min(tp_times))
    
    summary_text = f"""
    Performance Summary
    ═══════════════════════════════════════
    
    Sequential Baseline: {sequential_time:.4f}s
    
    multiprocessing.Pool:
      • Best: {worker_counts[best_mp_idx]} workers → {min(mp_times):.4f}s
      • Speedup: {mp_speedups[best_mp_idx]:.2f}x
      • Efficiency: {mp_efficiencies[best_mp_idx]:.1f}%
    
    ThreadPoolExecutor:
      • Best: {worker_counts[best_tp_idx]} workers → {min(tp_times):.4f}s
      • Speedup: {tp_speedups[best_tp_idx]:.2f}x
      • Efficiency: {tp_efficiencies[best_tp_idx]:.1f}%
    
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
    
    print(f"\n📊 Performance graphs saved to {output_dir}/")
    print(f"   • execution_time_comparison.png  (Bar Chart)")
    print(f"   • speedup_comparison.png         (Line Graph)")
    print(f"   • efficiency_comparison.png      (Line Graph)")
    print(f"   • performance_summary.png        (Combined Summary)")


def save_results(results, output_path, paradigm_name):
    """Save processed images to output directory."""
    os.makedirs(output_path, exist_ok=True)
    for res in results:
        save_image(res['image'], os.path.join(output_path, res['filename']))
    print(f"Saved {len(results)} images to {output_path}/")


def main():
    parser = argparse.ArgumentParser(
        description="CST435 Assignment 2: Compare multiprocessing.Pool vs ThreadPoolExecutor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m image_pipeline.main                    # Run with default worker counts [2, 4]
  python -m image_pipeline.main --workers 2 4 8    # Run with specific worker counts
  python -m image_pipeline.main --num-images 50    # Process 50 images instead of 100
        """
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        nargs='+',
        default=WORKER_COUNTS,
        help="Worker counts to test (e.g., --workers 2 4 8)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=NUM_IMAGES,
        help="Number of images to process"
    )
    args = parser.parse_args()

    print("="*80)
    print("CST435: PARALLEL AND CLOUD COMPUTING - ASSIGNMENT 2")
    print("Concurrent Image Processing Pipeline")
    print("="*80)
    print(f"\nComparing two parallel paradigms:")
    print("  1. multiprocessing.Pool      (Process-based, bypasses GIL)")
    print("  2. ThreadPoolExecutor        (Thread-based, limited by GIL)")
    print(f"\nWorker counts to test: {args.workers}")
    print(f"Number of images: {args.num_images}")
    print(f"CPU cores available: {os.cpu_count()}")
    if HAS_PSUTIL:
        print("psutil installed: Yes (CPU core tracking enabled)")
    else:
        print("psutil installed: No (install for detailed CPU tracking)")

    # 1. Download dataset
    print(f"\n{'='*60}")
    print("STEP 1: Dataset Preparation")
    print("="*60)
    download_food101_subset(DATASET_PATH, args.num_images)

    # 2. Load images
    print(f"\n{'='*60}")
    print("STEP 2: Loading Images")
    print("="*60)
    images_data = load_images(DATASET_PATH)
    print(f"Loaded {len(images_data)} images into memory")

    if not images_data:
        print("No images to process. Exiting.")
        return

    # 3. Run sequential baseline
    print(f"\n{'='*60}")
    print("STEP 3: Running Processing Pipelines")
    print("="*60)
    
    seq_results, seq_time = run_sequential(images_data)
    
    # Store all timing results
    all_times = {}
    all_results = {}

    # 4. Run both paradigms with different worker counts
    for num_workers in args.workers:
        # Paradigm 1: multiprocessing.Pool
        mp_results, mp_time = run_multiprocessing_pool(images_data, num_workers)
        all_times[f'multiprocessing_{num_workers}_workers'] = mp_time
        all_results[f'multiprocessing_{num_workers}_workers'] = mp_results
        
        # Paradigm 2: ThreadPoolExecutor
        tp_results, tp_time = run_threadpool_executor(images_data, num_workers)
        all_times[f'threadpool_{num_workers}_workers'] = tp_time
        all_results[f'threadpool_{num_workers}_workers'] = tp_results

    # 5. Save results (using the best performer's output)
    print(f"\n{'='*60}")
    print("STEP 4: Saving Processed Images")
    print("="*60)
    
    # Save from multiprocessing results
    best_mp_key = min([k for k in all_results if 'multiprocessing' in k], 
                       key=lambda x: all_times[x])
    save_results(all_results[best_mp_key], OUTPUT_PATH_MULTIPROCESSING, "multiprocessing.Pool")
    
    # Save from threadpool results
    best_tp_key = min([k for k in all_results if 'threadpool' in k],
                       key=lambda x: all_times[x])
    save_results(all_results[best_tp_key], OUTPUT_PATH_THREADPOOL, "ThreadPoolExecutor")

    # 6. Performance Analysis
    print(f"\n{'='*60}")
    print("STEP 5: Performance Analysis")
    print("="*60)
    
    print_performance_comparison(all_times, seq_time)
    print_analysis_summary(all_times, seq_time)
    
    # 7. Generate Performance Graphs
    print(f"\n{'='*60}")
    print("STEP 6: Generating Performance Graphs")
    print("="*60)
    generate_performance_graphs(all_times, seq_time, args.workers)
    
    print("\nProcessing complete!")
    print(f"  - Multiprocessing results: {OUTPUT_PATH_MULTIPROCESSING}/")
    print(f"  - ThreadPool results: {OUTPUT_PATH_THREADPOOL}/")
    print(f"  - Performance graphs: performance_graphs/")


if __name__ == "__main__":
    main()
