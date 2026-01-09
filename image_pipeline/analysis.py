"""
CST435: Parallel and Cloud Computing - Assignment 2
Performance Analysis Module

This module contains:
- Speedup and efficiency calculations
- Amdahl's Law analysis
- Scalability analysis
- Trade-offs comparison
"""


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
    """Print detailed analysis and observations including Amdahl's Law."""
    print("\n" + "="*80)
    print("ANALYSIS AND OBSERVATIONS")
    print("="*80)
    
    mp_times = {k: v for k, v in all_times.items() if 'multiprocessing' in k}
    tp_times = {k: v for k, v in all_times.items() if 'threadpool' in k}
    
    # Section 1: Best multiprocessing result
    if mp_times:
        _print_multiprocessing_summary(mp_times, sequential_time)
    
    # Section 2: Best threadpool result
    if tp_times:
        _print_threadpool_summary(tp_times, sequential_time)
    
    # Section 3: Key findings
    _print_key_findings()
    
    # Section 4: Amdahl's Law
    if mp_times:
        _print_amdahls_law(mp_times, sequential_time)
    
    # Section 5: Scalability
    if len(mp_times) >= 2:
        _print_scalability_analysis(mp_times, sequential_time)
    
    # Section 6: Trade-offs
    _print_tradeoffs_table()
    
    # Section 7: Conclusion
    if mp_times and tp_times:
        _print_conclusion(mp_times, tp_times)
    
    print("\n" + "="*80)


def _print_multiprocessing_summary(mp_times, sequential_time):
    """Print multiprocessing.Pool summary."""
    best_mp = min(mp_times.items(), key=lambda x: x[1])
    workers = int(best_mp[0].split('_')[-2])
    speedup, efficiency = calculate_metrics(sequential_time, best_mp[1], workers)
    
    print(f"\n1. MULTIPROCESSING.POOL (Best: {workers} workers)")
    print(f"   - Time: {best_mp[1]:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}%")
    print("   - Uses separate processes → Bypasses GIL → TRUE parallelism")
    print("   - Each worker has its own memory space and Python interpreter")


def _print_threadpool_summary(tp_times, sequential_time):
    """Print ThreadPoolExecutor summary."""
    best_tp = min(tp_times.items(), key=lambda x: x[1])
    workers = int(best_tp[0].split('_')[-2])
    speedup, efficiency = calculate_metrics(sequential_time, best_tp[1], workers)
    
    print(f"\n2. THREADPOOLEXECUTOR (Best: {workers} workers)")
    print(f"   - Time: {best_tp[1]:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2f}%")
    print("   - Uses threads → LIMITED by GIL → Constrained parallelism for CPU-bound tasks")
    print("   - All threads share the same memory space and Python interpreter")


def _print_key_findings():
    """Print key findings section."""
    print("\n3. KEY FINDINGS:")
    print("   - Image processing is CPU-BOUND (requires significant computation)")
    print("   - Python's GIL allows only ONE thread to execute Python bytecode at a time")
    print("   - multiprocessing.Pool BYPASSES GIL by using separate processes")
    print("   - ThreadPoolExecutor is LIMITED by GIL for CPU-bound tasks")
    print("   - For I/O-bound tasks (file reading, network), ThreadPoolExecutor would excel")


def _print_amdahls_law(mp_times, sequential_time):
    """Print Amdahl's Law analysis."""
    print("\n" + "-"*80)
    print("4. AMDAHL'S LAW ANALYSIS")
    print("-"*80)
    print("   Amdahl's Law: Speedup = 1 / ((1 - P) + P/N)")
    print("   Where: P = parallelizable fraction, N = number of processors")
    print("")
    
    best_mp = min(mp_times.items(), key=lambda x: x[1])
    workers = int(best_mp[0].split('_')[-2])
    observed_speedup, _ = calculate_metrics(sequential_time, best_mp[1], workers)
    
    if workers > 1 and observed_speedup > 1:
        # Calculate parallel fraction P
        P = (workers * (observed_speedup - 1)) / (observed_speedup * (workers - 1))
        P = min(max(P, 0), 1)
        serial_fraction = 1 - P
        
        print(f"   Based on multiprocessing.Pool ({workers} workers, {observed_speedup:.2f}x speedup):")
        print(f"   • Estimated Parallel Fraction (P): {P*100:.1f}%")
        print(f"   • Estimated Serial Fraction (1-P): {serial_fraction*100:.1f}%")
        print("")
        
        if serial_fraction > 0:
            max_speedup = 1 / serial_fraction
            print(f"   Theoretical Maximum Speedup (infinite processors): {max_speedup:.2f}x")
        
        # Predicted vs Observed table
        print("")
        print("   Predicted vs Observed Speedup (Amdahl's Law):")
        print(f"   {'Workers':<10} {'Predicted':<12} {'Observed':<12} {'Difference':<12}")
        print("   " + "-"*46)
        
        for key, time_val in sorted(mp_times.items()):
            n = int(key.split('_')[-2])
            predicted = 1 / ((1 - P) + P/n)
            observed, _ = calculate_metrics(sequential_time, time_val, n)
            diff = observed - predicted
            print(f"   {n:<10} {predicted:<12.2f}x {observed:<12.2f}x {diff:+.2f}x")
    
    print("")
    print("   BOTTLENECKS IDENTIFIED:")
    print("   • Serial portions: Image loading, result collection, process/thread creation")
    print("   • multiprocessing overhead: Data serialization (pickling), IPC communication")
    print("   • ThreadPool advantage: No serialization needed (shared memory)")
    print("   • OpenCV/NumPy: Release GIL during C-level computation")


def _print_scalability_analysis(mp_times, sequential_time):
    """Print scalability analysis."""
    print("\n" + "-"*80)
    print("5. SCALABILITY ANALYSIS")
    print("-"*80)
    
    sorted_mp = sorted(mp_times.items(), key=lambda x: int(x[0].split('_')[-2]))
    workers_list = [int(k.split('_')[-2]) for k, v in sorted_mp]
    speedups = [calculate_metrics(sequential_time, v, int(k.split('_')[-2]))[0] for k, v in sorted_mp]
    
    if len(speedups) >= 2:
        scaling_factor = speedups[-1] / speedups[0] if speedups[0] > 0 else 0
        worker_ratio = workers_list[-1] / workers_list[0]
        scaling_efficiency = (scaling_factor / worker_ratio) * 100
        
        print(f"   • Scaling from {workers_list[0]} to {workers_list[-1]} workers:")
        print(f"     Speedup increased: {speedups[0]:.2f}x → {speedups[-1]:.2f}x")
        print(f"     Scaling efficiency: {scaling_efficiency:.1f}%")
        
        if scaling_efficiency > 80:
            print("     → EXCELLENT scalability (>80%)")
        elif scaling_efficiency > 60:
            print("     → GOOD scalability (60-80%)")
        elif scaling_efficiency > 40:
            print("     → MODERATE scalability (40-60%)")
        else:
            print("     → LIMITED scalability (<40%) - diminishing returns")


def _print_tradeoffs_table():
    """Print trade-offs comparison table."""
    print("\n" + "-"*80)
    print("6. TRADE-OFFS BETWEEN PARADIGMS")
    print("-"*80)
    print("   ┌─────────────────────┬────────────────────┬────────────────────┐")
    print("   │ Aspect              │ multiprocessing    │ ThreadPoolExecutor │")
    print("   ├─────────────────────┼────────────────────┼────────────────────┤")
    print("   │ GIL Bypass          │ ✓ Yes              │ ✗ No               │")
    print("   │ Memory Overhead     │ High (separate)    │ Low (shared)       │")
    print("   │ Startup Cost        │ High               │ Low                │")
    print("   │ Data Serialization  │ Required (pickle)  │ Not needed         │")
    print("   │ Best For            │ CPU-bound tasks    │ I/O-bound tasks    │")
    print("   │ Communication       │ IPC (slow)         │ Shared memory      │")
    print("   └─────────────────────┴────────────────────┴────────────────────┘")


def _print_conclusion(mp_times, tp_times):
    """Print conclusion section."""
    best_mp_time = min(mp_times.values())
    best_tp_time = min(tp_times.values())
    
    if best_mp_time < best_tp_time:
        improvement = ((best_tp_time - best_mp_time) / best_tp_time) * 100
        print(f"\n7. CONCLUSION:")
        print(f"   multiprocessing.Pool is {improvement:.1f}% faster than ThreadPoolExecutor")
        print("   → Confirms that process-based parallelism is superior for CPU-bound tasks")
    else:
        winner_diff = ((best_mp_time - best_tp_time) / best_mp_time) * 100
        print(f"\n7. CONCLUSION:")
        print(f"   ThreadPoolExecutor is {winner_diff:.1f}% faster than multiprocessing.Pool")
        print("   → OpenCV/NumPy release GIL, allowing thread parallelism")
        print("   → Lower overhead (no pickling) benefits smaller workloads")
        print("   → For larger workloads, multiprocessing may outperform")
