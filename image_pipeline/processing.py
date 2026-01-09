"""
CST435: Parallel and Cloud Computing - Assignment 2
Image Processing Module

This module contains the parallel processing implementations:
- Sequential baseline
- multiprocessing.Pool (process-based parallelism)
- ThreadPoolExecutor (thread-based parallelism)
"""

import os
import time
import threading
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from image_pipeline.filters import apply_filters

# Try to import psutil for CPU core monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


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
    chunksize = max(1, len(images_data) // (num_workers * 4))
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_image_for_multiprocessing, images_data, chunksize=chunksize)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        _print_multiprocessing_results(results, total_time)
    
    return results, total_time


def _print_multiprocessing_results(results, total_time):
    """Print execution details for multiprocessing."""
    print(f"\n{'Image':<15} {'PID':<10} {'Core ID':<10} {'Time (s)':<12}")
    print("-"*50)
    
    unique_pids = set(res['pid'] for res in results)
    unique_cores = set(res['core_id'] for res in results if res['core_id'] != "N/A")
    
    for res in results[:10]:
        print(f"{res['filename']:<15} {res['pid']:<10} {res['core_id']:<10} {res['duration']:.4f}")
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more images")
    
    print("-"*50)
    print(f"Unique PIDs observed: {len(unique_pids)} → Confirms SEPARATE PROCESSES")
    print(f"All PIDs: {sorted(unique_pids)}")
    if unique_cores:
        print(f"CPU Cores utilized: {unique_cores} → True parallel execution")
    print(f"\nTotal time: {total_time:.4f} seconds")


def run_threadpool_executor(images_data, num_workers, verbose=True):
    """
    Paradigm 2: concurrent.futures.ThreadPoolExecutor (Thread-based parallelism)
    
    Synchronization & Load Balancing:
    - ThreadPoolExecutor context manager handles thread lifecycle
    - executor.map() automatically distributes work across threads
    - Shared memory space - but our tasks are independent (no shared state)
    - GIL provides implicit synchronization for Python objects
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
        _print_threadpool_results(results, total_time)
    
    return results, total_time


def _print_threadpool_results(results, total_time):
    """Print execution details for ThreadPoolExecutor."""
    print(f"\n{'Image':<15} {'PID':<10} {'TID':<20} {'Core ID':<10} {'Time (s)':<12}")
    print("-"*70)
    
    unique_pids = set(res['pid'] for res in results)
    unique_tids = set(res['tid'] for res in results)
    unique_cores = set(res['core_id'] for res in results if res['core_id'] != "N/A")
    
    for res in results[:10]:
        tid_short = str(res['tid'])[-8:]
        print(f"{res['filename']:<15} {res['pid']:<10} ...{tid_short:<16} {res['core_id']:<10} {res['duration']:.4f}")
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more images")
    
    print("-"*70)
    print(f"Unique PIDs observed: {len(unique_pids)} → Confirms SINGLE PROCESS (all threads share PID)")
    print(f"Unique Thread IDs: {len(unique_tids)} → Multiple threads within same process")
    if unique_cores:
        print(f"CPU Cores observed: {unique_cores} → May switch cores but GIL limits parallelism")
    print(f"\nTotal time: {total_time:.4f} seconds")
