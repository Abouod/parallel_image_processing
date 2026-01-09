"""
CST435: Parallel and Cloud Computing - Assignment 2
Concurrent Image Processing Pipeline

This module is the main entry point that compares two parallel processing paradigms:
1. multiprocessing.Pool (process-based parallelism - bypasses GIL)
2. concurrent.futures.ThreadPoolExecutor (thread-based parallelism - limited by GIL)

Module Structure:
- main.py: Entry point and orchestration
- processing.py: Parallel processing implementations
- analysis.py: Performance metrics and Amdahl's Law
- visualization.py: Graph generation
- filters.py: Image filter implementations
- utils.py: Dataset loading and saving utilities
"""

import os
import argparse

from image_pipeline.utils import download_food101_subset, load_images, save_image
from image_pipeline.processing import (
    run_sequential,
    run_multiprocessing_pool,
    run_threadpool_executor,
    HAS_PSUTIL
)
from image_pipeline.analysis import (
    print_performance_comparison,
    print_analysis_summary
)
from image_pipeline.visualization import generate_performance_graphs

# ================= Configuration Parameters =================
DATASET_PATH = "food101_subset"
OUTPUT_PATH_MULTIPROCESSING = "output_multiprocessing"
OUTPUT_PATH_THREADPOOL = "output_threadpool"
NUM_IMAGES = 100
WORKER_COUNTS = [2, 4]


def save_results(results, output_path):
    """Save processed images to output directory."""
    os.makedirs(output_path, exist_ok=True)
    for res in results:
        save_image(res['image'], os.path.join(output_path, res['filename']))
    print(f"Saved {len(results)} images to {output_path}/")


def print_header(args):
    """Print program header with configuration."""
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
    print(f"psutil installed: {'Yes (CPU core tracking enabled)' if HAS_PSUTIL else 'No'}")


def run_parallel_pipelines(images_data, worker_counts):
    """Run both parallel paradigms with different worker counts."""
    all_times = {}
    all_results = {}
    
    for num_workers in worker_counts:
        # Paradigm 1: multiprocessing.Pool
        mp_results, mp_time = run_multiprocessing_pool(images_data, num_workers)
        all_times[f'multiprocessing_{num_workers}_workers'] = mp_time
        all_results[f'multiprocessing_{num_workers}_workers'] = mp_results
        
        # Paradigm 2: ThreadPoolExecutor
        tp_results, tp_time = run_threadpool_executor(images_data, num_workers)
        all_times[f'threadpool_{num_workers}_workers'] = tp_time
        all_results[f'threadpool_{num_workers}_workers'] = tp_results
    
    return all_times, all_results


def save_best_results(all_times, all_results):
    """Save results from the best performing configuration."""
    print(f"\n{'='*60}")
    print("STEP 4: Saving Processed Images")
    print("="*60)
    
    # Save from multiprocessing results
    best_mp_key = min([k for k in all_results if 'multiprocessing' in k], 
                       key=lambda x: all_times[x])
    save_results(all_results[best_mp_key], OUTPUT_PATH_MULTIPROCESSING)
    
    # Save from threadpool results
    best_tp_key = min([k for k in all_results if 'threadpool' in k],
                       key=lambda x: all_times[x])
    save_results(all_results[best_tp_key], OUTPUT_PATH_THREADPOOL)


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CST435 Assignment 2: Compare multiprocessing.Pool vs ThreadPoolExecutor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m image_pipeline.main                    # Run with defaults [2, 4] workers
  python -m image_pipeline.main --workers 2 4 8    # Custom worker counts
  python -m image_pipeline.main --num-images 50    # Process 50 images
        """
    )
    parser.add_argument("--workers", type=int, nargs='+', default=WORKER_COUNTS,
                        help="Worker counts to test (e.g., --workers 2 4 8)")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help="Number of images to process")
    args = parser.parse_args()

    # Print header
    print_header(args)

    # Step 1: Download dataset
    print(f"\n{'='*60}")
    print("STEP 1: Dataset Preparation")
    print("="*60)
    download_food101_subset(DATASET_PATH, args.num_images)

    # Step 2: Load images
    print(f"\n{'='*60}")
    print("STEP 2: Loading Images")
    print("="*60)
    images_data = load_images(DATASET_PATH)
    print(f"Loaded {len(images_data)} images into memory")

    if not images_data:
        print("No images to process. Exiting.")
        return

    # Step 3: Run processing pipelines
    print(f"\n{'='*60}")
    print("STEP 3: Running Processing Pipelines")
    print("="*60)
    
    seq_results, seq_time = run_sequential(images_data)
    all_times, all_results = run_parallel_pipelines(images_data, args.workers)

    # Step 4: Save results
    save_best_results(all_times, all_results)

    # Step 5: Performance Analysis
    print(f"\n{'='*60}")
    print("STEP 5: Performance Analysis")
    print("="*60)
    print_performance_comparison(all_times, seq_time)
    print_analysis_summary(all_times, seq_time)
    
    # Step 6: Generate Graphs
    print(f"\n{'='*60}")
    print("STEP 6: Generating Performance Graphs")
    print("="*60)
    generate_performance_graphs(all_times, seq_time, args.workers)
    
    # Done
    print("\nProcessing complete!")
    print(f"  - Multiprocessing results: {OUTPUT_PATH_MULTIPROCESSING}/")
    print(f"  - ThreadPool results: {OUTPUT_PATH_THREADPOOL}/")
    print(f"  - Performance graphs: performance_graphs/")


if __name__ == "__main__":
    main()
