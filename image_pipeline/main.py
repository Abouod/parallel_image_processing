import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

from image_pipeline.filters import apply_filters
from image_pipeline.utils import download_food101_subset, load_images, save_image

# Configuration
DATASET_PATH = "food101_subset"
OUTPUT_PATH_SEQUENTIAL = "output_sequential"
OUTPUT_PATH_CONCURRENT_FUTURES_PROCESS = "output_concurrent_futures_process"
NUM_IMAGES = 100  # Number of images to process for demonstration
# NUM_PROCESSES will be set by command-line argument or default to os.cpu_count()

def process_image_wrapper(image_data):
    """Wrapper function to apply filters to a single image."""
    image, filename = image_data
    processed_image = apply_filters(image)
    return processed_image, filename

def run_sequential(images_data):
    print("Running sequentially...")
    start_time = time.time()
    results = []
    for image_data in images_data:
        results.append(process_image_wrapper(image_data))
    end_time = time.time()
    print(f"Sequential processing took {end_time - start_time:.2f} seconds.")
    return results, (end_time - start_time)

def run_concurrent_futures_process_pool(images_data, num_workers):
    print(f"Running with concurrent.futures.ProcessPoolExecutor using {num_workers} processes...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image_wrapper, images_data))
    end_time = time.time()
    print(f"Concurrent.futures.ProcessPoolExecutor took {end_time - start_time:.2f} seconds.")
    return results, (end_time - start_time)

def main():
    parser = argparse.ArgumentParser(description="Run image processing pipeline with configurable workers.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of worker processes for ProcessPoolExecutor.")
    args = parser.parse_args()
    num_workers = args.workers

    # 1. Download a subset of the Food-101 dataset
    print(f"Downloading a subset of Food-101 dataset to {DATASET_PATH}...")
    download_food101_subset(DATASET_PATH, NUM_IMAGES)

    # 2. Load images
    print(f"Loading images from {DATASET_PATH}...")
    images_data = load_images(DATASET_PATH)
    print(f"Loaded {len(images_data)} images.")

    if not images_data:
        print("No images to process. Exiting.")
        return

    # 3. Run processing pipelines
    all_results = {}
    all_times = {}

    # Sequential processing (for baseline T_p(1))
    seq_results, seq_time = run_sequential(images_data)
    all_results['sequential'] = seq_results
    all_times['sequential'] = seq_time

    # Save processed images from the sequential pipeline
    print(f"Saving processed images from the sequential pipeline to {OUTPUT_PATH_SEQUENTIAL}...")
    os.makedirs(OUTPUT_PATH_SEQUENTIAL, exist_ok=True)
    for (processed_image, filename) in all_results['sequential']:
        save_image(processed_image, os.path.join(OUTPUT_PATH_SEQUENTIAL, filename))
    print(f"Sequential image processing complete. Results saved to {OUTPUT_PATH_SEQUENTIAL}.")

    # Concurrent.futures ProcessPoolExecutor with configurable workers
    cf_process_results, cf_process_time = run_concurrent_futures_process_pool(images_data, num_workers)
    all_results[f'concurrent_futures_process_{num_workers}_workers'] = cf_process_results
    all_times[f'concurrent_futures_process_{num_workers}_workers'] = cf_process_time

    # Save processed images from the concurrent.futures.ProcessPoolExecutor pipeline
    print(f"Saving processed images from the concurrent.futures.ProcessPoolExecutor pipeline to {OUTPUT_PATH_CONCURRENT_FUTURES_PROCESS}...")
    os.makedirs(OUTPUT_PATH_CONCURRENT_FUTURES_PROCESS, exist_ok=True)
    for (processed_image, filename) in all_results[f'concurrent_futures_process_{num_workers}_workers']:
        save_image(processed_image, os.path.join(OUTPUT_PATH_CONCURRENT_FUTURES_PROCESS, filename))
    print(f"Concurrent.futures.ProcessPoolExecutor image processing complete with {num_workers} workers. Results saved to {OUTPUT_PATH_CONCURRENT_FUTURES_PROCESS}.")

    # 5. Performance Analysis Summary
    print("\n--- Performance Summary ---")
    sorted_times = sorted(all_times.items(), key=lambda item: item[1])
    for paradigm, exec_time in sorted_times:
        print(f"{paradigm.replace('_', ' ').title()}: {exec_time:.2f} seconds")

    print("\n--- Detailed Performance Analysis ---")
    if 'sequential' in all_times and all_times['sequential'] > 0:
        # Calculate speedup for the ProcessPoolExecutor run
        process_pool_key = f'concurrent_futures_process_{num_workers}_workers'
        if process_pool_key in all_times:
            speedup = all_times['sequential'] / all_times[process_pool_key]
            efficiency = speedup / num_workers
            print(f"Speedup of ProcessPoolExecutor with {num_workers} workers over Sequential: {speedup:.2f}x")
            print(f"Efficiency of ProcessPoolExecutor with {num_workers} workers: {efficiency:.2f}")
    else:
        print("Sequential processing time not available or zero, cannot calculate speedup.")

    print("\nObservations:")
    print("- ProcessPoolExecutor leverages multiple CPU cores, overcoming Python's GIL for CPU-bound tasks.")
    print("- The observed performance might vary based on the number of images, image sizes, filter complexity, and system resources.")
    print("- For this image processing task (CPU-bound), ProcessPoolExecutor is theoretically expected to perform better than sequential processing, assuming sufficient images and CPU cores.")
    print("- `concurrent.futures.ProcessPoolExecutor` offers a higher-level, simpler API compared to `multiprocessing.Pool`, abstracting some complexities of process management while providing similar performance characteristics for CPU-bound tasks.")


if __name__ == "__main__":
    main()
