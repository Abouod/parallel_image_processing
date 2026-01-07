# Concurrent Image Processing Pipeline

## Overview

This project demonstrates the performance benefits of concurrent processing for CPU-bound image manipulation tasks. It implements an image processing pipeline that applies a series of filters to a dataset of images, comparing sequential execution against parallel processing using Python's `concurrent.futures.ProcessPoolExecutor`.

The project is part of a CST435 course assignment focusing on parallel and concurrent programming concepts.

## What This Project Does

The pipeline performs the following operations on each image:

1. **Grayscale Conversion** - Converts the image to grayscale using luminance
2. **Gaussian Blur** - Applies a 3x3 Gaussian blur filter for smoothing
3. **Sobel Edge Detection** - Detects edges using the Sobel operator
4. **Sharpen Filter** - Enhances edge definition
5. **Brightness Adjustment** - Increases brightness by 20%

These filters are applied sequentially to each image, and the project compares:
- **Sequential Processing**: One image at a time
- **Parallel Processing**: Multiple images processed simultaneously using multiple CPU cores

## Project Structure

```
assignment2_435/
├── image_pipeline/              # Main Python package
│   ├── __init__.py
│   ├── main.py                  # Entry point and orchestration
│   ├── filters.py               # Image filter implementations
│   └── utils.py                 # Utility functions (download, load, save)
├── food101_subset/              # Downloaded images (generated)
├── output_sequential/           # Results from sequential processing
├── output_concurrent_futures_process/  # Results from parallel processing
├── requirements.txt             # Python dependencies
├── technical_report.md          # Detailed technical analysis
├── submission_checklist.md      # Assignment checklist
└── README.md                    # This file
```

## Requirements

- Python 3.7+
- Pillow (PIL)
- OpenCV (cv2)
- NumPy
- tqdm (for progress bars)
- requests (for downloading images)

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Note: If you encounter issues with OpenCV, you may need to install it separately:
```bash
pip install opencv-python
```

## Usage

### Basic Usage (Default Settings)

Run the pipeline with default settings (uses all available CPU cores):

```bash
python -m image_pipeline.main
```

### Specify Number of Workers

Control the number of parallel processes:

```bash
python -m image_pipeline.main --workers 4
```

### Command Line Arguments

- `--workers`: Number of worker processes for parallel processing (default: number of CPU cores)

### What Happens When You Run It

1. **Downloads Images**: Downloads 100 placeholder images to `food101_subset/` (only on first run)
2. **Loads Images**: Loads all images into memory
3. **Sequential Processing**: Processes all images sequentially and saves to `output_sequential/`
4. **Parallel Processing**: Processes all images in parallel using multiple processes and saves to `output_concurrent_futures_process/`
5. **Performance Report**: Displays execution times and speedup metrics

### Example Output

```
Downloading a subset of Food-101 dataset to food101_subset...
Loading images from food101_subset...
Loaded 100 images.

Running sequentially...
Sequential processing took 45.23 seconds.
Saving processed images from the sequential pipeline to output_sequential...

Running with concurrent.futures.ProcessPoolExecutor using 8 processes...
Concurrent.futures.ProcessPoolExecutor took 8.67 seconds.
Saving processed images from the concurrent.futures.ProcessPoolExecutor pipeline...

--- Performance Summary ---
Concurrent Futures Process 8 Workers: 8.67 seconds
Sequential: 45.23 seconds

--- Detailed Performance Analysis ---
Speedup (ProcessPoolExecutor with 8 workers): 5.22x
Efficiency: 65.25%
```

## Technical Details

### Why Parallel Processing Works Here

Image processing is a **CPU-bound task** - each filter operation requires significant computational work. Python's Global Interpreter Lock (GIL) prevents true parallelism with threads, but using **multiprocessing** (via `ProcessPoolExecutor`) creates separate Python processes, each with its own GIL, allowing true parallel execution across multiple CPU cores.

### Key Concepts Demonstrated

- **Process-based Parallelism**: Using `concurrent.futures.ProcessPoolExecutor` for CPU-bound tasks
- **Performance Metrics**: Speedup and efficiency calculations
- **Modular Design**: Separation of concerns (filters, utils, orchestration)
- **Baseline Comparison**: Sequential processing as a baseline for measuring improvements

### Filter Implementations

All filters are implemented using industry-standard libraries:
- **PIL (Pillow)**: For grayscale, blur, sharpen, and brightness
- **OpenCV**: For Sobel edge detection
- **NumPy**: For numerical operations in edge detection

## Performance Analysis

The project calculates two key metrics:

- **Speedup**: `T_sequential / T_parallel` - How much faster parallel processing is
- **Efficiency**: `Speedup / Number_of_Workers` - How effectively workers are utilized

Typical results show:
- **Linear speedup** is rare due to overhead (process creation, memory copying, synchronization)
- **Efficiency decreases** as more workers are added due to diminishing returns
- **Optimal worker count** is often around the number of physical CPU cores

## Files Included

- [image_pipeline/main.py](image_pipeline/main.py) - Main orchestration logic
- [image_pipeline/filters.py](image_pipeline/filters.py) - Filter implementations
- [image_pipeline/utils.py](image_pipeline/utils.py) - Helper functions
- [technical_report.md](technical_report.md) - Comprehensive technical analysis
- [requirements.txt](requirements.txt) - Python dependencies

## Troubleshooting

### Common Issues

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"No module named 'PIL'"**
```bash
pip install Pillow
```

**Images not downloading**
- Check your internet connection
- The script uses placehold.co for placeholder images
- If the service is down, the script will report errors but continue

**Low speedup on Windows**
- Windows has higher process creation overhead than Linux/macOS
- Try reducing the number of workers to match physical (not logical) cores

## Learning Objectives

This project demonstrates:
1. The difference between sequential and parallel processing
2. When to use process-based vs. thread-based concurrency
3. How to measure and analyze parallel performance
4. Real-world application of concurrent programming concepts
5. Python's `concurrent.futures` module for parallel processing

## Future Enhancements

Potential improvements:
- Add support for `multiprocessing.Pool` comparison
- Implement thread-based processing to show GIL limitations
- Add more sophisticated filters and filter chains
- Implement chunking strategies for very large datasets
- Add memory usage profiling
- Support for real Food-101 dataset download

## License

This is an academic project for CST435.

## Author

Course: CST435 - Parallel and Concurrent Programming
Assignment: Image Processing Pipeline with Concurrent Processing

---

For detailed technical analysis, performance benchmarks, and implementation rationale, see [technical_report.md](technical_report.md).
