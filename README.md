# Concurrent Image Processing Pipeline

## Overview

This project demonstrates the performance differences between two parallel programming paradigms in Python for CPU-bound image manipulation tasks. It implements an image processing pipeline that applies a series of filters to a dataset of images, comparing:

1. **`multiprocessing.Pool`** - Process-based parallelism (bypasses Python's GIL)
2. **`concurrent.futures.ThreadPoolExecutor`** - Thread-based parallelism (limited by GIL)

The project is part of a CST435 course assignment focusing on parallel and concurrent programming concepts.

## Key Concepts

### Why Compare These Two Paradigms?

| Aspect | `multiprocessing.Pool` | `ThreadPoolExecutor` |
|--------|----------------------|---------------------|
| **Parallelism Type** | Process-based | Thread-based |
| **Memory Model** | Separate memory spaces | Shared memory |
| **GIL Impact** | Bypasses GIL | Limited by GIL |
| **Best For** | CPU-bound tasks | I/O-bound tasks |
| **Overhead** | Higher (process creation) | Lower (thread creation) |

### Python's Global Interpreter Lock (GIL)

The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This means:

- **Threads** cannot achieve true parallelism for CPU-bound tasks
- **Processes** bypass the GIL by having separate Python interpreters

For image processing (CPU-bound), `multiprocessing.Pool` is expected to significantly outperform `ThreadPoolExecutor`.

## What This Project Does

The pipeline performs the following operations on each image:

1. **Grayscale Conversion** - Converts the image to grayscale using luminance formula
2. **Gaussian Blur** - Applies a 3x3 Gaussian blur filter for smoothing
3. **Sobel Edge Detection** - Detects edges using the Sobel operator
4. **Sharpen Filter** - Enhances edge definition
5. **Brightness Adjustment** - Increases brightness by 20%

The project compares execution across:
- Sequential processing (baseline)
- `multiprocessing.Pool` with 2, 4, and more workers
- `ThreadPoolExecutor` with 2, 4, and more workers

## Project Structure

```
assignment2_435/
├── image_pipeline/              # Main Python package
│   ├── __init__.py
│   ├── main.py                  # Entry point and orchestration
│   ├── processing.py            # Parallel processing implementations
│   ├── analysis.py              # Performance metrics & Amdahl's Law
│   ├── visualization.py         # Graph generation (matplotlib)
│   ├── filters.py               # Image filter implementations
│   └── utils.py                 # Utility functions (download, load, save)
├── food101_subset/              # Downloaded images (generated)
├── output_multiprocessing/      # Results from multiprocessing.Pool
├── output_threadpool/           # Results from ThreadPoolExecutor
├── performance_graphs/          # Generated performance charts
│   ├── execution_time_comparison.png
│   ├── speedup_comparison.png
│   ├── efficiency_comparison.png
│   └── performance_summary.png
├── Docs/                        # Documentation
│   └── HOW_IT_WORKS.md          # Detailed technical explanation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Requirements

- Python 3.7+
- Pillow (PIL)
- OpenCV (cv2)
- NumPy
- matplotlib (for performance graphs)
- tqdm (for progress bars)
- requests (for downloading images)
- psutil (for CPU core monitoring)

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

Run the pipeline with default settings (tests with 2 and 4 workers):

```bash
python -m image_pipeline.main
```

### Specify Worker Counts

Test with specific worker configurations:

```bash
python -m image_pipeline.main --workers 2 4 8
```

### Change Number of Images

Process a different number of images:

```bash
python -m image_pipeline.main --num-images 50
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--workers` | Worker counts to test (space-separated) | `2 4` |
| `--num-images` | Number of images to process | `100` |

### What Happens When You Run It

1. **Prepares Dataset**: If `food101_subset/` doesn't exist, downloads placeholder images (for testing). You can replace these with real Food-101 images.
2. **Loads Images**: Loads up to `--num-images` images (default: 100) from `food101_subset/` into memory
3. **Sequential Processing**: Processes all loaded images sequentially (baseline)
4. **Multiprocessing**: Processes images using `multiprocessing.Pool`
5. **ThreadPool**: Processes images using `ThreadPoolExecutor`
6. **Performance Report**: Displays execution times, speedup, and efficiency metrics
7. **Generates Graphs**: Creates performance charts in `performance_graphs/`

### Using Your Own Images (Recommended)

For the assignment, you should use images from the **Food-101 dataset**:

1. Download Food-101 from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)
2. Create the `food101_subset/` folder manually:
   ```bash
   mkdir -p food101_subset
   ```
3. Copy images into `food101_subset/`:
   ```bash
   # Example: Copy images from a Food-101 category
   cp /path/to/food-101/images/pizza/*.jpg food101_subset/
   ```
4. Run the pipeline with your desired image count:
   ```bash
   # Process 100 images (default)
   python -m image_pipeline.main
   
   # Process all 500 images
   python -m image_pipeline.main --num-images 500
   ```

### Example Output

```
================================================================================
CST435: PARALLEL AND CLOUD COMPUTING - ASSIGNMENT 2
Concurrent Image Processing Pipeline
================================================================================

Comparing two parallel paradigms:
  1. multiprocessing.Pool      (Process-based, bypasses GIL)
  2. ThreadPoolExecutor        (Thread-based, limited by GIL)

Worker counts to test: [2, 4]
Number of images: 100
CPU cores available: 4

============================================================
PARADIGM 1: multiprocessing.Pool (4 workers)
============================================================
Characteristics:
  - Process-based parallelism (separate memory spaces)
  - Bypasses Python's Global Interpreter Lock (GIL)
  - Each worker is a separate Python interpreter
  - Best for CPU-bound tasks

Image           PID        Core ID    Time (s)    
--------------------------------------------------
image_0000.jpg  12345      0          0.0234
image_0001.jpg  12346      1          0.0198
image_0002.jpg  12347      2          0.0215
...

Unique PIDs observed: 4 → Confirms SEPARATE PROCESSES
CPU Cores utilized: {0, 1, 2, 3} → True parallel execution

================================================================================
PERFORMANCE COMPARISON TABLE
================================================================================
Paradigm                                      Workers    Time (s)     Speedup    Efficiency  
--------------------------------------------------------------------------------
Sequential (Baseline)                         1          45.23        1.00x      100.00%
multiprocessing.Pool                          4          12.34        3.67x      91.75%
concurrent.futures.ThreadPoolExecutor         4          42.15        1.07x      26.75%
================================================================================
```

## Technical Details

### Key Observations from Execution

**Multiprocessing.Pool:**
- Different PIDs for each worker → Separate processes
- Can utilize multiple CPU cores simultaneously
- True parallelism for CPU-bound tasks

**ThreadPoolExecutor:**
- Same PID for all workers → Single process with multiple threads
- Limited by GIL for CPU-bound operations
- Better suited for I/O-bound tasks (file I/O, network requests)

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Speedup** | $S = T_{sequential} / T_{parallel}$ | How many times faster than sequential |
| **Efficiency** | $E = S / N \times 100\%$ | How well workers are utilized (N = worker count) |

### Amdahl's Law

The theoretical maximum speedup is limited by the sequential portion of the code:

$$S_{max} = \frac{1}{(1-P) + \frac{P}{N}}$$

Where:
- $P$ = Parallelizable fraction
- $N$ = Number of processors

## Filter Implementations

All filters are implemented using industry-standard libraries:
- **PIL (Pillow)**: For grayscale, blur, sharpen, and brightness
- **OpenCV**: For Sobel edge detection
- **NumPy**: For numerical operations in edge detection

## Deployment on GCP

For deployment on Google Cloud Platform (as required by the assignment):

1. Create a VM with multiple vCPUs (e.g., `e2-standard-4` with 4 vCPUs)
2. SSH into the VM
3. Clone the repository and install dependencies
4. Run the pipeline to observe true multi-core parallelism

## Files Included

| File | Description |
|------|-------------|
| [image_pipeline/main.py](image_pipeline/main.py) | Entry point and orchestration |
| [image_pipeline/processing.py](image_pipeline/processing.py) | Parallel processing (Pool & ThreadPool) |
| [image_pipeline/analysis.py](image_pipeline/analysis.py) | Performance metrics & Amdahl's Law |
| [image_pipeline/visualization.py](image_pipeline/visualization.py) | Graph generation (bar charts, line graphs) |
| [image_pipeline/filters.py](image_pipeline/filters.py) | 5 image filter implementations |
| [image_pipeline/utils.py](image_pipeline/utils.py) | Dataset download, load, save helpers |
| [Docs/HOW_IT_WORKS.md](Docs/HOW_IT_WORKS.md) | Detailed technical explanation |
| [requirements.txt](requirements.txt) | Python dependencies |

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

**"No module named 'psutil'"**
```bash
pip install psutil
```

**Low speedup with multiprocessing**
- Ensure you have multiple CPU cores available
- Check with `python -c "import os; print(os.cpu_count())"`

## Learning Objectives

This project demonstrates:
1. The difference between process-based and thread-based parallelism
2. Impact of Python's GIL on CPU-bound tasks
3. When to use multiprocessing vs threading
4. How to measure parallel performance (speedup, efficiency)
5. Real-world application of concurrent programming concepts

## License

This is an academic project for CST435.

## Authors

Course: CST435 - Parallel and Cloud Computing  
Assignment 2: Image Processing Pipeline with Concurrent Processing

---

For detailed technical analysis, see [Docs/HOW_IT_WORKS.md](Docs/HOW_IT_WORKS.md).
