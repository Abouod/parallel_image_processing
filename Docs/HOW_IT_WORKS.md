# How This Image Processing Pipeline Works

## Overview

This project compares **two different parallel programming paradigms** in Python for CPU-bound image processing tasks:

1. **`multiprocessing.Pool`** - Process-based parallelism (bypasses Python's GIL)
2. **`concurrent.futures.ThreadPoolExecutor`** - Thread-based parallelism (limited by GIL)

Think of it like an assembly line: sequential processing is one worker doing everything, while parallel processing divides work among multiple workers.

## The Big Picture

```
Download Images → Load into Memory → Apply Filters → Compare Paradigms → Analyze Performance
```

## Key Concepts: Processes vs Threads

### What's the Difference?

| Aspect | Processes (`multiprocessing.Pool`) | Threads (`ThreadPoolExecutor`) |
|--------|-----------------------------------|--------------------------------|
| **Memory** | Separate memory spaces | Shared memory |
| **GIL Impact** | Bypasses GIL | Limited by GIL |
| **Communication** | Via pickling/IPC | Direct memory access |
| **Overhead** | Higher (separate interpreters) | Lower (shared interpreter) |
| **Best For** | CPU-bound tasks | I/O-bound tasks |

### Python's Global Interpreter Lock (GIL)

The GIL is a mutex that prevents multiple threads from executing Python bytecode simultaneously:

```
With Threads (GIL):
Thread 1: ████████░░░░░░░░ (execute)
Thread 2: ░░░░░░░░████████ (wait, then execute)
→ Only ONE thread runs Python code at a time!

With Processes (No GIL constraint):
Process 1: ████████████████ (execute)
Process 2: ████████████████ (execute simultaneously)
→ TRUE parallel execution!
```

## Project Structure Explained

### Main Components

**[image_pipeline/main.py](../image_pipeline/main.py)** - The Orchestra Conductor
- Compares both parallel paradigms
- Tracks execution metadata (PID, CPU core, timing)
- Generates performance comparison tables
- Calculates speedup and efficiency metrics

**[image_pipeline/filters.py](../image_pipeline/filters.py)** - The Image Transformers
- Contains 5 different filters that modify images
- Each filter does one specific transformation

**[image_pipeline/utils.py](../image_pipeline/utils.py)** - The Helper Functions
- Downloads images from the internet
- Loads images from folders
- Saves processed images to disk

## The Two Parallel Paradigms

### Paradigm 1: multiprocessing.Pool (Process-based)

```python
from multiprocessing import Pool

def run_multiprocessing_pool(images_data, num_workers):
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_image, images_data)
    return results
```

**What happens:**
1. Creates `num_workers` separate Python processes
2. Each process has its own Python interpreter and memory space
3. Images are pickled (serialized) and sent to worker processes
4. Workers process images truly in parallel (bypass GIL)
5. Results are pickled and returned to main process

**Observable characteristics:**
- **Different PIDs** for each worker (separate processes)
- **Different CPU cores** utilized simultaneously
- **True parallelism** for CPU-bound tasks

### Paradigm 2: concurrent.futures.ThreadPoolExecutor (Thread-based)

```python
from concurrent.futures import ThreadPoolExecutor

def run_threadpool_executor(images_data, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image, images_data))
    return results
```

**What happens:**
1. Creates `num_workers` threads within the same process
2. All threads share the same Python interpreter and memory
3. GIL allows only one thread to execute Python bytecode at a time
4. Threads take turns executing (context switching)

**Observable characteristics:**
- **Same PID** for all workers (single process)
- **May use different CPU cores** but GIL limits actual parallelism
- **Limited speedup** for CPU-bound tasks due to GIL

## Execution Tracking (Like Lab 2)

Similar to Lab 2's concurrency comparison, we track:

```python
def get_cpu_core_id():
    """Get the current CPU core ID."""
    if HAS_PSUTIL:
        p = psutil.Process()
        return p.cpu_num()
    return "N/A"

def get_process_info():
    """Get PID and Thread ID."""
    pid = os.getpid()
    tid = threading.get_ident()
    return pid, tid
```

This allows us to observe:
- **PID allocation**: Same vs different PIDs
- **CPU core utilization**: Which cores are being used
- **Thread identification**: Multiple threads in same process

## The Filter Pipeline (filters.py)

Each image goes through 5 filters in order:

### Filter 1: Grayscale Conversion
```python
ImageOps.grayscale(image)
```
- Converts color image to black and white
- Uses luminance formula: $Y = 0.299R + 0.587G + 0.114B$
- Reduces 3 channels (RGB) to 1 channel

### Filter 2: Gaussian Blur (3×3 kernel)
```python
image.filter(ImageFilter.GaussianBlur(radius=1))
```
- Smooths the image by averaging nearby pixels
- Uses a 3×3 Gaussian kernel for smoothing
- Reduces noise and sharp details

### Filter 3: Sobel Edge Detection
```python
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
```
- Finds edges where brightness changes rapidly
- Computes horizontal and vertical gradients
- Combines into edge magnitude

### Filter 4: Sharpen
```python
image.filter(ImageFilter.SHARPEN)
```
- Enhances edges and fine details
- Makes edges more pronounced

### Filter 5: Brightness Adjustment
```python
ImageEnhance.Brightness(image).enhance(1.2)
```
- Increases brightness by 20% (factor = 1.2)

## Performance Metrics

### Speedup

$$S = \frac{T_{sequential}}{T_{parallel}}$$

- Measures how many times faster parallel is than sequential
- Example: 45s sequential ÷ 15s parallel = 3x speedup

### Efficiency

$$E = \frac{S}{N} \times 100\%$$

Where:
- $S$ = Speedup
- $N$ = Number of workers

- Measures how well each worker is utilized
- Perfect efficiency = 100% (each worker provides full speedup)
- Reality: 50-80% due to overhead

### Amdahl's Law

$$S_{max} = \frac{1}{(1-P) + \frac{P}{N}}$$

Where:
- $P$ = Parallelizable fraction of the code
- $N$ = Number of processors

This defines the theoretical maximum speedup for any parallel program.

## Expected Results

### Multiprocessing.Pool
- **High speedup** for CPU-bound image processing
- **Good efficiency** (60-80%) with moderate worker counts
- **Different PIDs** confirm separate processes
- **Multiple CPU cores** utilized simultaneously

### ThreadPoolExecutor
- **Low speedup** for CPU-bound tasks (limited by GIL)
- **Poor efficiency** for CPU-bound workloads
- **Same PID** confirms single process with multiple threads
- **GIL prevents true parallelism**

### Sample Performance Table

| Paradigm | Workers | Time (s) | Speedup | Efficiency |
|----------|---------|----------|---------|------------|
| Sequential | 1 | 45.00 | 1.00x | 100% |
| multiprocessing.Pool | 4 | 12.00 | 3.75x | 93.75% |
| ThreadPoolExecutor | 4 | 42.00 | 1.07x | 26.75% |

## Key Takeaways

1. **multiprocessing.Pool** is superior for CPU-bound tasks because it bypasses the GIL
2. **ThreadPoolExecutor** is better suited for I/O-bound tasks (file I/O, network)
3. **PID observation** confirms whether processes or threads are used
4. **CPU core tracking** shows actual hardware utilization
5. **Speedup is rarely linear** due to overhead (process creation, serialization)

## Running the Pipeline

```bash
# Default: Test with 2 and 4 workers
python -m image_pipeline.main

# Specify worker counts
python -m image_pipeline.main --workers 2 4 8

# Process fewer images for quick testing
python -m image_pipeline.main --num-images 50
```

## Code Entry Point

The main function in [main.py](../image_pipeline/main.py) orchestrates:
1. Dataset download/loading
2. Sequential baseline execution
3. Multiprocessing.Pool execution with various worker counts
4. ThreadPoolExecutor execution with various worker counts
5. Performance comparison table generation
6. Analysis and conclusions
