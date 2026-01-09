# Lab 2: Concurrency Models in Python – Multithreading vs. Multiprocessing

## Objective:

The purpose of this laboratory session is to evaluate distinct parallel processing architectures within Python. The experiment involves provisioning a multi-core cloud environment to benchmark Multithreading versus Multiprocessing by analyzing execution time, Process ID (PID) allocation, and CPU core utilization.

---

## Part 1: Infrastructure Setup

To accurately observe parallel execution across multiple physical hardware threads, a Virtual Machine (VM) with sufficient vCPU resources is required. The Python script simulates 4 concurrent workers; therefore, a single-core machine is insufficient for demonstrating true parallelism.

### 1. Instance Configuration

- Navigate to the Compute Engine section in the Google Cloud Console.
- Select **Create Instance**.
- **Machine Family:** Select General purpose, Series E2.
- **Machine Type:** Select **e2-highcpu-4** (4 vCPU, 4 GB memory) or **e2-standard-4** (4 vCPU, 16 GB memory).
  - **Rationale:** This specific configuration (4 vCPUs) allows the Operating System to schedule the 4 concurrent Python processes onto distinct cores, enabling the visualization of hardware-level parallelism.
- **Boot Disk:** Select Debian GNU/Linux 12 (bookworm) with 10 GB storage.
- **Firewall:** Default settings are sufficient.

### 2. Deployment

- Click **Create** to provision the resource.
- Once the Status indicator turns green, click **SSH** to establish a secure shell connection.

---

## Part 2: Environment and Dataset Preparation

The Linux environment requires specific libraries for data manipulation and system resource monitoring.

### 1. Dependency Installation

Execute the following commands to update the package manager and install the necessary Python libraries:

```bash
$ sudo apt update 
$ sudo apt install -y python3-pip python3-venv 
$ python3 -m venv venv
$ source venv/bin/activate 
$ pip install pandas numpy psutil
```

### 2. Dataset Acquisition

The experiment requires a specific dataset file named `shopping_behavior_updated.csv`.

- **Action:** Upload `shopping_behavior_updated.csv` to the VM's working directory.
- **Ensure** the file resides in the same folder where the Python script will be created.

---

## Part 3: Script Implementation

The benchmarking script utilizes `concurrent.futures` for threading and `multiprocessing.Pool` for process-based parallelism.

### 1. File Creation

Create a new Python file named `concurrency_test.py`

### 2. Source Code Integration

Insert the following code into the file (ensure indentation is preserved):

```python
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import time
import os
import threading
from datetime import datetime

# Try to import psutil to get the real CPU core ID
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ================= Configuration Parameters =================
FILENAME = "shopping_behavior_updated.csv"
CHUNK_SIZE = 1000

def get_core_id():
    """Get CPU core ID specifically"""
    if HAS_PSUTIL:
        try:
            p = psutil.Process()
            return p.cpu_num()
        except:
            return "N/A"
    return "N/A (psutil not installed)"

def get_thread_info():
    """Get PID and TID"""
    pid = os.getpid()
    tid = threading.get_ident()
    return f"PID:{pid} | TID:{tid}"

def process_chunk(chunk_data):
    """
    General processing logic
    """
    chunk_id, chunk_df = chunk_data
    start_time = time.time()
    
    # 1. Get core ID (obtained at the start of processing, representing the 
    # currently running core)
    # Note: The operating system may switch the task to another core during 
    # runtime, here we record the core at that moment
    core_id = get_core_id()
    
    # 2. Data cleaning
    chunk_df.columns = [c.strip() for c in chunk_df.columns]
    
    local_counts = pd.Series(dtype=int)
    
    # 3. Classification logic
    if 'Size' in chunk_df.columns and 'Gender' in chunk_df.columns:
        chunk_df['Size_Group'] = np.where(chunk_df['Size'].isin(['S', 'M']), 
                                          'Small', 'Large')
        chunk_df['Group_Key'] = chunk_df['Gender'] + ' - ' + chunk_df['Size_Group']
        local_counts = chunk_df['Group_Key'].value_counts()
    
    duration = time.time() - start_time
    
    # Return detailed results
    return {
        'counts': local_counts,
        'chunk_id': chunk_id,
        'duration': duration,
        'core_id': core_id,  # <--- Return Core ID separately
        'thread_info': get_thread_info()
    }

def run_threading():
    print(f"\n{'='*20} Method 1: Concurrent Futures {'='*20}")
    start_time = time.time()
    total_counts = pd.Series(dtype=int)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        reader = pd.read_csv(FILENAME, chunksize=CHUNK_SIZE)
        futures = []
        
        for i, chunk in enumerate(reader):
            futures.append(executor.submit(process_chunk, (i+1, chunk)))
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            total_counts = total_counts.add(res['counts'], fill_value=0)
            
            # === Modification: Log format optimization ===
            print(f" [Thread] Data Chunk ID: {res['chunk_id']} ---> CPU Core ID: {res['core_id']}")
            print(f" ℹ Identity Info: {res['thread_info']}")
            print(f" ⏱ Time Consumed: {res['duration']:.4f}s")
            print(f"  Classification Result: {res['counts'].to_dict()}")
            print("-" * 50)
    
    end_time = time.time()
    return end_time - start_time

def run_multiprocessing():
    print(f"\n{'='*20} Method 2: Multiprocessing {'='*20}")
    start_time = time.time()
    total_counts = pd.Series(dtype=int)
    
    chunks = []
    reader = pd.read_csv(FILENAME, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader):
        chunks.append((i+1, chunk))
    
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_chunk, chunks)
    
    for res in results:
        total_counts = total_counts.add(res['counts'], fill_value=0)
        
        # === Modification: Log format optimization ===
        print(f" [Process] Data Chunk ID: {res['chunk_id']} ---> CPU Core ID: {res['core_id']}")
        print(f" ℹ Identity Info: {res['thread_info']}")
        print(f" ⏱ Time Consumed: {res['duration']:.4f}s")
        print(f"  Classification Result: {res['counts'].to_dict()}")
        print("-" * 50)
    
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    if os.path.exists(FILENAME):
        # 1. Run multithreading
        time_thread = run_threading()
        
        # 2. Run multiprocessing
        time_process = run_multiprocessing()
        
        # 3. Final comparison
        print(f"\n{'='*20}  Final Comparison Result {'='*20}")
        print(f"1. Concurrent Futures Total Time : {time_thread:.4f} seconds")
        print(f"2. Multiprocessing Total Time : {time_process:.4f} seconds")
    else:
        print(f"❌ File not found: {FILENAME}")
```

### 3. Execute the Script

Run the script using the configured Python environment to observe the runtime behaviour of the operating system scheduler:

```bash
$ python concurrency_test.py
```

### 4. Analysis Output

#### Concurrent Futures (Threading)

Examine the terminal logs for the "Threading" section.

- **PID Observation:** Observe that the PID (Process ID) remains constant for every data chunk processed.
- **Core Allocation:** Observe that despite having 4 vCPUs, the threads may frequently execute on the same core or switch rapidly, constrained by the Python Global Interpreter Lock (GIL) within a single process context.

#### Multiprocessing

Examine the terminal logs for the "Multiprocessing" section.

- **PID Observation:** Observe distinct PIDs for different data chunks. This confirms that separate memory spaces and Python interpreters were spawned.
- **Core Allocation:** Observe the CPU Core ID column. With the 4-vCPU VM configuration, the logs should demonstrate simultaneous utilization of different cores (e.g., Core 0, Core 1, Core 2, Core 3), indicating true parallel execution.