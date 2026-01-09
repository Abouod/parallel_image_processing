# Lab 5: Distributed Memory Programming (MPI)

## Objective:
- Implement parallel data processing using MPI
- Calculate and display the memory footprint to understand the difference between shared vs. distributed memory models

## Scenario:

In this scenario, explore a common High-Performance Computing (HPC) challenge: processing a dataset that is too large to fit into the memory of a single computer. Perform Matrix Multiplication (C = A x B) on two matrices (2000 x 2000 integers).

---

## Part 1: Using OpenMP

### 1. Instance Configuration

- Navigate to the Compute Engine section in the Google Cloud Console.
- Select **Create Instance**.
- **Machine Family:** Select General purpose, Series E2.
- **Machine Type:** Select E2 series and e2-medium.
- **Boot Disk:** Select Debian GNU/Linux 12 (bookworm) with 10 GB storage.
- **Firewall:** Default settings are sufficient.

### 2. Deployment

- Click **Create** to provision the resource.
- Once the Status indicator turns green, click **SSH** to establish a secure shell connection.

### 3. Install Development Tools

Update the package list and install the GCC compiler by running the following commands in the SSH terminal:

```bash
$ sudo apt update
$ sudo apt install build-essential -y
```

### 4. Process Creation

Create `huge_matrix_openmp.c`

### 5. Source Code Integration

Insert the following code into the file:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// FAST TEST SIZE
#define N 2000

int main() {
  // 1. Calculate theoretical memory needed
  // 3 matrices * N * N * 4 bytes
  double mem_req = 3.0 * N * N * sizeof(int) / (1024.0 * 1024.0 * 1024.0);
  printf("=== OpenMP Matrix Mult (N=%d) ===\n", N);
  printf("Requested RAM: %.4f GB\n", mem_req);
  printf("Threads available: %d\n", omp_get_max_threads());

  // 2. Allocate Memory
  int *A = (int*)malloc((size_t)N * N * sizeof(int));
  int *B = (int*)malloc((size_t)N * N * sizeof(int));
  int *C = (int*)malloc((size_t)N * N * sizeof(int));

  if (!A || !B || !C) {
    printf("Memory Allocation Failed!\n");
    return 1;
  }

  // 3. Initialize
  // Parallelize initialization just for speed
  #pragma omp parallel for
  for (size_t i = 0; i < (size_t)N * N; i++) {
    A[i] = 1;
    B[i] = 1;
  }

  printf("Initialization Done. Computing...\n");

  // 4. Compute (Parallelized)
  double start = omp_get_wtime();
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      long long sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = (int)sum;
    }
  }
  double end = omp_get_wtime();

  printf("Success! Time: %.4f seconds\n", end - start);
  printf("Result Check: C[0][0] = %d\n", C[0]);

  // 5. Cleanup
  free(A);
  free(B);
  free(C);

  return 0;
}
```

### 6. Execute the Script

Compile and run:

```bash
$ gcc huge_matrix_openmp.c -o huge_matrix_openmp -fopenmp
$ ./huge_matrix_openmp
```

---

## Part 2: Using MPI

### 1. The Firewall

Use Network Tags to link computers.

- Go to **VPC network → Firewall**.
- Click **Create Firewall Rule**.
- **Name:** `allow-mpi-internal`
- **Targets:** Select Specified target tags.
- **Target tags:** Type `mpi-node`
- **Source filter:** Select Source tags.
- **Source tags:** Type `mpi-node`
- **Protocols and ports:** Select Allow all.
- Click **Create**.

### 2. Instance Configuration

- Navigate to the Compute Engine section in the Google Cloud Console.
- Select **Create Instance** (master-node).
- **Machine Family:** Select General purpose, Series E2.
- **Machine Type:** Select E2 series and e2-medium
- **Boot Disk:** Select Debian GNU/Linux 12 (bookworm) with 10 GB storage.
- **Networking → Firewall:** Network tags: `mpi-node`

### 3. Deployment

- Click **Create** to provision the resource.
- Once the Status indicator turns green, click **SSH** to establish a secure shell connection.

### 4. Install Development Tools

Update the package list and install the GCC compiler by running the following commands in the SSH terminal:

```bash
$ sudo apt update
$ sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev gcc tmux
```

### 5. Setup SSH Keys

```bash
$ ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

### 6. Process Creation

Create `matrix_mult.c`

### 7. Source Code Integration

Insert the following code into the file:

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Change this to 20000 for the big run later
#define N 2000

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_node = N / size;

  if (rank == 0) {
    printf("--- MPI Matrix Multiplication (N=%d) ---\n", N);
    printf("Nodes: %d\n", size);
    printf("Master initializing and allocating memory...\n");
  }

  // 1. Allocation
  int *A_local = (int*)malloc((size_t)rows_per_node * N * sizeof(int));
  int *B = (int*)malloc((size_t)N * N * sizeof(int));
  int *C_local = (int*)malloc((size_t)rows_per_node * N * sizeof(int));

  // 2. Initialize Data
  // We fill with 1s so the math is predictable
  for (size_t i = 0; i < (size_t)rows_per_node * N; i++) A_local[i] = 1;

  if (rank == 0) {
    for (size_t i = 0; i < (size_t)N * N; i++) B[i] = 1;
  }

  // 3. Broadcast B (Sending the data to everyone)
  // We barrier here to ensure everyone has B before we start the timer
  MPI_Bcast(B, (size_t)N * N, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // --- TIMER START ---
  double start_time = MPI_Wtime();
  if (rank == 0) printf("Computing...\n");

  // 4. Compute (The Heavy Lifting)
  for (int i = 0; i < rows_per_node; i++) {
    for (int j = 0; j < N; j++) {
      long long sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A_local[i * N + k] * B[k * N + j];
      }
      C_local[i * N + j] = (int)sum;
    }
  }

  // Wait for all nodes to finish calculation before stopping timer
  MPI_Barrier(MPI_COMM_WORLD);

  // --- TIMER END ---
  double end_time = MPI_Wtime();

  // 5. Output Result
  if (rank == 0) {
    printf("Success! Calculation Complete.\n");
    printf("Total Execution Time: %.4f seconds\n", end_time - start_time);
  }

  // Cleanup
  free(A_local);
  free(B);
  free(C_local);
  MPI_Finalize();

  return 0;
}
```

### 8. Execute the Script

Compile and run:

```bash
$ mpicc -O3 matrix_mult.c -o mpi_matrix
```

### 9. Create the Cluster (Instance Group)

- Stop the Master Node.
- Create a group based on this VM.
- **Create Template:** `mpi-template` (Source: mpi-medium-image, Machine Type: e2-medium).
- **Create Managed Instance Group:** Name: `mpi-workers`, Size: 4.

### 10. Authorize the SSH Key on the Master

- Start Master.
- Get IPs of the new workers (using `gcloud compute instances list` or the Console).
- Create/Edit `hostfile`:

```
localhost
10.128.0.20
10.128.0.17
10.128.0.18
10.128.0.19
```

- Verify the connection with worker nodes:

```bash
$ ssh 10.128.0.17 "echo Success"
```

### 11. Run the Code

Run the code:

```bash
$ mpirun -np 5 --hostfile hostfile ./mpi_matrix
```