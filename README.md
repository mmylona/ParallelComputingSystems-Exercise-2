# Assignment 2 – Programming with MPI & CUDA

## Table of Contents
* [System Information](#system-information)
* [Execution Instructions](#execution-instructions)
* [Exercise 2.1: Polynomial Multiplication with MPI](#exercise-21-polynomial-multiplication-with-mpi)
  * [2.1.1 Problem Description](#211-problem-description)
  * [2.1.2 Solution Description](#212-solution-description)
  * [2.1.3 Experimental Results](#213-experimental-results)
  * [2.1.4 Results Commentary](#214-results-commentary)
* [Exercise 2.2: Sparse Matrix × Vector Multiplication with MPI](#exercise-22-sparse-matrix--vector-multiplication-with-mpi)
  * [2.2.1 Problem Description](#221-problem-description)
  * [2.2.2 Solution Description](#222-solution-description)
  * [2.2.3 Scaling Number of Processes](#223-scaling-number-of-processes)
  * [2.2.4 Effect of Sparsity](#224-effect-of-sparsity)
  * [2.2.5 Effect of Matrix Size](#225-effect-of-matrix-size)
  * [2.2.6 Effect of Number of Iterations](#226-effect-of-number-of-iterations)
  * [2.2.7 Results Commentary](#227-results-commentary)
* [Exercise 2.3: Polynomial Multiplication with SIMD](#exercise-23-polynomial-multiplication-with-simd)
  * [2.3.1 Problem Description](#231-problem-description)
  * [2.3.2 Solution Description](#232-solution-description)
  * [2.3.3 Result 1: Effect of Polynomial Degree](#233-result-1-effect-of-polynomial-degree)
  * [2.3.4 Results Commentary](#234-results-commentary)
* [Exercise 2.4: Accelerating Maxwell Simulator with CUDA](#exercise-24-accelerating-maxwell-simulator-with-cuda)
  * [2.4.1 Problem Description](#241-problem-description)
  * [2.4.2 Solution Description](#242-solution-description)
  * [2.4.3 Results Commentary](#243-results-commentary)

---

## System Information
* **Hostname:** linux01
* **Processor model:** Intel Core i5-6500 @ 3.20GHz
* **Number of cores:** 4
* **OS version:** Ubuntu 20.04.6 LTS
* **Compiler version:** GCC 9.4.0

## Execution Instructions
To execute all exercises, first compile with the `make` command and then execute the programs with the `make run` command.

---

## Exercise 2.1: Polynomial Multiplication with MPI

### 2.1.1 Problem Description
The exercise studies the speedup offered by distributed processing via MPI in multiplying two random dense polynomials of degree *n* with the classical O(n²) complexity algorithm. Process 0 creates the two polynomials with integer, non-zero coefficients and sends to the other processes only the necessary data for parallel computation. The goal is to verify the correctness of the parallel result, as well as to measure and compare four individual times, for various polynomial degrees and numbers of processes, in relation to the serial execution but also to the shared memory implementations (Pthreads/OpenMP) of Exercise 1.1.

### 2.1.2 Solution Description
Each coefficient of the result C[k] is calculated independently of the others, eliminating any need for synchronization during the calculation. Because in MPI each process has its own separate address space, the two polynomials must be explicitly distributed to all processes via `MPI_Bcast` before the computation starts. Then, the output range [0, 2n] is divided into contiguous chunks among the processes. Finally, the results are gathered in process 0 with `MPI_Gatherv`. The timing of each phase is done with `MPI_Wtime()` in combination with `MPI_Barrier`, so that all processes are synchronized before each measurement, and a Bash script automates the compilation, the execution of experiments, the correctness check, and the recording of times and speedup in a CSV file.

### 2.1.3 Experimental Results

| Degree | Threads | AvgSerial | AvgSend | AvgReceive | AvgTotal | AvgSpeedup | Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50000 | 1 | 1.448718 | 0.000000 | 1.372585 | 0.000008 | 1.372595 | 1.0575 | YES |
| 50000 | 2 | 1.494965 | 0.000275 | 0.707585 | 0.000131 | 0.707987 | 2.1125 | YES |
| 50000 | 4 | 1.569390 | 0.000762 | 0.564932 | 0.000203 | 0.565898 | 2.7750 | YES |
| 50000 | 8 | 2.933275 | 0.008230 | 0.653282 | 0.003022 | 0.664532 | 4.4125 | YES |
| 100000 | 1 | 5.722500 | 0.000000 | 5.534875 | 0.000010 | 5.534900 | 1.0300 | YES |
| 100000 | 2 | 5.909375 | 0.000527 | 2.856200 | 0.000250 | 2.857000 | 2.0700 | YES |
| 100000 | 4 | 6.222525 | 0.001361 | 2.284902 | 0.000409 | 2.286672 | 2.7200 | YES |
| 100000 | 8 | 11.933875 | 0.008188 | 2.703075 | 0.005559 | 2.716825 | 4.3925 | YES |

### 2.1.4 Results Commentary
The program produced correct results in all executions. Increasing the number of processes significantly reduces the execution time, as the workload is distributed among them. However, at 8 processes the communication and synchronization cost increases, limiting further speedup. Compared to the shared memory implementations (OpenMP and Pthreads), MPI shows slightly worse performance, mainly due to the communication cost (`Bcast`/`Gatherv`), while OpenMP remains the fastest solution on a single node system. Therefore, MPI is more suitable for execution on multiple nodes, where the communication cost is offset by the distributed processing capability.

---

## Exercise 2.2: Sparse Matrix × Vector Multiplication with MPI

### 2.2.1 Problem Description
The exercise asks for the implementation of an MPI program for the efficient processing of sparse matrices, specifically the multiplication of a sparse square N×N matrix with a vector (SpMV), using the CSR (Compressed Sparse Row) format, for a given number of iterations where the result of each iteration becomes the input of the next. The program constructs the matrix and the vector, creates the CSR representation, and distributes to the other processes only the necessary data. The goal is to compare the performance of CSR against the classical dense representation, as well as to study the scaling with respect to the number of processes, sparsity percentage, matrix size, and number of iterations.

### 2.2.2 Solution Description
Process 0 creates a random dense matrix with the required sparsity percentage and an input vector, and constructs the CSR format. Then it distributes to each process only the CSR chunk of the rows assigned to it. In each iteration, each process calculates its local chunk `y = A_local · x`, and all processes exchange results with `MPI_Allgatherv` so that they all have the updated, complete vector `x` for the next iteration. To compare with the dense representation, the exact same logic of row distribution and `Allgatherv` is followed, but with the matrix in dense format. Correctness is verified by comparing the final `x` with the result of a serial CSR execution on process 0, and a Bash script automates the execution of the four experiments, recording the times in CSV files, and producing the relevant graphs.

### 2.2.3 Scaling Number of Processes
*(N = 2000, sparsity = 90%, iters = 10, processes = {1, 2, 4, 8})*

| Processes | CSR Build (s) | Distribution (s) | CSR SpMV Serial (s) | CSR SpMV Par (s) | Dense SpMV Par (s) | Speedup SpMV |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.010449 | 0.002744 | 0.002735 | 0.002897 | 0.042552 | 0.94 |
| 2 | 0.010732 | 0.002205 | 0.002903 | 0.001638 | 0.028500 | 1.77 |
| 4 | 0.011428 | 0.002055 | 0.002957 | 0.001033 | 0.024689 | 2.86 |
| 8 | 0.019920 | 0.002749 | 0.004747 | 0.002169 | 0.029869 | 2.19 |

**Observations:** Parallel CSR SpMV scales from 1 to 4 processes. At 8 processes, the speedup drops to 2.19x, likely due to contention and communication overhead. Dense SpMV benefits less from parallelism compared to CSR.

### 2.2.4 Effect of Sparsity
*(N = 2000, processes = 4, iters = 10, sparsity ∈ {0, 25, 50, 70, 80, 90, 95, 99}%)*

| Sparsity (%) | NNZ | Build (s) | CSR SpMV (s) | Dense SpMV (s) | CSR Total (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 4,000,000 | 0.028061 | 0.017777 | 0.024741 | 0.065963 |
| 25 | 3,000,119 | 0.029880 | 0.013174 | 0.024731 | 0.058339 |
| 50 | 1,998,332 | 0.030756 | 0.008596 | 0.024745 | 0.049615 |
| 70 | 1,197,700 | 0.022538 | 0.004649 | 0.024729 | 0.033412 |
| 80 | 798,216 | 0.016903 | 0.002699 | 0.024708 | 0.023709 |
| 90 | 398,755 | 0.011398 | 0.001028 | 0.024626 | 0.014475 |
| 95 | 199,398 | 0.008495 | 0.000515 | 0.024716 | 0.010053 |
| 99 | 40,137 | 0.006382 | 0.000224 | 0.024761 | 0.006855 |

**Observations:** CSR SpMV time decreases almost linearly with increasing sparsity, since the number of non-zero elements decreases proportionally. In contrast, Dense SpMV maintains a nearly constant time regardless of sparsity, as it always processes all N² elements. The total CSR time becomes less than Dense SpMV roughly from sparsity ≈ 70-80% and above, indicating that building the CSR representation is worthwhile only for sufficiently sparse matrices.

### 2.2.5 Effect of Matrix Size
*(sparsity = 90%, processes = 4, iters = 5, N ∈ {500, 1000, 2000, 3000, 5000})*

| N | CSR Build (s) | CSR SpMV (s) | Dense SpMV (s) | CSR Total (s) |
| :--- | :--- | :--- | :--- | :--- |
| 500 | 0.000784 | 0.000063 | 0.001119 | 0.001051 |
| 1000 | 0.002825 | 0.000147 | 0.004594 | 0.003538 |
| 2000 | 0.011279 | 0.000603 | 0.018934 | 0.013928 |
| 3000 | 0.026725 | 0.001750 | 0.043184 | 0.033147 |
| 5000 | 0.070191 | 0.005601 | 0.119749 | 0.088405 |

**Observations:** Both methods show increasing time with N, but CSR scales much more favorably due to the constant 90% sparsity. For N = 5000, Dense SpMV takes ~0.120s while CSR takes only ~0.088s, demonstrating a clear advantage for the CSR representation at larger matrix sizes, since the number of non-zero elements grows linearly with N, whereas the computational volume of the dense format grows with N².

### 2.2.6 Effect of Number of Iterations
*(N = 2000, sparsity = 90%, processes = 4, iters = {1, 3, 5, 8, 10, 15, 20})*

| Iterations | CSR Build (s) | CSR SpMV (s) | Dense SpMV (s) | CSR Total (s) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.011319 | 0.000166 | 0.014186 | 0.013537 |
| 3 | 0.011380 | 0.000399 | 0.016573 | 0.013829 |
| 5 | 0.011272 | 0.000592 | 0.018898 | 0.013916 |
| 8 | 0.011385 | 0.000851 | 0.022390 | 0.014281 |
| 10 | 0.011281 | 0.001065 | 0.024715 | 0.014404 |
| 15 | 0.011364 | 0.001453 | 0.030374 | 0.014863 |
| 20 | 0.011403 | 0.001874 | 0.036205 | 0.015329 |

**Observations:** The CSR build time remains constant regardless of the number of iterations, since it is done once before the loop. The CSR SpMV time increases linearly and slowly, while the Dense SpMV increases much faster, since each of its iterations costs significantly more. The crossover point is identified already from the 1st iteration under these parameters (N=2000, sparsity=90%); the total CSR time remains consistently smaller than Dense SpMV.

### 2.2.7 Results Commentary
The experimental results show that the CSR representation combined with distributed processing via MPI is an efficient solution for multiplying sparse matrices with a vector, provided the conditions match those for which the technique was designed: CSR outperforms dense when the sparsity percentage exceeds roughly 70-80%, when the matrix is large enough so that the O(nnz) advantage over O(N²) outweighs the construction cost, and when enough iterations are performed so that the fixed overhead of CSR is amortized.

---

## Exercise 2.3: Polynomial Multiplication with SIMD

### 2.3.1 Problem Description
The exercise asks for the extension of the serial polynomial multiplication program from Exercise 1.1 to use SIMD instructions, without any use of threads. The goal is to compare the performance of the serial O(n²) algorithm against a vectorized version of it for various polynomial degrees *n*, while simultaneously verifying that the two results match.

### 2.3.2 Solution Description
The vectorized implementation is based on the same algorithm as the serial one, but instead of processing one coefficient at a time, it processes four simultaneously using AVX2 instructions. Because the coefficients are of type `long long` (64-bit), each 256-bit register holds 4 values. The `_mm256_mul_epi32` instruction is used for the multiplications and is sufficient, as the coefficients have very small values (from -100 to 100), so the products are calculated accurately. Afterwards, the results are added normally to the output array using 64-bit operations.

### 2.3.3 Result 1: Effect of Polynomial Degree
*(n = {10000, 50000, 100000, 200000}, average of 4 iterations)*

| Degree | AvgCreation | AvgScalar | AvgSIMD | AvgSpeedup | Correct |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10000 | 0.000309 | 0.059914 | 0.029856 | 2.0075 | YES |
| 50000 | 0.001269 | 1.501122 | 0.913065 | 1.6425 | YES |
| 100000 | 0.002466 | 5.996575 | 3.628350 | 1.6525 | YES |
| 200000 | 0.004997 | 24.303000 | 14.709175 | 1.6525 | YES |

### 2.3.4 Results Commentary
Correctness is confirmed across all sizes; the vectorized algorithm produces exactly the same result as the serial one. A speedup is observed at all sizes, stabilizing around 1.65x for n ≥ 50000.

---

## Exercise 2.4: Accelerating Maxwell Simulator with CUDA

### 2.4.1 Problem Description
The exercise asks for the completion of the NVIDIA online course "Getting Started with Accelerated Computing in Modern CUDA C++", which provides a simple, functional 2D simulator for electromagnetic wave propagation (Maxwell's Equations Simulator), initially implemented in a CPU-only form with C++/std algorithms. The task requires a gradual acceleration of the application using CUDA/Thrust/CUB, preserving the correctness of the simulation, across three steps: porting to the GPU, avoiding intermediate buffers with fancy iterators, and implementing a cooperative CUDA kernel to construct a coarse grid of average values.

### 2.4.2 Solution Description
*   **Step 1 – Port CPU Simulator to GPU:**
    All functions (`update_hx`, `update_hy`, `update_dz`, `update_ez`) were converted to accept `thrust::device_vector<float>&` instead of `std::vector<float>&`, with `std::transform`/`std::for_each` replaced by the corresponding `thrust::transform`/`thrust::for_each` using `__device__` lambdas. The `copy_to_host()` function and the host-side copy of `hx/hy/dz/ez` inside `simulate()` were completely removed: the computation now happens directly on the incoming device containers, without any host-device transfers between simulation steps.
*   **Step 2 – Accelerate the Simulator with Fancy Iterators:**
    The intermediate buffer was completely removed: the difference `ez[i+n]-ez[i]` (and correspondingly `ez[i]-ez[i+1]`) is now calculated via a `transform_iterator` over a `zip_iterator` that "binds" the two shifted ranges of `ez`. Additionally, `cell_ids` was replaced with `thrust::counting_iterator<int>`, which produces the same index values without any memory allocation.
*   **Step 3 – Use Cooperative Algorithms to Coarse the Grid:**
    A CUDA kernel was implemented where each block processes a tile of the grid. Threads of the same block cooperate to calculate the sum of the segment's values, using shared memory so that the calculation is done faster. At the end, only the first thread of the block (thread 0) writes the average to the new grid, thereby avoiding simultaneous writes (race conditions) by the rest of the threads.

### 2.4.3 Results Commentary
All three steps successfully passed both the correctness check and the minimum throughput threshold. The largest performance improvement is observed between Step 1 and Step 2; the removal of the intermediate buffer via zip/transform iterators saves an entire global memory read/write pass per simulation step. Between Step 2 and Step 3 the difference is negligible, which is expected since Step 3 concerns a separate kernel and not the main update loop for the `hx/hy/dz/ez` fields.
assignment_2_mpi_cuda.md
