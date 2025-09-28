# **CUDA PageRank**

A high-performance, GPU-accelerated implementation of Google's PageRank algorithm using CUDA C++. This project analyzes two distinct GPU parallelization strategies—a simple **Scalar Kernel** and an optimized **Vector Kernel**—achieving up to **\~4800x speedup** over a standard CPU implementation on a real-world web graph.

## **🎯 Overview**

PageRank is the foundational algorithm that powered Google's search engine, ranking web pages based on their link structure. This project implements PageRank on the GPU, processing a graph of nearly one million web pages and over five million links. The core focus is to compare the performance of assigning one thread per page versus one warp (32 threads) per page.

### **Key Results**

The analysis clearly shows the superiority of the warp-per-row (Vector) approach, which better utilizes the GPU's memory bandwidth and parallel processing capabilities.

| Implementation | Time / Iter (ms) | Total Time (ms) | Speedup vs. CPU | Memory Bandwidth |
| :---- | :---- | :---- | :---- | :---- |
| **CPU (NumPy)** | 3391.960 | 210,301.53 | 1x | N/A |
| **GPU Scalar** | 3.178 | 197.06 | **1067x** | 32.62 GB/s |
| **GPU Vector** | **0.696** | **6.96** | **4873x** | **148.92 GB/s** |

## **🧠 Algorithm Background**

### **What is PageRank?**

PageRank determines the "importance" of web pages using this principle:

* **A page is important if other important pages link to it**.  
* It's analogous to academic citations—a paper is considered credible if cited by other credible papers.  
* The algorithm iteratively computes these importance scores (ranks) until they stabilize.

### **Mathematical Foundation**

The PageRank formula is implemented as follows:

PR(page)=N1−d​+d×p∈in-links∑​out-links(p)PR(p)​  
Where:

* d \= 0.85 is the damping factor, simulating an 85% chance a user clicks a link.  
* N is the total number of pages in the graph.  
* The summation is over all pages p that link *to* the current page.

## **🚀 Implementation Approach**

### **Data Structure: CSR (Compressed Sparse Row)**

To represent the sparse web graph efficiently on the GPU, we convert the edge list into an **in-neighbor CSR format**.

Original: "Page A links to Page B"  
CSR:      "For each page, store an array of all pages that link TO it"

This structure is ideal for a "pull-style" PageRank and enables:

* **Coalesced memory access** patterns on the GPU.  
* **Efficient sparse matrix-vector multiplication** (SpMV), the core operation.  
* **Minimal memory footprint** compared to a dense matrix.

### **GPU Kernels Compared**

This project implements and compares two CUDA kernels for the SpMV step:

1. **Scalar Kernel (spmv\_scalar\_kernel)**: The straightforward approach.  
   * **Strategy**: **One GPU thread processes one full row** (one web page).  
   * **Limitation**: Can lead to work imbalance if pages have vastly different numbers of incoming links.  
2. **Vector Kernel (spmv\_vector\_kernel)**: A more advanced, warp-centric approach.  
   * **Strategy**: **One GPU warp (32 threads) collaborates to process one row**.  
   * **Optimization**: Threads within a warp divide the work of summing up contributions from incoming links. A highly efficient warpReduceSum intrinsic is used to aggregate the final result, maximizing memory throughput.

## **📊 Performance Analysis**

### **Dataset: Stanford Web-Google Graph**

* **Nodes**: 916,428 web pages  
* **Edges**: 5,105,039 links  
* **Average links per page**: 5.57  
* **Source**: [SNAP Stanford](https://snap.stanford.edu/data/web-Google.html)

### **Convergence Behavior**

A fascinating result was the difference in convergence. While mathematically identical, the different order of floating-point operations in the parallel reduction led the Vector Kernel to converge much faster.

**Scalar Kernel Convergence:**

Iter | Residual (L1) | Time (ms) | Status  
\-----|---------------|-----------|--------  
   1 | 8.508467e-01 |    3.446 | Running  
  10 | 1.090636e-02 |    3.298 | Running  
  ...| ...           |    ...   | ...  
  60 | 1.219864e-06 |    2.993 | Running  
  62 | 8.729468e-07 |    2.972 | CONVERGED ✅

**Vector Kernel Convergence:**

Iter | Residual (L1) | Time (ms) | Status  
\-----|---------------|-----------|--------  
   1 | 6.913818e-01 |    0.737 | Running  
  10 | 4.030835e-07 |    0.682 | CONVERGED ✅

## **🛠 How to Reproduce**

### **Prerequisites**

1. **Google Colab** with a GPU runtime enabled.  
   * Go to Runtime → Change runtime type → Hardware accelerator → GPU.  
2. **CUDA Environment**  
   * CUDA 12.5+ (standard in Colab).  
   * NVIDIA GPU with compute capability 7.5+ (the notebook was tested on a Tesla T4).

### **Step-by-Step Instructions**

1. **Open the Notebook in Colab**: Use the main .ipynb file from the repository.  
2. **Run Environment Check**: The first cell verifies GPU availability and logs system details. It should detect a **Tesla T4** GPU.  
3. **Download and Prepare Data**: The second and third cells download the web-Google dataset and convert it into the optimized CSR format.  
4. **Run CPU Baseline**: This cell runs the pure NumPy implementation for comparison. It takes approximately 3-4 minutes to complete.  
5. **Compile and Run CUDA Kernels**: This cell writes the CUDA C++ source code, compiles it with nvcc, and runs the executable. The program will benchmark both the Scalar and Vector kernels, printing the results for each. This step completes in under a second.  
6. **Analyze Final Results**: The last cell collects the performance data from the CPU and GPU runs and displays the final summary table.

### **Expected Output**

\=== CUDA PageRank: Scalar vs Vector Kernel Comparison \===

Problem Size:  
  Nodes (web pages): 916428  
  Edges (links):     5105039  
  Avg links/page:    5.57

\--- Testing Scalar Method \---  
...  
Convergence: SUCCESS after 62 iterations.  
JSON\_RESULT: {"method":"GPU Scalar", ... "gbs":32.62}

\--- Testing Vector Method \---  
...  
Convergence: SUCCESS after 10 iterations.  
JSON\_RESULT: {"method":"GPU Vector", ... "gbs":148.92}

## **🎓 Educational Value**

This implementation is a practical case study in high-performance computing, demonstrating:

1. **Sparse Matrix Algorithms**: Efficiently handling large, sparse graphs common in real-world data.  
2. **GPU Parallelization Strategies**: Understanding the trade-offs between simple (thread-per-row) and advanced (warp-per-row) parallel designs.  
3. **Performance Optimization**: Highlighting the critical role of memory coalescing and maximizing memory bandwidth.  
4. **CUDA Programming**: Using CUDA C++, Thrust, and warp-level intrinsics (\_\_shfl\_down\_sync) for fine-grained optimization.  
5. **Numerical Stability**: Observing how parallel execution can affect the convergence path of iterative algorithms.

## **📚 References**

1. [Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/)  
2. [SNAP Stanford Network Analysis Project](https://snap.stanford.edu/)  
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## **📄 License**

This project is licensed under the MIT License. Feel free to use and adapt this code for educational and research purposes.
