# CUDA PageRank Implementation

A high-performance GPU-accelerated implementation of Google's PageRank algorithm using CUDA, demonstrating **~1000x speedup** over CPU implementation on real web graph data.

## 🎯 Overview

PageRank is the foundational algorithm that powered Google's search engine, ranking web pages based on their link structure. This project implements PageRank using CUDA for GPU acceleration, processing nearly a million web pages with over 5 million links between them.

### Key Results

| Implementation | Time | Speedup | Memory Bandwidth |
|----------------|------|---------|------------------|
| **CPU (NumPy)** | 218.4 seconds | 1x | Limited |
| **GPU (CUDA)** | **0.22 seconds** | **985x** | **43.0 GB/s** |

## 🧠 Algorithm Background

### What is PageRank?

PageRank determines the "importance" of web pages using this principle:
- **A page is important if other important pages link to it**
- It's like academic citations - papers are credible if cited by other credible papers
- The algorithm iteratively computes importance scores until convergence

### Mathematical Foundation

The PageRank formula:
```
PR(page) = (1-d)/N + d × Σ(PR(linking_page) / outbound_links(linking_page))
```

Where:
- `d = 0.85` (damping factor - simulates random web browsing)
- `N` = total number of pages
- The sum is over all pages linking to the current page

## 🚀 Implementation Approach

### Data Structure: CSR (Compressed Sparse Row)

We convert the web graph into CSR format for efficient GPU processing:

```
Original: "Page A links to Page B"
CSR:      "For each page, store all pages that link TO it"
```

This optimization enables:
- **Coalesced memory access** on GPU
- **Efficient sparse matrix-vector multiplication**
- **Minimal memory overhead**

### GPU Kernels

1. **SpMV Kernel**: Performs sparse matrix-vector multiplication
   ```cuda
   __global__ void spmv_pull_kernel(...)
   ```

2. **Finalize Kernel**: Applies damping factor and normalization
   ```cuda
   __global__ void finalize_kernel(...)
   ```

### Algorithm Steps

1. **Initialize**: All pages start with equal rank `1/N`
2. **Iterate**: 
   - Compute matrix-vector product (link propagation)
   - Handle dangling nodes (pages with no outbound links)
   - Apply damping factor
   - Check convergence
3. **Converge**: Stop when rank changes are below threshold

## 📊 Performance Analysis

### Dataset: Stanford Web-Google Graph
- **Nodes**: 916,428 web pages
- **Edges**: 5,105,039 links
- **Average links per page**: 5.57
- **Source**: [SNAP Stanford](https://snap.stanford.edu/data/web-Google.html)

### Convergence Behavior
```
Iteration | Residual (L1) | Time (ms) | Status
----------|---------------|-----------|--------
    1     | 8.508467e-01  |   3.668   | Running
   10     | 1.329880e-02  |   3.437   | Running
   20     | 1.827460e-03  |   3.445   | Running
   30     | 3.289641e-04  |   3.402   | Running
   40     | 6.311191e-05  |   3.221   | Running
   50     | 1.240221e-05  |   3.244   | Running
   60     | 2.463703e-06  |   3.294   | Running
   66     | 9.360145e-07  |   3.204   | CONVERGED ✅
```

### Memory Usage
- **Total GPU Memory**: 112.9 MB
- **Memory Transfer Time**: 197.7 ms (one-time overhead)
- **Memory Bandwidth**: 43.0 GB/s

## 🛠 How to Reproduce

### Prerequisites

1. **Google Colab** with GPU runtime enabled
   - Go to Runtime → Change runtime type → Hardware accelerator → GPU

2. **CUDA Environment**
   - CUDA 12.4+ (available in Colab)
   - NVIDIA GPU with compute capability 7.5+ (Tesla T4 works perfectly)

### Step-by-Step Instructions

1. **Clone and Open Notebook**
   ```bash
   # Open in Google Colab
   https://colab.research.google.com/github/anshulk-cmu/CUDA_PageRank/blob/main/CUDA_PageRank_(Google_Search_Engine).ipynb
   ```

2. **Run Environment Check**
   ```python
   # First cell - verifies GPU availability
   # Should show Tesla T4 or similar GPU
   ```

3. **Download and Prepare Data**
   ```python
   # Downloads Stanford web-Google dataset
   # Converts to CSR format for GPU processing
   ```

4. **Run CPU Baseline**
   ```python
   # Runs NumPy implementation for comparison
   # Takes ~3-4 minutes
   ```

5. **Compile and Run CUDA**
   ```python
   # Compiles CUDA code and executes
   # Takes ~0.2 seconds!
   ```

### Expected Output

```
=== CUDA PageRank Implementation ===
Problem Size:
  Nodes (web pages): 916428
  Edges (links):     5105039
  Avg links/page:    5.57

Memory transfer completed in 197.7 ms (112.9 MB total)

Starting PageRank iterations...
[Progress table showing convergence]

=== Results Summary ===
Convergence:       SUCCESS after 66 iterations
Total time:        221.915 ms
Memory bandwidth:  42.98 GB/s
```

## 📁 Project Structure

```
CUDA_PageRank/
├── CUDA_PageRank_(Google_Search_Engine).ipynb  # Main notebook
├── README.md                                   # This file
└── Generated during execution:
    ├── /content/drive/MyDrive/cuda-pagerank/
    │   ├── data/web-Google.txt                 # Raw graph data
    │   ├── prep/in_csr_webGoogle.npz          # Preprocessed CSR
    │   └── bin/                               # Binary files for CUDA
    │       ├── row_ptr.bin
    │       ├── col_idx.bin
    │       ├── val.bin
    │       ├── outdeg.bin
    │       └── meta.json
    └── /content/pagerank_pull.cu              # CUDA source code
```

## 🔧 Technical Details

### GPU Configuration
- **Threads per block**: 256
- **Total blocks**: 3,580
- **Architecture**: Compiled for sm_75 (Tesla T4)

### Numerical Precision
- **Data type**: Double precision (float64)
- **Convergence tolerance**: 1e-6
- **Damping factor**: 0.85 (standard PageRank value)

### Memory Optimization
- **CSR format**: Minimizes memory footprint
- **Sorted column indices**: Improves memory coalescing
- **Thrust library**: High-performance parallel primitives

### Error Handling
- Binary file validation
- GPU memory allocation checks
- Convergence monitoring with iteration limits

## 🎓 Educational Value

This implementation demonstrates several important concepts:

1. **Sparse Matrix Operations**: Efficient handling of large, sparse graphs
2. **GPU Memory Management**: Optimal data transfer and storage patterns
3. **Parallel Algorithm Design**: Mapping iterative algorithms to GPU architecture
4. **Performance Optimization**: Memory coalescing and bandwidth utilization
5. **Real-world Applications**: How academic algorithms power industry solutions

## 🔬 Extending the Project

### Potential Improvements

1. **Multi-GPU Support**: Scale to even larger graphs
2. **Mixed Precision**: Use float16 for memory savings
3. **Graph Partitioning**: Handle graphs larger than GPU memory
4. **Personalized PageRank**: Implement topic-sensitive variants
5. **Comparative Analysis**: Test other graph algorithms

### Alternative Datasets

Try other SNAP datasets:
- `web-Stanford.txt` (smaller, good for testing)
- `web-BerkStan.txt` (medium size)
- `web-NotreDame.txt` (different graph properties)

## 📚 References

1. [Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/)
2. [SNAP Stanford Network Analysis Project](https://snap.stanford.edu/)
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
4. [Thrust Documentation](https://thrust.github.io/)

## 📄 License

MIT License - feel free to use this code for research and educational purposes.

---

**Note**: This implementation prioritizes educational clarity and performance demonstration. For production use, consider additional optimizations like memory pooling, error recovery, and distributed computing support.
