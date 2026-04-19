# GPU Kernel Profiling and Performance Optimization

## Project Overview

This project explores **GPU kernel performance analysis and optimization** using these 3 kernels:

1. **Matrix Multiplication**
2. **Parallel Reduction**
3. **Fash Attention**

The project focuses on profiling these kernels on **NVIDIA A100 GPUs** and **AMD MI50 GPUs**, identifying performance bottlenecks, and applying architecture-specific optimizations to improve throughput, memory efficiency, and overall performance.

---

## Setup

### Dependencies

- **CUDA Toolkit** (for NVIDIA GPU support)
- **ROCm / HIP** (for AMD GPU support)
- **Python 3** with `numpy`, `pandas`, `matplotlib` (for plotting reduction results)

### Build Instructions

#### NVIDIA GPU

```bash
# FlashAttention (from repo root)
...

# Parallel Reduction
# Copy all files (reduction_nvidia.cu, run_nvidia.pbs) under directory kernels/parallel_reduction/nvidia to NVIDIA
qsub run_nvidia.pbs

# GEMM
...
```

Profiling (NVIDIA):

```bash
# FlashAttention profiling (from repo root)
...
```

#### AMD GPU

```bash
# FlashAttention (from repo root)
...

# Parallel Reduction
# Copy all files (reduction_amd.cpp, run_amd.sh) under directory kernels/parallel_reduction/amd to AMD pod
./run_amd.sh

# GEMM
...
```

Profiling (AMD):

```bash
# FlashAttention profiling (from repo root)
...
```

### Plot Results (Parallel Reduction)

Copy all results from NVIDIA and AMD to local and run:

```bash
python3 plot_results.py
```
