# GPU Kernel Profiling and Performance Optimization

## Project Overview

This project explores **GPU kernel performance analysis and optimization** using these 3 kernels:

1. **Matrix Multiplication**
2. **Parallel Reduction**
3. **Fash Attention**

The project focuses on profiling these kernels on **NVIDIA A100 GPUs** and **AMD MI50 GPUs**, identifying performance bottlenecks, and applying architecture-specific optimizations to improve throughput, memory efficiency, and overall performance.

For FlashAttention, the profiling is done on **NVIDIA H200 GPUs*** and **AMD MI300X GPUs** as AMD MI50 does not support Matrix Fused Multiply-Add (MFMA) and to ensure fair comparison (same generation).

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
# Change directory to flash attention
cd kernels/flashattention

make build

# Parallel Reduction
# Copy all files (reduction_nvidia.cu, run_nvidia.pbs) under directory kernels/parallel_reduction/nvidia to NVIDIA
qsub run_nvidia.pbs

# GEMM
```

#### AMD GPU

```bash
# FlashAttention (from repo root)
cd kernels/flashattention

make build-amd

# Parallel Reduction
# Copy all files (reduction_amd.cpp, run_amd.sh) under directory kernels/parallel_reduction/amd to AMD pod
./run_amd.sh

# GEMM

```

### Plot Results (Parallel Reduction)

Copy all results from NVIDIA and AMD to local and run:

```bash
python3 plot_results.py
```

### Matrix multiplication
nvcc cudaMatMult.cu -o run_gemm -O3 -DBLOCK_SIZE_X=16 -DREG_TILE_X=2 -DUF=4 -DVEC_TYPE=float4 -DNUM_STREAMS=4
qsub matMult.pbs

hipcc rocmMatMult.cpp -o run_gemm -O3 -DBLOCK_SIZE_X=16 -DREG_TILE_X=2 -DUF=4 -DVEC_TYPE=float4 -DNUM_STREAMS=4


