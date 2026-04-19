# ─────────────────────────────────────────────────────────────────────────────
# Flash Attention — Naive + Optimised CUDA/HIP Kernels
# ─────────────────────────────────────────────────────────────────────────────
#
# NVIDIA targets:
#   make naive               naive with CPU verification
#   make naive-nocheck       naive without CPU verification
#   make optimized           optimized with CPU verification
#   make optimized-nocheck   optimized without CPU verification
#   make build               all four NVIDIA binaries
#   make profile-naive       nsys profile (nocheck)
#   make profile-optimized   nsys profile (nocheck)
#   make profile             both nsys profiles
#
# AMD targets (same pattern, -amd suffix):
#   make naive-amd           make optimized-amd
#   make naive-amd-nocheck   make optimized-amd-nocheck
#   make build-amd
#   make profile-amd-naive   make profile-amd-optimized
#   make profile-amd
#
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := -std=c++17

# Compute capability: sm_70 = Volta, minimum for FP16 Tensor Cores (WMMA).
# Run: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Common: 70 (V100), 75 (T4/RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100/H200)
ARCH        := -gencode arch=compute_80,code=sm_80

# -lineinfo preserves source-line mappings so Nsight can annotate your code.
OPT_FLAGS   := -O3 -lineinfo
WARN_FLAGS  := -Xcompiler -Wall,-Wextra

NVCC_FLAGS  := $(CXX_STD) $(ARCH) $(OPT_FLAGS) $(WARN_FLAGS)

KERNEL_DIR  := /nvidia

NAIVE_SRC   := $(KERNEL_DIR)/naive.cu
OPT_SRC     := $(KERNEL_DIR)/optimized.cu

TIMESTAMP   := $(shell date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA — build targets
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all build naive naive-nocheck optimized optimized-nocheck

all: build

build: naive naive-nocheck optimized optimized-nocheck

naive: $(NAIVE_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

naive-nocheck: $(NAIVE_SRC)
	$(NVCC) $(NVCC_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

optimized: $(OPT_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

optimized-nocheck: $(OPT_SRC)
	$(NVCC) $(NVCC_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA — Nsight Systems profiling (always nocheck so nsys captures GPU only)
#   --trace=cuda        captures kernel launches, memcpy, and API calls
#   --cuda-memory-usage tracks device memory allocations over time
#   --output            names the .nsys-rep artefact
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: profile profile-naive profile-optimized

profile: profile-naive profile-optimized

profile-naive: naive-nocheck
	nsys profile \
	    --trace=cuda,nvtx \
	    --cuda-memory-usage=true \
	    --gpu-metrics-device=all \
	    --output=naive-$(TIMESTAMP) \
	    ./naive-nocheck
	@echo "Saved: naive-$(TIMESTAMP).nsys-rep"

profile-optimized: optimized-nocheck
	nsys profile \
	    --trace=cuda,nvtx \
	    --cuda-memory-usage=true \
	    --gpu-metrics-device=all \
	    --output=optimized-$(TIMESTAMP) \
	    ./optimized-nocheck
	@echo "Saved: optimized-$(TIMESTAMP).nsys-rep"

# ─────────────────────────────────────────────────────────────────────────────
# AMD / HIP
# ─────────────────────────────────────────────────────────────────────────────

HIPCC       := hipcc

AMD_DIR     := amd
NAIVE_AMD_SRC   := $(AMD_DIR)/naive.cpp
OPT_AMD_SRC     := $(AMD_DIR)/optimized.cpp

# Set --offload-arch to match your GPU:
#   gfx942  = MI300X
#   gfx90a  = MI250X
#   gfx908  = MI100
#   gfx1100 = RX 7900 XT
AMD_ARCH    := gfx942
HIP_FLAGS   := -std=c++17 -O3 -g --offload-arch=$(AMD_ARCH)

.PHONY: build-amd naive-amd naive-amd-nocheck optimized-amd optimized-amd-nocheck

build-amd: naive-amd naive-amd-nocheck optimized-amd optimized-amd-nocheck

naive-amd: $(NAIVE_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

naive-amd-nocheck: $(NAIVE_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

optimized-amd: $(OPT_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

optimized-amd-nocheck: $(OPT_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

# ─────────────────────────────────────────────────────────────────────────────
# AMD — rocprof profiling
#   --hip-trace   captures HIP API calls and kernel launches
#   --hsa-trace   captures HSA runtime calls (lower-level)
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: profile-amd profile-amd-naive profile-amd-optimized

profile-amd: profile-amd-naive profile-amd-optimized

profile-amd-naive: naive-amd-nocheck
	rocprof --hip-trace --hsa-trace \
	    -o naive-amd-$(TIMESTAMP).csv \
	    ./naive-amd-nocheck
	@echo "Saved: naive-amd-$(TIMESTAMP).csv"

profile-amd-optimized: optimized-amd-nocheck
	rocprof --hip-trace --hsa-trace \
	    -o optimized-amd-$(TIMESTAMP).csv \
	    ./optimized-amd-nocheck
	@echo "Saved: optimized-amd-$(TIMESTAMP).csv"

# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f naive naive-nocheck optimized optimized-nocheck
	rm -f naive-amd naive-amd-nocheck optimized-amd optimized-amd-nocheck
