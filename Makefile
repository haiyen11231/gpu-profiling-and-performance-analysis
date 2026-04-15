# ─────────────────────────────────────────────────────────────────────────────
# Flash Attention — Naive + Optimised CUDA Kernels
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := -std=c++17

# Compute capability: sm_70 = Volta, minimum for FP16 Tensor Cores (WMMA).
# Run: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Common: 70 (V100), 75 (T4/RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
ARCH        := -gencode arch=compute_80,code=sm_80

# -lineinfo preserves source-line mappings so Nsight can annotate your code.
OPT_FLAGS   := -O3 -lineinfo
WARN_FLAGS  := -Xcompiler -Wall,-Wextra

NVCC_FLAGS  := $(CXX_STD) $(ARCH) $(OPT_FLAGS) $(WARN_FLAGS)

KERNEL_DIR  := kernels/flashattention/nvidia

NAIVE_SRC   := $(KERNEL_DIR)/naive.cu
OPT_SRC     := $(KERNEL_DIR)/optimized.cu

NAIVE_BIN        := naive
OPT_BIN          := optimized
NAIVE_PROF_BIN   := naive_profile
OPT_PROF_BIN     := optimized_profile

TIMESTAMP        := $(shell date +%Y%m%d-%H%M%S)

# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all build clean profile profile-naive profile-optimized

all: build

build: $(NAIVE_BIN) $(OPT_BIN)

$(NAIVE_BIN): $(NAIVE_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(OPT_BIN): $(OPT_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Profile binaries skip CPU verification so nsys only captures GPU work.
$(NAIVE_PROF_BIN): $(NAIVE_SRC)
	$(NVCC) $(NVCC_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

$(OPT_PROF_BIN): $(OPT_SRC)
	$(NVCC) $(NVCC_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

# ─────────────────────────────────────────────────────────────────────────────
# Profiling — Nsight Systems
#   --trace=cuda        captures kernel launches, memcpy, and API calls
#   --cuda-memory-usage tracks device memory allocations over time
#   --output            names the .nsys-rep artefact
# ─────────────────────────────────────────────────────────────────────────────

profile: profile-naive profile-optimized

profile-naive: $(NAIVE_PROF_BIN)
	nsys profile \
	    --trace=cuda,nvtx \
	    --cuda-memory-usage=true \
		--gpu-metrics-device all \
	    --output=naive-$(TIMESTAMP) \
	    ./$(NAIVE_PROF_BIN)
	@echo "Saved: naive-$(TIMESTAMP).nsys-rep"

profile-optimized: $(OPT_PROF_BIN)
	nsys profile \
	    --trace=cuda,nvtx \
	    --cuda-memory-usage=true \
		--gpu-metrics-device all \
	    --output=optimized-$(TIMESTAMP) \
	    ./$(OPT_PROF_BIN)
	@echo "Saved: optimized-$(TIMESTAMP).nsys-rep"

# ─────────────────────────────────────────────────────────────────────────────
# AMD / HIP
# ─────────────────────────────────────────────────────────────────────────────

HIPCC       := hipcc

AMD_DIR     := kernels/flashattention/amd
NAIVE_AMD_SRC   := $(AMD_DIR)/naive.cpp
OPT_AMD_SRC     := $(AMD_DIR)/optimized.cpp

NAIVE_AMD_BIN       := naive_amd
OPT_AMD_BIN         := optimized_amd
NAIVE_AMD_PROF_BIN  := naive_amd_profile
OPT_AMD_PROF_BIN    := optimized_amd_profile

# Set --offload-arch to match your GPU (gfx90a = MI250X, gfx908 = MI100, gfx1100 = RX 7900).
AMD_ARCH    := gfx942
HIP_FLAGS   := -std=c++17 -O3 -g --offload-arch=$(AMD_ARCH)

.PHONY: build-amd profile-amd profile-amd-naive profile-amd-optimized

build-amd: $(NAIVE_AMD_BIN) $(OPT_AMD_BIN)

$(NAIVE_AMD_BIN): $(NAIVE_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

$(OPT_AMD_BIN): $(OPT_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -o $@ $<

$(NAIVE_AMD_PROF_BIN): $(NAIVE_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

$(OPT_AMD_PROF_BIN): $(OPT_AMD_SRC)
	$(HIPCC) $(HIP_FLAGS) -DSKIP_CPU_VERIFY -o $@ $<

# Profiling — rocprof
#   --hip-trace   captures HIP API calls and kernel launches
#   --hsa-trace   captures HSA runtime calls (lower-level)
profile-amd: profile-amd-naive profile-amd-optimized

profile-amd-naive: $(NAIVE_AMD_PROF_BIN)
	rocprof --hip-trace --hsa-trace \
	    -o naive-amd-$(TIMESTAMP).csv \
	    ./$(NAIVE_AMD_PROF_BIN)
	@echo "Saved: naive-amd-$(TIMESTAMP).csv"

profile-amd-optimized: $(OPT_AMD_PROF_BIN)
	rocprof --hip-trace --hsa-trace \
	    -o optimized-amd-$(TIMESTAMP).csv \
	    ./$(OPT_AMD_PROF_BIN)
	@echo "Saved: optimized-amd-$(TIMESTAMP).csv"

# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(NAIVE_BIN) $(OPT_BIN) $(NAIVE_PROF_BIN) $(OPT_PROF_BIN)
	rm -f $(NAIVE_AMD_BIN) $(OPT_AMD_BIN) $(NAIVE_AMD_PROF_BIN) $(OPT_AMD_PROF_BIN)
