# ─────────────────────────────────────────────────────────────────────────────
# Flash Attention — Naive + Optimised CUDA Kernels
# ─────────────────────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := -std=c++17

# Compute capability: sm_70 = Volta, minimum for FP16 Tensor Cores (WMMA).
# Run: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Common: 70 (V100), 75 (T4/RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
ARCH        := -gencode arch=compute_70,code=sm_70

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
		--gpu-metrics-device all
	    --output=naive-$(TIMESTAMP) \
	    ./$(NAIVE_PROF_BIN)
	@echo "Saved: naive-$(TIMESTAMP).nsys-rep"

profile-optimized: $(OPT_PROF_BIN)
	nsys profile \
	    --trace=cuda,nvtx \
	    --cuda-memory-usage=true \
		--gpu-metrics-device all
	    --output=optimized-$(TIMESTAMP) \
	    ./$(OPT_PROF_BIN)
	@echo "Saved: optimized-$(TIMESTAMP).nsys-rep"

# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(NAIVE_BIN) $(OPT_BIN) $(NAIVE_PROF_BIN) $(OPT_PROF_BIN)
