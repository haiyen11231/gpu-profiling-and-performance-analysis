#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Flash Attention — shared configuration (AMD / HIP)
// ─────────────────────────────────────────────────────────────────────────────

// Sequence length — must be divisible by Br and Bc.
#define SEQ_LEN  (1 << 15)

// Head dimension.
// naive:     D ≤ 64 recommended (O_acc[D] lives in VGPRs; larger D spills).
// optimized: designed for D=256; supports any multiple of WMMA_N.
#define D  64

// Tile sizes for the naive kernel.
// Constrained by dim3(Bc, Br) ≤ 1024 threads and O_acc[D] register budget.
#define Br_NAIVE  16
#define Bc_NAIVE  32

// Tile sizes for the optimized (FA2-style) kernel.
// Br/WMMA_M must equal NUM_WAVEFRONTS so every wavefront owns one row-strip.
#define Br  64    // Q-tile rows  (= NUM_WAVEFRONTS × WMMA_M = 4 × 16)
#define Bc  64    // K/V-tile rows

// ── optimized kernel only ────────────────────────────────────────────────────

// Shared memory padding (in __half elements) to avoid bank conflicts.
// Constraint: (D + PAD) / 2 must not be divisible by 32.
#define PAD  8

// rocWMMA tile shape — 16×16×16 supported on CDNA and RDNA3.
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// Thread block: 256 threads = 4 wavefronts, matching NVIDIA's 4-warp layout.
#define NUM_THREADS  256

// AMD wavefront size: 64 for CDNA (MI100/MI200/MI300), 32 for RDNA3 (RX 7000).
// Must match --offload-arch: gfx9xx → 64, gfx11xx → 32.
#define WAVEFRONT_SIZE   64
#define NUM_WAVEFRONTS   (NUM_THREADS / WAVEFRONT_SIZE)
