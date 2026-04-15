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

// Tile sizes — shared by both kernels.
#define Br  16    // Q-tile rows
#define Bc  16    // K/V-tile rows

// ── optimized kernel only ────────────────────────────────────────────────────

// Shared memory padding (in __half elements) to avoid bank conflicts.
// Constraint: (D + PAD) / 2 must not be divisible by 32.
#define PAD  8

// rocWMMA tile shape — 16×16×16 supported on CDNA and RDNA3.
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// Thread block size.
#define NUM_THREADS  128

// AMD wavefront size: 64 for CDNA (MI100/MI200/MI300), 32 for RDNA3 (RX 7000).
// Must match --offload-arch: gfx9xx → 64, gfx11xx → 32.
#define WAVEFRONT_SIZE   64
#define NUM_WAVEFRONTS   (NUM_THREADS / WAVEFRONT_SIZE)
