#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Flash Attention — shared configuration
// ─────────────────────────────────────────────────────────────────────────────

// Sequence length — must be divisible by Br and Bc.
#define N  (1 << 15)

// Head dimension.
// naive:     D ≤ 64 recommended (O_acc[D] lives in registers; 256 regs/thread
//            × 256 threads/block hits the 64K SM limit at D=256).
// optimized: designed for D=256; supports up to D=256 with current tile sizes.
#define D  64

// Tile sizes — shared by both kernels.
#define Br  16    // Q-tile rows
#define Bc  16    // K/V-tile rows

// ── optimized kernel only ────────────────────────────────────────────────────

// Shared memory padding (in half elements) to avoid bank conflicts.
// Constraint: (D + PAD) / 2 must not be divisible by 32.
#define PAD  8

// WMMA tile shape — fixed by the hardware (FP16 Tensor Cores).
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// Thread block: 128 threads = 4 warps.
#define NUM_THREADS  128
#define NUM_WARPS    (NUM_THREADS / 32)
