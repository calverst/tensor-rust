# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build --release      # Optimized build (recommended for benchmarking)
cargo build                # Debug build
cargo run --release        # Build and run benchmark
cargo clippy               # Lint
```

The binary requires an OpenCL-capable GPU and the `hgemm.cl` kernel file. At runtime, the code searches for `hgemm.cl` in the working directory and its parent.

## Architecture

This is a single-binary Rust tool that auto-tunes an OpenCL FP16 batched GEMM kernel with a Q4_K_M-quantized B matrix. It has no library interface — the entire program is `src/main.rs` plus the kernel `hgemm.cl`.

**Fixed problem dimensions** (hardcoded at top of `main.rs`): `BATCH=36`, `M=256`, `N=32`, `K=256`, column-major layout. K must equal 256 (= QK_K) for Q4_K_M to work correctly.

**Execution flow:**
1. **Device selection** — scans all OpenCL platforms/devices, scores them by vendor (AMD/NVIDIA > Intel), device type (GPU > CPU), and OpenCL version, picks the best.
2. **Data generation** — `generate_data()` builds deterministic FP16 test matrices; `reference_gemm()` computes the CPU ground truth against the original FP16 B matrix.
3. **B quantization** — `quantize_b_q4k()` converts the FP16 B matrix to Q4_K_M blocks (144 bytes per 256-element column, 3.6× compression). This runs once before the tuning loop.
4. **Auto-tuning loop** — exhaustively tests combinations of kernel compile-time parameters (`MDIMC`, `NDIMC`, `MWG`, `NWG`, `KWG`, `SA`, `SB`, `VWM`, `VWN`). Each configuration is compiled via `opencl3`, run 4 times, and timed with OpenCL profiling events.
5. **`run_config()`** — compiles the kernel with `-D` flags, allocates GPU buffers (B as `Buffer::<u8>`), enqueues the kernel, and returns averaged elapsed time plus MSE against the FP16 CPU reference.
6. Results for every configuration are printed; the best (minimum time) is highlighted at the end. Expected MSE is ~0.003 due to 4-bit quantization error.

**`hgemm.cl`** implements the kernel using NVIDIA `wmma` (warp matrix multiply-accumulate) intrinsics for Tensor Core acceleration. The B matrix is always in Q4_K_M format (enabled via `-DQ4K_B` in `CL_BASE_ARGS`).

## Q4_K_M format

`-DQ4K_B` is always set in `CL_BASE_ARGS`. When active:

- B is passed to the kernel as `__global uchar*` instead of `__global half*`.
- Each column of B (256 elements, one per K-slice) is one `block_q4_K` (144 bytes):
  - bytes 0–1: `d` (f16 super-scale for sub-scales)
  - bytes 2–3: `dmin` (f16 super-scale for sub-mins)
  - bytes 4–15: `scales[12]` — eight (6-bit scale, 6-bit min) pairs, packed per `get_scale_min_k4`
  - bytes 16–143: `qs[128]` — 256 × 4-bit quants, two per byte
- Dequant formula per element: `d * sc * q4 - dmin * mn`
- `Q4K_B` forces `SB=1` (local B staging is mandatory for dequantization). The `GlobalToLocalB_q4k()` function dequantizes each element from its block into local half memory during the tile copy.
- `b_offset` in `HgemmBatched` is in bytes: `kSizeN * 144 * batch`.
- The kernel requires `cl_khr_fp16` (pragma at top of `hgemm.cl`).

## Tuning parameters

| Parameter | Meaning |
|-----------|---------|
| `MDIMC` / `NDIMC` | Warp-tile dimensions in M/N |
| `MWG` / `NWG` | Workgroup tile size in M/N (must be ≥ MDIMC/NDIMC) |
| `KWG` | K-loop unroll tile (16 or 32) |
| `SA` | Cache A tile in local memory (0/1) |
| `SB` | Cache B tile in local memory (0/1); forced to 1 by Q4K_B |
| `VWM` / `VWN` | Vector width for A/B local copies (unused for B in Q4K mode) |
