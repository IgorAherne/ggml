#pragma once

// The W8A8 path's tile constants and the layout of its quantized activations (q_x int8 rows, the per-32 sub-scales sy
// in the token tile's order, d_t), shared by its kernels (w8a8.cuh) and the fused norm writer (norm.cu).

#include <cuda_runtime.h>
#include <cstdint>

// The activations' sub-scale range: sy in 1 .. W8A8_SY_MAX (5 bits), d_g = d_t * sy / W8A8_SY_MAX. The GEMM adds
// sy * (a 32-value int8 dot product) into one int32 per output, so |acc| <= W8A8_SY_MAX * 127^2 * K: with 31,
// 2,047,995,904 < 2^31 for K = 4096 (the attention's and the [gate|up] products: no wrap possible); for K = 12288 (the
// down product) the bound is 3x 2^31; on the real activation dumps the largest accumulator stays 41x below 2^31 (the
// 2026-10-23 review's measurement with the model's weights). (8-bit
// sub-scales, 255, left only a 3.2x margin on real to_out activations: the residual stream's massive channels.)
#define W8A8_SY_MAX   31

#ifndef W8A8_BM
// tokens per block tile (128: 4 warps, two blocks per SM - 255 registers x 128 threads = half the 64K register file,
// 2 x (49,920 + 1,024 reserved) bytes of shared memory: cudaOccupancyMaxActiveBlocksPerMultiprocessor gives 2 on the RTX
// 3060, w8a8-tools/occ.cu; 256: 8 warps, one block per SM)
#define W8A8_BM       128
#endif
#define W8A8_BN       128   // output channels per block tile
#define W8A8_BK       64    // K values per pipeline stage (2 sub-scale groups)
#ifndef W8A8_WM
#define W8A8_WM       64    // tokens per warp tile (64: 128 accumulators per thread; 32: 64, twice the warps)
#endif
#define W8A8_MF       (W8A8_WM/16)        // m16 fragments per warp
#define W8A8_WARPS_M  (W8A8_BM/W8A8_WM)
#define W8A8_NTHREADS (W8A8_WARPS_M*64)   // W8A8_WARPS_M (tokens) x 2 (channels) warps, W8A8_WM x 64 outputs each
#define W8A8_MINBLOCKS ((W8A8_WM == 64 ? 256 : 512)/W8A8_NTHREADS > 1 ? (W8A8_WM == 64 ? 256 : 512)/W8A8_NTHREADS : 1)

// position of row r (0..BM-1) of a token tile in its sy block: r = wm*WM + mf*16 + h*8 + gid  ->
// wm*WM + gid*2*MF + mf*2 + h (a thread's 2*MF values of one group are consecutive bytes)
static __host__ __device__ __forceinline__ int w8a8_sy_pos(const int r) {
    const int wm = r / W8A8_WM, mf = (r % W8A8_WM) >> 4, h = (r >> 3) & 1, gid = r & 7;
    return wm*W8A8_WM + gid*2*W8A8_MF + mf*2 + h;
}

static __host__ __device__ __forceinline__ int64_t w8a8_pad_tokens(const int64_t T) {
    return (T + W8A8_BM - 1) / W8A8_BM * W8A8_BM;
}

// the quantized activations of T rows of K values: q_x [T_pad][K], sy [T_pad/BM][K/32][BM] (w8a8_sy_pos order), d_t [T_pad]
struct w8a8_act_ptrs {
    int8_t *  qx;
    uint8_t * sy;
    float *   dt;
};
