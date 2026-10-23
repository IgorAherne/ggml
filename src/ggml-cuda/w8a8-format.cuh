#pragma once

// The W8A8 path's tile constants and the layout of its quantized activations (q_x int8 rows, the per-32 sub-scales sy
// in the token tile's order, d_t), shared by its kernels (w8a8.cuh) and the fused norm writer (norm.cu).

#include <cuda_runtime.h>
#include <cstdint>

#ifndef W8A8_BM
#define W8A8_BM       128   // tokens per block tile (128: 4 warps, two blocks per SM; 256: 8 warps, one block per SM)
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
