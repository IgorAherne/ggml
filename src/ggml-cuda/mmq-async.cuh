#pragma once

// Asynchronous (cp.async, sm_80+) tile pipeline for the Q4_K / Q5_K MMA kernels with I == J == 128.
//
// The default mul_mat_q_process_tile loads every K step synchronously: the x (weight) tile is read from global memory,
// unpacked and stored to shared memory, then the y half tiles, with __syncthreads around each part, so all warps wait
// for global memory four times per K step while the tensor cores are idle. Here the raw weight blocks and the y half
// tiles are copied with cp.async into double buffers one half step ahead, the 4/5 bit values are unpacked in registers
// right after ldmatrix, and the per sub-block scales are computed once per tile row into a float buffer, so the
// global memory latency is hidden behind the MMA work and there are two barriers per K step.
//
// The int8 values, the per sub-block scales (same half precision operations, same float conversion) and the d*C terms
// of the float accumulation are those of the default path (ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma's loop, same order and
// operations). The min terms (-dmin*m)*s are summed on the tensor cores instead (see ggml_cuda_mmq_async_vec_dot):
// their products are exact in f32 as before, only the rounding of their accumulation differs (results within f32
// rounding of the default path, not bit-identical).

#include "cp-async.cuh"
#include "unary.cuh"

#define GGML_CUDA_MMQ_ASYNC_I        128
#define GGML_CUDA_MMQ_ASYNC_J        128
#define GGML_CUDA_MMQ_ASYNC_NTHREADS 256
#define GGML_CUDA_MMQ_ASYNC_DM_ROW   8   // scales per tile row (one per 32 values of the 256 value K step)

// Shared memory row length of a raw weight block in ints (the block itself, 144 or 176 bytes). Consecutive rows then
// start 4 or 12 banks apart, so the 8 rows of an ldmatrix 8x8 matrix hit different banks.
static constexpr __host__ __device__ int ggml_cuda_mmq_async_raw_ints(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? int(sizeof(block_q4_K)/4) : (type == GGML_TYPE_Q5_K ? int(sizeof(block_q5_K)/4) : 0);
}

static constexpr __host__ __device__ bool ggml_cuda_mmq_async_config_ok(const ggml_cuda_mmq_config & c) {
    return (c.type == GGML_TYPE_Q4_K || c.type == GGML_TYPE_Q5_K) && c.I == GGML_CUDA_MMQ_ASYNC_I && c.J == GGML_CUDA_MMQ_ASYNC_J &&
        c.nthreads == GGML_CUDA_MMQ_ASYNC_NTHREADS && c.K_vram == MMQ_ITER_K && c.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;
}

// Scale buffer of one K step: d*sc as float [I][DM_ROW], then -dmin*m as half [I][DM_ROW].
static constexpr __host__ __device__ int ggml_cuda_mmq_async_dm_bytes() {
    return GGML_CUDA_MMQ_ASYNC_I*GGML_CUDA_MMQ_ASYNC_DM_ROW*(sizeof(float) + sizeof(half));
}

// ids[J] | raw x: 2 buffers | y: 2 half tiles | scales: 2 buffers | y half 0's (d, s) for the min term (J*4 half2)
static __host__ size_t ggml_cuda_mmq_async_nbytes_shared(const ggml_cuda_mmq_config & c) {
    return c.J*sizeof(int) + 2*c.I*ggml_cuda_mmq_async_raw_ints(c.type)*sizeof(int) + 2*c.J*MMQ_TILE_Y_K*sizeof(int) +
        2*ggml_cuda_mmq_async_dm_bytes() + c.J*4*sizeof(half2);
}

// Whether the kernel compiled for this device contains the asynchronous path for this config.
static __host__ bool ggml_cuda_mmq_async_available(const ggml_cuda_mmq_config & c, const int cc, const size_t smpbo) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE && c.use_mma_data_layout(cc) &&
        ggml_cuda_mmq_async_config_ok(c) && ggml_cuda_mmq_async_nbytes_shared(c) <= smpbo;
}

#if defined(CP_ASYNC_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)
#define GGML_CUDA_MMQ_ASYNC_DEVICE
#endif // defined(CP_ASYNC_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)

template <ggml_type type, int J, bool fallback>
static constexpr __device__ bool ggml_cuda_mmq_async_device_ok() {
#ifdef GGML_CUDA_MMQ_ASYNC_DEVICE
    return ggml_cuda_mmq_async_config_ok(ggml_cuda_mmq_get_config(type, J, fallback));
#else
    return false;
#endif // GGML_CUDA_MMQ_ASYNC_DEVICE
}

#ifdef GGML_CUDA_MMQ_ASYNC_DEVICE

static __device__ __forceinline__ void ggml_cuda_mmq_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int n_pending>
static __device__ __forceinline__ void ggml_cuda_mmq_async_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(n_pending));
}

// Raw weight blocks of one K step (one block per row) -> shared memory. Rows beyond i_max (fallback) copy row i_max,
// like the default loader, whose results for those rows are never written back. glu: tile rows I/2.. are read
// glu_shift rows further on (the up rows matching the tile's gate rows 0..I/2-1, see ggml_cuda_mmq_glu_args).
template <ggml_type type, bool fallback, bool glu = false>
static __device__ __forceinline__ void ggml_cuda_mmq_async_load_x(
        int * __restrict__ rx, const char * __restrict__ x, const int kbx0, const int i_max, const int stride,
        const int glu_shift = 0) {
    constexpr int I        = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int nthreads = GGML_CUDA_MMQ_ASYNC_NTHREADS;
    constexpr int row_ints = ggml_cuda_mmq_async_raw_ints(type);
    constexpr int nchunk   = row_ints/4; // 16 byte chunks per row
    constexpr size_t block_size = type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const unsigned int base = ggml_cuda_cvta_generic_to_shared(rx);
#pragma unroll
    for (int c0 = 0; c0 < I*nchunk; c0 += nthreads) {
        const int c = c0 + tid;
        if (c0 + nthreads <= I*nchunk || c < I*nchunk) {
            const int i  = c / nchunk;
            const int ch = c % nchunk;
            const int is = (fallback ? min(i, i_max) : i) + (glu && i >= I/2 ? glu_shift : 0);
            const char * src = x + (int64_t(kbx0) + int64_t(is)*stride)*block_size + 16*ch;
            cp_async_cg_16<0>(base + i*(4*row_ints) + 16*ch, src);
        }
    }
}

// One y half tile (J columns x 128 values as block_q8_1_mmq, contiguous in global memory) -> shared memory.
template <int J>
static __device__ __forceinline__ void ggml_cuda_mmq_async_load_y(int * __restrict__ tile_y, const int * __restrict__ by0) {
    constexpr int nthreads = GGML_CUDA_MMQ_ASYNC_NTHREADS;
    constexpr int n16      = J*MMQ_TILE_Y_K/4;

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const unsigned int base = ggml_cuda_cvta_generic_to_shared(tile_y);
#pragma unroll
    for (int c0 = 0; c0 < n16; c0 += nthreads) {
        const int c = c0 + tid;
        if (c0 + nthreads <= n16 || c < n16) {
            cp_async_cg_16<0>(base + 16*c, by0 + 4*c);
        }
    }
}

// The per row and sub-block (d*sc, -dmin*m) of ggml_cuda_mmq_load_tiles_q4_K / _q5_K (same half precision
// operations): d*sc as float (the exact float value ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma converts it to) to dmd,
// -dmin*m as half to dmm. Thread t writes row t/2, sub-blocks 4*(t%2) .. +3 (consecutive 16 / 8 byte stores).
template <ggml_type type>
static __device__ __forceinline__ void ggml_cuda_mmq_async_scales(
        float * __restrict__ dmd, half2 * __restrict__ dmm, const int * __restrict__ rx) {
    constexpr int I        = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int nthreads = GGML_CUDA_MMQ_ASYNC_NTHREADS;
    constexpr int row_ints = ggml_cuda_mmq_async_raw_ints(type);

    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
#pragma unroll
    for (int t0 = 0; t0 < 2*I; t0 += nthreads) {
        const int t   = t0 + tid;
        const int i   = t / 2;
        const int ksc = t % 2;
        const int4 hdr = *(const int4 *) (rx + i*row_ints); // dm, scales[0..11]

        // = unpack_scales_q45_K(scales, ksc + 0) and unpack_scales_q45_K(scales, ksc + 2), scales = {hdr.y, hdr.z, hdr.w}
        const int sc32 = ksc == 0 ? (((hdr.y >> 0) & 0x0F0F0F0F) | ((hdr.y >> 0) & 0x30303030))
                                  : (((hdr.w >> 0) & 0x0F0F0F0F) | ((hdr.y >> 2) & 0x30303030));
        const int  m32 = ksc == 0 ? (((hdr.z >> 0) & 0x0F0F0F0F) | ((hdr.z >> 0) & 0x30303030))
                                  : (((hdr.w >> 4) & 0x0F0F0F0F) | ((hdr.z >> 2) & 0x30303030));

        const uint8_t * sc8 = (const uint8_t *) &sc32;
        const uint8_t *  m8 = (const uint8_t *)  &m32;

        union { int i; half2 h; } u;
        u.i = hdr.x;
        const half2 dm = u.h * make_half2(1.0f, -1.0f);

        half2 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = dm*make_half2(sc8[k], m8[k]);
        }
        *(float4 *) (dmd + i*GGML_CUDA_MMQ_ASYNC_DM_ROW + 4*ksc) =
            make_float4(__low2float(v[0]), __low2float(v[1]), __low2float(v[2]), __low2float(v[3]));
        half2 m[2] = {__highs2half2(v[0], v[1]), __highs2half2(v[2], v[3])};
        *(int2 *) (dmm + i*(GGML_CUDA_MMQ_ASYNC_DM_ROW/2) + 2*ksc) = *(const int2 *) m;
    }
}

// ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma (NVIDIA path) for half h of the K step (x sub-blocks 4h..4h+3) against the y half
// tile y, with the x tiles built from the raw blocks: the A fragments are ldmatrix loads of the raw 4 bit data split
// into low and high nibbles (+ the 5th bit for Q5_K), i.e. the same ints the default loader stores to shared memory.
//
// The min term sum_b (-dmin*m)[i, b] * s[j, b] over the 8 sub-blocks b of a K step is not added per sub-block and
// element with FFMAs but once per K step and 16x8 output tile with an f16 MMA m16n8k8 (f32 accumulation) into the same
// sums: its f32 C fragment has tile_C's layout. A = the rows' -dmin*m (dmm: the half products themselves),
// B = the columns' block sums s (the .y halves of y's ds4). Each lane's fragments hold sub-blocks 2*(lane%4) and
// 2*(lane%4) + 1 (A: rows lane/4 and lane/4 + 8, B: column lane/4), which lie in half 0 for lane%4 < 2 and in half 1
// otherwise. The MMAs run in half 1: A comes from dmm (valid for the whole K step), B from y half 1 or, for the half 0
// sub-blocks, from ds_h0, where the warps copy those (d, s) pairs during half 0 (ty0 is refilled during half 1), so no
// state stays in registers across the barrier. The tensor time is 1/8 of the int8 MMAs' on GeForce cards (f16 MMA
// with f32 accumulation at a quarter of the int8 rate).

// D += A*B, m16n8k8, A/B f16 (row/col), D f32 (sm_80+ only, see GGML_CUDA_MMQ_ASYNC_DEVICE).
static __device__ __forceinline__ void ggml_cuda_mmq_async_mma_f16_f32(
        float * __restrict__ D, const int A0, const int A1, const int B) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A0), "r"(A1), "r"(B));
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_async_vec_dot(
        const int * __restrict__ rx, const float * __restrict__ dmd, const half2 * __restrict__ dmm,
        const int * __restrict__ y, float * __restrict__ sum, int2 * __restrict__ ds_h0, const int h) {
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
    constexpr int row_ints      = ggml_cuda_mmq_async_raw_ints(type);
    constexpr int qs_off        = type == GGML_TYPE_Q4_K ? 4 : 12; // ints: dm + scales (+ qh for Q5_K)
    constexpr int qh_off        = 4;
    constexpr int nwarps        = GGML_CUDA_MMQ_ASYNC_NTHREADS/WARP_SIZE;
    static_assert(tile_C::ne == 4 && tile_C::J == 8 && nwarps % ntx == 0, "min term fragments");

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
    // the (d, s) pairs of half 0 for this warp's column set: [column tile][column][pair 0/1 = sub-blocks 0, 1 / 2, 3]
    ds_h0 += (threadIdx.y % ntx) * (J/ntx*2);
    const int ty = threadIdx.x % 4; // the min term fragments' sub-blocks are 2*ty, 2*ty + 1

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float  dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1]; // d*sc of the rows get_i(2*l), sub-blocks 4h .. 4h + 3

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        const int * rxn = rx + (i0 + n*tile_A::I)*row_ints;
        tile_A H;
        if constexpr (type == GGML_TYPE_Q5_K) {
            load_ldmatrix(H, rxn + qh_off, row_ints); // qh ints 0..7 of the 16 rows
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            // raw qs ints 16h + 8p .. +7: the low nibbles are sub-block 4h + 2p, the high nibbles sub-block 4h + 2p + 1
            tile_A R;
            load_ldmatrix(R, rxn + qs_off + 16*h + 8*p, row_ints);
#pragma unroll
            for (int r = 0; r < tile_A::ne; ++r) {
                if constexpr (type == GGML_TYPE_Q4_K) {
                    A[n][2*p + 0].x[r] = (R.x[r] >> 0) & 0x0F0F0F0F;
                    A[n][2*p + 1].x[r] = (R.x[r] >> 4) & 0x0F0F0F0F;
                } else {
                    const int qh  = H.x[r];
                    const int sh  = 2*(2*h + p);
                    A[n][2*p + 0].x[r] = ((R.x[r] >> 0) & 0x0F0F0F0F) | (((qh >> (sh + 0)) << 4) & 0x10101010);
                    A[n][2*p + 1].x[r] = ((R.x[r] >> 4) & 0x0F0F0F0F) | (((qh >> (sh + 1)) << 4) & 0x10101010);
                }
            }
        }
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
            const float4 v = *(const float4 *) (dmd + i*GGML_CUDA_MMQ_ASYNC_DM_ROW + 4*h);
            dmA[n][l][0] = v.x;
            dmA[n][l][1] = v.y;
            dmA[n][l][2] = v.z;
            dmA[n][l][3] = v.w;
        }
    }

    // min term A (half 1): -dmin*m of rows lane/4 and lane/4 + 8, sub-blocks 2*ty, 2*ty + 1
    int min_A[ntx][2];
    if (h == 1) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
                min_A[n][l] = *(const int *) (dmm + i*(GGML_CUDA_MMQ_ASYNC_DM_ROW/2) + ty);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        const int jt = j0/(ntx*tile_C::J);
        // min term B: s of column j0 + lane/4, sub-blocks 2*ty, 2*ty + 1. Half 0: the (d, s) pairs of sub-blocks 0 .. 3
        // to ds_h0, column tile jt by the warp (threadIdx.y/ntx) == jt % (nwarps/ntx) of the column set.
        int min_B = 0;
        if (h == 0) {
            if (ty < 2 && jt % (nwarps/ntx) == threadIdx.y/ntx) {
                ds_h0[(jt*tile_C::J + threadIdx.x/4)*2 + ty] =
                    *(const int2 *) (y_dm + (j0 + threadIdx.x/4)*MMQ_TILE_Y_K + 2*ty);
            }
        } else {
            const int2 ds = ty < 2 ? ds_h0[(jt*tile_C::J + threadIdx.x/4)*2 + ty] :
                                     *(const int2 *) (y_dm + (j0 + threadIdx.x/4)*MMQ_TILE_Y_K + 2*(ty - 2));
            min_B = __byte_perm(ds.x, ds.y, 0x7632); // (ds.x.y, ds.y.y): the two s
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float    dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = __low2float(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1]*dB[l%2]*C.x[l];
                }
            }
        }

        if (h == 1) {
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                ggml_cuda_mmq_async_mma_f16_f32(sum + (j0/tile_C::J + n)*tile_C::ne, min_A[n][0], min_A[n][1], min_B);
            }
        }
    }
}

// The SwiGLU epilogue (ggml_cuda_mmq_async_process_tile with glu_qlayout >= 0): the tile's rows 0..I/2-1 are I/2
// consecutive gate rows (features feat0..) of a [gate | up] weight and rows I/2..I-1 the matching up rows (shift rows
// further on); instead of the F32 products, silu(gate)*up of the tile's I/2 features and J tokens is written to vq as
// the q8_1 data the quantization with the SWIGLU pre-op writes for the following quantized MUL_MAT (ntok rows).
struct ggml_cuda_mmq_glu_args {
    void * vq    = nullptr;
    int    shift = 0;
    int    feat0 = 0;
    int    tok0  = 0;
    int    ntok  = 0;
};

// quantize_mmq_q8_1's arithmetic and writes for the four values of columns 4*group .. 4*group+3 of row `row` (of
// nrows), as in norm.cu's norm_store_q8_1 (lanes 8k .. 8k+7 of a warp hold one 32-value block; every lane of the warp
// must call it); write == false: shuffle only.
template <int qlayout>
static __device__ __forceinline__ void ggml_cuda_mmq_async_store_q8_1(
        const float4 xi, void * __restrict__ vq, const int64_t group, const int row, const int nrows, const bool write) {
    constexpr mmq_q8_1_ds_layout ds_layout = (mmq_q8_1_ds_layout) qlayout;
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vq;
    const int64_t i0      = 4*group;
    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    if (!write) {
        return;
    }

    const int64_t ib = k_block*nrows + row;

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs/16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs/64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32]  = d;
        }
    }
}

// The SwiGLU epilogue: the accumulators (whole K) go to shared memory as [token][row], then each warp takes two tokens
// per pass, lanes 16t .. 16t+15 the 64 features of token t in groups of 4 (8 lanes per 32-value q8_1 block):
// silu(gate) * up as quantize_mmq_q8_1's SWIGLU pre-op computes it, then its quantization.
template <ggml_type type, int J, int qlayout>
static __device__ __forceinline__ void ggml_cuda_mmq_async_glu_q8_1(
        const float * __restrict__ sum, float * __restrict__ tbuf, const ggml_cuda_mmq_glu_args & glu, const int tile_y_max_j) {
    typedef tile<16, 8, int> tile_C;
    constexpr int I             = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int nwarps        = GGML_CUDA_MMQ_ASYNC_NTHREADS / WARP_SIZE;
    constexpr int pitch         = I + 4; // floats per token row: conflict-free stores, 16-byte aligned float4 loads
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, false);
    constexpr int ntx           = rows_per_warp/tile_C::I;
    static_assert(J % (2*nwarps) == 0 && I/2 == 4*16, "two tokens of 64 features per warp and pass");

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                tbuf[j*pitch + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
    __syncthreads();

    const int f = 4*(threadIdx.x % 16);
#pragma unroll 2
    for (int p = 0; p < J/(2*nwarps); ++p) {
        const int j = 2*(p*nwarps + threadIdx.y) + threadIdx.x / 16;
        const float4 g = *(const float4 *) (tbuf + j*pitch + f);
        const float4 u = *(const float4 *) (tbuf + j*pitch + I/2 + f);
        float4 xi;
        xi.x = __fmul_rn(ggml_cuda_op_silu_single(g.x), u.x);
        xi.y = __fmul_rn(ggml_cuda_op_silu_single(g.y), u.y);
        xi.z = __fmul_rn(ggml_cuda_op_silu_single(g.z), u.z);
        xi.w = __fmul_rn(ggml_cuda_op_silu_single(g.w), u.w);
        ggml_cuda_mmq_async_store_q8_1<qlayout>(xi, glu.vq, (glu.feat0 + f)/4, glu.tok0 + j, glu.ntok, j <= tile_y_max_j);
    }
}

// Drop-in for mul_mat_q_process_tile (same arguments and results). Pipeline per K step kb (buffers b = step parity):
//   vec_dot half 0 (x[kb] raw + scales, y half 0)        | in flight: y half 1 of kb, raw x of kb + 1
//   wait, barrier; copy y half 0 of kb + 1; scales of kb + 1 from its raw x
//   vec_dot half 1 (x[kb], y half 1)                     | in flight: y half 0 of kb + 1
//   wait, barrier; copy y half 1 of kb + 1 and the raw x of kb + 2 (into the buffer of kb)
// glu_qlayout >= 0: the SwiGLU epilogue (see ggml_cuda_mmq_glu_args) writing q8_1 of that layout instead of dst.
template <ggml_type type, int J, bool fallback, bool fixup, int glu_qlayout = -1>
static __device__ __forceinline__ void ggml_cuda_mmq_async_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop,
        const ggml_cuda_mmq_glu_args glu = ggml_cuda_mmq_glu_args()) {
    constexpr bool is_glu  = glu_qlayout >= 0;
    constexpr int nthreads = GGML_CUDA_MMQ_ASYNC_NTHREADS;
    constexpr int I        = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int qk       = ggml_cuda_type_traits<type>::qk;
    constexpr int row_ints = ggml_cuda_mmq_async_raw_ints(type);
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback>();
    constexpr int ne_block = QK8_1_MMQ;
    constexpr int sz       = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(ggml_cuda_mmq_get_K_vram(type, J, fallback) == qk, "one block per row and K step expected");
    static_assert(J*I / nthreads == 64, "unexpected tile shape");

    extern __shared__ int data_mul_mat_q[];
    int    * rx0  = data_mul_mat_q + J;
    int    * ty0  = rx0 + 2*I*row_ints;
    int    * ty1  = ty0 + J*MMQ_TILE_Y_K;
    char   * dm0   = (char *) (ty1 + J*MMQ_TILE_Y_K); // 2 scale buffers (ggml_cuda_mmq_async_dm_bytes each)
    int2   * ds_h0 = (int2 *) (dm0 + 2*ggml_cuda_mmq_async_dm_bytes());
    constexpr int rx_size  = I*row_ints;
    constexpr int dm_bytes = ggml_cuda_mmq_async_dm_bytes();
    // d*sc and -dmin*m of scale buffer b
    auto dmd = [&](const int b) { return (float *) (dm0 + b*dm_bytes); };
    auto dmm = [&](const int b) { return (half2 *) (dm0 + b*dm_bytes + I*GGML_CUDA_MMQ_ASYNC_DM_ROW*sizeof(float)); };

    float sum[J*I / nthreads] = {0.0f};

    // prologue: x and both y halves of kb0_start, then the raw x of kb0_start + 1
    {
        const int * by = y + ncols_y*((kb0_start*qk/ne_block)*sz);
        ggml_cuda_mmq_async_load_x<type, fallback, is_glu>(rx0, x, offset_x + kb0_start, tile_x_max_i, stride_row_x, glu.shift);
        ggml_cuda_mmq_async_load_y<J>(ty0, by);
        ggml_cuda_mmq_async_load_y<J>(ty1, by + ncols_y*sz);
        ggml_cuda_mmq_async_commit();
        if (kb0_start + 1 < kb0_stop) {
            ggml_cuda_mmq_async_load_x<type, fallback, is_glu>(rx0 + rx_size, x, offset_x + kb0_start + 1, tile_x_max_i, stride_row_x, glu.shift);
        }
        ggml_cuda_mmq_async_commit();
        ggml_cuda_mmq_async_wait<1>();
        __syncthreads();
        ggml_cuda_mmq_async_scales<type>(dmd(0), dmm(0), rx0);
        __syncthreads();
    }

    int b = 0;
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const bool has_next = kb0 + 1 < kb0_stop;
        const int * by_next = y + ncols_y*(((kb0 + 1)*qk/ne_block)*sz);

        ggml_cuda_mmq_async_vec_dot<type, J, fallback>(rx0 + b*rx_size, dmd(b), dmm(b), ty0, sum, ds_h0, 0);

        ggml_cuda_mmq_async_wait<0>();
        __syncthreads();

        if (has_next) {
            ggml_cuda_mmq_async_load_y<J>(ty0, by_next);
            ggml_cuda_mmq_async_commit();
            ggml_cuda_mmq_async_scales<type>(dmd(b ^ 1), dmm(b ^ 1), rx0 + (b ^ 1)*rx_size);
        }

        ggml_cuda_mmq_async_vec_dot<type, J, fallback>(rx0 + b*rx_size, dmd(b), dmm(b), ty1, sum, ds_h0, 1);

        ggml_cuda_mmq_async_wait<0>();
        __syncthreads();

        if (has_next) {
            ggml_cuda_mmq_async_load_y<J>(ty1, by_next + ncols_y*sz);
            if (kb0 + 2 < kb0_stop) {
                ggml_cuda_mmq_async_load_x<type, fallback, is_glu>(rx0 + b*rx_size, x, offset_x + kb0 + 2, tile_x_max_i, stride_row_x, glu.shift);
            }
            ggml_cuda_mmq_async_commit();
        }
        b ^= 1;
    }

    if constexpr (is_glu) {
        static_assert(!fixup && !fallback, "the SwiGLU epilogue needs whole tiles");
        // the loop's last barrier came after the last shared memory reads: the tile buffers can be reused
        ggml_cuda_mmq_async_glu_q8_1<type, J, glu_qlayout>(sum, (float *) rx0, glu, tile_y_max_j);
        GGML_UNUSED(write_back);
    } else if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

#endif // GGML_CUDA_MMQ_ASYNC_DEVICE

// [gate | up] MUL_MAT -> SWIGLU -> quantized MUL_MAT: the first product with the SwiGLU epilogue, one CUDA block per
// output tile (I/2 features x J tokens, whole K: the same sums as the default launch in tiling mode), visited in groups
// of tile_group row tiles like mul_mat_q's grouped order. nfeat = rows/2, ntx = token tiles, nty = nfeat/(I/2).
template <ggml_type type, int qlayout>
static __global__ void __launch_bounds__(GGML_CUDA_MMQ_ASYNC_NTHREADS, 1) mul_mat_q_glu_q8_1(
        const char * __restrict__ x, const int * __restrict__ y, void * __restrict__ vq,
        const int blocks_per_ne00, const int stride_row_x, const int ncols_y, const int nfeat,
        const int ntx, const int nty, const int tile_group) {
#ifdef GGML_CUDA_MMQ_ASYNC_DEVICE
    constexpr int J = GGML_CUDA_MMQ_ASYNC_J;
    constexpr int I = GGML_CUDA_MMQ_ASYNC_I;

    extern __shared__ int ids_dst_shared[]; // the layout of process_tile starts after J ids (unused here)

    int bidx = blockIdx.x;
    if (tile_group > 0) {
        const int group_size = tile_group*ntx;
        const int group      = bidx / group_size;
        const int rows_group = min(tile_group, nty - group*tile_group);
        const int l          = bidx - group*group_size;
        bidx = (group*tile_group + l % rows_group)*ntx + l / rows_group;
    }
    const int it = bidx / ntx;
    const int jt = bidx - it*ntx;

    ggml_cuda_mmq_glu_args glu;
    glu.vq    = vq;
    glu.shift = nfeat - I/2;
    glu.feat0 = it*(I/2);
    glu.tok0  = jt*J;
    glu.ntok  = ncols_y;

    const int offset_x = it*(I/2)*stride_row_x;
    const int offset_y = jt*J*(sizeof(block_q8_1_mmq)/sizeof(int));
    ggml_cuda_mmq_async_process_tile<type, J, false, false, qlayout>
        (x, offset_x, y + offset_y, ids_dst_shared, nullptr, nullptr, nullptr,
         stride_row_x, ncols_y, 0, I - 1, ncols_y - jt*J - 1, 0, blocks_per_ne00, glu);
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(vq); GGML_UNUSED(blocks_per_ne00); GGML_UNUSED(stride_row_x);
    GGML_UNUSED(ncols_y); GGML_UNUSED(nfeat); GGML_UNUSED(ntx); GGML_UNUSED(nty); GGML_UNUSED(tile_group);
    NO_DEVICE_CODE;
#endif // GGML_CUDA_MMQ_ASYNC_DEVICE
}
