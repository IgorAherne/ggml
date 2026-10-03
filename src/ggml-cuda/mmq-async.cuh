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
// The results are bit-identical to the default path: the same int8 values reach the same mma instructions, every
// per sub-block scale is computed with the same half precision operations and converted to float exactly as before,
// and the float accumulation is the unchanged loop of ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma (same order, same operations).

#include "cp-async.cuh"

#define GGML_CUDA_MMQ_ASYNC_I        128
#define GGML_CUDA_MMQ_ASYNC_J        128
#define GGML_CUDA_MMQ_ASYNC_NTHREADS 256
#define GGML_CUDA_MMQ_ASYNC_DM_ROW   8   // float2 scales per tile row (one per 32 values of the 256 value K step)

// Shared memory row length of a raw weight block in ints (the block itself, 144 or 176 bytes). Consecutive rows then
// start 4 or 12 banks apart, so the 8 rows of an ldmatrix 8x8 matrix hit different banks.
static constexpr __host__ __device__ int ggml_cuda_mmq_async_raw_ints(const ggml_type type) {
    return type == GGML_TYPE_Q4_K ? int(sizeof(block_q4_K)/4) : (type == GGML_TYPE_Q5_K ? int(sizeof(block_q5_K)/4) : 0);
}

static constexpr __host__ __device__ bool ggml_cuda_mmq_async_config_ok(const ggml_cuda_mmq_config & c) {
    return (c.type == GGML_TYPE_Q4_K || c.type == GGML_TYPE_Q5_K) && c.I == GGML_CUDA_MMQ_ASYNC_I && c.J == GGML_CUDA_MMQ_ASYNC_J &&
        c.nthreads == GGML_CUDA_MMQ_ASYNC_NTHREADS && c.K_vram == MMQ_ITER_K && c.sram_layout == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;
}

// ids[J] | raw x: 2 buffers | y: 2 half tiles | scales: 2 buffers
static __host__ size_t ggml_cuda_mmq_async_nbytes_shared(const ggml_cuda_mmq_config & c) {
    return c.J*sizeof(int) + 2*c.I*ggml_cuda_mmq_async_raw_ints(c.type)*sizeof(int) + 2*c.J*MMQ_TILE_Y_K*sizeof(int) +
        2*c.I*GGML_CUDA_MMQ_ASYNC_DM_ROW*sizeof(float2);
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
// like the default loader, whose results for those rows are never written back.
template <ggml_type type, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_async_load_x(
        int * __restrict__ rx, const char * __restrict__ x, const int kbx0, const int i_max, const int stride) {
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
            const int is = fallback ? min(i, i_max) : i;
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

// Position of the float2 scale of sub-block sb in tile row i: 4 chunks of 16 bytes (2 sub-blocks each) per row,
// XOR swizzled so that the 8 rows read by one ldmatrix-style access pattern use 8 different groups of 4 banks.
static __device__ __forceinline__ int ggml_cuda_mmq_async_dm_index(const int i, const int sb) {
    return i*GGML_CUDA_MMQ_ASYNC_DM_ROW + 2*((sb/2) ^ ((i >> 1) & 3)) + sb % 2;
}

// The per row and sub-block (d*sc, -dmin*m) of ggml_cuda_mmq_load_tiles_q4_K / _q5_K (same half precision
// operations), stored as float2 (the exact float values ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma converts them to).
template <ggml_type type>
static __device__ __forceinline__ void ggml_cuda_mmq_async_scales(float2 * __restrict__ dmf, const int * __restrict__ rx) {
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

#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const float2 v0 = __half22float2(dm*make_half2(sc8[2*q + 0], m8[2*q + 0]));
            const float2 v1 = __half22float2(dm*make_half2(sc8[2*q + 1], m8[2*q + 1]));
            *(float4 *) (dmf + ggml_cuda_mmq_async_dm_index(i, 4*ksc + 2*q)) = make_float4(v0.x, v0.y, v1.x, v1.y);
        }
    }
}

// ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma (NVIDIA path) for half h of the K step (x sub-blocks 4h..4h+3) against the y half
// tile y, with the x tiles built from the raw blocks: the A fragments are ldmatrix loads of the raw 4 bit data split
// into low and high nibbles (+ the 5th bit for Q5_K), i.e. the same ints the default loader stores to shared memory.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_async_vec_dot(
        const int * __restrict__ rx, const float2 * __restrict__ dmf, const int * __restrict__ y, float * __restrict__ sum, const int h) {
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
    constexpr int row_ints      = ggml_cuda_mmq_async_raw_ints(type);
    constexpr int qs_off        = type == GGML_TYPE_Q4_K ? 4 : 12; // ints: dm + scales (+ qh for Q5_K)
    constexpr int qh_off        = 4;

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

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
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float4 v = *(const float4 *) (dmf + ggml_cuda_mmq_async_dm_index(i, 4*h + 2*q));
                dmA[n][l][2*q + 0] = make_float2(v.x, v.y);
                dmA[n][l][2*q + 1] = make_float2(v.z, v.w);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

// Drop-in for mul_mat_q_process_tile (same arguments and results). Pipeline per K step kb (buffers b = step parity):
//   vec_dot half 0 (x[kb] raw + scales, y half 0)        | in flight: y half 1 of kb, raw x of kb + 1
//   wait, barrier; copy y half 0 of kb + 1; scales of kb + 1 from its raw x
//   vec_dot half 1 (x[kb], y half 1)                     | in flight: y half 0 of kb + 1
//   wait, barrier; copy y half 1 of kb + 1 and the raw x of kb + 2 (into the buffer of kb)
template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void ggml_cuda_mmq_async_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
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
    float2 * dmf0 = (float2 *) (ty1 + J*MMQ_TILE_Y_K);
    constexpr int rx_size  = I*row_ints;
    constexpr int dmf_size = I*GGML_CUDA_MMQ_ASYNC_DM_ROW;

    float sum[J*I / nthreads] = {0.0f};

    // prologue: x and both y halves of kb0_start, then the raw x of kb0_start + 1
    {
        const int * by = y + ncols_y*((kb0_start*qk/ne_block)*sz);
        ggml_cuda_mmq_async_load_x<type, fallback>(rx0, x, offset_x + kb0_start, tile_x_max_i, stride_row_x);
        ggml_cuda_mmq_async_load_y<J>(ty0, by);
        ggml_cuda_mmq_async_load_y<J>(ty1, by + ncols_y*sz);
        ggml_cuda_mmq_async_commit();
        if (kb0_start + 1 < kb0_stop) {
            ggml_cuda_mmq_async_load_x<type, fallback>(rx0 + rx_size, x, offset_x + kb0_start + 1, tile_x_max_i, stride_row_x);
        }
        ggml_cuda_mmq_async_commit();
        ggml_cuda_mmq_async_wait<1>();
        __syncthreads();
        ggml_cuda_mmq_async_scales<type>(dmf0, rx0);
        __syncthreads();
    }

    int b = 0;
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const bool has_next = kb0 + 1 < kb0_stop;
        const int * by_next = y + ncols_y*(((kb0 + 1)*qk/ne_block)*sz);

        ggml_cuda_mmq_async_vec_dot<type, J, fallback>(rx0 + b*rx_size, dmf0 + b*dmf_size, ty0, sum, 0);

        ggml_cuda_mmq_async_wait<0>();
        __syncthreads();

        if (has_next) {
            ggml_cuda_mmq_async_load_y<J>(ty0, by_next);
            ggml_cuda_mmq_async_commit();
            ggml_cuda_mmq_async_scales<type>(dmf0 + (b ^ 1)*dmf_size, rx0 + (b ^ 1)*rx_size);
        }

        ggml_cuda_mmq_async_vec_dot<type, J, fallback>(rx0 + b*rx_size, dmf0 + b*dmf_size, ty1, sum, 1);

        ggml_cuda_mmq_async_wait<0>();
        __syncthreads();

        if (has_next) {
            ggml_cuda_mmq_async_load_y<J>(ty1, by_next + ncols_y*sz);
            if (kb0 + 2 < kb0_stop) {
                ggml_cuda_mmq_async_load_x<type, fallback>(rx0 + b*rx_size, x, offset_x + kb0 + 2, tile_x_max_i, stride_row_x);
            }
            ggml_cuda_mmq_async_commit();
        }
        b ^= 1;
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

#endif // GGML_CUDA_MMQ_ASYNC_DEVICE
