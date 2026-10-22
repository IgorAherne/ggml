#pragma once

// Ring tile pipeline for the Q4_K / Q5_K MMA kernels with I == J == 128 (sm_80+; GGML_CUDA_MMQ_PIPE=ring) on activations
// in the q8_1 layout MMQ_Q8_1_DS_LAYOUT_DSY (per 128 values: one f16 scale d128, an 8-bit sub-scale sy_j per 32 values,
// the per-32 float sums s_j; see block_q8_1_mmq and ggml_cuda_q8_1_dsy_quantize).
//
// Arithmetic. Weight row r, sub-block j (32 values) of a Q4_K/Q5_K block: w = d*sc_j*q - dmin*m_j (6-bit sc_j, m_j, f16
// d, dmin). Activation column c: x ~ d128*sy_j/MAX * y (int8 y). Per 128-value half of the K step and output (r, c):
//   acc = sum_j (sc_j[r] * sy_j[c]) * C_j[r][c]       int32, exact: C_j = the m16n8k32 int8 MMA of (q - centre) and y,
//                                                     one sub-block per MMA; sc*sy as one IDP4A, then one IMAD
//   sum += (d[r]/MAX * d128[c]) * float(acc)          once per G sub-blocks (I2F, FMUL, FFMA)
// and once per K step, as an f16 m16n8k8 MMA with f32 accumulation over the 8 sub-blocks (as in mmq-async.cuh):
//   sum += sum_j m'_j[r] * s_j[c],  m'_j = centre*d*sc_j - dmin*m_j (rounded once to f16; -dmin*m_j as the half product
//   of the other paths when centre == 0).
// The centred weights also make the result closer to the exact product than the uncentred per-32 path: the part of
// the weights the f32 sums s_j carry exactly grows from -dmin*m_j to the sub-block offset centre*d*sc_j - dmin*m_j.
// Overflow: |C_j| <= 32*max|q - centre|*127, sc*sy <= 63*MAX, so |acc| <= G*63*MAX*32*max|q - centre|*127 < 2^31 with
// MAX = 255: Q4_K centre 8, G 4 (2.089e9); Q5_K centre 16, G 2 (2.089e9). (MAX = 127: Q4_K uncentred, Q5_K G 4.)
//
// Pipeline. The K loop runs in stages of 128 values (one half of a 256-value block). The y half tiles go through a ring
// of 3 buffers and are issued two stages ahead; the raw weight blocks are double buffered per K step, the block of
// K step kb + 1 issued with y(kb + 1, half 0) during half 0 of kb. One cp.async group and one barrier per stage. Each
// warp reads its rows' scales and mins from the raw 12-byte scales field (no scale buffers in shared memory); the
// min term's s values of half 0 stay in registers until half 1 (the ring reuses half 0's buffer during half 1).
// Shared memory: the s pairs of half 0 (J*2 ints; over mul_mat_q's J ids, which the ring never reads: no MoE) + 2 raw
// x blocks + 3 y half tiles = 93,184 bytes (Q4_K), 101,376 bytes (Q5_K: the sm_86 / sm_89 / sm_120 limit).
//
// Warp tiles: RN row tiles of 16 rows (rows (w/CG)*16*RN ..) x 128/CG columns in 8-column tiles (columns (w%CG)*8 +
// 8*CG*jt + 0..7, jt = 0 .. 16/CG - 1), (128/(16*RN))*CG warps: RN 1, CG 1 -> 16 x 128, 8 warps (256 threads);
// RN 1, CG 2 -> 16 x 64, 16 warps (512 threads); RN 2, CG 2 -> 32 x 64, 8 warps (the asynchronous path's tiles).

#define GGML_CUDA_MMQ_RING_STAGES 3
#ifndef GGML_CUDA_MMQ_RING_RN
#define GGML_CUDA_MMQ_RING_RN     1
#endif // GGML_CUDA_MMQ_RING_RN
#ifndef GGML_CUDA_MMQ_RING_CG
#define GGML_CUDA_MMQ_RING_CG     1
#endif // GGML_CUDA_MMQ_RING_CG
#define GGML_CUDA_MMQ_RING_NTHREADS ((GGML_CUDA_MMQ_ASYNC_I/(16*GGML_CUDA_MMQ_RING_RN))*GGML_CUDA_MMQ_RING_CG*32)

template <ggml_type type>
static constexpr __host__ __device__ int ggml_cuda_mmq_ring_centre() {
    return type == GGML_TYPE_Q5_K ? 16 : (GGML_CUDA_MMQ_DSY_MAX > 127 ? 8 : 0);
}

template <ggml_type type>
static constexpr __host__ __device__ int ggml_cuda_mmq_ring_G() {
    return type == GGML_TYPE_Q5_K && GGML_CUDA_MMQ_DSY_MAX > 127 ? 2 : 4;
}

template <ggml_type type>
static constexpr __host__ __device__ long long ggml_cuda_mmq_ring_acc_bound() {
    constexpr int qmax = type == GGML_TYPE_Q5_K ? 31 : 15;
    constexpr int c    = ggml_cuda_mmq_ring_centre<type>();
    constexpr int qabs = c == 0 ? qmax : (c > qmax - c ? c : qmax - c);
    return (long long) ggml_cuda_mmq_ring_G<type>() * 63 * GGML_CUDA_MMQ_DSY_MAX * 32 * qabs * 127;
}
static_assert(ggml_cuda_mmq_ring_acc_bound<GGML_TYPE_Q4_K>() < 2147483648LL, "Q4_K ring accumulator may overflow");
static_assert(ggml_cuda_mmq_ring_acc_bound<GGML_TYPE_Q5_K>() < 2147483648LL, "Q5_K ring accumulator may overflow");

// the s pairs of half 0 (J*2 ints, for the min term in half 1; the first J are mul_mat_q's ids) | raw x: 2 buffers |
// y: 3 half tiles
static __host__ size_t ggml_cuda_mmq_ring_nbytes_shared(const ggml_cuda_mmq_config & c) {
    return c.J*2*sizeof(int) + 2*c.I*ggml_cuda_mmq_async_raw_ints(c.type)*sizeof(int) +
        GGML_CUDA_MMQ_RING_STAGES*c.J*MMQ_TILE_Y_K*sizeof(int);
}

// Whether the kernel compiled for this device contains the ring pipeline for this config (as the asynchronous one).
static __host__ bool ggml_cuda_mmq_ring_available(const ggml_cuda_mmq_config & c, const int cc, const size_t smpbo) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE && c.use_mma_data_layout(cc) &&
        ggml_cuda_mmq_async_config_ok(c) && ggml_cuda_mmq_ring_nbytes_shared(c) <= smpbo;
}

#ifdef GGML_CUDA_MMQ_ASYNC_DEVICE

// Raw weight blocks of one K step -> shared memory, as ggml_cuda_mmq_async_load_x with the ring's thread count.
template <ggml_type type, bool fallback, bool glu = false>
static __device__ __forceinline__ void ggml_cuda_mmq_ring_load_x(
        int * __restrict__ rx, const char * __restrict__ x, const int kbx0, const int i_max, const int stride,
        const int glu_shift = 0) {
    constexpr int I        = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int nthreads = GGML_CUDA_MMQ_RING_NTHREADS;
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

// One y half tile (J columns x 128 values, contiguous in global memory) -> shared memory.
template <int J>
static __device__ __forceinline__ void ggml_cuda_mmq_ring_load_y(int * __restrict__ tile_y, const int * __restrict__ by0) {
    constexpr int nthreads = GGML_CUDA_MMQ_RING_NTHREADS;
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

// The half H (sub-blocks 4H .. 4H + 3 of the K step) of the warp tile (RN x njt fragments, see above);
// sum[(jt*RN + n)*4 + l] (C fragment layout).
// rx: the raw blocks of the K step; y: the half tile; ds_h0 (H == 1): the s pairs of half 0 per column ([J][2] ints).
template <ggml_type type, int J, int H>
static __device__ __forceinline__ void ggml_cuda_mmq_ring_vec_dot(
        const int * __restrict__ rx, const int * __restrict__ y, float * __restrict__ sum, const int * __restrict__ ds_h0) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int ntx      = GGML_CUDA_MMQ_RING_RN;  // row tiles per warp
    constexpr int cg       = GGML_CUDA_MMQ_RING_CG;  // column groups
    constexpr int row_ints = ggml_cuda_mmq_async_raw_ints(type);
    constexpr int qs_off   = type == GGML_TYPE_Q4_K ? 4 : 12; // ints: dm + scales (+ qh for Q5_K)
    constexpr int qh_off   = 4;
    constexpr int centre   = ggml_cuda_mmq_ring_centre<type>();
    constexpr int G        = ggml_cuda_mmq_ring_G<type>();
    constexpr int nsb      = 4;                    // sub-blocks per half
    constexpr int njt      = J/(cg*tile_C::J);     // column tiles per warp
    static_assert(njt <= 16 && tile_C::ne == 4 && nsb % G == 0, "unexpected warp tile");

    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int t    = lane % 4;
    const int i0   = (threadIdx.y/cg)*(ntx*tile_A::I);
    y     += (threadIdx.y % cg) * (tile_C::J*MMQ_TILE_Y_K);
    ds_h0 += (threadIdx.y % cg) * (tile_C::J*2);
    // the ldmatrix row addresses of the B fragments (shared state space): lanes 8m .. 8m+7 address matrix m = column
    // lane%8, ints 4 + 8*(m/2) + 4*(m%2) of its block (+ 16 ints for the sub-blocks 2, 3; + 8*CG columns per column tile)
    const unsigned int y_s = ggml_cuda_cvta_generic_to_shared((void *) y) +
        sizeof(int)*((lane % 8)*MMQ_TILE_Y_K + 4 + 8*(lane/16) + 4*((lane/8) % 2));

    // A: the 4 sub-blocks of the half per row tile, from ldmatrix loads of the raw 4-bit data (as mmq-async.cuh)
    tile_A A[ntx][nsb];
    // per lane row (n, r: rows g, g + 8 of row tile n): sc_j of the half as byte j of an otherwise zero int (one IDP4A
    // with the sy bytes of a column gives sc_j * sy_j), d / MAX
    int   scb[ntx][2][nsb];
    float dxs[ntx][2];
    // min term A (H == 1): m'_j of rows g, g + 8 for the sub-blocks 2t, 2t + 1 of the K step, as half2
    int   min_A[ntx][2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
        const int * rxn = rx + (i0 + n*tile_A::I)*row_ints;
        tile_A Hq;
        if constexpr (type == GGML_TYPE_Q5_K) {
            load_ldmatrix(Hq, rxn + qh_off, row_ints); // qh ints 0..7 of the 16 rows
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            // raw qs ints 16H + 8p .. +7: the low nibbles are sub-block 4H + 2p, the high nibbles sub-block 4H + 2p + 1
            tile_A R;
            load_ldmatrix(R, rxn + qs_off + 16*H + 8*p, row_ints);
#pragma unroll
            for (int l = 0; l < tile_A::ne; ++l) {
                int lo = (R.x[l] >> 0) & 0x0F0F0F0F;
                int hi = (R.x[l] >> 4) & 0x0F0F0F0F;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    const int qh = Hq.x[l];
                    const int sh = 2*(2*H + p);
                    lo |= ((qh >> (sh + 0)) << 4) & 0x10101010;
                    hi |= ((qh >> (sh + 1)) << 4) & 0x10101010;
                }
                if constexpr (centre != 0) {
                    // per byte v -> v - centre as int8: (v + 128 - centre) has no carries, then flip bit 7
                    lo = (lo + 0x01010101*(128 - centre)) ^ 0x80808080;
                    hi = (hi + 0x01010101*(128 - centre)) ^ 0x80808080;
                }
                A[n][2*p + 0].x[l] = lo;
                A[n][2*p + 1].x[l] = hi;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int4 hdr = *(const int4 *) (rx + (i0 + n*tile_A::I + 8*r + g)*row_ints); // dm, scales[0..11]
            union { int i; half2 h; } dm;
            dm.i = hdr.x;
            // = unpack_scales_q45_K: sub-blocks 0..3 (H = 0) or 4..7 (H = 1)
            const int sc32 = H == 0 ? (hdr.y & 0x3F3F3F3F) : ((hdr.w & 0x0F0F0F0F) | ((hdr.y >> 2) & 0x30303030));
#pragma unroll
            for (int j = 0; j < nsb; ++j) {
                scb[n][r][j] = sc32 & (0xFF << (8*j));
            }
            dxs[n][r] = __low2float(dm.h) * (1.0f/GGML_CUDA_MMQ_DSY_MAX);

            if constexpr (H == 1) {
                // sub-blocks 2t, 2t + 1 of the K step: bytes 2*(t%2), +1 of the low (t < 2) or high scales / mins
                const int m32 = t < 2 ? (hdr.z & 0x3F3F3F3F) : (((hdr.w >> 4) & 0x0F0F0F0F) | ((hdr.z >> 2) & 0x30303030));
                const int sh  = 16*(t & 1);
                const int ma  = (m32 >> (sh + 0)) & 0x3F;
                const int mb  = (m32 >> (sh + 8)) & 0x3F;
                union { half2 h; int i; } mm;
                if constexpr (centre != 0) {
                    const int s32 = t < 2 ? (hdr.y & 0x3F3F3F3F) : ((hdr.w & 0x0F0F0F0F) | ((hdr.y >> 2) & 0x30303030));
                    const int sa  = (s32 >> (sh + 0)) & 0x3F;
                    const int sb  = (s32 >> (sh + 8)) & 0x3F;
                    const float d    = __low2float(dm.h);
                    const float dmin = __high2float(dm.h);
                    mm.h = __floats2half2_rn(d*float(centre*sa) - dmin*float(ma), d*float(centre*sb) - dmin*float(mb));
                } else {
                    const half nd = -__high2half(dm.h);
                    mm.h = __hmul2(make_half2(nd, nd), make_half2(__int2half_rn(ma), __int2half_rn(mb)));
                }
                min_A[n][r] = mm.i;
            }
        }
    }

#pragma unroll
    for (int jt = 0; jt < njt; ++jt) {
        const int * yt = y + jt*(cg*tile_C::J)*MMQ_TILE_Y_K; // the 8 blocks of the column tile

        // min term B (H == 1): (s_2t, s_2t+1) of column g: header int t%2 of the block of half t/2 (half 0: ds_h0)
        int min_B = 0;
        if constexpr (H == 1) {
            min_B = t < 2 ? ds_h0[(jt*cg*tile_C::J + g)*2 + t] : yt[g*MMQ_TILE_Y_K + t - 2];
        }

        // B: two ldmatrix.x4 (sub-blocks 0, 1 and 2, 3): registers = B[2q].x[0], B[2q].x[1], B[2q+1].x[0], B[2q+1].x[1]
        tile_B B[nsb];
#pragma unroll
        for (int q = 0; q < nsb/2; ++q) {
            const unsigned int src = y_s + sizeof(int)*(jt*(cg*tile_C::J)*MMQ_TILE_Y_K + 16*q);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(B[2*q].x[0]), "=r"(B[2*q].x[1]), "=r"(B[2*q + 1].x[0]), "=r"(B[2*q + 1].x[1])
                : "r"(src));
        }

        // the columns 2t, 2t + 1 of the tile: d128 (f32) and the sy bytes
        float d128[2];
        int   sy[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int2 v = *(const int2 *) (yt + (2*t + c)*MMQ_TILE_Y_K + 2);
            d128[c] = __int_as_float(v.x);
            sy[c]   = v.y;
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            int acc[tile_C::ne];
#pragma unroll
            for (int j = 0; j < nsb; ++j) {
                tile_C C;
                mma(C, A[n][j], B[j]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int m = (int) __dp4a((unsigned int) scb[n][l/2][j], (unsigned int) sy[l%2], 0u); // sc_j * sy_j, unsigned
                    acc[l] = j % G == 0 ? m*C.x[l] : acc[l] + m*C.x[l];
                }
                if (j % G == G - 1) {
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        sum[(jt*ntx + n)*tile_C::ne + l] += (dxs[n][l/2]*d128[l%2]) * float(acc[l]);
                    }
                }
            }
        }

        if constexpr (H == 1) {
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                ggml_cuda_mmq_async_mma_f16_f32(sum + (jt*ntx + n)*tile_C::ne, min_A[n][0], min_A[n][1], min_B);
            }
        }
    }
}

// The warp tile's sums to dst (column-major, stride): as ggml_cuda_mmq_write_back_mma for the ring's warp tiles, with
// the identity for the ids (the ring runs MUL_MATs without ids only; their shared memory holds ds_h0).
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_ring_write_back(
        const float * __restrict__ sum, float * __restrict__ dst, const int stride, const int i_max, const int j_max) {
    typedef tile<16, 8, int> tile_C;
    constexpr int ntx = GGML_CUDA_MMQ_RING_RN;
    constexpr int cg  = GGML_CUDA_MMQ_RING_CG;
    const int i0 = (threadIdx.y/cg)*(ntx*tile_C::I);
#pragma unroll
    for (int jt = 0; jt < J/(cg*tile_C::J); ++jt) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = (threadIdx.y % cg)*tile_C::J + jt*(cg*tile_C::J) + tile_C::get_j(l);
                if (j > j_max) {
                    continue;
                }
                const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                if (fallback && i > i_max) {
                    continue;
                }
                dst[j*stride + i] = sum[(jt*ntx + n)*tile_C::ne + l];
            }
        }
    }
}

// Drop-in for mul_mat_q_process_tile (same arguments; y in the DSY layout; GGML_CUDA_MMQ_RING_NTHREADS threads).
// Stage (kb, h), buffers: y ring slot s = (2*(kb - kb0_start) + h) % 3, raw x buffer b = (kb - kb0_start) % 2:
//   wait for the group of stage (kb, h), barrier; issue stage (kb, h) + 2: y(kb + 1, h) and, for h == 0, x[kb + 1];
//   commit (possibly empty); vec_dot half h.
template <ggml_type type, int J, bool fallback, bool fixup>
static __device__ __forceinline__ void ggml_cuda_mmq_ring_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
    constexpr int nthreads = GGML_CUDA_MMQ_RING_NTHREADS;
    constexpr int I        = GGML_CUDA_MMQ_ASYNC_I;
    constexpr int qk       = ggml_cuda_type_traits<type>::qk;
    constexpr int row_ints = ggml_cuda_mmq_async_raw_ints(type);
    constexpr int ne_block = QK8_1_MMQ;
    constexpr int sz       = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int rx_size  = I*row_ints;
    constexpr int ty_size  = J*MMQ_TILE_Y_K;
    static_assert(ggml_cuda_mmq_get_K_vram(type, J, fallback) == qk, "one block per row and K step expected");
    static_assert(J == GGML_CUDA_MMQ_ASYNC_J && I == GGML_CUDA_MMQ_ASYNC_I, "unexpected tile shape");

    extern __shared__ int data_mul_mat_q[];
    int * ds_h0 = data_mul_mat_q;     // over the J ids of mul_mat_q (unused here)
    int * rx0   = ds_h0 + 2*J;
    int * ty0   = rx0 + 2*rx_size;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    static_assert(nthreads >= 2*J, "one s pair int per thread");

    float sum[J*I / nthreads] = {0.0f};

    // prologue: group of stage (kb0_start, 0) = x[kb0_start] + y(kb0_start, 0); group of stage (kb0_start, 1)
    {
        const int * by = y + ncols_y*((kb0_start*qk/ne_block)*sz);
        ggml_cuda_mmq_ring_load_x<type, fallback>(rx0, x, offset_x + kb0_start, tile_x_max_i, stride_row_x);
        ggml_cuda_mmq_ring_load_y<J>(ty0, by);
        ggml_cuda_mmq_async_commit();
        ggml_cuda_mmq_ring_load_y<J>(ty0 + ty_size, by + ncols_y*sz);
        ggml_cuda_mmq_async_commit();
    }

    int s0 = 0; // ring slot of half 0 of kb (half 1: s1 = s0 + 1 mod 3)
    int b  = 0;
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const bool has_next = kb0 + 1 < kb0_stop;
        const int * by_next = y + ncols_y*(((kb0 + 1)*qk/ne_block)*sz);
        const int s1 = s0 == 2 ? 0 : s0 + 1;
        const int s2 = s1 == 2 ? 0 : s1 + 1;

        // half 0: y(kb, 0) in s0; issue y(kb + 1, 0) -> s2 (consumed: y(kb - 1, 1)) and x[kb + 1] -> b^1 (x[kb - 1])
        ggml_cuda_mmq_async_wait<1>();
        __syncthreads();
        if (has_next) {
            ggml_cuda_mmq_ring_load_y<J>(ty0 + s2*ty_size, by_next);
            ggml_cuda_mmq_ring_load_x<type, fallback>(rx0 + (b ^ 1)*rx_size, x, offset_x + kb0 + 1, tile_x_max_i, stride_row_x);
        }
        ggml_cuda_mmq_async_commit();
        if (tid < 2*J) { // the s pairs of half 0 for the min term in half 1 (its buffer is refilled during half 1)
            ds_h0[tid] = ty0[s0*ty_size + (tid/2)*MMQ_TILE_Y_K + tid%2];
        }
        ggml_cuda_mmq_ring_vec_dot<type, J, 0>(rx0 + b*rx_size, ty0 + s0*ty_size, sum, ds_h0);

        // half 1: y(kb, 1) in s1; issue y(kb + 1, 1) -> s0 (consumed: y(kb, 0))
        ggml_cuda_mmq_async_wait<1>();
        __syncthreads();
        if (has_next) {
            ggml_cuda_mmq_ring_load_y<J>(ty0 + s0*ty_size, by_next + ncols_y*sz);
        }
        ggml_cuda_mmq_async_commit();
        ggml_cuda_mmq_ring_vec_dot<type, J, 1>(rx0 + b*rx_size, ty0 + s1*ty_size, sum, ds_h0);

        s0 = s2;
        b ^= 1;
    }

    // the shared memory reads of the last half come before the next tile's prologue copies (stream-k: several tiles
    // per CUDA block)
    __syncthreads();

    if (fixup) {
        ggml_cuda_mmq_ring_write_back<type, J, fallback>(sum, tmp_fixup + blockIdx.x*(J*I), I, I, J);
    } else {
        ggml_cuda_mmq_ring_write_back<type, J, fallback>(sum, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
    GGML_UNUSED(y_scale);
    GGML_UNUSED(ids_dst);
}

#endif // GGML_CUDA_MMQ_ASYNC_DEVICE
