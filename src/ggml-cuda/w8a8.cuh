#pragma once

// "W8 x A8-sub32": an int8 tensor-core matmul for Q4_K / Q5_K weights on sm_80+ (device code and launchers; the ggml
// integration is w8a8.cu). Self-contained (CUDA headers only) so that the stand-alone tests in w8a8-tools/ compile the
// very same kernels.
//
// Number formats (the precision contract; the research emulation GGML_CUDA_W8A8=sub:8, w8a8-emu.cu, uses the same
// formulas and expressions, so the int8 values and sub-scales are bit-identical to it):
//   activations X [T][K] f32 (or the f16 SwiGLU output of the gate|up epilogue below), per token t:
//     amax_t = max_k |x|, d_t = amax_t / 127; per group g of 32 values: amax_g = the group's max,
//     sy = max(1, min(M, ceil(M * amax_g / amax_t))) (1 when amax_t == 0), M = 255; d_g = d_t * sy / M, inv_g = 1 / d_g;
//     q_x = clamp(rint(x * inv_g), -127, 127).
//   weights W [N][K] = ggml's dequantized Q4_K / Q5_K values: amax_n = max_k |w|, dW_n = amax_n / 127,
//     q_w = clamp(rint(w * (127 / amax_n)), -127, 127).
//   product: acc[t][n] = sum_g sy[t][g] * (sum_{k in g} q_x * q_w) (int32, exact), y = float(acc) * (d_t / 255) * dW_n.
// The GEMM issues every m16n8k32 MMA (one k step = one sub-scale group) with a zero C operand and adds sy * P to the
// int32 accumulator with one IMAD per output: no float work in the main loop.
//
// Buffers:
//   q_x  int8  [T_pad][K]          T_pad = T rounded up to W8A8_BM; rows >= T are 0
//   sy   uint8 [T_pad/BM][K/32][BM] per token tile and group, the tile's rows in the order the warps read them
//                                  (w8a8_sy_pos): a thread's 8 values of one group are 8 consecutive bytes
//   d_t  f32   [T_pad]             rows >= T are 0
//   q_w  int8  [N][K], dW f32 [N]  (glu: [gate | up] rows interleaved, gate_j at 2j and up_j at 2j+1)
// Overflow: |acc| <= 255 * 127^2 * K in theory, but acc = 255 * 127^2 * y / (amax_t * amax_n): an overflow needs
// |y| > 522 * amax_t * amax_n (K = 4096), impossible for real activations. The MMA is the saturating one; the IMAD
// accumulate would wrap in that pathological case.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define W8A8_DEVICE_OK 1
#else
#define W8A8_DEVICE_OK 0
#endif

#ifndef W8A8_BM
#define W8A8_BM       256   // tokens per block tile (256: 8 warps, one block per SM; 128: 4 warps, two blocks per SM)
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

#ifndef W8A8_STAGES
#define W8A8_STAGES   3
#endif
#ifndef W8A8_WHATIF
#define W8A8_WHATIF   0     // timing experiments of the stand-alone tests only (w8a8-tools); 0 = the real kernel
#endif
#ifndef W8A8_GROUP_N
#define W8A8_GROUP_N  0     // > 0: tile order by groups of this many channel tiles instead (token tile inner)
#endif
#ifndef W8A8_GROUP_M
#define W8A8_GROUP_M  4     // tile order: groups of this many token tiles, channel tiles inner
#endif

static constexpr int w8a8_stage_bytes() { return W8A8_BM*W8A8_BK + W8A8_BN*W8A8_BK + 2*W8A8_BM; }
static constexpr int w8a8_smem_bytes()  { return W8A8_STAGES*w8a8_stage_bytes(); }

// position of row r (0..BM-1) of a token tile in its sy block: r = wm*WM + mf*16 + h*8 + gid  ->
// wm*WM + gid*2*MF + mf*2 + h (a thread's 2*MF values of one group are consecutive bytes)
static __host__ __device__ __forceinline__ int w8a8_sy_pos(const int r) {
    const int wm = r / W8A8_WM, mf = (r % W8A8_WM) >> 4, h = (r >> 3) & 1, gid = r & 7;
    return wm*W8A8_WM + gid*2*W8A8_MF + mf*2 + h;
}

static __host__ __device__ __forceinline__ int64_t w8a8_pad_tokens(const int64_t T) {
    return (T + W8A8_BM - 1) / W8A8_BM * W8A8_BM;
}

// ---- weights: Q4_K / Q5_K rows -> int8 rows with a per-row scale (single pass, the row max computed in the kernel) ----

struct w8a8_block_q4_K { __half d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct w8a8_block_q5_K { __half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
static_assert(sizeof(w8a8_block_q4_K) == 144 && sizeof(w8a8_block_q5_K) == 176, "k-quant block sizes");

// ggml's get_scale_min_k4
static __device__ __forceinline__ void w8a8_scale_min_k4(const int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// One warp per weight row; lane l makes the values 8l .. 8l+7 of every 256-value block (sub-block j = l/4, as
// dequantize_q4_K / _q5_K compute them: d1 = d*sc, m1 = dmin*m in f32, w = d1*q - m1 as one FMA). Pass 1: the row max;
// pass 2 (the row again, from the caches): q = clamp(rint(w * (127/amax))). glu: row n of the [gate | up] weight
// (N rows) is written to row 2n (gate, n < N/2) or 2(n - N/2) + 1 (up).
template <bool q5>
static __device__ __forceinline__ void w8a8_dequant8(const uint8_t * __restrict__ blk, const int lane, float (&w)[8]) {
    const int j   = lane / 4;              // 32-value sub-block
    const int g2  = j / 2;                 // 64-value group: qs bytes 32*g2 .. +31, low nibbles = sub-block 2*g2
    const int hi  = j & 1;
    const int pos = 8 * (lane % 4);        // value position inside the sub-block
    const __half2 dm = *(const __half2 *) blk;
    const float dall = __low2float(dm);
    const float dmin = __high2float(dm);
    uint8_t sc, m;
    w8a8_scale_min_k4(j, blk + 4, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    const uint8_t * qs = blk + (q5 ? 48 : 16) + 32*g2 + pos;
    const uint2 raw = *(const uint2 *) qs;
    uint2 qhv = make_uint2(0, 0);
    if (q5) {
        qhv = *(const uint2 *) (blk + 16 + pos);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t r = i < 4 ? raw.x : raw.y;
        int q = (r >> (8*(i % 4) + 4*hi)) & 0xF;
        if (q5) {
            const uint32_t h = i < 4 ? qhv.x : qhv.y;
            q += ((h >> (8*(i % 4) + j)) & 1) ? 16 : 0;
        }
        w[i] = __fmaf_rn(d1, (float) q, -m1);
    }
}

template <bool q5, bool glu>
static __global__ void __launch_bounds__(256) w8a8_convert_weights(
        const void * __restrict__ vx, const int64_t row_bytes, int8_t * __restrict__ qw, float * __restrict__ dw,
        const int N, const int K) {
#if W8A8_DEVICE_OK
    const int n    = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (n >= N) {
        return;
    }
    constexpr int bsz = q5 ? (int) sizeof(w8a8_block_q5_K) : (int) sizeof(w8a8_block_q4_K);
    const uint8_t * row = (const uint8_t *) vx + (int64_t) n * row_bytes;
    const int nb = K / 256;
    float amax = 0.0f;
    for (int b = 0; b < nb; ++b) {
        float w[8];
        w8a8_dequant8<q5>(row + (int64_t) b*bsz, lane, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            amax = fmaxf(amax, fabsf(w[i]));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
    }
    // the emulation's w8a8_rowquant (bits == 0): inv = 127 / amax, d_row = amax / 127
    const float inv  = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int   nout = glu ? (n < N/2 ? 2*n : 2*(n - N/2) + 1) : n;
    if (lane == 0) {
        dw[nout] = amax / 127.0f;
    }
    int8_t * out = qw + (int64_t) nout * K;
    const int j = lane / 4, pos = 8 * (lane % 4);
    for (int b = 0; b < nb; ++b) {
        float w[8];
        w8a8_dequant8<q5>(row + (int64_t) b*bsz, lane, w);
        uint32_t o[2] = {0, 0};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = max(-127, min(127, __float2int_rn(w[i] * inv)));
            o[i / 4] |= (uint32_t) (q & 0xFF) << (8*(i % 4));
        }
        *(uint2 *) (out + b*256 + 32*j + pos) = make_uint2(o[0], o[1]);
    }
#else
    (void) vx; (void) row_bytes; (void) qw; (void) dw; (void) N; (void) K;
#endif
}

// ---- activations: [T][K] f32 / f16 -> q_x, sy, d_t (one block of 256 threads per token row, the row in registers) ----

enum w8a8_act_pre {
    W8A8_PRE_NONE  = 0,
    W8A8_PRE_SCALE = 1,   // GGML_OP_SCALE fused: scale * x + bias as one FMA (scale_f32's rounding)
};

// NV = K / 1024 float4 (or half4) per thread. The arithmetic is the emulation's (w8a8_rowquant with bits > 0), with M
// a run-time value as there, so that the compiler emits the same instructions for the divisions.
template <int NV, typename src_t, int pre>
static __global__ void __launch_bounds__(256) w8a8_quantize_act(
        const src_t * __restrict__ x, const int64_t s01, int8_t * __restrict__ qx, uint8_t * __restrict__ sy,
        float * __restrict__ dt, const int T, const int K, const int M, const float scale, const float bias) {
#if W8A8_DEVICE_OK
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const int G   = K / 32;
    int8_t * qrow = qx + (int64_t) t * K;
    uint8_t * sy_tile = sy + (int64_t) (t / W8A8_BM) * G * W8A8_BM + w8a8_sy_pos(t % W8A8_BM);
    if (t >= T) {  // padding row of the last token tile
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            *(char4 *) (qrow + 4*(i*256 + tid)) = make_char4(0, 0, 0, 0);
            if (tid % 8 == 0) {
                sy_tile[(int64_t) ((i*256 + tid) / 8) * W8A8_BM] = 0;
            }
        }
        if (tid == 0) {
            dt[t] = 0.0f;
        }
        return;
    }
    float4 v[NV];
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int64_t k = 4*(int64_t) (i*256 + tid);
        if constexpr (sizeof(src_t) == 2) {
            const uint2 raw = *(const uint2 *) ((const __half *) x + t*s01 + k);
            const float2 a = __half22float2(*(const __half2 *) &raw.x);
            const float2 b = __half22float2(*(const __half2 *) &raw.y);
            v[i] = make_float4(a.x, a.y, b.x, b.y);
        } else {
            v[i] = *(const float4 *) ((const float *) x + t*s01 + k);
        }
        if constexpr (pre == W8A8_PRE_SCALE) {
            v[i].x = __fmaf_rn(scale, v[i].x, bias);
            v[i].y = __fmaf_rn(scale, v[i].y, bias);
            v[i].z = __fmaf_rn(scale, v[i].z, bias);
            v[i].w = __fmaf_rn(scale, v[i].w, bias);
        }
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    }
    __shared__ float red[8];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off, 32));
    }
    if (tid % 32 == 0) {
        red[tid / 32] = m;
    }
    __syncthreads();
    float amax = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
        amax = fmaxf(amax, red[w]);
    }
    const float d_row = amax / 127.0f;
    if (tid == 0) {
        dt[t] = d_row;
    }
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        float g = fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w)));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            g = fmaxf(g, __shfl_xor_sync(0xFFFFFFFF, g, off, 32));
        }
        const int   s   = amax > 0.0f ? max(1, min(M, (int) ceilf((float) M * g / amax))) : 1;
        const float d   = d_row * (float) s / (float) M;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        char4 q;
        q.x = (signed char) max(-127, min(127, __float2int_rn(v[i].x * inv)));
        q.y = (signed char) max(-127, min(127, __float2int_rn(v[i].y * inv)));
        q.z = (signed char) max(-127, min(127, __float2int_rn(v[i].z * inv)));
        q.w = (signed char) max(-127, min(127, __float2int_rn(v[i].w * inv)));
        *(char4 *) (qrow + 4*(i*256 + tid)) = q;
        if (tid % 8 == 0) {
            sy_tile[(int64_t) ((i*256 + tid) / 8) * W8A8_BM] = (uint8_t) s;
        }
    }
#else
    (void) x; (void) s01; (void) qx; (void) sy; (void) dt; (void) T; (void) K; (void) M; (void) scale; (void) bias;
#endif
}

// ---- the GEMM ----

enum w8a8_epilogue {
    W8A8_EPI_F32      = 0,  // y = acc * d_t/255 * dW_n -> f32 dst[t][n] (ggml's [N, T] layout)
    W8A8_EPI_GLU_F16  = 1,  // interleaved [gate | up] rows: silu(y_gate) * y_up -> f16 [t][n/2]
    W8A8_EPI_GLU_F32  = 2,  // the same, f32 out
    W8A8_EPI_I32      = 3,  // the raw int32 accumulators [t][n] (tests)
};

#if W8A8_DEVICE_OK
// 16-byte copy global -> shared, L1 bypassed; L2::128B: the L2 fetches the whole 128-byte line (each row's next
// K step, i.e. the next stage, is then an L2 hit; CUTLASS's sm80 copies use the same hint)
static __device__ __forceinline__ void w8a8_cp_async16(const unsigned dst, const void * src) {
#ifndef W8A8_NO_L2_PREFETCH
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" : : "r"(dst), "l"(src));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(dst), "l"(src));
#endif
}
static __device__ __forceinline__ void w8a8_cp_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int n>
static __device__ __forceinline__ void w8a8_cp_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
}
static __device__ __forceinline__ void w8a8_ldsm_x4(int (&r)[4], const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
static __device__ __forceinline__ void w8a8_mma_zero(int (&d)[4], const int (&a)[4], const int b0, const int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %10, %10, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
// byte offset of 16-byte chunk c (0..3) of row r in a [rows][64 bytes] tile: XOR swizzle over row pairs, so that the 8
// rows of an ldmatrix 8x8 matrix and the 8 chunks of a cp.async phase hit 8 different 16-byte bank groups
static __device__ __forceinline__ int w8a8_swz(const int r, const int c) {
    return r*64 + ((c ^ ((r >> 1) & 3)) << 4);
}
#endif // W8A8_DEVICE_OK

// One block per BM x BN output tile; grid = ntm * ntn (grouped order); dst: f32 [T][ldd] (EPI_F32) or the GLU output
// [T][ldd] (ldd = N/2 values per row).
template <int epi>
static __global__ void __launch_bounds__(W8A8_NTHREADS, W8A8_MINBLOCKS) w8a8_gemm(
        const int8_t * __restrict__ qx, const uint8_t * __restrict__ sy, const float * __restrict__ dt,
        const int8_t * __restrict__ qw, const float * __restrict__ dw, void * __restrict__ dst, const int64_t ldd,
        const int T, const int N, const int K) {
#if W8A8_DEVICE_OK
    extern __shared__ __align__(128) uint8_t w8a8_smem[];
    constexpr int BM = W8A8_BM, BN = W8A8_BN, BK = W8A8_BK, S = W8A8_STAGES;
    constexpr int A_BYTES = BM*BK, B_BYTES = BN*BK, SY_BYTES = 2*BM, STAGE = A_BYTES + B_BYTES + SY_BYTES;

    const int ntm = (T + BM - 1) / BM;
    const int ntn = N / BN;
    const int bid = blockIdx.x;
#if W8A8_GROUP_N > 0
    // grouped tile order: W8A8_GROUP_N channel tiles x all token tiles per group, the channel tile fastest
    const int per_grp  = W8A8_GROUP_N * ntm;
    const int grp      = bid / per_grp;
    const int first_n  = grp * W8A8_GROUP_N;
    const int gsize    = min(W8A8_GROUP_N, ntn - first_n);
    const int in_grp   = bid % per_grp;
    const int tn       = first_n + in_grp % gsize;
    const int tm       = in_grp / gsize;
#else
    // grouped tile order: W8A8_GROUP_M token tiles x all channel tiles per group, the token tile fastest
    const int per_grp  = W8A8_GROUP_M * ntn;
    const int grp      = bid / per_grp;
    const int first_m  = grp * W8A8_GROUP_M;
    const int gsize    = min(W8A8_GROUP_M, ntm - first_m);
    const int in_grp   = bid % per_grp;
    const int tm       = first_m + in_grp % gsize;
    const int tn       = in_grp / gsize;
#endif
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int wm   = warp % W8A8_WARPS_M;   // W8A8_WM-token slice
    constexpr int MF = W8A8_MF, WM = W8A8_WM;
    const int wn   = warp / W8A8_WARPS_M;   // 64-channel slice
    const int gid  = lane / 4;
    const int tig  = lane % 4;

    const int G  = K / 32;
    const int nk = K / BK;

    const unsigned sbase = (unsigned) __cvta_generic_to_shared(w8a8_smem);

    // per-thread copy addresses: chunk i*NTHREADS + tid of a stage = row i*NTHREADS/4 + tid/4, 16-byte chunk tid%4
    // (rows r and r + 32*i have the same swizzle); the sub-scales: threads 0..31 copy 16 of the stage's 512 bytes each
    const unsigned cp_soff = (unsigned) w8a8_swz(tid / 4, tid % 4);
    const int8_t  * pA  = qx + ((int64_t) m0 + tid/4) * K + 16*(tid % 4);
    const int8_t  * pB  = qw + ((int64_t) n0 + tid/4) * K + 16*(tid % 4);
    const uint8_t * pS  = sy + (int64_t) tm * G * BM + 16*(tid % 32);
    const int64_t   KR  = (W8A8_NTHREADS/4) * (int64_t) K;   // the rows one pass of the threads copies
    auto load_stage = [&](const int slot) {  // the stage after the last one copied: pA / pB / pS advance
        const unsigned s = sbase + slot*STAGE + cp_soff;
#pragma unroll
        for (int i = 0; i < A_BYTES/16/W8A8_NTHREADS; ++i) {
            w8a8_cp_async16(s + i*(W8A8_NTHREADS/4)*BK, pA + i*KR);
        }
#pragma unroll
        for (int i = 0; i < B_BYTES/16/W8A8_NTHREADS; ++i) {
            w8a8_cp_async16(s + A_BYTES + i*(W8A8_NTHREADS/4)*BK, pB + i*KR);
        }
        if (tid < SY_BYTES/16) {
            w8a8_cp_async16(sbase + slot*STAGE + A_BYTES + B_BYTES + 16*tid, pS);
        }
        pA += BK;
        pB += BK;
        pS += 2*BM;
    };

    int acc[MF][8][4];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int nf = 0; nf < 8; ++nf)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[mf][nf][j] = 0;

    // per-lane fragment offsets in a stage (rows r and r + 16*i have the same swizzle, so the m / n fragment is an
    // immediate offset of mf*1024 / np*1024); index = k step
    unsigned a_off[2], b_off[2];
    {
        const int a_row = wm*WM + (lane % 16);
        const int a_x   = (lane / 16) ^ ((a_row >> 1) & 3);
        const int b_row = wn*64 + (lane % 8) + (lane / 16)*8;
        const int b_x   = ((lane / 8) % 2) ^ ((b_row >> 1) & 3);
        a_off[0] = a_row*64 + (a_x << 4);
        a_off[1] = a_row*64 + ((a_x ^ 2) << 4);
        b_off[0] = A_BYTES + b_row*64 + (b_x << 4);
        b_off[1] = A_BYTES + b_row*64 + ((b_x ^ 2) << 4);
    }
    const unsigned sy_off = A_BYTES + B_BYTES + wm*WM + gid*2*MF;

    // The pipeline. Per k step (32 MMAs per warp, 4 pairs of n fragments x 2 x 4 m fragments) the B fragments are
    // loaded one pair ahead; the A fragments and sub-scales of the next k step are loaded during the last pair, each
    // m fragment right after its last MMA into the same registers (the m order alternates with the k step parity, so
    // the fragment loaded first is the one needed first). One fragment set + one B pair in flight: the registers this
    // saves hold the zero-C MMAs' products until their IMADs. One barrier per stage, at the last pair of k step 1:
    // every read of the stage's slot is issued before it, so the slot takes stage kt+S right after it (S stages
    // resident, S-1 copies in flight), and stage kt+1 is complete (wait) before its fragments are read.
    auto load_a1 = [&](int (&f)[4], const unsigned st, const int ks, const int mf) {
#if W8A8_WHATIF == 13   // what-if (wrong results): no fragment loads (registers change by a cheap op)
        f[0] += (int) st; f[1] ^= ks; (void) mf;
#else
        w8a8_ldsm_x4(f, st + a_off[ks] + mf*1024);
#endif
    };
    auto load_s = [&](uint2 & fs, const unsigned st, const int ks) {
#if W8A8_WHATIF == 2 || W8A8_WHATIF == 7 || W8A8_WHATIF == 13   // what-if (wrong results): constant sub-scales, no shared-memory reads of sy
        fs = make_uint2(0x01020304u + st, 0x05060708u + ks);
#else
        if constexpr (MF == 4) {
            asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(fs.x), "=r"(fs.y) : "r"(st + sy_off + ks*BM));
        } else {
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(fs.x) : "r"(st + sy_off + ks*BM));
            fs.y = 0;
        }
#endif
    };
    auto load_b = [&](int (&fb)[4], const unsigned st, const int ks, const int np) {
#if W8A8_WHATIF == 13
        fb[0] += (int) st; fb[2] ^= np + ks;
#else
        w8a8_ldsm_x4(fb, st + b_off[ks] + np*1024);
#endif
    };

    int   fa[MF][4] = {};
    uint2 fs;
    int   fb[2][4] = {};

    // one k step (parity par = ks): fa / fs / fb[0] hold its A fragments, sub-scales and B pair 0; its B pairs 1..3
    // are at st / ks; the next k step is at nst / nks; hook(np) runs before the loads and MMAs of pair np
    auto kstep = [&](const int par, const unsigned st, const int ks, const unsigned nst, const int nks, auto && hook) {
        int sv[MF][2];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) {
            const unsigned w = mf < 2 ? fs.x : fs.y;
            sv[mf][0] = (w >> (16*(mf % 2)))     & 0xFF;
            sv[mf][1] = (w >> (16*(mf % 2) + 8)) & 0xFF;
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            hook(np);
            if (np < 3) {
                load_b(fb[(np + 1) % 2], st, ks, np + 1);
            } else {
                load_b(fb[0], nst, nks, 0);
                load_s(fs, nst, nks);
            }
            const int (&b)[4] = fb[np % 2];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
                for (int j = 0; j < MF; ++j) {
                    // m order: par 0: 0 1 2 3 | 3 2 1 0 ; par 1: 3 2 1 0 | 0 1 2 3 (serpentine: the A fragment of the
                    // last MMA is reused; the last pair's second half frees 3 2 1 0 / 0 1 2 3 = the next k step's order)
                    const int mf = (nn ^ par) ? MF - 1 - j : j;
                    int (&c)[4] = acc[mf][2*np + nn];
#if W8A8_WHATIF == 1 || W8A8_WHATIF == 6   // what-if (wrong results): plain accumulation in the MMA, no sub-scales
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                                 : "r"(fa[mf][0]), "r"(fa[mf][1]), "r"(fa[mf][2]), "r"(fa[mf][3]), "r"(b[2*nn]), "r"(b[2*nn + 1]));
#else
                    int p[4];
                    w8a8_mma_zero(p, fa[mf], b[2*nn], b[2*nn + 1]);
#if W8A8_WHATIF == 8   // what-if (wrong results): IADD instead of the IMAD (kept out of the MMA by asm)
#pragma unroll
                    for (int q = 0; q < 4; ++q) asm volatile("add.s32 %0, %0, %1;" : "+r"(c[q]) : "r"(p[q]));
#elif W8A8_WHATIF == 9   // what-if (wrong results): one FADD per element (P bits as float)
#pragma unroll
                    for (int q = 0; q < 4; ++q) c[q] = __float_as_int(__int_as_float(c[q]) + __int_as_float(p[q]));
#elif W8A8_WHATIF == 10  // what-if (wrong results): one FFMA per element
#pragma unroll
                    for (int q = 0; q < 4; ++q) c[q] = __float_as_int(fmaf(__int_as_float(sv[mf][q/2]), __int_as_float(p[q]), __int_as_float(c[q])));
#elif W8A8_WHATIF == 11  // what-if (wrong results): FADD + FFMA per element (the exact float route's cost)
#pragma unroll
                    for (int q = 0; q < 4; ++q) c[q] = __float_as_int(fmaf(__int_as_float(sv[mf][q/2]), __int_as_float(p[q]) - 12582912.0f, __int_as_float(c[q])));
#elif W8A8_WHATIF == 12  // what-if (wrong results): LEA (shift-add, ALU pipe) per element
#pragma unroll
                    for (int q = 0; q < 4; ++q) asm volatile("{ .reg .u32 t; shl.b32 t, %1, 3; add.s32 %0, %0, t; }" : "+r"(c[q]) : "r"(p[q]));
#else
                    c[0] += sv[mf][0] * p[0];
                    c[1] += sv[mf][0] * p[1];
                    c[2] += sv[mf][1] * p[2];
                    c[3] += sv[mf][1] * p[3];
#endif
#endif
                    if (np == 3 && nn == 1) {
                        load_a1(fa[mf], nst, nks, mf);   // its last use in this k step
                    }
                }
            }
        }
    };

    // prologue: stages 0 .. S-1 (every slot), then the fragments of k step 0
#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < nk) {
            load_stage(s);
        }
        w8a8_cp_commit();
    }
    w8a8_cp_wait<S - 1>();
    __syncthreads();
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        load_a1(fa[mf], sbase, 0, mf);
    }
    load_s(fs, sbase, 0);
    load_b(fb[0], sbase, 0, 0);

    int slot = 0;        // the stage being computed
    int left = nk - S;   // stages still to copy
    for (int kt = 0; kt < nk; ++kt) {
        const int nslot = slot + 1 == S ? 0 : slot + 1;
        const unsigned st  = sbase + slot*STAGE;
        const unsigned nst = sbase + nslot*STAGE;
        kstep(0, st, 0, st, 1, [&](const int np) { (void) np; });
        kstep(1, st, 1, nst, 0, [&](const int np) {
            if (np == 3) {
                // every read of this stage's slot is issued: after the barrier it takes stage kt+S; stage kt+1 is
                // complete (the copies of stages <= kt+1 done: S-2 newer groups may be pending)
                w8a8_cp_wait<S - 2>();
#if W8A8_WHATIF != 4   // what-if 4 (wrong results, races): no barrier
                __syncthreads();
#endif
#if W8A8_WHATIF == 3 || W8A8_WHATIF == 6 || W8A8_WHATIF >= 7   // what-if (wrong results): no global loads after the prologue
                if (false) {
#else
                if (left > 0) {
#endif
                    load_stage(slot);
                }
                w8a8_cp_commit();
                --left;
            }
        });
        slot = nslot;
    }
    w8a8_cp_wait<0>();

    // epilogue: per m fragment (16 rows of the warp's 64 x 64 outputs) the scaled values go through the warp's own
    // shared-memory buffer and leave as 16-byte rows (a warp store = 2 x 256 contiguous bytes, f32; 8 x 64, GLU f16),
    // with streaming stores (evict-first: the output does not push the next tiles' operands out of L2)
    __syncthreads();  // the pipeline's slots are free (the last stages' fragment reads are done)
    constexpr int EPI_COLS = epi == W8A8_EPI_F32 || epi == W8A8_EPI_I32 ? 64 : 32;   // values per row of the warp
    constexpr int EPI_ESZ  = epi == W8A8_EPI_GLU_F16 ? 2 : 4;                         // bytes per value
    constexpr int EPI_LD   = EPI_COLS + 32/EPI_ESZ;   // row stride in values (+32 bytes: conflict-free fragment writes)
    uint8_t * ebuf = w8a8_smem + warp * (16 * EPI_LD * EPI_ESZ);
    float wsc[8][2];
#pragma unroll
    for (int nf = 0; nf < 8; ++nf) {
        const float2 d2 = *(const float2 *) (dw + n0 + wn*64 + nf*8 + 2*tig);
        wsc[nf][0] = d2.x;
        wsc[nf][1] = d2.y;
    }
    float st[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = m0 + wm*WM + mf*16 + h*8 + gid;
            st[mf][h] = t < T ? dt[t] * (1.0f / 255.0f) : 0.0f;
        }
    }
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = h*8 + gid;   // row in the 16-row chunk
#pragma unroll
            for (int nf = 0; nf < 8; ++nf) {
                const int c = nf*8 + 2*tig;
                if constexpr (epi == W8A8_EPI_I32) {
                    *(int2 *) (ebuf + (r*EPI_LD + c)*4) = make_int2(acc[mf][nf][2*h + 0], acc[mf][nf][2*h + 1]);
                } else {
                    const float y0 = (float) acc[mf][nf][2*h + 0] * st[mf][h] * wsc[nf][0];
                    const float y1 = (float) acc[mf][nf][2*h + 1] * st[mf][h] * wsc[nf][1];
                    if constexpr (epi == W8A8_EPI_F32) {
                        *(float2 *) (ebuf + (r*EPI_LD + c)*4) = make_float2(y0, y1);
                    } else {
                        // ggml's swiglu: silu(gate) * up (unary_gated_op_kernel<op_silu>, silu = x / (1 + exp(-x)))
                        const float g = __fmul_rn(y0 / (1.0f + expf(-y0)), y1);
                        if constexpr (epi == W8A8_EPI_GLU_F16) {
                            *(__half *) (ebuf + (r*EPI_LD + c/2)*2) = __float2half(g);
                        } else {
                            *(float *) (ebuf + (r*EPI_LD + c/2)*4) = g;
                        }
                    }
                }
            }
        }
        __syncwarp();
        // 16 rows x EPI_COLS*EPI_ESZ bytes = 16 x 256 (f32) / 64 (f16) / 128 (GLU f32) bytes, 16 bytes per lane
        constexpr int V_ROW = EPI_COLS*EPI_ESZ/16;     // 16-byte vectors per row
        constexpr int ROWS  = 32 / V_ROW;              // rows per warp store
        const int vr = lane / V_ROW, vc = lane % V_ROW;
#pragma unroll
        for (int r0 = 0; r0 < 16; r0 += ROWS) {
            const int r = r0 + vr;
            const int t = m0 + wm*WM + mf*16 + r;
            const int4 v = *(const int4 *) (ebuf + r*EPI_LD*EPI_ESZ + vc*16);
#if W8A8_WHATIF == 5   // what-if (wrong results): no epilogue stores unless the value is a magic one
            if (v.x != 0x12345678) continue;
#endif
            if (t < T) {
                // column of this vector's first value in the output row
                const int64_t col = epi == W8A8_EPI_F32 || epi == W8A8_EPI_I32 ? n0 + wn*64 + vc*4 : (n0 + wn*64)/2 + vc*(16/EPI_ESZ);
                __stcs((int4 *) ((uint8_t *) dst + ((int64_t) t*ldd + col)*EPI_ESZ), v);
            }
        }
    }
#else
    (void) qx; (void) sy; (void) dt; (void) qw; (void) dw; (void) dst; (void) ldd; (void) T; (void) N; (void) K;
#endif
}

// ---- launchers (raw pointers; stream-ordered, no synchronization) ----

static inline void w8a8_convert_weights_cuda(const void * vx, const bool q5, const bool glu, const int64_t row_bytes,
                                             int8_t * qw, float * dw, const int N, const int K, cudaStream_t stream) {
    const dim3 grid((unsigned) ((N + 7) / 8));
    if (q5) {
        if (glu) w8a8_convert_weights<true,  true ><<<grid, 256, 0, stream>>>(vx, row_bytes, qw, dw, N, K);
        else     w8a8_convert_weights<true,  false><<<grid, 256, 0, stream>>>(vx, row_bytes, qw, dw, N, K);
    } else {
        if (glu) w8a8_convert_weights<false, true ><<<grid, 256, 0, stream>>>(vx, row_bytes, qw, dw, N, K);
        else     w8a8_convert_weights<false, false><<<grid, 256, 0, stream>>>(vx, row_bytes, qw, dw, N, K);
    }
}

// K % 1024 == 0, K <= 16384. src: f32 or f16 rows with stride s01 (elements). Launches T_pad blocks.
template <typename src_t>
static inline bool w8a8_quantize_act_cuda(const src_t * x, const int64_t s01, const int pre, const float scale, const float bias,
                                          int8_t * qx, uint8_t * sy, float * dt, const int T, const int K, cudaStream_t stream) {
    const unsigned grid = (unsigned) w8a8_pad_tokens(T);
    const int M = 255;
#define W8A8_Q(NV) case NV: \
        if (pre == W8A8_PRE_SCALE) w8a8_quantize_act<NV, src_t, W8A8_PRE_SCALE><<<grid, 256, 0, stream>>>(x, s01, qx, sy, dt, T, K, M, scale, bias); \
        else                       w8a8_quantize_act<NV, src_t, W8A8_PRE_NONE ><<<grid, 256, 0, stream>>>(x, s01, qx, sy, dt, T, K, M, scale, bias); \
        return true;
    switch (K / 1024) {
        W8A8_Q(1) W8A8_Q(2) W8A8_Q(3) W8A8_Q(4) W8A8_Q(5) W8A8_Q(6) W8A8_Q(8) W8A8_Q(12) W8A8_Q(16)
        default: return false;
    }
#undef W8A8_Q
}

static inline cudaError_t w8a8_gemm_cuda(const int epi, const int8_t * qx, const uint8_t * sy, const float * dt, const int8_t * qw,
                                         const float * dw, void * dst, const int64_t ldd, const int T, const int N, const int K,
                                         cudaStream_t stream) {
    const int ntm = (T + W8A8_BM - 1) / W8A8_BM;
    const int ntn = N / W8A8_BN;
    const dim3 grid((unsigned) (ntm * ntn));
    const int smem = w8a8_smem_bytes();
    cudaError_t err = cudaSuccess;
#define W8A8_G(E) { \
        static bool attr_set = false; \
        if (!attr_set) { err = cudaFuncSetAttribute(w8a8_gemm<E>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);             if (err == cudaSuccess) err = cudaFuncSetAttribute(w8a8_gemm<E>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);             attr_set = err == cudaSuccess; } \
        if (err != cudaSuccess) return err; \
        w8a8_gemm<E><<<grid, W8A8_NTHREADS, smem, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K); }
    switch (epi) {
        case W8A8_EPI_F32:     W8A8_G(W8A8_EPI_F32);     break;
        case W8A8_EPI_GLU_F16: W8A8_G(W8A8_EPI_GLU_F16); break;
        case W8A8_EPI_GLU_F32: W8A8_G(W8A8_EPI_GLU_F32); break;
        case W8A8_EPI_I32:     W8A8_G(W8A8_EPI_I32);     break;
        default: return cudaErrorInvalidValue;
    }
#undef W8A8_G
    return cudaGetLastError();
}
