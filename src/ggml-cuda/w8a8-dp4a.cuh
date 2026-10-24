#pragma once

// The W8 x A8-sub32 GEMM for cards without int8 tensor cores (Pascal, cc 6.1): the very buffers, number formats and
// epilogues of w8a8_gemm (w8a8.cuh), the products as dp4a (4 int8 products a lane per instruction). Self-contained
// (CUDA headers + w8a8.cuh) so that the stand-alone tests in w8a8-tools/ compile the same kernel.
//
// Work split: a block computes a 128-token x BN-channel tile, each of its 2*BN threads 8 tokens x 8 channels. The
// operands of a 64-value K stage sit in shared memory transposed into int32 words ([k4][token], [k4][channel]: 4 int8
// values of K per word), so that a thread reads its 8 tokens' (8 channels') words of one k4 with two 16-byte loads and
// issues 64 dp4a with them. Per 32-value group the thread's 64 dot products (8 k4 steps) are added to the accumulators
// times the tokens' sub-scales sy (acc += sy * P): the format's integer sum, exact (int32, as the tensor-core kernel's:
// the f32 outputs are bit-identical to w8a8_gemm's) or, W8A8D_SCALE_F32, in f32 (I2F + FFMA: cheaper on Pascal, whose
// 32-bit integer multiply is three XMADs; a rounding-class difference).
//
// The thread's 8 tokens are the 8 rows whose sub-scales are consecutive bytes of the token tile's sy block (w8a8_sy_pos:
// rows wm*64 + gid + 8i, i = 0..7, are positions wm*64 + gid*8 + i), so that one 8-byte load gives them; the A tile is
// stored in that position order too. Columns are XOR-swizzled (bit 2 by bit 5) so that both the copies (32 consecutive
// rows per warp) and the 16-byte reads (4 token groups x 8 channel groups per warp) are free of bank conflicts.
//
// Pipeline: no cp.async on Pascal; the next stage's global loads go to registers before the current stage's math and
// to the other shared buffer after it (one barrier per stage). BN 64 (128 threads) lets two blocks share an SM, so
// that one computes while the other waits at its barrier.

#include "w8a8.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
#define W8A8D_DEVICE_OK 1
#else
#define W8A8D_DEVICE_OK 0
#endif

#define W8A8D_BM       128   // tokens per block tile (= W8A8_BM: the sy blocks are per W8A8_BM tokens)
#ifndef W8A8D_BN
#define W8A8D_BN       64    // channels per block tile (64 or 128)
#endif
#define W8A8D_BK       64    // K values per stage (2 sub-scale groups)
#define W8A8D_NTHREADS (2*W8A8D_BN)
#ifndef W8A8D_MINB
#define W8A8D_MINB     (W8A8D_BN == 64 ? 2 : 1)   // blocks per SM the registers are budgeted for
#endif
#ifndef W8A8D_SCALE_F32
#define W8A8D_SCALE_F32 1    // the group sums scaled in f32 (I2F + FFMA): 10 % faster on a GTX 1070 Ti than the exact int32
#endif                       // sums (3 XMADs per multiply-add on Pascal); 0 = exact, the tensor-core kernel's bits
#ifndef W8A8D_PREFETCH
#define W8A8D_PREFETCH 0     // 1: the next k4 step's words loaded before this step's dot products (no gain measured)
#endif

static_assert(W8A8D_BM == W8A8_BM && W8A8_WM == 64 && W8A8_MF == 4, "the dp4a kernel's token mapping assumes the 64-row warp tiles of w8a8_sy_pos");
static_assert(W8A8D_BN == 64 || W8A8D_BN == 128, "BN");
#ifndef W8A8D_GI_UNROLL
#define W8A8D_GI_UNROLL 1    // the stage's two sub-scale groups as a loop: unrolled, the stage's ~1,600 instructions
#endif                       // (17 KB with the control words) overflow Pascal's instruction cache: 13.6 vs 18.9 T/s
#ifndef W8A8D_K4_UNROLL
#define W8A8D_K4_UNROLL 8    // a group's 8 k4 steps unrolled
#endif
static constexpr int W8A8D_GU = W8A8D_GI_UNROLL;
static constexpr int W8A8D_KU = W8A8D_K4_UNROLL;

// the tile row whose sub-scale sits at position p of the sy block (the inverse of w8a8_sy_pos)
static __host__ __device__ __forceinline__ int w8a8d_sy_row(const int p) {
    const int wm = p / 64, gid = (p % 64) / 8, i = p % 8;   // i = mf*2 + h
    return wm*64 + (i / 2)*16 + (i % 2)*8 + gid;
}

// swizzled column of word c in a [k4][128 or 64] shared tile
static __host__ __device__ __forceinline__ int w8a8d_swz(const int c) {
    return c ^ (((c >> 5) & 1) << 2);
}

template <int epi>
static __global__ void __launch_bounds__(W8A8D_NTHREADS, W8A8D_MINB) w8a8d_gemm(
        const int8_t * __restrict__ qx, const uint8_t * __restrict__ sy, const float * __restrict__ dt,
        const int8_t * __restrict__ qw, const float * __restrict__ dw, void * __restrict__ dst, const int64_t ldd,
        const int T, const int N, const int K, const int group) {
#if W8A8D_DEVICE_OK
    constexpr int BM = W8A8D_BM, BN = W8A8D_BN, BK = W8A8D_BK, KR = BK/4, NG = BK/32, NT = W8A8D_NTHREADS;
    __shared__ __align__(16) int     As[2][KR][BM];
    __shared__ __align__(16) int     Bs[2][KR][BN];
    __shared__ __align__(16) uint8_t Ss[2][NG][BM];

    const int ntm = (T + BM - 1) / BM;
    const int ntn = N / BN;
    // grouped tile order, as w8a8_gemm's
    const int bid = blockIdx.x;
    int tm, tn;
    if (group > 0) {
        const int per_grp = group * ntn;
        const int first_m = (bid / per_grp) * group;
        const int gsize   = min(group, ntm - first_m);
        const int in_grp  = bid % per_grp;
        tm = first_m + in_grp % gsize;
        tn = in_grp / gsize;
    } else {
        const int per_grp = -group * ntm;
        const int first_n = (bid / per_grp) * -group;
        const int gsize   = min(-group, ntn - first_n);
        const int in_grp  = bid % per_grp;
        tn = first_n + in_grp % gsize;
        tm = in_grp / gsize;
    }
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    constexpr int WN = BN / 64;                    // warps across the channels
    const int tg   = (warp / WN) * 4 + lane / 8;   // token group: sy positions tg*8 .. tg*8+7
    const int cg   = (warp % WN) * 8 + lane % 8;   // channel group: tile channels cg*8 .. cg*8+7
    const int pa   = w8a8d_swz(tg*8);             // where the group's tokens 0..3 are (4..7 at pa ^ 4)
    const int pb   = w8a8d_swz(cg*8);

    const int G  = K / 32;
    const int nk = K / BK;

    // copies: 16-byte chunk q = tid + NT*s of the A tile is row q % BM, chunk q / BM (k4 rows 4*chunk .. +3); the same for
    // the B tile with BN rows; a warp = 32 consecutive rows of one chunk
    constexpr int CA = BM*BK/16/NT, CB = BN*BK/16/NT;   // chunks per thread
    const int8_t * gA[CA];
    int ca[CA], ka[CA];
#pragma unroll
    for (int s = 0; s < CA; ++s) {
        const int q = tid + NT*s, r = q % BM, c = q / BM;
        gA[s] = qx + ((int64_t) m0 + r) * K + 16*c;
        ca[s] = w8a8d_swz(w8a8_sy_pos(r));
        ka[s] = 4*c;
    }
    const int8_t * gB[CB];
    int cb[CB], kb[CB];
#pragma unroll
    for (int s = 0; s < CB; ++s) {
        const int q = tid + NT*s, r = q % BN, c = q / BN;
        gB[s] = qw + ((int64_t) n0 + r) * K + 16*c;
        cb[s] = w8a8d_swz(r);
        kb[s] = 4*c;
    }
    const uint8_t * gS = sy + (int64_t) tm * G * BM + 16*tid;   // a stage's sy: NG*BM = 256 contiguous bytes, 16 threads

    int4 ra[CA], rb[CB], rs = make_int4(0, 0, 0, 0);
    auto gload = [&](const int kt) {
        const int64_t ko = (int64_t) kt * BK;
#pragma unroll
        for (int s = 0; s < CA; ++s) ra[s] = __ldg((const int4 *) (gA[s] + ko));
#pragma unroll
        for (int s = 0; s < CB; ++s) rb[s] = __ldg((const int4 *) (gB[s] + ko));
        if (tid < NG*BM/16) {
            rs = __ldg((const int4 *) (gS + (int64_t) kt * NG*BM));
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int s = 0; s < CA; ++s) {
            As[buf][ka[s] + 0][ca[s]] = ra[s].x; As[buf][ka[s] + 1][ca[s]] = ra[s].y;
            As[buf][ka[s] + 2][ca[s]] = ra[s].z; As[buf][ka[s] + 3][ca[s]] = ra[s].w;
        }
#pragma unroll
        for (int s = 0; s < CB; ++s) {
            Bs[buf][kb[s] + 0][cb[s]] = rb[s].x; Bs[buf][kb[s] + 1][cb[s]] = rb[s].y;
            Bs[buf][kb[s] + 2][cb[s]] = rb[s].z; Bs[buf][kb[s] + 3][cb[s]] = rb[s].w;
        }
        if (tid < NG*BM/16) {
            *(int4 *) (&Ss[buf][0][0] + 16*tid) = rs;
        }
    };

#if W8A8D_SCALE_F32
    float acc[8][8];
#else
    int   acc[8][8];
#endif
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0;

    gload(0);
    sstore(0);
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {
            gload(kt + 1);
        }
#pragma unroll W8A8D_GU
        for (int gi = 0; gi < NG; ++gi) {
            const uint2 s8 = *(const uint2 *) &Ss[buf][gi][tg*8];   // the 8 tokens' sub-scales, token i = byte i
            int p[8][8];
            auto frag = [&](const int k4, int (&fa)[8], int (&fb)[8]) {
#if W8A8D_WHATIF_NOLDS
                const int row = gi*8 + (k4 & 0);   // what-if: the same words every k4
#else
                const int row = gi*8 + k4;
#endif
                const int4 a0 = *(const int4 *) &As[buf][row][pa];
                const int4 a1 = *(const int4 *) &As[buf][row][pa ^ 4];
                const int4 b0 = *(const int4 *) &Bs[buf][row][pb];
                const int4 b1 = *(const int4 *) &Bs[buf][row][pb ^ 4];
                fa[0] = a0.x; fa[1] = a0.y; fa[2] = a0.z; fa[3] = a0.w; fa[4] = a1.x; fa[5] = a1.y; fa[6] = a1.z; fa[7] = a1.w;
                fb[0] = b0.x; fb[1] = b0.y; fb[2] = b0.z; fb[3] = b0.w; fb[4] = b1.x; fb[5] = b1.y; fb[6] = b1.z; fb[7] = b1.w;
            };
            int fa[2][8], fb[2][8];
#if W8A8D_PREFETCH
            frag(0, fa[0], fb[0]);
#endif
#if W8A8D_K4_UNROLL < 8
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) p[i][j] = 0;
#endif
#pragma unroll W8A8D_KU
            for (int k4 = 0; k4 < 8; ++k4) {
#if W8A8D_PREFETCH
                const int cur = k4 & 1;
                if (k4 < 7) {
                    frag(k4 + 1, fa[cur ^ 1], fb[cur ^ 1]);
                }
#else
                const int cur = 0;
                frag(k4, fa[0], fb[0]);
#endif
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) p[i][j] = __dp4a(fa[cur][i], fb[cur][j], (W8A8D_K4_UNROLL == 8 && k4 == 0) ? 0 : p[i][j]);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned s = ((i < 4 ? s8.x : s8.y) >> (8*(i % 4))) & 0xFF;
#if W8A8D_WHATIF_NOSCALE
                (void) s;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += p[i][j];
#elif W8A8D_SCALE_F32
                const float sf = (float) s;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(sf, (float) p[i][j], acc[i][j]);
#else
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = (int) ((unsigned) acc[i][j] + s * (unsigned) p[i][j]);
#endif
            }
        }
        if (kt + 1 < nk) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

    // epilogue: y = acc * d_t/W8A8_SY_MAX * dW_n (w8a8_gemm's expression and order); token i of the thread is tile row
    // w8a8d_sy_row(tg*8 + i), its channels n0 + cg*8 + j
    float wsc[8];
    {
        const float4 w0 = *(const float4 *) (dw + n0 + cg*8);
        const float4 w1 = *(const float4 *) (dw + n0 + cg*8 + 4);
        wsc[0] = w0.x; wsc[1] = w0.y; wsc[2] = w0.z; wsc[3] = w0.w;
        wsc[4] = w1.x; wsc[5] = w1.y; wsc[6] = w1.z; wsc[7] = w1.w;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int t = m0 + w8a8d_sy_row(tg*8 + i);
        if (t >= T) {
            continue;
        }
        if constexpr (epi == W8A8_EPI_I32) {
            int4 * o = (int4 *) ((int *) dst + (int64_t) t*ldd + n0 + cg*8);
            __stcs(o + 0, make_int4((int) acc[i][0], (int) acc[i][1], (int) acc[i][2], (int) acc[i][3]));
            __stcs(o + 1, make_int4((int) acc[i][4], (int) acc[i][5], (int) acc[i][6], (int) acc[i][7]));
            continue;
        }
        const float st = dt[t] * (1.0f / (float) W8A8_SY_MAX);
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            y[j] = (float) acc[i][j] * st * wsc[j];
        }
        if constexpr (epi == W8A8_EPI_F32) {
            float4 * o = (float4 *) ((float *) dst + (int64_t) t*ldd + n0 + cg*8);
            __stcs(o + 0, make_float4(y[0], y[1], y[2], y[3]));
            __stcs(o + 1, make_float4(y[4], y[5], y[6], y[7]));
        } else {
            // interleaved [gate | up] channels: 2jj = gate, 2jj + 1 = up of output column (n0 + cg*8)/2 + jj;
            // ggml's swiglu silu(gate) * up, as w8a8_gemm's epilogue
            float g[4];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float y0 = y[2*jj], y1 = y[2*jj + 1];
                g[jj] = __fmul_rn(y0 / (1.0f + expf(-y0)), y1);
            }
            const int64_t col = (n0 + cg*8) / 2;
            if constexpr (epi == W8A8_EPI_GLU_F16) {
                // clamped to the f16 range first, as w8a8_gemm's
                __half2 h[2];
                h[0] = __halves2half2(__float2half(fminf(fmaxf(g[0], -65504.f), 65504.f)), __float2half(fminf(fmaxf(g[1], -65504.f), 65504.f)));
                h[1] = __halves2half2(__float2half(fminf(fmaxf(g[2], -65504.f), 65504.f)), __float2half(fminf(fmaxf(g[3], -65504.f), 65504.f)));
                __stcs((int2 *) ((__half *) dst + (int64_t) t*ldd + col), *(const int2 *) h);
            } else {
                __stcs((float4 *) ((float *) dst + (int64_t) t*ldd + col), make_float4(g[0], g[1], g[2], g[3]));
            }
        }
    }
#else
    (void) qx; (void) sy; (void) dt; (void) qw; (void) dw; (void) dst; (void) ldd; (void) T; (void) N; (void) K; (void) group;
#endif
}

// the tile order for the dp4a kernel
static inline int w8a8d_default_group(const int T, const int N, const int K) {
    (void) T; (void) K;
    return N >= 16384 ? 4 : -4;
}

static inline cudaError_t w8a8d_gemm_cuda(const int epi, const int8_t * qx, const uint8_t * sy, const float * dt, const int8_t * qw,
                                          const float * dw, void * dst, const int64_t ldd, const int T, const int N, const int K,
                                          cudaStream_t stream, int group = 0) {
    if (group == 0) {
        group = w8a8d_default_group(T, N, K);
    }
    if (K % W8A8D_BK != 0 || N % W8A8D_BN != 0) {
        return cudaErrorInvalidValue;
    }
    const int ntm = (T + W8A8D_BM - 1) / W8A8D_BM;
    const int ntn = N / W8A8D_BN;
    const dim3 grid((unsigned) (ntm * ntn));
    switch (epi) {
        case W8A8_EPI_F32:     w8a8d_gemm<W8A8_EPI_F32    ><<<grid, W8A8D_NTHREADS, 0, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K, group); break;
        case W8A8_EPI_GLU_F16: w8a8d_gemm<W8A8_EPI_GLU_F16><<<grid, W8A8D_NTHREADS, 0, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K, group); break;
        case W8A8_EPI_GLU_F32: w8a8d_gemm<W8A8_EPI_GLU_F32><<<grid, W8A8D_NTHREADS, 0, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K, group); break;
        case W8A8_EPI_I32:     w8a8d_gemm<W8A8_EPI_I32    ><<<grid, W8A8D_NTHREADS, 0, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K, group); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
