#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-dp4a.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Spellcast3D: FlashAttention with INT8 Q.K^T scores for cards without tensor cores (see fattn-tile-dp4a.cuh).
//
// The op in three steps:
// 1. the mean of each K channel over the keys (per head), K minus that mean and Q (times the softmax scale and
//    log2(e)) to INT8 per row (per key / query and head) with one F32 scale each (the row's absolute maximum / 127),
//    into a pool buffer. Subtracting the key mean adds the same constant to every score of a query, which the softmax
//    takes out again (exact without sinks and logit softcap, which therefore keep the F32 tile kernel); it keeps a
//    channel offset that all keys share out of the INT8 range (SageAttention's "smooth K").
// 2. the attention kernel: a block of 8 warps works on 64 queries of one head, 32 keys an iteration. The scores are
//    __dp4a sums over the INT8 rows (32 a score) times the two scales; the online softmax (exp2, the scale holds
//    log2(e)) and the P.V product (V converted to F32 in shared memory, F32 accumulators) as in the F32 tile kernel.
//    A warp owns 8 queries: for the scores a lane takes one query and 8 of the 32 keys (the INT8 P.V kernel below: 2
//    queries x 4 keys), for P.V 4 of the 128 output channels of all 8 queries.
// 3. nothing: the kernel writes the normalised output (one block a tile of queries, no parallel blocks over the keys).

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr int   FDP4A_D      = 128;                 // head size
static constexpr int   FDP4A_NCOLS  = 64;                  // queries a block
static constexpr int   FDP4A_NWARPS = 8;                   // a warp: 8 queries
static constexpr int   FDP4A_NBATCH = 32;                  // keys an iteration
static constexpr int   FDP4A_QKS    = FDP4A_D/4 + 4;       // ints a Q / K row in shared memory (padded: no bank conflicts)
static constexpr int   FDP4A_PS     = FDP4A_NCOLS + 8;     // floats a P row in shared memory (padded likewise)
static constexpr float FDP4A_LOG2E  = 1.4426950408889634f;

static_assert(FDP4A_NCOLS == 8*FDP4A_NWARPS, "a warp works on 8 queries");
static_assert(FDP4A_NBATCH == 32, "a lane takes 8 of the 32 keys of an iteration");

static __device__ __forceinline__ void fdp4a_load4(const float * p, float v[4]) {
    const float4 t = *(const float4 *) p;
    v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
}

static __device__ __forceinline__ void fdp4a_load4(const half * p, float v[4]) {
    const uint2  t = *(const uint2 *) p;
    const float2 a = __half22float2(*(const half2 *) &t.x);
    const float2 b = __half22float2(*(const half2 *) &t.y);
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

static __device__ __forceinline__ float2 fdp4a_load2(const float * p) {
    return *(const float2 *) p;
}

static __device__ __forceinline__ float2 fdp4a_load2(const half * p) {
    return __half22float2(*(const half2 *) p);
}

// The mean of every K channel over the nk keys: one block of 256 threads (64 channel pairs x 4 key rows) a head.
template <typename T>
static __global__ void flash_attn_dp4a_key_mean(
        const char * __restrict__ K, float * __restrict__ mean, const int nk, const int nkh,
        const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    const int hs   = blockIdx.x;                           // sequence*nkh + head
    const int pair = threadIdx.x % (FDP4A_D/2);
    const int row  = threadIdx.x / (FDP4A_D/2);
    const char * base = K + (hs / nkh)*nb3 + (hs % nkh)*nb2;

    float2 sum = make_float2(0.0f, 0.0f);
    for (int t = row; t < nk; t += 4) {
        const float2 v = fdp4a_load2((const T *) (base + t*nb1) + 2*pair);
        sum.x += v.x;
        sum.y += v.y;
    }

    __shared__ float2 sums[4*FDP4A_D/2];
    sums[threadIdx.x] = sum;
    __syncthreads();
    if (row == 0) {
#pragma unroll
        for (int i = 1; i < 4; ++i) {
            sum.x += sums[threadIdx.x + i*(FDP4A_D/2)].x;
            sum.y += sums[threadIdx.x + i*(FDP4A_D/2)].y;
        }
        mean[int64_t(hs)*FDP4A_D + 2*pair + 0] = sum.x / nk;
        mean[int64_t(hs)*FDP4A_D + 2*pair + 1] = sum.y / nk;
    }
}

// One warp a row of 128 values (minus the key mean if given) to INT8 with one F32 scale: the row's absolute maximum
// / 127 times scale_mul. Rows n_rows .. n_rows_pad-1 become zeros with scale 0. Output: [nh*nseq][n_rows_pad][128].
template <typename T>
static __global__ void flash_attn_dp4a_quantize(
        const char * __restrict__ x, const float * __restrict__ mean, int * __restrict__ x8, float * __restrict__ scales,
        const int n_rows, const int n_rows_pad, const int nh, const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const float scale_mul) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int hs   = blockIdx.y;                           // sequence*nh + head
    if (row >= n_rows_pad) {
        return;
    }

    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (row < n_rows) {
        fdp4a_load4((const T *) (x + (hs / nh)*nb3 + (hs % nh)*nb2 + row*nb1) + 4*lane, v);
        if (mean) {
            const float4 m = *(const float4 *) (mean + int64_t(hs)*FDP4A_D + 4*lane);
            v[0] -= m.x; v[1] -= m.y; v[2] -= m.z; v[3] -= m.w;
        }
    }

    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = warp_reduce_max(amax);
    const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;

    int packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = max(-127, min(127, __float2int_rn(v[i]*inv)));
        packed |= (q & 0xFF) << (8*i);
    }
    const int64_t r = int64_t(hs)*n_rows_pad + row;
    x8[r*(FDP4A_D/4) + lane] = packed;
    if (lane == 0) {
        scales[r] = amax/127.0f * scale_mul;
    }
}

template <bool use_mask>
__launch_bounds__(FDP4A_NWARPS*WARP_SIZE, 2)
static __global__ void flash_attn_tile_dp4a(
        const int   * __restrict__ Q8,
        const float * __restrict__ Qs,
        const int   * __restrict__ K8,
        const float * __restrict__ Ks,
        const char  * __restrict__ V,
        const half  * __restrict__ mask,
        const int   * __restrict__ KV_max,
        float       * __restrict__ dst,
        const int nq, const int nq_pad, const int nk, const int nk_pad, const int nh, const int gqa_ratio,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t stride_mask, const int64_t stride_mask3, const int ne31, const int ne33,
        const float max_bias, const float m0, const float m1, const uint32_t n_head_log2) {
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D   = FDP4A_D;
    constexpr int NC  = FDP4A_NCOLS;
    constexpr int NB  = FDP4A_NBATCH;
    constexpr int QKS = FDP4A_QKS;
    constexpr int PS  = FDP4A_PS;
    constexpr int NT  = FDP4A_NWARPS*WARP_SIZE;

    __shared__ __align__(16) int   Q8s[NC*QKS];            // the block's queries, INT8
    __shared__ __align__(16) int   K8s[NB*QKS];            // the iteration's keys, INT8
    __shared__ __align__(16) float Vs[NB*D];               // the iteration's values, F32
    __shared__ __align__(16) float Ps[NB*PS];              // softmax numerators [key][query]
    __shared__               float Kss[NB];                // the keys' scales

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int col0 = blockIdx.x*NC;
    const int hs   = blockIdx.z;                           // sequence*nh + head
    const int head = hs % nh;
    const int seq  = hs / nh;
    const int nkh  = nh / gqa_ratio;
    const int khs  = seq*nkh + head/gqa_ratio;

    // Scores: the lane takes query jc of the block and the keys 4*i + g (i = 0..7) of an iteration.
    const int c  = lane % 8;
    const int g  = lane / 8;
    const int jc = w*8 + c;

    const int   * Q8h = Q8 + (int64_t(hs)*nq_pad + col0)*(D/4);
    const int   * K8h = K8 + int64_t(khs)*nk_pad*(D/4);
    const float * Ksh = Ks + int64_t(khs)*nk_pad;
    const char  * Vh  = V + seq*nbv3 + (head/gqa_ratio)*nbv2;

    for (int i = tid; i < NC*(D/16); i += NT) {
        const int row = i / (D/16);
        const int seg = i % (D/16);
        *(int4 *) &Q8s[row*QKS + seg*4] = *(const int4 *) &Q8h[row*(D/4) + seg*4];
    }
    const float qs = Qs[int64_t(hs)*nq_pad + col0 + jc];   // includes the softmax scale and log2(e)

    const half * maskh = nullptr;
    float slope = 0.0f;
    if (use_mask) {
        maskh = mask + (seq % ne33)*stride_mask3 + int64_t(min(col0 + jc, ne31 - 1))*stride_mask;
        slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1)*FDP4A_LOG2E;
    }

    const int k_max = KV_max ? KV_max[seq*gridDim.x + blockIdx.x] : nk;

    // P.V: the lane accumulates the queries w*8 + 0..7 (all of its warp), channels 4*lane + 0..3.
    float acc[8][4];
#pragma unroll
    for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            acc[cc][d] = 0.0f;
        }
    }
    float m_run = -FLT_MAX/2.0f;                           // running maximum of query jc (log2 units)
    float l_run = 0.0f;                                    // this lane's part of query jc's softmax denominator

    // Registers that carry the next tiles from global to shared memory across a compute phase:
    int4  kreg  = make_int4(0, 0, 0, 0);
    float ksreg = 0.0f;
    uint2 vreg[NB/FDP4A_NWARPS];

    auto load_K = [&](const int k0) {                      // key k0 + tid/8, bytes 16*(tid%8) ..
        kreg = *(const int4 *) &K8h[int64_t(k0 + tid/8)*(D/4) + (tid%8)*4];
        if (tid < NB) {
            ksreg = Ksh[k0 + tid];
        }
    };
    auto store_K = [&]() {
        *(int4 *) &K8s[(tid/8)*QKS + (tid%8)*4] = kreg;
        if (tid < NB) {
            Kss[tid] = ksreg;
        }
    };
    auto load_V = [&](const int k0) {                      // keys k0 + w + 8*m, channels 4*lane ..
#pragma unroll
        for (int m = 0; m < NB/FDP4A_NWARPS; ++m) {
            const int t = k0 + w + FDP4A_NWARPS*m;
            vreg[m] = t < nk ? *(const uint2 *) (Vh + t*nbv1 + lane*(4*sizeof(half))) : make_uint2(0, 0);
        }
    };
    auto store_V = [&]() {
#pragma unroll
        for (int m = 0; m < NB/FDP4A_NWARPS; ++m) {
            const float2 a = __half22float2(*(const half2 *) &vreg[m].x);
            const float2 b = __half22float2(*(const half2 *) &vreg[m].y);
            *(float4 *) &Vs[(w + FDP4A_NWARPS*m)*D + 4*lane] = make_float4(a.x, a.y, b.x, b.y);
        }
    };

    // The tiles travel from global memory through registers one iteration ahead (values) and two ahead (keys):
    if (k_max > 0) {
        load_K(0);
        load_V(0);
        store_K();
    }
    __syncthreads();
    if (NB < k_max) {
        load_K(NB);
    }

    for (int k0 = 0; k0 < k_max; k0 += NB) {
        // Scores of query jc with the keys 4*i + g:
        int sacc[NB/4];
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            sacc[i] = 0;
        }
#pragma unroll
        for (int kc = 0; kc < D/16; ++kc) {
            const int4 q = *(const int4 *) &Q8s[jc*QKS + kc*4];
#pragma unroll
            for (int i = 0; i < NB/4; ++i) {
                const int4 k = *(const int4 *) &K8s[(4*i + g)*QKS + kc*4];
                sacc[i] = ggml_cuda_dp4a(q.x, k.x, sacc[i]);
                sacc[i] = ggml_cuda_dp4a(q.y, k.y, sacc[i]);
                sacc[i] = ggml_cuda_dp4a(q.z, k.z, sacc[i]);
                sacc[i] = ggml_cuda_dp4a(q.w, k.w, sacc[i]);
            }
        }

        float s[NB/4];
        float s_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            const int t = 4*i + g;
            float x = -FLT_MAX; // a key past the end: exp2 gives 0
            if (k0 + t < k_max) {
                x = float(sacc[i]) * (qs*Kss[t]);
                if (use_mask) {
                    x += slope*__half2float(maskh[k0 + t]);
                }
            }
            s[i]  = x;
            s_max = fmaxf(s_max, x);
        }
        s_max = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFF, s_max,  8, WARP_SIZE));
        s_max = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFF, s_max, 16, WARP_SIZE));

        // FATTN_KQ_MAX_OFFSET of the tile kernel in log2 units: 3.
        const float m_new = fmaxf(m_run, s_max + 3.0f);
        const float corr  = exp2f(m_run - m_new);
        m_run = m_new;

        __syncwarp(); // the warp's P.V reads of Ps (previous iteration) are done before it is rewritten
        float l_add = 0.0f;
#pragma unroll
        for (int i = 0; i < NB/4; ++i) {
            const float p = exp2f(s[i] - m_new);
            l_add += p;
            Ps[(4*i + g)*PS + jc] = p;
        }
        l_run = l_run*corr + l_add;

        if (!__all_sync(0xFFFFFFFF, corr == 1.0f)) { // no query's maximum moved: nothing to rescale
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
                const float corr_cc = __shfl_sync(0xFFFFFFFF, corr, cc, WARP_SIZE); // lane cc has query w*8 + cc
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    acc[cc][d] *= corr_cc;
                }
            }
        }

        __syncthreads(); // all warps are done with K8s and with the previous Vs; Ps is complete
        store_V();
        if (k0 + NB < k_max) {
            store_K();
            load_V(k0 + NB);
        }
        if (k0 + 2*NB < k_max) {
            load_K(k0 + 2*NB);
        }
        __syncthreads();

        // P.V: per key one 16-byte value load (all lanes) and two broadcast loads of the warp's 8 numerators.
#pragma unroll 16
        for (int t = 0; t < NB; ++t) {
            const float4 v  = *(const float4 *) &Vs[t*D + 4*lane];
            const float4 p0 = *(const float4 *) &Ps[t*PS + 8*w + 0];
            const float4 p1 = *(const float4 *) &Ps[t*PS + 8*w + 4];
            const float p[8]  = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
            const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    acc[cc][d] += p[cc]*vv[d];
                }
            }
        }
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run,  8, WARP_SIZE);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, WARP_SIZE);

#pragma unroll
    for (int cc = 0; cc < 8; ++cc) {
        const float l   = __shfl_sync(0xFFFFFFFF, l_run, cc, WARP_SIZE);
        const int   col = col0 + 8*w + cc;
        if (col < nq) {
            const float inv = 1.0f/l;
            *(float4 *) &dst[((int64_t(seq)*nq + col)*nh + head)*D + 4*lane] =
                make_float4(acc[cc][0]*inv, acc[cc][1]*inv, acc[cc][2]*inv, acc[cc][3]*inv);
        }
    }
#else
    GGML_UNUSED_VARS(Q8, Qs, K8, Ks, V, mask, KV_max, dst, nq, nq_pad, nk, nk_pad, nh, gqa_ratio, nbv1, nbv2, nbv3,
        stride_mask, stride_mask3, ne31, ne33, max_bias, m0, m1, n_head_log2);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------------
// The P.V product in INT8 too (GGML_CUDA_FATTN_DP4A_PV, see ggml_cuda_fattn_dp4a_pv_mode):
//
// - P: the online softmax's numerators of a query over one 32-key block as unsigned 8-bit integers relative to the
//   block's largest: q = rint(255 * exp2(s - s_max_block)), scale exp2(s_max_block - m) / 255 (m: the running maximum).
//   The largest key of a block is exact (255); a block of small numerators keeps its own relative precision. The
//   softmax denominator sums the same dequantised numerators (scale * sum q), so the weights stay normalised.
// - V: INT8 with one F32 scale per (channel, 32-key block) (the block's absolute maximum of the channel / 127), in a
//   prologue, stored [head][block][key/4][channel] as words of 4 keys of one channel (the dp4a operand along the keys).
// - The product of a block: per output (query, channel) 8 dp4a (u8 x s8, 32 keys) into an int32 that starts at the
//   bit pattern of 1.5 * 2^23, so that the sum (|sum| <= 32*255*127 < 2^22) reads as a float with one FADD (no I2F,
//   a quarter-rate instruction on Pascal); then one FMA with the two scales into the F32 accumulator.
// - Outlier blocks keep the F32 P.V: a block with a token whose V (its absolute maximum over the channels) is above
//   ratio x or below 1/ratio x the mean over the head's keys (Qwen-Image's text tokens: attention sinks with ~1/30 of
//   the typical V and outliers with 7-20x; about 2 of 130 blocks). An INT8 scale shared with such a token is either
//   too coarse for the small one or clipped... measured on dumped Q/K/V of the model (NOTES-fattn-dp4a.md).
//
// Shared memory as the F32 kernel's (+2.3 KB): the F32 value tile and the INT8 one (+ its scales) share a buffer; the
// F32 numerators and the INT8 ones are separate (a warp rewrites its numerators while other warps may still read theirs).

static constexpr int FDP4A_PV_BLK = 32;                    // keys a P / V scale block (= FDP4A_NBATCH)
static constexpr int FDP4A_P8S    = FDP4A_NCOLS + 8;       // words a row of 4 keys of the INT8 numerators (padded: the
                                                           // byte stores of the 2 x 4 score tile hit 16 distinct banks)

// Unroll depths of the INT8 P.V kernel's loops (instruction-cache size on Pascal; see NOTES-fattn-dp4a.md):
#ifndef FDP4A_PV8_SC_UNROLL
#define FDP4A_PV8_SC_UNROLL 8    // the score loop over 8 chunks of 16 channels
#endif
#ifndef FDP4A_PV8_KW_UNROLL
#define FDP4A_PV8_KW_UNROLL 8    // the INT8 P.V loop over the 8 words of 4 keys
#endif
#ifndef FDP4A_PV8_F32_UNROLL
#define FDP4A_PV8_F32_UNROLL 2   // the F32 P.V of outlier blocks (rare) over the 32 keys
#endif
// as constants: a macro inside #pragma unroll is expanded by MSVC's preprocessor but not by GCC's (Linux builds failed)
static constexpr int FDP4A_PV8_SC_U  = FDP4A_PV8_SC_UNROLL;
static constexpr int FDP4A_PV8_KW_U  = FDP4A_PV8_KW_UNROLL;
static constexpr int FDP4A_PV8_F32_U = FDP4A_PV8_F32_UNROLL;

enum fdp4a_pv_mode {
    FDP4A_PV_F32      = 0, // the F32 P.V kernel above
    FDP4A_PV_INT8     = 1, // INT8 P.V, outlier blocks in F32
    FDP4A_PV_INT8_ALL = 2, // research: INT8 P.V for every block
    FDP4A_PV_INT8_F32 = 3, // research: the INT8 kernel with every block flagged (its F32 path everywhere)
};

static __device__ __forceinline__ int fdp4a_dp4a_us(const int a_u8, const int b_s8, const int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_u8), "r"(b_s8), "r"(c));
    return d;
#else
    GGML_UNUSED_VARS(a_u8, b_s8);
    NO_DEVICE_CODE;
    return c;
#endif
}

// V of one (head, 32-key block) a CTA of 8 warps: warp w the keys 4w .. 4w+3, lane the channels 4*lane .. 4*lane+3.
// Out: V8 [hs][block][key/4][128] words (byte i = key 4*(key/4) + i), Vsc [hs][block][128] channel scales, Vstat
// [hs][block][3] = {min, max, sum} over the block's keys of the key's absolute maximum over the channels.
template <typename T>
static __global__ void flash_attn_dp4a_quantize_v(
        const char * __restrict__ V, int * __restrict__ V8, float * __restrict__ Vsc, float * __restrict__ Vstat,
        const int nk, const int nkh, const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int blk  = blockIdx.x;
    const int nblk = gridDim.x;
    const int hs   = blockIdx.y;                           // sequence*nkh + head
    const char * base = V + (hs / nkh)*nb3 + (hs % nkh)*nb2;

    float v[4][4];
    float tmin = FLT_MAX, tmax = 0.0f, tsum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = blk*FDP4A_PV_BLK + 4*w + i;
        if (t < nk) {
            fdp4a_load4((const T *) (base + t*nb1) + 4*lane, v[i]);
        } else {
            v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.0f;
        }
        float am = fmaxf(fmaxf(fabsf(v[i][0]), fabsf(v[i][1])), fmaxf(fabsf(v[i][2]), fabsf(v[i][3])));
        am = warp_reduce_max(am);
        if (t < nk) {
            tmin  = fminf(tmin, am);
            tmax  = fmaxf(tmax, am);
            tsum += am;
        }
    }

    __shared__ float cmax[FDP4A_NWARPS][FDP4A_D];
    __shared__ float stat[FDP4A_NWARPS][3];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        cmax[w][4*lane + j] = fmaxf(fmaxf(fabsf(v[0][j]), fabsf(v[1][j])), fmaxf(fabsf(v[2][j]), fabsf(v[3][j])));
    }
    if (lane == 0) {
        stat[w][0] = tmin;
        stat[w][1] = tmax;
        stat[w][2] = tsum;
    }
    __syncthreads();

    float scale[4];
    int   word[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float c = 0.0f;
#pragma unroll
        for (int ww = 0; ww < FDP4A_NWARPS; ++ww) {
            c = fmaxf(c, cmax[ww][4*lane + j]);
        }
        scale[j] = c/127.0f;
        const float inv = c > 0.0f ? 127.0f/c : 0.0f;
        int packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = max(-127, min(127, __float2int_rn(v[i][j]*inv)));
            packed |= (q & 0xFF) << (8*i);
        }
        word[j] = packed;
    }
    const int64_t hb = int64_t(hs)*nblk + blk;
    *(int4 *) &V8[(hb*(FDP4A_PV_BLK/4) + w)*FDP4A_D + 4*lane] = make_int4(word[0], word[1], word[2], word[3]);
    if (w == 0) {
        *(float4 *) &Vsc[hb*FDP4A_D + 4*lane] = make_float4(scale[0], scale[1], scale[2], scale[3]);
    }
    if (w == 0 && lane == 0) {
        float mn = FLT_MAX, mx = 0.0f, sm = 0.0f;
        for (int ww = 0; ww < FDP4A_NWARPS; ++ww) {
            mn  = fminf(mn, stat[ww][0]);
            mx  = fmaxf(mx, stat[ww][1]);
            sm += stat[ww][2];
        }
        Vstat[hb*3 + 0] = mn;
        Vstat[hb*3 + 1] = mx;
        Vstat[hb*3 + 2] = sm;
    }
}

// The outlier flags of a head's blocks (one CTA of 256 threads a head): 1 = the block keeps the F32 P.V. The mean
// over the keys is summed in a fixed order (deterministic flags).
static __global__ void flash_attn_dp4a_v_flags(
        const float * __restrict__ Vstat, int * __restrict__ flags, const int nblk, const int nk, const float ratio,
        const int mode) {
    const int hs  = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = tid; b < nblk; b += blockDim.x) {
        sum += Vstat[(int64_t(hs)*nblk + b)*3 + 2];
    }
    __shared__ float part[256];
    part[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (tid < s) {
            part[tid] += part[tid + s];
        }
        __syncthreads();
    }
    const float mean = part[0] / nk;
    for (int b = tid; b < nblk; b += blockDim.x) {
        const float mn = Vstat[(int64_t(hs)*nblk + b)*3 + 0];
        const float mx = Vstat[(int64_t(hs)*nblk + b)*3 + 1];
        int flag = mx > ratio*mean || mn*ratio < mean;
        if (mode == FDP4A_PV_INT8_ALL) {
            flag = 0;
        } else if (mode == FDP4A_PV_INT8_F32) {
            flag = 1;
        }
        flags[int64_t(hs)*nblk + b] = flag;
    }
}

template <bool use_mask, typename TV>
__launch_bounds__(FDP4A_NWARPS*WARP_SIZE, 2)
static __global__ void flash_attn_tile_dp4a_pv8(
        const int   * __restrict__ Q8,
        const float * __restrict__ Qs,
        const int   * __restrict__ K8,
        const float * __restrict__ Ks,
        const int   * __restrict__ V8,
        const float * __restrict__ Vsc,
        const int   * __restrict__ Vflag,
        const char  * __restrict__ V,
        const half  * __restrict__ mask,
        const int   * __restrict__ KV_max,
        float       * __restrict__ dst,
        const int nq, const int nq_pad, const int nk, const int nk_pad, const int nh, const int gqa_ratio,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t stride_mask, const int64_t stride_mask3, const int ne31, const int ne33,
        const float max_bias, const float m0, const float m1, const uint32_t n_head_log2) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    constexpr int D   = FDP4A_D;
    constexpr int NC  = FDP4A_NCOLS;
    constexpr int NB  = FDP4A_NBATCH;
    constexpr int QKS = FDP4A_QKS;
    constexpr int PS  = FDP4A_PS;
    constexpr int NT  = FDP4A_NWARPS*WARP_SIZE;
    constexpr int KW  = NB/4;                              // words of 4 keys a block
    static_assert(NB == FDP4A_PV_BLK, "a scale block is one iteration's keys");
    static_assert(NT == KW*D/4, "one int4 of the INT8 value tile a thread");

    __shared__ __align__(16) int   Q8s[NC*QKS];            // the block's queries, INT8
    __shared__ __align__(16) int   K8s[NB*QKS];            // the iteration's keys, INT8
    __shared__ __align__(16) float VA[NB*D];               // the iteration's values: F32 [key][channel] (outlier
                                                           // blocks) or INT8 words [key/4][channel] + scales [channel]
    __shared__ __align__(16) float Ps[NB*PS];              // F32 numerators [key][query] (outlier blocks)
    __shared__ __align__(16) int   P8s[KW*FDP4A_P8S];      // INT8 numerators: words [key/4][query] (rows padded)
    __shared__ __align__(16) float Pss[NC];                // their scales
    __shared__               float Kss[NB];                // the keys' scales
    int   * V8s  = (int *) VA;
    float * Vscs = VA + KW*D;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*WARP_SIZE + lane;

    const int col0 = blockIdx.x*NC;
    const int hs   = blockIdx.z;                           // sequence*nh + head
    const int head = hs % nh;
    const int seq  = hs / nh;
    const int nkh  = nh / gqa_ratio;
    const int khs  = seq*nkh + head/gqa_ratio;
    const int nblk = nk_pad / NB;

    // The scores as a 2 x 4 register tile a lane (the F32 P.V kernel above: 1 query x 8 keys). Per 16-byte chunk of
    // the rows a lane loads 2 query chunks and 4 key chunks (6 LDS.128) for 64 dp4a; 1 x 8 took 9 LDS.128 for 32 dp4a,
    // and on Pascal a 16-byte shared load costs ~2.2 clocks of the SM's one shared-memory pipe even when 4-8 lanes
    // share its address (measured, NOTES-fattn-dp4a.md), so the loop was shared-memory bound.
    //   lane = 8*g + c, c = 2*qp + h: queries jq0 = 8*w + 2*qp and jq0 + 1, keys 8*j + 4*h + g (j = 0..3).
    //   A query's chunk is read by 8 lanes (4 distinct a load: banks 8*qp (+4) of the 36-int rows), a key's by the 4
    //   lanes of one (h, g) (8 distinct a load: banks 16*h + 4*g, all 32 once).
    // The output is bit-identical to the 1 x 8 tile's:
    // - the scores are exact int32 sums times the same two scales in the same expression; a query's maximum (and so
    //   m_run, corr, the numerators and their u8 codes) does not depend on the order of the fmaxf;
    // - every per-query F32 sum keeps today's partition and order: lane (c, g) still owns query jc = 8*w + c and its
    //   keys 4*i + g (i = 0..7) for the softmax denominator. The tile holds half of them (i = 2*j + h), lane ^ 1 the
    //   other half: the u8 codes' integer sum is completed with one shuffle (exact); an outlier block's F32
    //   numerators are summed from shared memory in today's order;
    // - the numerators land in the same [key/4][query] bytes / [key][query] floats.
    const int c  = lane % 8;
    const int g  = lane / 8;
    const int h  = c % 2;
    const int jc = w*8 + c;                                // the query whose denominator this lane keeps
    const int jq0 = w*8 + (c & ~1);                        // the tile's queries: jq0, jq0 + 1 (jc is one of them)
    const int jq1 = jq0 + 1;

    const int   * Q8h = Q8 + (int64_t(hs)*nq_pad + col0)*(D/4);
    const int   * K8h = K8 + int64_t(khs)*nk_pad*(D/4);
    const float * Ksh = Ks + int64_t(khs)*nk_pad;
    const int   * V8h = V8 + int64_t(khs)*nblk*KW*D;
    const float * Vsh = Vsc + int64_t(khs)*nblk*D;
    const int   * Vfh = Vflag + int64_t(khs)*nblk;
    const char  * Vh  = V + seq*nbv3 + (head/gqa_ratio)*nbv2;

    for (int i = tid; i < NC*(D/16); i += NT) {
        const int row = i / (D/16);
        const int seg = i % (D/16);
        *(int4 *) &Q8s[row*QKS + seg*4] = *(const int4 *) &Q8h[row*(D/4) + seg*4];
    }
    const float qs0 = Qs[int64_t(hs)*nq_pad + col0 + jq0]; // include the softmax scale and log2(e)
    const float qs1 = Qs[int64_t(hs)*nq_pad + col0 + jq1];

    const half * maskh0 = nullptr;
    const half * maskh1 = nullptr;
    float slope = 0.0f;
    if (use_mask) {
        maskh0 = mask + (seq % ne33)*stride_mask3 + int64_t(min(col0 + jq0, ne31 - 1))*stride_mask;
        maskh1 = mask + (seq % ne33)*stride_mask3 + int64_t(min(col0 + jq1, ne31 - 1))*stride_mask;
        slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1)*FDP4A_LOG2E;
    }

    const int k_max = KV_max ? KV_max[seq*gridDim.x + blockIdx.x] : nk;

    float acc[8][4];
#pragma unroll
    for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            acc[cc][d] = 0.0f;
        }
    }
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    int4   kreg  = make_int4(0, 0, 0, 0);
    float  ksreg = 0.0f;
    int4   v8reg = make_int4(0, 0, 0, 0);
    float4 vsreg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int    flag_cur = 0, flag_next = 0;

    auto load_K = [&](const int k0) {
        kreg = *(const int4 *) &K8h[int64_t(k0 + tid/8)*(D/4) + (tid%8)*4];
        if (tid < NB) {
            ksreg = Ksh[k0 + tid];
        }
    };
    auto store_K = [&]() {
        *(int4 *) &K8s[(tid/8)*QKS + (tid%8)*4] = kreg;
        if (tid < NB) {
            Kss[tid] = ksreg;
        }
    };
    auto load_V8 = [&](const int k0) {                     // the INT8 tile of block k0/NB, its scales and flag
        const int b = k0 / NB;
        v8reg = *(const int4 *) &V8h[int64_t(b)*KW*D + 4*tid];
        if (tid < D/4) {
            vsreg = *(const float4 *) &Vsh[int64_t(b)*D + 4*tid];
        }
        flag_next = Vfh[b];
    };

    if (k_max > 0) {
        load_K(0);
        load_V8(0);
        store_K();
    }
    __syncthreads();
    flag_cur = flag_next;
    if (NB < k_max) {
        load_K(NB);
    }

    for (int k0 = 0; k0 < k_max; k0 += NB) {
        // the dot products start at the bits of 1.5*2^23 (|sum| <= 128*127*127 < 2^22): one FADD reads one as a
        // float, exactly as I2F would (a quarter-rate instruction on Pascal)
        // the 2 x 4 tile: sacc0 query jq0, sacc1 query jq1, [j] key 8*j + 4*h + g
        constexpr int TK = NB/8;                           // keys a lane: 4
        int sacc0[TK];
        int sacc1[TK];
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            sacc0[j] = 0x4B400000;
            sacc1[j] = 0x4B400000;
        }
#pragma unroll FDP4A_PV8_SC_U
        for (int kc = 0; kc < D/16; ++kc) {
            const int4 qa = *(const int4 *) &Q8s[jq0*QKS + kc*4];
            const int4 qb = *(const int4 *) &Q8s[jq1*QKS + kc*4];
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                const int4 k = *(const int4 *) &K8s[(8*j + 4*h + g)*QKS + kc*4];
                sacc0[j] = ggml_cuda_dp4a(qa.x, k.x, sacc0[j]);
                sacc0[j] = ggml_cuda_dp4a(qa.y, k.y, sacc0[j]);
                sacc0[j] = ggml_cuda_dp4a(qa.z, k.z, sacc0[j]);
                sacc0[j] = ggml_cuda_dp4a(qa.w, k.w, sacc0[j]);
                sacc1[j] = ggml_cuda_dp4a(qb.x, k.x, sacc1[j]);
                sacc1[j] = ggml_cuda_dp4a(qb.y, k.y, sacc1[j]);
                sacc1[j] = ggml_cuda_dp4a(qb.z, k.z, sacc1[j]);
                sacc1[j] = ggml_cuda_dp4a(qb.w, k.w, sacc1[j]);
            }
        }

        float s0[TK];
        float s1[TK];
        float s_max0 = -FLT_MAX;
        float s_max1 = -FLT_MAX;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            const int t = 8*j + 4*h + g;
            float x0 = -FLT_MAX; // a key past the end: exp2 gives 0
            float x1 = -FLT_MAX;
            if (k0 + t < k_max) {
                const float ks = Kss[t];
                x0 = (__int_as_float(sacc0[j]) - 12582912.0f) * (qs0*ks);
                x1 = (__int_as_float(sacc1[j]) - 12582912.0f) * (qs1*ks);
                if (use_mask) {
                    x0 += slope*__half2float(maskh0[k0 + t]);
                    x1 += slope*__half2float(maskh1[k0 + t]);
                }
            }
            s0[j]  = x0;
            s1[j]  = x1;
            s_max0 = fmaxf(s_max0, x0);
            s_max1 = fmaxf(s_max1, x1);
        }
        // the maxima over the 32 keys: this lane's query jc with lane ^ 1's half of it, then over g; then the
        // tile's other query from lane ^ 1 (whose query it is)
        float s_max = h ? s_max1 : s_max0;
        s_max = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFF, h ? s_max0 : s_max1, 1, WARP_SIZE));
        s_max = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFF, s_max,  8, WARP_SIZE));
        s_max = fmaxf(s_max, __shfl_xor_sync(0xFFFFFFFF, s_max, 16, WARP_SIZE));
        {
            const float s_max_x = __shfl_xor_sync(0xFFFFFFFF, s_max, 1, WARP_SIZE);
            s_max0 = h ? s_max_x : s_max;
            s_max1 = h ? s_max : s_max_x;
        }

        const float m_new = fmaxf(m_run, s_max + 3.0f);    // query jc
        const float corr  = exp2f(m_run - m_new);
        m_run = m_new;

        __syncwarp(); // the warp's P.V reads of its numerators (previous iteration) are done before they are rewritten
        if (flag_cur) {
            const float m_new_x = __shfl_xor_sync(0xFFFFFFFF, m_new, 1, WARP_SIZE);
            const float m_new0  = h ? m_new_x : m_new;
            const float m_new1  = h ? m_new : m_new_x;
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                const int t = 8*j + 4*h + g;
                Ps[t*PS + jq0] = exp2f(s0[j] - m_new0);
                Ps[t*PS + jq1] = exp2f(s1[j] - m_new1);
            }
            __syncwarp();
            // the denominator of query jc over its keys 4*i + g in today's order (an F32 sum: the order matters)
            float l_add = 0.0f;
#pragma unroll
            for (int i = 0; i < NB/4; ++i) {
                l_add += Ps[(4*i + g)*PS + jc];
            }
            l_run = l_run*corr + l_add;
        } else {
            // numerators relative to the block's largest: 0 .. 255 (an all-masked block: scale 0)
            // q = rint(255 e) as the low byte of the bits of 1.5*2^23 + 255 e (FFMA instead of FMUL + F2I, a
            // quarter-rate conversion on Pascal); a NaN of an all-masked block (byte 255) meets the scale 0.
            const float sc = exp2f(s_max - m_new) * (1.0f/255.0f);
            unsigned int qbits0 = 0;
            unsigned int qbits1 = 0;
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                const int i = 2*j + h;                     // key 4*i + g
                const unsigned int qb0 = __float_as_uint(fmaf(255.0f, exp2f(s0[j] - s_max0), 12582912.0f));
                const unsigned int qb1 = __float_as_uint(fmaf(255.0f, exp2f(s1[j] - s_max1), 12582912.0f));
                qbits0 += qb0;
                qbits1 += qb1;
                ((unsigned char *) P8s)[4*(i*FDP4A_P8S + jq0) + g] = (unsigned char) qb0;
                ((unsigned char *) P8s)[4*(i*FDP4A_P8S + jq1) + g] = (unsigned char) qb1;
            }
            // query jc's 8 keys 4*i + g: 4 here, 4 on lane ^ 1 (integer sums: exact in any order)
            const unsigned int qbits = (h ? qbits1 : qbits0) +
                (unsigned int) __shfl_xor_sync(0xFFFFFFFF, h ? qbits0 : qbits1, 1, WARP_SIZE);
            const int qsum = int(qbits - (NB/4)*0x4B400000u);
            l_run = l_run*corr + sc*float(qsum);
            if (g == 0) {
                Pss[jc] = sc;
            }
        }

        if (!__all_sync(0xFFFFFFFF, corr == 1.0f)) {
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
                const float corr_cc = __shfl_sync(0xFFFFFFFF, corr, cc, WARP_SIZE);
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    acc[cc][d] *= corr_cc;
                }
            }
        }

        __syncthreads(); // all warps are done with K8s and with the previous value tile; the numerators are complete
        if (flag_cur) {
            // an outlier block: its values in F32 straight from V (rare: no prefetch)
#pragma unroll
            for (int m = 0; m < NB/FDP4A_NWARPS; ++m) {
                const int t = k0 + w + FDP4A_NWARPS*m;
                float vv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (t < nk) {
                    fdp4a_load4((const TV *) (Vh + t*nbv1) + 4*lane, vv);
                }
                *(float4 *) &VA[(w + FDP4A_NWARPS*m)*D + 4*lane] = make_float4(vv[0], vv[1], vv[2], vv[3]);
            }
        } else {
            *(int4 *) &V8s[4*tid] = v8reg;
            if (tid < D/4) {
                *(float4 *) &Vscs[4*tid] = vsreg;
            }
        }
        if (k0 + NB < k_max) {
            store_K();
            load_V8(k0 + NB);
        }
        if (k0 + 2*NB < k_max) {
            load_K(k0 + 2*NB);
        }
        __syncthreads();

        if (flag_cur) {
#pragma unroll FDP4A_PV8_F32_U
            for (int t = 0; t < NB; ++t) {
                const float4 v  = *(const float4 *) &VA[t*D + 4*lane];
                const float4 p0 = *(const float4 *) &Ps[t*PS + 8*w + 0];
                const float4 p1 = *(const float4 *) &Ps[t*PS + 8*w + 4];
                const float p[8]  = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        acc[cc][d] += p[cc]*vv[d];
                    }
                }
            }
        } else {
            // per output 8 dp4a over the block's 32 keys into an int32 that starts at the bits of 1.5*2^23
            int isum[8][4];
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    isum[cc][d] = 0x4B400000;
                }
            }
#pragma unroll FDP4A_PV8_KW_U
            for (int kw = 0; kw < KW; ++kw) {
                const int4 v  = *(const int4 *) &V8s[kw*D + 4*lane];
                const int4 p0 = *(const int4 *) &P8s[kw*FDP4A_P8S + 8*w + 0];
                const int4 p1 = *(const int4 *) &P8s[kw*FDP4A_P8S + 8*w + 4];
                const int p[8]  = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
                const int vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        isum[cc][d] = fdp4a_dp4a_us(p[cc], vv[d], isum[cc][d]);
                    }
                }
            }
            const float4 ps0 = *(const float4 *) &Pss[8*w + 0];
            const float4 ps1 = *(const float4 *) &Pss[8*w + 4];
            const float4 vs  = *(const float4 *) &Vscs[4*lane];
            const float ps[8]  = {ps0.x, ps0.y, ps0.z, ps0.w, ps1.x, ps1.y, ps1.z, ps1.w};
            const float vsa[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    acc[cc][d] = fmaf(ps[cc]*vsa[d], __int_as_float(isum[cc][d]) - 12582912.0f, acc[cc][d]);
                }
            }
        }
        flag_cur = flag_next;
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run,  8, WARP_SIZE);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, WARP_SIZE);

#pragma unroll
    for (int cc = 0; cc < 8; ++cc) {
        const float l   = __shfl_sync(0xFFFFFFFF, l_run, cc, WARP_SIZE);
        const int   col = col0 + 8*w + cc;
        if (col < nq) {
            const float inv = 1.0f/l;
            *(float4 *) &dst[((int64_t(seq)*nq + col)*nh + head)*D + 4*lane] =
                make_float4(acc[cc][0]*inv, acc[cc][1]*inv, acc[cc][2]*inv, acc[cc][3]*inv);
        }
    }
#else
    GGML_UNUSED_VARS(Q8, Qs, K8, Ks, V8, Vsc, Vflag, V, mask, KV_max, dst, nq, nq_pad, nk, nk_pad, nh, gqa_ratio,
        nbv1, nbv2, nbv3, stride_mask, stride_mask3, ne31, ne33, max_bias, m0, m1, n_head_log2);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && __CUDA_ARCH__ >= 610
}

// GGML_CUDA_FATTN_DP4A_PV: f32 = the F32 P.V kernel; int8 = INT8 P.V with outlier blocks in F32 (default); research:
// int8-all (no outlier blocks), int8-f32 (the INT8 kernel's F32 path for every block). GGML_CUDA_FATTN_DP4A_PV_RATIO
// (default 4): the outlier ratio.
static int ggml_cuda_fattn_dp4a_pv_mode() {
    static const int mode = [] {
        const char * value = getenv("GGML_CUDA_FATTN_DP4A_PV");
        if (value == nullptr) {
            return int(FDP4A_PV_INT8);
        }
        if (strcmp(value, "f32") == 0) {
            return int(FDP4A_PV_F32);
        }
        if (strcmp(value, "int8-all") == 0) {
            return int(FDP4A_PV_INT8_ALL);
        }
        if (strcmp(value, "int8-f32") == 0) {
            return int(FDP4A_PV_INT8_F32);
        }
        return int(FDP4A_PV_INT8);
    }();
    return mode;
}

static float ggml_cuda_fattn_dp4a_pv_ratio() {
    static const float ratio = [] {
        const char * value = getenv("GGML_CUDA_FATTN_DP4A_PV_RATIO");
        const float r = value ? float(atof(value)) : 4.0f;
        return r > 1.0f ? r : 4.0f;
    }();
    return ratio;
}

static bool ggml_cuda_fattn_dp4a_env_off() {
    static const bool off = [] {
        const char * value = getenv("GGML_CUDA_FATTN_DP4A");
        return value != nullptr && strcmp(value, "off") == 0;
    }();
    return off;
}

bool ggml_cuda_flash_attn_ext_tile_dp4a_use(const int device, const ggml_tensor * dst) {
    if (ggml_cuda_fattn_dp4a_env_off()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_DP4A || cc >= GGML_CUDA_CC_VOLTA ||
            ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (sinks != nullptr || logit_softcap != 0.0f) {
        return false; // the key-mean shift is not softmax-invariant with these
    }
    if (Q->ne[0] != FDP4A_D || K->ne[0] != FDP4A_D || V->ne[0] != FDP4A_D || Q->ne[1] < 1024) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) || (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32)) {
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1)) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3] || Q->ne[3] != V->ne[3] || K->ne[1] != V->ne[1] ||
            K->ne[2] != V->ne[2] || Q->ne[2]*Q->ne[3] > 65535) {
        return false;
    }
    // The kernels read rows of 4 elements a lane: aligned rows.
    auto aligned = [](const ggml_tensor * t, const size_t alignment) {
        return t->nb[0] == ggml_type_size(t->type) && uintptr_t(t->data) % alignment == 0 &&
            t->nb[1] % alignment == 0 && t->nb[2] % alignment == 0 && t->nb[3] % alignment == 0;
    };
    if (!aligned(Q, 16) || !aligned(K, K->type == GGML_TYPE_F16 ? 8 : 16) || !aligned(V, V->type == GGML_TYPE_F16 ? 8 : 16)) {
        return false;
    }
    // A non-F16 V is converted to F16 like the tile kernel does; not when it is a view of K (MLA), whose F16 copy this
    // path does not make.
    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    if (V->type != GGML_TYPE_F16 && V_is_K_view) {
        return false;
    }
    return true;
}

static void ggml_cuda_flash_attn_ext_tile_dp4a_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Research hook (inactive unless GGML_CUDA_FATTN_DUMP=<dir> is set): the inputs and the output of chosen calls of this
// kernel for offline precision studies. Calls are counted from 0 over this kernel's calls; GGML_CUDA_FATTN_DUMP_CALLS
// lists the ones to write (comma-separated; none listed: only the index), GGML_CUDA_FATTN_DUMP_QSTRIDE keeps every
// n-th query (default 8). Files <dir>/fa<call>.bin: int32 header {0x34414446, D, nq_sub, nk, nh, nkh, qstride,
// K type, V type} + float scale, then Q [nh][nq_sub][D] f32, K [nkh][nk][D] and V [nkh][nk][D] in their types
// (f16 / f32), then the output [nh][nq_sub][D] f32. Synchronous; one sequence only.
static int fdp4a_dump_pending = -1;
static void fdp4a_dump_gather(const ggml_tensor * t, const int64_t row_stride, FILE * f) {
    const int64_t bytes = (t->ne[3] - 1)*t->nb[3] + (t->ne[2] - 1)*t->nb[2] + (t->ne[1] - 1)*t->nb[1] + t->ne[0]*t->nb[0];
    std::vector<char> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), t->data, bytes, cudaMemcpyDeviceToHost));
    const size_t row = t->ne[0]*ggml_type_size(t->type);
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
        for (int64_t i1 = 0; i1 < t->ne[1]; i1 += row_stride) {
            fwrite(h.data() + i2*t->nb[2] + i1*t->nb[1], 1, row, f);
        }
    }
}
static void fdp4a_dump_in(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    static const char * dir = getenv("GGML_CUDA_FATTN_DUMP");
    if (dir == nullptr) {
        return;
    }
    static std::vector<int> calls = [] {
        std::vector<int> v;
        const char * e = getenv("GGML_CUDA_FATTN_DUMP_CALLS");
        for (const char * q = e; q && *q; ) {
            char * end = nullptr;
            v.push_back((int) strtol(q, &end, 10));
            q = (end && *end == ',') ? end + 1 : end;
        }
        return v;
    }();
    static const int qstride = getenv("GGML_CUDA_FATTN_DUMP_QSTRIDE") ? atoi(getenv("GGML_CUDA_FATTN_DUMP_QSTRIDE")) : 8;
    static int serial = 0;
    const int call = serial++;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    fprintf(stderr, "fattn-dp4a dump: call %d, Q %lld x %lld x %lld, K %lld x %lld (%s), V %s\n", call,
        (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3], (long long) K->ne[1], (long long) K->ne[2],
        ggml_type_name(K->type), ggml_type_name(V->type));
    if (std::find(calls.begin(), calls.end(), call) == calls.end() || Q->ne[3] != 1) {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/fa%04d.bin", dir, call);
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    const int32_t nq_sub = (int32_t) ((Q->ne[1] + qstride - 1) / qstride);
    const int32_t header[9] = { 0x34414446, (int32_t) Q->ne[0], nq_sub, (int32_t) K->ne[1], (int32_t) Q->ne[2],
        (int32_t) K->ne[2], qstride, (int32_t) K->type, (int32_t) V->type };
    fwrite(header, sizeof(header), 1, f);
    fwrite((const float *) dst->op_params + 0, sizeof(float), 1, f);
    fdp4a_dump_gather(Q, qstride, f);
    fdp4a_dump_gather(K, 1, f);
    fdp4a_dump_gather(V, 1, f);
    fclose(f);
    fdp4a_dump_pending = call;
}
static void fdp4a_dump_out(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (fdp4a_dump_pending < 0) {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/fa%04d.bin", getenv("GGML_CUDA_FATTN_DUMP"), fdp4a_dump_pending);
    fdp4a_dump_pending = -1;
    static const int qstride = getenv("GGML_CUDA_FATTN_DUMP_QSTRIDE") ? atoi(getenv("GGML_CUDA_FATTN_DUMP_QSTRIDE")) : 8;
    FILE * f = fopen(path, "ab");
    if (f == nullptr) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    // dst is [D][nh][nq] (a query's heads adjacent): out [nh][nq_sub][D]
    const int64_t D = dst->ne[0], nh = dst->ne[1], nq = dst->ne[2];
    std::vector<float> h(D*nh*nq);
    CUDA_CHECK(cudaMemcpy(h.data(), dst->data, h.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (int64_t head = 0; head < nh; ++head) {
        for (int64_t q = 0; q < nq; q += qstride) {
            fwrite(h.data() + (q*nh + head)*D, sizeof(float), D, f);
        }
    }
    fclose(f);
}

void ggml_cuda_flash_attn_ext_tile_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fdp4a_dump_in(ctx, dst);
    ggml_cuda_flash_attn_ext_tile_dp4a_impl(ctx, dst);
    fdp4a_dump_out(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_tile_dp4a_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(ggml_cuda_flash_attn_ext_tile_dp4a_use(ctx.device, dst));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int nq   = Q->ne[1];
    const int nh   = Q->ne[2];
    const int nseq = Q->ne[3];
    const int nk   = K->ne[1];
    const int nkh  = K->ne[2];
    const int gqa_ratio = nh / nkh;
    const int nq_pad   = GGML_PAD(nq, FDP4A_NCOLS);
    const int nk_pad   = GGML_PAD(nk, FDP4A_NBATCH);
    const int ntiles_x = nq_pad / FDP4A_NCOLS;

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int pv_mode = ggml_cuda_fattn_dp4a_pv_mode();
    const bool pv8 = pv_mode != FDP4A_PV_F32;

    // V as F16 the way the tile kernel takes it: converted into the op's F16 extra space (reserved by
    // ggml_cuda_flash_attn_ext_get_alloc_size for the tile kernel). The INT8 P.V kernel reads V in its own type.
    const char * V_data = (const char *) V->data;
    int64_t nb21 = V->nb[1];
    int64_t nb22 = V->nb[2];
    int64_t nb23 = V->nb[3];
    if (V->type != GGML_TYPE_F16 && !pv8) {
        const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra = ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
        GGML_ASSERT(f16_extra.V != 0);
        half * V_f16 = (half *) f16_extra.V;
        const int64_t bs = ggml_blck_size(V->type);
        const int64_t ts = ggml_type_size(V->type);
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V->data, V_f16, ggml_nelements(V), stream);
            nb21 = nb21*bs*sizeof(half)/ts;
            nb22 = nb22*bs*sizeof(half)/ts;
            nb23 = nb23*bs*sizeof(half)/ts;
        } else {
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            to_fp16(V->data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], nb21/ts, nb22/ts, nb23/ts, stream);
            nb21 = V->ne[0]*sizeof(half);
            nb22 = V->ne[1]*nb21;
            nb23 = V->ne[2]*nb22;
        }
        V_data = (const char *) V_f16;
    }

    // Scratch: INT8 Q and K with their scales, the key mean, the KV_max list of a mask.
    const bool use_KV_max = mask != nullptr && K->ne[1] % FATTN_KQ_STRIDE == 0 && mask->ne[3] == nseq &&
        mask->ne[1] >= int64_t(ntiles_x)*FDP4A_NCOLS;
    size_t off = 0;
    auto reserve = [&off](const size_t bytes) {
        const size_t at = off;
        off += GGML_PAD(bytes, 256);
        return at;
    };
    const size_t off_q8 = reserve(size_t(nh)*nseq*nq_pad*FDP4A_D);
    const size_t off_qs = reserve(size_t(nh)*nseq*nq_pad*sizeof(float));
    const size_t off_k8 = reserve(size_t(nkh)*nseq*nk_pad*FDP4A_D);
    const size_t off_ks = reserve(size_t(nkh)*nseq*nk_pad*sizeof(float));
    const size_t off_km = reserve(size_t(nkh)*nseq*FDP4A_D*sizeof(float));
    const size_t off_kv = reserve(use_KV_max ? size_t(ntiles_x)*nseq*sizeof(int) : 0);
    const int nblk = nk_pad / FDP4A_PV_BLK;
    const size_t off_v8 = reserve(pv8 ? size_t(nkh)*nseq*nk_pad*FDP4A_D : 0);
    const size_t off_vs = reserve(pv8 ? size_t(nkh)*nseq*nblk*FDP4A_D*sizeof(float) : 0);
    const size_t off_vt = reserve(pv8 ? size_t(nkh)*nseq*nblk*3*sizeof(float) : 0);
    const size_t off_vf = reserve(pv8 ? size_t(nkh)*nseq*nblk*sizeof(int) : 0);
    ggml_cuda_pool_alloc<char> scratch(pool, off);
    int   * Q8 = (int   *) (scratch.get() + off_q8);
    float * Qs = (float *) (scratch.get() + off_qs);
    int   * K8 = (int   *) (scratch.get() + off_k8);
    float * Ks = (float *) (scratch.get() + off_ks);
    float * Km = (float *) (scratch.get() + off_km);
    int   * KV_max = use_KV_max ? (int *) (scratch.get() + off_kv) : nullptr;

    constexpr int rows_per_block = 8; // 256 threads, a warp a row
    if (K->type == GGML_TYPE_F16) {
        flash_attn_dp4a_key_mean<half><<<nkh*nseq, 4*FDP4A_D/2, 0, stream>>>(
            (const char *) K->data, Km, nk, nkh, K->nb[1], K->nb[2], K->nb[3]);
        flash_attn_dp4a_quantize<half><<<dim3(nk_pad/rows_per_block, nkh*nseq), rows_per_block*WARP_SIZE, 0, stream>>>(
            (const char *) K->data, Km, K8, Ks, nk, nk_pad, nkh, K->nb[1], K->nb[2], K->nb[3], 1.0f);
    } else {
        flash_attn_dp4a_key_mean<float><<<nkh*nseq, 4*FDP4A_D/2, 0, stream>>>(
            (const char *) K->data, Km, nk, nkh, K->nb[1], K->nb[2], K->nb[3]);
        flash_attn_dp4a_quantize<float><<<dim3(nk_pad/rows_per_block, nkh*nseq), rows_per_block*WARP_SIZE, 0, stream>>>(
            (const char *) K->data, Km, K8, Ks, nk, nk_pad, nkh, K->nb[1], K->nb[2], K->nb[3], 1.0f);
    }
    flash_attn_dp4a_quantize<float><<<dim3(nq_pad/rows_per_block, nh*nseq), rows_per_block*WARP_SIZE, 0, stream>>>(
        (const char *) Q->data, nullptr, Q8, Qs, nq, nq_pad, nh, Q->nb[1], Q->nb[2], Q->nb[3], scale*FDP4A_LOG2E);
    CUDA_CHECK(cudaGetLastError());

    if (use_KV_max) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);
        flash_attn_mask_to_KV_max<FDP4A_NCOLS><<<dim3(ntiles_x, nseq, 1), FATTN_KQ_STRIDE/2, 0, stream>>>(
            (const half2 *) mask->data, KV_max, K->ne[1] / FATTN_KQ_STRIDE, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(nh))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const dim3 grid(ntiles_x, 1, nh*nseq);
    const dim3 block(WARP_SIZE, FDP4A_NWARPS, 1);
    const half * mask_data      = mask ? (const half *) mask->data : nullptr;
    const int64_t stride_mask   = mask ? mask->nb[1] / sizeof(half) : 0;
    const int64_t stride_mask3  = mask ? mask->nb[3] / sizeof(half) : 0;
    const int ne31 = mask ? mask->ne[1] : 1;
    const int ne33 = mask ? mask->ne[3] : 1;
    if (pv8) {
        int   * V8    = (int   *) (scratch.get() + off_v8);
        float * Vsc   = (float *) (scratch.get() + off_vs);
        float * Vstat = (float *) (scratch.get() + off_vt);
        int   * Vflag = (int   *) (scratch.get() + off_vf);
        const dim3 vgrid(nblk, nkh*nseq, 1);
        if (V->type == GGML_TYPE_F16) {
            flash_attn_dp4a_quantize_v<half><<<vgrid, block, 0, stream>>>(
                V_data, V8, Vsc, Vstat, nk, nkh, nb21, nb22, nb23);
        } else {
            flash_attn_dp4a_quantize_v<float><<<vgrid, block, 0, stream>>>(
                V_data, V8, Vsc, Vstat, nk, nkh, nb21, nb22, nb23);
        }
        flash_attn_dp4a_v_flags<<<nkh*nseq, 256, 0, stream>>>(Vstat, Vflag, nblk, nk, ggml_cuda_fattn_dp4a_pv_ratio(), pv_mode);
        CUDA_CHECK(cudaGetLastError());
#define FDP4A_PV8_LAUNCH(use_mask, TV) \
        flash_attn_tile_dp4a_pv8<use_mask, TV><<<grid, block, 0, stream>>>( \
            Q8, Qs, K8, Ks, V8, Vsc, Vflag, V_data, use_mask ? mask_data : nullptr, use_mask ? KV_max : nullptr, \
            (float *) dst->data, nq, nq_pad, nk, nk_pad, nh, gqa_ratio, nb21, nb22, nb23, \
            use_mask ? stride_mask : 0, use_mask ? stride_mask3 : 0, use_mask ? ne31 : 1, use_mask ? ne33 : 1, \
            max_bias, m0, m1, n_head_log2)
        if (V->type == GGML_TYPE_F16) {
            if (mask) {
                FDP4A_PV8_LAUNCH(true, half);
            } else {
                FDP4A_PV8_LAUNCH(false, half);
            }
        } else {
            if (mask) {
                FDP4A_PV8_LAUNCH(true, float);
            } else {
                FDP4A_PV8_LAUNCH(false, float);
            }
        }
#undef FDP4A_PV8_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mask) {
        flash_attn_tile_dp4a<true><<<grid, block, 0, stream>>>(
            Q8, Qs, K8, Ks, V_data, mask_data, KV_max, (float *) dst->data,
            nq, nq_pad, nk, nk_pad, nh, gqa_ratio, nb21, nb22, nb23,
            stride_mask, stride_mask3, ne31, ne33, max_bias, m0, m1, n_head_log2);
    } else {
        flash_attn_tile_dp4a<false><<<grid, block, 0, stream>>>(
            Q8, Qs, K8, Ks, V_data, nullptr, nullptr, (float *) dst->data,
            nq, nq_pad, nk, nk_pad, nh, gqa_ratio, nb21, nb22, nb23,
            0, 0, 1, 1, max_bias, m0, m1, n_head_log2);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_flash_attn_ext_tile_dp4a_use(const int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_flash_attn_ext_tile_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("the dp4a flash attention kernel is CUDA only");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
