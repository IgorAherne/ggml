#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-dp4a.cuh"

#include <cstdlib>
#include <cstring>

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
//    A warp owns 8 queries: for the scores a lane takes one query and 8 of the 32 keys, for P.V 4 of the 128 output
//    channels of all 8 queries.
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

void ggml_cuda_flash_attn_ext_tile_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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

    // V as F16 the way the tile kernel takes it: converted into the op's F16 extra space (reserved by
    // ggml_cuda_flash_attn_ext_get_alloc_size for the tile kernel).
    const char * V_data = (const char *) V->data;
    int64_t nb21 = V->nb[1];
    int64_t nb22 = V->nb[2];
    int64_t nb23 = V->nb[3];
    if (V->type != GGML_TYPE_F16) {
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
