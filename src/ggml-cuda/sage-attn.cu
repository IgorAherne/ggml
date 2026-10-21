#include "sage-attn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDACC_VER_MAJOR__ >= 12

#include <climits>
#include <cstdlib>
#include <cstring>
#include "convert.cuh"
#include "sageattention/attn.cuh"
#include "sageattention/attn_fp8.cuh"

static constexpr int sage_cuda_version = __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10;

static ggml_sage_attn_mode sage_mode(int device, const ggml_tensor* op) {
    const auto requested = static_cast<ggml_sage_attn_mode>(op->op_params[1]);
    if (requested != GGML_SAGE_ATTN_AUTO) {
        return requested;
    }
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10) >= 12040
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc >= GGML_CUDA_CC_ADA_LOVELACE && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_ADA_LOVELACE) {
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10) >= 12080
        if (cc != GGML_CUDA_CC_HOPPER) {
            return GGML_SAGE_ATTN_2_PLUS_PLUS;
        }
#endif
        return GGML_SAGE_ATTN_2;
    }
#endif
    return GGML_SAGE_ATTN_AUTO;
}

// The kernel an op runs (Spellcast3D):
// - SAGE_PATH_TUNED: per-warp INT8 Q (16 rows) / K (64 keys, smoothed by the key mean), FP16 P.V accumulated in FP16
//   over each 64-key block and then in FP32 (SageAttention's "fp16+fp32", its instruction buffer), the tuned launch of
//   sageattention/attn.cuh (two blocks an SM, the P.V step one tile at a time) and the F32 output written straight
//   from the accumulators. The default of GGML_SAGE_ATTN_AUTO for head 128 on every NVIDIA card from the RTX 30
//   series (cc 8.6) on: on GeForce and workstation cards an FP16 MMA with FP16 accumulation runs at twice the rate of
//   one with FP32 accumulation, so this P.V step is the fast one there (RTX 3060, Qwen-Image 2.1's attention, 4,096
//   queries x 4,152 keys x 32 heads: see the measurements in sageattention/README.md). It is also the default on the
//   RTX 40 / 50 series instead of the FP8 kernels below: the same numbers on every card, the ones Spellcast3D checked
//   in pictures (the FP8 P.V of SageAttention2 / 2++ was not checked: no such card). GGML_CUDA_SAGE=fp8 / fp8pp in the
//   environment asks for those where the card has them, GGML_CUDA_SAGE=sm80 for the kernel below.
// - SAGE_PATH_SM80: upstream's SM80 compatibility path (per-warp INT8 Q of 32 rows, FP16 V, FP32 accumulation, FP16
//   output converted to F32): GGML_SAGE_ATTN_AUTO for head 64, and on cc 8.0 - 8.5 (A100 / A30: FP32 accumulation
//   costs nothing there).
// - SAGE_PATH_FP8 / SAGE_PATH_FP8_PLUS_PLUS: upstream's SageAttention2 / 2++ (GGML_SAGE_ATTN_2 / _2_PLUS_PLUS asked,
//   or GGML_CUDA_SAGE).
enum sage_path {
    SAGE_PATH_SM80,
    SAGE_PATH_TUNED,
    SAGE_PATH_FP8,
    SAGE_PATH_FP8_PLUS_PLUS,
};

// GGML_CUDA_SAGE: tuned | sm80 | fp8 | fp8pp, read once; -1 when unset or unknown.
static int sage_env_path() {
    static const int path = [] {
        const char* value = getenv("GGML_CUDA_SAGE");
        if (value == nullptr) {
            return -1;
        }
        if (strcmp(value, "tuned") == 0) {
            return int(SAGE_PATH_TUNED);
        }
        if (strcmp(value, "sm80") == 0) {
            return int(SAGE_PATH_SM80);
        }
        if (strcmp(value, "fp8") == 0) {
            return int(SAGE_PATH_FP8);
        }
        if (strcmp(value, "fp8pp") == 0) {
            return int(SAGE_PATH_FP8_PLUS_PLUS);
        }
        return -1;
    }();
    return path;
}

static bool sage_fp8_available(int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    return sage_cuda_version >= 12040 && cc >= GGML_CUDA_CC_ADA_LOVELACE &&
           ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_ADA_LOVELACE;
}

static sage_path sage_choose(int device, const ggml_tensor* op) {
    const int cc     = ggml_cuda_info().devices[device].cc;
    const int64_t D  = op->src[0]->ne[0];
    const int env    = sage_env_path();
    const bool fp8   = sage_fp8_available(device);
    const bool fp8pp = fp8 && sage_cuda_version >= 12080;
    if (env == SAGE_PATH_TUNED && D == 128) {
        return SAGE_PATH_TUNED;
    }
    if (env == SAGE_PATH_SM80) {
        return SAGE_PATH_SM80;
    }
    if (env == SAGE_PATH_FP8 && fp8) {
        return SAGE_PATH_FP8;
    }
    if (env == SAGE_PATH_FP8_PLUS_PLUS && fp8pp) {
        return SAGE_PATH_FP8_PLUS_PLUS;
    }
    const auto requested = static_cast<ggml_sage_attn_mode>(op->op_params[1]);
    if (requested == GGML_SAGE_ATTN_AUTO) {
        return D == 128 && cc >= 860 ? SAGE_PATH_TUNED : SAGE_PATH_SM80;
    }
    const auto mode = sage_mode(device, op);
    return mode == GGML_SAGE_ATTN_2_PLUS_PLUS ? SAGE_PATH_FP8_PLUS_PLUS : mode == GGML_SAGE_ATTN_2 ? SAGE_PATH_FP8 : SAGE_PATH_SM80;
}

struct SageWorkspace {
    size_t q;
    size_t k;
    size_t q_scale;
    size_t k_scale;
    size_t k_mean;
    size_t v;
    size_t v_scale;
    size_t output;
    size_t size;

    // The scratch lives behind the op's own output: the FP8 paths' V copy and scales only where they run, the FP16
    // output only where it is converted to F32 afterwards (the tuned path writes F32 itself).
    SageWorkspace(const ggml_tensor* dst, sage_path path) {
        const ggml_tensor* qt = dst->src[0];
        const ggml_tensor* kt = dst->src[1];
        const bool fp8        = path == SAGE_PATH_FP8 || path == SAGE_PATH_FP8_PLUS_PLUS;
        size                  = GGML_PAD(ggml_nbytes(dst), 256);
        q                     = reserve(ggml_nelements(qt));
        k                     = reserve(ggml_nelements(kt));
        q_scale               = reserve(GGML_PAD(qt->ne[1], 128) / 32 * 8 * qt->ne[2] * qt->ne[3] * sizeof(float));
        k_scale               = reserve(GGML_PAD(kt->ne[1], 64) / 64 * 4 * kt->ne[2] * kt->ne[3] * sizeof(float));
        k_mean                = reserve(kt->ne[0] * kt->ne[2] * kt->ne[3] * sizeof(float));
        v                     = fp8 ? reserve(kt->ne[0] * GGML_PAD(kt->ne[1], 64) * kt->ne[2] * kt->ne[3]) : 0;
        v_scale               = fp8 ? reserve(kt->ne[0] * kt->ne[2] * kt->ne[3] * sizeof(float)) : 0;
        output                = path != SAGE_PATH_TUNED ? reserve(ggml_nelements(dst) * sizeof(half)) : 0;
    }

    size_t reserve(size_t bytes) {
        const size_t offset = size;
        size += GGML_PAD(bytes, 256);
        return offset;
    }
};

// Q and K arrive as F32 or F16 (Spellcast3D: F16 straight from Qwen-Image 2.1's RoPE, as flash attention takes them).
static __device__ __forceinline__ float sage_to_float(const float x) {
    return x;
}

static __device__ __forceinline__ float sage_to_float(const half x) {
    return __half2float(x);
}

template <int D, typename T>
static __global__ void sage_key_mean(const T* k, float* mean, int length) {
    const int channel = blockIdx.x * 32 + threadIdx.x % 32;
    const int row     = threadIdx.x / 32;
    const size_t base = size_t(blockIdx.y) * length * D;
    float sum         = 0.0f;
    for (int token = row; token < length; token += 8) {
        sum += sage_to_float(k[base + size_t(token) * D + channel]);
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    if (row == 0) {
        for (int i = 1; i < 8; ++i) {
            sum += sums[threadIdx.x + i * 32];
        }
        mean[size_t(blockIdx.y) * D + channel] = sum / length;
    }
}

template <int D, int TOKENS, bool SMOOTH, typename T>
static __global__ void sage_quantize(const T* input, const float* mean, int8_t* output, float* scales, int length) {
    constexpr int VALUES = D * TOKENS / 256;
    const int head       = blockIdx.y;
    const int first      = blockIdx.x * TOKENS * D;
    const size_t base    = size_t(head) * length * D;
    float values[VALUES];
    float amax = 1e-7f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = first + threadIdx.x + i * 256;
        float value     = 0.0f;
        if (index < length * D) {
            value = sage_to_float(input[base + index]);
            if constexpr (SMOOTH) {
                value -= mean[size_t(head) * D + index % D];
            }
        }
        values[i] = value;
        amax      = fmaxf(amax, fabsf(value));
    }
    amax = warp_reduce_max(amax);
    __shared__ float maxima[8];
    if (threadIdx.x % 32 == 0) {
        maxima[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        amax = threadIdx.x < 8 ? maxima[threadIdx.x] : 1e-7f;
        amax = warp_reduce_max(amax);
        if (threadIdx.x == 0) {
            maxima[0]                                     = amax;
            scales[size_t(head) * gridDim.x + blockIdx.x] = amax / 127.0f;
        }
    }
    __syncthreads();
    const float inverse_scale = 127.0f / maxima[0];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = first + threadIdx.x + i * 256;
        if (index < length * D) {
            output[base + index] = int8_t(__float2int_rn(values[i] * inverse_scale));
        }
    }
}

template <int D, bool KEY, typename T>
static __global__ void sage_quantize_per_thread(const T* input, const float* mean, int8_t* output, float* scales, int length) {
    constexpr int GROUPS = KEY ? 4 : 8;
    constexpr int TOKENS = KEY ? 64 : 32;
    constexpr int VALUES = D * TOKENS / GROUPS / 32;
    const int group      = threadIdx.x / 32;
    const int lane       = threadIdx.x % 32;
    const size_t base    = size_t(blockIdx.y) * length * D;
    float values[VALUES];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = lane + i * 32;
        const int row   = index / D;
        const int token = blockIdx.x * TOKENS + (KEY ? row / 2 * 8 + group * 2 + row % 2 : row * 8 + group);
        float value     = token < length ? sage_to_float(input[base + size_t(token) * D + index % D]) : 0.0f;
        if constexpr (KEY) {
            if (token < length) {
                value -= mean[size_t(blockIdx.y) * D + index % D];
            }
        }
        values[i] = value;
        amax      = fmaxf(amax, fabsf(value));
    }
    const float scale = warp_reduce_max(amax) / 127.0f + 1e-7f;
    if (lane == 0) {
        scales[(size_t(blockIdx.y) * gridDim.x + blockIdx.x) * GROUPS + group] = scale;
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int index = lane + i * 32;
        const int row   = index / D;
        const int token = blockIdx.x * TOKENS + (KEY ? row / 2 * 8 + group * 2 + row % 2 : row * 8 + group);
        if (token < length) {
            const float value                            = values[i] / scale;
            output[base + size_t(token) * D + index % D] = int8_t(__float2int_rz(value + copysignf(0.5f, value)));
        }
    }
}

template <int D>
static __global__ void sage_value_scale(const half* v, float* scales, int length, float scale_max) {
    const int channel = blockIdx.x * 32 + threadIdx.x % 32;
    const int row     = threadIdx.x / 32;
    const size_t base = size_t(blockIdx.y) * length * D;
    float amax        = 1e-7f;
    for (int token = row; token < length; token += 8) {
        amax = fmaxf(amax, fabsf(__half2float(v[base + size_t(token) * D + channel])));
    }
    __shared__ float maxima[256];
    maxima[threadIdx.x] = amax;
    __syncthreads();
    if (row == 0) {
        for (int i = 1; i < 8; ++i) {
            amax = fmaxf(amax, maxima[threadIdx.x + i * 32]);
        }
        scales[size_t(blockIdx.y) * D + channel] = amax / scale_max;
    }
}

template <int D>
static __global__ void sage_quantize_value(const half* v, const float* scales, int8_t* output, int length, int padded) {
    __shared__ uint8_t tile[32][D + 4];
    const int head    = blockIdx.y;
    const int first   = blockIdx.x * 32;
    const size_t base = size_t(head) * length * D;
    for (int i = threadIdx.x; i < 32 * D; i += 256) {
        const int token   = first + i / D;
        const int channel = i % D;
        const float value = token < length ? __half2float(v[base + size_t(token) * D + channel]) / scales[size_t(head) * D + channel] : 0.0f;
        // FP8 MMA consumes pairs of tokens in a permuted, transposed V tile.
        const int row           = i / D;
        const int permuted      = row / 16 * 16 + row % 16 / 8 * 2 + row % 8 / 2 * 4 + row % 2;
        tile[permuted][channel] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * D; i += 256) {
        output[(size_t(head) * D + i / 32) * padded + first + i % 32] = tile[i % 32][i / 32];
    }
}

// The key mean and the K quantization (smoothed), for Q / K of type QT / KT.
template <int D, typename KT>
static void sage_prepare_k(const ggml_tensor* k, float* km, int8_t* ki, float* ks, cudaStream_t stream) {
    const int lk = k->ne[1], hk = k->ne[2], batch = k->ne[3];
    sage_key_mean<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const KT*>(k->data), km, lk);
    sage_quantize<D, 64, true><<<dim3(GGML_PAD(lk, 64) / 64, hk * batch), 256, 0, stream>>>(
        static_cast<const KT*>(k->data), km, ki, ks, lk);
}

template <int D, bool PLUS_PLUS, typename QT, typename KT>
static void sage_attn_fp8(ggml_backend_cuda_context& ctx, ggml_tensor* dst, const SageWorkspace& work) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const int lq = q->ne[1], lk = k->ne[1], padded = GGML_PAD(lk, 64);
    const int hq = q->ne[2], hk = k->ne[2], batch = q->ne[3];
    char* data                = static_cast<char*>(dst->data);
    auto* qi                  = reinterpret_cast<int8_t*>(data + work.q);
    auto* ki                  = reinterpret_cast<int8_t*>(data + work.k);
    auto* vi                  = reinterpret_cast<int8_t*>(data + work.v);
    auto* qs                  = reinterpret_cast<float*>(data + work.q_scale);
    auto* ks                  = reinterpret_cast<float*>(data + work.k_scale);
    auto* vs                  = reinterpret_cast<float*>(data + work.v_scale);
    auto* km                  = reinterpret_cast<float*>(data + work.k_mean);
    auto* out                 = reinterpret_cast<half*>(data + work.output);
    const cudaStream_t stream = ctx.stream();
    sage_key_mean<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const KT*>(k->data), km, lk);
    sage_quantize_per_thread<D, false><<<dim3(GGML_PAD(lq, 128) / 32, hq * batch), 256, 0, stream>>>(
        static_cast<const QT*>(q->data), nullptr, qi, qs, lq);
    sage_quantize_per_thread<D, true><<<dim3(padded / 64, hk * batch), 128, 0, stream>>>(
        static_cast<const KT*>(k->data), km, ki, ks, lk);
    // The smaller V range keeps the FP16 instruction accumulator finite in SageAttention2++.
    sage_value_scale<D><<<dim3(D / 32, hk * batch), 256, 0, stream>>>(static_cast<const half*>(v->data), vs, lk, PLUS_PLUS ? 2.25f : 448.0f);
    sage_quantize_value<D><<<dim3(padded / 32, hk * batch), 256, 0, stream>>>(static_cast<const half*>(v->data), vs, vi, lk, padded);
    auto kernel = sageattention::qk_int_sv_f8_attn_kernel<128, 64, 32, 64, D,
                                                          sageattention::DataType::kInt8, sageattention::QuantGranularity::kPerThread,
                                                          sageattention::QuantGranularity::kPerThread, float, true, half,
                                                          sageattention::ComputeUnit::kCudaCore, sageattention::MaskMode::kNone, false, true, false, PLUS_PLUS>;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    kernel<<<dim3(GGML_PAD(lq, 128) / 128, hq, batch), dim3(32, 4), 256 * D, stream>>>(
        qi, ki, vi, out, nullptr, qs, ks, vs, nullptr, lq, lk, hq / hk,
        D * lq * hq, D, D * lq, D * lk * hk, D, D * lk,
        D * padded * hk, D * padded, padded, D * lq * hq, D * hq, D, scale);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(out, static_cast<float*>(dst->data), ggml_nelements(dst), stream);
}

// V's strides in halves (Spellcast3D: V may be a permuted view, e.g. Qwen-Image 2.1's [D, heads, tokens] memory read
// as [D, tokens, heads]; the kernels read it through these). The FP8 paths read V as contiguous.
struct SageVStrides {
    uint32_t bz, seq, h;
    explicit SageVStrides(const ggml_tensor* v) :
        bz(uint32_t(v->nb[3] / sizeof(half))), seq(uint32_t(v->nb[1] / sizeof(half))), h(uint32_t(v->nb[2] / sizeof(half))) {}
};

template <int D, typename QT, typename KT>
static void sage_attn_sm80(ggml_backend_cuda_context& ctx, ggml_tensor* dst, const SageWorkspace& work) {
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const int lq = q->ne[1], lk = k->ne[1];
    const int hq = q->ne[2], hk = k->ne[2], batch = q->ne[3];
    char* data                = static_cast<char*>(dst->data);
    auto* qi                  = reinterpret_cast<int8_t*>(data + work.q);
    auto* ki                  = reinterpret_cast<int8_t*>(data + work.k);
    auto* qs                  = reinterpret_cast<float*>(data + work.q_scale);
    auto* ks                  = reinterpret_cast<float*>(data + work.k_scale);
    auto* km                  = reinterpret_cast<float*>(data + work.k_mean);
    auto* out                 = reinterpret_cast<half*>(data + work.output);
    const cudaStream_t stream = ctx.stream();
    const SageVStrides vst(v);

    sage_prepare_k<D, KT>(k, km, ki, ks, stream);
    sage_quantize<D, 32, false><<<dim3(GGML_PAD(lq, 128) / 32, hq * batch), 256, 0, stream>>>(
        static_cast<const QT*>(q->data), nullptr, qi, qs, lq);

    constexpr size_t shared_bytes = (128 + 64 + 64 * 2) * D;
    auto kernel                   = sageattention::qk_int_sv_f16_attn_kernel<128, 64, 32, 64, D,
                                                           sageattention::DataType::kInt8, sageattention::QuantGranularity::kPerWarp,
                                                           sageattention::QuantGranularity::kPerWarp, float, false, half,
                                                           sageattention::ComputeUnit::kTensorCore, sageattention::MaskMode::kNone, false, false>;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    kernel<<<dim3(GGML_PAD(lq, 128) / 128, hq, batch), dim3(32, 4), shared_bytes, stream>>>(
        qi, ki, static_cast<half*>(v->data), out, nullptr, qs, ks, nullptr,
        lq, lk, hq / hk,
        D * lq * hq, D, D * lq,
        D * lk * hk, D, D * lk,
        vst.bz, vst.seq, vst.h,
        D * lq * hq, D * hq, D,
        scale);
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(out, static_cast<float*>(dst->data), ggml_nelements(dst), stream);
}

// SAGE_PATH_TUNED (see sage_path): head 128, Q in groups of 16 rows (8 warps a block of 128 queries).
template <typename QT, typename KT>
static void sage_attn_tuned(ggml_backend_cuda_context& ctx, ggml_tensor* dst, const SageWorkspace& work) {
    constexpr int D      = 128;
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* k = dst->src[1];
    const ggml_tensor* v = dst->src[2];
    const int lq = q->ne[1], lk = k->ne[1];
    const int hq = q->ne[2], hk = k->ne[2], batch = q->ne[3];
    char* data                = static_cast<char*>(dst->data);
    auto* qi                  = reinterpret_cast<int8_t*>(data + work.q);
    auto* ki                  = reinterpret_cast<int8_t*>(data + work.k);
    auto* qs                  = reinterpret_cast<float*>(data + work.q_scale);
    auto* ks                  = reinterpret_cast<float*>(data + work.k_scale);
    auto* km                  = reinterpret_cast<float*>(data + work.k_mean);
    const cudaStream_t stream = ctx.stream();
    const SageVStrides vst(v);

    sage_prepare_k<D, KT>(k, km, ki, ks, stream);
    sage_quantize<D, 16, false><<<dim3(GGML_PAD(lq, 128) / 16, hq * batch), 256, 0, stream>>>(
        static_cast<const QT*>(q->data), nullptr, qi, qs, lq);

    constexpr size_t shared_bytes = (128 + 64 + 64 * 2) * D;
    auto kernel                   = sageattention::qk_int_sv_f16_attn_kernel<128, 64, 16, 64, D,
                                                           sageattention::DataType::kInt8, sageattention::QuantGranularity::kPerWarp,
                                                           sageattention::QuantGranularity::kPerWarp, float, true, float,
                                                           sageattention::ComputeUnit::kTensorCore, sageattention::MaskMode::kNone, false, false, true>;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    kernel<<<dim3(GGML_PAD(lq, 128) / 128, hq, batch), dim3(32, 8), shared_bytes, stream>>>(
        qi, ki, static_cast<half*>(v->data), static_cast<float*>(dst->data), nullptr, qs, ks, nullptr,
        lq, lk, hq / hk,
        D * lq * hq, D, D * lq,
        D * lk * hk, D, D * lk,
        vst.bz, vst.seq, vst.h,
        D * lq * hq, D * hq, D,
        scale);
}

template <int D, typename QT, typename KT>
static void sage_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const sage_path path = sage_choose(ctx.device, dst);
    const SageWorkspace work(dst, path);
    switch (path) {
        case SAGE_PATH_FP8_PLUS_PLUS:
            sage_attn_fp8<D, true, QT, KT>(ctx, dst, work);
            return;
        case SAGE_PATH_FP8:
            sage_attn_fp8<D, false, QT, KT>(ctx, dst, work);
            return;
        case SAGE_PATH_TUNED:
            if constexpr (D == 128) {
                sage_attn_tuned<QT, KT>(ctx, dst, work);
                return;
            }
            break;
        case SAGE_PATH_SM80:
            break;
    }
    sage_attn_sm80<D, QT, KT>(ctx, dst, work);
}

template <int D>
static void sage_attn_typed(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const bool q16 = dst->src[0]->type == GGML_TYPE_F16;
    const bool k16 = dst->src[1]->type == GGML_TYPE_F16;
    if (q16 && k16) {
        sage_attn<D, half, half>(ctx, dst);
    } else if (q16) {
        sage_attn<D, half, float>(ctx, dst);
    } else if (k16) {
        sage_attn<D, float, half>(ctx, dst);
    } else {
        sage_attn<D, float, float>(ctx, dst);
    }
}

bool ggml_cuda_sage_attn_supported(int device, const ggml_tensor* op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE ||
        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const auto mode = sage_mode(device, op);
    if (mode < GGML_SAGE_ATTN_AUTO || mode > GGML_SAGE_ATTN_2_PLUS_PLUS) {
        return false;
    }
    if (mode != GGML_SAGE_ATTN_AUTO && (sage_cuda_version < 12040 || cc < GGML_CUDA_CC_ADA_LOVELACE ||
                                        ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_ADA_LOVELACE)) {
        return false;
    }
    if (mode == GGML_SAGE_ATTN_2_PLUS_PLUS && sage_cuda_version < 12080) {
        return false;
    }
    const ggml_tensor* q = op->src[0];
    const ggml_tensor* k = op->src[1];
    const ggml_tensor* v = op->src[2];
    if (!q || !k || !v || op->src[3] || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) ||
        (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) || v->type != GGML_TYPE_F16 ||
        op->type != GGML_TYPE_F32 || (q->ne[0] != 64 && q->ne[0] != 128)) {
        return false;
    }
    for (const ggml_tensor* t : {q, k, op}) {
        if (!ggml_is_contiguous(t) || ggml_nelements(t) > INT_MAX || t->ne[2] * t->ne[3] > 65535) {
            return false;
        }
    }
    // V: contiguous for the FP8 paths (their V quantizers read it so; a view, e.g. Qwen-Image 2.1's permuted V, makes
    // the op unsupported there and the caller falls back to its other attention); for the others any view whose rows
    // are contiguous and 16-byte aligned (the kernels copy 16 bytes at a time) and whose last element's offset fits
    // the 32-bit stride arithmetic.
    const sage_path path = sage_choose(device, op);
    if (path == SAGE_PATH_FP8 || path == SAGE_PATH_FP8_PLUS_PLUS) {
        if (!ggml_is_contiguous(v)) {
            return false;
        }
    } else {
        const size_t v_last = (v->ne[1] - 1) * v->nb[1] + (v->ne[2] - 1) * v->nb[2] + (v->ne[3] - 1) * v->nb[3];
        if (v->nb[0] != sizeof(half) || v->nb[1] % 16 != 0 || v->nb[2] % 16 != 0 || v->nb[3] % 16 != 0 ||
            uintptr_t(v->data) % 16 != 0 || v_last / sizeof(half) > UINT32_MAX) {
            return false;
        }
    }
    if (ggml_nelements(v) > INT_MAX || v->ne[2] * v->ne[3] > 65535) {
        return false;
    }
    if (k->ne[0] * GGML_PAD(k->ne[1], 64) * k->ne[2] * k->ne[3] > INT_MAX) {
        return false;
    }
    return q->ne[0] == k->ne[0] && ggml_are_same_shape(k, v) && q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] == k->ne[3] && op->ne[0] == q->ne[0] && op->ne[1] == q->ne[2] &&
           op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3];
}

size_t ggml_cuda_sage_attn_get_alloc_size(int device, const ggml_tensor* op) {
    // Scratch belongs to the graph allocation, so graph capture and VRAM planning see its full lifetime.
    return SageWorkspace(op, sage_choose(device, op)).size;
}

void ggml_cuda_sage_attn(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(ggml_cuda_sage_attn_supported(ctx.device, dst));
    if (dst->src[0]->ne[0] == 64) {
        sage_attn_typed<64>(ctx, dst);
    } else {
        sage_attn_typed<128>(ctx, dst);
    }
}

#else

bool ggml_cuda_sage_attn_supported(int, const ggml_tensor*) {
    return false;
}

size_t ggml_cuda_sage_attn_get_alloc_size(int, const ggml_tensor* op) {
    return ggml_nbytes(op);
}

void ggml_cuda_sage_attn(ggml_backend_cuda_context&, ggml_tensor*) {
    GGML_ABORT("SageAttention was not compiled into this backend");
}

#endif
