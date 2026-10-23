// Research emulation of int8 matmul formats for picture tests, through exact f32 GEMMs of quantized-then-dequantized
// operands (every format is reproduced to f32 rounding). Inactive unless GGML_CUDA_W8A8 is set:
//   tok                       int8 activations per token, int8 weights per output channel
//   sub:<bits>                int8 activations per token with per-32 integer sub-scales of <bits> bits
//                             (d_g = d_t * sy_g / M, sy_g = ceil(M * amax_g / amax_t), M = 2^bits - 1), int8 weights
//                             per output channel
//   sub:<bits>:wg:<G>         as sub, the int8 weights with one scale per (row, G K-values) instead of per row
//   sub:<bits>:bal:<beta>     as sub, the weights first divided per input channel by c_k = (max_n |w_nk|)^beta and the
//                             activations multiplied by c_k (the balance folded into the quantizer)
//   dyn:<alpha>               per token + SmoothQuant smoothing s_k = cmax_k^a / wmax_k^(1-a) from this call's column maxima
//   static:<alpha>:<file>     as dyn with the column maxima from a calibration file ([i32 len][name][i32 K][K f32] records)
// Applies to Q4_K / Q5_K weights of the diffusion transformer's blocks with at least GGML_CUDA_W8A8_MIN_TOKENS (256)
// tokens. The weights are dequantized to f32 per call; the GEMM runs without TF32 (slow by design).
#include "w8a8-emu.cuh"
#include "convert.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct w8a8_cfg {
    enum mode_t { OFF, TOK, SUB, DYN, STATIC } mode = OFF;
    float   alpha      = 0.5f;
    int     bits       = 8;
    float   beta       = 0.0f;  // 0 = no column balance
    int     wgroup     = 0;     // int8 weight scale per (row, wgroup K-values); 0 = per row
    int64_t min_tokens = 256;
    std::map<std::string, float *> cal;  // weight name -> device column maxima [K]
};

static w8a8_cfg & w8a8_config() {
    static w8a8_cfg cfg;
    static bool     init = false;
    if (init) {
        return cfg;
    }
    init = true;
    const char * e = getenv("GGML_CUDA_W8A8");
    if (e == nullptr || strcmp(e, "off") == 0) {
        return cfg;
    }
    const std::string s(e);
    if (s == "tok") {
        cfg.mode = w8a8_cfg::TOK;
    } else if (s.rfind("sub:", 0) == 0) {
        cfg.mode = w8a8_cfg::SUB;
        cfg.bits = atoi(s.c_str() + 4);
        const size_t bal = s.find(":bal:");
        if (bal != std::string::npos) {
            cfg.beta = (float) atof(s.c_str() + bal + 5);
        }
        const size_t wg = s.find(":wg:");
        if (wg != std::string::npos) {
            cfg.wgroup = atoi(s.c_str() + wg + 4);
        }
    } else if (s.rfind("dyn:", 0) == 0) {
        cfg.mode  = w8a8_cfg::DYN;
        cfg.alpha = (float) atof(s.c_str() + 4);
    } else if (s.rfind("static:", 0) == 0) {
        cfg.mode = w8a8_cfg::STATIC;
        const size_t colon = s.find(':', 7);
        cfg.alpha = (float) atof(s.substr(7, colon - 7).c_str());
        std::ifstream f(s.substr(colon + 1), std::ios::binary);
        while (f) {
            int32_t len = 0;
            if (!f.read((char *) &len, 4)) {
                break;
            }
            std::string name(len, '\0');
            f.read(&name[0], len);
            int32_t K = 0;
            f.read((char *) &K, 4);
            std::vector<float> v(K);
            f.read((char *) v.data(), 4 * (size_t) K);
            float * d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, 4 * (size_t) K));
            CUDA_CHECK(cudaMemcpy(d, v.data(), 4 * (size_t) K, cudaMemcpyHostToDevice));
            cfg.cal[name] = d;
        }
        fprintf(stderr, "w8a8 emulation: %zu calibration records\n", cfg.cal.size());
    } else {
        fprintf(stderr, "w8a8 emulation: unknown mode %s\n", e);
    }
    if (const char * mt = getenv("GGML_CUDA_W8A8_MIN_TOKENS")) {
        cfg.min_tokens = atoll(mt);
    }
    fprintf(stderr, "w8a8 emulation: mode %d alpha %g bits %d beta %g min tokens %lld\n",
            (int) cfg.mode, cfg.alpha, cfg.bits, cfg.beta, (long long) cfg.min_tokens);
    return cfg;
}

// column maxima of |x| over the rows of a row-major [R, K] matrix (atomic max on non-negative floats; out zeroed first)
static __global__ void w8a8_colmax(const float * __restrict__ x, const int64_t K, const int64_t R, const int64_t rows_per_block,
                                   float * __restrict__ out) {
    const int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }
    const int64_t r0 = (int64_t) blockIdx.y * rows_per_block;
    const int64_t r1 = min(R, r0 + rows_per_block);
    float m = 0.0f;
    for (int64_t r = r0; r < r1; ++r) {
        m = fmaxf(m, fabsf(x[r * K + k]));
    }
    atomicMax((int *) out + k, __float_as_int(m));
}

static void w8a8_colmax_cuda(const float * x, const int64_t K, const int64_t R, float * out, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, K * sizeof(float), stream));
    const int64_t rows_per_block = 256;
    const dim3    grid((unsigned) ((K + 255) / 256), (unsigned) ((R + rows_per_block - 1) / rows_per_block));
    w8a8_colmax<<<grid, 256, 0, stream>>>(x, K, R, rows_per_block, out);
}

// per-channel factors: smoothing (alpha) fx = 1/s, fw = s with s = cmax^a / wmax^(1-a); or the column balance (beta)
// fx = c, fw = 1/c with c = wmax^beta
static __global__ void w8a8_factors(const float * __restrict__ cmax, const float * __restrict__ wmax, const int64_t K,
                                    const float alpha, const float beta, float * __restrict__ fx, float * __restrict__ fw) {
    const int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) {
        return;
    }
    const float w = wmax[k];
    if (cmax != nullptr) {
        const float c = cmax[k];
        const float v = (c > 0.0f && w > 0.0f) ? powf(c, alpha) / powf(w, 1.0f - alpha) : 1.0f;
        fx[k] = 1.0f / v;
        fw[k] = v;
    } else {
        const float c = w > 0.0f ? powf(w, beta) : 1.0f;
        fx[k] = c;
        fw[k] = 1.0f / c;
    }
}

// one block per row: y = the int8 rounding of x[r, :] * f, dequantized. bits == 0: one scale per row (max / 127);
// bits > 0: per-32 integer sub-scales of that width under the row scale.
static __global__ void w8a8_rowquant(const float * __restrict__ x, const int64_t K, const float * __restrict__ f,
                                     const int bits, float * __restrict__ y) {
    const int64_t r  = blockIdx.x;
    const float * xr = x + r * K;
    float *       yr = y + r * K;
    __shared__ float sm[32];
    float m = 0.0f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(xr[k] * (f ? f[k] : 1.0f)));
    }
    m = warp_reduce_max(m);
    if (threadIdx.x % WARP_SIZE == 0) {
        sm[threadIdx.x / WARP_SIZE] = m;
    }
    __syncthreads();
    m = threadIdx.x < blockDim.x / WARP_SIZE ? sm[threadIdx.x] : 0.0f;
    m = warp_reduce_max(m);
    __syncthreads();
    if (threadIdx.x == 0) {
        sm[0] = m;
    }
    __syncthreads();
    const float amax = sm[0];
    const float d_row = amax / 127.0f;
    if (bits == 0) {
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
            const float v = xr[k] * (f ? f[k] : 1.0f);
            yr[k] = (float) max(-127, min(127, __float2int_rn(v * inv))) * d_row;
        }
        return;
    }
    // 8 consecutive lanes hold one group of 32 (4 values each)
    const int M = (1 << bits) - 1;
    for (int64_t base = (int64_t) threadIdx.x * 4; base < K; base += (int64_t) blockDim.x * 4) {
        float v[4];
        float g = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = xr[base + i] * (f ? f[base + i] : 1.0f);
            g    = fmaxf(g, fabsf(v[i]));
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            g = fmaxf(g, __shfl_xor_sync(0xFFFFFFFF, g, offset, WARP_SIZE));
        }
        const int   sy  = amax > 0.0f ? max(1, min(M, (int) ceilf((float) M * g / amax))) : 1;
        const float d   = d_row * (float) sy / (float) M;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            yr[base + i] = (float) max(-127, min(127, __float2int_rn(v[i] * inv))) * d;
        }
    }
}

// one block per row: int8 rounding of x[r, :] with one scale per group of G consecutive values (G % 32 == 0), dequantized
static __global__ void w8a8_rowquant_grouped(const float * __restrict__ x, const int64_t K, const int G, float * __restrict__ y) {
    const int64_t r  = blockIdx.x;
    const float * xr = x + r * K;
    float *       yr = y + r * K;
    for (int64_t g0 = 0; g0 < K; g0 += G) {
        float m = 0.0f;
        for (int k = threadIdx.x; k < G; k += blockDim.x) {
            m = fmaxf(m, fabsf(xr[g0 + k]));
        }
        m = warp_reduce_max(m);
        __shared__ float sm[32];
        if (threadIdx.x % WARP_SIZE == 0) {
            sm[threadIdx.x / WARP_SIZE] = m;
        }
        __syncthreads();
        m = threadIdx.x < blockDim.x / WARP_SIZE ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        __syncthreads();
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
        __syncthreads();
        const float amax = sm[0];
        const float d    = amax / 127.0f;
        const float inv  = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int k = threadIdx.x; k < G; k += blockDim.x) {
            yr[g0 + k] = (float) max(-127, min(127, __float2int_rn(xr[g0 + k] * inv))) * d;
        }
        __syncthreads();
    }
}

bool ggml_cuda_w8a8_emu(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    w8a8_cfg & cfg = w8a8_config();
    if (cfg.mode == w8a8_cfg::OFF) {
        return false;
    }
    if ((src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || src1->ne[1] < cfg.min_tokens || strstr(src0->name, "transformer_blocks") == nullptr) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t T = src1->ne[1];
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N && dst->ne[1] == T && K % 32 == 0);

    const float * cal = nullptr;
    if (cfg.mode == w8a8_cfg::STATIC) {
        const auto it = cfg.cal.find(src0->name);
        if (it == cfg.cal.end()) {
            static int warned = 0;
            if (warned++ < 8) {
                fprintf(stderr, "w8a8 emulation: no calibration record for %s\n", src0->name);
            }
            return false;
        }
        cal = it->second;
    }

    cudaStream_t     stream = ctx.stream();
    ggml_cuda_pool & pool   = ctx.pool();

    ggml_cuda_pool_alloc<float> w32(pool, (size_t) N * K);
    const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
    GGML_ASSERT(to_fp32 != nullptr);
    to_fp32(src0->data, w32.get(), N * K, stream);

    ggml_cuda_pool_alloc<float> stats(pool, 4 * (size_t) K);
    float * wmax = stats.get();
    float * cmax = wmax + K;
    float * fx   = cmax + K;
    float * fw   = fx + K;
    const bool factors = cfg.mode == w8a8_cfg::DYN || cfg.mode == w8a8_cfg::STATIC || (cfg.mode == w8a8_cfg::SUB && cfg.beta != 0.0f);
    if (factors) {
        w8a8_colmax_cuda(w32.get(), K, N, wmax, stream);
        if (cfg.mode == w8a8_cfg::DYN) {
            w8a8_colmax_cuda((const float *) src1->data, K, T, cmax, stream);
            cal = cmax;
        }
        w8a8_factors<<<(unsigned) ((K + 255) / 256), 256, 0, stream>>>(cal, wmax, K, cfg.alpha, cfg.beta, fx, fw);
    }

    ggml_cuda_pool_alloc<float> xq(pool, (size_t) T * K);
    ggml_cuda_pool_alloc<float> wq(pool, (size_t) N * K);
    const int bits = cfg.mode == w8a8_cfg::SUB ? cfg.bits : 0;
    w8a8_rowquant<<<(unsigned) T, 256, 0, stream>>>((const float *) src1->data, K, factors ? fx : nullptr, bits, xq.get());
    if (cfg.wgroup > 0 && !factors) {
        GGML_ASSERT(K % cfg.wgroup == 0 && cfg.wgroup % 32 == 0);
        w8a8_rowquant_grouped<<<(unsigned) N, 256, 0, stream>>>(w32.get(), K, cfg.wgroup, wq.get());
    } else {
        w8a8_rowquant<<<(unsigned) N, 256, 0, stream>>>(w32.get(), K, factors ? fw : nullptr, 0, wq.get());
    }

    // dst [N, T] (N contiguous) = wq [K, N]^T x xq [K, T] in column-major terms; exact f32 (no TF32)
    cublasHandle_t h = ctx.cublas_handle();
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, wq.get(), (int) K, xq.get(), (int) K,
                             &zero, (float *) dst->data, (int) N));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    return true;
}
