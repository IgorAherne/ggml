// Bit-identity of the W8A8 format kernels (src/ggml-cuda/w8a8.cuh) with the research emulation (GGML_CUDA_W8A8=sub:5,
// main tree src/ggml-cuda/w8a8-emu.cu) and ggml's Q4_K / Q5_K dequantizers, compiled here VERBATIM with the same nvcc
// flags as ggml-cuda (-O3 -use_fast_math): same instructions -> any difference is a formula difference.
//   weights:     ggml dequantize_q4_K / _q5_K (f32) -> emulation w8a8_rowquant(bits 0): q, d_row; ours: q_w, dW.
//                Also our in-kernel dequantized values against ggml's (every value).
//   activations: emulation w8a8_rowquant(bits = log2(W8A8_SY_MAX + 1) = 5): q, sy, d_row per row; ours: q_x, sy
//                (tile-permuted), d_t;
//                inputs: synthetic (gaussian, outlier channels, zero row, tiny rows, huge range) and real DiT
//                activation rows (dumps of the engine, GGML_CUDA_ACT_DUMP); the SCALE pre-op (x/32 of to_out) against
//                scale_f32's FMA then the emulation; the f16 input path against the emulation of the f16-rounded
//                values (the SwiGLU output the gate|up epilogue stores).
// usage: test_formats [dump.bin ...]
#include "w8a8.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define WARP_SIZE 32
// ---- verbatim from ggml (common.cuh warp_reduce_max, dequantize.cuh dequantize_q4_K / _q5_K / get_scale_min_k4,
// convert.cu dequantize_block_q4_K / _q5_K launches) ----
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}
typedef struct { half2 dm; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;
    dst_t * y = yy + 64*il + n*ir;
    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);
    const uint8_t * q = x[ib].qs + 32*il + n*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = (dst_t)(d1 * (q[l] & 0xF) - m1);
        y[l +32] = (dst_t)(d2 * (q[l] >>  4) - m2);
    }
}
template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;
    const int64_t il  = tid/16;
    const int64_t ir  = tid%16;
    const int64_t is  = 2*il;
    dst_t * y = yy + 64*il + 2*ir;
    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);
    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    uint8_t   hm  = 1 << (2*il);
    y[ 0] = (dst_t)(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = (dst_t)(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = (dst_t)(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = (dst_t)(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}
template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    dequantize_q4_K(vx, i, yy + i*256, threadIdx.x);
}
template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;
    dequantize_q5_K(vx, i, yy + i*256, threadIdx.x);
}
// ---- verbatim from the main tree's w8a8-emu.cu (w8a8_rowquant), instrumented: the int8 values, the sub-scale of each
// group and the row scale are stored as well (extra stores only; the arithmetic is unchanged) ----
static __global__ void w8a8_rowquant(const float * __restrict__ x, const int64_t K, const float * __restrict__ f,
                                     const int bits, float * __restrict__ y,
                                     int8_t * __restrict__ q_out, uint8_t * __restrict__ s_out, float * __restrict__ d_out) {
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
    if (threadIdx.x == 0) d_out[r] = d_row;
    if (bits == 0) {
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
            const float v = xr[k] * (f ? f[k] : 1.0f);
            const int qq = max(-127, min(127, __float2int_rn(v * inv)));
            yr[k] = (float) qq * d_row;
            q_out[r*K + k] = (int8_t) qq;
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
            const int qq = max(-127, min(127, __float2int_rn(v[i] * inv)));
            yr[base + i] = (float) qq * d;
            q_out[r*K + base + i] = (int8_t) qq;
        }
        if (threadIdx.x % 8 == 0) s_out[r*(K/32) + base/32] = (uint8_t) sy;
    }
}
// scale_f32 of ggml (scale.cu): dst = scale * x + bias
static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i < nelements) dst[i] = scale * x[i] + bias;
}
static __global__ void to_half_and_back(const float * x, __half * h, float * y, const int64_t n) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) { h[i] = __float2half(x[i]); y[i] = __half2float(h[i]); }
}
// our dequantization of every value (w8a8_dequant8), for the value-level comparison with ggml's
template <bool q5>
static __global__ void our_dequant(const void * vx, float * y, const int nblocks) {
    const int b = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (b >= nblocks) return;
    float w[8];
    w8a8_dequant8<q5>((const uint8_t *) vx + (int64_t) b * (q5 ? 176 : 144), lane, w);
    const int j = lane / 4, pos = 8 * (lane % 4);
    for (int i = 0; i < 8; ++i) y[(int64_t) b*256 + 32*j + pos + i] = w[i];
}

static long cmp_bytes(const void * a, const void * b, size_t n) {
    const uint8_t * x = (const uint8_t *) a; const uint8_t * y = (const uint8_t *) b; long bad = 0;
    for (size_t i = 0; i < n; ++i) bad += x[i] != y[i];
    return bad;
}

static void test_weights(bool q5, int N, int K, uint32_t seed) {
    const int nb = K / 256, bsz = q5 ? 176 : 144;
    std::vector<uint8_t> h((size_t) N * nb * bsz);
    std::mt19937 rng(seed);
    for (auto & c : h) c = (uint8_t) rng();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t b = 0; b < (size_t) N * nb; ++b) {  // realistic f16 d / dmin (random bytes could be inf / nan)
        const float d = 2e-4f * std::exp(0.5f * nd(rng)), dm = 2e-4f * std::exp(0.5f * nd(rng));
        __half hd = __float2half(d), hm = __float2half(dm);
        memcpy(&h[b*bsz], &hd, 2); memcpy(&h[b*bsz + 2], &hm, 2);
    }
    if (N > 3) {  // an all-zero row (amax = 0)
        for (int b = 0; b < nb; ++b) memset(&h[((size_t) 3*nb + b)*bsz], 0, bsz);
    }
    void * dx; float * dwf; float * dwq; int8_t * deq; uint8_t * dsdummy; float * ddrow; int8_t * dqw; float * ddw; float * dours;
    CK(cudaMalloc(&dx, h.size())); CK(cudaMemcpy(dx, h.data(), h.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwf, (size_t) N*K*4)); CK(cudaMalloc(&dwq, (size_t) N*K*4)); CK(cudaMalloc(&deq, (size_t) N*K));
    CK(cudaMalloc(&dsdummy, (size_t) N*K/32)); CK(cudaMalloc(&ddrow, N*4)); CK(cudaMalloc(&dqw, (size_t) N*K));
    CK(cudaMalloc(&ddw, N*4)); CK(cudaMalloc(&dours, (size_t) N*K*4));
    if (q5) dequantize_block_q5_K<float><<<N*nb, 64>>>(dx, dwf); else dequantize_block_q4_K<float><<<N*nb, 32>>>(dx, dwf);
    w8a8_rowquant<<<N, 256>>>(dwf, K, nullptr, 0, dwq, deq, dsdummy, ddrow);
    w8a8_convert_weights_cuda(dx, q5, false, (int64_t) nb*bsz, dqw, ddw, N, K, 0);
    if (q5) our_dequant<true><<<(N*nb + 7)/8, 256>>>(dx, dours, N*nb); else our_dequant<false><<<(N*nb + 7)/8, 256>>>(dx, dours, N*nb);
    CK(cudaDeviceSynchronize());
    std::vector<float> wf((size_t) N*K), wo((size_t) N*K), drow(N), dw(N);
    std::vector<int8_t> qe((size_t) N*K), qo((size_t) N*K);
    CK(cudaMemcpy(wf.data(), dwf, wf.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(wo.data(), dours, wo.size()*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qe.data(), deq, qe.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qo.data(), dqw, qo.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(drow.data(), ddrow, N*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dw.data(), ddw, N*4, cudaMemcpyDeviceToHost));
    // glu interleaving: row n -> 2n (n < N/2) / 2(n - N/2) + 1
    int8_t * dqg; float * ddg;
    CK(cudaMalloc(&dqg, (size_t) N*K)); CK(cudaMalloc(&ddg, N*4));
    w8a8_convert_weights_cuda(dx, q5, true, (int64_t) nb*bsz, dqg, ddg, N, K, 0);
    std::vector<int8_t> qg((size_t) N*K); std::vector<float> dg(N);
    CK(cudaMemcpy(qg.data(), dqg, qg.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dg.data(), ddg, N*4, cudaMemcpyDeviceToHost));
    long badg = 0;
    for (int n = 0; n < N; ++n) {
        const int o = n < N/2 ? 2*n : 2*(n - N/2) + 1;
        badg += cmp_bytes(&qg[(size_t) o*K], &qe[(size_t) n*K], K) + (memcmp(&dg[o], &drow[n], 4) != 0);
    }
    printf("weights %s %5d x %5d: dequantized values differ from ggml's: %ld of %zu | int8 differ from the emulation: %ld | "
           "dW differ: %ld of %d | glu-interleaved rows/scales differ: %ld\n", q5 ? "Q5_K" : "Q4_K", N, K,
           cmp_bytes(wf.data(), wo.data(), wf.size()*4) ? (long) std::count_if(wf.begin(), wf.end(), [&](const float & v) {
               const size_t i = &v - wf.data(); return memcmp(&v, &wo[i], 4) != 0; }) : 0L, wf.size(),
           cmp_bytes(qe.data(), qo.data(), qe.size()), cmp_bytes(drow.data(), dw.data(), N*4) ? (long) std::count_if(
               drow.begin(), drow.end(), [&](const float & v) { const size_t i = &v - drow.data(); return memcmp(&v, &dw[i], 4) != 0; }) : 0L,
           N, badg);
    CK(cudaFree(dx)); CK(cudaFree(dwf)); CK(cudaFree(dwq)); CK(cudaFree(deq)); CK(cudaFree(dsdummy)); CK(cudaFree(ddrow));
    CK(cudaFree(dqw)); CK(cudaFree(ddw)); CK(cudaFree(dours)); CK(cudaFree(dqg)); CK(cudaFree(ddg));
}

// compares our quantizer on x (T x K, f32; or f16 when half_in, then the reference sees the f16-rounded values) with
// the emulation's; pre: SCALE pre-op (the reference sees scale_f32's output)
static void test_act(const char * name, const std::vector<float> & hx, int T, int K, bool half_in, bool pre, float scale, float bias) {
    const int G = K / 32;
    const int64_t Tp = w8a8_pad_tokens(T);
    float * dx; float * dref_in; __half * dh; float * dy; int8_t * deq; uint8_t * des; float * ded;
    int8_t * dq; uint8_t * dsy; float * ddt;
    CK(cudaMalloc(&dx, (size_t) T*K*4)); CK(cudaMalloc(&dref_in, (size_t) T*K*4)); CK(cudaMalloc(&dh, (size_t) T*K*2));
    CK(cudaMalloc(&dy, (size_t) T*K*4)); CK(cudaMalloc(&deq, (size_t) T*K)); CK(cudaMalloc(&des, (size_t) T*G));
    CK(cudaMalloc(&ded, T*4)); CK(cudaMalloc(&dq, (size_t) Tp*K)); CK(cudaMalloc(&dsy, (size_t) Tp*G)); CK(cudaMalloc(&ddt, Tp*4));
    CK(cudaMemcpy(dx, hx.data(), (size_t) T*K*4, cudaMemcpyHostToDevice));
    const int64_t n = (int64_t) T*K;
    if (half_in) {
        to_half_and_back<<<(unsigned) ((n + 255)/256), 256>>>(dx, dh, dref_in, n);
    } else if (pre) {
        scale_f32<<<(unsigned) ((n + 255)/256), 256>>>(dx, dref_in, scale, bias, n);
    } else {
        CK(cudaMemcpy(dref_in, dx, (size_t) n*4, cudaMemcpyDeviceToDevice));
    }
    constexpr int SY_BITS = 5;
    static_assert((1 << SY_BITS) - 1 == W8A8_SY_MAX, "the emulation's sub-scale width");
    w8a8_rowquant<<<T, 256>>>(dref_in, K, nullptr, SY_BITS, dy, deq, des, ded);
    CK(cudaMemset(dq, 0x55, (size_t) Tp*K)); CK(cudaMemset(dsy, 0x55, (size_t) Tp*G)); CK(cudaMemset(ddt, 0x55, Tp*4));
    bool ok;
    if (half_in) ok = w8a8_quantize_act_cuda<__half>(dh, K, W8A8_PRE_NONE, 1.0f, 0.0f, dq, dsy, ddt, T, K, 0);
    else         ok = w8a8_quantize_act_cuda<float>(dx, K, pre ? W8A8_PRE_SCALE : W8A8_PRE_NONE, scale, bias, dq, dsy, ddt, T, K, 0);
    if (!ok) { printf("%s: K %d not supported by the quantizer\n", name, K); exit(1); }
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> qe((size_t) n), qo((size_t) Tp*K);
    std::vector<uint8_t> se((size_t) T*G), so((size_t) Tp*G);
    std::vector<float> de(T), dd(Tp);
    CK(cudaMemcpy(qe.data(), deq, qe.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(qo.data(), dq, qo.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(se.data(), des, se.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(so.data(), dsy, so.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(de.data(), ded, T*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(dd.data(), ddt, Tp*4, cudaMemcpyDeviceToHost));
    long badq = cmp_bytes(qe.data(), qo.data(), (size_t) n), bads = 0, badd = 0, badpad = 0;
    long sy_hist[4] = {0, 0, 0, 0};  // sy = 1, 2..15, 16..M-1, M
    for (int t = 0; t < T; ++t) {
        badd += memcmp(&de[t], &dd[t], 4) != 0;
        for (int g = 0; g < G; ++g) {
            const uint8_t s = so[((size_t) (t / W8A8_BM) * G + g) * W8A8_BM + w8a8_sy_pos(t % W8A8_BM)];
            bads += s != se[(size_t) t*G + g];
            const int e = se[(size_t) t*G + g];
            sy_hist[e == 1 ? 0 : (e < 16 ? 1 : (e < W8A8_SY_MAX ? 2 : 3))]++;
        }
    }
    for (int64_t t = T; t < Tp; ++t) {
        for (int k = 0; k < K; ++k) badpad += qo[(size_t) t*K + k] != 0;
        badpad += dd[t] != 0.0f;
    }
    printf("act %-34s T %4d K %5d: int8 differ %ld of %lld | sy differ %ld of %lld | d_t differ %ld | padding rows not 0: %ld"
           " | sy: =1 %ld, 2-15 %ld, 16-30 %ld, 31 %ld\n", name, T, K, badq, (long long) n, bads, (long long) T*G, badd, badpad,
           sy_hist[0], sy_hist[1], sy_hist[2], sy_hist[3]);
    CK(cudaFree(dx)); CK(cudaFree(dref_in)); CK(cudaFree(dh)); CK(cudaFree(dy)); CK(cudaFree(deq)); CK(cudaFree(des));
    CK(cudaFree(ded)); CK(cudaFree(dq)); CK(cudaFree(dsy)); CK(cudaFree(ddt));
}

static std::vector<float> synth(int T, int K, uint32_t seed) {
    std::vector<float> x((size_t) T*K);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int> outl(8); for (auto & c : outl) c = (int) (rng() % K);
    for (int t = 0; t < T; ++t) {
        const float rs = t % 7 == 3 ? 1e-30f : (t % 11 == 5 ? 1e4f : 1.0f);   // tiny (denormal-range) and huge rows
        for (int k = 0; k < K; ++k) x[(size_t) t*K + k] = rs * nd(rng);
        for (int c : outl) x[(size_t) t*K + c] *= 50.0f;
        if (t % 13 == 7) for (int k = 0; k < K; k += 64) for (int i = 0; i < 32; ++i) x[(size_t) t*K + k + i] *= 1e-4f;
    }
    if (T > 2) for (int k = 0; k < K; ++k) x[(size_t) 2*K + k] = 0.0f;   // a zero row
    return x;
}

int main(int argc, char ** argv) {
    test_weights(false, 4096, 4096, 1);
    test_weights(false, 512, 12288, 2);
    test_weights(true, 4096, 4096, 3);
    test_weights(true, 512, 12288, 4);
    for (int K : {4096, 12288}) {
        for (int T : {300, 1811}) {
            auto x = synth(T, K, 10 + T + K);
            test_act("synthetic f32", x, T, K, false, false, 1, 0);
            test_act("synthetic SCALE x/32", x, T, K, false, true, 1.0f/32, 0);
            test_act("synthetic SCALE 0.7x+0.1", x, T, K, false, true, 0.7f, 0.1f);
            for (auto & v : x) v = std::max(-60000.0f, std::min(60000.0f, v));
            test_act("synthetic f16 in", x, T, K, true, false, 1, 0);
        }
    }
    for (int a = 1; a < argc; ++a) {   // engine dumps: [magic K T nrows] cmax[K] csq[K] rows[nrows][K]
        FILE * f = fopen(argv[a], "rb");
        if (!f) { printf("cannot open %s\n", argv[a]); continue; }
        int32_t hd[4]; fread(hd, 4, 4, f);
        const int K = hd[1], nr = hd[3];
        fseek(f, 16 + 8L*K, SEEK_SET);
        std::vector<float> x((size_t) nr*K);
        fread(x.data(), 4, x.size(), f); fclose(f);
        char nm[64]; snprintf(nm, sizeof nm, "dump %s", strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a]);
        test_act(nm, x, nr, K, false, false, 1, 0);
        test_act("  same, SCALE x/32", x, nr, K, false, true, 1.0f/32, 0);
        test_act("  same, f16 in", x, nr, K, true, false, 1, 0);
    }
    return 0;
}
