// Speed of the W8A8 route's conversion kernels on sm_86:
//  1. per-layer weight dequantizer: Q4_K (or Q5_K) blocks of N rows x K -> int8 N x K, q8 = rint(w / s_row[n]) clamped to
//     +-127, w = d*sc*q - dmin*m as ggml's dequantize_row_q4_K / q5_K computes it, s_row[n] = max|w[n,:]| / 127
//     precomputed once per model load (its kernel is timed too). One warp per 256-value block: lane l makes values
//     8l .. 8l+7 (one 8-byte load of qs (+ qh), one 8-byte store), so a warp writes 256 contiguous bytes.
//     Variants: exact division w / s_row (__fdiv_rn) and multiplication by the reciprocal.
//  2. per-token activation quantizer: M x K f32 -> int8, s_tok[m] = max|x[m,:]| / 127, q = rint(x * (127 / amax)); one CUDA
//     block of 256 threads per row, the row in registers (K/1024 float4 per thread).
// Checks: every output of a sample of rows against a host reference (exact match expected; rint ties are measure zero).
// usage: w8a8_quant
#include "common.cuh"
#include <cuda_fp16.h>

struct BlockQ4K { __half d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BlockQ5K { __half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
static_assert(sizeof(BlockQ4K) == 144 && sizeof(BlockQ5K) == 176, "block sizes");

__host__ __device__ static inline void scale_min_k4(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// one warp per block; blocks of a row are consecutive: global warp id = row * nb + b
template <bool q5, bool exact_div>
__global__ void __launch_bounds__(256) dequant_k_to_i8(const void * __restrict__ vx, const float * __restrict__ s_row,
                                                        int8_t * __restrict__ out, const int N, const int K) {
    const int nb   = K / 256;
    const int64_t wid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (wid >= (int64_t) N * nb) return;
    const int n  = (int) (wid / nb);
    const int bsz = q5 ? 176 : 144;
    const uint8_t * blk = (const uint8_t *) vx + wid * bsz;
    const int4 hdr4 = *(const int4 *) blk;                      // d, dmin, scales[0..11] (16 bytes, broadcast)
    const uint8_t * hb = (const uint8_t *) &hdr4;
    const float d    = __half2float(*(const __half *) (hb + 0));
    const float dmin = __half2float(*(const __half *) (hb + 2));
    const int g2   = lane / 8;            // 64-value group
    const int hi   = (lane % 8) >= 4;     // high nibbles: sub-block 2*g2 + 1
    const int pos  = 8 * (lane % 4);      // position of the 8 values inside the 32-value sub-block
    const int j    = 2*g2 + hi;
    int sc, m;
    scale_min_k4(j, hb + 4, sc, m);
    const float d1 = d * sc, m1 = dmin * m;
    const uint8_t * qs = blk + (q5 ? 48 : 16) + 32*g2 + pos;
    const uint2 raw = *(const uint2 *) qs;
    uint2 qhv = make_uint2(0, 0);
    if (q5) qhv = *(const uint2 *) (blk + 16 + pos);
    const float sr = s_row[n];
    const float inv = 1.0f / sr;
    uint32_t o[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t r = h ? raw.y : raw.x;
        const uint32_t qh = h ? qhv.y : qhv.x;
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = (r >> (8*i + 4*hi)) & 0xF;
            if (q5) q += ((qh >> (8*i + j)) & 1) * 16;
            const float w = d1 * q - m1;
            const float t = exact_div ? __fdiv_rn(w, sr) : w * inv;
            const int qi = max(-127, min(127, __float2int_rn(t)));
            packed |= (uint32_t) (qi & 0xFF) << (8*i);
        }
        o[h] = packed;
    }
    *(uint2 *) (out + (int64_t) n * K + (wid % nb) * 256 + 8*lane) = make_uint2(o[0], o[1]);
}


// 16 lanes per block, 16 values per lane (one 16-byte load of qs (+ qh), one 16-byte store)
template <bool q5>
__global__ void __launch_bounds__(256) dequant_k_to_i8_w16(const void * __restrict__ vx, const float * __restrict__ s_row,
                                                            int8_t * __restrict__ out, const int N, const int K) {
    const int nb   = K / 256;
    const int64_t bid = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 16;
    const int lane = threadIdx.x % 16;
    if (bid >= (int64_t) N * nb) return;
    const int n  = (int) (bid / nb);
    const int bsz = q5 ? 176 : 144;
    const uint8_t * blk = (const uint8_t *) vx + bid * bsz;
    const int4 hdr4 = *(const int4 *) blk;
    const uint8_t * hb = (const uint8_t *) &hdr4;
    const float d    = __half2float(*(const __half *) (hb + 0));
    const float dmin = __half2float(*(const __half *) (hb + 2));
    const int g2  = lane / 4;
    const int hi  = (lane % 4) >= 2;
    const int pos = 16 * (lane % 2);
    const int j   = 2*g2 + hi;
    int sc, m;
    scale_min_k4(j, hb + 4, sc, m);
    const float d1 = d * sc, m1 = dmin * m;
    const int4 raw = *(const int4 *) (blk + (q5 ? 48 : 16) + 32*g2 + pos);
    int4 qhv = make_int4(0, 0, 0, 0);
    if (q5) qhv = *(const int4 *) (blk + 16 + pos);
    const float inv = 1.0f / s_row[n];
    const uint32_t rr[4] = {(uint32_t) raw.x, (uint32_t) raw.y, (uint32_t) raw.z, (uint32_t) raw.w};
    const uint32_t hh[4] = {(uint32_t) qhv.x, (uint32_t) qhv.y, (uint32_t) qhv.z, (uint32_t) qhv.w};
    uint32_t o[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = (rr[h] >> (8*i + 4*hi)) & 0xF;
            if (q5) q += ((hh[h] >> (8*i + j)) & 1) * 16;
            const int qi = max(-127, min(127, __float2int_rn((d1 * q - m1) * inv)));
            packed |= (uint32_t) (qi & 0xFF) << (8*i);
        }
        o[h] = packed;
    }
    *(int4 *) (out + (int64_t) n * K + (bid % nb) * 256 + 64*g2 + 32*hi + pos) = make_int4(o[0], o[1], o[2], o[3]);
}

// s_row[n] = max |w[n,:]| / 127 (one block of 256 threads per row; offline, once per model load)
template <bool q5>
__global__ void row_absmax_k(const void * __restrict__ vx, float * __restrict__ s_row, const int K) {
    const int n = blockIdx.x, nb = K / 256, bsz = q5 ? 176 : 144;
    float amax = 0.0f;
    for (int v = threadIdx.x; v < K; v += blockDim.x) {
        const uint8_t * blk = (const uint8_t *) vx + ((int64_t) n * nb + v / 256) * bsz;
        const int l = v % 256, g2 = l / 64, o = l % 64, j = 2*g2 + (o >= 32);
        const float d = __half2float(*(const __half *) blk), dmin = __half2float(*(const __half *) (blk + 2));
        int sc, m; scale_min_k4(j, blk + 4, sc, m);
        const uint8_t * qs = blk + (q5 ? 48 : 16);
        int q = (qs[32*g2 + (o % 32)] >> (o >= 32 ? 4 : 0)) & 0xF;
        if (q5) q += ((blk[16 + (o % 32)] >> j) & 1) * 16;
        amax = fmaxf(amax, fabsf(d * sc * q - dmin * m));
    }
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    __shared__ float red[8];
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0; for (int i = 0; i < 8; ++i) a = fmaxf(a, red[i]);
        s_row[n] = a / 127.0f;
    }
}

// activations: one block of 256 threads per row, NV float4 per thread (K = 1024*NV)
template <int NV>
__global__ void __launch_bounds__(256) quant_tok(const float * __restrict__ x, int8_t * __restrict__ q, float * __restrict__ s_tok, const int K) {
    const int m = blockIdx.x;
    const float4 * xr = (const float4 *) (x + (int64_t) m * K);
    float4 v[NV];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        v[i] = xr[i*256 + threadIdx.x];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    }
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    __shared__ float red[8];
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) amax = fmaxf(amax, red[i]);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) s_tok[m] = amax / 127.0f;
    char4 * qr = (char4 *) (q + (int64_t) m * K);
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        char4 c;
        c.x = (signed char) __float2int_rn(v[i].x * inv); c.y = (signed char) __float2int_rn(v[i].y * inv);
        c.z = (signed char) __float2int_rn(v[i].z * inv); c.w = (signed char) __float2int_rn(v[i].w * inv);
        qr[i*256 + threadIdx.x] = c;
    }
}


// SwiGLU of a [gate | up] f32 output (M x 2F, gate first) then the per-token int8 quantization of the M x F product, as
// the W8A8 route needs between the [gate | up] GEMM and the down GEMM (F = 1024*NV)
template <int NV>
__global__ void __launch_bounds__(256) swiglu_quant_tok(const float * __restrict__ gu, int8_t * __restrict__ q, float * __restrict__ s_tok, const int F) {
    const int m = blockIdx.x;
    const float4 * g4 = (const float4 *) (gu + (int64_t) m * 2 * F);
    const float4 * u4 = g4 + F/4;
    float4 v[NV];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const float4 g = g4[i*256 + threadIdx.x], u = u4[i*256 + threadIdx.x];
        v[i] = make_float4(g.x/(1.0f + __expf(-g.x))*u.x, g.y/(1.0f + __expf(-g.y))*u.y, g.z/(1.0f + __expf(-g.z))*u.z, g.w/(1.0f + __expf(-g.w))*u.w);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    }
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    __shared__ float red[8];
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) amax = fmaxf(amax, red[i]);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) s_tok[m] = amax / 127.0f;
    char4 * qr = (char4 *) (q + (int64_t) m * F);
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        char4 c;
        c.x = (signed char) __float2int_rn(v[i].x * inv); c.y = (signed char) __float2int_rn(v[i].y * inv);
        c.z = (signed char) __float2int_rn(v[i].z * inv); c.w = (signed char) __float2int_rn(v[i].w * inv);
        qr[i*256 + threadIdx.x] = c;
    }
}

template <int NV>
static void bench_swiglu(int M) {
    const int F = 1024 * NV;
    float * dgu; int8_t * dq; float * ds;
    CK(cudaMalloc(&dgu, (size_t) M * 2 * F * 4)); CK(cudaMalloc(&dq, (size_t) M * F)); CK(cudaMalloc(&ds, sizeof(float)*M));
    CK(cudaMemset(dgu, 0x3c, (size_t) M * 2 * F * 4)); // ~0.011 everywhere (timing only)
    const float ms = time_median([&] { swiglu_quant_tok<NV><<<M, 256>>>(dgu, dq, ds, F); });
    const double bytes = (double) M * F * 9;
    printf("  swiglu + per-token quant %d x [gate|up] 2 x %5d f32 -> int8 %5d: %.3f ms (%.0f GB/s)\n", M, F, F, ms, bytes/ms/1e6);
    CK(cudaFree(dgu)); CK(cudaFree(dq)); CK(cudaFree(ds));
}

static float host_w(const uint8_t * blk, bool q5, int l) {
    const int g2 = l / 64, o = l % 64, j = 2*g2 + (o >= 32);
    __half_raw hd, hm; memcpy(&hd.x, blk, 2); memcpy(&hm.x, blk + 2, 2);
    const float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
    int sc, m; scale_min_k4(j, blk + 4, sc, m);
    const uint8_t * qs = blk + (q5 ? 48 : 16);
    int q = (qs[32*g2 + (o % 32)] >> (o >= 32 ? 4 : 0)) & 0xF;
    if (q5) q += ((blk[16 + (o % 32)] >> j) & 1) * 16;
    const float d1 = d * sc, m1 = dmin * m;
    return d1 * q - m1;
}

template <bool q5>
static void bench_weights(int N, int K) {
    const int nb = K / 256, bsz = q5 ? 176 : 144;
    const size_t bytes_in = (size_t) N * nb * bsz, bytes_out = (size_t) N * K;
    std::vector<uint8_t> h(bytes_in);
    std::mt19937 rng(11);
    for (auto & c : h) c = (uint8_t) rng();
    for (size_t b = 0; b < (size_t) N * nb; ++b) { // sane f16 d / dmin (random bytes could be inf / nan)
        __half hd = __float2half(1e-3f * (1 + (rng() % 1000) / 1000.0f)), hm = __float2half(1e-3f * (rng() % 1000) / 1000.0f);
        memcpy(&h[b*bsz], &hd, 2); memcpy(&h[b*bsz + 2], &hm, 2);
    }
    void * dx; float * dsr; int8_t * dout;
    CK(cudaMalloc(&dx, bytes_in)); CK(cudaMalloc(&dsr, sizeof(float)*N)); CK(cudaMalloc(&dout, bytes_out));
    CK(cudaMemcpy(dx, h.data(), bytes_in, cudaMemcpyHostToDevice));
    const float ms_amax = time_median([&] { row_absmax_k<q5><<<N, 256>>>(dx, dsr, K); }, 10, 2);
    const int64_t nthr = (int64_t) N * nb * 32;
    const unsigned grid = (unsigned) ((nthr + 255) / 256);
    const float ms_div = time_median([&] { dequant_k_to_i8<q5, true ><<<grid, 256>>>(dx, dsr, dout, N, K); });
    const float ms_rcp = time_median([&] { dequant_k_to_i8<q5, false><<<grid, 256>>>(dx, dsr, dout, N, K); });
    const unsigned grid16 = (unsigned) (((int64_t) N * nb * 16 + 255) / 256);
    const float ms_w16 = time_median([&] { dequant_k_to_i8_w16<q5><<<grid16, 256>>>(dx, dsr, dout, N, K); });
    // check the 16-lane reciprocal variant on 64 rows (rcp vs exact division: counted separately below)
    dequant_k_to_i8_w16<q5><<<grid16, 256>>>(dx, dsr, dout, N, K);
    CK(cudaDeviceSynchronize());
    std::vector<float> sr(N); CK(cudaMemcpy(sr.data(), dsr, sizeof(float)*N, cudaMemcpyDeviceToHost));
    std::vector<int8_t> o(K);
    long bad = 0, badmax = 0; double maxdiff_s = 0;
    for (int t = 0; t < 64; ++t) {
        const int n = (int) (rng() % N);
        CK(cudaMemcpy(o.data(), dout + (size_t) n * K, K, cudaMemcpyDeviceToHost));
        float amax = 0;
        for (int v = 0; v < K; ++v) amax = std::max(amax, std::fabs(host_w(&h[((size_t) n*nb + v/256)*bsz], q5, v % 256)));
        maxdiff_s = std::max(maxdiff_s, (double) std::fabs(amax / 127.0f - sr[n]) / sr[n]);
        for (int v = 0; v < K; ++v) {
            const float w = host_w(&h[((size_t) n*nb + v/256)*bsz], q5, v % 256);
            const int e = std::max(-127, std::min(127, (int) std::nearbyint(w * (1.0f / sr[n]))));
            if (e != o[v]) { ++bad; badmax = std::max(badmax, (long) std::abs(e - o[v])); }
        }
    }
    printf("  %s %5d x %5d: in %6.1f MB out %6.1f MB | dequant div %.3f ms (%.0f GB/s) rcp %.3f ms (%.0f GB/s) | row amax "
           "(once per load) %.3f ms | 16-lane rcp %.3f ms (%.0f GB/s) | check (16-lane) 64 rows: %ld mismatches (max %ld), "
           "s_row rel diff %.1e\n",
           q5 ? "Q5_K" : "Q4_K", N, K, bytes_in/1e6, bytes_out/1e6, ms_div, (bytes_in + bytes_out)/ms_div/1e6, ms_rcp,
           (bytes_in + bytes_out)/ms_rcp/1e6, ms_amax, ms_w16, (bytes_in + bytes_out)/ms_w16/1e6, bad, badmax, maxdiff_s);
    CK(cudaFree(dx)); CK(cudaFree(dsr)); CK(cudaFree(dout));
}

template <int NV>
static void bench_tokens(int M) {
    const int K = 1024 * NV;
    std::vector<float> hx((size_t) M * K);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : hx) v = nd(rng);
    float * dx; int8_t * dq; float * ds;
    CK(cudaMalloc(&dx, hx.size()*4)); CK(cudaMalloc(&dq, hx.size())); CK(cudaMalloc(&ds, sizeof(float)*M));
    CK(cudaMemcpy(dx, hx.data(), hx.size()*4, cudaMemcpyHostToDevice));
    const float ms = time_median([&] { quant_tok<NV><<<M, 256>>>(dx, dq, ds, K); });
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> q(K); long bad = 0;
    for (int t = 0; t < 32; ++t) {
        const int m = (int) (rng() % M);
        CK(cudaMemcpy(q.data(), dq + (size_t) m * K, K, cudaMemcpyDeviceToHost));
        float amax = 0; for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(hx[(size_t) m*K + k]));
        const float inv = 127.0f / amax;
        for (int k = 0; k < K; ++k) if ((int) std::nearbyint(hx[(size_t) m*K + k] * inv) != q[k]) ++bad;
    }
    const double bytes = (double) M * K * 5;
    printf("  tokens %d x %5d f32 -> int8: %.3f ms (%.0f GB/s) | check 32 rows: %ld mismatches\n", M, K, ms, bytes/ms/1e6, bad);
    CK(cudaFree(dx)); CK(cudaFree(dq)); CK(cudaFree(ds));
}

int main() {
    printf("weights (Q4_K / Q5_K -> int8 per output row)\n");
    bench_weights<false>(4096, 4096);
    bench_weights<false>(24576, 4096);
    bench_weights<false>(4096, 12288);
    bench_weights<true>(4096, 4096);
    bench_weights<true>(16384, 4096);
    bench_weights<true>(4096, 12288);
    printf("activations (per-token int8)\n");
    bench_tokens<4>(4096);
    bench_tokens<12>(4096);
    bench_swiglu<12>(4096);
    return 0;
}
