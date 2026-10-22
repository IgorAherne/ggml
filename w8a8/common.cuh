// Shared helpers of the W8A8 speed probes (timing, data, the scaling kernel, the exact reference on a sample).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// median of n timed runs (CUDA events around each launch), after `warm` untimed runs
template <class F>
static float time_median(F && launch, int n = 30, int warm = 5) {
    for (int i = 0; i < warm; ++i) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> t;
    for (int i = 0; i < n; ++i) {
        CK(cudaEventRecord(e0));
        launch();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void fill_i8(std::vector<int8_t> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(-127, 127);
    for (auto & x : v) x = (int8_t) d(rng);
}
static void fill_scale(std::vector<float> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(0.5e-3f, 1.5e-3f);
    for (auto & x : v) x = d(rng);
}

// D[m][n] = float(C[m][n]) * s_tok[m] * s_row[n] (C int32 or f32, D f32, row-major M x N; in place allowed)
template <typename T>
__global__ void scale_rows_cols(const T * __restrict__ C, float * __restrict__ D, const float * __restrict__ s_tok,
                                const float * __restrict__ s_row, const int M, const int N) {
    const int64_t i4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; // index of a 4-element group
    const int64_t i  = 4*i4;
    if (i >= (int64_t) M * N) return;
    const int m = (int) (i / N);
    const int n = (int) (i % N);
    const float st = s_tok[m];
    const float4 sr = *(const float4 *) (s_row + n);
    float4 c;
    if constexpr (std::is_same<T, int32_t>::value) {
        const int4 v = *(const int4 *) (C + i);
        c = make_float4((float) v.x, (float) v.y, (float) v.z, (float) v.w);
    } else {
        c = *(const float4 *) (C + i);
    }
    *(float4 *) (D + i) = make_float4(c.x*st*sr.x, c.y*st*sr.y, c.z*st*sr.z, c.w*st*sr.w);
}

template <typename T>
static void launch_scale(const T * C, float * D, const float * s_tok, const float * s_row, int M, int N, cudaStream_t s = 0) {
    const int64_t n4 = (int64_t) M * N / 4;
    scale_rows_cols<T><<<(unsigned) ((n4 + 255) / 256), 256, 0, s>>>(C, D, s_tok, s_row, M, N);
}

// exact reference of nsamp outputs: ref[i] = sum_k x[m][k] * w[n][k] (int64), (m, n) = samples
__global__ void ref_dot(const int8_t * __restrict__ X, const int8_t * __restrict__ W, const int2 * __restrict__ mn,
                        long long * __restrict__ ref, const int K, const int nsamp) {
    const int s = blockIdx.x;
    if (s >= nsamp) return;
    const int m = mn[s].x, n = mn[s].y;
    long long acc = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) acc += (int) X[(int64_t) m*K + k] * (int) W[(int64_t) n*K + k];
    __shared__ long long red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) ref[s] = red[0];
}

// checks D (f32, row-major M x N) against the exact reference * s_tok * s_row on nsamp random outputs; returns max rel err
static double check_sample(const int8_t * dX, const int8_t * dW, const float * dD, const std::vector<float> & s_tok,
                           const std::vector<float> & s_row, int M, int N, int K, bool scaled = true) {
    const int nsamp = 2048;
    std::vector<int2> mn(nsamp);
    std::mt19937 rng(7);
    for (auto & p : mn) { p.x = (int) (rng() % M); p.y = (int) (rng() % N); }
    int2 * dmn; long long * dref;
    CK(cudaMalloc(&dmn, nsamp*sizeof(int2))); CK(cudaMalloc(&dref, nsamp*sizeof(long long)));
    CK(cudaMemcpy(dmn, mn.data(), nsamp*sizeof(int2), cudaMemcpyHostToDevice));
    ref_dot<<<nsamp, 256>>>(dX, dW, dmn, dref, K, nsamp);
    std::vector<long long> ref(nsamp);
    CK(cudaMemcpy(ref.data(), dref, nsamp*sizeof(long long), cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (int s = 0; s < nsamp; ++s) {
        float d; CK(cudaMemcpy(&d, dD + (int64_t) mn[s].x*N + mn[s].y, sizeof(float), cudaMemcpyDeviceToHost));
        const double e = scaled ? (double) ref[s] * s_tok[mn[s].x] * s_row[mn[s].y] : (double) ref[s];
        const double scale = scaled ? 1e-6 * std::sqrt((double) K) * 127 * 127 / 3 : std::sqrt((double) K) * 127 * 127 / 3; // ~ rms
        maxrel = std::max(maxrel, std::abs(d - e) / std::max(std::abs(e), scale));
    }
    CK(cudaFree(dmn)); CK(cudaFree(dref));
    return maxrel;
}
