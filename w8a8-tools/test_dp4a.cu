// Correctness and speed of the dp4a W8A8 GEMM (src/ggml-cuda/w8a8-dp4a.cuh, cards without int8 tensor cores) on random
// int8 data:
//   - the int32 accumulators (W8A8_EPI_I32; exact mode only) against an int64 reference on sampled rows and columns;
//   - the f32 epilogue against y = acc * d_t/31 * dW_n from the int64 reference (max relative error);
//   - on an sm_80+ card also against the tensor-core kernel w8a8_gemm, byte for byte (f32 and GLU f16 epilogues);
//   - the median kernel time, T/s = 2 T N K / t.
// usage: test_dp4a [T N K]...        (default: the Qwen shapes at T = 4096 + odd token counts)
//        test_dp4a bench T N K [reps] (timing only; W8A8_EPI=0|1|2 the epilogue, W8A8_GROUP the tile order)
#include "w8a8-dp4a.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void ref_rows(const int8_t * qx, const uint8_t * sy, const int8_t * qw, const int * rows, int nrows, int N, int K,
                         long long * out) {
    const int i = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows || n >= N) return;
    const int t = rows[i];
    const int G = K / 32;
    long long acc = 0;
    for (int g = 0; g < G; ++g) {
        int d = 0;
        for (int k = 0; k < 32; ++k) d += (int) qx[(long long) t*K + g*32 + k] * (int) qw[(long long) n*K + g*32 + k];
        acc += (long long) sy[((long long) (t / W8A8_BM) * G + g) * W8A8_BM + w8a8_sy_pos(t % W8A8_BM)] * d;
    }
    out[(long long) i*N + n] = acc;
}
__global__ void ref_cols(const int8_t * qx, const uint8_t * sy, const int8_t * qw, const int * cols, int ncols, int T, int K,
                         long long * out) {
    const int j = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ncols || t >= T) return;
    const int n = cols[j];
    const int G = K / 32;
    long long acc = 0;
    for (int g = 0; g < G; ++g) {
        int d = 0;
        for (int k = 0; k < 32; ++k) d += (int) qx[(long long) t*K + g*32 + k] * (int) qw[(long long) n*K + g*32 + k];
        acc += (long long) sy[((long long) (t / W8A8_BM) * G + g) * W8A8_BM + w8a8_sy_pos(t % W8A8_BM)] * d;
    }
    out[(long long) j*T + t] = acc;
}

template <class F>
static float time_median(F && f, int n = 20, int warm = 3) {
    for (int i = 0; i < warm; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> v;
    for (int i = 0; i < n; ++i) {
        CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static bool g_tc = false;   // an sm_80+ card: compare with w8a8_gemm too

static void run_shape(int T, int N, int K, bool check, int reps) {
    const int64_t Tp = w8a8_pad_tokens(T);
    const int G = K / 32;
    std::vector<int8_t> hx((size_t) Tp * K, 0), hw((size_t) N * K);
    std::vector<uint8_t> hs((size_t) Tp * G);
    std::vector<float> hdt(Tp, 0.0f), hdw(N);
    std::mt19937 rng(1234 + T + N + K);
    std::uniform_int_distribution<int> d8(-127, 127), dsy(1, W8A8_SY_MAX);
    std::uniform_real_distribution<float> dsc(0.5f, 2.0f);
    for (int64_t t = 0; t < T; ++t) for (int k = 0; k < K; ++k) hx[(size_t) t*K + k] = (int8_t) d8(rng);
    for (auto & v : hw) v = (int8_t) d8(rng);
    for (auto & v : hs) v = (uint8_t) dsy(rng);
    for (int t = 0; t < T; ++t) hdt[t] = dsc(rng) * 1e-3f;
    for (auto & v : hdw) v = dsc(rng) * 1e-3f;
    int8_t * dx; int8_t * dw8; uint8_t * dsy_; float * ddt; float * ddw; void * dout; void * dout2;
    CK(cudaMalloc(&dx, hx.size())); CK(cudaMalloc(&dw8, hw.size())); CK(cudaMalloc(&dsy_, hs.size()));
    CK(cudaMalloc(&ddt, Tp * 4)); CK(cudaMalloc(&ddw, (size_t) N * 4));
    CK(cudaMalloc(&dout, (size_t) T * N * 4)); CK(cudaMalloc(&dout2, (size_t) T * N * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw8, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsy_, hs.data(), hs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddt, hdt.data(), Tp * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddw, hdw.data(), (size_t) N * 4, cudaMemcpyHostToDevice));
    const double flops = 2.0 * T * (double) N * K;
    printf("T %5d N %5d K %5d:", T, N, K);
    if (check) {
        const int nr = std::min(T, 48), nc = std::min(N, 48);
        std::vector<int> rows(nr), cols(nc);
        for (int i = 0; i < nr; ++i) rows[i] = i < 8 ? (i < 4 ? i : T - 8 + i) : (int) (rng() % T);
        for (int j = 0; j < nc; ++j) cols[j] = j < 4 ? j : (j < 8 ? N - 8 + j : (int) (rng() % N));
        int * drows; int * dcols; long long * dr; long long * dc;
        CK(cudaMalloc(&drows, nr * 4)); CK(cudaMalloc(&dcols, nc * 4));
        CK(cudaMalloc(&dr, (size_t) nr * N * 8)); CK(cudaMalloc(&dc, (size_t) nc * T * 8));
        CK(cudaMemcpy(drows, rows.data(), nr * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcols, cols.data(), nc * 4, cudaMemcpyHostToDevice));
        ref_rows<<<dim3((N + 127) / 128, nr), 128>>>(dx, dsy_, dw8, drows, nr, N, K, dr);
        ref_cols<<<dim3((T + 127) / 128, nc), 128>>>(dx, dsy_, dw8, dcols, nc, T, K, dc);
        CK(cudaDeviceSynchronize());
        std::vector<long long> hr((size_t) nr * N), hc((size_t) nc * T);
        CK(cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), dc, hc.size() * 8, cudaMemcpyDeviceToHost));
        auto each = [&](auto && f) {   // f(t, n, exact int64)
            for (int i = 0; i < nr; ++i) for (int n = 0; n < N; ++n) f(rows[i], n, hr[(size_t) i*N + n]);
            for (int j = 0; j < nc; ++j) for (int t = 0; t < T; ++t) f(t, cols[j], hc[(size_t) j*T + t]);
        };
#if !W8A8D_SCALE_F32
        CK(w8a8d_gemm_cuda(W8A8_EPI_I32, dx, dsy_, ddt, dw8, ddw, dout, N, T, N, K, 0));
        CK(cudaDeviceSynchronize());
        {
            std::vector<int> out((size_t) T * N);
            CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
            long bad = 0, tot = 0;
            each([&](int t, int n, long long e) {
                ++tot;
                if ((long long) out[(size_t) t*N + n] != e) { if (bad < 5) printf("\n  mismatch t %d n %d: %d vs %lld", t, n, out[(size_t) t*N + n], e); ++bad; }
            });
            printf(" i32 %ld/%ld differ%s", bad, tot, bad ? " <- FAIL" : " (exact)");
        }
#endif
        CK(w8a8d_gemm_cuda(W8A8_EPI_F32, dx, dsy_, ddt, dw8, ddw, dout, N, T, N, K, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> outf((size_t) T * N);
        CK(cudaMemcpy(outf.data(), dout, outf.size() * 4, cudaMemcpyDeviceToHost));
        double maxrel = 0.0;
        each([&](int t, int n, long long e) {
            const double y = (double) e * hdt[t] / W8A8_SY_MAX * hdw[n];
            const double d = std::fabs((double) outf[(size_t) t*N + n] - y) / std::max(std::fabs(y), 1e-30);
            maxrel = std::max(maxrel, d);
        });
        printf(" | f32 max rel err %.2e", maxrel);
        if (g_tc) {
            CK(w8a8_gemm_cuda(W8A8_EPI_F32, dx, dsy_, ddt, dw8, ddw, dout2, N, T, N, K, 0));
            CK(cudaDeviceSynchronize());
            std::vector<float> o2((size_t) T * N);
            CK(cudaMemcpy(o2.data(), dout2, o2.size() * 4, cudaMemcpyDeviceToHost));
            long diff = 0;
            for (size_t i = 0; i < o2.size(); ++i) diff += memcmp(&o2[i], &outf[i], 4) != 0;
            printf(" | vs tensor-core f32: %ld of %zu differ", diff, o2.size());
            if (N % 256 == 0) {
                CK(w8a8d_gemm_cuda(W8A8_EPI_GLU_F16, dx, dsy_, ddt, dw8, ddw, dout, N/2, T, N, K, 0));
                CK(w8a8_gemm_cuda(W8A8_EPI_GLU_F16, dx, dsy_, ddt, dw8, ddw, dout2, N/2, T, N, K, 0));
                CK(cudaDeviceSynchronize());
                std::vector<uint16_t> g1((size_t) T * N/2), g2((size_t) T * N/2);
                CK(cudaMemcpy(g1.data(), dout, g1.size() * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(g2.data(), dout2, g2.size() * 2, cudaMemcpyDeviceToHost));
                long gd = 0;
                for (size_t i = 0; i < g1.size(); ++i) gd += g1[i] != g2[i];
                printf(", glu-f16 %ld of %zu differ", gd, g1.size());
            }
        }
        CK(cudaFree(drows)); CK(cudaFree(dcols)); CK(cudaFree(dr)); CK(cudaFree(dc));
    }
    if (reps > 0) {
        const int epi = getenv("W8A8_EPI") ? atoi(getenv("W8A8_EPI")) : W8A8_EPI_F32;
        const int64_t ldd = epi == W8A8_EPI_F32 ? N : N/2;
        const int group = getenv("W8A8_GROUP") ? atoi(getenv("W8A8_GROUP")) : 0;
        const float ms = time_median([&] { CK(w8a8d_gemm_cuda(epi, dx, dsy_, ddt, dw8, ddw, dout, ldd, T, N, K, 0, group)); }, reps);
        printf(" | dp4a %s %.3f ms %.2f T/s", epi == W8A8_EPI_F32 ? "f32" : (epi == W8A8_EPI_GLU_F16 ? "glu-f16" : "glu-f32"),
               ms, flops / ms / 1e9);
    }
    printf("\n");
    CK(cudaFree(dx)); CK(cudaFree(dw8)); CK(cudaFree(dsy_)); CK(cudaFree(ddt)); CK(cudaFree(ddw)); CK(cudaFree(dout)); CK(cudaFree(dout2));
}

int main(int argc, char ** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_tc = prop.major >= 8 && !getenv("W8A8_NO_TC");
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, w8a8d_gemm<W8A8_EPI_F32>));
    printf("%s (cc %d.%d), %d SMs; dp4a kernel %dx%dx%d, %d threads, %d registers, %zu B shared, scale %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, W8A8D_BM, W8A8D_BN, W8A8D_BK, W8A8D_NTHREADS, fa.numRegs, fa.sharedSizeBytes,
           W8A8D_SCALE_F32 ? "f32" : "int32 (exact)");
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        const int reps = argc > 5 ? atoi(argv[5]) : 20;
        run_shape(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), false, reps);
        return 0;
    }
    std::vector<int> s = {4096, 4096, 4096,  4096, 24576, 4096,  4096, 4096, 12288,  4145, 4096, 4096,  1811, 4096, 4096,
                          8360, 4096, 4096,  4145, 24576, 4096,  1811, 4096, 12288, 100, 256, 1024};
    if (argc > 3) { s.clear(); for (int i = 1; i + 2 < argc; i += 3) { s.push_back(atoi(argv[i])); s.push_back(atoi(argv[i+1])); s.push_back(atoi(argv[i+2])); } }
    for (size_t i = 0; i + 2 < s.size(); i += 3) run_shape(s[i], s[i+1], s[i+2], true, 10);
    return 0;
}
