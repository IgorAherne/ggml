// Correctness and speed of the W8 x A8-sub32 GEMM (src/ggml-cuda/w8a8.cuh; the Turing variant w8a8-turing.cuh) on random
// int8 data:
//   acc[t][n] = sum_g sy[t][g] * sum_{k in g} qx[t][k] * qw[n][k]  against an int64 reference (exact match required),
//   then the median kernel time of the f32 epilogue (CUDA events) and T/s = 2 T N K / t.
// usage: test_gemm [T N K]...          (default: the three Qwen shapes at T = 4096 + odd token counts)
//        test_gemm bench T N K [reps]   (timing only)
//        test_gemm cmp [T N K]...       (w8a8_gemm vs w8a8t_gemm on the same inputs, every epilogue and tile order: the
//                                        outputs compared byte for byte; needs both kernels compiled for the card, i.e.
//                                        an sm_80+ build with -DW8A8T_ANY_ARCH)
//        test_gemm hash [T N K]...      (the selected kernel's output hashes for every epilogue and tile order, to compare
//                                        builds: e.g. the compute_75-only binary's Turing kernel with the sm_86 binary's
//                                        w8a8_gemm)
// W8A8_KERNEL=turing selects w8a8t_gemm for the check, bench and hash modes (default: w8a8_gemm).
#include "w8a8-turing.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static bool use_turing() {
    const char * e = getenv("W8A8_KERNEL");
    return e && !strcmp(e, "turing");
}

static cudaError_t gemm(const bool turing, const int epi, const int8_t * qx, const uint8_t * sy, const float * dt, const int8_t * qw,
                        const float * dw, void * dst, const int64_t ldd, const int T, const int N, const int K, cudaStream_t stream,
                        const int group = 0) {
    return turing ? w8a8t_gemm_cuda(epi, qx, sy, dt, qw, dw, dst, ldd, T, N, K, stream, group)
                  : w8a8_gemm_cuda (epi, qx, sy, dt, qw, dw, dst, ldd, T, N, K, stream, group);
}

// exact reference for the outputs (rows[i], all n) and (all t, cols[j]): int64
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

// bytes in which a and b differ; a position-dependent 64-bit hash of a (sum of splitmix64(word ^ index*phi))
__global__ void count_diff(const uint8_t * a, const uint8_t * b, const size_t n, unsigned long long * cnt) {
    unsigned long long c = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) c += a[i] != b[i];
    if (c) atomicAdd(cnt, c);
}
__device__ __forceinline__ unsigned long long splitmix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__global__ void hash64(const unsigned long long * a, const size_t nw, unsigned long long * h) {
    unsigned long long s = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < nw; i += (size_t) gridDim.x*blockDim.x) {
        s += splitmix(a[i] ^ (i * 0x9E3779B97F4A7C15ull));
    }
    atomicAdd(h, s);
}
static unsigned long long dev_hash(const void * p, const size_t bytes) {   // bytes % 8 == 0
    unsigned long long * d; CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8));
    hash64<<<1024, 256>>>((const unsigned long long *) p, bytes / 8, d);
    unsigned long long h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); CK(cudaFree(d));
    return h;
}
static unsigned long long dev_diff(const void * a, const void * b, const size_t bytes) {
    unsigned long long * d; CK(cudaMalloc(&d, 8)); CK(cudaMemset(d, 0, 8));
    count_diff<<<1024, 256>>>((const uint8_t *) a, (const uint8_t *) b, bytes, d);
    unsigned long long h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); CK(cudaFree(d));
    return h;
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

enum test_mode { MODE_CHECK, MODE_BENCH, MODE_CMP, MODE_HASH };

static const char * epi_name(const int epi) {
    return epi == W8A8_EPI_F32 ? "f32" : epi == W8A8_EPI_GLU_F16 ? "glu-f16" : epi == W8A8_EPI_GLU_F32 ? "glu-f32" : "i32";
}

static long g_bad_total = 0;

static void run_shape(int T, int N, int K, const test_mode mode, int reps) {
    const bool turing = use_turing();
    const int64_t Tp = w8a8_pad_tokens(T);
    const int G = K / 32;
    std::vector<int8_t> hx((size_t) Tp * K, 0), hw((size_t) N * K);
    std::vector<uint8_t> hs((size_t) Tp * G);
    std::mt19937 rng(1234 + T + N + K);
    std::uniform_int_distribution<int> d8(-127, 127), dsy(1, W8A8_SY_MAX);
    for (int64_t t = 0; t < T; ++t) for (int k = 0; k < K; ++k) hx[(size_t) t*K + k] = (int8_t) d8(rng);
    for (auto & v : hw) v = (int8_t) d8(rng);
    for (auto & v : hs) v = (uint8_t) dsy(rng);
    int8_t * dx; int8_t * dw8; uint8_t * dsy_; float * ddt; float * ddw; void * dout;
    const size_t out_bytes = (size_t) T * N * 4;
    CK(cudaMalloc(&dx, hx.size())); CK(cudaMalloc(&dw8, hw.size())); CK(cudaMalloc(&dsy_, hs.size()));
    CK(cudaMalloc(&ddt, Tp * 4)); CK(cudaMalloc(&ddw, (size_t) N * 4)); CK(cudaMalloc(&dout, out_bytes));
    CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw8, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsy_, hs.data(), hs.size(), cudaMemcpyHostToDevice));
    std::vector<float> ones(std::max<int64_t>(Tp, N), 1.0f);
    CK(cudaMemcpy(ddt, ones.data(), Tp * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddw, ones.data(), (size_t) N * 4, cudaMemcpyHostToDevice));
    const double flops = 2.0 * T * (double) N * K;
    printf("T %5d N %5d K %5d:", T, N, K);
    if (mode == MODE_CHECK) {
        CK(gemm(turing, W8A8_EPI_I32, dx, dsy_, ddt, dw8, ddw, dout, N, T, N, K, 0));
        CK(cudaDeviceSynchronize());
        std::vector<int> out((size_t) T * N);
        CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
        const int nr = std::min(T, 48), nc = std::min(N, 48);
        std::vector<int> rows(nr), cols(nc);
        for (int i = 0; i < nr; ++i) rows[i] = i < 8 && T >= 8 ? (i < 4 ? i : T - 8 + i) : (int) (rng() % T);
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
        long bad = 0, tot = 0;
        for (int i = 0; i < nr; ++i) for (int n = 0; n < N; ++n, ++tot) {
            const long long e = hr[(size_t) i*N + n];
            if ((long long) out[(size_t) rows[i]*N + n] != e) { if (bad < 5) printf("\n  mismatch t %d n %d: %d vs %lld", rows[i], n, out[(size_t) rows[i]*N + n], e); ++bad; }
        }
        for (int j = 0; j < nc; ++j) for (int t = 0; t < T; ++t, ++tot) {
            const long long e = hc[(size_t) j*T + t];
            if ((long long) out[(size_t) t*N + cols[j]] != e) { if (bad < 10) printf("\n  mismatch t %d n %d: %d vs %lld", t, cols[j], out[(size_t) t*N + cols[j]], e); ++bad; }
        }
        printf(" [%s] check %ld of %ld outputs differ from the int64 reference%s", turing ? "turing" : "ampere", bad, tot,
               bad ? " <- FAIL" : " (exact)");
        g_bad_total += bad;
        CK(cudaFree(drows)); CK(cudaFree(dcols)); CK(cudaFree(dr)); CK(cudaFree(dc));
    }
    if (mode == MODE_CMP || mode == MODE_HASH) {
        // realistic scales: d_t per token in [1e-4, 1e-2) (one zero row: amax 0), dW per channel in [1e-4, 1e-2) with a few
        // large ones (the GLU f16 clamp), the padding rows' d_t 0 as the quantizer writes them
        std::uniform_real_distribution<float> du(1e-4f, 1e-2f);
        std::vector<float> hdt(Tp, 0.0f), hdw(N);
        for (int t = 0; t < T; ++t) hdt[t] = t == T / 2 ? 0.0f : du(rng);
        for (int n = 0; n < N; ++n) hdw[n] = n % 97 == 5 ? 1.0f : du(rng);
        CK(cudaMemcpy(ddt, hdt.data(), Tp * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ddw, hdw.data(), (size_t) N * 4, cudaMemcpyHostToDevice));
        void * dout2 = nullptr;
        if (mode == MODE_CMP) CK(cudaMalloc(&dout2, out_bytes));
        const int groups[] = {0, 4, -4, 1, 3, -2, -8, 16};
        long nbad = 0;
        printf("\n");
        for (const int epi : {W8A8_EPI_F32, W8A8_EPI_GLU_F16, W8A8_EPI_GLU_F32, W8A8_EPI_I32}) {
            const int64_t ldd = epi == W8A8_EPI_F32 || epi == W8A8_EPI_I32 ? N : N/2;
            for (const int group : groups) {
                if (mode == MODE_CMP) {
                    CK(cudaMemset(dout, 0xCD, out_bytes)); CK(cudaMemset(dout2, 0xCD, out_bytes));
                    CK(gemm(false, epi, dx, dsy_, ddt, dw8, ddw, dout,  ldd, T, N, K, 0, group));
                    CK(gemm(true,  epi, dx, dsy_, ddt, dw8, ddw, dout2, ldd, T, N, K, 0, group));
                    CK(cudaDeviceSynchronize());
                    const unsigned long long diff = dev_diff(dout, dout2, out_bytes);
                    const unsigned long long h1 = dev_hash(dout, out_bytes), h2 = dev_hash(dout2, out_bytes);
                    printf("  HASH T %d N %d K %d %-7s group %3d  ampere %016llx turing %016llx  %llu bytes differ%s\n", T, N, K,
                           epi_name(epi), group, h1, h2, diff, diff || h1 != h2 ? " <- FAIL" : "");
                    nbad += diff != 0 || h1 != h2;
                } else {
                    CK(cudaMemset(dout, 0xCD, out_bytes));
                    CK(gemm(turing, epi, dx, dsy_, ddt, dw8, ddw, dout, ldd, T, N, K, 0, group));
                    CK(cudaDeviceSynchronize());
                    printf("  HASH T %d N %d K %d %-7s group %3d  %s %016llx\n", T, N, K, epi_name(epi), group,
                           turing ? "turing" : "ampere", dev_hash(dout, out_bytes));
                }
            }
        }
        if (mode == MODE_CMP) {
            printf("  -> %ld of %d (epilogue, tile order) runs differ%s", nbad, 4 * (int) (sizeof(groups)/sizeof(groups[0])),
                   nbad ? " <- FAIL" : " (byte-identical)");
            g_bad_total += nbad;
            CK(cudaFree(dout2));
        }
    }
    if (mode == MODE_BENCH || (mode == MODE_CHECK && reps > 0)) {
        const int epi = getenv("W8A8_EPI") ? atoi(getenv("W8A8_EPI")) : W8A8_EPI_F32;
        const int64_t ldd = epi == W8A8_EPI_F32 ? N : N/2;
        const int group = getenv("W8A8_GROUP") ? atoi(getenv("W8A8_GROUP")) : 0;
        const float ms = time_median([&] { gemm(turing, epi, dx, dsy_, ddt, dw8, ddw, dout, ldd, T, N, K, 0, group); }, reps);
        CK(cudaGetLastError());
        printf(" | [%s] %s epilogue %.3f ms %.1f T/s", turing ? "turing" : "ampere",
               epi == W8A8_EPI_F32 ? "f32" : (epi == W8A8_EPI_GLU_F16 ? "glu-f16" : "glu-f32"), ms, flops / ms / 1e9);
    }
    printf("\n");
    fflush(stdout);
    CK(cudaFree(dx)); CK(cudaFree(dw8)); CK(cudaFree(dsy_)); CK(cudaFree(ddt)); CK(cudaFree(ddw)); CK(cudaFree(dout));
}

int main(int argc, char ** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs; tiles %dx%dx%d, warp tile %dx64, %d stages, smem %d B (turing: 2 stages, %d B, %d pieces); kernel %s\n",
           prop.name, prop.multiProcessorCount, W8A8_BM, W8A8_BN, W8A8_BK, W8A8_WM, W8A8_STAGES, w8a8_smem_bytes(),
           w8a8t_smem_bytes(), W8A8T_PIECES, use_turing() ? "turing" : "ampere");
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        const int reps = argc > 5 ? atoi(argv[5]) : 20;
        run_shape(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), MODE_BENCH, reps);
        return 0;
    }
    test_mode mode = MODE_CHECK;
    int a0 = 1;
    if (argc > 1 && !strcmp(argv[1], "cmp"))  { mode = MODE_CMP;  a0 = 2; }
    if (argc > 1 && !strcmp(argv[1], "hash")) { mode = MODE_HASH; a0 = 2; }
    std::vector<int> s = {4096, 4096, 4096,  4096, 24576, 4096,  4096, 4096, 12288,  4145, 4096, 4096,  1811, 4096, 4096,
                          8360, 4096, 4096,  4145, 24576, 4096,  1811, 4096, 12288, 100, 256, 1024};
    if (argc > a0 + 2) { s.clear(); for (int i = a0; i + 2 < argc; i += 3) { s.push_back(atoi(argv[i])); s.push_back(atoi(argv[i+1])); s.push_back(atoi(argv[i+2])); } }
    for (size_t i = 0; i + 2 < s.size(); i += 3) run_shape(s[i], s[i+1], s[i+2], mode, mode == MODE_CHECK ? 20 : 0);
    printf("TOTAL %ld failures\n", g_bad_total);
    return g_bad_total ? 1 : 0;
}
