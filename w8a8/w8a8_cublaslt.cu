// cuBLASLt int8 x int8 -> int32 (CUBLAS_COMPUTE_32I) for the W8A8 route, then the scaling kernel of common.cuh:
//   D[m][n] = float(sum_k X[m][k] * W[n][k]) * s_tok[m] * s_row[n]   (X: M x K int8 row-major, W: N x K row-major)
// layout "TN-W": A = W (K x N column-major, transa = T), B = X (K x M column-major), C = N x M column-major int32
//               = D row-major M x N (the layout the next op wants);
// layout "TN-X": A = X, B = W: C = M x N column-major (= D transposed).
// Every heuristic algorithm (up to 16, 64 MB workspace) is timed; the best and the first (default) are reported.
// usage: w8a8_cublaslt [M K N ...]
#include "common.cuh"
#include <cublasLt.h>

#define CL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLASLt error %d at %s:%d\n", (int) s_, __FILE__, __LINE__); exit(1); } } while (0)

struct LtCase {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
};

static LtCase make_case(int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC, int lda, int ldb, int ldc) {
    LtCase cs;
    CL(cublasLtMatmulDescCreate(&cs.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(cs.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CL(cublasLtMatmulDescSetAttribute(cs.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CL(cublasLtMatrixLayoutCreate(&cs.a, CUDA_R_8I, rowsA, colsA, lda));
    CL(cublasLtMatrixLayoutCreate(&cs.b, CUDA_R_8I, rowsB, colsB, ldb));
    CL(cublasLtMatrixLayoutCreate(&cs.c, CUDA_R_32I, rowsC, colsC, ldc));
    return cs;
}

int main(int argc, char ** argv) {
    std::vector<int> shapes = {4096, 4096, 4096, 4096, 4096, 24576, 4096, 12288, 4096};
    if (argc > 3) { shapes.clear(); for (int i = 1; i + 2 < argc; i += 3) { shapes.push_back(atoi(argv[i])); shapes.push_back(atoi(argv[i+1])); shapes.push_back(atoi(argv[i+2])); } }
    cublasLtHandle_t lt; CL(cublasLtCreate(&lt));
    const size_t wsz = 64u << 20;
    void * ws; CK(cudaMalloc(&ws, wsz));
    printf("cuBLASLt %d\n", (int) cublasLtGetVersion());
    for (size_t s = 0; s < shapes.size(); s += 3) {
        const int M = shapes[s], K = shapes[s+1], N = shapes[s+2];
        const double flops = 2.0 * M * (double) N * K;
        std::vector<int8_t> hx((size_t) M*K), hw((size_t) N*K);
        fill_i8(hx, 1); fill_i8(hw, 2);
        std::vector<float> htok(M), hrow(N); fill_scale(htok, 3); fill_scale(hrow, 4);
        int8_t * X; int8_t * W; int32_t * C; float * D; float * st; float * sr;
        CK(cudaMalloc(&X, hx.size())); CK(cudaMalloc(&W, hw.size()));
        CK(cudaMalloc(&C, sizeof(int32_t)*(size_t) M*N)); CK(cudaMalloc(&D, sizeof(float)*(size_t) M*N));
        CK(cudaMalloc(&st, sizeof(float)*M)); CK(cudaMalloc(&sr, sizeof(float)*N));
        CK(cudaMemcpy(X, hx.data(), hx.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(st, htok.data(), sizeof(float)*M, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sr, hrow.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
        printf("M %d K %d N %d\n", M, K, N);
        for (int variant = 0; variant < 2; ++variant) {
            // variant 0: A = W (K x N), B = X (K x M), C N x M col-major; variant 1: A = X (K x M), B = W, C M x N col-major
            LtCase cs = variant == 0 ? make_case(K, N, K, M, N, M, K, K, N) : make_case(K, M, K, N, M, N, K, K, M);
            const void * pa = variant == 0 ? (const void *) W : (const void *) X;
            const void * pb = variant == 0 ? (const void *) X : (const void *) W;
            cublasLtMatmulPreference_t pref; CL(cublasLtMatmulPreferenceCreate(&pref));
            CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
            cublasLtMatmulHeuristicResult_t res[16]; int nres = 0;
            CL(cublasLtMatmulAlgoGetHeuristic(lt, cs.op, cs.a, cs.b, cs.c, cs.c, pref, 16, res, &nres));
            const int32_t alpha = 1, beta = 0;
            float best = 1e30f, first = 0; int ibest = -1;
            for (int i = 0; i < nres; ++i) {
                auto launch = [&] { CL(cublasLtMatmul(lt, cs.op, &alpha, pa, cs.a, pb, cs.b, &beta, C, cs.c, C, cs.c,
                                                      &res[i].algo, ws, wsz, 0)); };
                const float ms = time_median(launch, 25, 3);
                if (i == 0) first = ms;
                if (ms < best) { best = ms; ibest = i; }
            }
            const char * name = variant == 0 ? "TN-W (D row-major M x N)" : "TN-X (D col-major M x N)";
            printf("  %-26s %2d algos: first %8.3f ms %6.1f T/s | best #%d %8.3f ms %6.1f T/s (%.1f %% of 102)\n", name, nres,
                   first, flops/first/1e9, ibest, best, flops/best/1e9, flops/best/1e9/102*100);
            if (variant == 0 && ibest >= 0) {
                auto gemm = [&] { CL(cublasLtMatmul(lt, cs.op, &alpha, pa, cs.a, pb, cs.b, &beta, C, cs.c, C, cs.c,
                                                    &res[ibest].algo, ws, wsz, 0)); };
                const float ms_s = time_median([&] { launch_scale<int32_t>(C, D, st, sr, M, N); });
                const float ms_b = time_median([&] { gemm(); launch_scale<int32_t>(C, D, st, sr, M, N); });
                gemm(); launch_scale<int32_t>(C, D, st, sr, M, N); CK(cudaDeviceSynchronize());
                const double err = check_sample(X, W, D, htok, hrow, M, N, K);
                printf("  %-26s + scale kernel (int32 -> f32) %.3f ms: GEMM + scale %8.3f ms %6.1f T/s  maxrel %.1e\n", "",
                       ms_s, ms_b, flops/ms_b/1e9, err);
            }
            CL(cublasLtMatmulPreferenceDestroy(pref));
            CL(cublasLtMatrixLayoutDestroy(cs.a)); CL(cublasLtMatrixLayoutDestroy(cs.b)); CL(cublasLtMatrixLayoutDestroy(cs.c));
            CL(cublasLtMatmulDescDestroy(cs.op));
        }
        CK(cudaFree(X)); CK(cudaFree(W)); CK(cudaFree(C)); CK(cudaFree(D)); CK(cudaFree(st)); CK(cudaFree(sr));
    }
    CK(cudaFree(ws));
    CL(cublasLtDestroy(lt));
    return 0;
}
