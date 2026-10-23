// ggml-level benchmark and verification of the W8A8 path (after scratchpad qs/bench/mmbench.cpp of the coordinator):
//   w8a8bench <bin dir> mm <type> <K> <N> <T> [iters]        MUL_MAT: weights [K, N] of <type>, activations [K, T] f32
//   w8a8bench <bin dir> scalemm <type> <K> <N> <T> [iters]   SCALE(x, 1/32) + MUL_MAT (to_out's input scale)
//   w8a8bench <bin dir> mm3 <type> <K> <N> <T> [iters]       three MUL_MATs of one src1 (q, k, v), outputs concatenated
//   w8a8bench <bin dir> glu <type> <K> <N> <T> [iters]       [gate|up] MUL_MAT (N = 2*intermediate) + SWIGLU + MUL_MAT down
//   w8a8bench <bin dir> normmm3 <type> <K> <N> <T> [iters]   h = norm(x + a*b) * m (a DiT block's gated residual, layer norm
//                                                            and modulation; b, m [K] rows), then q, k, v = 3 MUL_MATs of h
//                                                            (built one after the other, as sd.cpp does with the prefix
//                                                            cache); output: the 3 products and the residual, concatenated
//   w8a8bench <bin dir> normglu <type> <K> <N> <T> [iters]   the same h into the [gate|up] + SWIGLU + down chain
// The weights are named "model.diffusion_model.transformer_blocks.0.test.weight" so that the research emulation
// (GGML_CUDA_W8A8=sub:8, its name filter) and the W8A8 path (default name filter) both take the products.
// Activations: N(0, 1), or the token rows of an engine dump (GGML_CUDA_ACT_DUMP file, W8A8BENCH_ACT=<file>, rows
// repeated to T; K must match). Prints the median time per graph, T/s, an FNV-1a hash of the output; MMBENCH_DUMP=<file>
// writes the output (f32) for comparisons (compare.py).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint64_t fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static ggml_type type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* name = ggml_type_name((ggml_type)t);
        if (name && s == name) return (ggml_type)t;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(2);
}

static void set_tensor(ggml_tensor* t, uint32_t seed, float scale) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    const char* act = getenv("W8A8BENCH_ACT");
    if (strcmp(t->name, "b_vec") == 0 || strcmp(t->name, "m_vec") == 0) {  // a gate row ~ N(0, 0.1), a modulation 1 + N(0, 0.1)
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 0.1f);
        for (auto& x : f) x = (t->name[0] == 'm' ? 1.0f : 0.0f) + nd(rng);
    } else if (t->type == GGML_TYPE_F32 && act && *act) {
        FILE* fh = fopen(act, "rb");
        if (!fh) { fprintf(stderr, "cannot open %s\n", act); exit(2); }
        int32_t hd[4]; fread(hd, 4, 4, fh);
        const int K = hd[1], nr = hd[3];
        if (K != t->ne[0] || nr <= 0) { fprintf(stderr, "dump K %d rows %d does not fit K %lld\n", K, nr, (long long) t->ne[0]); exit(2); }
        fseek(fh, 16 + 8L*K, SEEK_SET);
        std::vector<float> rows((size_t) nr * K);
        fread(rows.data(), 4, rows.size(), fh); fclose(fh);
        for (int64_t r = 0; r < t->ne[1]; ++r) memcpy(&f[r * K], &rows[(size_t) (r % nr) * K], 4 * (size_t) K);
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, scale);
        for (auto& x : f) x = nd(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1] * t->ne[2] * t->ne[3], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see the source\n"); return 2; }
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU backend\n"); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    std::string op = argv[2];

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* out = nullptr;
    std::vector<ggml_tensor*> inputs;
    std::vector<ggml_tensor*> pre_expand;  // nodes expanded into the graph before the output (their order)
    double flops = 0;
    int iters = argc > 7 ? atoi(argv[7]) : 20;

    ggml_type type = type_of(argv[3]);
    int64_t K = atoll(argv[4]), N = atoll(argv[5]), T = atoll(argv[6]);
    ggml_tensor* W = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_set_name(W, "model.diffusion_model.transformer_blocks.0.test.weight");
    ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    inputs = {W, X};
    flops = 2.0 * K * N * T;
    if (op == "mm") {
        out = ggml_mul_mat(ctx, W, X);
    } else if (op == "scalemm") {
        out = ggml_mul_mat(ctx, W, ggml_scale(ctx, X, 1.0f / 32));
    } else if (op == "mm3") {
        ggml_tensor* W2 = ggml_new_tensor_2d(ctx, type, K, N);
        ggml_tensor* W3 = ggml_new_tensor_2d(ctx, type, K, N);
        ggml_set_name(W2, "model.diffusion_model.transformer_blocks.0.test2.weight");
        ggml_set_name(W3, "model.diffusion_model.transformer_blocks.0.test3.weight");
        inputs.push_back(W2);
        inputs.push_back(W3);
        ggml_tensor* a = ggml_mul_mat(ctx, W, X);
        ggml_tensor* b = ggml_mul_mat(ctx, W2, X);
        ggml_tensor* c = ggml_mul_mat(ctx, W3, X);
        out = ggml_concat(ctx, ggml_concat(ctx, a, b, 0), c, 0);
        flops *= 3;
    } else if (op == "normmm3" || op == "normglu") {
        ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        ggml_tensor* bv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        ggml_tensor* mv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        ggml_set_name(bv, "b_vec");
        ggml_set_name(mv, "m_vec");
        inputs.push_back(A);
        inputs.push_back(bv);
        inputs.push_back(mv);
        ggml_tensor* res = ggml_add(ctx, X, ggml_mul(ctx, A, bv));        // the gated residual (kept: an output)
        ggml_tensor* h = ggml_mul(ctx, ggml_norm(ctx, res, 1e-6f), mv);    // layer norm without affine, modulated
        pre_expand.push_back(res);
        if (op == "normmm3") {
            ggml_tensor* W2 = ggml_new_tensor_2d(ctx, type, K, N);
            ggml_tensor* W3 = ggml_new_tensor_2d(ctx, type, K, N);
            ggml_set_name(W2, "model.diffusion_model.transformer_blocks.0.test2.weight");
            ggml_set_name(W3, "model.diffusion_model.transformer_blocks.0.test3.weight");
            inputs.push_back(W2);
            inputs.push_back(W3);
            ggml_tensor* q = ggml_mul_mat(ctx, W, h);
            ggml_tensor* k = ggml_mul_mat(ctx, W2, h);
            ggml_tensor* v = ggml_mul_mat(ctx, W3, h);
            pre_expand.push_back(q);
            pre_expand.push_back(k);
            pre_expand.push_back(v);
            out = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0), res, 0);
            flops *= 3;
        } else {
            ggml_tensor* gu = ggml_mul_mat(ctx, W, h);
            ggml_tensor* Wd = ggml_new_tensor_2d(ctx, type, N / 2, K);
            ggml_set_name(Wd, "model.diffusion_model.transformer_blocks.0.test_down.weight");
            inputs.push_back(Wd);
            ggml_tensor* dn = ggml_mul_mat(ctx, Wd, ggml_swiglu(ctx, gu));
            pre_expand.push_back(dn);
            out = ggml_concat(ctx, dn, res, 0);
            flops += 2.0 * (N / 2) * K * T;
        }
    } else if (op == "glu") {
        out = ggml_mul_mat(ctx, W, X);
        ggml_tensor* Wd = ggml_new_tensor_2d(ctx, type, N / 2, K);
        ggml_set_name(Wd, "model.diffusion_model.transformer_blocks.0.test_down.weight");
        inputs.push_back(Wd);
        ggml_tensor* g = ggml_swiglu(ctx, out);
        out = ggml_mul_mat(ctx, Wd, g);
        flops += 2.0 * (N / 2) * K * T;
    } else {
        fprintf(stderr, "unknown op\n");
        return 2;
    }

    size_t free0 = 0, total = 0;
    ggml_backend_dev_memory(dev, &free0, &total);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    uint32_t seed = 1234;
    for (ggml_tensor* t : inputs) {
        // weights ~ N(0, 0.02), activations ~ N(0, 1)
        set_tensor(t, seed++, t->type == GGML_TYPE_F32 ? 1.0f : 0.02f);
    }
    ggml_cgraph* gf = ggml_new_graph(ctx);
    for (ggml_tensor* t : pre_expand) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, out);

    ggml_backend_graph_compute(be, gf);  // warm-up
    ggml_backend_synchronize(be);
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    std::vector<uint8_t> host(ggml_nbytes(out));
    ggml_backend_tensor_get(out, host.data(), 0, host.size());
    double sum = 0, asum = 0;
    const float* f = (const float*)host.data();
    for (size_t i = 0; i < host.size() / 4; ++i) { sum += f[i]; asum += fabs(f[i]); }
    printf("%s %s K %lld N %lld T %lld: median %.3f ms (min %.3f), %.1f T/s, hash %016llx, sum %.6g, |sum| %.6g\n", op.c_str(),
           argv[3], (long long) K, (long long) N, (long long) T, med * 1e3, times[0] * 1e3, flops / med / 1e12,
           (unsigned long long)fnv(host.data(), host.size()), sum, asum);
    size_t free1 = 0;
    ggml_backend_dev_memory(dev, &free1, &total);
    // the tensors' buffer + the backend's pool after the runs (the pool keeps its peak) + kernels / library memory
    printf("   device memory used by this run: %.1f MiB (tensors %.1f MiB; the rest = pool peak + module / library memory)\n",
           (double) (free0 - free1) / 1048576.0, (double) ggml_backend_buffer_get_size(buf) / 1048576.0);
    if (const char* dump = getenv("MMBENCH_DUMP")) {
        FILE* fh = fopen(dump, "wb");
        if (fh) { fwrite(host.data(), 1, host.size(), fh); fclose(fh); }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
