// Micro-benchmark of single ggml ops with Qwen-Image 2.1 shapes, against the DLLs of a stable-diffusion.cpp build.
//   mmbench <bin dir> mm <type> <K> <N> <T> [iters]       MUL_MAT: weights [K, N] of <type>, activations [K, T] f32
//   mmbench <bin dir> fa <D> <n_q> <n_kv> <heads> [iters]  FLASH_ATTN_EXT: q f32, k/v f16, no mask
//   mmbench <bin dir> glu <type> <K> <N> <T> [iters]      [gate|up] MUL_MAT (N = 2*intermediate) + SWIGLU + MUL_MAT down
// Prints the median time per op, the throughput and an FNV-1a hash of the output (bit-identity between builds).
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

static void fill_f32(std::vector<float>& v, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& x : v) x = nd(rng);
}

static void set_tensor(ggml_tensor* t, uint32_t seed, float scale) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    fill_f32(f, seed, scale);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1] * t->ne[2] * t->ne[3], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: see the source\n"); return 2; }
    ggml_backend_load_all_from_path(argv[1]);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU backend\n"); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    std::string op = argv[2];

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* out = nullptr;
    std::vector<ggml_tensor*> inputs;
    double flops = 0;
    int iters = 20;

    if (op == "mm" || op == "glu") {
        ggml_type type = type_of(argv[3]);
        int64_t K = atoll(argv[4]), N = atoll(argv[5]), T = atoll(argv[6]);
        if (argc > 7) iters = atoi(argv[7]);
        ggml_tensor* W = ggml_new_tensor_2d(ctx, type, K, N);
        ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        inputs = {W, X};
        out = ggml_mul_mat(ctx, W, X);
        flops = 2.0 * K * N * T;
        if (op == "glu") {
            ggml_tensor* Wd = ggml_new_tensor_2d(ctx, type, N / 2, K);
            inputs.push_back(Wd);
            ggml_tensor* g = ggml_swiglu(ctx, out);
            out = ggml_mul_mat(ctx, Wd, g);
            flops += 2.0 * (N / 2) * K * T;
        }
    } else if (op == "fa") {
        int64_t D = atoll(argv[3]), nq = atoll(argv[4]), nkv = atoll(argv[5]), H = atoll(argv[6]);
        if (argc > 7) iters = atoi(argv[7]);
        ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, H);
        ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, nkv, H);
        ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, nkv, H);
        inputs = {q, k, v};
        out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / sqrtf((float)D), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        flops = 4.0 * D * nq * nkv * H;
    } else if (op == "sage") {
        // ggml_sage_attn: q/k F32 [D, tokens, heads], v F16 [D, tokens, heads]; output F32 [D, heads, tokens]
        int64_t D = atoll(argv[3]), nq = atoll(argv[4]), nkv = atoll(argv[5]), H = atoll(argv[6]);
        if (argc > 7) iters = atoi(argv[7]);
        int mode = argc > 8 ? atoi(argv[8]) : 0;
        ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, H);
        ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nkv, H);
        ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, nkv, H);
        inputs = {q, k, v};
        out = ggml_sage_attn(ctx, q, k, v, 1.0f / sqrtf((float)D), (ggml_sage_attn_mode)mode);
        flops = 4.0 * D * nq * nkv * H;
    } else if (op == "fap") {
        // flash attention on the sage op's inputs (q/k F32, v F16 [D, tokens, heads] -> permuted to [D, tokens, heads]
        // as ggml_flash_attn_ext wants them: the same memory): the reference for the sage op's output
        int64_t D = atoll(argv[3]), nq = atoll(argv[4]), nkv = atoll(argv[5]), H = atoll(argv[6]);
        if (argc > 7) iters = atoi(argv[7]);
        ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, H);
        ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nkv, H);
        ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, nkv, H);
        inputs = {q, k, v};
        ggml_tensor* kh = ggml_cast(ctx, k, GGML_TYPE_F16);
        out = ggml_flash_attn_ext(ctx, q, kh, v, nullptr, 1.0f / sqrtf((float)D), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        flops = 4.0 * D * nq * nkv * H;
    } else {
        fprintf(stderr, "unknown op\n");
        return 2;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    uint32_t seed = 1234;
    for (ggml_tensor* t : inputs) {
        // weights ~ N(0, 0.02), activations ~ N(0, 1), q/k/v ~ N(0, 1)
        set_tensor(t, seed++, (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) ? 1.0f : 0.02f);
    }
    ggml_cgraph* gf = ggml_new_graph(ctx);
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
    if (out->type == GGML_TYPE_F32) {
        const float* f = (const float*)host.data();
        for (size_t i = 0; i < host.size() / 4; ++i) { sum += f[i]; asum += fabs(f[i]); }
    }
    printf("%s %s: median %.3f ms (min %.3f), %.1f T/s, hash %016llx, sum %.6g, |sum| %.6g\n", op.c_str(),
           argc > 3 ? argv[3] : "", med * 1e3, times[0] * 1e3, flops / med / 1e12, (unsigned long long)fnv(host.data(), host.size()),
           sum, asum);
    // dump the output for accuracy comparisons between builds: MMBENCH_DUMP=<file>
    if (const char* dump = getenv("MMBENCH_DUMP")) {
        FILE* fh = fopen(dump, "wb");
        if (fh) { fwrite(host.data(), 1, host.size(), fh); fclose(fh); }
    }
    // dump the inputs (raw tensor bytes) for reference computations: MMBENCH_DUMP_IN=<prefix> -> <prefix>_in<k>.bin
    if (const char* pre = getenv("MMBENCH_DUMP_IN")) {
        for (size_t k = 0; k < inputs.size(); ++k) {
            std::vector<uint8_t> hb(ggml_nbytes(inputs[k]));
            ggml_backend_tensor_get(inputs[k], hb.data(), 0, hb.size());
            std::string fn = std::string(pre) + "_in" + std::to_string(k) + ".bin";
            FILE* fh = fopen(fn.c_str(), "wb");
            if (fh) { fwrite(hb.data(), 1, hb.size(), fh); fclose(fh); }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
