// Spellcast3D: numeric check of the CUDA flash attention with INT8 (dp4a) scores (ggml-cuda/fattn-tile-dp4a.cu)
// against an F64 CPU reference, beside the F32 tile kernel and CPU emulations of two other roundings of Q and K.
//
// Run with the GTX 10xx dispatch (a real cc 6.1 card, or GGML_CUDA_FAKE_CC=610 with a compute_61 build), once as is
// (dp4a) and once with GGML_CUDA_FATTN_DP4A=off (the F32 tile kernel):
//   test_fattn_dp4a <backend dir> [nq nk nh nkh] [--mask] [--f32kv] [--bias B] [--seed S] [--rows N] [--out file]
// Q / K rows of norm ~12..36 (a few outlier channels five times the others), a per-channel K offset shared by all keys
// (bias B, default 1.5), V ~ N(0, 1). Q F32, K F16 (or F32 with --f32kv), V F16 as a permuted view of
// [D, heads, keys] memory (F32 with --f32kv), the way Qwen-Image 2.1 hands them over.
// Reported per sampled output row (query, head): the relative L2 error against the F64 reference, max and median.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Stats {
    double max_rel = 0.0, median_rel = 0.0, p99_rel = 0.0, max_abs = 0.0;
};

static Stats summarize(std::vector<double> rel, double max_abs) {
    Stats s;
    if (rel.empty()) {
        return s;
    }
    std::sort(rel.begin(), rel.end());
    s.max_rel    = rel.back();
    s.median_rel = rel[rel.size() / 2];
    s.p99_rel    = rel[std::min(rel.size() - 1, size_t(rel.size() * 0.99))];
    s.max_abs    = max_abs;
    return s;
}

static float round_f16(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

// INT8 per group of rows with one scale (absolute maximum / 127), round to nearest; rows: n x 128.
static void quantize_groups(std::vector<double> & x, int n, int group) {
    for (int r0 = 0; r0 < n; r0 += group) {
        double amax = 0.0;
        for (int r = r0; r < std::min(n, r0 + group); ++r) {
            for (int d = 0; d < 128; ++d) {
                amax = std::max(amax, std::fabs(x[size_t(r) * 128 + d]));
            }
        }
        const double s = amax > 0 ? amax / 127.0 : 0.0;
        for (int r = r0; r < std::min(n, r0 + group); ++r) {
            for (int d = 0; d < 128; ++d) {
                double & v = x[size_t(r) * 128 + d];
                v          = s > 0 ? std::nearbyint(v / s) * s : 0.0;
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend dir> [nq nk nh nkh] [--mask] [--f32kv] [--bias B] [--seed S] [--rows N] [--out file]\n", argv[0]);
        return 2;
    }
    const char * backend_dir = argv[1];
    int nq = 4096, nk = 4145, nh = 32, nkh = 32;
    int argi = 2;
    if (argc >= 6 && argv[2][0] != '-') {
        nq  = atoi(argv[2]);
        nk  = atoi(argv[3]);
        nh  = atoi(argv[4]);
        nkh = atoi(argv[5]);
        argi = 6;
    }
    bool use_mask = false, f32kv = false, exact = false, causal = false;
    float scale_arg = 0.0f;
    double bias = 1.5;
    unsigned seed = 1234;
    int rows_per_head = 64;
    int bench = 0;
    int ns = 1; // sequences (ne[3]); the mask is [nk, nq, 1, ns] with the same values in every sequence
    float max_bias = 0.0f, softcap = 0.0f;
    std::string out_file;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--mask") {
            use_mask = true;
        } else if (a == "--f32kv") {
            f32kv = true;
        } else if (a == "--causal") {
            use_mask = causal = true; // key s visible to query t if s <= t (fully masked key tiles: the KV_max path)
        } else if (a == "--exact") {
            exact = true; // integer Q / K rows with |max| = 127 and K pairs t, t + nk/2 of opposite sign (key mean 0):
                          // the INT8 rows are exact, the dp4a kernel must match the reference to F32 rounding
        } else if (a == "--scale" && argi + 1 < argc) {
            scale_arg = float(atof(argv[++argi]));
        } else if (a == "--bias" && argi + 1 < argc) {
            bias = atof(argv[++argi]);
        } else if (a == "--seed" && argi + 1 < argc) {
            seed = unsigned(atoi(argv[++argi]));
        } else if (a == "--alibi" && argi + 1 < argc) {
            max_bias = float(atof(argv[++argi])); // ALiBi (needs --mask or --causal)
        } else if (a == "--softcap" && argi + 1 < argc) {
            softcap = float(atof(argv[++argi])); // logit softcap: the dp4a path must decline (F32 tile kernel)
        } else if (a == "--nseq" && argi + 1 < argc) {
            ns = atoi(argv[++argi]);
        } else if (a == "--bench" && argi + 1 < argc) {
            bench = atoi(argv[++argi]); // time this many runs of the op, no reference
        } else if (a == "--rows" && argi + 1 < argc) {
            rows_per_head = atoi(argv[++argi]);
        } else if (a == "--out" && argi + 1 < argc) {
            out_file = argv[++argi];
        }
    }
    const int D = 128;
    const float scale = scale_arg != 0.0f ? scale_arg : 1.0f / std::sqrt(float(D));

    ggml_backend_load_all_from_path(backend_dir);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    // Data (host, as the kernels will see it: K / V rounded to F16 unless --f32kv).
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.8f, 2.4f);
    std::vector<float> sigma(D, 1.0f);
    for (int i = 0; i < 4; ++i) {
        sigma[(rng() % (D / 2)) * 2 + (i & 1)] = 5.0f; // outlier channels
    }
    // Heads are numbered across the sequences: H = sequence*nh + head (Q), sequence*nkh + head (K, V).
    std::vector<float> q(size_t(nq) * nh * ns * D), k(size_t(nk) * nkh * ns * D), v(size_t(nk) * nkh * ns * D);
    // Q: [D, nq, nh, ns] contiguous
    for (int h = 0; h < nh * ns; ++h) {
        for (int t = 0; t < nq; ++t) {
            const float r = uni(rng);
            for (int d = 0; d < D; ++d) {
                q[(size_t(h) * nq + t) * D + d] = normal(rng) * sigma[d] * r;
            }
        }
    }
    // K: [D, nk, nkh, ns] contiguous, a per-head channel offset
    for (int h = 0; h < nkh * ns; ++h) {
        std::vector<float> off(D);
        for (int d = 0; d < D; ++d) {
            off[d] = float(bias) * normal(rng);
        }
        for (int t = 0; t < nk; ++t) {
            const float r = uni(rng);
            for (int d = 0; d < D; ++d) {
                float x = normal(rng) * sigma[d] * r + off[d];
                k[(size_t(h) * nk + t) * D + d] = f32kv ? x : round_f16(x);
            }
        }
    }
    // V memory: [D, nkh, nk] (keys outermost), read as the permuted view [D, nk, nkh]
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = normal(rng);
        v[i] = f32kv ? x : round_f16(x);
    }
    if (exact) {
        std::uniform_int_distribution<int> ui(-60, 60);
        for (size_t r = 0; r < q.size() / D; ++r) {
            for (int d = 0; d < D; ++d) q[r * D + d] = float(ui(rng));
            q[r * D + (r % D)] = (r & 1) ? 127.0f : -127.0f;
        }
        for (int h = 0; h < nkh * ns; ++h) {
            for (int t = 0; t < nk / 2; ++t) {
                for (int d = 0; d < D; ++d) k[(size_t(h) * nk + t) * D + d] = float(ui(rng));
                k[(size_t(h) * nk + t) * D + (t % D)] = (t & 1) ? -127.0f : 127.0f;
                for (int d = 0; d < D; ++d) k[(size_t(h) * nk + t + nk / 2) * D + d] = -k[(size_t(h) * nk + t) * D + d];
            }
            if (nk % 2) for (int d = 0; d < D; ++d) k[(size_t(h) * nk + nk - 1) * D + d] = 0.0f;
        }
    }
    // mask [nk, nq]: blocks of -inf, some ALiBi-like gradient
    std::vector<float> maskf;
    if (use_mask) {
        maskf.resize(size_t(nq) * nk);
        for (int t = 0; t < nq; ++t) {
            for (int s = 0; s < nk; ++s) {
                float m = -0.002f * float(std::abs(t - s) % 97);
                if (causal ? s > t : ((t / 37) + (s / 211)) % 5 == 0) {
                    m = -INFINITY;
                }
                if (s == t % nk) {
                    m = 0.0f; // every query sees at least one key
                }
                maskf[size_t(t) * nk + s] = round_f16(m);
            }
        }
    }

    // Graph
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, nh, ns);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, f32kv ? GGML_TYPE_F32 : GGML_TYPE_F16, D, nk, nkh, ns);
    ggml_tensor * tvm = ggml_new_tensor_4d(ctx, f32kv ? GGML_TYPE_F32 : GGML_TYPE_F16, D, nkh, nk, ns);
    ggml_tensor * tv = ggml_permute(ctx, tvm, 0, 2, 1, 3); // [D, nk, nkh, ns]
    ggml_tensor * tm = use_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nk, nq, 1, ns) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale, max_bias, softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_input(tq);
    ggml_set_input(tk);
    ggml_set_input(tvm);
    if (tm) {
        ggml_set_input(tm);
    }
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "flash attention not supported\n");
        return 1;
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "alloc failed\n");
        return 1;
    }
    ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
    if (f32kv) {
        ggml_backend_tensor_set(tk, k.data(), 0, ggml_nbytes(tk));
        // V memory [D, nkh, nk, ns] from v laid out [ns][nkh][nk][D]
        std::vector<float> vm(v.size());
        for (int b = 0; b < ns; ++b) for (int h = 0; h < nkh; ++h) for (int t = 0; t < nk; ++t) for (int d = 0; d < D; ++d)
            vm[((size_t(b) * nk + t) * nkh + h) * D + d] = v[((size_t(b) * nkh + h) * nk + t) * D + d];
        ggml_backend_tensor_set(tvm, vm.data(), 0, ggml_nbytes(tvm));
    } else {
        std::vector<ggml_fp16_t> kh(k.size()), vm(v.size());
        ggml_fp32_to_fp16_row(k.data(), kh.data(), k.size());
        for (int b = 0; b < ns; ++b) for (int h = 0; h < nkh; ++h) for (int t = 0; t < nk; ++t) for (int d = 0; d < D; ++d)
            vm[((size_t(b) * nk + t) * nkh + h) * D + d] = ggml_fp32_to_fp16(v[((size_t(b) * nkh + h) * nk + t) * D + d]);
        ggml_backend_tensor_set(tk, kh.data(), 0, ggml_nbytes(tk));
        ggml_backend_tensor_set(tvm, vm.data(), 0, ggml_nbytes(tvm));
    }
    if (tm) {
        std::vector<ggml_fp16_t> mh(maskf.size() * ns);
        for (int b = 0; b < ns; ++b) ggml_fp32_to_fp16_row(maskf.data(), mh.data() + b * maskf.size(), maskf.size());
        ggml_backend_tensor_set(tm, mh.data(), 0, ggml_nbytes(tm));
    }

    const char * env = getenv("GGML_CUDA_FATTN_DP4A");
    const bool dp4a_on = !(env && std::string(env) == "off");
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ok = ok && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS; // twice: the pool reused
    if (bench > 0) {
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) {
            ok = ok && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / bench;
        std::printf("nq=%d nk=%d nh=%d nkh=%d mask=%d  GGML_CUDA_FATTN_DP4A=%s  %.3f ms per op (%d runs)\n",
            nq, nk, nh, nkh, int(use_mask), dp4a_on ? "(on)" : "off", ms, bench);
        return ok ? 0 : 1;
    }
    std::vector<float> res(ggml_nelements(out)); // [D, nh, nq]
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    int n_nonfinite = 0;
    for (float x : res) {
        n_nonfinite += !std::isfinite(x);
    }
    if (!out_file.empty()) {
        FILE * f = std::fopen(out_file.c_str(), "wb");
        std::fwrite(res.data(), sizeof(float), res.size(), f);
        std::fclose(f);
    }

    // F64 reference on sampled rows, and two CPU emulations (Q / K as F16; SageAttention-style INT8: Q per 16 rows, K
    // smoothed by its mean and per 64 keys).
    std::vector<int> rows;
    for (int i = 0; i < rows_per_head; ++i) {
        rows.push_back(int((size_t(i) * 7919 + 13) % nq));
    }
    rows.push_back(nq - 1);
    struct Job { int h, t; };
    std::vector<Job> jobs;
    for (int h = 0; h < nh * ns; ++h) for (int t : rows) jobs.push_back({h, t});

    // Precompute per KV head: K in F64, K as F16, K sage-quantized (smoothed, per 64 keys); Q likewise per Q head.
    std::vector<std::vector<double>> k64(nkh * ns), kf16(nkh * ns), ksage(nkh * ns), kmean(nkh * ns);
    for (int h = 0; h < nkh * ns; ++h) {
        k64[h].resize(size_t(nk) * D); kf16[h].resize(size_t(nk) * D); ksage[h].resize(size_t(nk) * D); kmean[h].assign(D, 0.0);
        for (int t = 0; t < nk; ++t) for (int d = 0; d < D; ++d) {
            const double x = k[(size_t(h) * nk + t) * D + d];
            k64[h][size_t(t) * D + d] = x;
            kf16[h][size_t(t) * D + d] = round_f16(float(x));
            kmean[h][d] += x / nk;
        }
        for (int t = 0; t < nk; ++t) for (int d = 0; d < D; ++d) ksage[h][size_t(t) * D + d] = k64[h][size_t(t) * D + d] - kmean[h][d];
        quantize_groups(ksage[h], nk, 64);
    }
    std::vector<std::vector<double>> qsage(nh * ns);
    for (int h = 0; h < nh * ns; ++h) {
        qsage[h].resize(size_t(nq) * D);
        for (size_t i = 0; i < size_t(nq) * D; ++i) qsage[h][i] = q[size_t(h) * nq * D + i] * double(scale);
        quantize_groups(qsage[h], nq, 16);
    }

    std::vector<double> rel_gpu(jobs.size()), rel_f16(jobs.size()), rel_sage(jobs.size()), abs_gpu(jobs.size());
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        std::vector<double> s(nk), o(D), qa(D);
        for (size_t j; (j = next.fetch_add(1)) < jobs.size();) {
            const int H = jobs[j].h, t = jobs[j].t, b = H / nh, h = H % nh, kh = b * nkh + h / (nh / nkh);
            double slope_h = 1.0;
            if (max_bias > 0.0f) {
                const int n_head_log2 = 1 << int(std::floor(std::log2(double(nh))));
                const double m0 = std::pow(2.0, -double(max_bias) / n_head_log2);
                const double m1 = std::pow(2.0, -double(max_bias) / 2.0 / n_head_log2);
                slope_h = h < n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1);
            }
            auto attend = [&](const double * qv, const std::vector<double> & kk, std::vector<double> & outv) {
                double mx = -INFINITY;
                for (int u = 0; u < nk; ++u) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += qv[d] * kk[size_t(u) * D + d];
                    if (softcap != 0.0f) dot = softcap * std::tanh(dot / softcap);
                    if (use_mask) dot += slope_h * maskf[size_t(t) * nk + u];
                    s[u] = dot;
                    mx = std::max(mx, dot);
                }
                double sum = 0;
                for (int u = 0; u < nk; ++u) { s[u] = std::isinf(s[u]) ? 0.0 : std::exp(s[u] - mx); sum += s[u]; }
                for (int d = 0; d < D; ++d) outv[d] = 0;
                for (int u = 0; u < nk; ++u) {
                    const double p = s[u] / sum;
                    if (p == 0) continue;
                    const float * vr = &v[(size_t(kh) * nk + u) * D];
                    for (int d = 0; d < D; ++d) outv[d] += p * vr[d];
                }
            };
            std::vector<double> ref(D), alt(D);
            for (int d = 0; d < D; ++d) qa[d] = double(q[(size_t(H) * nq + t) * D + d]) * scale;
            attend(qa.data(), k64[kh], ref);
            double nr = 0, ne = 0, ma = 0;
            for (int d = 0; d < D; ++d) {
                const double g = res[((size_t(b) * nq + t) * nh + h) * D + d];
                nr += ref[d] * ref[d];
                ne += (g - ref[d]) * (g - ref[d]);
                ma = std::max(ma, std::fabs(g - ref[d]));
            }
            rel_gpu[j] = std::sqrt(ne / nr);
            abs_gpu[j] = ma;
            // Q and K rounded to F16 (what an F16 tensor-core kernel sees):
            for (int d = 0; d < D; ++d) qa[d] = double(round_f16(q[(size_t(H) * nq + t) * D + d])) * scale;
            attend(qa.data(), kf16[kh], alt);
            ne = 0;
            for (int d = 0; d < D; ++d) ne += (alt[d] - ref[d]) * (alt[d] - ref[d]);
            rel_f16[j] = std::sqrt(ne / nr);
            // SageAttention-style INT8 Q (per 16 queries) and smoothed K (per 64 keys):
            attend(&qsage[H][size_t(t) * D], ksage[kh], alt);
            ne = 0;
            for (int d = 0; d < D; ++d) ne += (alt[d] - ref[d]) * (alt[d] - ref[d]);
            rel_sage[j] = std::sqrt(ne / nr);
        }
    };
    std::vector<std::thread> threads;
    const int nthreads = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 0; i < nthreads; ++i) threads.emplace_back(worker);
    for (auto & th : threads) th.join();

    const Stats g = summarize(rel_gpu, *std::max_element(abs_gpu.begin(), abs_gpu.end()));
    const Stats f = summarize(rel_f16, 0.0);
    const Stats sg = summarize(rel_sage, 0.0);
    std::printf("nseq=%d alibi=%.2f softcap=%.1f ", ns, max_bias, softcap);
    std::printf("nq=%d nk=%d nh=%d nkh=%d mask=%d%s f32kv=%d bias=%.2f seed=%u rows=%zu  GGML_CUDA_FATTN_DP4A=%s\n",
        nq, nk, nh, nkh, int(use_mask), causal ? " (causal)" : "", int(f32kv), bias, seed, jobs.size(), dp4a_on ? "(on)" : "off");
    std::printf("  GPU kernel       : rel L2 per row  max %.3e  p99 %.3e  median %.3e   max abs %.3e   non-finite %d\n",
        g.max_rel, g.p99_rel, g.median_rel, g.max_abs, n_nonfinite);
    std::printf("  CPU Q,K as F16   : rel L2 per row  max %.3e  p99 %.3e  median %.3e\n", f.max_rel, f.p99_rel, f.median_rel);
    std::printf("  CPU sage-style   : rel L2 per row  max %.3e  p99 %.3e  median %.3e\n", sg.max_rel, sg.p99_rel, sg.median_rel);
    ok = ok && n_nonfinite == 0;
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
