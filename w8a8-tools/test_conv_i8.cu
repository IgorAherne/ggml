// The GTX 10xx int8 (dp4a) convolution of src/ggml-cuda/conv2d.cu (conv2d_i8_prepare: the maxima / scales / weight /
// input passes; conv2d_i8_conv: conv2d_implicit_gemm_i8), included with GGML_CUDA_CONV2D_KERNELS_ONLY and run as
// ggml_cuda_op_conv2d runs it.
//   1. exactness: the kernel's int32 sums (its RAW epilogue) against an int64 reference over the very quantized operands
//      it read (every output); the quantized operands against a host recomputation from the F32 input / F16 weights
//      and the scales the passes computed (every word);
//   2. the quantization error: the int8 outputs and the F32 path's (conv2d_simt_f32 on the F16-rounded input) against
//      an F64 reference of the convolution on sampled outputs (relative RMS and max error over the outputs' RMS);
//   3. speed: the int8 kernel alone, the whole int8 path, the F32 path (F16 pad + weight transpose + kernel); TOPS =
//      2 OC OW OH IC KW KH B / t.
//
// build (from w8a8-tools, MSVC + nvcc 12.8):   build_tool.cmd test_conv_i8      (sm_86 SASS)
//        NVCC_ARCH="-gencode=arch=compute_61,code=compute_61" build_tool.cmd test_conv_i8 test_conv_i8_c61
//        (the compute_61 PTX, JIT-compiled for the card: the code the FAKE_CC=610 emulation runs)
// usage: test_conv_i8                          the Wan VAE decoder's int8 shapes + odd ones
//        test_conv_i8 IW IH OC IC K [B]        one shape (K 1 or 3, padding (K-1)/2)
// env:   CONV_ALPHA=a[,b..] (default 0,0.5,1)  CONV_REPS=n  CONV_NOTIME=1  CONV_SPREAD=s (channel scales e^U(-s,s),
//        default 1)
#define GGML_CUDA_CONV2D_KERNELS_ONLY
#include "conv2d.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static __device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static __device__ __forceinline__ float rnd_normal(uint32_t i, uint32_t seed) {
    const uint32_t h0 = hash32(i * 3u + seed), h1 = hash32(h0 ^ 0x9e3779b9U), h2 = hash32(h1 + 0x85ebca6bU);
    const float    u  = (h0 >> 8) * (1.0f / 16777216.0f) + (h1 >> 8) * (1.0f / 16777216.0f) + (h2 >> 8) * (1.0f / 16777216.0f);
    return (u - 1.5f) * 2.0f;
}
// input [B][IC][IH][IW]: channel c times cs[c]; a SiLU-like skew on odd channels (mostly positive values)
static __global__ void init_x(float * x, int64_t total, int64_t hw, int ic, const float * cs, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = int(i / hw % ic);
    float v = rnd_normal(uint32_t(i), seed);
    if (c % 2) v = v / (1.0f + expf(-v));
    x[i] = v * cs[c];
}
static __global__ void init_w(half * w, int64_t total, float scale, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    w[i] = __float2half(rnd_normal(uint32_t(i), seed) * scale);
}
// the int64 reference of the kernel's sums from the operands it read (xq, wt in the kernel's layouts); mism[0] = count,
// mism[1] = the first differing output, amax[0] = max |sum|
static __global__ void ref_sums(const int * xq, const int * wt, const float * raw, const conv_params PP, int k4_pad, int bm,
                                int ic4, int khw, int kw, unsigned long long * mism, unsigned long long * amax) {
    const int64_t npos = PP.OW * PP.OH, total = PP.B * PP.OC * npos;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int p = int(i % npos), o = int(i / npos % PP.OC), n = int(i / npos / PP.OC);
    const int y = p / int(PP.OW), x = p % int(PP.OW);
    const int mt = o / bm, ol = o % bm;
    long long s = 0;
    for (int k4 = 0; k4 < ic4 * khw; ++k4) {
        const int g = k4 / khw, kk = k4 % khw, ky = kk / kw, kx = kk % kw;
        const int a = xq[((int64_t(n) * ic4 + g) * PP.IH + y + ky) * PP.IW + x + kx];
        const int b = wt[(int64_t(mt) * k4_pad + k4) * bm + ol];
        for (int j = 0; j < 4; ++j) s += (long long) (int8_t) (a >> (8 * j)) * (long long) (int8_t) (b >> (8 * j));
    }
    atomicMax(amax, (unsigned long long) (s < 0 ? -s : s));
    if ((long long) __float_as_int(raw[i]) != s) {
        atomicAdd(mism, 1ull);
        atomicMin(mism + 1, (unsigned long long) i);
    }
}

template <class F> static float time_median(F && f, int n) {
    f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> v;
    for (int i = 0; i < n; ++i) {
        CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(ms);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct shape { int iw, ih, oc, ic, k, b; const char * note; };
static int g_nsm = 0;
static std::vector<float> g_alphas = { 0.0f, 0.5f, 1.0f };
static float g_spread = 1.0f;
static bool g_time = true;
static int g_reps = 0;

template <int TM, int MODE> static void print_attrs() {
    cudaFuncAttributes fa;
    auto f = conv2d_implicit_gemm_i8<TM, MODE, false>;
    CK(cudaFuncGetAttributes(&fa, f));
    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256, 0));
    printf("  i8<tm %d, %s>: %3d registers, %5zu B shared, %3zu B local, %d blocks/SM\n", TM,
           MODE == CONV2D_SIMT_SEG3 ? "seg3" : "rows", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, nb);
}

static bool run_shape(const shape & s) {
    const int pad = (s.k - 1) / 2;
    const int64_t IW = s.iw, IH = s.ih, K = s.k, IC = s.ic, OC = s.oc, B = s.b;
    const int64_t OW = IW, OH = IH, npos = OW * OH, total = B * OC * npos;
    const int pw = int(IW + 2 * pad), ph = int(IH + 2 * pad);
    const conv_params P  = { IW, IH, OW, OH, K, K, 1, 1, pad, pad, 1, 1, IC, OC, B, total };
    const conv_params PP = { pw, ph, OW, OH, K, K, 1, 1, 0, 0, 1, 1, IC, OC, B, total };
    // ggml_cuda_op_conv2d's split_k policy
    const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
    const int     target  = 8 * g_nsm;
    const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * K * K + 63) / 64,
                                           std::max(int64_t(1), (target + blocks - 1) / blocks) }));
    const conv2d_i8_plan plan = conv2d_i8_make_plan(PP, split_k, IW * IH);
    printf("%4lldx%-4lld OC %4lld IC %4lld k%lld B%lld split %d: ", (long long) OW, (long long) OH, (long long) OC,
           (long long) IC, (long long) K, (long long) B, split_k);
    if (plan.mode < 0) {
        printf("not an int8 shape (F32 path)  %s\n", s.note);
        return true;
    }
    printf("%s tm %d k4 %d/%d  %s\n", plan.mode == CONV2D_SIMT_SEG3 ? "seg3" : "rows", plan.tm, plan.k4_total, plan.k4_pad, s.note);

    const int64_t nx = B * IC * IH * IW, nw = OC * IC * K * K;
    float * x; half * w; float * y_raw; float * y_i8; float * y_f; float * cs; float * f32; int * words;
    CK(cudaMalloc(&x, nx * 4)); CK(cudaMalloc(&w, nw * 2)); CK(cudaMalloc(&cs, IC * 4));
    CK(cudaMalloc(&y_raw, total * 4)); CK(cudaMalloc(&y_i8, total * 4)); CK(cudaMalloc(&y_f, total * 4));
    CK(cudaMalloc(&f32, plan.f32_elems * 4)); CK(cudaMalloc(&words, (plan.wt_words + plan.in_words) * 4));
    // channel scales e^U(-spread, spread), one outlier channel x8
    std::vector<float> hcs(IC);
    std::mt19937 rng(uint32_t(s.iw * 7 + s.oc * 13 + s.ic));
    std::uniform_real_distribution<float> U(-g_spread, g_spread);
    for (auto & v : hcs) v = expf(U(rng));
    hcs[IC / 3] *= 8.0f;
    CK(cudaMemcpy(cs, hcs.data(), IC * 4, cudaMemcpyHostToDevice));
    const uint32_t seed = uint32_t(s.iw * 131 + s.oc * 7 + s.ic * 3 + s.k);
    init_x<<<unsigned((nx + 255) / 256), 256>>>(x, nx, IH * IW, int(IC), cs, seed);
    init_w<<<unsigned((nw + 255) / 256), 256>>>(w, nw, 1.0f / sqrtf(float(IC * K * K)), seed + 17);
    CK(cudaGetLastError());
    std::vector<float> hx(nx); std::vector<half> hw(nw);
    CK(cudaMemcpy(hx.data(), x, nx * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hw.data(), w, nw * 2, cudaMemcpyDeviceToHost));

    // the F32 path (as ggml_cuda_op_conv2d runs it for this shape)
    half * xh; CK(cudaMalloc(&xh, int64_t(pw) * ph * IC * B * 2));
    const conv2d_simt_plan fplan = conv2d_simt_f32_plan(PP, split_k, xh, y_f);
    half * wt16; CK(cudaMalloc(&wt16, std::max(fplan.wt_elems, int64_t(1)) * 2));
    auto f32_path = [&]() {
        const int padded_total = int(int64_t(pw) * ph * IC * B);
        conv2d_pad_f16<<<(padded_total + 255) / 256, 256>>>(x, xh, int(IW), int(IH), pw, ph, pad, pad, padded_total);
        conv2d_simt_f32(xh, w, wt16, y_f, PP, split_k, fplan, 0);
    };
    f32_path();
    CK(cudaDeviceSynchronize());

    // sampled F64 reference
    const int nsample = 4096;
    std::vector<int64_t> idx(nsample);
    std::uniform_int_distribution<int64_t> UI(0, total - 1);
    for (auto & v : idx) v = UI(rng);
    std::vector<double> ref(nsample);
    double ref_ss = 0;
    for (int t = 0; t < nsample; ++t) {
        const int64_t i = idx[t];
        const int p = int(i % npos), o = int(i / npos % OC), n = int(i / npos / OC), y0 = p / int(OW), x0 = p % int(OW);
        double acc = 0;
        for (int c = 0; c < IC; ++c)
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx) {
                    const int yy = y0 + ky - pad, xx = x0 + kx - pad;
                    if (yy < 0 || yy >= IH || xx < 0 || xx >= IW) continue;
                    acc += double(hx[((int64_t(n) * IC + c) * IH + yy) * IW + xx]) *
                           double(__half2float(hw[((int64_t(o) * IC + c) * K + ky) * K + kx]));
                }
        ref[t] = acc; ref_ss += acc * acc;
    }
    const double ref_rms = sqrt(ref_ss / nsample);
    auto err = [&](const float * dev, double & rms, double & mx) {
        std::vector<float> h(total);
        CK(cudaMemcpy(h.data(), dev, total * 4, cudaMemcpyDeviceToHost));
        double ss = 0; mx = 0;
        for (int t = 0; t < nsample; ++t) { const double d = h[idx[t]] - ref[t]; ss += d * d; mx = std::max(mx, fabs(d)); }
        rms = sqrt(ss / nsample) / ref_rms; mx /= ref_rms;
    };
    double f_rms, f_max;
    err(y_f, f_rms, f_max);
    printf("    F32 path: rel. error rms %.2e max %.2e (F16-rounded input)\n", f_rms, f_max);

    bool ok = true;
    unsigned long long * dm; CK(cudaMalloc(&dm, 4 * sizeof(unsigned long long)));
    for (const float alpha : g_alphas) {
        conv2d_i8_prepare(x, w, P, PP, plan, f32, words, alpha, 0);
        conv2d_i8_conv<true>(y_raw, P, PP, plan, f32, words, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        // 1a. the sums against the int64 reference
        unsigned long long hinit[4] = { 0, ~0ull, 0, 0 }, hm[4];
        CK(cudaMemcpy(dm, hinit, sizeof(hinit), cudaMemcpyHostToDevice));
        ref_sums<<<unsigned((total + 255) / 256), 256>>>(words + plan.wt_words, words, y_raw, PP, plan.k4_pad,
                                                         16 * plan.tm, int(IC / 4), int(K * K), int(K), dm, dm + 2);
        CK(cudaGetLastError());
        CK(cudaMemcpy(hm, dm, sizeof(hm), cudaMemcpyDeviceToHost));
        // 1b. the operands against the host's quantization with the passes' scales
        std::vector<float> hf(plan.f32_elems);
        CK(cudaMemcpy(hf.data(), f32, plan.f32_elems * 4, cudaMemcpyDeviceToHost));
        const float * hs = hf.data() + IC * B * plan.nchunk, * hinv_s = hs + IC, * hdw = hinv_s + IC, * hinv_dw = hdw + OC;
        std::vector<int> hwords(plan.wt_words + plan.in_words);
        CK(cudaMemcpy(hwords.data(), words, hwords.size() * 4, cudaMemcpyDeviceToHost));
        auto q8 = [](float v) { return std::max(-127, std::min(127, int(std::nearbyint(v)))); };
        long long bad_x = 0, bad_w = 0;
        const int bm = 16 * plan.tm, ic4 = int(IC / 4), khw = int(K * K);
        for (int64_t i = 0; i < plan.in_words; ++i) {
            const int xx = int(i % pw) - pad, r = int(i / pw), yy = r % ph - pad, ng = r / ph, g = ng % ic4, n = ng / ic4;
            int word = 0;
            if (xx >= 0 && xx < IW && yy >= 0 && yy < IH)
                for (int j = 0; j < 4; ++j)
                    word |= (q8(hx[((int64_t(n) * IC + 4 * g + j) * IH + yy) * IW + xx] * hinv_s[4 * g + j]) & 0xFF) << (8 * j);
            bad_x += word != hwords[plan.wt_words + i];
        }
        const int mtiles = int((OC + bm - 1) / bm);
        for (int mt = 0; mt < mtiles; ++mt)
            for (int k4 = 0; k4 < plan.k4_pad; ++k4)
                for (int c = 0; c < bm; ++c) {
                    const int o = mt * bm + c, g = k4 / khw, kk = k4 % khw;
                    int word = 0;
                    if (o < OC && k4 < plan.k4_total)
                        for (int j = 0; j < 4; ++j)
                            word |= (q8(__half2float(hw[(int64_t(o) * IC + 4 * g + j) * khw + kk]) * hs[4 * g + j] * hinv_dw[o]) & 0xFF) << (8 * j);
                    bad_w += word != hwords[(int64_t(mt) * plan.k4_pad + k4) * bm + c];
                }
        // the scales against the host's (amax exact; s, 1/s, d within rounding)
        double s_rel = 0;
        {
            std::vector<float> am(IC, 0.0f);
            for (int64_t i = 0; i < nx; ++i) { const int c = int(i / (IH * IW) % IC); am[c] = std::max(am[c], fabsf(hx[i])); }
            double S = 0;
            for (int c = 0; c < IC; ++c) if (am[c] > 0) S = std::max(S, double(am[c]) / pow(double(am[c]), double(alpha)));
            S /= 127;
            for (int c = 0; c < IC; ++c) {
                const double sc = am[c] > 0 ? pow(double(am[c]), double(alpha)) * S : 0;
                if (sc > 0) s_rel = std::max(s_rel, fabs(hs[c] / sc - 1));
            }
        }
        const bool pass = hm[0] == 0 && bad_x == 0 && bad_w == 0;
        ok = ok && pass;
        // 2. the error of the int8 outputs
        conv2d_i8_run(x, w, y_i8, P, PP, plan, f32, words, alpha, 0);
        CK(cudaDeviceSynchronize());
        double q_rms, q_max;
        err(y_i8, q_rms, q_max);
        printf("    alpha %.2f: sums %llu/%lld differ (max |sum| %.3g = 2^%.1f), x words %lld, w words %lld differ, "
               "scales rel %.1e%s | rel. error rms %.2e max %.2e\n", alpha, hm[0], (long long) total, double(hm[2]),
               log2(double(hm[2]) + 1), bad_x, bad_w, s_rel, pass ? "" : "  <- FAIL", q_rms, q_max);
    }
    CK(cudaFree(dm));

    // 3. speed
    if (g_time) {
        const double work = 2.0 * OC * OW * OH * IC * K * K * B;
        const int reps = g_reps > 0 ? g_reps : (work > 2e11 ? 7 : (work > 2e10 ? 15 : 41));
        const float alpha = 0.5f;
        conv2d_i8_prepare(x, w, P, PP, plan, f32, words, alpha, 0);
        const float t_k  = time_median([&]() { conv2d_i8_conv(y_i8, P, PP, plan, f32, words, 0); }, reps);
        const float t_i8 = time_median([&]() { conv2d_i8_run(x, w, y_i8, P, PP, plan, f32, words, alpha, 0); }, reps);
        const float t_f  = time_median(f32_path, reps);
        printf("    time: int8 kernel %8.3f ms %6.2f T/s | int8 path %8.3f ms %6.2f T/s | F32 path %8.3f ms %5.2f T/s | x%.2f\n",
               t_k, work / t_k * 1e-9, t_i8, work / t_i8 * 1e-9, t_f, work / t_f * 1e-9, t_f / t_i8);
    }
    fflush(stdout);
    CK(cudaFree(x)); CK(cudaFree(w)); CK(cudaFree(cs)); CK(cudaFree(y_raw)); CK(cudaFree(y_i8)); CK(cudaFree(y_f));
    CK(cudaFree(f32)); CK(cudaFree(words)); CK(cudaFree(xh)); CK(cudaFree(wt16));
    return ok;
}

int main(int argc, char ** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_nsm = prop.multiProcessorCount;
    if (const char * v = getenv("CONV_ALPHA")) {
        g_alphas.clear();
        std::string str(v);
        for (size_t p = 0; p < str.size();) {
            const size_t q = str.find(',', p);
            g_alphas.push_back(float(atof(str.substr(p, q - p).c_str())));
            p = q == std::string::npos ? str.size() : q + 1;
        }
    }
    if (const char * v = getenv("CONV_SPREAD")) g_spread = float(atof(v));
    if (const char * v = getenv("CONV_NOTIME")) g_time = atoi(v) == 0;
    if (const char * v = getenv("CONV_REPS")) g_reps = atoi(v);
    printf("%s (cc %d.%d), %d SMs, %d MHz; channel spread e^+-%.2f\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000, g_spread);
    print_attrs<9, CONV2D_SIMT_SEG3>();
    print_attrs<9, CONV2D_SIMT_ROWS>();
    print_attrs<8, CONV2D_SIMT_SEG3>();
    print_attrs<8, CONV2D_SIMT_ROWS>();
    std::vector<shape> shapes;
    if (argc >= 6) {
        shapes.push_back({ atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 1, "" });
    } else {
        shapes = {
            // the Wan VAE decoder of a 1024x1024 picture (calls)
            { 1024, 1024,  144,  144, 3, 1, "x5" },
            { 1024, 1024,  144,  288, 3, 1, "x1" },
            { 1024, 1024,  144,  288, 1, 1, "x1" },
            { 1024, 1024,  288,  288, 3, 1, "x1" },
            { 1024, 1024,    4,  144, 3, 1, "x1 (final conv: F32 narrow)" },
            {  512,  512,  288,  288, 3, 1, "x5" },
            {  512,  512,  288,  576, 3, 1, "x1" },
            {  512,  512,  288,  576, 1, 1, "x1" },
            {  512,  512,  576,  576, 3, 1, "x1" },
            {  256,  256,  576,  576, 3, 1, "x5" },
            {  256,  256,  576, 1152, 3, 1, "x1" },
            {  256,  256,  576, 1152, 1, 1, "x1" },
            {  256,  256, 1152, 1152, 3, 1, "x1" },
            {  128,  128, 1152, 1152, 3, 1, "x7" },
            {   64,   64, 1152, 1152, 3, 1, "x12" },
            {   64,   64, 1152, 1152, 1, 1, "x1" },
            {   64,   64, 1152,   64, 3, 1, "x1" },
            {   64,   64, 3456, 1152, 1, 1, "x1 (to_qkv)" },
            {   64,   64,   64,   64, 1, 1, "x1" },
            // odd ones
            {  192,   64,  200,   12, 3, 1, "OC 200 (tile tail), K 108 (k4 27)" },
            {   64,    2,  128,    4, 3, 2, "batch 2, IC 4, tm 8, 2-row map" },
            {  128,   64,  256,   20, 1, 2, "1x1 K 20 (k4 5 -> 8), batch 2, tm 8" },
            {  128,  128,   16,   64, 3, 1, "OC 16" },
            {   64,   64,   12,  132, 1, 1, "OC 12, 1x1 k4 33 -> 40" },
            {   37,   53,  144,  144, 3, 1, "odd map: F32" },
            {  100,   64,  144,  144, 3, 1, "width 100: F32" },
            {   64,   64,  144,  142, 3, 1, "IC 142: F32" },
        };
    }
    bool ok = true;
    for (const shape & s : shapes) ok = run_shape(s) && ok;
    printf(ok ? "all int8 shapes exact\n" : "FAIL\n");
    return ok ? 0 : 1;
}
