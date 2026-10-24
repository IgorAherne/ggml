// Byte identity and speed of the Pascal F32 implicit-GEMM convolution (conv2d_implicit_gemm_simt_f32 in
// src/ggml-cuda/conv2d.cu, the GTX 10xx path) against the SIMT branch of conv2d_implicit_gemm_f16 that it replaces.
// Both kernels come from conv2d.cu itself (included with GGML_CUDA_CONV2D_KERNELS_ONLY) and are launched as
// ggml_cuda_op_conv2d launches them: the padded F16 input, F16 weights, the same split_k policy, the reduce kernel.
// The raw kernel outputs (the per-split partial sums when split_k > 1) and the reduced outputs are compared byte for
// byte over the whole tensor; times are medians of cudaEvent-timed launches, TFLOPS = 2 OC OW OH IC KW KH B / t.
//
// build (the sdlinux container):
//   nvcc -O3 -std=c++17 -arch=sm_61 -use_fast_math -I<ggml>/src/ggml-cuda -I<ggml>/src -I<ggml>/include \
//        -o test_conv test_conv_simt.cu
// usage: test_conv                      the Wan VAE decoder's shapes + tail shapes (identity + timing)
//        test_conv IW IH OC IC K [S [D [B]]]   one shape (input size, kernel K x K, padding (K-1)/2 * D, stride S,
//                                           dilation D, batch B)
// env:   CONV_MODE=auto|gather|rows  force the new kernel's input staging (rows: 1x1 shapes only); CONV_REPS=n reps
//        CONV_NOREF_TIME=1  skip timing the reference (identity only for it)
#define GGML_CUDA_CONV2D_KERNELS_ONLY
#include "conv2d.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static __device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// ~N(0, 1) from three uniforms, times scale, rounded to F16.
static __device__ __forceinline__ half rnd_half(uint32_t i, uint32_t seed, float scale) {
    const uint32_t h0 = hash32(i * 3u + seed), h1 = hash32(h0 ^ 0x9e3779b9U), h2 = hash32(h1 + 0x85ebca6bU);
    const float    u  = (h0 >> 8) * (1.0f / 16777216.0f) + (h1 >> 8) * (1.0f / 16777216.0f) + (h2 >> 8) * (1.0f / 16777216.0f);
    return __float2half((u - 1.5f) * 2.0f * scale);
}
// The padded input as conv2d_pad_f16 leaves it: zeros on the border, data inside.
static __global__ void init_input(half * x, int64_t total, int pw, int ph, int px, int py, int iw, int ih, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int xx = int(i % pw) - px, yy = int(i / pw % ph) - py;
    x[i] = (unsigned) xx < (unsigned) iw && (unsigned) yy < (unsigned) ih ? rnd_half(uint32_t(i), seed, 1.0f) : __float2half(0.0f);
}
static __global__ void init_weight(half * w, int64_t total, float scale, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    w[i] = rnd_half(uint32_t(i), seed, scale);
}
static __global__ void fill_u32(uint32_t * p, int64_t n, uint32_t v) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
// Counts differing 32-bit words; first[0] = the lowest differing index.
static __global__ void cmp_words(const uint32_t * a, const uint32_t * b, int64_t n, unsigned long long * cnt,
                                 unsigned long long * first) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) {
        atomicAdd(cnt, 1ull);
        atomicMin(first, (unsigned long long) i);
    }
}

template <class F>
static float time_median(F && f, int n, int warm = 1) {
    for (int i = 0; i < warm; ++i) f();
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

struct shape { int iw, ih, oc, ic, k, s, d, b; const char * note; };

static int g_nsm = 0;
static std::string g_mode = "auto";

static const char * mode_name(int m) {
    return m == CONV2D_SIMT_SEG3 ? "seg3" : m == CONV2D_SIMT_ROWS ? "rows" : m == CONV2D_SIMT_NARROW ? "narrow" :
           m == CONV2D_SIMT_64 ? "64x64" : "gather";
}
template <int KW, int KH, int TM, int MODE> static void print_attrs() {
    cudaFuncAttributes fa;
    auto               f = conv2d_implicit_gemm_simt_f32<KW, KH, TM, MODE>;
    CK(cudaFuncGetAttributes(&fa, f));
    int nb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 256, 0));
    printf("  <%d,%d,tm %d,%-6s>: %3d registers, %5zu B shared, %3zu B local, %d blocks/SM\n", KW, KH, TM,
           mode_name(MODE), fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, nb);
}
static void print_kernel_attrs() {
    print_attrs<3, 3, 9, CONV2D_SIMT_SEG3>();
    print_attrs<1, 1, 9, CONV2D_SIMT_ROWS>();
    print_attrs<3, 3, 9, CONV2D_SIMT_GATHER>();
    print_attrs<1, 1, 9, CONV2D_SIMT_GATHER>();
    print_attrs<0, 0, 9, CONV2D_SIMT_GATHER>();
    print_attrs<3, 3, 8, CONV2D_SIMT_SEG3>();
    print_attrs<1, 1, 8, CONV2D_SIMT_ROWS>();
    print_attrs<3, 3, 8, CONV2D_SIMT_GATHER>();
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, conv2d_simt_narrow3x3<4, 16, true>));
    printf("  narrow3x3<4,16>: %3d registers, %5zu B static shared (+ 16 B a k for the weights), %3zu B local\n",
           fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
    CK(cudaFuncGetAttributes(&fa, conv2d_simt_weights<9>));
    printf("  weights<9>: %3d registers, %5zu B shared, %3zu B local\n", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
}

// returns the number of differing words (raw + reduced)
static long long run_shape(const shape & s, int reps, bool time_ref) {
    const int pad = (s.k - 1) / 2 * s.d;
    const int64_t IW = s.iw, IH = s.ih, KW = s.k, KH = s.k, IC = s.ic, OC = s.oc, B = s.b;
    const int64_t OW = (IW + 2 * pad - s.d * (KW - 1) - 1) / s.s + 1, OH = (IH + 2 * pad - s.d * (KH - 1) - 1) / s.s + 1;
    const int64_t total = B * OC * OH * OW;
    const int pw = int(IW + 2 * pad), ph = int(IH + 2 * pad);
    const int64_t padded_total = int64_t(pw) * ph * IC * B;
    const conv_params P = { pw, ph, OW, OH, KW, KH, s.s, s.s, 0, 0, s.d, s.d, IC, OC, B, total };

    // ggml_cuda_op_conv2d's split_k policy, verbatim
    const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
    const int     target  = 8 * g_nsm;
    const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                           std::max(int64_t(1), (target + blocks - 1) / blocks) }));
    const int64_t nout = total * split_k;

    half * x; half * w; float * o_ref; float * o_new; float * r_ref = nullptr; float * r_new = nullptr;
    CK(cudaMalloc(&x, padded_total * sizeof(half)));
    CK(cudaMalloc(&w, OC * IC * KW * KH * sizeof(half)));
    CK(cudaMalloc(&o_ref, nout * sizeof(float)));
    CK(cudaMalloc(&o_new, nout * sizeof(float)));
    if (split_k > 1) { CK(cudaMalloc(&r_ref, total * sizeof(float))); CK(cudaMalloc(&r_new, total * sizeof(float))); }
    const uint32_t seed = uint32_t(s.iw * 131 + s.oc * 7 + s.ic * 3 + s.k);
    init_input<<<unsigned((padded_total + 255) / 256), 256>>>(x, padded_total, pw, ph, pad, pad, int(IW), int(IH), seed);
    init_weight<<<unsigned((OC * IC * KW * KH + 255) / 256), 256>>>(w, OC * IC * KW * KH, 1.0f / sqrtf(float(IC * KW * KH)), seed + 17);
    // different garbage in the two outputs: every word must be written by both kernels
    fill_u32<<<unsigned((nout + 255) / 256), 256>>>((uint32_t *) o_ref, nout, 0x7fc00001u);
    fill_u32<<<unsigned((nout + 255) / 256), 256>>>((uint32_t *) o_new, nout, 0xffc00002u);
    CK(cudaGetLastError());

    const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
    const dim3 block(32, 4);
    conv2d_simt_plan plan = conv2d_simt_f32_plan(P, split_k, x, o_new);
    if (g_mode == "gather" || g_mode == "rows") {  // force a mode (rows only where its conditions hold)
        plan.mode = g_mode == "gather" ? CONV2D_SIMT_GATHER : CONV2D_SIMT_ROWS;
        plan.k_pad = int((IC * KW * KH + 7) / 8 * 8);
        plan.wt_elems = (OC + 16 * plan.tm - 1) / (16 * plan.tm) * plan.k_pad * 16 * plan.tm;
    }
    half * wt; CK(cudaMalloc(&wt, std::max(plan.wt_elems, int64_t(1)) * sizeof(half)));
    auto ref = [&]() { conv2d_launch_implicit_gemm<false>(x, w, o_ref, P, split_k, grid, block, 0); };
    auto nw  = [&]() { conv2d_simt_f32(x, w, wt, o_new, P, split_k, plan, 0); };
    auto wpass = [&]() {
        if (plan.mode == CONV2D_SIMT_NARROW || plan.mode == CONV2D_SIMT_64) return;
        const dim3 gw(unsigned((plan.k_pad + 31) / 32), unsigned((OC + 16 * plan.tm - 1) / (16 * plan.tm)));
        if (plan.tm == 8) conv2d_simt_weights<8><<<gw, 256>>>(w, wt, int(OC), int(IC * KW * KH), plan.k_pad);
        else              conv2d_simt_weights<9><<<gw, 256>>>(w, wt, int(OC), int(IC * KW * KH), plan.k_pad);
    };
    ref(); nw();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    if (split_k > 1) {
        conv2d_reduce_split_k<<<unsigned((total + 255) / 256), 256>>>(o_ref, r_ref, int(total), int(OC * OW * OH), split_k);
        conv2d_reduce_split_k<<<unsigned((total + 255) / 256), 256>>>(o_new, r_new, int(total), int(OC * OW * OH), split_k);
    }
    unsigned long long * dc; CK(cudaMalloc(&dc, 4 * sizeof(unsigned long long)));
    unsigned long long hinit[4] = { 0, ~0ull, 0, ~0ull };
    CK(cudaMemcpy(dc, hinit, sizeof(hinit), cudaMemcpyHostToDevice));
    cmp_words<<<unsigned((nout + 255) / 256), 256>>>((const uint32_t *) o_ref, (const uint32_t *) o_new, nout, dc, dc + 1);
    if (split_k > 1) {
        cmp_words<<<unsigned((total + 255) / 256), 256>>>((const uint32_t *) r_ref, (const uint32_t *) r_new, total, dc + 2, dc + 3);
    }
    unsigned long long hc[4];
    CK(cudaMemcpy(hc, dc, sizeof(hc), cudaMemcpyDeviceToHost));
    // a sample of values, to show the data are not trivial
    float sample[2] = {};
    CK(cudaMemcpy(sample, o_ref + nout / 3, sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sample + 1, o_new + nout / 3, sizeof(float), cudaMemcpyDeviceToHost));

    const double flops = 2.0 * OC * OW * OH * IC * KW * KH * B;
    const float t_ref = time_ref ? time_median(ref, reps) : 0.0f;
    const float t_new = time_median(nw, reps);
    const float t_w   = time_median(wpass, reps);
    printf("%4lldx%-4lld OC %4lld IC %4lld k%lld s%d d%d B%lld split %2d | raw %lld/%lld differ%s",
           (long long) OW, (long long) OH, (long long) OC, (long long) IC, (long long) KW, s.s, s.d, (long long) B, split_k,
           (long long) hc[0], (long long) nout, hc[0] ? " <- FAIL" : "");
    if (hc[0]) {
        float a, b2; CK(cudaMemcpy(&a, o_ref + hc[1], 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&b2, o_new + hc[1], 4, cudaMemcpyDeviceToHost));
        printf(" (first at %llu: %.9g vs %.9g)", hc[1], a, b2);
    }
    if (split_k > 1) printf(", reduced %lld/%lld differ%s", (long long) hc[2], (long long) total, hc[2] ? " <- FAIL" : "");
    printf(" | sample %.5g", sample[0]);
    if (time_ref) printf(" | old %8.3f ms %5.2f TF", t_ref, flops / t_ref * 1e-9);
    printf(" | new %-6s bm%d %8.3f ms (w %.3f) %5.2f TF", mode_name(plan.mode), plan.tm * 16, t_new, t_w, flops / t_new * 1e-9);
    if (time_ref) printf(" | x%.2f", t_ref / t_new);
    printf("  %s\n", s.note);
    fflush(stdout);
    CK(cudaFree(x)); CK(cudaFree(w)); CK(cudaFree(o_ref)); CK(cudaFree(o_new)); CK(cudaFree(dc)); CK(cudaFree(wt));
    if (r_ref) { CK(cudaFree(r_ref)); CK(cudaFree(r_new)); }
    return (long long) (hc[0] + hc[2]);
}

int main(int argc, char ** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_nsm = prop.multiProcessorCount;
    if (const char * v = getenv("CONV_MODE")) g_mode = v;
    const int  reps_env = getenv("CONV_REPS") ? atoi(getenv("CONV_REPS")) : 0;
    const bool time_ref = !(getenv("CONV_NOREF_TIME") && atoi(getenv("CONV_NOREF_TIME")));
    printf("%s (cc %d.%d), %d SMs, %d MHz; mode %s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000, g_mode.c_str());
    print_kernel_attrs();
    std::vector<shape> shapes;
    if (argc >= 6) {
        shapes.push_back({ atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                           argc > 6 ? atoi(argv[6]) : 1, argc > 7 ? atoi(argv[7]) : 1, argc > 8 ? atoi(argv[8]) : 1, "" });
    } else {
        shapes = {
            // the Wan VAE decoder of a 1024x1024 picture (calls in the note)
            { 1024, 1024,  144,  144, 3, 1, 1, 1, "x5" },
            { 1024, 1024,  144,  288, 3, 1, 1, 1, "x1" },
            { 1024, 1024,  144,  288, 1, 1, 1, 1, "x1" },
            { 1024, 1024,  288,  288, 3, 1, 1, 1, "x1" },
            { 1024, 1024,    4,  144, 3, 1, 1, 1, "x1 (final conv)" },
            {  512,  512,  288,  288, 3, 1, 1, 1, "x5" },
            {  512,  512,  288,  576, 3, 1, 1, 1, "x1" },
            {  512,  512,  288,  576, 1, 1, 1, 1, "x1" },
            {  512,  512,  576,  576, 3, 1, 1, 1, "x1" },
            {  256,  256,  576,  576, 3, 1, 1, 1, "x5" },
            {  256,  256,  576, 1152, 3, 1, 1, 1, "x1" },
            {  256,  256,  576, 1152, 1, 1, 1, 1, "x1" },
            {  256,  256, 1152, 1152, 3, 1, 1, 1, "x1" },
            {  128,  128, 1152, 1152, 3, 1, 1, 1, "x7" },
            {   64,   64, 1152, 1152, 3, 1, 1, 1, "x12" },
            {   64,   64, 1152, 1152, 1, 1, 1, 1, "x1" },
            {   64,   64, 1152,   64, 3, 1, 1, 1, "x1" },
            {   64,   64, 3456, 1152, 1, 1, 1, 1, "x1 (to_qkv)" },
            {   64,   64,   64,   64, 1, 1, 1, 1, "x1" },
            // tails
            {   37,   53,  144,  144, 3, 1, 1, 1, "odd map (N tail)" },
            {   37,   53,    4,   13, 3, 1, 1, 1, "M 4, K 117 (K tail)" },
            {   53,   37,  100,   13, 1, 1, 1, 1, "1x1, K 13, M 100" },
            {   53,   37,  200,  150, 1, 1, 1, 1, "1x1, K 150" },
            {   16,   16,  144,  576, 3, 1, 1, 1, "split_k" },
            {   23,   19,   70,   37, 3, 1, 1, 2, "batch 2, split_k, K 333" },
            {   80,   60,  144,   32, 3, 2, 1, 1, "stride 2" },
            {   61,   47,   96,   24, 3, 1, 2, 1, "dilation 2" },
            {   41,   29,   40,    7, 5, 1, 1, 2, "5x5, K 175, batch 2" },
            {  129,  127,  288,  288, 3, 1, 1, 1, "N tail, big" },
            {  128,   64,    4,   13, 3, 1, 1, 1, "narrow: M 4, K 117 (K tail)" },
            {  128,   64,    8,   20, 3, 1, 1, 1, "narrow: M 8, K 180" },
            {   96,   96,    3,   64, 3, 1, 1, 1, "narrow: M 3, K 576 (no K tail)" },
            {   64,   64,    4,   16, 3, 1, 1, 2, "narrow: batch 2" },
            {  200,   48,    5,   40, 3, 1, 1, 1, "narrow: M 5, width 200" },
        };
    }
    long long bad = 0;
    for (const shape & s : shapes) {
        const double work = 2.0 * s.oc * s.iw * s.ih * s.ic * s.k * s.k * s.b;
        const int reps = reps_env > 0 ? reps_env : (work > 2e11 ? 5 : (work > 2e10 ? 10 : 30));
        bad += run_shape(s, reps, time_ref);
    }
    printf(bad ? "FAIL: %lld words differ\n" : "all shapes bit-identical (%lld words differ)\n", bad);
    return bad != 0;
}
