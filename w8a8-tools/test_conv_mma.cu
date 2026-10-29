// Exactness and speed of the tensor-core implicit-GEMM convolution (src/ggml-cuda/conv2d-mma.cuh: conv2d_mma_f16, the
// cp.async / m16n8k16 kernel of cc >= 8.0 and the register-staged / m16n8k8 Turing kernel) against the kernel it
// replaces on Turing and newer (conv2d_implicit_gemm_f16<.., true> over the padded F16 copy, the engine's split_k
// policy and reduce kernel) and an F64 reference. Kernels from conv2d.cu itself (GGML_CUDA_CONV2D_KERNELS_ONLY),
// launched as ggml_cuda_op_conv2d launches them.
//
// Operands: F16 values from a hash (the input generated in F32 as exactly representable halves, so every path reads the
// same F16 operands), weights scaled 1/sqrt(K). Checks: new vs old over every output (max |d|, max |d| / max |old|,
// relative RMS); the Turing kernel vs the new one; every kernel vs an F64 sum over sampled outputs (error / sum |w x|).
// Times: medians and minima of cudaEvent-timed launches; "path" = the F16 copy (+ the weight reorder) + the kernel.
//
// build: NVCC_EXTRA="-I..\include -I..\src" build_tool.cmd test_conv_mma          (sm_86)
// usage: test_conv_mma                  the Qwen-Image 2.1 VAE decoder's shapes + odd ones (exactness + timing)
//        test_conv_mma IW IH OC IC K [S [D [B [P]]]]   one shape (padding P, default (K-1)/2 * D)
// env:   CONV_REPS=n   timed launches (default 10); CONV_SAMPLES=n F64 samples (default 3000); CONV_SET=decoder|odd|all
//        CONV_NOOLD=1  skip the old kernel (timing); CONV_TURING=0 skip the Turing kernel
#define GGML_CUDA_CONV2D_KERNELS_ONLY
#include "conv2d.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// what the launch helpers' CUDA_SET_SHARED_MEMORY_LIMIT needs from the backend
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  %s in %s at %s:%d\n", msg, stmt, func, file, line);
    exit(1);
}
int ggml_cuda_get_device() {
    int d = 0;
    cudaGetDevice(&d);
    return d;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static __host__ __device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// ~N(0, 1) * scale from three uniforms (a float; rounded to F16 by the caller)
static __host__ __device__ __forceinline__ float rnd(uint32_t i, uint32_t seed, float scale) {
    const uint32_t h0 = hash32(i * 3u + seed), h1 = hash32(h0 ^ 0x9e3779b9U), h2 = hash32(h1 + 0x85ebca6bU);
    const float    u  = (h0 >> 8) * (1.0f / 16777216.0f) + (h1 >> 8) * (1.0f / 16777216.0f) + (h2 >> 8) * (1.0f / 16777216.0f);
    return (u - 1.5f) * 2.0f * scale;
}
static float h_round_half(float v) { return __half2float(__float2half(v)); }

static __global__ void init_input(float * x, int64_t n, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__float2half(rnd(uint32_t(i), seed, 1.0f)));
}
static __global__ void init_weight(half * w, int64_t n, float scale, uint32_t seed) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) w[i] = __float2half(rnd(uint32_t(i), seed, scale));
}
static __global__ void fill_f32(float * p, int64_t n, float v) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
// st[0] = max |a - b|, st[1] = max |b| (as float bits), st2[0] = sum (a - b)^2, st2[1] = sum b^2, cnt = non-finite
static __global__ void cmp_stats(const float * a, const float * b, int64_t n, unsigned * st, double * st2,
                                 unsigned long long * nonfinite) {
    double s0 = 0, s1 = 0;
    float  m0 = 0, m1 = 0;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const float x = a[i], y = b[i];
        if (!isfinite(x) || !isfinite(y)) { atomicAdd(nonfinite, 1ull); continue; }
        const float d = fabsf(x - y);
        m0 = fmaxf(m0, d); m1 = fmaxf(m1, fabsf(y));
        s0 += double(d) * d; s1 += double(y) * y;
    }
    for (int o = 16; o > 0; o >>= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, o));
        s0 += __shfl_xor_sync(0xffffffff, s0, o); s1 += __shfl_xor_sync(0xffffffff, s1, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(st + 0, __float_as_uint(m0)); atomicMax(st + 1, __float_as_uint(m1));
        atomicAdd(st2 + 0, s0); atomicAdd(st2 + 1, s1);
    }
}

template <class F>
static void time_it(F && f, int n, float & med, float & mn) {
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
    med = v[v.size() / 2]; mn = v[0];
}

struct shape { int iw, ih, oc, ic, k, s, d, b, p; const char * note; };

struct cmp_result { float maxd, maxref; double rrms; unsigned long long nonfinite; };
static cmp_result compare(const float * a, const float * b, int64_t n) {
    unsigned * st; double * st2; unsigned long long * nf;
    CK(cudaMalloc(&st, 8)); CK(cudaMalloc(&st2, 16)); CK(cudaMalloc(&nf, 8));
    CK(cudaMemset(st, 0, 8)); CK(cudaMemset(st2, 0, 16)); CK(cudaMemset(nf, 0, 8));
    cmp_stats<<<1024, 256>>>(a, b, n, st, st2, nf);
    unsigned h[2]; double h2[2]; unsigned long long hn;
    CK(cudaMemcpy(h, st, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2, st2, 16, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hn, nf, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(st)); CK(cudaFree(st2)); CK(cudaFree(nf));
    cmp_result r; memcpy(&r.maxd, &h[0], 4); memcpy(&r.maxref, &h[1], 4);
    r.rrms = h2[1] > 0 ? std::sqrt(h2[0] / h2[1]) : 0; r.nonfinite = hn;
    return r;
}

static int g_reps = 10, g_samples = 3000;
static bool g_old = true, g_turing = true;
static double g_sum_old = 0, g_sum_new = 0, g_sum_path = 0, g_sum_kernel = 0, g_sum_flop = 0;

static bool run_shape(const shape & s) {
    const int64_t IW = s.iw, IH = s.ih, IC = s.ic, OC = s.oc, KW = s.k, KH = s.k, B = s.b;
    const int     ST = s.s, DL = s.d, PD = s.p >= 0 ? s.p : (s.k - 1) / 2 * s.d;
    const int64_t OW = (IW + 2 * PD - DL * (KW - 1) - 1) / ST + 1, OH = (IH + 2 * PD - DL * (KH - 1) - 1) / ST + 1;
    const int64_t K = IC * KW * KH, nin = B * IC * IH * IW, nw = OC * K, nout = B * OC * OH * OW;
    const double  flop = 2.0 * double(nout) * double(K);
    if (!conv2d_mma_shape_ok(IW, IH, OW, OH, KW, KH, IC, OC, B, ST, ST, PD, PD, DL, DL)) {
        printf("%-34s not taken by the MMA kernel\n", s.note);
        return true;
    }
    float * x; half * w; float * y_old; float * y_new; float * y_tur;
    CK(cudaMalloc(&x, nin * 4)); CK(cudaMalloc(&w, nw * 2));
    CK(cudaMalloc(&y_old, nout * 4)); CK(cudaMalloc(&y_new, nout * 4)); CK(cudaMalloc(&y_tur, nout * 4));
    const uint32_t sx = 1234u + uint32_t(IC * 7 + OC), sw = 777u + uint32_t(K);
    const float    wscale = 1.0f / std::sqrt(float(K));
    init_input<<<unsigned((nin + 255) / 256), 256>>>(x, nin, sx);
    init_weight<<<unsigned((nw + 255) / 256), 256>>>(w, nw, wscale, sw);
    fill_f32<<<unsigned((nout + 255) / 256), 256>>>(y_old, nout, NAN);
    fill_f32<<<unsigned((nout + 255) / 256), 256>>>(y_new, nout, NAN);
    fill_f32<<<unsigned((nout + 255) / 256), 256>>>(y_tur, nout, NAN);
    CK(cudaGetLastError());

    // the new path
    half * xh; half * wt;
    CK(cudaMalloc(&xh, nin * 2)); CK(cudaMalloc(&wt, nw * 2));
    conv2d_mma_params p = {};
    p.iw = int(IW), p.ih = int(IH), p.ic = int(IC), p.ow = int(OW), p.oh = int(OH), p.oc = int(OC);
    p.kw = int(KW), p.kh = int(KH), p.sx = ST, p.sy = ST, p.px = PD, p.py = PD, p.dx = DL, p.dy = DL;
    p.k = int(K), p.ohw = int(OW * OH), p.npos = int(B * OW * OH);
    const bool reorder = KW * KH > 1;
    auto prep = [&]() {
        conv2d_mma_to_nhwc(x, xh, int(IC), int(IH), int(IW), int(B), 1, false, 1.0f, 0.0f, 0);
        if (reorder) conv2d_mma_reorder_weights(w, wt, int(OC), int(IC), int(KW * KH), 0);
    };
    const half * wk = reorder ? wt : w;
    auto run_new = [&]() { conv2d_mma_launch(xh, wk, y_new, p, true, 0); };
    auto run_tur = [&]() { conv2d_mma_launch(xh, wk, y_tur, p, false, 0); };
    prep(); run_new(); if (g_turing) run_tur();
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // the old path (ggml_cuda_op_conv2d's MMA branch)
    const int64_t pw = IW + 2 * PD, ph = IH + 2 * PD, padded_total = pw * ph * IC * B;
    const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
    int dev = 0, nsm = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    const int target  = 8 * nsm;
    const int split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                       std::max(int64_t(1), (target + blocks - 1) / blocks) }));
    const conv_params pp = { pw, ph, OW, OH, KW, KH, ST, ST, 0, 0, DL, DL, IC, OC, B, nout };
    half * xpad = nullptr; float * partial = nullptr;
    const bool do_old = g_old || true;  // exactness always; timing only with g_old
    if (do_old) {
        CK(cudaMalloc(&xpad, padded_total * 2));
        if (split_k > 1) CK(cudaMalloc(&partial, nout * split_k * 4));
    }
    auto run_old = [&]() {
        conv2d_pad_f16<<<unsigned((padded_total + 255) / 256), 256>>>(x, xpad, int(IW), int(IH), int(pw), int(ph), PD, PD,
                                                                        int(padded_total));
        float *    res = split_k == 1 ? y_old : partial;
        const dim3 block(32, 4);
        const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        conv2d_launch_implicit_gemm<true>(xpad, w, res, pp, split_k, grid, block, 0);
        if (split_k > 1) {
            conv2d_reduce_split_k<<<unsigned((nout + 255) / 256), 256>>>(res, y_old, int(nout), int(OC * OW * OH), split_k);
        }
    };
    if (do_old) { run_old(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }

    // exactness
    const cmp_result r_no = do_old ? compare(y_new, y_old, nout) : cmp_result{};
    const cmp_result r_tn = g_turing ? compare(y_tur, y_new, nout) : cmp_result{};
    // F64 reference over sampled outputs
    std::vector<float> h_new(nout > 0 ? 1 : 0);
    double e_new = 0, e_old = 0, e_tur = 0, e_new_rel = 0;
    {
        const int ns = int(std::min<int64_t>(g_samples, nout));
        std::vector<int64_t> idx(ns);
        for (int i = 0; i < ns; ++i) idx[i] = int64_t(hash32(uint32_t(i) * 2654435761u + 17u)) * 7919 % nout;
        std::vector<float> vn(ns), vo(ns), vt(ns);
        for (int i = 0; i < ns; ++i) {
            CK(cudaMemcpy(&vn[i], y_new + idx[i], 4, cudaMemcpyDeviceToHost));
            if (do_old) CK(cudaMemcpy(&vo[i], y_old + idx[i], 4, cudaMemcpyDeviceToHost));
            if (g_turing) CK(cudaMemcpy(&vt[i], y_tur + idx[i], 4, cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < ns; ++i) {
            const int64_t o = idx[i];
            const int64_t ox = o % OW, oy = o / OW % OH, oc = o / (OW * OH) % OC, n = o / (OW * OH * OC);
            double sum = 0, sabs = 0;
            for (int64_t ci = 0; ci < IC; ++ci) {
                for (int64_t ky = 0; ky < KH; ++ky) {
                    const int64_t iy = oy * ST + ky * DL - PD;
                    if (iy < 0 || iy >= IH) continue;
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const int64_t ix = ox * ST + kx * DL - PD;
                        if (ix < 0 || ix >= IW) continue;
                        const double xv = h_round_half(rnd(uint32_t(((n * IC + ci) * IH + iy) * IW + ix), sx, 1.0f));
                        const double wv = h_round_half(rnd(uint32_t((oc * IC + ci) * KH * KW + ky * KW + kx), sw, wscale));
                        sum += xv * wv; sabs += std::fabs(xv * wv);
                    }
                }
            }
            const double den = sabs > 0 ? sabs : 1;
            e_new = std::max(e_new, std::fabs(vn[i] - sum) / den);
            e_new_rel = std::max(e_new_rel, std::fabs(vn[i] - sum) / std::max(std::fabs(sum), 1e-3 * den));
            if (do_old) e_old = std::max(e_old, std::fabs(vo[i] - sum) / den);
            if (g_turing) e_tur = std::max(e_tur, std::fabs(vt[i] - sum) / den);
        }
    }

    // timing
    float t_old = 0, t_old_min = 0, t_path = 0, t_path_min = 0, t_k = 0, t_k_min = 0, t_tur = 0, t_tur_min = 0, t_prep = 0,
          t_prep_min = 0;
    if (g_old) time_it(run_old, g_reps, t_old, t_old_min);
    time_it([&]() { prep(); run_new(); }, g_reps, t_path, t_path_min);
    time_it(run_new, g_reps, t_k, t_k_min);
    time_it(prep, g_reps, t_prep, t_prep_min);
    if (g_turing) time_it(run_tur, g_reps, t_tur, t_tur_min);

    printf("%-34s K %5lld sk %2d | old %8.3f ms (%5.2f T) | path %8.3f (prep %6.3f) kernel %8.3f ms (%5.2f T, min %5.2f T)"
           " | tur %8.3f (%5.2f T) | new-old max %.2e rel %.2e rrms %.2e nf %llu | tur-new max %.2e rrms %.2e | f64 new %.2e"
           " (rel %.1e) old %.2e tur %.2e\n",
           s.note, (long long) K, split_k, t_old, g_old ? flop / t_old * 1e-9 : 0.0, t_path, t_prep, t_k, flop / t_k * 1e-9,
           flop / t_k_min * 1e-9, t_tur, g_turing ? flop / t_tur * 1e-9 : 0.0, r_no.maxd,
           r_no.maxref > 0 ? r_no.maxd / r_no.maxref : 0.0, r_no.rrms, r_no.nonfinite, r_tn.maxd, r_tn.rrms, e_new,
           e_new_rel, e_old, e_tur);
    fflush(stdout);
    g_sum_old += t_old; g_sum_path += t_path; g_sum_kernel += t_k; g_sum_flop += flop;
    const bool ok = r_no.nonfinite == 0 && r_tn.nonfinite == 0 && e_new < 1e-5 && (!g_turing || e_tur < 1e-5) &&
                    (!do_old || r_no.maxd <= 1e-4 * std::max(1.0f, r_no.maxref));
    if (!ok) printf("  ^^^ FAILED\n");
    CK(cudaFree(x)); CK(cudaFree(w)); CK(cudaFree(y_old)); CK(cudaFree(y_new)); CK(cudaFree(y_tur));
    CK(cudaFree(xh)); CK(cudaFree(wt));
    if (xpad) CK(cudaFree(xpad));
    if (partial) CK(cudaFree(partial));
    return ok;
}

int main(int argc, char ** argv) {
    if (getenv("CONV_REPS")) g_reps = atoi(getenv("CONV_REPS"));
    if (getenv("CONV_SAMPLES")) g_samples = atoi(getenv("CONV_SAMPLES"));
    if (getenv("CONV_NOOLD")) g_old = atoi(getenv("CONV_NOOLD")) == 0;
    if (getenv("CONV_TURING")) g_turing = atoi(getenv("CONV_TURING")) != 0;
    std::vector<shape> shapes;
    if (argc >= 6) {
        shape s = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 1,
                    argc > 7 ? atoi(argv[7]) : 1, argc > 8 ? atoi(argv[8]) : 1, argc > 9 ? atoi(argv[9]) : -1, "custom" };
        shapes.push_back(s);
    } else {
        const std::string set = getenv("CONV_SET") ? getenv("CONV_SET") : "all";
        // the decoder (a 1024x1024 picture): map, OC <- IC, kernel; calls in the decode in the note
        const shape dec[] = {
            { 64, 64, 64, 64, 1, 1, 1, 1, -1, "64^2 64<-64 1x1 (x1)" },
            { 64, 64, 1152, 64, 3, 1, 1, 1, -1, "64^2 1152<-64 3x3 (x1)" },
            { 64, 64, 1152, 1152, 3, 1, 1, 1, -1, "64^2 1152<-1152 3x3 (x10)" },
            { 64, 64, 3456, 1152, 1, 1, 1, 1, -1, "64^2 3456<-1152 1x1 (x1)" },
            { 64, 64, 1152, 1152, 1, 1, 1, 1, -1, "64^2 1152<-1152 1x1 (x1)" },
            { 128, 128, 1152, 1152, 3, 1, 1, 1, -1, "128^2 1152<-1152 3x3 (x7)" },
            { 256, 256, 1152, 1152, 3, 1, 1, 1, -1, "256^2 1152<-1152 3x3 (x1)" },
            { 256, 256, 576, 1152, 3, 1, 1, 1, -1, "256^2 576<-1152 3x3 (x1)" },
            { 256, 256, 576, 1152, 1, 1, 1, 1, -1, "256^2 576<-1152 1x1 (x1)" },
            { 256, 256, 576, 576, 3, 1, 1, 1, -1, "256^2 576<-576 3x3 (x5)" },
            { 512, 512, 576, 576, 3, 1, 1, 1, -1, "512^2 576<-576 3x3 (x1)" },
            { 512, 512, 288, 576, 3, 1, 1, 1, -1, "512^2 288<-576 3x3 (x1)" },
            { 512, 512, 288, 576, 1, 1, 1, 1, -1, "512^2 288<-576 1x1 (x1)" },
            { 512, 512, 288, 288, 3, 1, 1, 1, -1, "512^2 288<-288 3x3 (x5)" },
            { 1024, 1024, 288, 288, 3, 1, 1, 1, -1, "1024^2 288<-288 3x3 (x1)" },
            { 1024, 1024, 144, 288, 3, 1, 1, 1, -1, "1024^2 144<-288 3x3 (x1)" },
            { 1024, 1024, 144, 288, 1, 1, 1, 1, -1, "1024^2 144<-288 1x1 (x1)" },
            { 1024, 1024, 144, 144, 3, 1, 1, 1, -1, "1024^2 144<-144 3x3 (x5)" },
            { 1024, 1024, 4, 144, 3, 1, 1, 1, -1, "1024^2 4<-144 3x3 (x1)" },
        };
        const shape odd[] = {
            { 37, 53, 100, 32, 3, 1, 1, 1, -1, "37x53 100<-32 3x3" },
            { 129, 127, 200, 48, 3, 1, 1, 1, -1, "129x127 200<-48 3x3" },
            { 64, 40, 7, 8, 3, 1, 1, 1, -1, "64x40 7<-8 3x3" },
            { 50, 50, 9, 16, 3, 1, 1, 2, -1, "50x50 9<-16 3x3 b2" },
            { 33, 31, 300, 24, 3, 1, 1, 1, -1, "33x31 300<-24 3x3" },
            { 80, 80, 128, 40, 3, 2, 1, 1, -1, "80x80 128<-40 3x3 s2" },
            { 81, 79, 64, 64, 3, 2, 1, 1, 0, "81x79 64<-64 3x3 s2 p0" },
            { 60, 60, 144, 144, 3, 1, 2, 1, -1, "60x60 144<-144 3x3 d2" },
            { 40, 40, 96, 64, 5, 1, 1, 2, -1, "40x40 96<-64 5x5 b2" },
            { 70, 30, 1, 8, 1, 1, 1, 1, -1, "70x30 1<-8 1x1" },
            { 100, 100, 130, 136, 1, 1, 1, 3, -1, "100x100 130<-136 1x1 b3" },
            { 16, 16, 512, 512, 3, 1, 1, 1, -1, "16^2 512<-512 3x3 (split_k old)" },
            { 7, 9, 64, 256, 3, 1, 1, 1, 2, "7x9 64<-256 3x3 p2" },
            { 128, 128, 256, 128, 3, 1, 1, 1, -1, "128^2 256<-128 3x3 (T128)" },
            { 256, 256, 512, 256, 3, 1, 1, 1, -1, "256^2 512<-256 3x3 (T128)" },
        };
        if (set == "decoder" || set == "all") for (const auto & s : dec) shapes.push_back(s);
        if (set == "odd" || set == "all") for (const auto & s : odd) shapes.push_back(s);
    }
    int fails = 0;
    for (const auto & s : shapes) fails += run_shape(s) ? 0 : 1;
    printf("sums over the shapes listed (one call each): old %.1f ms, path %.1f ms, kernel %.1f ms (%.2f T)\n", g_sum_old,
           g_sum_path, g_sum_kernel, g_sum_flop / g_sum_kernel * 1e-9);
    printf("%s (%d failed)\n", fails ? "FAILED" : "ALL OK", fails);
    return fails ? 1 : 0;
}
