#include "w8a8-mm.cuh"
#include "w8a8.cuh"
#if W8A8_BM == 128 && W8A8_WM == 64   // the dp4a / Turing kernels' token mapping (research builds with other tiles: neither)
#define W8A8_HAS_DP4A 1
#include "w8a8-dp4a.cuh"
#define W8A8_HAS_TURING 1
#include "w8a8-turing.cuh"
#else
#define W8A8_HAS_DP4A 0
#define W8A8_HAS_TURING 0
#endif

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Turing (cc 7.5: RTX 20xx, GTX 16xx) takes the path with the sm_75 variant of the tensor-core GEMM (w8a8-turing.cuh)
// by default; GGML_CUDA_W8A8_TURING=0 keeps ggml's MMQ there (as GGML_CUDA_W8A8_KERNEL=off does on every card).
// Measured on the RTX 3060 running the compute_75 code (NOTES-w8a8.md, "Turing port"); not yet on a Turing card.
constexpr bool W8A8_TURING_DEFAULT = true;

struct w8a8_config {
    bool        enabled    = true;
    int64_t     min_tokens = 1024;
    std::string names      = "transformer_blocks";
    bool        glu_f16    = true;
    bool        norm_fuse  = true;
    bool        norm_check = false;
    bool        turing     = W8A8_TURING_DEFAULT;
};

const w8a8_config & w8a8_cfg() {
    static const w8a8_config cfg = [] {
        w8a8_config c;
        if (const char * e = getenv("GGML_CUDA_W8A8_KERNEL")) {
            c.enabled = strcmp(e, "off") != 0 && strcmp(e, "0") != 0;
        }
        if (const char * e = getenv("GGML_CUDA_W8A8")) {  // the research emulation takes the products instead
            if (strcmp(e, "off") != 0) {
                c.enabled = false;
            }
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_MIN_TOKENS")) {
            c.min_tokens = atoll(e);
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_NAMES")) {
            c.names = strcmp(e, "*") == 0 ? "" : e;
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_GLU")) {
            c.glu_f16 = strcmp(e, "f32") != 0;
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_NORM")) {
            c.norm_fuse = strcmp(e, "off") != 0 && strcmp(e, "0") != 0;
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_CHECK_NORM")) {
            c.norm_check = atoi(e) != 0;
        }
        if (const char * e = getenv("GGML_CUDA_W8A8_TURING")) {
            c.turing = strcmp(e, "off") != 0 && strcmp(e, "0") != 0;
        }
        return c;
    }();
    return cfg;
}

// the dp4a kernel's cards: Pascal with dp4a (cc 6.1 / 6.2; the GTX 10xx); Volta keeps MMQ (no int8 tensor cores; not
// measured with the dp4a kernel), Turing has its own tensor-core kernel (below)
bool w8a8_dp4a_device(const int cc) {
    return W8A8_HAS_DP4A && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_DP4A && cc < GGML_CUDA_CC_VOLTA &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A;
}

// the Turing kernel's cards: cc 7.5 with sm_75 / compute_75 code in the build (another arch's code below 8.0 would be
// the kernel's empty stub). Like ggml's MMQ, which takes its int8 MMA path on every cc 7.5 card, this includes the
// GTX 16xx, whose MMA instructions run without the RTX cards' tensor cores (ggml only logs a warning for them).
bool w8a8_turing_device(const int cc) {
    return W8A8_HAS_TURING && w8a8_cfg().turing && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_TURING &&
        ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING;
}

// the device side: an sm_80+ card whose kernels were compiled with the path (the release build also carries
// sm_61 .. sm_75 code, where the sm_80 tensor-core kernels are empty), a dp4a card (w8a8-dp4a.cuh) or a Turing card
// (w8a8-turing.cuh)
bool w8a8_device_ok(const int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (w8a8_dp4a_device(cc)) {
        return true;
    }
#if W8A8_HAS_TURING
    if (w8a8_turing_device(cc)) {
        return ggml_cuda_info().devices[device].smpbo >= (size_t) w8a8t_smem_bytes();
    }
#endif
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
        ggml_cuda_info().devices[device].smpbo >= (size_t) w8a8_smem_bytes();
}

void w8a8_log_once(const int device) {
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;
    const w8a8_config & c = w8a8_cfg();
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    GGML_LOG_INFO("ggml_cuda: W8A8 int8 matmul path (W8 x A8-sub32, %s) on %s (cc %d): Q4_K/Q5_K products with >= %lld tokens%s%s%s, "
                  "SwiGLU intermediate %s, norm writer %s%s, sub-scales 1..%d\n",
                  w8a8_dp4a_device(ggml_cuda_info().devices[device].cc) ? "dp4a" :
                  w8a8_turing_device(ggml_cuda_info().devices[device].cc) ? "int8 tensor cores, sm_75 kernel" : "int8 tensor cores",
                  prop.name, ggml_cuda_info().devices[device].cc, (long long) c.min_tokens,
                  c.names.empty() ? "" : ", weights named *", c.names.c_str(), c.names.empty() ? "" : "*", c.glu_f16 ? "f16" : "f32",
                  c.norm_fuse ? "fused" : "off", c.norm_check ? " (checked)" : "", W8A8_SY_MAX);
}

}  // namespace

bool ggml_cuda_w8a8_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * src1) {
    const w8a8_config & c = w8a8_cfg();
    if (!c.enabled || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    if (src1 == nullptr) {
        src1 = dst->src[1];
    }
    if ((src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1], T = src1->ne[1];
    if (T < c.min_tokens || T > INT_MAX || N > INT_MAX || !w8a8_quantize_act_supported(K) || N % W8A8_BN != 0 || src1->ne[0] != K ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[0] != N || dst->ne[1] != T) {
        return false;
    }
    // weights: plain k-quant rows in a weight buffer, 8-byte aligned; activations: f32 rows, 16-byte aligned; dst rows
    // contiguous and 16-byte aligned
    if (src0->buffer == nullptr || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
        src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] != ggml_row_size(src0->type, K) || (uintptr_t) src0->data % 8 != 0 ||
        src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || (uintptr_t) src1->data % 16 != 0 ||
        dst->nb[0] != sizeof(float) || dst->nb[1] % 16 != 0 || (uintptr_t) dst->data % 16 != 0) {
        return false;
    }
    if (!c.names.empty() && strstr(src0->name, c.names.c_str()) == nullptr) {
        return false;
    }
    return w8a8_device_ok(ctx.device);
}

bool ggml_cuda_w8a8_norm_fusion_enabled() {
    return w8a8_cfg().norm_fuse;
}

bool ggml_cuda_w8a8_norm_check_enabled() {
    return w8a8_cfg().norm_check;
}

void ggml_cuda_w8a8_act_alloc(ggml_backend_cuda_context & ctx, ggml_cuda_w8a8_act & act, const int64_t T, const int64_t K) {
    const int64_t Tp = w8a8_pad_tokens(T);
    act.qx.alloc(Tp*K);
    act.sy.alloc(Tp*(K/32));
    act.dt.alloc(Tp);
    act.T = T;
    act.K = K;
    if (Tp > T) {  // the last token tile's padding rows: q_x and d_t 0, its sy block 0 (the writer then fills rows < T)
        cudaStream_t stream = ctx.stream();
        CUDA_CHECK(cudaMemsetAsync(act.qx.get() + T*K, 0, (Tp - T)*K, stream));
        CUDA_CHECK(cudaMemsetAsync(act.sy.get() + (Tp - W8A8_BM)*(K/32), 0, W8A8_BM*(K/32), stream));
        CUDA_CHECK(cudaMemsetAsync(act.dt.get() + T, 0, (Tp - T)*sizeof(float), stream));
    }
}

static __global__ void w8a8_count_mismatches(const uint8_t * a, const uint8_t * b, const int64_t n, unsigned long long * count) {
    unsigned long long c = 0;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        c += a[i] != b[i];
    }
    if (c) {
        atomicAdd(count, c);
    }
}

int64_t ggml_cuda_w8a8_act_mismatches(ggml_backend_cuda_context & ctx, ggml_cuda_w8a8_act & a, ggml_cuda_w8a8_act & b) {
    GGML_ASSERT(a.T == b.T && a.K == b.K);
    const int64_t Tp = w8a8_pad_tokens(a.T);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<unsigned long long> count(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(count.get(), 0, sizeof(unsigned long long), stream));
    w8a8_count_mismatches<<<256, 256, 0, stream>>>((const uint8_t *) a.qx.get(), (const uint8_t *) b.qx.get(), Tp*a.K, count.get());
    w8a8_count_mismatches<<<256, 256, 0, stream>>>(a.sy.get(), b.sy.get(), Tp*(a.K/32), count.get());
    w8a8_count_mismatches<<<64, 256, 0, stream>>>((const uint8_t *) a.dt.get(), (const uint8_t *) b.dt.get(), Tp*4, count.get());
    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h, count.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return (int64_t) h;
}

void ggml_cuda_w8a8_quantize(ggml_backend_cuda_context & ctx, ggml_cuda_w8a8_act & act, const void * x, const bool x_f16,
                             const int64_t s01, const int64_t T, const int64_t K, const bool scale_pre, const float scale,
                             const float bias) {
    GGML_ASSERT(!x_f16 || !scale_pre);
    const int64_t Tp = w8a8_pad_tokens(T);
    act.qx.alloc(Tp*K);
    act.sy.alloc(Tp*(K/32));
    act.dt.alloc(Tp);
    act.T = T;
    act.K = K;
    cudaStream_t stream = ctx.stream();
    bool ok;
    if (x_f16) {
        ok = w8a8_quantize_act_cuda<__half>((const __half *) x, s01, W8A8_PRE_NONE, 1.0f, 0.0f, act.qx.get(), act.sy.get(),
                                            act.dt.get(), (int) T, (int) K, stream);
    } else {
        ok = w8a8_quantize_act_cuda<float>((const float *) x, s01, scale_pre ? W8A8_PRE_SCALE : W8A8_PRE_NONE, scale, bias,
                                           act.qx.get(), act.sy.get(), act.dt.get(), (int) T, (int) K, stream);
    }
    GGML_ASSERT(ok);
    CUDA_CHECK(cudaGetLastError());
}

// converts src0 (rows interleaved for glu) into pool memory and runs the GEMM with epilogue epi into dst
static void w8a8_product(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_cuda_w8a8_act & act, const int epi,
                         void * dst, const int64_t ldd) {
    const int64_t K = src0->ne[0], N = src0->ne[1];
    GGML_ASSERT(act.K == K);
    w8a8_log_once(ctx.device);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> qw(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float>  dw(ctx.pool(), N);
    const bool glu = epi == W8A8_EPI_GLU_F16 || epi == W8A8_EPI_GLU_F32;
    w8a8_convert_weights_cuda(src0->data, src0->type == GGML_TYPE_Q5_K, glu, (int64_t) src0->nb[1], qw.get(), dw.get(), (int) N, (int) K,
                              stream);
    CUDA_CHECK(cudaGetLastError());
#if W8A8_HAS_DP4A
    if (w8a8_dp4a_device(ggml_cuda_info().devices[ctx.device].cc)) {
        CUDA_CHECK(w8a8d_gemm_cuda(epi, act.qx.ptr, act.sy.ptr, act.dt.ptr, qw.get(), dw.get(), dst, ldd, (int) act.T, (int) N, (int) K, stream));
    } else
#endif
#if W8A8_HAS_TURING
    if (w8a8_turing_device(ggml_cuda_info().devices[ctx.device].cc)) {
        CUDA_CHECK(w8a8t_gemm_cuda(epi, act.qx.ptr, act.sy.ptr, act.dt.ptr, qw.get(), dw.get(), dst, ldd, (int) act.T, (int) N, (int) K, stream));
    } else
#endif
    {
        CUDA_CHECK(w8a8_gemm_cuda(epi, act.qx.ptr, act.sy.ptr, act.dt.ptr, qw.get(), dw.get(), dst, ldd, (int) act.T, (int) N, (int) K, stream));
    }
}

void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_cuda_w8a8_act & act, ggml_tensor * dst) {
    GGML_ASSERT(dst->ne[0] == src0->ne[1] && dst->ne[1] == act.T && dst->type == GGML_TYPE_F32);
    w8a8_product(ctx, src0, act, W8A8_EPI_F32, dst->data, dst->nb[1] / sizeof(float));
}

void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_w8a8_act act(ctx.pool());
    ggml_cuda_w8a8_quantize(ctx, act, src1->data, false, src1->nb[1] / sizeof(float), src1->ne[1], src1->ne[0]);
    ggml_cuda_w8a8_mul_mat(ctx, src0, act, dst);
}

bool ggml_cuda_w8a8_glu_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * glu,
                                  const ggml_tensor * mm_out) {
    if (!ggml_cuda_w8a8_supported(ctx, mm) || !ggml_cuda_w8a8_supported(ctx, mm_out, glu)) {
        return false;
    }
    const int64_t N = mm->src[0]->ne[1];
    // glu: [N/2, T] contiguous (the epilogue writes N/2 values per row, 16-byte stores); mm_out reads it as its src1
    return N % 256 == 0 && glu->ne[0] == N/2 && glu->ne[1] == mm->ne[1] && ggml_is_contiguous(glu) && mm_out->src[1] == glu &&
        mm_out->src[0]->ne[0] == N/2;
}

void ggml_cuda_w8a8_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_cuda_w8a8_act & act, ggml_tensor * glu,
                        ggml_tensor * mm_out) {
    const bool    f16 = w8a8_cfg().glu_f16;
    const int64_t F   = glu->ne[0];
    // [gate | up] product -> silu(gate) * up, written into glu's buffer as f16 / f32 rows of F values
    w8a8_product(ctx, mm->src[0], act, f16 ? W8A8_EPI_GLU_F16 : W8A8_EPI_GLU_F32, glu->data, F);
    // the down product of it
    ggml_cuda_w8a8_act act2(ctx.pool());
    ggml_cuda_w8a8_quantize(ctx, act2, glu->data, f16, F, glu->ne[1], F);
    ggml_cuda_w8a8_mul_mat(ctx, mm_out->src[0], act2, mm_out);
}
