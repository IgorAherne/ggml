#pragma once

#include "common.cuh"

// The "W8 x A8-sub32" matmul path (kernels in w8a8.cuh) for the Q4_K / Q5_K products of large token counts on sm_80+:
// int8 weights per output channel (converted from the k-quant blocks at every call, into pool memory) x int8
// activations per token with 8-bit integer sub-scales per 32 values, on the int8 tensor cores with one int32
// accumulator per output. The number formats are those of the research emulation GGML_CUDA_W8A8=sub:8 (w8a8-emu.cu),
// bit for bit.
//
// Environment:
//   GGML_CUDA_W8A8_KERNEL     off | on (default on where supported: an sm_80+ card and kernels compiled for it)
//   GGML_CUDA_W8A8_MIN_TOKENS the products with fewer tokens (src1 rows) keep MMQ (default 1024)
//   GGML_CUDA_W8A8_NAMES      only weights whose name contains this text (default "transformer_blocks": the scope the
//                             format was judged on, the diffusion transformer; "*" = every Q4_K / Q5_K weight)
//   GGML_CUDA_W8A8_GLU        f16 | f32: the SwiGLU output that the [gate | up] product's epilogue writes for the down
//                             product (default f16)
// The path is off while the research emulation (GGML_CUDA_W8A8) is on.

// the quantized activations of one src1 (pool memory; one set can feed several products of the same src1)
struct ggml_cuda_w8a8_act {
    ggml_cuda_pool_alloc<int8_t>  qx;
    ggml_cuda_pool_alloc<uint8_t> sy;
    ggml_cuda_pool_alloc<float>   dt;
    int64_t T = 0;
    int64_t K = 0;
    explicit ggml_cuda_w8a8_act(ggml_cuda_pool & pool) : qx(pool), sy(pool), dt(pool) {}
};

// Whether the path computes dst (a MUL_MAT) when its src1 is `src1` (default dst->src[1]; a fusion passes the
// tensor whose values the product reads instead, e.g. a SCALE's input).
bool ggml_cuda_w8a8_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const ggml_tensor * src1 = nullptr);

// Quantizes T rows of K values (f32, or f16 when x_f16) with row stride s01 (elements); scale_pre: x*scale + bias first
// (GGML_OP_SCALE fused, scale_f32's FMA).
void ggml_cuda_w8a8_quantize(ggml_backend_cuda_context & ctx, ggml_cuda_w8a8_act & act, const void * x, bool x_f16, int64_t s01,
                             int64_t T, int64_t K, bool scale_pre = false, float scale = 1.0f, float bias = 0.0f);

// dst = src0 x act (dst F32 [N, T])
void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_cuda_w8a8_act & act, ggml_tensor * dst);

// dst = src0 x src1 (quantizes src1, then the product)
void ggml_cuda_w8a8_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Whether mm ([gate | up] MUL_MAT) -> glu (SWIGLU of mm, not swapped) -> mm_out (MUL_MAT of glu) can run as one chain:
// both products on this path and the shapes the GLU epilogue handles.
bool ggml_cuda_w8a8_glu_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * glu,
                                  const ggml_tensor * mm_out);

// The chain: the [gate | up] weight converted with its gate / up rows interleaved, the product's epilogue writes
// silu(gate) * up (ggml's swiglu) into glu's own buffer (f16 or f32, GGML_CUDA_W8A8_GLU), which is quantized for the
// down product. mm's F32 output is never written; glu->data holds the f16 / f32 values only for this chain.
void ggml_cuda_w8a8_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_cuda_w8a8_act & act, ggml_tensor * glu,
                        ggml_tensor * mm_out);
