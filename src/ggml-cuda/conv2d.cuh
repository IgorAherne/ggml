#pragma once
#include "common.cuh"

#define CUDA_CONV2D_BLOCK_SIZE 256
void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Spellcast3D: the tensor-core convolution (conv2d-mma.cuh) with neighbouring graph nodes folded in (ggml-cuda.cu's
// fusion): the input read from x (the CONV_2D's src[1], or what a folded SCALE / nearest 2x UPSCALE of it reads) as
// fma(in_scale, x, in_sbias) when in_affine; the result fma(out_scale, conv, out_sbias) when out_affine, + bias[oc]
// (an ADD of a [1, 1, OC, 1] tensor), + z (an ADD of a same-shape tensor), written to out.
struct ggml_cuda_conv2d_fusion {
    const ggml_tensor * x         = nullptr;
    int                 up        = 1;
    bool                in_affine = false;
    float               in_scale  = 1.0f;
    float               in_sbias  = 0.0f;
    bool                out_affine = false;
    float               out_scale  = 1.0f;
    float               out_sbias  = 0.0f;
    const ggml_tensor * bias       = nullptr;
    const ggml_tensor * z          = nullptr;
    ggml_tensor *       out        = nullptr;
    const half *        xh         = nullptr;  // the F16 channels-last input made by the producer (x then unused)
};

bool ggml_cuda_conv2d_mma_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * conv);
void ggml_cuda_conv2d_mma_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * conv,
                                const ggml_cuda_conv2d_fusion & f);
