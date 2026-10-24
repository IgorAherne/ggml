#pragma once

#include "common.cuh"

// FlashAttention for cards without tensor cores (Spellcast3D: the GTX 10xx, cc 6.1) with the Q.K^T scores as INT8
// dot products (__dp4a, four times the FP32 rate there): Q and K quantized per row to INT8 with one F32 scale each (K
// smoothed by its mean over the keys, as SageAttention does), the online softmax in F32 and, by default, the P.V
// product as dp4a too (numerators u8 per 32-key block, V int8 per channel and 32-key block, blocks with outlier V
// keys in F32; GGML_CUDA_FATTN_DP4A_PV=f32 keeps the F32 P.V of the generic tile kernel). Head size 128 only.

// Whether the op dst (whose best kernel is the generic tile kernel) runs through this kernel: an NVIDIA card with
// dp4a and without tensor cores (cc 6.1 - 6.x), head size 128, Q with at least 1024 columns, no attention sinks and no
// logit softcap, F16 / F32 K and V. GGML_CUDA_FATTN_DP4A=off in the environment keeps the F32 tile kernel.
bool ggml_cuda_flash_attn_ext_tile_dp4a_use(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_tile_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
