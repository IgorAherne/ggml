#pragma once

#include "common.cuh"

// Research emulation of the W8A8 route (see w8a8-emu.cu); returns true when it computed dst. Inactive unless the
// environment variable GGML_CUDA_W8A8 is set.
bool ggml_cuda_w8a8_emu(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
