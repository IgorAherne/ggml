#include "common.cuh"

#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// a 2D IM2COL (dst) of a SCALE's output, reading the SCALE's input x and scaling it itself; false: not applicable
bool ggml_cuda_op_im2col_scaled(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * x, float scale, float bias);
void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
