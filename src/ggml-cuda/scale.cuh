#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a SCALE followed by an ADD of its output and y (repeatable to it), and optionally another ADD of that and z (same
// shape), in one kernel writing dst (the last ADD's output); y_first / z_first: the other operand is the ADD's src0
void ggml_cuda_op_scale_add_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node, const ggml_tensor * y,
                                  bool y_first, const ggml_tensor * z, bool z_first, ggml_tensor * dst);
