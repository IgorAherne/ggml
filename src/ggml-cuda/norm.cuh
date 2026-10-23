#include "common.cuh"
#include "w8a8-format.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// NORM (dst) followed by mul_tensor = dst * (broadcast F32 operand) in one kernel, written to mul_tensor; with q8_1,
// written instead as the q8_1 data (mmq_q8_1_ds_layout q8_1_layout) quantize_mmq_q8_1 would make of mul_tensor
// (requires ne0 % 128 == 0 and ne2 == ne3 == 1)
void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                             void * q8_1 = nullptr, int q8_1_layout = -1);

// add = x + mul1 (mul1 = a * b) and mul2 = norm(add) * m in one kernel; writes add and mul2 (or, with q8_1, mul2's
// q8_1 data as above; or, with w8a8, mul2 as the W8A8 path's quantized activations - exactly what
// ggml_cuda_w8a8_quantize makes of mul2 - into buffers for w8a8_pad_tokens(rows) rows, rows >= nrows not written;
// requires ne0 % 1024 == 0, ne0 <= 16384 and ne2 == ne3 == 1)
void ggml_cuda_op_mul_add_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * mul1, ggml_tensor * add,
                                         ggml_tensor * norm, ggml_tensor * mul2, void * q8_1 = nullptr, int q8_1_layout = -1,
                                         const w8a8_act_ptrs * w8a8 = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// cont(permute) -> RMS_NORM -> MUL by a [C] gamma -> cont(permute back) [-> SILU] over the channels (dim 3) of a
// contiguous F32 x [W, H, T, C], written to dst (same shape) in one kernel with the separate kernels' exact arithmetic;
// dst may be x itself but must not overlap it otherwise; scale (with silu): a following SCALE node, applied as well
void ggml_cuda_op_rms_norm_cf_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
                                    float eps, ggml_tensor * dst, bool silu, const ggml_tensor * scale = nullptr);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
