#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// layout: the mmq_q8_1_ds_layout to write (-1: the per-32 layout of type_src0; MMQ_Q8_1_DS_LAYOUT_DSY: no ids)
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream, int layout = -1);

// An elementwise op whose output is a quantized MUL_MAT's src1, applied while quantizing src1 instead of writing the
// output (ggml_cuda_try_fuse): the op's own arithmetic on its input, so the q8_1 data are the same.
enum ggml_cuda_quantize_pre_op_type {
    GGML_CUDA_QUANTIZE_PRE_NONE   = 0,
    GGML_CUDA_QUANTIZE_PRE_SCALE  = 1,  // GGML_OP_SCALE: scale * x + bias, one FMA like scale_f32
    GGML_CUDA_QUANTIZE_PRE_SWIGLU = 2,  // GGML_GLU_OP_SWIGLU: silu(x) * x[up_offset], like unary_gated_op_kernel
};

struct ggml_cuda_quantize_pre_op {
    int           type = GGML_CUDA_QUANTIZE_PRE_NONE;
    const float * x    = nullptr;       // the op's input at the first value of src1 (SWIGLU: the first gate value)
    int64_t       s01 = 0, s02 = 0, s03 = 0;  // its row / channel / sample strides, in floats
    float         scale = 1.0f, bias = 0.0f;
    int64_t       up_offset = 0;        // SWIGLU: from a gate value to its up value, in floats
};

// quantize_mmq_q8_1_cuda (no ids) of the output of `pre` (ne00 values per row, as the MUL_MAT's src1 has them)
void quantize_mmq_q8_1_pre_op_cuda(
        const ggml_cuda_quantize_pre_op & pre, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream, int layout = -1);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
