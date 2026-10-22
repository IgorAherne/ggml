#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_quantize_pre_op * src1_pre_op, char * src1_q8_1_given, bool src1_q8_1_filled) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!src1_pre_op || !ids);
    GGML_ASSERT(!src1_q8_1_given || !ids);
    GGML_ASSERT(src1_q8_1_given || !src1_q8_1_filled);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(!src1_pre_op || !use_native_fp4);
    GGML_ASSERT(!src1_q8_1_given || !use_native_fp4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        // the q8_1 layout every producer of this product's activations uses (DSY: the ring pipeline)
        const int y_layout = use_native_fp4 ? -1 : ggml_cuda_mul_mat_q_y_layout(src0, src1);
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        if (src1_q8_1_given == nullptr) {
            src1_q8_1.alloc(nbytes_src1_q8_1);
        }
        char * src1_q8_1_ptr = src1_q8_1_given != nullptr ? src1_q8_1_given : src1_q8_1.get();
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        if (!src1_q8_1_filled) {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1_ptr, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else if (src1_pre_op) {
                quantize_mmq_q8_1_pre_op_cuda(*src1_pre_op, src1_q8_1_ptr, src0->type, ne10, ne10_padded,
                                              ne11, ne12, ne13, stream, y_layout);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1_ptr, src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream, y_layout);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1_ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1, y_layout};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

// ---- [gate | up] MUL_MAT -> SWIGLU -> quantized MUL_MAT (mul_mat_q_glu_q8_1, mmq-async.cuh) -----------------------------

// Whether ggml_cuda_mul_mat_q would compute every output of src0 x src1 (no ids, one channel and sample) with J = 128
// and one CUDA block per whole output tile (no stream-k split along K), which is what the SwiGLU kernel does: then its
// sums are the same.
static bool ggml_cuda_mul_mat_q_whole_tiles_J128(const ggml_type type, const int64_t nrows_x, const int64_t ncols, const int cc,
                                                  const int nsm, const size_t smpbo) {
    const bool fallback = nrows_x % 128 != 0;
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) { // as mul_mat_q_switch_J chooses J
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles_x = (ncols + config.J - 1) / config.J;
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    if (J_best != 128) {
        return false;
    }
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 128, fallback, cc);
    if (!ggml_cuda_mmq_get_stream_k(type, 128, fallback, cc)) {
        return true;
    }
    // launch_mul_mat_q_impl: stream-k kernel with one block per tile when the tiles fill >= 90% of the waves (NVIDIA)
    const int64_t nty    = (nrows_x + config.I - 1) / config.I;
    const int64_t ntx    = (ncols   + config.J - 1) / config.J;
    const int64_t ntiles = nty*ntx;
    const int64_t nwaves = (ntiles + nsm - 1) / nsm;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && 100*ntiles / (nsm*nwaves) >= 90;
}

bool ggml_cuda_mul_mat_q_glu_supported(const ggml_tensor * mm, const ggml_tensor * mm_out) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if ((src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) || src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
        mm->src[2] != nullptr || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src0->ne[1] % 128 != 0 || src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) ||
        (uintptr_t) src0->data % 16 != 0 || src0->nb[1] % 16 != 0) {
        return false;
    }
    // the second MUL_MAT's src1 rows: the SWIGLU's ne01/2 values, no padding (the epilogue writes every q8_1 block)
    if (mm_out->src[1]->ne[0] != src0->ne[1]/2 || (src0->ne[1]/2) % MATRIX_ROW_PADDING != 0 || mm_out->src[1]->ne[1] != src1->ne[1]) {
        return false;
    }
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, 128, false, cc);
    if (ggml_cuda_mmq_get_pipe() == GGML_CUDA_MMQ_PIPE_OFF || !ggml_cuda_mmq_async_available(config, cc, ggml_cuda_info().devices[id].smpbo)) {
        return false;
    }
    if (ggml_cuda_mul_mat_q_y_layout(src0, src1) == MMQ_Q8_1_DS_LAYOUT_DSY ||
            ggml_cuda_mul_mat_q_y_layout(mm_out->src[0], mm_out->src[1]) == MMQ_Q8_1_DS_LAYOUT_DSY) {
        return false; // TODO: the SwiGLU kernel on the ring pipeline
    }
    return ggml_cuda_mul_mat_q_whole_tiles_J128(src0->type, src0->ne[1], src1->ne[1], cc, ggml_cuda_info().devices[id].nsm,
                                                ggml_cuda_info().devices[id].smpbo);
}

template <ggml_type type, int qlayout>
static void ggml_cuda_mul_mat_q_glu_launch(ggml_backend_cuda_context & ctx, const char * x, const int * y, void * vq,
                                           const int64_t ne00, const int64_t ne01, const int64_t stride_row_x, const int64_t ne11,
                                           cudaStream_t stream) {
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, 128, false, cc);
    const int nbytes_shared = (int) ggml_cuda_mmq_async_nbytes_shared(config);
    GGML_ASSERT(nbytes_shared >= int(128*sizeof(int) + 128*(128 + 4)*sizeof(float))); // the epilogue's [token][row] buffer
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_glu_q8_1<type, qlayout>), nbytes_shared);

    const int ntx = (int) ((ne11 + 127) / 128);
    const int nty = (int) (ne01 / 128);
    // the grouped tile order of the default launch (same groups as ggml_cuda_mmq_get_tile_group would give)
    mmq_args args = {};
    args.type_x = type; args.ncols_x = ne00; args.nrows_x = ne01; args.nchannels_y = 1; args.nsamples_y = 1;
    args.ncols_max = ne11; args.ncols_opt = ne11;
    const int tile_group = ggml_cuda_mmq_get_tile_group(config, args, cc, nsm);

    GGML_ASSERT(int64_t(nty)*ntx < INT_MAX);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMQ_ASYNC_NTHREADS/WARP_SIZE, 1);
    mul_mat_q_glu_q8_1<type, qlayout><<<nty*ntx, block_dims, nbytes_shared, stream>>>(
        x, y, vq, (int) (ne00 / ggml_cuda_type_traits<type>::qk), (int) stride_row_x, (int) ne11, (int) (ne01/2), ntx, nty, tile_group);
}

void ggml_cuda_mul_mat_q_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, char * src1_q8_1_given, bool src1_q8_1_filled,
                             void * vq_out, int out_layout) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    cudaStream_t stream = ctx.stream();

    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * y = src1_q8_1_given;
    if (y == nullptr) {
        y = src1_q8_1.alloc(ggml_cuda_mul_mat_q_src1_q8_1_bytes(src0, src1));
    }
    if (!src1_q8_1_filled) {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, y, src0->type, src1->ne[0], s11, s12, s13, ne10_padded,
                               src1->ne[1], 1, 1, stream);
        CUDA_CHECK(cudaGetLastError());
    }
    GGML_ASSERT((uintptr_t) y % 16 == 0 && (uintptr_t) vq_out % 16 == 0);

    const int64_t stride_row_x = src0->nb[1] / ggml_type_size(src0->type);
#define GLU_LAUNCH(T, L) ggml_cuda_mul_mat_q_glu_launch<T, L>(ctx, (const char *) src0->data, (const int *) y, vq_out, \
        src0->ne[0], src0->ne[1], stride_row_x, src1->ne[1], stream)
    switch (out_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            if (src0->type == GGML_TYPE_Q4_K) { GLU_LAUNCH(GGML_TYPE_Q4_K, MMQ_Q8_1_DS_LAYOUT_D4); } else { GLU_LAUNCH(GGML_TYPE_Q5_K, MMQ_Q8_1_DS_LAYOUT_D4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            if (src0->type == GGML_TYPE_Q4_K) { GLU_LAUNCH(GGML_TYPE_Q4_K, MMQ_Q8_1_DS_LAYOUT_DS4); } else { GLU_LAUNCH(GGML_TYPE_Q5_K, MMQ_Q8_1_DS_LAYOUT_DS4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            if (src0->type == GGML_TYPE_Q4_K) { GLU_LAUNCH(GGML_TYPE_Q4_K, MMQ_Q8_1_DS_LAYOUT_D2S6); } else { GLU_LAUNCH(GGML_TYPE_Q5_K, MMQ_Q8_1_DS_LAYOUT_D2S6); }
            break;
        default:
            GGML_ABORT("unknown q8_1 layout %d", out_layout);
    }
#undef GLU_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

size_t ggml_cuda_mul_mat_q_src1_q8_1_bytes(const ggml_tensor * src0, const ggml_tensor * src1) {
    // as ggml_cuda_mul_mat_q computes nbytes_src1_q8_1 without ids and with q8_1 activations
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const bool fallback = src0->ne[1] % 128 != 0;
    return src1->ne[3]*src1->ne[2] * src1->ne[1]*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, src1->ne[1]) * sizeof(block_q8_1_mmq);
}

int ggml_cuda_mul_mat_q_y_layout(const ggml_tensor * src0, const ggml_tensor * src1) {
    const bool x_aligned = (uintptr_t) src0->data % 16 == 0 && src0->nb[1] % 16 == 0;
    return ggml_cuda_mmq_y_layout(src0->type, src0->ne[1], src1->ne[1], x_aligned);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
