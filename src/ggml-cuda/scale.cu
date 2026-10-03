#include "scale.cuh"

#include <algorithm>
#include <climits>
#include <cstring>

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32, launch_params, x, dst, scale, bias, nelements);
}

// SCALE followed by one or two ADDs of its result (e.g. a convolution's output scale, its bias and a residual):
// v = scale*x + bias with scale_f32's FFMA, then v + y[broadcast] and optionally v + z with the ADD kernels' FADD,
// operands in the graph's order. x, z and dst contiguous with dst's shape, y repeatable to it (element strides).
// Elementwise: dst may be x or z itself.
template <bool add2>
static __global__ void scale_add_f32(const float * x, const float * y, const float * z, float * dst,
                                     const float scale, const float bias, const uint32_t n,
                                     const uint3 ne0, const uint3 ne1, const uint3 ne2,
                                     const uint3 yne0, const uint3 yne1, const uint3 yne2, const uint3 yne3,
                                     const uint32_t ys1, const uint32_t ys2, const uint32_t ys3,
                                     const bool y_first, const bool z_first) {
    for (uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
        const uint2 d0 = fast_div_modulo(i, ne0);
        const uint2 d1 = fast_div_modulo(d0.x, ne1);
        const uint2 d2 = fast_div_modulo(d1.x, ne2);
        const uint32_t yi = fastmodulo(d0.y, yne0) + fastmodulo(d1.y, yne1)*ys1 + fastmodulo(d2.y, yne2)*ys2 +
                            fastmodulo(d2.x, yne3)*ys3;
        float v = __fmaf_rn(scale, x[i], bias);
        v = y_first ? __fadd_rn(y[yi], v) : __fadd_rn(v, y[yi]);
        if constexpr (add2) {
            v = z_first ? __fadd_rn(z[i], v) : __fadd_rn(v, z[i]);
        }
        dst[i] = v;
    }
}

void ggml_cuda_op_scale_add_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node, const ggml_tensor * y,
                                  const bool y_first, const ggml_tensor * z, const bool z_first, ggml_tensor * dst) {
    const ggml_tensor * x = scale_node->src[0];
    GGML_ASSERT(ggml_nelements(dst) < INT32_MAX && ggml_nelements(dst) == ggml_nelements(scale_node));
    const uint32_t n = (uint32_t) ggml_nelements(dst);
    // y broadcasts over the first ADD's shape, which is the SCALE's (dst may be a reshape of it)
    const uint3 ne0 = init_fastdiv_values(scale_node->ne[0]);
    const uint3 ne1 = init_fastdiv_values(scale_node->ne[1]);
    const uint3 ne2 = init_fastdiv_values(scale_node->ne[2]);
    const uint3 yne0 = init_fastdiv_values(y->ne[0]);
    const uint3 yne1 = init_fastdiv_values(y->ne[1]);
    const uint3 yne2 = init_fastdiv_values(y->ne[2]);
    const uint3 yne3 = init_fastdiv_values(y->ne[3]);
    const uint32_t ys1 = (uint32_t) (y->nb[1] / sizeof(float));
    const uint32_t ys2 = (uint32_t) (y->nb[2] / sizeof(float));
    const uint32_t ys3 = (uint32_t) (y->nb[3] / sizeof(float));
    float sc;
    float sb;
    memcpy(&sc, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&sb, (const float *) scale_node->op_params + 1, sizeof(float));
    const int block = 256;
    const int grid  = (int) std::min<int64_t>((n + block - 1) / block, 65535*8);
    cudaStream_t stream = ctx.stream();
    if (z) {
        scale_add_f32<true><<<grid, block, 0, stream>>>((const float *) x->data, (const float *) y->data, (const float *) z->data,
            (float *) dst->data, sc, sb, n, ne0, ne1, ne2, yne0, yne1, yne2, yne3, ys1, ys2, ys3, y_first, z_first);
    } else {
        scale_add_f32<false><<<grid, block, 0, stream>>>((const float *) x->data, (const float *) y->data, nullptr,
            (float *) dst->data, sc, sb, n, ne0, ne1, ne2, yne0, yne1, yne2, yne3, ys1, ys2, ys3, y_first, false);
    }
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
}
