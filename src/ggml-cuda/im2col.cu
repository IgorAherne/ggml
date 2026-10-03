#include "im2col.cuh"

#include <algorithm>
#include <climits>

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t N, int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int tid = threadIdx.x;

    const int64_t total_channels = IC * KH * KW;
    const int threads_per_pos = blockDim.x;
    const int64_t start_ch = tid;
    const int64_t stride_ch = threads_per_pos;

    for (int64_t iow = blockIdx.x; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.y; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in = iz / OH;
            const int64_t ioh = iz - in * OH;

            for (int64_t iic_khw = start_ch; iic_khw < total_channels; iic_khw += stride_ch) {
                const int64_t iic = iic_khw / KH_KW;
                const int64_t rem = iic_khw - iic * KH_KW;
                const int64_t ikh = rem / KW;
                const int64_t ikw = rem - ikh * KW;

                const int64_t iiw = iow * s0 + ikw * d0 - p0;
                const int64_t iih = ioh * s1 + ikh * d1 - p1;

                const int64_t offset_dst =
                    ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                    dst[offset_dst] = 0.0f;
                } else {
                    const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
                    dst[offset_dst] = x[offset_src + iih * IW + iiw];
                }
            }
        }
    }
}

// The same 2D im2col in tiles, for speed (the plain kernel above does 64-bit divisions per value and reads the input
// scattered): a block takes TW consecutive output columns of one output row and image and TC input channels, stages the
// KH input rows they read into shared memory (rows loaded coalesced, each warp's loads issued together, 0 for the
// padding), then writes every output position's TC*KH*KW consecutive values with consecutive threads. Each value and
// its conversion to T are exactly the plain kernel's (a copy of x, or 0 for the padding).
#define IM2COL_TILED_TW       64
#define IM2COL_TILED_THREADS 256
#define IM2COL_TILED_SMEM    (32*1024)
#define IM2COL_TILED_ROWS      4
#define IM2COL_TILED_COLS    160

template <typename T>
static __global__ void __launch_bounds__(IM2COL_TILED_THREADS) im2col_tiled_kernel(
        const float * x, T * dst, const int IC, const int IW, const int IH, const int OW, const int OH,
        const int KW, const int KH, const int64_t s_c, const int64_t s_n, const int TC, const int SW,
        const int s0, const int s1, const int p0, const int p1, const int d0, const int d1) {
    extern __shared__ float im2col_tile[];  // [TC][KH][SW]

    // grid: x = column tile, y = image * output row, z = channel chunk
    const int ow0 = blockIdx.x * IM2COL_TILED_TW;
    const int n   = blockIdx.y / OH;
    const int oh  = blockIdx.y - n*OH;
    const int ic0 = blockIdx.z * TC;
    const int tc  = min(TC, IC - ic0);
    const int iw0 = ow0*s0 - p0;

    // each warp loads IM2COL_TILED_ROWS tile rows (of SW <= IM2COL_TILED_COLS columns) at a time, all loads first
    const int warp   = threadIdx.x / WARP_SIZE;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;  // full warps (a last partial warp does not load)
    const int nrows  = warp < nwarps ? tc*KH : 0;
    for (int r0 = warp*IM2COL_TILED_ROWS; r0 < nrows; r0 += nwarps*IM2COL_TILED_ROWS) {
        float v[IM2COL_TILED_ROWS][IM2COL_TILED_COLS/WARP_SIZE];
#pragma unroll
        for (int q = 0; q < IM2COL_TILED_ROWS; ++q) {
            const int  r  = r0 + q;
            const int  c  = r / KH;
            const int  kh = r - c*KH;
            const int  ih = oh*s1 + kh*d1 - p1;
            const bool row_ok = r < nrows && ih >= 0 && ih < IH;
            const float * xrow = x + n*s_n + (ic0 + (row_ok ? c : 0))*s_c + (int64_t) (row_ok ? ih : 0)*IW;
#pragma unroll
            for (int jj = 0; jj < IM2COL_TILED_COLS/WARP_SIZE; ++jj) {
                const int j  = lane + jj*WARP_SIZE;
                const int iw = iw0 + j;
                v[q][jj] = row_ok && j < SW && iw >= 0 && iw < IW ? xrow[iw] : 0.0f;
            }
        }
#pragma unroll
        for (int q = 0; q < IM2COL_TILED_ROWS; ++q) {
#pragma unroll
            for (int jj = 0; jj < IM2COL_TILED_COLS/WARP_SIZE; ++jj) {
                const int j = lane + jj*WARP_SIZE;
                if (r0 + q < nrows && j < SW) {
                    im2col_tile[(r0 + q)*SW + j] = v[q][jj];
                }
            }
        }
    }
    __syncthreads();

    const int KHKW = KH*KW;
    const int L    = TC*KHKW;          // values of one position in a full channel chunk
    const int ppb  = blockDim.x / L;   // positions per pass
    if ((int) threadIdx.x >= ppb*L) {
        return;
    }
    const int e  = threadIdx.x % L;
    const int c  = e / KHKW;
    const int k  = e - c*KHKW;
    const int kh = k / KW;
    const int kw = k - kh*KW;
    if (c >= tc) {
        return;
    }
    const int     tw  = min(IM2COL_TILED_TW, OW - ow0);
    const int64_t row = (int64_t) IC*KHKW;
    T * d = dst + ((int64_t) (n*OH + oh)*OW + ow0)*row + (int64_t) ic0*KHKW + e;
    const float * t = im2col_tile + (c*KH + kh)*SW + kw*d0;
    for (int pos = threadIdx.x / L; pos < tw; pos += ppb) {
        d[pos*row] = t[pos*s0];
    }
}

// the tiled kernel for a 2D im2col when its tiles fit; false: use the plain kernel
template <typename T>
static bool im2col_tiled_cuda(const float * x, T * dst,
        int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
        int64_t N, int64_t s_c, int64_t s_n, int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream) {
    const int64_t KHKW = KH*KW;
    const int64_t SW   = (IM2COL_TILED_TW - 1)*(int64_t) s0 + (KW - 1)*(int64_t) d0 + 1;
    if (KHKW > IM2COL_TILED_THREADS || N*OH > 65535 || SW > IM2COL_TILED_COLS || s0 < 1 || s1 < 1 || d0 < 1 || d1 < 1 ||
        IW > INT_MAX/2 || IH > INT_MAX/2 || OW > INT_MAX/2 || IC > INT_MAX/2 || KH*SW*(int64_t) sizeof(float) > IM2COL_TILED_SMEM) {
        return false;
    }
    // channels per block: as many as the threads (one value each per position) and the shared memory allow, evened
    // out over the chunks
    int64_t tc_max = std::min<int64_t>(IM2COL_TILED_THREADS / KHKW, IM2COL_TILED_SMEM / (KH*SW*(int64_t) sizeof(float)));
    tc_max = std::max<int64_t>(1, std::min<int64_t>(tc_max, IC));
    const int64_t nchunks = (IC + tc_max - 1) / tc_max;
    const int64_t TC      = (IC + nchunks - 1) / nchunks;
    if (nchunks > 65535) {
        return false;
    }
    const int  threads = (int) ((IM2COL_TILED_THREADS / (TC*KHKW)) * (TC*KHKW));
    const dim3 grid((unsigned) ((OW + IM2COL_TILED_TW - 1) / IM2COL_TILED_TW), (unsigned) (N*OH), (unsigned) nchunks);
    const size_t smem = (size_t) (TC*KH*SW) * sizeof(float);
    im2col_tiled_kernel<T><<<grid, threads, smem, stream>>>(x, dst, (int) IC, (int) IW, (int) IH, (int) OW, (int) OH,
        (int) KW, (int) KH, s_c, s_n, (int) TC, (int) SW, s0, s1, p0, p1, d0, d1);
    return true;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    if (IH > 1 && im2col_tiled_cuda<T>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream)) {
        return;
    }
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    const int threads_per_block = MIN((int)IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE);
    dim3 block_nums(MIN(OW, MAX_GRIDDIM_Y), MIN(N_OH, MAX_GRIDDIM_Z));

    im2col_kernel<<<block_nums, threads_per_block, 0, stream>>>(
        x, dst, IC, IW, IH, OH, OW, KW, KH,
        N, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
        s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static  __global__ void im2col_3d_kernel(
        const float * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KD_KH_KW) {
        return;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    const int64_t iic = i / KD_KH_KW;
    const int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    const int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    const int64_t ikw = i % KW;

    for (int64_t iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y) {
        for (int64_t iz = blockIdx.z; iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
            const int64_t in  = iz / OD_OH;
            const int64_t iod = (iz - in*OD_OH) / OH;
            const int64_t ioh = iz % OH;

            const int64_t iiw = iow * s0 + ikw * d0 - p0;
            const int64_t iih = ioh * s1 + ikh * d1 - p1;
            const int64_t iid = iod * s2 + ikd * d2 - p2;

            const int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                dst[offset_dst] = 0.0f;
            } else {
                const int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
                dst[offset_dst] = src[offset_src];
            }
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void im2col_3d_cuda(const float * src, T* dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {
    const int64_t OH_OW = OH*OW;
    const int64_t KD_KH_KW = KD*KH*KW;
    const int64_t ID_IH_IW = ID*IH*IW;
    const int64_t KH_KW = KH*KW;
    const int64_t IH_IW = IH*IW;
    const int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    const int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    const int64_t N_OD_OH = N*OD*OH;
    const int64_t OD_OH = OD*OH;
    const int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    const int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    const int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    const int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    const int64_t num_blocks = (IC_KD_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    dim3 block_nums(num_blocks, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OD_OH, MAX_GRIDDIM_Z));
    im2col_3d_kernel<<<block_nums, MIN(IC_KD_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                                                                                           OH_OW, KD_KH_KW, ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW,
                                                                                           IC_KD_KH_KW, OW_KD_KH_KW, OD_OH_OW_IC_KD_KH_KW,
                                                                                           OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                                                                           stride_q, stride_z, stride_y, stride_x,
                                                                                           s0, s1, s2, p0, p1, p2, d0, d1, d2);
}

static void im2col_3d_cuda_f16(const float * src, half * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                         stride_q, stride_z, stride_y, stride_x,
                         s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

static void im2col_3d_cuda_f32(const float * src, float * dst,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream) {

    im2col_3d_cuda<float>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t OC = ne03 / IC;
    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const size_t  es       = ggml_element_size(src1);
    const int64_t stride_x = src1->nb[0] / es;
    const int64_t stride_y = src1->nb[1] / es;
    const int64_t stride_z = src1->nb[2] / es;
    const int64_t stride_q = src1->nb[3] / es;

    if(dst->type == GGML_TYPE_F16) {
        im2col_3d_cuda_f16(src1_d, (half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } else {
        im2col_3d_cuda_f32(src1_d, (float *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
