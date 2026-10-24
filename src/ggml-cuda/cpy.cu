#include "cpy.cuh"
#include "dequantize.cuh"
#include "cpy-utils.cuh"
#include "fp8.cuh"
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
#include "ggml-musa/mudnn.cuh"
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

const int CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

template <cpy_kernel_t cpy_1>
static __global__ void cpy_scalar(const char * cx, char * cdst, const int64_t ne,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                  const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                  const int64_t nb12, const int64_t nb13) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    ggml_cuda_pdl_sync();
    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <typename T>
static __global__ void cpy_scalar_transpose(const char * cx, char * cdst, const int64_t ne,
                               const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                               const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                               const int64_t nb12, const int64_t nb13) {

    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const int64_t x  = (int64_t) blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int64_t y  = (int64_t) blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    const int64_t tx = (int64_t) blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset
    const int64_t ty = (int64_t) blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    __shared__ float tile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];
    int cur_tile_buf = 0;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {

        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM + i;
        if (imat >= nmat)
            break;

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if(x < ne01 && y + j < ne00){
                const int row = threadIdx.y+j;
                const int col = threadIdx.x * sizeof(float)/sizeof(T);
                T *tile2 = reinterpret_cast<T*>(tile[cur_tile_buf][row]);
                tile2[col] = src[imat*n + (y+j)*ne01 + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (ty + j < ne01 && tx < ne00) {
                const int col = (threadIdx.y+j)*sizeof(float)/sizeof(T);
                const T *tile2 = reinterpret_cast<const T*>(tile[cur_tile_buf][threadIdx.x]);
                dst[imat*n + (ty+j)*ne00 + tx] = tile2[col];
            }
        }

        cur_tile_buf = (cur_tile_buf + 1) % 2;
    }

    GGML_UNUSED_VARS(ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
        nb12, nb13);
}

// Spellcast3D: a same-type copy whose source is a permuted view of a contiguous tensor that amounts to ONE 2D
// transpose once the dimensions that stay neighbours in both the source's memory and the destination are merged.
// Wan's VAE moves its maps between channel-first and channel-last for every RMS norm: [c][y][x] -> [y][x][c] is a
// 144 x 1,048,576 matrix transposed (and back). The generic kernel above reads such a source with a whole plane
// between neighbouring lanes (28 GB/s on a GTX 1070: 1.5 s of a 7.7 s decode); the tiled transpose below runs at
// the memory speed. The same bytes end up in the same places - a copy is a copy.
template <typename T>
static __global__ void cpy_transpose2d_tiled(const char * cx, char * cdst, const int64_t rows, const int64_t cols,
                                             const int64_t tiles_x) {
    // src: [rows][cols] row-major; dst: [cols][rows]
    __shared__ T tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];
    ggml_cuda_pdl_lc();
    const T * src = (const T *) cx;
    T *       dst = (T *) cdst;
    const int64_t t  = blockIdx.x;
    const int64_t bx = (t % tiles_x) * CUDA_CPY_TILE_DIM_2D;   // first column of the source tile
    const int64_t by = (t / tiles_x) * CUDA_CPY_TILE_DIM_2D;   // first row of the source tile
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
        const int64_t r = by + threadIdx.y + j, c = bx + threadIdx.x;
        if (r < rows && c < cols) {
            tile[threadIdx.y + j][threadIdx.x] = src[r*cols + c];
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
        const int64_t c = bx + threadIdx.y + j, r = by + threadIdx.x;   // the destination's row is a source column
        if (r < rows && c < cols) {
            dst[c*rows + r] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// true (with the source seen as a [rows][cols] row-major matrix) when the copy is that 2D transpose: same type,
// a contiguous destination, and the source's non-unit dimensions, with logical neighbours that are also memory
// neighbours merged, form exactly two groups, the logical inner one strided by the outer one's extent
static bool ggml_cuda_cpy_as_transpose2d(const ggml_tensor * src0, const ggml_tensor * src1, int64_t & rows, int64_t & cols) {
    if (src0->type != src1->type || ggml_blck_size(src0->type) != 1 || !ggml_is_contiguous(src1) || ggml_is_contiguous(src0)) {
        return false;
    }
    const int64_t esize = (int64_t) ggml_type_size(src0->type);
    int64_t gne[GGML_MAX_DIMS], gnb[GGML_MAX_DIMS];
    int     g = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (src0->ne[i] <= 1) {
            continue;
        }
        if (g > 0 && (int64_t) src0->nb[i] == gnb[g - 1]*gne[g - 1]) {
            gne[g - 1] *= src0->ne[i];
        } else {
            gne[g] = src0->ne[i];
            gnb[g] = (int64_t) src0->nb[i];
            g++;
        }
    }
    if (g != 2 || gnb[1] != esize || gnb[0] != esize*gne[1]) {
        return false;
    }
    rows = gne[0];
    cols = gne[1];
    return true;
}

template <typename T>
static void ggml_cpy_transpose2d_cuda(const char * cx, char * cdst, const int64_t rows, const int64_t cols, cudaStream_t stream) {
    const int64_t tiles_x = (cols + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    const int64_t tiles   = tiles_x * ((rows + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D);
    GGML_ASSERT(tiles <= INT_MAX);
    const dim3 block(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3) (unsigned) tiles, block, 0, stream);
    ggml_cuda_kernel_launch(cpy_transpose2d_tiled<T>, launch_params, cx, cdst, rows, cols, tiles_x);
}

static __device__ void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        float2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}

template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < qk/2; j++) {
        float2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + qk/2) = dq.y;
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    ggml_cuda_pdl_sync();
    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    ggml_cuda_pdl_sync();
    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template<typename src_t, typename dst_t>
static __global__ void cpy_scalar_contiguous(const char * cx, char * cdst, const int64_t ne) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const src_t * x = (const src_t *) cx;
    dst_t *     dst = (dst_t *) cdst;

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<dst_t>(x[i]);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_scalar_contiguous_cuda(
    const char * cx, char * cdst, const int64_t ne,
cudaStream_t stream) {

    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(cpy_scalar_contiguous<src_t, dst_t>, launch_params, cx, cdst, ne);
}

template<typename src_t, typename dst_t, bool transposed = false>
static void ggml_cpy_scalar_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    const auto launch_scalar_generic = [&]() {
        const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        GGML_ASSERT(num_blocks <= INT_MAX);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(cpy_scalar<cpy_1_scalar<src_t, dst_t>>, launch_params,
            cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    };

    if (transposed) {
        GGML_ASSERT(ne == ne00*ne01*ne02);  // ne[3] is 1 assumed
        int64_t ne00n, ne01n, ne02n;
        if (nb00 <= nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = ne00;
            ne01n = ne01;
            ne02n = ne02;
        } else {
            ne00n = ne00;
            ne01n = ne01*ne02;
            ne02n = 1;
        }

        int64_t grid_x = (ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_y = (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_z = (ne/(ne01n*ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM;
        GGML_ASSERT(grid_x <= INT_MAX);
        if (grid_y > USHRT_MAX || grid_z > USHRT_MAX) {
            launch_scalar_generic();
        } else {
            dim3 dimGrid(grid_x, grid_y, grid_z);
            dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dimGrid, dimBlock, 0, stream);
            ggml_cuda_kernel_launch(cpy_scalar_transpose<dst_t>, launch_params,
                cx, cdst, ne, ne00n, ne01n, ne02n, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
        }
    } else {
        launch_scalar_generic();
    }
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int64_t num_blocks = (ne/QK8_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q8_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    const int64_t num_blocks = (ne/QK8_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int64_t num_blocks = (ne/QK4_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = (ne/QK4_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int64_t num_blocks = (ne/QK4_1 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = (ne/QK4_1 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int64_t num_blocks = (ne/QK5_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = (ne/QK5_0 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int64_t num_blocks = (ne/QK5_1 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = (ne/QK5_1 + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int64_t num_blocks = (ne/QK4_NL + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

// check if a same-type copy reduces to a 2D strided copy (height rows of width
// contiguous bytes), so it can use cudaMemcpy2DAsync instead of the scalar kernel
static bool ggml_cuda_cpy_as_memcpy_2d(const ggml_tensor * src0, const ggml_tensor * src1,
        size_t & width, size_t & height, size_t & spitch, size_t & dpitch) {
    // require matching shape: a reshaped copy maps elements by flat order, which the
    // prefix walk below does not handle
    if (src0->type != src1->type || !ggml_are_same_shape(src0, src1)) {
        return false;
    }

    // grow the contiguous prefix block shared by both tensors
    size_t block_nb = ggml_element_size(src0);
    int d = 0;
    for (; d < GGML_MAX_DIMS; ++d) {
        if (src0->nb[d] != block_nb || src1->nb[d] != block_nb) {
            break;
        }
        block_nb *= src0->ne[d];
    }

    // d == 0: nothing contiguous; d == GGML_MAX_DIMS: fully contiguous (handled by memcpy)
    if (d == 0 || d == GGML_MAX_DIMS) {
        return false;
    }

    // dim d carries the rows; everything above it must be a single element
    for (int i = d + 1; i < GGML_MAX_DIMS; ++i) {
        if (src0->ne[i] != 1) {
            return false;
        }
    }

    width  = block_nb;
    height = src0->ne[d];
    spitch = src0->nb[d];
    dpitch = src1->nb[d];

    return spitch >= width && dpitch >= width;
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
    const bool can_be_transposed = nb01 == (int64_t)ggml_element_size(src0) &&
        src0->ne[3] == 1 && nb02 == ne00 * ne01 * (int64_t)ggml_element_size(src0);

    size_t mc_width = 0, mc_height = 0, mc_spitch = 0, mc_dpitch = 0;
    int64_t t_rows = 0, t_cols = 0;

    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        } else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
        }
    } else if (ggml_cuda_cpy_as_memcpy_2d(src0, src1, mc_width, mc_height, mc_spitch, mc_dpitch)) {
        CUDA_CHECK(cudaMemcpy2DAsync(src1_ddc, mc_dpitch, src0_ddc, mc_spitch,
                                     mc_width, mc_height, cudaMemcpyDeviceToDevice, main_stream));
    } else if (ggml_cuda_cpy_as_transpose2d(src0, src1, t_rows, t_cols)) {
        switch (ggml_type_size(src0->type)) {   // Spellcast3D: the permute that is a 2D transpose, by element size
            case 1:  ggml_cpy_transpose2d_cuda<uint8_t> (src0_ddc, src1_ddc, t_rows, t_cols, main_stream); break;
            case 2:  ggml_cpy_transpose2d_cuda<uint16_t>(src0_ddc, src1_ddc, t_rows, t_cols, main_stream); break;
            case 4:  ggml_cpy_transpose2d_cuda<uint32_t>(src0_ddc, src1_ddc, t_rows, t_cols, main_stream); break;
            case 8:  ggml_cpy_transpose2d_cuda<uint64_t>(src0_ddc, src1_ddc, t_rows, t_cols, main_stream); break;
            default: GGML_ABORT("%s: unexpected element size %zu\n", __func__, ggml_type_size(src0->type));
        }
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<ggml_fp8_e4m3_cuda, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<ggml_fp8_e4m3_cuda, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F8_E5M2 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<ggml_fp8_e5m2_cuda, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<ggml_fp8_e5m2_cuda, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F8_E4M3 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<ggml_fp8_e4m3_cuda, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<ggml_fp8_e4m3_cuda, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F8_E5M2 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<ggml_fp8_e5m2_cuda, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<ggml_fp8_e5m2_cuda, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<float, float, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            // the vectorized conversion flash attention uses for F32 K/V (four elements per thread where aligned):
            // the same ggml_cuda_cast<half> per element as cpy_scalar_contiguous, at close to full bandwidth
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src0_ddc, (half *) src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<half, half, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<int32_t, int32_t, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<int16_t, int16_t, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int16_t, int16_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}
