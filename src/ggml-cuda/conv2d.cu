#include "conv2d.cuh"
#include "convert.cuh"
#include "mma.cuh"
#include "conv2d-mma.cuh"

struct conv_params {
    const int64_t IW, IH;
    const int64_t OW, OH;
    const int64_t KW, KH;
    const int64_t ST_X, ST_Y;
    const int64_t PD_X, PD_Y;
    const int64_t DL_X, DL_Y;
    const int64_t IC, OC;
    const int64_t B;
    const int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

__device__ __forceinline__ int64_t max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int64_t out_coord,
                                                     int64_t kern_coord,
                                                     int64_t stride,
                                                     int64_t dilation,
                                                     int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
    }

    __device__ static int64_t kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & P) {
        return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
        return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
    }

    __device__ static void unpack_indices(int64_t             global_idx,
                                          const conv_params & P,
                                          int64_t &           n,
                                          int64_t &           c,
                                          int64_t &           out_y,
                                          int64_t &           out_x) {
        out_x = global_idx % P.OW;
        out_y = (global_idx / P.OW) % P.OH;
        c     = (global_idx / (P.OW * P.OH)) % P.OC;
        n     = global_idx / (P.OW * P.OH * P.OC);
    }
};

template <typename T, typename Layout>
static __global__ void conv2d_kernel(const float * __restrict__ input,
                                     const T * __restrict__ kernel,
                                     float * __restrict__ output,
                                     const conv_params P) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    Layout::unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val = input[Layout::input_index(n, c_in, in_y, in_x, P)];
                const T kernel_val = kernel[Layout::kernel_index(c_out, c_in, ky, kx, P)];
                acc += (input_val * ggml_cuda_cast<float>(kernel_val));
            }
        }
    }

    // [N, OC, OH, OW]
    output[Layout::output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_cuda(const float * X_D, const T * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    const int blocks = (P.TOTAL + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
    conv2d_kernel<T, whcn_layout><<<blocks, CUDA_CONV2D_BLOCK_SIZE, 0, st>>>(X_D, K_D, Y_D, P);
}

static __global__ void
conv2d_pad_f16(const float * input, half * output, int iw, int ih, int pw, int ph, int px, int py, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int x = i % pw - px, y = i / pw % ph - py, nc = i / (pw * ph);
    output[i] = __float2half(
        (unsigned) x < (unsigned) iw && (unsigned) y < (unsigned) ih ? input[(nc * ih + y) * iw + x] : 0.0f);
}

template <int KW, int KH, bool use_mma>
static __global__ void conv2d_implicit_gemm_f16(const half * __restrict__ input,
                                                const half * __restrict__ weight,
                                                float * __restrict__ output,
                                                const conv_params P,
                                                const int         split_k) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nthreads  = 4 * warp_size;
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int AS = BK / 2 + 4;
    constexpr int BS = BN / 2 + 4;
    __shared__ __align__(16) half2 a_s[BM][AS];
    __shared__ __align__(16) half2 b_s[BK][BS];

    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int kw = KW ? KW : int(P.KW), kh = KH ? KH : int(P.KH);
    const int ic = int(P.IC), oc = int(P.OC);
    const int sx = int(P.ST_X), sy = int(P.ST_Y);
    const int dx = int(P.DL_X), dy = int(P.DL_Y);
    const int n = blockIdx.z / split_k, split = blockIdx.z % split_k;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    const int k_total   = ic * kw * kh;
    const int load_lane = warp_size == 32 ? threadIdx.x : threadIdx.x % (BN / 2);
    const int load_row  = threadIdx.y * (warp_size / (BN / 2)) + (warp_size == 32 ? 0 : threadIdx.x / (BN / 2));
    const int spatial   = n0 + 2 * load_lane;
    const int spatial0 = min(spatial, ow * oh - 1), spatial1 = min(spatial + 1, ow * oh - 1);
    const int y0 = spatial0 / ow, x0 = spatial0 % ow;
    const int y1 = spatial1 / ow, x1 = spatial1 % ow;
    const int pos0 = y0 * sy * iw + x0 * sx, pos1 = y1 * sy * iw + x1 * sx;

    [[maybe_unused]] const int wm = threadIdx.y / 2 * 32, wn = threadIdx.y % 2 * 32;
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    using tile_ab = tile<16, 8, half2, get_input_data_layout()>;
#    if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    // AMD accumulator fragments transpose the input fragment's row/column mapping.
    using tile_c = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
#    else
    using tile_c = tile<16, 16, float>;
#    endif
    [[maybe_unused]] tile_c c[2][2];
#else
    if constexpr (use_mma) {
        NO_DEVICE_CODE;
        return;
    }
#endif
    constexpr int              RM = 4, RN = BM * BN / (nthreads * RM);
    [[maybe_unused]] const int simt_m = tid / (BN / RN) * RM, simt_n = tid % (BN / RN) * RN;
    [[maybe_unused]] float     c_simt[RM][RN] = {};
    const int                  tiles          = (k_total + BK - 1) / BK;
    const int                  begin          = int(int64_t(tiles) * split / split_k) * BK;
    const int                  end            = int(int64_t(tiles) * (split + 1) / split_k) * BK;
    for (int k0 = begin; k0 < end; k0 += BK) {
        if (k_total % 8 == 0 && uintptr_t(weight) % 16 == 0) {
#pragma unroll
            for (int i = tid; i < BM * BK / 8; i += nthreads) {
                const int  row = i / (BK / 8), col = 8 * (i % (BK / 8));
                const int4 v                 = m0 + row < oc && k0 + col < k_total ?
                                                   ((const int4 *) weight)[((m0 + row) * k_total + k0 + col) / 8] :
                                                   make_int4(0, 0, 0, 0);
                *(int4 *) &a_s[row][col / 2] = v;
            }
        } else {
#pragma unroll
            for (int i = tid; i < BM * BK / 2; i += nthreads) {
                const int row = i / (BK / 2), col = 2 * (i % (BK / 2));
                half      lo = __float2half(0.0f), hi = lo;
                if (m0 + row < oc && k0 + col < k_total) {
                    lo = weight[(m0 + row) * k_total + k0 + col];
                    if (k0 + col + 1 < k_total) {
                        hi = weight[(m0 + row) * k_total + k0 + col + 1];
                    }
                }
                a_s[row][col / 2] = __halves2half2(lo, hi);
            }
        }
#pragma unroll
        for (int k = load_row; k < BK; k += nthreads / (BN / 2)) {
            const int ki = k0 + k;
            const int ci = ki / (kw * kh), ky = ki / kw % kh, kx = ki % kw;
            const int offset = ki < k_total ? (n * ic + ci) * ih * iw + ky * dy * iw + kx * dx : 0;
            half      lo = __float2half(0.0f), hi = lo;
            if (ki < k_total && spatial < ow * oh) {
                lo = input[offset + pos0];
            }
            if (ki < k_total && spatial + 1 < ow * oh) {
                hi = input[offset + pos1];
            }
            b_s[k][load_lane] = __halves2half2(lo, hi);
        }
        __syncthreads();
        if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
            for (int k = 0; k < BK; k += 16) {
                tile_ab a[2], b[2];
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_ldmatrix(a[i], &a_s[wm + 16 * i][k / 2], AS);
                    load_ldmatrix_trans(b[i], &b_s[k][(wn + 16 * i) / 2], BS);
                }
#    pragma unroll
                for (int i = 0; i < 2; ++i) {
#    pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        mma(c[i][j], a[i], b[j]);
                    }
                }
            }
#endif
        } else {
#pragma unroll 4
            for (int k = 0; k < BK; ++k) {
                float a[RM], b[RN];
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    a[i] = __half2float(((const half *) a_s[simt_m + i])[k]);
                }
#pragma unroll
                for (int j = 0; j < RN; ++j) {
                    b[j] = __half2float(((const half *) b_s[k])[simt_n + j]);
                }
#pragma unroll
                for (int i = 0; i < RM; ++i) {
#pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        c_simt[i][j] += a[i] * b[j];
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (use_mma) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#    pragma unroll
        for (int i = 0; i < 2; ++i) {
#    pragma unroll
            for (int j = 0; j < 2; ++j) {
#    pragma unroll
                for (int l = 0; l < c[i][j].ne; ++l) {
                    const int co  = m0 + wm + 16 * i + c[i][j].get_i(l);
                    const int pos = n0 + wn + 16 * j + c[i][j].get_j(l);
                    if (co < oc && pos < ow * oh) {
                        output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c[i][j].x[l];
                    }
                }
            }
        }
#endif
    } else {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int j = 0; j < RN; ++j) {
                const int co = m0 + simt_m + i, pos = n0 + simt_n + j;
                if (co < oc && pos < ow * oh) {
                    output[(int64_t(blockIdx.z) * oc + co) * ow * oh + pos] = c_simt[i][j];
                }
            }
        }
    }
}

// Spellcast3D: the F32 implicit GEMM for NVIDIA cards before Volta (the GTX 10xx), in place of the SIMT branch above
// (which stays for MooreThreads). Its output is bit-identical to that branch's: the same F16 operands converted to F32
// exactly, one F32 accumulator per output, `c += a * b` (one FFMA under -use_fast_math, as there) for every k of the
// split in increasing order, starting from 0. What differs is only data movement and tiling:
//   - a block computes TM*16 output channels x 128 positions (TM = 9: 144 channels, a divisor of the Wan VAE's 144 ..
//     1152, 3456), a thread TM channels x 8 positions, reading its operands of one k with four 16-byte shared loads
//     (and one 4-byte load for the 9th channel); 2 blocks of 256 threads a SM;
//   - the weights are transposed once per call into k rows of the block's channels (conv2d_simt_weights), so that a k
//     tile of them is one contiguous copy;
//   - the operands are converted F16 -> F32 when a k tile is staged into shared memory (global -> registers before the
//     current tile's math, registers -> the other shared buffer after it, one barrier per tile; no cp.async on Pascal);
//   - the input tile is gathered per k row (any shape), read as rows of 4 values (1x1, stride 1), or, for 3x3 / stride 1
//     maps whose width is a multiple of 64, staged a channel (9 k) at a time from the 3 input rows of each 64-position
//     run: every input value is loaded and converted once and stored at its 3 kx shifts.
// The reference zero-pads its last 64-wide k tile: those products are 0 * 0 and leave every sum as it is, except that
// they turn a -0 (a negative sum flushed to zero by FTZ) into +0; the epilogue repeats that when the split's 64-wide
// range ends past k_total. The split_k policy and conv2d_reduce_split_k are unchanged.
enum conv2d_simt_mode {
    CONV2D_SIMT_GATHER = 0,  // any shape: per k row, one 16-bit load per position (k tiles of 8)
    CONV2D_SIMT_ROWS   = 1,  // 1x1, stride 1, no padding, maps of whole 128-position tiles (k tiles of 8)
    CONV2D_SIMT_SEG3   = 2,  // 3x3, stride 1, dilation 1, width % 64 == 0, whole tiles, split_k == 1 (k tiles of 9)
    CONV2D_SIMT_NARROW = 3,  // <= 8 output channels, 3x3, stride 1, dilation 1, width % 8 == 0, split_k == 1
    CONV2D_SIMT_64     = 4,  // other shapes with <= 16 output channels: the 64x64 kernel itself (wastes less of a tile)
};

// Thread row ty (0..15) owns the block's channels 8 ty .. 8 ty + 7 and, for TM 9, 128 + ty: a k row of the weight tile
// is the block's BM channels in order, read by a thread as two 16-byte pieces and (TM 9) one value.
template <int TM> struct conv2d_simt_tile {
    static_assert(TM == 8 || TM == 9, "8 or 9 channels a thread");
    static constexpr int THR_M = 16, THR_N = 16;  // the thread grid
    static constexpr int TN    = 8;               // positions per thread: two groups of 4, 64 apart
    static constexpr int BM = TM * THR_M, BN = TN * THR_N;
};

// The weights transposed: [M tile][k < k_pad][BM channels], zero outside OC x k_total. A block transposes the BM
// channels x 32 k of one M tile through shared memory (grid: k_pad / 32 x M tiles).
template <int TM>
static __global__ void __launch_bounds__(256) conv2d_simt_weights(const half * __restrict__ w,
                                                                 half * __restrict__ wt,
                                                                 const int oc,
                                                                 const int k_total,
                                                                 const int k_pad) {
    static constexpr int BM = conv2d_simt_tile<TM>::BM, KT = 32;
    __shared__ half      t[BM][KT + 2];
    const int            mt = blockIdx.y, k0 = blockIdx.x * KT;
    for (int e = threadIdx.x; e < BM * KT; e += blockDim.x) {
        const int c = e / KT, kk = e - c * KT, co = mt * BM + c, k = k0 + kk;
        t[c][kk]    = co < oc && k < k_total ? w[int64_t(co) * k_total + k] : __float2half(0.0f);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < BM * KT; e += blockDim.x) {
        const int kk = e / BM, c = e - kk * BM;
        if (k0 + kk < k_pad) {
            wt[(int64_t(mt) * k_pad + k0 + kk) * BM + c] = t[c][kk];
        }
    }
}

// An empty asm that takes the value, so that the load producing it stays before the math: left alone, the compiler sinks
// the next tile's loads to their use after the math and their latency shows (GTX 1070: 4.7 -> 5.1 TFLOPS with this).
static __device__ __forceinline__ void conv2d_simt_issue(const uint32_t v) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    asm volatile("" ::"r"(v));
#else
    GGML_UNUSED(v);
#endif
}

template <int KW, int KH, int TM, int MODE>
static __global__ void __launch_bounds__(256, 2) conv2d_implicit_gemm_simt_f32(const half * __restrict__ input,
                                                                              const half * __restrict__ wt,
                                                                              float * __restrict__ output,
                                                                              const conv_params P,
                                                                              const int         split_k,
                                                                              const int         k_pad) {
    using T                       = conv2d_simt_tile<TM>;
    static constexpr int nthreads = 256, nwarps = 8;
    static constexpr int BM = T::BM, BN = T::BN, TN = T::TN;
    static constexpr int BK        = MODE == CONV2D_SIMT_SEG3 ? 9 : 8;
    static constexpr int A_VEC     = BK * BM / 8;  // 8-value (16-byte) pieces of a weight tile
    static constexpr int A_PER     = (A_VEC + nthreads - 1) / nthreads;
    static_assert(A_PER == 1, "a thread stages one 16-byte piece of the weight tile");
    static constexpr int A_S       = A_PER * 8 * nthreads;  // >= BK * BM: every thread stores its pieces (no branch)
    static constexpr int SEG_WORDS = 2 * 3 * 33;            // SEG3: 2 runs of 64 positions x 3 input rows x 66 values
    // A k row of the input tile: the two runs of 64 positions with 4 guard values on each side (SEG3 stores the values
    // that fall off a run there instead of branching): position p at 4 + p + 8 (p / 64).
    static constexpr int BS = 2 * (64 + 8);
    static_assert(BN == 4 * 32 && nwarps == 8 && BK % 8 == (MODE == CONV2D_SIMT_SEG3 ? 1 : 0), "tile layout");
    static_assert(MODE != CONV2D_SIMT_SEG3 || (KW == 3 && KH == 3), "SEG3 is 3x3 only");
    __shared__ __align__(16) float a_s[2][A_S];
    __shared__ __align__(16) float b_s[2][BK * BS];
    __shared__ int                 pos_s[MODE == CONV2D_SIMT_GATHER ? BN : 1];

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int kw = KW ? KW : int(P.KW), kh = KH ? KH : int(P.KH);
    const int ic = int(P.IC), oc = int(P.OC);
    const int sx = int(P.ST_X), sy = int(P.ST_Y);
    const int dx = int(P.DL_X), dy = int(P.DL_Y);
    const int n = blockIdx.z / split_k, split = blockIdx.z % split_k;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int k_total = ic * kw * kh, npos = ow * oh, ihw = ih * iw;

    // The reference's split of k: 64-wide tiles, `begin` and `end` multiples of 64; `end` may pass k_total.
    const int tiles = (k_total + 63) / 64;
    const int begin = int(int64_t(tiles) * split / split_k) * 64;
    const int end   = int(int64_t(tiles) * (split + 1) / split_k) * 64;
    const int kend  = min(end, k_total);

    const half * in_n = input + int64_t(n) * ic * ihw;

    // B staging, per mode. GATHER: the block's positions, clamped to the map (the extra columns are computed, never
    // stored). SEG3: thread t < SEG_WORDS loads the 32-bit word wd of input row ky of run r (the map row of positions
    // n0 + 64 r .. +63 and the two below it), values 2 wd and 2 wd + 1, and stores value e at its 3 kx shifts, position
    // 64 r + e - kx of k row ky*3 + kx (in a guard when e - kx is -1 or 64); the threads past SEG_WORDS repeat the last
    // word's loads and stores (same values, same places).
    const half * b_p     = MODE == CONV2D_SIMT_ROWS ? in_n + n0 + 4 * lane : in_n;
    int          seg_dst = 0;
    if (MODE == CONV2D_SIMT_GATHER && tid < BN) {
        const int s = min(n0 + tid, npos - 1);
        const int y = s / ow, x = s - y * ow;
        pos_s[tid]  = y * sy * iw + x * sx;
    }
    if (MODE == CONV2D_SIMT_SEG3) {
        const int t = min(tid, SEG_WORDS - 1);
        const int r = t / 99, ky = t % 99 / 33, wd = t % 33;
        const int s = n0 + 64 * r, y = s / ow, x = s - y * ow;
        b_p     = in_n + (y + ky) * iw + x + 2 * wd;
        seg_dst = ky * 3 * BS + 4 + 72 * r + 2 * wd;
    }

    // The loads of a tile go to registers unconditionally (the threads past A_VEC load a piece again and store it past
    // the tile): a load whose use is conditional is sunk by the compiler to that use, after the math.
    int4     ra[A_PER];
    half     rb[4];
    uint2    rv;
    uint32_t rs;
    // a_t: the thread's piece of the tile's weights; b_t: SEG3 the thread's input word, ROWS the warp's k row's offset
    // from b_p
    auto     load_tile = [&](const int k0, const int4 * a_t, const half * b_t_seg, const int b_t) {
#pragma unroll
        for (int r = 0; r < A_PER; ++r) {
            ra[r] = a_t[0];
            conv2d_simt_issue(ra[r].x);
            conv2d_simt_issue(ra[r].y);
            conv2d_simt_issue(ra[r].z);
            conv2d_simt_issue(ra[r].w);
        }
        if (MODE == CONV2D_SIMT_GATHER) {
            const int    ki = k0 + warp;
            const int    ci = ki / (kw * kh), kr = ki - ci * (kw * kh), ky = kr / kw, kx = kr - ky * kw;
            const bool   ok   = ki < k_total;
            const half * src  = in_n + (ok ? ci * ihw + ky * dy * iw + kx * dx : 0);
            const half   zero = __float2half(0.0f);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const half v = src[pos_s[lane + 32 * j]];
                conv2d_simt_issue(__half_as_ushort(v));
                rb[j] = ok ? v : zero;
            }
        } else if (MODE == CONV2D_SIMT_ROWS) {
            const bool  ok = k0 + warp < k_total;
            const uint2 v  = *(const uint2 *) (b_p + (ok ? b_t : 0));
            conv2d_simt_issue(v.x);
            conv2d_simt_issue(v.y);
            rv = ok ? v : make_uint2(0, 0);
        } else {
            rs = *(const uint32_t *) b_t_seg;
            conv2d_simt_issue(rs);
        }
    };
    auto store_tile = [&](const int buf) {
#pragma unroll
        for (int r = 0; r < A_PER; ++r) {
            const half2 * h = (const half2 *) &ra[r];
            float4 *      d = (float4 *) &a_s[buf][8 * (tid + nthreads * r)];
            d[0]            = make_float4(__low2float(h[0]), __high2float(h[0]), __low2float(h[1]), __high2float(h[1]));
            d[1]            = make_float4(__low2float(h[2]), __high2float(h[2]), __low2float(h[3]), __high2float(h[3]));
        }
        if (MODE == CONV2D_SIMT_GATHER) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_s[buf][warp * BS + 4 + 8 * (j / 2) + lane + 32 * j] = __half2float(rb[j]);
            }
        } else if (MODE == CONV2D_SIMT_ROWS) {
            const half2 lo = *(const half2 *) &rv.x, hi = *(const half2 *) &rv.y;
            *(float4 *) &b_s[buf][warp * BS + 4 + 8 * (lane / 16) + 4 * lane] =
                make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
        } else {
            const half2 h  = *(const half2 *) &rs;
            const float e0 = __low2float(h), e1 = __high2float(h);
            float *     d  = &b_s[buf][seg_dst];
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                d[kx * (BS - 1)]     = e0;
                d[kx * (BS - 1) + 1] = e1;
            }
        }
    };

    // The math: warp w covers thread rows 4 (w / 2) .. +3 and thread columns 8 (w % 2) .. +7.
    const int ty = warp / 2 * 4 + lane / 8, tx = warp % 2 * 8 + lane % 8;
    float     acc[TM][TN] = {};
    int       cur         = 0;
    if (MODE == CONV2D_SIMT_GATHER) {
        __syncthreads();  // pos_s
    }
    const int    b_step = MODE == CONV2D_SIMT_ROWS ? BK * ihw : 0;
    const int4 * a_t    = (const int4 *) (wt + (int64_t(blockIdx.y) * k_pad + begin) * BM) + min(tid, A_VEC - 1);
    const half * b_t_seg = b_p + begin / 9 * ihw;
    int          b_t     = MODE == CONV2D_SIMT_ROWS ? (begin + warp) * ihw : 0;
    if (begin < kend) {
        load_tile(begin, a_t, b_t_seg, b_t);
        store_tile(0);
    }
    __syncthreads();
    for (int k0 = begin; k0 < kend; k0 += BK) {
        // the next tile (the last tile again at the end)
        const bool more = k0 + BK < kend;
        a_t             = more ? a_t + A_VEC : a_t;
        b_t_seg         = more ? b_t_seg + ihw : b_t_seg;
        b_t             = more ? b_t + b_step : b_t;
        load_tile(more ? k0 + BK : k0, a_t, b_t_seg, b_t);
        const float * a_k = &a_s[cur][8 * ty];
        const float * b_k = &b_s[cur][4 + 4 * tx];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const float4 v = *(const float4 *) &a_k[k * BM + 4 * g];
                a[4 * g] = v.x, a[4 * g + 1] = v.y, a[4 * g + 2] = v.z, a[4 * g + 3] = v.w;
            }
            if (TM == 9) {
                a[TM - 1] = a_s[cur][k * BM + 128 + ty];
            }
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const float4 v = *(const float4 *) &b_k[k * BS + g * (BS / 2)];
                b[4 * g] = v.x, b[4 * g + 1] = v.y, b[4 * g + 2] = v.z, b[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        store_tile(cur ^ 1);
        __syncthreads();
        cur ^= 1;
    }

    if (end > k_total) {
        // The reference's 0 * 0 products past k_total: -0 -> +0, every other value unchanged.
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] = __int_as_float(__float_as_int(acc[i][j]) == int(0x80000000) ? 0 : __float_as_int(acc[i][j]));
            }
        }
    }
    float *    out_z = output + int64_t(blockIdx.z) * oc * npos;
    const bool o_vec = npos % 4 == 0 && uintptr_t(output) % 16 == 0;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int co = m0 + (i < 8 ? 8 * ty + i : 128 + ty);
        if (co >= oc) {
            continue;
        }
        float * o_row = out_z + int64_t(co) * npos;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int p = n0 + g * (BN / 2) + tx * 4;
            if (o_vec) {
                if (p < npos) {
                    *(float4 *) &o_row[p] =
                        make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2], acc[i][4 * g + 3]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (p + j < npos) {
                        o_row[p + j] = acc[i][4 * g + j];
                    }
                }
            }
        }
    }
}

static __global__ void conv2d_reduce_split_k(const float * __restrict__ partial,
                                             float * __restrict__ output,
                                             const int total,
                                             const int per_batch,
                                             const int split_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int     n   = i / per_batch;
    const float * src = partial + int64_t(n) * (split_k - 1) * per_batch + i;
    float         sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += src[int64_t(k) * per_batch];
    }
    output[i] = sum;
}

template <bool use_mma>
static void conv2d_launch_implicit_gemm(const half *        input,
                                        const half *        weight,
                                        float *             output,
                                        const conv_params & params,
                                        int                 split_k,
                                        dim3                grid,
                                        dim3                block,
                                        cudaStream_t        stream) {
    if (params.KW == 3 && params.KH == 3) {
        conv2d_implicit_gemm_f16<3, 3, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else if (params.KW == 1 && params.KH == 1) {
        conv2d_implicit_gemm_f16<1, 1, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    } else {
        conv2d_implicit_gemm_f16<0, 0, use_mma><<<grid, block, 0, stream>>>(input, weight, output, params, split_k);
    }
}

// Few output channels (the Wan VAE's last conv: 4), 3x3, stride 1, dilation 1, split_k == 1: a thread computes OCB
// channels x PT consecutive positions of one map row. Per input channel and kernel row it converts the PT + 2 input
// values once and runs the 3 kx shifts over them; the weights (F32, [k][OCB]) sit in shared memory. Every output is the
// same FFMA chain over k = (ci, ky, kx) in order as in the 64x64 kernel, and its -0 -> +0 at the end (see above).
// EVEN: the rows start at even halves (even padded width and map size), loaded as 32-bit words.
template <int OCB, int PT, bool EVEN>
static __global__ void __launch_bounds__(128) conv2d_simt_narrow3x3(const half * __restrict__ input,
                                                                    const half * __restrict__ weight,
                                                                    float * __restrict__ output,
                                                                    const conv_params P) {
    extern __shared__ float w_s[];  // [k_total][OCB]
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int ic = int(P.IC), oc = int(P.OC);
    const int k_total = ic * 9, npos = ow * oh, ihw = ih * iw;
    for (int i = threadIdx.x; i < k_total * OCB; i += blockDim.x) {
        const int k = i / OCB, o = i - k * OCB;
        w_s[i]      = o < oc ? __half2float(weight[int64_t(o) * k_total + k]) : 0.0f;
    }
    __syncthreads();
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g * PT >= npos) {
        return;
    }
    const int    n = blockIdx.z, p0 = g * PT, y = p0 / ow, x = p0 - y * ow;
    const half * src = input + int64_t(n) * ic * ihw + y * iw + x;
    float        acc[OCB][PT] = {};
    for (int ci = 0; ci < ic; ++ci) {
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            const half * r = src + ci * ihw + ky * iw;
            float        v[PT + 2];
            if (EVEN) {
#pragma unroll
                for (int e = 0; e < PT + 2; e += 2) {
                    const half2 h = *(const half2 *) (r + e);
                    v[e]          = __low2float(h);
                    v[e + 1]      = __high2float(h);
                }
            } else {
#pragma unroll
                for (int e = 0; e < PT + 2; ++e) {
                    v[e] = __half2float(r[e]);
                }
            }
            const float * wk = &w_s[(ci * 9 + ky * 3) * OCB];
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                float wv[OCB];
#pragma unroll
                for (int o = 0; o < OCB; o += 4) {
                    const float4 t = *(const float4 *) &wk[kx * OCB + o];
                    wv[o] = t.x, wv[o + 1] = t.y, wv[o + 2] = t.z, wv[o + 3] = t.w;
                }
#pragma unroll
                for (int o = 0; o < OCB; ++o) {
#pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        acc[o][j] += wv[o] * v[j + kx];
                    }
                }
            }
        }
    }
    const bool tail = k_total % 64 != 0;  // the reference's last 64-wide tile ends past k_total
#pragma unroll
    for (int o = 0; o < OCB; ++o) {
        if (o >= oc) {
            continue;
        }
        float * out = output + (int64_t(n) * oc + o) * npos + p0;
#pragma unroll
        for (int j = 0; j < PT; j += 4) {
            float4 t;
            t.x = acc[o][j], t.y = acc[o][j + 1], t.z = acc[o][j + 2], t.w = acc[o][j + 3];
            if (tail) {
                t.x = __int_as_float(__float_as_int(t.x) == int(0x80000000) ? 0 : __float_as_int(t.x));
                t.y = __int_as_float(__float_as_int(t.y) == int(0x80000000) ? 0 : __float_as_int(t.y));
                t.z = __int_as_float(__float_as_int(t.z) == int(0x80000000) ? 0 : __float_as_int(t.z));
                t.w = __int_as_float(__float_as_int(t.w) == int(0x80000000) ? 0 : __float_as_int(t.w));
            }
            *(float4 *) &out[j] = t;
        }
    }
}

struct conv2d_simt_plan {
    int     tm;        // channels a thread: 9 (144 a block) or 8 (128)
    int     mode;      // conv2d_simt_mode
    int     k_pad;     // k_total rounded up to the mode's k tile
    int64_t wt_elems;  // the transposed weights (conv2d_simt_weights), halves
};

// P: the padded parameters (padding 0, IW/IH the padded input).
static conv2d_simt_plan conv2d_simt_f32_plan(const conv_params & P,
                                             const int           split_k,
                                             const void *        input,
                                             const void *        output) {
    conv2d_simt_plan plan;
    // 144 channels a block (the Wan VAE's channel counts are multiples of 144), 128 when that wastes less.
    const int64_t    pad144 = (P.OC + 143) / 144 * 144 - P.OC, pad128 = (P.OC + 127) / 128 * 128 - P.OC;
    plan.tm                 = pad128 * 144 < pad144 * 128 ? 8 : 9;
    const int64_t npos      = P.OW * P.OH;
    const bool    s1        = P.ST_X == 1 && P.ST_Y == 1;
    const bool    s3        = P.KW == 3 && P.KH == 3 && s1 && P.DL_X == 1 && P.DL_Y == 1 && split_k == 1;
    if (s3 && P.OC <= 8 && P.OW % 8 == 0 && P.IC * 9 * 8 * 4 <= 48 * 1024 && uintptr_t(output) % 16 == 0) {
        plan.mode = CONV2D_SIMT_NARROW;
    } else if (s3 && P.OW % 64 == 0 && npos % 128 == 0 && P.IW % 2 == 0 && uintptr_t(input) % 4 == 0) {
        plan.mode = CONV2D_SIMT_SEG3;
    } else if (P.KW == 1 && P.KH == 1 && s1 && P.IW == P.OW && P.IH == P.OH && npos % 128 == 0 &&
               uintptr_t(input) % 8 == 0) {
        plan.mode = CONV2D_SIMT_ROWS;
    } else if (P.OC <= 16) {
        plan.mode = CONV2D_SIMT_64;
    } else {
        plan.mode = CONV2D_SIMT_GATHER;
    }
    const int bk      = plan.mode == CONV2D_SIMT_SEG3 ? 9 : 8;
    const int k_total = int(P.IC * P.KW * P.KH);
    plan.k_pad        = (k_total + bk - 1) / bk * bk;
    const int bm      = 16 * plan.tm;
    plan.wt_elems     = plan.mode == CONV2D_SIMT_NARROW || plan.mode == CONV2D_SIMT_64 ? 0 :
                                                                                 (P.OC + bm - 1) / bm * plan.k_pad * bm;
    return plan;
}

template <int TM>
static void conv2d_simt_f32_tm(const half *             input,
                                const half *             weight,
                                half *                   wt,
                                float *                  output,
                                const conv_params &      P,
                                const int                split_k,
                                const conv2d_simt_plan & plan,
                                cudaStream_t             st) {
    using T = conv2d_simt_tile<TM>;
    conv2d_simt_weights<TM><<<dim3(unsigned((plan.k_pad + 31) / 32), unsigned((P.OC + T::BM - 1) / T::BM)), 256, 0, st>>>(
        weight, wt, int(P.OC), int(P.IC * P.KW * P.KH), plan.k_pad);
    const dim3 grid(unsigned((P.OW * P.OH + T::BN - 1) / T::BN), unsigned((P.OC + T::BM - 1) / T::BM),
                    unsigned(P.B * split_k));
    if (plan.mode == CONV2D_SIMT_SEG3) {
        conv2d_implicit_gemm_simt_f32<3, 3, TM, CONV2D_SIMT_SEG3>
            <<<grid, 256, 0, st>>>(input, wt, output, P, split_k, plan.k_pad);
    } else if (plan.mode == CONV2D_SIMT_ROWS) {
        conv2d_implicit_gemm_simt_f32<1, 1, TM, CONV2D_SIMT_ROWS>
            <<<grid, 256, 0, st>>>(input, wt, output, P, split_k, plan.k_pad);
    } else if (P.KW == 3 && P.KH == 3) {
        conv2d_implicit_gemm_simt_f32<3, 3, TM, CONV2D_SIMT_GATHER>
            <<<grid, 256, 0, st>>>(input, wt, output, P, split_k, plan.k_pad);
    } else if (P.KW == 1 && P.KH == 1) {
        conv2d_implicit_gemm_simt_f32<1, 1, TM, CONV2D_SIMT_GATHER>
            <<<grid, 256, 0, st>>>(input, wt, output, P, split_k, plan.k_pad);
    } else {
        conv2d_implicit_gemm_simt_f32<0, 0, TM, CONV2D_SIMT_GATHER>
            <<<grid, 256, 0, st>>>(input, wt, output, P, split_k, plan.k_pad);
    }
}

// wt: plan.wt_elems halves of scratch.
static void conv2d_simt_f32(const half *             input,
                            const half *             weight,
                            half *                   wt,
                            float *                  output,
                            const conv_params &      P,
                            const int                split_k,
                            const conv2d_simt_plan & plan,
                            cudaStream_t             st) {
    if (plan.mode == CONV2D_SIMT_64) {
        const dim3 grid(unsigned((P.OW * P.OH + 63) / 64), unsigned((P.OC + 63) / 64), unsigned(P.B * split_k));
        conv2d_launch_implicit_gemm<false>(input, weight, output, P, split_k, grid, dim3(32, 4), st);
        return;
    }
    if (plan.mode == CONV2D_SIMT_NARROW) {
        const int64_t npos = P.OW * P.OH;
        const bool    even = P.IW % 2 == 0 && P.IH * P.IW % 2 == 0 && uintptr_t(input) % 4 == 0;
        const size_t  smem = size_t(P.IC * 9) * (P.OC <= 4 ? 4 : 8) * sizeof(float);
        if (P.OC <= 4 && P.OW % 16 == 0) {
            const dim3 grid(unsigned((npos / 16 + 127) / 128), 1, unsigned(P.B));
            if (even) {
                conv2d_simt_narrow3x3<4, 16, true><<<grid, 128, smem, st>>>(input, weight, output, P);
            } else {
                conv2d_simt_narrow3x3<4, 16, false><<<grid, 128, smem, st>>>(input, weight, output, P);
            }
        } else {
            const dim3 grid(unsigned((npos / 8 + 127) / 128), 1, unsigned(P.B));
            if (P.OC <= 4) {
                if (even) {
                    conv2d_simt_narrow3x3<4, 8, true><<<grid, 128, smem, st>>>(input, weight, output, P);
                } else {
                    conv2d_simt_narrow3x3<4, 8, false><<<grid, 128, smem, st>>>(input, weight, output, P);
                }
            } else if (even) {
                conv2d_simt_narrow3x3<8, 8, true><<<grid, 128, smem, st>>>(input, weight, output, P);
            } else {
                conv2d_simt_narrow3x3<8, 8, false><<<grid, 128, smem, st>>>(input, weight, output, P);
            }
        }
        return;
    }
    if (plan.tm == 8) {
        conv2d_simt_f32_tm<8>(input, weight, wt, output, P, split_k, plan, st);
    } else {
        conv2d_simt_f32_tm<9>(input, weight, wt, output, P, split_k, plan, st);
    }
}

// Spellcast3D: int8 convolutions for the GTX 10xx (cc 6.1 / 6.2), whose dp4a makes 4 int8 products a lane per
// instruction where an FFMA makes one. The tiling of conv2d_implicit_gemm_simt_f32 (144 or 128 output channels x 128
// positions a block, 9 or 8 channels x 8 positions a thread, the SEG3 and ROWS input staging), on int32 words of 4 int8
// values instead of floats: k runs over (input channel group of 4, ky, kx), a word holds the 4 channels of a group, so a
// word of the input is one 32-bit value at one position of the map and the products of 4 channels are one dp4a.
// The input is quantized into [B][IC/4][PH][PW] words (the padding included, as 0), the weights into the kernel's order
// [M tile][k4][BM channels], both once per call. Number formats (one formula, alpha a knob):
//   activations: amax_c = max |x| over input channel c (the batch and the map), m_c = amax_c^alpha, S = max_c (amax_c /
//     m_c) / 127, s_c = m_c * S (0 when amax_c is 0), x_q = clamp(rint(x / s_c), -127, 127). alpha 0: one scale for
//     the tensor; 1: one a channel; in between, SmoothQuant's migration of the channels' ranges into the weights.
//   weights: w' = w * s_c (the activations' scales folded into the weights, so the int32 sum needs one scale per
//     output channel), d_o = max |w'[o]| / 127, w_q = clamp(rint(w' / d_o), -127, 127).
//   y[o][p] = d_o * sum w_q x_q, the sum in int32, exact: |sum| <= 127^2 K < 2^31 for K < CONV2D_I8_KMAX.
// The engine uses alpha 0.5 (the Wan VAE decoder, a 1024x1024 picture against the F32 path: 50.8 dB; alpha 0 42.7 dB,
// 0.3 48.0, 0.4 49.5, 0.6 47.7, 0.7 49.1, 1 39.0) and only for the convolutions conv2d_i8_input_ok admits.
// The order of the int32 sum does not matter (it is exact), so any split of k would give the same bits; split_k > 1 is
// left to the F32 path all the same (the decoder's shapes never split).
static constexpr int CONV2D_I8_KMAX  = 133000;
static constexpr int CONV2D_I8_CHUNK = 16384;  // map values a block of the maxima pass reduces
static constexpr int CONV2D_I8_KC    = 72;     // k values (whole channel groups) a block of the weight pass quantizes

static __device__ __forceinline__ float conv2d_i8_block_max(float m, float * red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off, 32));
    }
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = m;
    }
    __syncthreads();
    m = 0.0f;
    for (int i = 0; i < int(blockDim.x / 32); ++i) {
        m = fmaxf(m, red[i]);
    }
    return m;
}

// Pass 1: the maxima of |x| per channel and chunk of the map: part[(c * B + n) * nchunk + chunk]. Grid (nchunk, IC, B).
static __global__ void __launch_bounds__(256) conv2d_i8_amax(const float * __restrict__ x,
                                                             float * __restrict__ part,
                                                             const int ic,
                                                             const int hw) {
    __shared__ float red[8];
    const int        c = blockIdx.y, n = blockIdx.z;
    const float *    p     = x + (int64_t(n) * ic + c) * hw;
    const int        begin = blockIdx.x * CONV2D_I8_CHUNK, end = min(hw, begin + CONV2D_I8_CHUNK);
    float            m     = 0.0f;
    if (hw % 4 == 0 && uintptr_t(x) % 16 == 0) {
        for (int i = begin + 4 * int(threadIdx.x); i < end; i += 4 * blockDim.x) {
            const float4 v = *(const float4 *) (p + i);
            m              = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
            m = fmaxf(m, fabsf(p[i]));
        }
    }
    m = conv2d_i8_block_max(m, red);
    if (threadIdx.x == 0) {
        part[(int64_t(c) * gridDim.z + n) * gridDim.x + blockIdx.x] = m;
    }
}

// Pass 2 (one block): s_c and 1 / s_c from the maxima (see above); nparts = B * nchunk values per channel.
static __global__ void __launch_bounds__(1024) conv2d_i8_scales(const float * __restrict__ part,
                                                                float * __restrict__ s,
                                                                float * __restrict__ inv_s,
                                                                const int ic,
                                                                const int nparts,
                                                                const float alpha) {
    __shared__ float red[32];
    float            r = 0.0f;
    for (int c = threadIdx.x; c < ic; c += blockDim.x) {
        float a = 0.0f;
        for (int i = 0; i < nparts; ++i) {
            a = fmaxf(a, part[int64_t(c) * nparts + i]);
        }
        s[c] = a;  // amax_c for now
        if (a > 0.0f) {
            r = fmaxf(r, alpha == 0.0f ? a : alpha == 1.0f ? 1.0f : a / powf(a, alpha));
        }
    }
    const float S = conv2d_i8_block_max(r, red) / 127.0f;
    for (int c = threadIdx.x; c < ic; c += blockDim.x) {
        const float a  = s[c];
        const float sc = a > 0.0f ? (alpha == 0.0f ? 1.0f : alpha == 1.0f ? a : powf(a, alpha)) * S : 0.0f;
        s[c]           = sc;
        inv_s[c]       = sc > 0.0f ? 1.0f / sc : 0.0f;
    }
}

// Pass 3: d_o = max |w[o] * s_c| / 127 and 1 / d_o, a warp per output channel.
static __global__ void __launch_bounds__(256) conv2d_i8_wscale(const half * __restrict__ w,
                                                               const float * __restrict__ s,
                                                               float * __restrict__ dw,
                                                               float * __restrict__ inv_dw,
                                                               const int oc,
                                                               const int ic,
                                                               const int khw) {
    const int o = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (o >= oc) {
        return;
    }
    const half * row = w + int64_t(o) * ic * khw;
    float        m   = 0.0f;
    for (int k = lane; k < ic * khw; k += 32) {
        m = fmaxf(m, fabsf(__half2float(row[k]) * s[k / khw]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off, 32));
    }
    if (lane == 0) {
        dw[o]     = m / 127.0f;
        inv_dw[o] = m > 0.0f ? 127.0f / m : 0.0f;
    }
}

static __device__ __forceinline__ int conv2d_i8_q(const float v) {
    return max(-127, min(127, __float2int_rn(v)));
}

// Pass 4: the weights quantized into the kernel's words, wt[M tile][k4 < k4_pad][BM channels]; word (o, k4 = g * khw +
// kk) holds w_q[o][4g + j][kk] in byte j; 0 outside OC x k4_total. A block quantizes BM channels x CONV2D_I8_KC k values
// (18 words a channel) into shared memory, then writes their words (consecutive threads: consecutive channels).
// Grid: (k4_pad / 18 rounded up, M tiles). khw divides 18 (1x1 and 3x3).
template <int TM>
static __global__ void __launch_bounds__(256) conv2d_i8_weights(const half * __restrict__ w,
                                                                const float * __restrict__ s,
                                                                const float * __restrict__ inv_dw,
                                                                int * __restrict__ wt,
                                                                const int oc,
                                                                const int ic,
                                                                const int khw,
                                                                const int k4_pad) {
    static constexpr int BM = conv2d_simt_tile<TM>::BM, KC = CONV2D_I8_KC, KW4 = KC / 4;
    __shared__ int8_t    q[BM][KC + 4];
    const int            mt = blockIdx.y, kc0 = blockIdx.x * KC, k_total = ic * khw;
    for (int e = threadIdx.x; e < BM * KC; e += blockDim.x) {
        const int c = e / KC, kl = e - c * KC, o = mt * BM + c, k = kc0 + kl;
        q[c][kl]    = o < oc && k < k_total ? int8_t(conv2d_i8_q(__half2float(w[int64_t(o) * k_total + k]) * s[k / khw] *
                                                                 inv_dw[o])) :
                                              int8_t(0);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < BM * KW4; e += blockDim.x) {
        const int k4l = e / BM, c = e - k4l * BM, k4 = blockIdx.x * KW4 + k4l;
        if (k4 >= k4_pad) {
            break;
        }
        const int g = k4l / khw, kk = k4l - g * khw;
        int       word = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            word |= (int(q[c][(4 * g + j) * khw + kk]) & 0xFF) << (8 * j);
        }
        wt[(int64_t(mt) * k4_pad + k4) * BM + c] = word;
    }
}

// Pass 5: the input quantized and padded, xq[n][g][y][x] (PH x PW, IC / 4 groups) = x_q of channels 4g .. 4g + 3 at
// (y - py, x - px) in bytes 0..3, 0 in the padding.
static __global__ void __launch_bounds__(256) conv2d_i8_input(const float * __restrict__ x,
                                                              const float * __restrict__ inv_s,
                                                              int * __restrict__ xq,
                                                              const int iw,
                                                              const int ih,
                                                              const int pw,
                                                              const int ph,
                                                              const int px,
                                                              const int py,
                                                              const int ic,
                                                              const int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int xx = i % pw - px, r = i / pw, yy = r % ph - py, ng = r / ph, g = ng % (ic / 4), n = ng / (ic / 4);
    int       word = 0;
    if ((unsigned) xx < (unsigned) iw && (unsigned) yy < (unsigned) ih) {
        const int64_t plane = int64_t(ih) * iw;
        const float * src   = x + (int64_t(n) * ic + 4 * g) * plane + int64_t(yy) * iw + xx;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            word |= (conv2d_i8_q(src[j * plane] * inv_s[4 * g + j]) & 0xFF) << (8 * j);
        }
    }
    xq[i] = word;
}

// The convolution: conv2d_implicit_gemm_simt_f32's SEG3 / ROWS kernels on words (k tiles of 9 / 8 words), dp4a in place
// of FFMA, y = d_o * acc in the epilogue. RAW (test programs): the int32 sums are written instead, as the float's bits.
template <int TM, int MODE, bool RAW>
static __global__ void __launch_bounds__(256, 2) conv2d_implicit_gemm_i8(const int * __restrict__ input,
                                                                         const int * __restrict__ wt,
                                                                         const float * __restrict__ dw,
                                                                         float * __restrict__ output,
                                                                         const conv_params P,
                                                                         const int         k4_pad) {
    using T                         = conv2d_simt_tile<TM>;
    static constexpr int nthreads   = 256, nwarps = 8;
    static constexpr int BM = T::BM, BN = T::BN, TN = T::TN;
    static constexpr int BK          = MODE == CONV2D_SIMT_SEG3 ? 9 : 8;  // words of k a tile
    static constexpr int A_VEC       = BK * BM / 4;                        // 16-byte pieces of a weight tile
    static constexpr int A_PER       = (A_VEC + nthreads - 1) / nthreads;
    static constexpr int A_S         = A_PER * 4 * nthreads;  // >= BK * BM: every thread stores its pieces (no branch)
    static constexpr int SEG_THREADS = 2 * 3 * 33;            // SEG3: 2 runs x 3 input rows x 33 pairs of positions
    static constexpr int BS          = 2 * (64 + 8);          // a k row: position p at 4 + p + 8 (p / 64)
    static_assert(MODE == CONV2D_SIMT_SEG3 || MODE == CONV2D_SIMT_ROWS, "SEG3 or ROWS");
    static_assert(BN == 128 && (MODE == CONV2D_SIMT_SEG3 || BK == nwarps) && BK * BM % 4 == 0, "tile layout");
    __shared__ __align__(16) int a_s[2][A_S];
    __shared__ __align__(16) int b_s[2][BK * BS];

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int iw = int(P.IW), ih = int(P.IH), ow = int(P.OW), oh = int(P.OH);
    const int oc = int(P.OC), ic4 = int(P.IC) / 4;
    const int n = blockIdx.z, m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int npos = ow * oh, ihw = ih * iw;
    const int k4_total = MODE == CONV2D_SIMT_SEG3 ? 9 * ic4 : ic4;

    const int * in_n = input + int64_t(n) * ic4 * ihw;
    // ROWS: the warp's k row is one channel group's map, its lane loads positions n0 + 4 lane .. +3. SEG3: thread t <
    // SEG_THREADS loads the positions 2 wd, 2 wd + 1 of input row ky of run r (the map row of positions n0 + 64 r ..
    // +63 and the two below it) and stores each at its 3 kx shifts, as the F32 kernel does with its F16 pairs; the
    // threads past SEG_THREADS repeat the last pair's loads and stores (same values, same places).
    const int * b_p     = MODE == CONV2D_SIMT_ROWS ? in_n + n0 + 4 * lane : in_n;
    int         seg_dst = 0;
    if (MODE == CONV2D_SIMT_SEG3) {
        const int t = min(tid, SEG_THREADS - 1);
        const int r = t / 99, ky = t % 99 / 33, wd = t % 33;
        const int s = n0 + 64 * r, y = s / ow, x = s - y * ow;
        b_p         = in_n + (y + ky) * iw + x + 2 * wd;
        seg_dst     = ky * 3 * BS + 4 + 72 * r + 2 * wd;
    }

    int4  ra[A_PER];
    int4  rv;
    uint2 rs;
    int   a_off[A_PER];
#pragma unroll
    for (int r = 0; r < A_PER; ++r) {
        a_off[r] = min(tid + nthreads * r, A_VEC - 1);
    }
    auto load_tile = [&](const int k0, const int4 * a_t, const int b_t) {
#pragma unroll
        for (int r = 0; r < A_PER; ++r) {
            ra[r] = a_t[a_off[r]];
            conv2d_simt_issue(ra[r].x);
            conv2d_simt_issue(ra[r].y);
            conv2d_simt_issue(ra[r].z);
            conv2d_simt_issue(ra[r].w);
        }
        if (MODE == CONV2D_SIMT_ROWS) {
            const bool ok = k0 + warp < k4_total;
            const int4 v  = *(const int4 *) (b_p + (ok ? b_t : 0));
            conv2d_simt_issue(v.x);
            conv2d_simt_issue(v.y);
            conv2d_simt_issue(v.z);
            conv2d_simt_issue(v.w);
            rv = ok ? v : make_int4(0, 0, 0, 0);
        } else {
            rs = *(const uint2 *) (b_p + b_t);
            conv2d_simt_issue(rs.x);
            conv2d_simt_issue(rs.y);
        }
    };
    auto store_tile = [&](const int buf) {
#pragma unroll
        for (int r = 0; r < A_PER; ++r) {
            *(int4 *) &a_s[buf][4 * (tid + nthreads * r)] = ra[r];
        }
        if (MODE == CONV2D_SIMT_ROWS) {
            *(int4 *) &b_s[buf][warp * BS + 4 + 8 * (lane / 16) + 4 * lane] = rv;
        } else {
            int * d = &b_s[buf][seg_dst];
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                d[kx * (BS - 1)]     = int(rs.x);
                d[kx * (BS - 1) + 1] = int(rs.y);
            }
        }
    };

    // The math: warp w covers thread rows 4 (w / 2) .. +3 and thread columns 8 (w % 2) .. +7.
    const int ty = warp / 2 * 4 + lane / 8, tx = warp % 2 * 8 + lane % 8;
    int       acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0;
        }
    }
    int          cur    = 0;
    const int    b_step = MODE == CONV2D_SIMT_ROWS ? BK * ihw : ihw;
    const int4 * a_t    = (const int4 *) (wt + int64_t(blockIdx.y) * k4_pad * BM);
    int          b_t    = MODE == CONV2D_SIMT_ROWS ? warp * ihw : 0;
    load_tile(0, a_t, b_t);
    store_tile(0);
    __syncthreads();
    for (int k0 = 0; k0 < k4_total; k0 += BK) {
        // the next tile (the last tile again at the end)
        const bool more = k0 + BK < k4_total;
        a_t             = more ? a_t + A_VEC : a_t;
        b_t             = more ? b_t + b_step : b_t;
        load_tile(more ? k0 + BK : k0, a_t, b_t);
        const int * a_k = &a_s[cur][8 * ty];
        const int * b_k = &b_s[cur][4 + 4 * tx];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            int a[TM], b[TN];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int4 v = *(const int4 *) &a_k[k * BM + 4 * g];
                a[4 * g] = v.x, a[4 * g + 1] = v.y, a[4 * g + 2] = v.z, a[4 * g + 3] = v.w;
            }
            if (TM == 9) {
                a[TM - 1] = a_s[cur][k * BM + 128 + ty];
            }
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int4 v = *(const int4 *) &b_k[k * BS + g * (BS / 2)];
                b[4 * g] = v.x, b[4 * g + 1] = v.y, b[4 * g + 2] = v.z, b[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = ggml_cuda_dp4a(a[i], b[j], acc[i][j]);
                }
            }
        }
        store_tile(cur ^ 1);
        __syncthreads();
        cur ^= 1;
    }

    float *    out_z = output + int64_t(n) * oc * npos;
    const bool o_vec = npos % 4 == 0 && uintptr_t(output) % 16 == 0;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int co = m0 + (i < 8 ? 8 * ty + i : 128 + ty);
        if (co >= oc) {
            continue;
        }
        const float d     = dw[co];
        float *     o_row = out_z + int64_t(co) * npos;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int p = n0 + g * (BN / 2) + tx * 4;
            float     y[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                y[j] = RAW ? __int_as_float(acc[i][4 * g + j]) : float(acc[i][4 * g + j]) * d;
            }
            if (o_vec) {
                if (p < npos) {
                    *(float4 *) &o_row[p] = make_float4(y[0], y[1], y[2], y[3]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (p + j < npos) {
                        o_row[p + j] = y[j];
                    }
                }
            }
        }
    }
}

struct conv2d_i8_plan {
    int     mode;       // CONV2D_SIMT_SEG3 or CONV2D_SIMT_ROWS; -1: the shape is not for the int8 path
    int     tm;         // channels a thread: 9 (144 a block) or 8 (128)
    int     k4_total;   // words of k: IC / 4 * KW * KH
    int     k4_pad;     // rounded up to the mode's k tile (9 / 8 words)
    int     nchunk;     // map chunks of the maxima pass
    int64_t wt_words;   // the quantized weights
    int64_t in_words;   // the quantized, padded input
    int64_t f32_elems;  // the maxima, scales and inverses
};

// P: the padded parameters (padding 0, IW / IH the padded input), as conv2d_simt_f32_plan; ih_iw: the unpadded map size.
static conv2d_i8_plan conv2d_i8_make_plan(const conv_params & P, const int split_k, const int64_t ih_iw) {
    conv2d_i8_plan plan = {};
    plan.mode           = -1;
    const int64_t npos  = P.OW * P.OH;
    const bool    s1    = P.ST_X == 1 && P.ST_Y == 1;
    if (split_k != 1 || P.IC % 4 != 0 || P.IC * P.KW * P.KH >= CONV2D_I8_KMAX || P.OC <= 8 || npos % 128 != 0) {
        return plan;  // (OC <= 8: the F32 narrow kernel, the decoder's last conv)
    }
    if (P.KW == 3 && P.KH == 3 && s1 && P.DL_X == 1 && P.DL_Y == 1 && P.OW % 64 == 0 && P.IW % 2 == 0) {
        plan.mode = CONV2D_SIMT_SEG3;
    } else if (P.KW == 1 && P.KH == 1 && s1 && P.IW == P.OW && P.IH == P.OH) {
        plan.mode = CONV2D_SIMT_ROWS;
    } else {
        return plan;
    }
    const int64_t pad144 = (P.OC + 143) / 144 * 144 - P.OC, pad128 = (P.OC + 127) / 128 * 128 - P.OC;
    plan.tm              = pad128 * 144 < pad144 * 128 ? 8 : 9;
    plan.k4_total        = int(P.IC / 4 * P.KW * P.KH);
    plan.k4_pad          = plan.mode == CONV2D_SIMT_SEG3 ? plan.k4_total : (plan.k4_total + 7) / 8 * 8;
    plan.nchunk          = int((ih_iw + CONV2D_I8_CHUNK - 1) / CONV2D_I8_CHUNK);
    const int bm         = 16 * plan.tm;
    plan.wt_words        = (P.OC + bm - 1) / bm * plan.k4_pad * bm;
    plan.in_words        = P.B * (P.IC / 4) * P.IW * P.IH;
    plan.f32_elems       = P.IC * P.B * plan.nchunk + 2 * P.IC + 2 * P.OC;
    return plan;
}

// The int8 path: x the unpadded F32 input [B][IC][IH][IW] (P its parameters), w the F16 weights, PP the padded
// parameters. Scratch: f32 plan.f32_elems floats, wt plan.wt_words + plan.in_words ints (the quantized weights, then
// the quantized input at wt + plan.wt_words). Test programs read the quantized operands there:
// the per-channel maxima, then s, 1 / s, d, 1 / d in f32 (IC * B * nchunk, IC, IC, OC, OC).
template <int TM>
static void conv2d_i8_prepare_tm(const float *          x,
                                 const half *           w,
                                 const conv_params &    P,
                                 const conv_params &    PP,
                                 const conv2d_i8_plan & plan,
                                 float *                f32,
                                 int *                  wt,
                                 const float            alpha,
                                 cudaStream_t           st) {
    using T          = conv2d_simt_tile<TM>;
    const int ic     = int(P.IC), oc = int(P.OC), khw = int(P.KW * P.KH), hw = int(P.IW * P.IH);
    const int mtiles = int((P.OC + T::BM - 1) / T::BM);
    float *   part   = f32;
    float *   s      = part + P.IC * P.B * plan.nchunk;
    float *   inv_s  = s + ic;
    float *   dw     = inv_s + ic;
    float *   inv_dw = dw + oc;
    int *     xq     = wt + plan.wt_words;
    conv2d_i8_amax<<<dim3(unsigned(plan.nchunk), unsigned(ic), unsigned(P.B)), 256, 0, st>>>(x, part, ic, hw);
    conv2d_i8_scales<<<1, 1024, 0, st>>>(part, s, inv_s, ic, int(P.B) * plan.nchunk, alpha);
    conv2d_i8_wscale<<<unsigned((oc + 7) / 8), 256, 0, st>>>(w, s, dw, inv_dw, oc, ic, khw);
    conv2d_i8_weights<TM><<<dim3(unsigned((plan.k4_pad + CONV2D_I8_KC / 4 - 1) / (CONV2D_I8_KC / 4)), unsigned(mtiles)),
                            256, 0, st>>>(w, s, inv_dw, wt, oc, ic, khw, plan.k4_pad);
    conv2d_i8_input<<<unsigned((plan.in_words + 255) / 256), 256, 0, st>>>(
        x, inv_s, xq, int(P.IW), int(P.IH), int(PP.IW), int(PP.IH), int(P.PD_X), int(P.PD_Y), ic, int(plan.in_words));
}

template <int TM, bool RAW>
static void conv2d_i8_conv_tm(const int *            xq,
                              const int *            wt,
                              const float *          dw,
                              float *                output,
                              const conv_params &    PP,
                              const conv2d_i8_plan & plan,
                              cudaStream_t           st) {
    using T = conv2d_simt_tile<TM>;
    const dim3 grid(unsigned(PP.OW * PP.OH / T::BN), unsigned((PP.OC + T::BM - 1) / T::BM), unsigned(PP.B));
    if (plan.mode == CONV2D_SIMT_SEG3) {
        conv2d_implicit_gemm_i8<TM, CONV2D_SIMT_SEG3, RAW><<<grid, 256, 0, st>>>(xq, wt, dw, output, PP, plan.k4_pad);
    } else {
        conv2d_implicit_gemm_i8<TM, CONV2D_SIMT_ROWS, RAW><<<grid, 256, 0, st>>>(xq, wt, dw, output, PP, plan.k4_pad);
    }
}

static void conv2d_i8_prepare(const float *          x,
                              const half *           w,
                              const conv_params &    P,
                              const conv_params &    PP,
                              const conv2d_i8_plan & plan,
                              float *                f32,
                              int *                  wt,
                              const float            alpha,
                              cudaStream_t           st) {
    if (plan.tm == 8) {
        conv2d_i8_prepare_tm<8>(x, w, P, PP, plan, f32, wt, alpha, st);
    } else {
        conv2d_i8_prepare_tm<9>(x, w, P, PP, plan, f32, wt, alpha, st);
    }
}

// The convolution of the prepared operands; RAW: the sums.
template <bool RAW = false>
static void conv2d_i8_conv(float *                output,
                           const conv_params &    P,
                           const conv_params &    PP,
                           const conv2d_i8_plan & plan,
                           const float *          f32,
                           const int *            wt,
                           cudaStream_t           st) {
    const float * dw = f32 + P.IC * P.B * plan.nchunk + 2 * P.IC;
    const int *   xq = wt + plan.wt_words;
    if (plan.tm == 8) {
        conv2d_i8_conv_tm<8, RAW>(xq, wt, dw, output, PP, plan, st);
    } else {
        conv2d_i8_conv_tm<9, RAW>(xq, wt, dw, output, PP, plan, st);
    }
}

static void conv2d_i8_run(const float *          x,
                          const half *           w,
                          float *                output,
                          const conv_params &    P,
                          const conv_params &    PP,
                          const conv2d_i8_plan & plan,
                          float *                f32,
                          int *                  wt,
                          const float            alpha,
                          cudaStream_t           st) {
    conv2d_i8_prepare(x, w, P, PP, plan, f32, wt, alpha, st);
    conv2d_i8_conv(output, P, PP, plan, f32, wt, st);
}

// Test programs (ggml/w8a8-tools/test_conv_simt.cu, test_conv_i8.cu) include this file for its kernels alone.
#ifndef GGML_CUDA_CONV2D_KERNELS_ONLY
static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
}

// Spellcast3D: whether the int8 path may take a convolution, by what produced its input: an activation (SiLU), seen
// through views, reshapes, copies and scales. Measured on the Wan VAE decoder (a 1024x1024 Qwen-Image picture against
// the F32 path, int8 activations with a scale per channel): its 34 convolutions fed by norm + SiLU leave the picture as
// it was (one at a time 63.7-67.9 dB, the 128^2 / 256^2 groups 60.5 / 61.4 dB), while those on the residual stream ruin
// it even alone - the shortcuts' 1x1 convolutions (the input an ADD: 512^2 37.0 dB, 1024^2 16.9 dB) and the upsampling
// convolutions (an UPSCALE: 1024^2 23.2 dB): the stream's outliers (max |x| up to 140 times its rms) leave the rest of
// its values a few int8 steps. Those stay F32 (ggml/NOTES-conv2d-simt.md, "int8 convolutions").
static bool conv2d_i8_input_ok(const ggml_tensor * t) {
    for (int i = 0; t != nullptr && i < 8; ++i) {
        switch (t->op) {
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_SCALE:
                t = t->src[0];
                break;
            case GGML_OP_UNARY:
                return ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
            default:
                return false;
        }
    }
    return false;
}

// Spellcast3D: the tensor-core implicit GEMM (conv2d-mma.cuh) for NVIDIA cards with Turing MMA (cc >= 7.5), in place
// of conv2d_implicit_gemm_f16<.., true> and of cuBLAS for 1x1 (RTX 3060, the Qwen-Image 2.1 VAE decode at 1024x1024:
// see ggml/NOTES-conv2d-mma.md). Volta, the GTX 10xx (their F32 and int8 kernels above), AMD and MooreThreads keep
// their paths. GGML_CUDA_CONV2D_MMA=0: the old kernels; 2: the Turing kernel (register-staged, m16n8k8) on every
// card that has it (research).
static int conv2d_mma_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_CONV2D_MMA");
        return e == nullptr ? 1 : atoi(e);
    }();
    return mode;
}

bool ggml_cuda_conv2d_mma_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * conv) {
    if (conv->op != GGML_OP_CONV_2D || conv2d_mma_mode() == 0) {
        return false;
    }
    const ggml_tensor * kernel = conv->src[0];
    const ggml_tensor * input  = conv->src[1];
    const int           cc     = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc) || kernel->type != GGML_TYPE_F16 ||
        input->type != GGML_TYPE_F32 || conv->type != GGML_TYPE_F32 || !ggml_is_contiguous(kernel) ||
        !ggml_is_contiguous(input) || !ggml_is_contiguous(conv) || input->ne[2] != kernel->ne[2]) {
        return false;
    }
    const int32_t * p = (const int32_t *) conv->op_params;
    if (p[6] != 0) {  // cwhn
        return false;
    }
    return conv2d_mma_shape_ok(input->ne[0], input->ne[1], conv->ne[0], conv->ne[1], kernel->ne[0], kernel->ne[1],
                               input->ne[2], kernel->ne[3], input->ne[3], p[0], p[1], p[2], p[3], p[4], p[5]);
}

void ggml_cuda_conv2d_mma_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * conv,
                                const ggml_cuda_conv2d_fusion & f) {
    const ggml_tensor * kernel = conv->src[0];
    const ggml_tensor * input  = conv->src[1];
    const int32_t *     op     = (const int32_t *) conv->op_params;
    const int           cc     = ggml_cuda_info().devices[ctx.device].cc;
    cudaStream_t        st     = ctx.stream();

    const int iw = int(input->ne[0]), ih = int(input->ne[1]), ic = int(input->ne[2]), b = int(input->ne[3]);
    const int kw = int(kernel->ne[0]), kh = int(kernel->ne[1]), oc = int(kernel->ne[3]);
    const int ow = int(conv->ne[0]), oh = int(conv->ne[1]);
    GGML_ASSERT(f.out != nullptr && (f.up == 1 || f.up == 2));

    ggml_cuda_pool_alloc<half> xbuf(ctx.pool());
    const half *               xh = f.xh;
    if (xh == nullptr) {
        GGML_ASSERT(f.x != nullptr && ggml_nelements(f.x) * f.up * f.up == ggml_nelements(input));
        xbuf.alloc(int64_t(b) * ih * iw * ic);
        conv2d_mma_to_nhwc((const float *) f.x->data, xbuf.get(), ic, ih, iw, b, f.up, f.in_affine, f.in_scale,
                           f.in_sbias, st);
        xh = xbuf.get();
    }

    const half *               w = (const half *) kernel->data;
    ggml_cuda_pool_alloc<half> wbuf(ctx.pool());
    if (kw * kh > 1 || uintptr_t(w) % 16 != 0) {
        wbuf.alloc(int64_t(oc) * kw * kh * ic);
        conv2d_mma_reorder_weights(w, wbuf.get(), oc, ic, kw * kh, st);
        w = wbuf.get();
    }

    conv2d_mma_params p = {};
    p.iw = iw, p.ih = ih, p.ic = ic, p.ow = ow, p.oh = oh, p.oc = oc, p.kw = kw, p.kh = kh;
    p.sx = op[0], p.sy = op[1], p.px = op[2], p.py = op[3], p.dx = op[4], p.dy = op[5];
    p.k = kw * kh * ic, p.ohw = ow * oh, p.npos = b * ow * oh;
    p.affine = f.out_affine, p.scale = f.out_scale, p.sbias = f.out_sbias;
    p.bias   = f.bias ? (const float *) f.bias->data : nullptr;
    p.z      = f.z ? (const float *) f.z->data : nullptr;
    // GGML_CUDA_CONV2D_MMA_ACC=16: F16 accumulation (research: twice the tensor rate of F32 accumulation on GeForce cards)
    static const bool acc16 = [] {
        const char * e = getenv("GGML_CUDA_CONV2D_MMA_ACC");
        return e != nullptr && atoi(e) == 16;
    }();
    conv2d_mma_launch(xh, w, (float *) f.out->data, p, cp_async_available(cc) && conv2d_mma_mode() != 2, st, acc16);
}

void ggml_cuda_op_conv2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    cudaStream_t st = ctx.stream();

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int64_t IW = input->ne[0];   // input_w
    const int64_t IH = input->ne[1];   // input_h
    const int64_t OW = dst->ne[0];     // output_w
    const int64_t OH = dst->ne[1];     // output_h
    const int64_t KW = kernel->ne[0];  // kernel_w
    const int64_t KH = kernel->ne[1];  // kernel_h
    const int64_t IC = input->ne[2];   // input_channels
    const int64_t OC = kernel->ne[3];  // ouptut_chanles
    const int64_t B  = input->ne[3];   // n_batches

    const int64_t total  = B * OC * OH * OW;
    conv_params   params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (ggml_cuda_conv2d_mma_supported(ctx, dst)) {
        ggml_cuda_conv2d_fusion f;
        f.x   = input;
        f.out = dst;
        ggml_cuda_conv2d_mma_fused(ctx, dst, f);
        return;
    }
    const bool   use_mma =
        turing_mma_available(device.cc) || amd_wmma_available(device.cc) || amd_mfma_available(device.cc);
    // MUSA can share the tiling without a native fragment implementation in mma.cuh. Spellcast3D: so can the NVIDIA
    // cards before Volta (the GTX 10xx): their only other kernel is conv2d_kernel, one thread per output looping over
    // every input (a GTX 1070 Ti decoded a 1024x1024 Qwen-Image picture in 139 s with it, 7.1 s through the tiled
    // kernel's F32 SIMT path); GGML_CUDA_CONV2D_SIMT=0 restores it. Volta (no Turing MMA either) is left as it was:
    // never measured, and the tiled kernel rounds its input to F16 first.
    static const bool simt_nv = [] {
        const char * e = getenv("GGML_CUDA_CONV2D_SIMT");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool use_simt   = GGML_CUDA_CC_IS_MTHREADS(device.cc) ||
                            (simt_nv && GGML_CUDA_CC_IS_NVIDIA(device.cc) && !use_mma && device.cc < GGML_CUDA_CC_VOLTA);
    const bool pointwise  = KW == 1 && KH == 1 && ST_X == 1 && ST_Y == 1 && PD_X == 0 && PD_Y == 0;
    const bool use_blas   = pointwise && fast_fp16_hardware_available(device.cc);
    // Short reductions on small maps do not amortize conversion and launch costs.
    const bool small_conv = IC * KW * KH < 64 && OW * OH < 512;

    const int64_t limit    = INT_MAX - 256;
    const int64_t padded_w = IW + 2 * int64_t(PD_X), padded_h = IH + 2 * int64_t(PD_Y);
    const bool    padded_fits = padded_w > 0 && padded_w <= limit && padded_h > 0 && padded_h <= limit &&
                             padded_w * padded_h <= limit && IC * B <= limit / (padded_w * padded_h);
    if (kernel->type == GGML_TYPE_F16 && (use_mma || use_blas || use_simt) && (use_blas || !small_conv) &&
        ggml_nelements(input) <= limit && ggml_nelements(kernel) <= limit && total <= limit && padded_fits &&
        PD_X >= 0 && PD_Y >= 0 && ST_X > 0 && ST_Y > 0 && DL_X > 0 && DL_Y > 0 &&
        (OW - 1) * ST_X + (KW - 1) * DL_X < padded_w && (OH - 1) * ST_Y + (KH - 1) * DL_Y < padded_h &&
        (OC + 63) / 64 <= 65535 && B <= 65535) {
        const int pw = int(padded_w), ph = int(padded_h);
        const int padded_total = int(padded_w * padded_h * IC * B);

        // Spellcast3D: GGML_CUDA_CONV2D_PASCAL=0 takes the GTX 10xx back to the SIMT branch of the 64x64 kernel (the
        // same output bit for bit; GTX 1070, the Wan VAE decode of a 1024x1024 picture: its convolutions 6.3 -> 3.4 s).
        static const bool simt_pascal = [] {
            const char * e = getenv("GGML_CUDA_CONV2D_PASCAL");
            return e == nullptr || atoi(e) != 0;
        }();
        const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
        const int     target  = 8 * ggml_cuda_info().devices[ctx.device].nsm;
        // Split long reductions so small spatial maps still occupy the GPU.
        const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                               std::max(int64_t(1), (target + blocks - 1) / blocks) }));
        const conv_params padded_params = { pw, ph, OW, OH, KW, KH, ST_X, ST_Y, 0, 0, DL_X, DL_Y, IC, OC, B, total };

        // Spellcast3D: on the GTX 10xx (cc 6.1 / 6.2, dp4a) the convolutions fed by a SiLU (conv2d_i8_input_ok) whose
        // shape is one of the F32 kernel's SEG3 / ROWS modes with 4n input channels take the int8 path
        // (conv2d_implicit_gemm_i8, the operands quantized at each call); the rest stays F32. GGML_CUDA_CONV2D_INT8=0:
        // every convolution F32 (the pictures of the engine before it, bit for bit); 2: every int8 shape whatever feeds
        // it (research only: it ruins the Wan VAE's pictures). GGML_CUDA_CONV2D_INT8_ALPHA: the activations' per-channel
        // share (default 0.5, see above).
        static const int   int8_mode  = [] {
            const char * e = getenv("GGML_CUDA_CONV2D_INT8");
            return e == nullptr ? 1 : atoi(e);
        }();
        static const float int8_alpha = [] {
            const char * e = getenv("GGML_CUDA_CONV2D_INT8_ALPHA");
            return e != nullptr ? std::min(1.0f, std::max(0.0f, float(atof(e)))) : 0.5f;
        }();
        if (int8_mode != 0 && simt_pascal && !use_mma && !use_blas && GGML_CUDA_CC_IS_NVIDIA(device.cc) &&
            device.cc >= GGML_CUDA_CC_DP4A && device.cc < GGML_CUDA_CC_VOLTA &&
            (int8_mode == 2 || conv2d_i8_input_ok(input))) {
            const conv2d_i8_plan plan = conv2d_i8_make_plan(padded_params, split_k, IW * IH);
            if (plan.mode >= 0) {
                ggml_cuda_pool_alloc<float> f32(ctx.pool(), plan.f32_elems);
                ggml_cuda_pool_alloc<int>   words(ctx.pool(), plan.wt_words + plan.in_words);
                conv2d_i8_run(X_D, (const half *) K_D, Y_D, params, padded_params, plan, f32.get(), words.get(),
                              int8_alpha, st);
                return;
            }
        }

        ggml_cuda_pool_alloc<half> x_half(ctx.pool(), padded_total);
        // Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32.
        if (PD_X == 0 && PD_Y == 0) {
            ggml_get_to_fp16_cuda(input->type)(X_D, x_half.get(), padded_total, st);
        } else {
            conv2d_pad_f16<<<(padded_total + 255) / 256, 256, 0, st>>>(X_D, x_half.get(), int(IW), int(IH), pw, ph,
                                                                       PD_X, PD_Y, padded_total);
        }
        if (use_blas) {
            const float    alpha = 1.0f, beta = 0.0f;
            const int      positions = int(OW * OH);
            cublasHandle_t cublas_h  = ctx.cublas_handle();
            for (int n = 0; n < B; ++n) {
                CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, positions, int(OC), int(IC), &alpha,
                                          x_half.get() + int64_t(n) * IC * positions, CUDA_R_16F, positions, K_D,
                                          CUDA_R_16F, int(IC), &beta, Y_D + int64_t(n) * OC * positions, CUDA_R_32F,
                                          positions, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            return;
        }
        ggml_cuda_pool_alloc<float> partial(ctx.pool());
        float *                     result = split_k == 1 ? Y_D : partial.alloc(total * split_k);
        const dim3                  block(device.warp_size, 4);
        const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        if (use_mma) {
            conv2d_launch_implicit_gemm<true>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                              block, st);
        } else if (simt_pascal && GGML_CUDA_CC_IS_NVIDIA(device.cc)) {
            const conv2d_simt_plan     plan = conv2d_simt_f32_plan(padded_params, split_k, x_half.get(), result);
            ggml_cuda_pool_alloc<half> wt(ctx.pool(), std::max(plan.wt_elems, int64_t(1)));
            conv2d_simt_f32(x_half.get(), (const half *) K_D, wt.get(), result, padded_params, split_k, plan, st);
        } else {
            conv2d_launch_implicit_gemm<false>(x_half.get(), (const half *) K_D, result, padded_params, split_k, grid,
                                               block, st);
        }
        if (split_k > 1) {
            conv2d_reduce_split_k<<<(total + 255) / 256, 256, 0, st>>>(result, Y_D, int(total), int(OC * OW * OH),
                                                                       split_k);
        }
        return;
    }

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_cuda_f16(X_D, (half *) K_D, Y_D, params, st);
    } else {
        conv2d_cuda_f32(X_D, K_D, Y_D, params, st);
    }
}
#endif // GGML_CUDA_CONV2D_KERNELS_ONLY
