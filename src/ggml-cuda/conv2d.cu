#include "conv2d.cuh"
#include "convert.cuh"
#include "mma.cuh"

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

// Test programs (ggml/w8a8-tools/test_conv_simt.cu) include this file for its kernels alone.
#ifndef GGML_CUDA_CONV2D_KERNELS_ONLY
static void conv2d_cuda_f16(const float * X_D, const half * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<half>(X_D, K_D, Y_D, P, st);
}

static void conv2d_cuda_f32(const float * X_D, const float * K_D, float * Y_D, const conv_params P, cudaStream_t st) {
    conv2d_cuda<float>(X_D, K_D, Y_D, P, st);
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

        ggml_cuda_pool_alloc<half> x_half(ctx.pool(), padded_total);
        // Match im2col's F16 input precision, but expand patches only in shared memory and accumulate in F32.
        if (PD_X == 0 && PD_Y == 0) {
            ggml_get_to_fp16_cuda(input->type)(X_D, x_half.get(), padded_total, st);
        } else {
            conv2d_pad_f16<<<(padded_total + 255) / 256, 256, 0, st>>>(X_D, x_half.get(), int(IW), int(IH), pw, ph,
                                                                       PD_X, PD_Y, padded_total);
        }
        const conv_params padded_params = { pw, ph, OW, OH, KW, KH, ST_X, ST_Y, 0, 0, DL_X, DL_Y, IC, OC, B, total };
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
        const int64_t blocks  = ((OW * OH + 63) / 64) * ((OC + 63) / 64) * B;
        const int     target  = 8 * ggml_cuda_info().devices[ctx.device].nsm;
        // Split long reductions so small spatial maps still occupy the GPU.
        const int     split_k = int(std::min({ int64_t(32), int64_t(65535) / B, (IC * KW * KH + 63) / 64,
                                               std::max(int64_t(1), (target + blocks - 1) / blocks) }));

        ggml_cuda_pool_alloc<float> partial(ctx.pool());
        float *                     result = split_k == 1 ? Y_D : partial.alloc(total * split_k);
        const dim3                  block(device.warp_size, 4);
        const dim3 grid(unsigned((OW * OH + 63) / 64), unsigned((OC + 63) / 64), unsigned(B * split_k));
        // Spellcast3D: GGML_CUDA_CONV2D_PASCAL=0 takes the GTX 10xx back to the SIMT branch of the 64x64 kernel (the
        // same output bit for bit; GTX 1070, the Wan VAE decode of a 1024x1024 picture: its convolutions 6.3 -> 3.4 s).
        static const bool simt_pascal = [] {
            const char * e = getenv("GGML_CUDA_CONV2D_PASCAL");
            return e == nullptr || atoi(e) != 0;
        }();
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
