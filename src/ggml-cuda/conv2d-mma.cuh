#pragma once

// Spellcast3D: the F16 tensor-core convolution for NVIDIA cards with Turing MMA (cc >= 7.5: RTX 20xx and newer), in
// place of conv2d_implicit_gemm_f16<.., true> (64x64x64 tiles, one shared stage with two barriers a tile, 16-bit
// gathers of a padded F16 copy: ~8 TFLOPS on an RTX 3060, a third of its F16 x F16 -> F32 tensor rate).
//
// An implicit GEMM: rows = output positions (b, oy, ox), columns = output channels, k = (ky * KW + kx) * IC + ci.
//   - The input is first copied to F16 in channels-last order, x_h[b][iy][ix][ci] (conv2d_mma_nhwc: a tiled transpose,
//     optionally with the graph's input scale and a nearest 2x upsampling folded in), so that 8 consecutive k of a
//     row are one 16-byte load (IC % 8 == 0). Out-of-map taps (the padding) are zero-filled by the loads themselves:
//     no padded copy.
//   - The weights [OC][IC][KH][KW] are reordered once per call to [OC][KH][KW][IC] (conv2d_mma_weights; 1x1 needs no
//     reorder).
//   - A block computes BM positions x BN channels with 8 warps (each WM x WN), k in tiles of 32: on cc >= 8.0 a
//     two-slot cp.async pipeline (16-byte copies, zero-filled where the tap leaves the map or k passes K) with one
//     barrier a tile, ldmatrix from rows padded to 80 bytes (no bank conflicts), mma.sync m16n8k16 F16 x F16 -> F32,
//     two blocks a SM (<= 128 registers, 43.5 KB of shared memory each: RTX 3060, the 144-channel tile, 23-24 TFLOPS
//     against 19-20 for one block with four slots: the other block's math hides a block's epilogue and pipeline fill);
//     on Turing (no cp.async, no m16n8k16) the same tiles through registers into two shared slots (the next tile's
//     global loads issued before the current tile's math) and two m16n8k8 per k16, one block a SM (64 KB there).
//   - The epilogue writes F32 [b][oc][oy][ox] and applies what the graph fusion hands it, in the unfused kernels'
//     arithmetic: v = fma(scale, acc, sbias) (the SCALE after the convolution), then + bias[oc], then + z (residual).
// Accumulation is F32 throughout (sd.cpp's im2col path accumulates in F16 inside cuBLAS); against the old kernel the
// sums differ in order only.

#include "common.cuh"

static constexpr int CONV2D_MMA_BK  = 32;                 // k values a tile: 4 chunks of 8 halves (16 bytes)
static constexpr int CONV2D_MMA_LDS = CONV2D_MMA_BK + 8;  // shared row stride in halves (80 bytes)

struct conv2d_mma_params {
    int iw, ih, ic;                 // the F16 channels-last input map
    int ow, oh, oc;                 // the F32 output [b][oc][oh][ow]
    int kw, kh;
    int sx, sy, px, py, dx, dy;
    int k;                          // kh * kw * ic
    int ohw, npos;                  // oh * ow, b * oh * ow
    int ntn;                        // channel tiles: blockIdx.x = position tile * ntn + channel tile
    int   affine;                   // epilogue: v = fma(scale, acc, sbias) [+ bias[oc]] [+ z[i]]
    float scale, sbias;
    const float * bias;
    const float * z;
};

template <int BM_, int BN_, int WM_, int WN_, int STAGES_> struct conv2d_mma_cfg {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, STAGES = STAGES_;
    static constexpr int NWM = BM / WM, NWN = BN / WN, NT = NWM * NWN * 32;
    static constexpr int MF = WM / 16, NF = WN / 8;            // m16 / n8 fragments a warp
    static constexpr int A_PER = BM * 4 / NT;                  // 16-byte input chunks a thread and tile
    static constexpr int B_PER = (BN * 4 + NT - 1) / NT;       // 16-byte weight chunks a thread and tile
    static constexpr int SLOT  = (BM + BN) * CONV2D_MMA_LDS;   // halves a shared slot
    static constexpr size_t smem(const bool async) {
        return size_t(async ? STAGES : 2) * SLOT * sizeof(half);
    }
    static_assert(NT == 256 && BM * 4 % NT == 0 && WM % 16 == 0 && WN % 8 == 0, "tile layout");
};

static __device__ __forceinline__ void conv2d_mma_cp16(const uint32_t dst, const void * src, const bool ok) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    GGML_UNUSED(ok);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void conv2d_mma_ldsm4(uint32_t (&r)[4], const uint32_t addr) {
#ifdef TURING_MMA_AVAILABLE
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
#else
    GGML_UNUSED(r);
    GGML_UNUSED(addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void conv2d_mma_ldsm2(uint32_t & r0, uint32_t & r1, const uint32_t addr) {
#ifdef TURING_MMA_AVAILABLE
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
#else
    GGML_UNUSED(r0);
    GGML_UNUSED(r1);
    GGML_UNUSED(addr);
    NO_DEVICE_CODE;
#endif
}

// d += a * b: one m16n8k16 (K16), or two m16n8k8 (k 0-7: a0 a1 b0, then k 8-15: a2 a3 b1; Turing has no m16n8k16).
template <bool K16>
static __device__ __forceinline__ void conv2d_mma_mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t b0,
                                                      const uint32_t b1) {
#ifdef TURING_MMA_AVAILABLE
    if constexpr (K16) {
#    ifdef AMPERE_MMA_AVAILABLE
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#    else
        GGML_UNUSED(d);
        GGML_UNUSED(a);
        GGML_UNUSED(b0);
        GGML_UNUSED(b1);
        NO_DEVICE_CODE;
#    endif
    } else {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b0));
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                     "{%0, %1, %2, %3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[2]), "r"(a[3]), "r"(b1));
    }
#else
    GGML_UNUSED(d);
    GGML_UNUSED(a);
    GGML_UNUSED(b0);
    GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif
}

// The same with F16 accumulation (d: two half2 registers, rows g and g + 8); research option (GGML_CUDA_CONV2D_MMA_ACC=16).
template <bool K16>
static __device__ __forceinline__ void conv2d_mma_mma_h(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t b0,
                                                        const uint32_t b1) {
#ifdef TURING_MMA_AVAILABLE
    if constexpr (K16) {
#    ifdef AMPERE_MMA_AVAILABLE
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                     : "+r"(d[0]), "+r"(d[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#    else
        GGML_UNUSED(d);
        GGML_UNUSED(a);
        GGML_UNUSED(b0);
        GGML_UNUSED(b1);
        NO_DEVICE_CODE;
#    endif
    } else {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
                     : "+r"(d[0]), "+r"(d[1])
                     : "r"(a[0]), "r"(a[1]), "r"(b0));
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
                     : "+r"(d[0]), "+r"(d[1])
                     : "r"(a[2]), "r"(a[3]), "r"(b1));
    }
#else
    GGML_UNUSED(d);
    GGML_UNUSED(a);
    GGML_UNUSED(b0);
    GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif
}

// ASYNC: the cc >= 8.0 kernel (cp.async pipeline of CFG::STAGES slots, m16n8k16); otherwise the Turing kernel (two
// register-staged slots, m16n8k8), which also runs on newer cards (the test programs compare the two there).
template <class CFG, bool ASYNC, bool ACC16>
static __global__ void __launch_bounds__(CFG::NT, ASYNC ? 2 : 1) conv2d_mma_f16(const half * __restrict__ x,
                                                                    const half * __restrict__ w,
                                                                    float * __restrict__ y,
                                                                    const conv2d_mma_params p) {
#if defined(TURING_MMA_AVAILABLE)
#    if !defined(CP_ASYNC_AVAILABLE)
    if constexpr (ASYNC) {
        NO_DEVICE_CODE;
        return;
    } else
#    endif
    {
        constexpr int BM = CFG::BM, BN = CFG::BN, WM = CFG::WM, NT = CFG::NT, MF = CFG::MF, NF = CFG::NF;
        constexpr int A_PER = CFG::A_PER, B_PER = CFG::B_PER, SLOT = CFG::SLOT, LDS = CONV2D_MMA_LDS;
        constexpr int BK = CONV2D_MMA_BK, NSLOT = ASYNC ? CFG::STAGES : 2;
        extern __shared__ int4 conv2d_mma_smem[];
        half * smem = (half *) conv2d_mma_smem;

        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        const int tn = blockIdx.x % p.ntn, tm = blockIdx.x / p.ntn;
        const int p0 = tm * BM, n0 = tn * BN;
        const int wm = (warp % CFG::NWM) * WM, wn = (warp / CFG::NWM) * CFG::WN;
        const int cj = tid & 3;  // the thread's 16-byte chunk of a k row: k0 + 8 cj .. + 7

        // Input rows (positions) of this thread: (tid >> 2) + i * NT / 4. a_iy / a_ix: the tap (0, 0) input
        // coordinates, a_off its element offset (negative when outside the map; only used where in bounds).
        int a_off[A_PER], a_iy[A_PER], a_ix[A_PER];
#    pragma unroll
        for (int i = 0; i < A_PER; ++i) {
            const int pos = p0 + (tid >> 2) + i * (NT / 4);
            if (pos < p.npos) {
                const int n = pos / p.ohw, q = pos - n * p.ohw, oy = q / p.ow, ox = q - oy * p.ow;
                a_iy[i]     = oy * p.sy - p.py;
                a_ix[i]     = ox * p.sx - p.px;
                a_off[i]    = ((n * p.ih + a_iy[i]) * p.iw + a_ix[i]) * p.ic;
            } else {
                a_iy[i]  = INT_MIN / 2;  // fails every bounds test
                a_ix[i]  = 0;
                a_off[i] = 0;
            }
        }
        // Weight rows (channels) of this thread: chunk c = tid + i * NT, row c / 4.
        const half * b_src[B_PER];
        bool         b_ok[B_PER];
#    pragma unroll
        for (int i = 0; i < B_PER; ++i) {
            const int c = tid + i * NT, r = c >> 2;
            b_ok[i]     = c < BN * 4 && n0 + r < p.oc;
            b_src[i]    = w + (b_ok[i] ? int64_t(n0 + r) * p.k + 8 * cj : 0);
        }
        // The tap (ky, kx) and channel ci of the thread's k chunk in the next tile to load.
        int t_ci = 8 * cj, t_kx = 0, t_ky = 0;
        while (t_ci >= p.ic) {
            t_ci -= p.ic;
            if (++t_kx == p.kw) {
                t_kx = 0;
                ++t_ky;
            }
        }
        auto advance = [&]() {
            t_ci += BK;
            while (t_ci >= p.ic) {
                t_ci -= p.ic;
                if (++t_kx == p.kw) {
                    t_kx = 0;
                    ++t_ky;
                }
            }
        };

        const int KT = (p.k + BK - 1) / BK;

        // The source of every 16-byte chunk of tile kt (in the tap state's order), nullptr where it is zero.
        auto chunk_src = [&](const int kt, const half ** a, const half ** b) {
            const bool kv   = t_ky < p.kh;  // k < K
            const int  dyo  = t_ky * p.dy, dxo = t_kx * p.dx;
            const int  toff = (dyo * p.iw + dxo) * p.ic + t_ci;
#    pragma unroll
            for (int i = 0; i < A_PER; ++i) {
                const int iy = a_iy[i] + dyo, ix = a_ix[i] + dxo;
                const bool ok = kv && unsigned(iy) < unsigned(p.ih) && unsigned(ix) < unsigned(p.iw);
                a[i]          = ok ? x + (a_off[i] + toff) : nullptr;
            }
            const bool kb = kt * BK + 8 * cj < p.k;
#    pragma unroll
            for (int i = 0; i < B_PER; ++i) {
                b[i] = b_ok[i] && kb ? b_src[i] + kt * BK : nullptr;
            }
        };

        float    acc[MF][NF][ACC16 ? 1 : 4];
        uint32_t acch[MF][NF][ACC16 ? 2 : 1];  // F16 accumulation: half2 (rows g, g + 8)
#    pragma unroll
        for (int i = 0; i < MF; ++i) {
#    pragma unroll
            for (int j = 0; j < NF; ++j) {
                if constexpr (ACC16) {
                    acch[i][j][0] = 0;
                    acch[i][j][1] = 0;
                } else {
#    pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        acc[i][j][l] = 0.0f;
                    }
                }
            }
        }

        const uint32_t smem_base = (uint32_t) __cvta_generic_to_shared(smem);

        // One k tile from shared slot `slot`: per k16, MF ldmatrix.x4 of the positions, NF / 2 ldmatrix.x4 (+ one
        // .x2 when NF is odd) of the channels, MF x NF MMAs.
        auto compute = [&](const int slot) {
            const uint32_t sa = smem_base + slot * SLOT * 2, sb = sa + BM * LDS * 2;
#    pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                uint32_t a[MF][4], b[NF][2];
#    pragma unroll
                for (int i = 0; i < MF; ++i) {
                    conv2d_mma_ldsm4(a[i], sa + ((wm + 16 * i + (lane & 15)) * LDS + kk + (lane >> 4) * 8) * 2);
                }
#    pragma unroll
                for (int j = 0; j < NF; j += 2) {
                    if (j + 1 < NF) {
                        uint32_t r[4];
                        conv2d_mma_ldsm4(
                            r, sb + ((wn + 8 * j + (lane & 7) + ((lane >> 4) << 3)) * LDS + kk + ((lane >> 3) & 1) * 8) * 2);
                        b[j][0]     = r[0];
                        b[j][1]     = r[1];
                        b[j + 1][0] = r[2];
                        b[j + 1][1] = r[3];
                    } else {
                        conv2d_mma_ldsm2(b[j][0], b[j][1],
                                         sb + ((wn + 8 * j + (lane & 7)) * LDS + kk + ((lane >> 3) & 1) * 8) * 2);
                    }
                }
#    pragma unroll
                for (int i = 0; i < MF; ++i) {
#    pragma unroll
                    for (int j = 0; j < NF; ++j) {
                        if constexpr (ACC16) {
                            conv2d_mma_mma_h<ASYNC>(acch[i][j], a[i], b[j][0], b[j][1]);
                        } else {
                            conv2d_mma_mma<ASYNC>(acc[i][j], a[i], b[j][0], b[j][1]);
                        }
                    }
                }
            }
        };

        // Shared offsets (halves) of the thread's chunks in a slot.
        auto a_dst = [&](const int i) { return ((tid >> 2) + i * (NT / 4)) * LDS + 8 * cj; };
        auto b_dst = [&](const int i) { return BM * LDS + ((tid + i * NT) >> 2) * LDS + 8 * cj; };

        if constexpr (ASYNC) {
            auto load = [&](const int slot, const int kt) {
                const half * a[A_PER];
                const half * b[B_PER];
                chunk_src(kt, a, b);
                advance();
                const uint32_t s = smem_base + slot * SLOT * 2;
#    pragma unroll
                for (int i = 0; i < A_PER; ++i) {
                    conv2d_mma_cp16(s + a_dst(i) * 2, a[i] ? a[i] : x, a[i] != nullptr);
                }
#    pragma unroll
                for (int i = 0; i < B_PER; ++i) {
                    if (B_PER * NT == BN * 4 || tid + i * NT < BN * 4) {
                        conv2d_mma_cp16(s + b_dst(i) * 2, b[i] ? b[i] : w, b[i] != nullptr);
                    }
                }
            };
#    pragma unroll
            for (int s = 0; s < NSLOT - 1; ++s) {
                if (s < KT) {
                    load(s, s);
                }
                asm volatile("cp.async.commit_group;");
            }
            for (int kt = 0; kt < KT; ++kt) {
                asm volatile("cp.async.wait_group %0;" ::"n"(NSLOT - 2));
                __syncthreads();
                const int nk = kt + NSLOT - 1;
                if (nk < KT) {
                    load(nk % NSLOT, nk);
                }
                asm volatile("cp.async.commit_group;");
                compute(kt % NSLOT);
            }
        } else {
            int4 ra[A_PER], rb[B_PER];
            auto gload = [&](const int kt) {
                const half * a[A_PER];
                const half * b[B_PER];
                chunk_src(kt, a, b);
                advance();
#    pragma unroll
                for (int i = 0; i < A_PER; ++i) {
                    ra[i] = a[i] ? *(const int4 *) a[i] : make_int4(0, 0, 0, 0);
                }
#    pragma unroll
                for (int i = 0; i < B_PER; ++i) {
                    rb[i] = b[i] ? *(const int4 *) b[i] : make_int4(0, 0, 0, 0);
                }
            };
            auto sstore = [&](const int slot) {
                half * s = smem + slot * SLOT;
#    pragma unroll
                for (int i = 0; i < A_PER; ++i) {
                    *(int4 *) (s + a_dst(i)) = ra[i];
                }
#    pragma unroll
                for (int i = 0; i < B_PER; ++i) {
                    if (B_PER * NT == BN * 4 || tid + i * NT < BN * 4) {
                        *(int4 *) (s + b_dst(i)) = rb[i];
                    }
                }
            };
            gload(0);
            sstore(0);
            __syncthreads();
            for (int kt = 0; kt < KT; ++kt) {
                if (kt + 1 < KT) {
                    gload(kt + 1);
                }
                compute(kt & 1);
                if (kt + 1 < KT) {
                    sstore((kt + 1) & 1);
                }
                __syncthreads();
            }
        }

        // Epilogue: c0 c1 = (row g, channels 2t, 2t+1), c2 c3 = (row g + 8, same channels).
        const int g = lane >> 2, t = lane & 3;
#    pragma unroll
        for (int i = 0; i < MF; ++i) {
#    pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int pos = p0 + wm + 16 * i + g + 8 * h;
                if (pos >= p.npos) {
                    continue;
                }
                const int     n    = pos / p.ohw;
                const int64_t base = int64_t(n) * p.oc * p.ohw + (pos - n * p.ohw);
#    pragma unroll
                for (int j = 0; j < NF; ++j) {
#    pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int oc = n0 + wn + 8 * j + 2 * t + e;
                        if (oc < p.oc) {
                            const int64_t o = base + int64_t(oc) * p.ohw;
                            float         v;
                            if constexpr (ACC16) {
                                const half2 hv = *(const half2 *) &acch[i][j][h];
                                v              = __half2float(e ? __high2half(hv) : __low2half(hv));
                            } else {
                                v = acc[i][j][2 * h + e];
                            }
                            if (p.affine) {
                                v = __fmaf_rn(p.scale, v, p.sbias);
                            }
                            if (p.bias) {
                                v = __fadd_rn(v, p.bias[oc]);
                            }
                            if (p.z) {
                                v = __fadd_rn(v, p.z[o]);
                            }
                            y[o] = v;
                        }
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED(x);
    GGML_UNUSED(w);
    GGML_UNUSED(y);
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif  // defined(TURING_MMA_AVAILABLE)
}

// The input to F16, channels last: y[b][yy][xx][ch] = x[b][ch][yy / UP][xx / UP] (h, w: the output map; UP = 2: the
// graph's nearest 2x UPSCALE folded in), as fma(scale, v, sbias) when affine (the graph's SCALE: scale_f32's FFMA).
// A block: 64 positions x 32 channels through shared memory (coalesced F32 reads along the map, 16-byte F16 writes).
// Grid: (ceil(h * w / 64), ceil(c / 32), b). c % 8 == 0.
template <int UP>
static __global__ void __launch_bounds__(256) conv2d_mma_nhwc(const float * __restrict__ x,
                                                             half * __restrict__ y,
                                                             const int  c,
                                                             const int  h,
                                                             const int  w,
                                                             const int  affine,
                                                             const float scale,
                                                             const float sbias) {
    __shared__ float t[32][65];
    const int        hw = h * w, b = blockIdx.z, q0 = blockIdx.x * 64, c0 = blockIdx.y * 32;
    const int        sw = w / UP, shw = (h / UP) * sw;
    {
        const int q = q0 + (threadIdx.x & 63);
        int       src = 0;
        if (q < hw) {
            if constexpr (UP == 1) {
                src = q;
            } else {
                const int yy = q / w, xx = q - yy * w;
                src          = (yy / UP) * sw + xx / UP;
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int ch = (threadIdx.x >> 6) + 4 * r;
            float     v  = 0.0f;
            if (q < hw && c0 + ch < c) {
                v = x[(int64_t(b) * c + c0 + ch) * shw + src];
                if (affine) {
                    v = __fmaf_rn(scale, v, sbias);
                }
            }
            t[ch][threadIdx.x & 63] = v;
        }
    }
    __syncthreads();
    const int pq = threadIdx.x >> 2, j = threadIdx.x & 3, q = q0 + pq;
    if (q < hw && c0 + 8 * j < c) {
        half2 o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            o[e] = __floats2half2_rn(t[8 * j + 2 * e][pq], t[8 * j + 2 * e + 1][pq]);
        }
        *(int4 *) (y + (int64_t(b) * hw + q) * c + c0 + 8 * j) = *(const int4 *) o;
    }
}

// The weights [oc][ic][khw] -> [oc][khw][ic] (k = tap * ic + ci). SMEM: a block per output channel through shared
// memory (k halves of dynamic shared memory); otherwise a thread per value.
template <bool SMEM>
static __global__ void __launch_bounds__(256) conv2d_mma_weights(const half * __restrict__ w,
                                                                 half * __restrict__ wt,
                                                                 const int ic,
                                                                 const int khw,
                                                                 const int64_t total) {
    const int k = ic * khw;
    if constexpr (SMEM) {
        extern __shared__ int4 conv2d_mma_wsmem[];
        half *               s  = (half *) conv2d_mma_wsmem;
        const int64_t        o0 = int64_t(blockIdx.x) * k;
        for (int e = threadIdx.x; e < k; e += blockDim.x) {
            s[e] = w[o0 + e];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < k; e += blockDim.x) {
            const int tap = e / ic, ci = e - tap * ic;
            wt[o0 + e]    = s[ci * khw + tap];
        }
    } else {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= total) {
            return;
        }
        const int64_t o = i / k;
        const int     e = int(i - o * k), tap = e / ic, ci = e - tap * ic;
        wt[i]           = w[o * k + ci * khw + tap];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Host side.

// The tile configurations: channels a block (BN) from the output channel count; 128 positions a block (256 for the
// narrow one), 8 warps.
using conv2d_mma_cfg144 = conv2d_mma_cfg<128, 144, 32, 72, 2>;  // 144 | the Wan 2.2 / Qwen-Image 2.1 VAE's channels
using conv2d_mma_cfg128 = conv2d_mma_cfg<128, 128, 32, 64, 2>;
using conv2d_mma_cfg64  = conv2d_mma_cfg<128, 64, 32, 32, 2>;
using conv2d_mma_cfg8   = conv2d_mma_cfg<256, 8, 32, 8, 2>;     // <= 8 channels (the VAE's last conv: 4)

enum conv2d_mma_tile { CONV2D_MMA_T144 = 0, CONV2D_MMA_T128 = 1, CONV2D_MMA_T64 = 2, CONV2D_MMA_T8 = 3 };

// The tile that pads the fewest channels (ties: the wider one).
static int conv2d_mma_pick_tile(const int oc) {
    if (oc <= 8) {
        return CONV2D_MMA_T8;
    }
    if (oc <= 64) {
        return CONV2D_MMA_T64;
    }
    const int pad144 = (oc + 143) / 144 * 144 - oc;
    const int pad128 = (oc + 127) / 128 * 128 - oc;
    const int pad64  = (oc + 63) / 64 * 64 - oc;
    if (pad144 <= pad128 && pad144 <= pad64) {
        return CONV2D_MMA_T144;
    }
    return pad128 <= pad64 ? CONV2D_MMA_T128 : CONV2D_MMA_T64;
}

// Whether the kernel takes this convolution: IC % 8 == 0 (16-byte chunks of channels), every index in int32 (the
// padded map's too: a tap's offset is formed before its bounds test); any kernel size, stride, dilation, padding.
static bool conv2d_mma_shape_ok(const int64_t IW, const int64_t IH, const int64_t OW, const int64_t OH, const int64_t KW,
                                const int64_t KH, const int64_t IC, const int64_t OC, const int64_t B, const int ST_X,
                                const int ST_Y, const int PD_X, const int PD_Y, const int DL_X, const int DL_Y) {
    const int64_t lim = INT_MAX - 1024;
    return IC % 8 == 0 && IC >= 8 && OC >= 1 && ST_X > 0 && ST_Y > 0 && DL_X > 0 && DL_Y > 0 && PD_X >= 0 && PD_Y >= 0 &&
           IW * IH * IC * B <= lim && OW * OH * OC * B <= lim && OW * OH * B <= lim && KW * KH * IC <= lim / 8 &&
           OC * KW * KH * IC <= lim && (IW + 2 * PD_X + 1) * (IH + 2 * PD_Y + 1) * IC <= lim && B <= 65535 &&
           OW >= 1 && OH >= 1;
}

template <class CFG, bool ASYNC, bool ACC16>
static void conv2d_mma_launch_cfg(const half * xh, const half * wt, float * y, conv2d_mma_params p, cudaStream_t st) {
    p.ntn              = (p.oc + CFG::BN - 1) / CFG::BN;
    const int64_t ntm  = (int64_t(p.npos) + CFG::BM - 1) / CFG::BM;
    const size_t  smem = CFG::smem(ASYNC);
    CUDA_SET_SHARED_MEMORY_LIMIT((conv2d_mma_f16<CFG, ASYNC, ACC16>), smem);
    conv2d_mma_f16<CFG, ASYNC, ACC16><<<dim3(unsigned(ntm * p.ntn)), CFG::NT, smem, st>>>(xh, wt, y, p);
}

template <class CFG>
static void conv2d_mma_launch_tile(const half * xh, const half * wt, float * y, const conv2d_mma_params & p,
                                   const bool async, const bool acc16, cudaStream_t st) {
    if (acc16) {
        async ? conv2d_mma_launch_cfg<CFG, true, true>(xh, wt, y, p, st) :
                conv2d_mma_launch_cfg<CFG, false, true>(xh, wt, y, p, st);
    } else {
        async ? conv2d_mma_launch_cfg<CFG, true, false>(xh, wt, y, p, st) :
                conv2d_mma_launch_cfg<CFG, false, false>(xh, wt, y, p, st);
    }
}

// async: the cc >= 8.0 kernel, else the Turing one; acc16: F16 accumulation (research).
static void conv2d_mma_launch(const half * xh, const half * wt, float * y, const conv2d_mma_params & p, const bool async,
                              cudaStream_t st, const bool acc16 = false) {
    switch (conv2d_mma_pick_tile(p.oc)) {
        case CONV2D_MMA_T144:
            conv2d_mma_launch_tile<conv2d_mma_cfg144>(xh, wt, y, p, async, acc16, st);
            break;
        case CONV2D_MMA_T128:
            conv2d_mma_launch_tile<conv2d_mma_cfg128>(xh, wt, y, p, async, acc16, st);
            break;
        case CONV2D_MMA_T64:
            conv2d_mma_launch_tile<conv2d_mma_cfg64>(xh, wt, y, p, async, acc16, st);
            break;
        default:
            conv2d_mma_launch_tile<conv2d_mma_cfg8>(xh, wt, y, p, async, acc16, st);
            break;
    }
}

// The input's F16 channels-last copy (B x H x W x C halves; H, W the map the convolution reads, UP the upsampling of
// x into it).
static void conv2d_mma_to_nhwc(const float * x, half * xh, const int c, const int h, const int w, const int b,
                               const int up, const bool affine, const float scale, const float sbias, cudaStream_t st) {
    const dim3 grid(unsigned((int64_t(h) * w + 63) / 64), unsigned((c + 31) / 32), unsigned(b));
    if (up == 2) {
        conv2d_mma_nhwc<2><<<grid, 256, 0, st>>>(x, xh, c, h, w, affine, scale, sbias);
    } else {
        conv2d_mma_nhwc<1><<<grid, 256, 0, st>>>(x, xh, c, h, w, affine, scale, sbias);
    }
}

static void conv2d_mma_reorder_weights(const half * w, half * wt, const int oc, const int ic, const int khw,
                                       cudaStream_t st) {
    const int     k     = ic * khw;
    const int64_t total = int64_t(oc) * k;
    if (int64_t(k) * 2 <= 48 * 1024) {
        conv2d_mma_weights<true><<<oc, 256, k * 2, st>>>(w, wt, ic, khw, total);
    } else {
        conv2d_mma_weights<false><<<unsigned((total + 255) / 256), 256, 0, st>>>(w, wt, ic, khw, total);
    }
}
