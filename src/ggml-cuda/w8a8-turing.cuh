#pragma once

// The W8A8 GEMM for Turing (cc 7.5: RTX 20xx, GTX 16xx; sm_75): w8a8_gemm (w8a8.cuh) on what sm_75 has. The same
// inputs (q_x / sy / d_t of the activation quantizer or the norm writer, q_w / dW of the weight converter), the same
// sy layout (w8a8_sy_pos), tile order (`group`), epilogues (the expressions verbatim) and launch contract
// (w8a8t_gemm_cuda = w8a8_gemm_cuda's signature); the outputs are bit-identical to w8a8_gemm's: the int32 sums are
// exact whatever the MMA shape, and every output value goes through the same epilogue expressions in the same order.
//
// What differs from the sm_80 kernel (measurements: NOTES-w8a8.md, "Turing port"):
// - MMA: sm_75 has no m16n8k32 int8 shape; the m16n8k32 fragment registers (ldmatrix x4, unchanged) feed four
//   mma.sync.m8n8k16.s32.s8.s8.s32: rows 0-7 (a0, a2) and rows 8-15 (a1, a3), k 0-15 (a0 / a1 x b0) then k 16-31
//   (a2 / a3 x b1) through the C operand, the first of each pair with a zero C. The 32-value group's dot product is
//   then the same exact int32 P as the m16n8k32's (|P| <= 32*127^2), and the sub-scale IMAD is the sm_80 kernel's.
//   satfinite as there: it cannot trigger on a 32-value sum; the accumulation across groups is the IMAD's.
// - Warp tile W8A8T_WM x 64, default 32 x 64: 8 warps, 64 accumulators a thread, one block of 256 threads per SM.
//   Four m8n8k16 per m16n8k32 make ptxas hoist many independent MMAs whose products then hold registers: with the
//   sm_80 kernel's 64 x 64 warp tile (128 accumulators, two blocks of 128 threads per SM) the k loop spilled and
//   shuffled accumulators (~45-100 MOVs per stage, 255 registers + 24-64 bytes of stack); at 32 x 64 it has no spills,
//   ~2 MOVs, ~236 registers, a ~320-instruction (5 KB) loop for 128 MMAs. The block tile stays 128 x 128 (the sy
//   layout is per 128-token tile; w8a8_sy_pos gives a 32-row warp slice's 2*MF values of a group as consecutive bytes).
// - No cp.async: each stage goes global -> registers (16-byte ld.global.nc with the L2::128B prefetch hint: the
//   row's next stage is an L2 hit) -> shared. Default: three slots (3 x 16 KB of Turing's 64 KB: one block per SM);
//   stage kt + 2 is loaded into registers during stage kt, in W8A8T_PIECES pieces, and stored during stage kt + 1, so
//   a load has a whole stage to arrive and the one barrier per stage (the sm_80 kernel's place) never waits for one.
//   (W8A8T_SLOTS 2, the 64 x 64 tile's: two slots, each piece stored 8/PIECES pairs of n fragments after its load,
//   the last before the barrier.) The sub-scales are read from global memory (2*MF bytes per thread and group, one
//   k step ahead) instead of a shared-memory stage.
//
// The device code is compiled for sm_75 only (W8A8T_DEVICE_OK; elsewhere empty stubs): w8a8-mm.cu dispatches it on
// cc 7.5 when the build has sm_75 / compute_75 code. W8A8T_ANY_ARCH (tests only) compiles it for every arch >= 7.5,
// so that one sm_86 test binary can compare it with w8a8_gemm byte for byte.

#include "w8a8.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750 || (defined(W8A8T_ANY_ARCH) && __CUDA_ARCH__ >= 750))
#define W8A8T_DEVICE_OK 1
#else
#define W8A8T_DEVICE_OK 0
#endif

// Research knobs (the defaults are the measured choice; NOTES-w8a8.md):
#ifndef W8A8T_WM
#define W8A8T_WM 32        // tokens per warp tile: 32 (8 warps, one block per SM) or 64 (4 warps, two blocks per SM)
#endif
#ifndef W8A8T_SLOTS
#define W8A8T_SLOTS (W8A8T_WM == 32 ? 3 : 2)   // shared-memory stages of 16 KB
#endif
#ifndef W8A8T_PIECES
#define W8A8T_PIECES (W8A8T_WM == 32 ? 2 : 4)  // a stage's 16-byte chunks per thread (A and B) go in this many pieces
#endif
#ifndef W8A8T_LDG_GAP
#define W8A8T_LDG_GAP 1    // 3 slots: pairs of n fragments between a piece's store and the load into its registers
#endif
#ifndef W8A8T_LDG_HINT
#define W8A8T_LDG_HINT 1   // 1: ld.global.nc.L2::128B, 0: ld.global.nc
#endif

static constexpr int W8A8T_MF       = W8A8T_WM / 16;              // m16 fragments per warp
static constexpr int W8A8T_WARPS_M  = W8A8_BM / W8A8T_WM;
static constexpr int W8A8T_NTHREADS = W8A8T_WARPS_M * 64;         // W8A8T_WARPS_M (tokens) x 2 (channels) warps
static constexpr int W8A8T_MINBLOCKS = W8A8T_WM == 64 ? 2 : 1;    // 255 registers a thread either way

static constexpr int w8a8t_stage_bytes() { return W8A8_BM*W8A8_BK + W8A8_BN*W8A8_BK; }   // 16,384
// the pipeline slots, or the epilogue's per-warp staging (16 rows of 64 values + 32 bytes, f32) if larger
static constexpr int w8a8t_smem_bytes() {
    return W8A8T_SLOTS*w8a8t_stage_bytes() > (W8A8T_NTHREADS/32) * 16 * (64*4 + 32) ? W8A8T_SLOTS*w8a8t_stage_bytes()
                                                                                    : (W8A8T_NTHREADS/32) * 16 * (64*4 + 32);
}

#if W8A8T_DEVICE_OK
static __device__ __forceinline__ int4 w8a8t_ldg16(const int8_t * p) {
    int4 v;
#if W8A8T_LDG_HINT
    asm volatile("ld.global.nc.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
    asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#endif
    return v;
}
static __device__ __forceinline__ void w8a8t_sts16(const unsigned addr, const int4 v) {
    asm volatile("st.shared.v4.s32 [%0], {%1, %2, %3, %4};" : : "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// the thread's 2*MF sub-scale bytes of one group (8 or 4 consecutive bytes)
static __device__ __forceinline__ void w8a8t_ldg_sy(uint2 & s, const uint8_t * p) {
    if constexpr (W8A8T_MF == 4) {
        asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(s.x), "=r"(s.y) : "l"(p));
    } else {
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(s.x) : "l"(p));
        s.y = 0;
    }
}
// w8a8_swz (the sm_80 kernel's swizzle, defined there for sm_80+ only)
static __device__ __forceinline__ int w8a8t_swz(const int r, const int c) {
    return r*64 + ((c ^ ((r >> 1) & 3)) << 4);
}
static __device__ __forceinline__ void w8a8t_ldsm_x4(int (&r)[4], const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// d = a x b + c for one 8x8 output block of 16 k values (a: row gid, k 4*tig..+3; b: column gid, the same k; d / c:
// row gid, columns 2*tig, 2*tig+1)
static __device__ __forceinline__ void w8a8t_mma8816(int & d0, int & d1, const int a, const int b, const int c0, const int c1) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
// the m16n8k32 product of the sm_80 kernel (zero C) from the same fragment registers: p[0..1] row gid, p[2..3] row gid+8
static __device__ __forceinline__ void w8a8t_mma_zero(int (&p)[4], const int (&a)[4], const int b0, const int b1) {
    int l0, l1, h0, h1;
    w8a8t_mma8816(l0, l1, a[0], b0, 0, 0);
    w8a8t_mma8816(h0, h1, a[1], b0, 0, 0);
    w8a8t_mma8816(p[0], p[1], a[2], b1, l0, l1);
    w8a8t_mma8816(p[2], p[3], a[3], b1, h0, h1);
}
#endif // W8A8T_DEVICE_OK

// One block per BM x BN output tile; grid = ntm * ntn (grouped order); dst: f32 [T][ldd] (EPI_F32) or the GLU output
// [T][ldd] (ldd = N/2 values per row). The k loop's structure (fragment schedule, the serpentine m order, the barrier
// position) is w8a8_gemm's; see there.
template <int epi>
static __global__ void __launch_bounds__(W8A8T_NTHREADS, W8A8T_MINBLOCKS) w8a8t_gemm(
        const int8_t * __restrict__ qx, const uint8_t * __restrict__ sy, const float * __restrict__ dt,
        const int8_t * __restrict__ qw, const float * __restrict__ dw, void * __restrict__ dst, const int64_t ldd,
        const int T, const int N, const int K, const int group) {
#if W8A8T_DEVICE_OK
    static_assert(W8A8_BM == 128 && W8A8_BN == 128 && W8A8_BK == 64 && W8A8_WM == 64, "w8a8_sy_pos's layout");
    static_assert(W8A8T_WM == 32 || W8A8T_WM == 64, "W8A8T_WM");
    constexpr int CPO = W8A8_BM*W8A8_BK/16 / W8A8T_NTHREADS;   // 16-byte chunks per thread and operand per stage (2 or 4)
    static_assert(W8A8T_PIECES >= 1 && W8A8T_PIECES <= 2*CPO && (2*CPO) % W8A8T_PIECES == 0 && 8 % W8A8T_PIECES == 0, "W8A8T_PIECES");
    static_assert(W8A8T_LDG_GAP >= 0 && W8A8T_LDG_GAP < 8 / W8A8T_PIECES, "W8A8T_LDG_GAP");
    extern __shared__ __align__(128) uint8_t w8a8t_smem[];
    constexpr int BM = W8A8_BM, BN = W8A8_BN, BK = W8A8_BK;
    constexpr int A_BYTES = BM*BK, B_BYTES = BN*BK, STAGE = A_BYTES + B_BYTES;
    constexpr int NT = W8A8T_NTHREADS;

    const int ntm = (T + BM - 1) / BM;
    const int ntn = N / BN;
    // grouped tile order (w8a8_gemm's, verbatim)
    const int bid = blockIdx.x;
    int tm, tn;
    if (group > 0) {
        const int per_grp = group * ntn;
        const int first_m = (bid / per_grp) * group;
        const int gsize   = min(group, ntm - first_m);
        const int in_grp  = bid % per_grp;
        tm = first_m + in_grp % gsize;
        tn = in_grp / gsize;
    } else {
        const int per_grp = -group * ntm;
        const int first_n = (bid / per_grp) * -group;
        const int gsize   = min(-group, ntn - first_n);
        const int in_grp  = bid % per_grp;
        tn = first_n + in_grp % gsize;
        tm = in_grp / gsize;
    }
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int wm   = warp % W8A8T_WARPS_M;  // W8A8T_WM-token slice
    constexpr int MF = W8A8T_MF, WM = W8A8T_WM;
    const int wn   = warp / W8A8T_WARPS_M;  // 64-channel slice
    const int gid  = lane / 4;
    const int tig  = lane % 4;

    const int G  = K / 32;
    const int nk = K / BK;

    const unsigned sbase = (unsigned) __cvta_generic_to_shared(w8a8t_smem);

    // per-thread copies (w8a8_gemm's chunk layout): chunk c (0 .. 2*CPO-1) of a stage = A (c < CPO) or B, row
    // (c % CPO)*NT/4 + tid/4, 16-byte chunk tid%4 (rows r and r + 32*i have the same swizzle)
    constexpr int CH = 2*CPO / W8A8T_PIECES;  // chunks per piece
    const unsigned cp_soff = (unsigned) w8a8t_swz(tid / 4, tid % 4);
    const int8_t * pA = qx + ((int64_t) m0 + tid/4) * K + 16*(tid % 4);
    const int8_t * pB = qw + ((int64_t) n0 + tid/4) * K + 16*(tid % 4);
    const int      KR = (NT/4) * K;   // the rows one pass of the threads copies (<= 64 * 16384: int)
    auto gptr = [&](const int c) { return c < CPO ? pA + (c % CPO)*KR : pB + (c % CPO)*KR; };
    auto soff = [&](const int c) { return cp_soff + (c < CPO ? 0 : A_BYTES) + (c % CPO)*(NT/4)*BK; };
    constexpr int SLOTS = W8A8T_SLOTS;
    static_assert(SLOTS == 2 || SLOTS == 3, "W8A8T_SLOTS");
    constexpr int NSTG = SLOTS == 3 ? 2*CPO : CH;  // staging registers (int4): 3 slots hold a whole stage for a stage
    int4 stg[NSTG];
    auto ldg_piece = [&](const int j) {           // piece j of the stage after the last one loaded: pA / pB advance
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            const int c = j*CH + i;
            stg[SLOTS == 3 ? c : i] = w8a8t_ldg16(gptr(c));
            if (c == CPO - 1) {
                pA += BK;
            }
            if (c == 2*CPO - 1) {
                pB += BK;
            }
        }
    };
    auto sts_piece = [&](const unsigned slot_base, const int j) {
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            w8a8t_sts16(slot_base + soff(j*CH + i), stg[SLOTS == 3 ? j*CH + i : i]);
        }
    };
    // the sub-scales of the thread's 2*MF rows of group g: consecutive bytes at sy + (tm*G + g)*BM + w8a8_sy_pos(row
    // wm*WM + gid) (its mf = 0, h = 0 row; w8a8_gemm's sy_off)
    const uint8_t * pS = sy + (int64_t) tm * G * BM + w8a8_sy_pos(wm*WM + gid);

    int acc[MF][8][4];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int nf = 0; nf < 8; ++nf)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[mf][nf][j] = 0;

    // per-lane fragment offsets in a stage (w8a8_gemm's): rows r and r + 16*i have the same swizzle, so the m / n
    // fragment is an immediate offset of mf*1024 / np*1024; index = k step
    unsigned a_off[2], b_off[2];
    {
        const int a_row = wm*WM + (lane % 16);
        const int a_x   = (lane / 16) ^ ((a_row >> 1) & 3);
        const int b_row = wn*64 + (lane % 8) + (lane / 16)*8;
        const int b_x   = ((lane / 8) % 2) ^ ((b_row >> 1) & 3);
        a_off[0] = a_row*64 + (a_x << 4);
        a_off[1] = a_row*64 + ((a_x ^ 2) << 4);
        b_off[0] = A_BYTES + b_row*64 + (b_x << 4);
        b_off[1] = A_BYTES + b_row*64 + ((b_x ^ 2) << 4);
    }

    auto load_a1 = [&](int (&f)[4], const unsigned st, const int ks, const int mf) {
        w8a8t_ldsm_x4(f, st + a_off[ks] + mf*1024);
    };
    auto load_b = [&](int (&fb)[4], const unsigned st, const int ks, const int np) {
        w8a8t_ldsm_x4(fb, st + b_off[ks] + np*1024);
    };

    int   fa[MF][4] = {};
    uint2 fs;
    int   fb[2][4] = {};

    // one k step (w8a8_gemm's kstep): fa / fs / fb[0] hold its A fragments, sub-scales and B pair 0; its B pairs 1..3
    // are at st / ks; the next k step is at nst / nks; the next group's sub-scales are loaded into fs from global
    // memory (when next_s) right after this step's are extracted; hook(np) runs before the loads and MMAs of pair np
    auto kstep = [&](const int par, const unsigned st, const int ks, const unsigned nst, const int nks, const bool next_s,
                     auto && hook) {
        int sv[MF][2];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf) {
            const unsigned w = mf < 2 ? fs.x : fs.y;
            sv[mf][0] = (w >> (16*(mf % 2)))     & 0xFF;
            sv[mf][1] = (w >> (16*(mf % 2) + 8)) & 0xFF;
        }
        pS += BM;
        if (next_s) {
            w8a8t_ldg_sy(fs, pS);
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            hook(np);
            if (np < 3) {
                load_b(fb[(np + 1) % 2], st, ks, np + 1);
            } else {
                load_b(fb[0], nst, nks, 0);
            }
            const int (&b)[4] = fb[np % 2];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
                for (int j = 0; j < MF; ++j) {
                    // the serpentine m order of w8a8_gemm (the A fragment of the last MMA is reused)
                    const int mf = (nn ^ par) ? MF - 1 - j : j;
                    int (&c)[4] = acc[mf][2*np + nn];
                    int p[4];
                    w8a8t_mma_zero(p, fa[mf], b[2*nn], b[2*nn + 1]);
                    c[0] += sv[mf][0] * p[0];
                    c[1] += sv[mf][0] * p[1];
                    c[2] += sv[mf][1] * p[2];
                    c[3] += sv[mf][1] * p[3];
                    if (np == 3 && nn == 1) {
                        load_a1(fa[mf], nst, nks, mf);   // its last use in this k step
                    }
                }
            }
        }
    };

    // prologue: stage 0 into slot 0, the first piece of stage 1, the sub-scales of group 0, then the fragments of k step 0
    {
        int4 v[2*CPO];
#pragma unroll
        for (int c = 0; c < 2*CPO; ++c) {
            v[c] = w8a8t_ldg16(gptr(c));
        }
#pragma unroll
        for (int c = 0; c < 2*CPO; ++c) {
            w8a8t_sts16(sbase + soff(c), v[c]);
        }
        pA += BK;
        pB += BK;
    }
    if (nk > 1) {   // stage 1: its first piece (2 slots) or all of it (3 slots)
#pragma unroll
        for (int j = 0; j < (SLOTS == 3 ? W8A8T_PIECES : 1); ++j) {
            ldg_piece(j);
        }
    }
    w8a8t_ldg_sy(fs, pS);
    __syncthreads();
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        load_a1(fa[mf], sbase, 0, mf);
    }
    load_b(fb[0], sbase, 0, 0);

    // The copy schedule, in pairs q = 4*ks + np = 0..7 of stage kt (LAG = 8/PIECES); the barrier at q = 7 (before the
    // loads of the next stage's first fragments: every read of slot kt is issued, stage kt + 1 is complete after it).
    // 3 slots: stage kt + 1 (loaded into the registers during stage kt - 1) is stored into its slot (last read in stage
    // kt - 2) during stage kt, piece j at q = j*LAG, and piece j of stage kt + 2 is loaded W8A8T_LDG_GAP pairs later
    // (the store has read the registers by then): a whole stage between a load and its store, and the barrier never
    // waits for a load. 2 slots: piece j of stage kt + 1 is stored at q = (j+1)*LAG - 1 and the next piece loaded
    // right after; the last piece is stored at q = 7 before the barrier, and piece 0 of stage kt + 2 loaded after it.
    constexpr int LAG = 8 / W8A8T_PIECES;
    int slot = 0;   // the slot of the stage being computed
    for (int kt = 0; kt < nk; ++kt) {
        const bool more1 = kt + 1 < nk;   // stage kt + 1 exists (it is being copied)
        const bool more2 = kt + 2 < nk;
        const int nslot = SLOTS == 2 ? slot ^ 1 : (slot == SLOTS - 1 ? 0 : slot + 1);
        const unsigned st  = sbase + slot*STAGE;
        const unsigned nst = sbase + nslot*STAGE;
        auto copy_hook = [&](const int q) {
            if (SLOTS == 3) {
                if (q % LAG == 0 && more1) {
                    sts_piece(nst, q / LAG);
                }
                if ((q + 8 - W8A8T_LDG_GAP) % LAG == 0 && more2) {   // W8A8T_LDG_GAP pairs after the piece's store
                    ldg_piece(((q + 8 - W8A8T_LDG_GAP) % 8) / LAG);
                }
                if (q == 7) {
                    __syncthreads();
                }
            } else if ((q + 1) % LAG == 0) {
                const int j = (q + 1)/LAG - 1;
                if (more1) {
                    sts_piece(nst, j);
                }
                if (q == 7) {
                    __syncthreads();
                    if (more2) {
                        ldg_piece(0);
                    }
                } else if (more1) {
                    ldg_piece(j + 1);
                }
            }
        };
        kstep(0, st, 0, st, 1, true,  [&](const int np) { copy_hook(np); });
        kstep(1, st, 1, nst, 0, more1, [&](const int np) { copy_hook(4 + np); });
        slot = nslot;
    }

    // epilogue: w8a8_gemm's (per m fragment through the warp's shared-memory buffer, 16-byte streaming stores), with
    // this kernel's warp slice (WM / MF); the output expressions verbatim
    __syncthreads();  // the pipeline's slots are free (the last stages' fragment reads are done)
    constexpr int EPI_COLS = epi == W8A8_EPI_F32 || epi == W8A8_EPI_I32 ? 64 : 32;   // values per row of the warp
    constexpr int EPI_ESZ  = epi == W8A8_EPI_GLU_F16 ? 2 : 4;                         // bytes per value
    constexpr int EPI_LD   = EPI_COLS + 32/EPI_ESZ;   // row stride in values (+32 bytes: conflict-free fragment writes)
    uint8_t * ebuf = w8a8t_smem + warp * (16 * EPI_LD * EPI_ESZ);
    float wsc[8][2];
#pragma unroll
    for (int nf = 0; nf < 8; ++nf) {
        const float2 d2 = *(const float2 *) (dw + n0 + wn*64 + nf*8 + 2*tig);
        wsc[nf][0] = d2.x;
        wsc[nf][1] = d2.y;
    }
    float st[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = m0 + wm*WM + mf*16 + h*8 + gid;
            st[mf][h] = t < T ? dt[t] * (1.0f / (float) W8A8_SY_MAX) : 0.0f;
        }
    }
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = h*8 + gid;   // row in the 16-row chunk
#pragma unroll
            for (int nf = 0; nf < 8; ++nf) {
                const int c = nf*8 + 2*tig;
                if constexpr (epi == W8A8_EPI_I32) {
                    *(int2 *) (ebuf + (r*EPI_LD + c)*4) = make_int2(acc[mf][nf][2*h + 0], acc[mf][nf][2*h + 1]);
                } else {
                    const float y0 = (float) acc[mf][nf][2*h + 0] * st[mf][h] * wsc[nf][0];
                    const float y1 = (float) acc[mf][nf][2*h + 1] * st[mf][h] * wsc[nf][1];
                    if constexpr (epi == W8A8_EPI_F32) {
                        *(float2 *) (ebuf + (r*EPI_LD + c)*4) = make_float2(y0, y1);
                    } else {
                        // ggml's swiglu: silu(gate) * up (unary_gated_op_kernel<op_silu>, silu = x / (1 + exp(-x)))
                        const float g = __fmul_rn(y0 / (1.0f + expf(-y0)), y1);
                        if constexpr (epi == W8A8_EPI_GLU_F16) {
                            // clamped to the f16 range first (w8a8_gemm's comment)
                            *(__half *) (ebuf + (r*EPI_LD + c/2)*2) = __float2half(fminf(fmaxf(g, -65504.f), 65504.f));
                        } else {
                            *(float *) (ebuf + (r*EPI_LD + c/2)*4) = g;
                        }
                    }
                }
            }
        }
        __syncwarp();
        // 16 rows x EPI_COLS*EPI_ESZ bytes = 16 x 256 (f32) / 64 (f16) / 128 (GLU f32) bytes, 16 bytes per lane
        constexpr int V_ROW = EPI_COLS*EPI_ESZ/16;     // 16-byte vectors per row
        constexpr int ROWS  = 32 / V_ROW;              // rows per warp store
        const int vr = lane / V_ROW, vc = lane % V_ROW;
#pragma unroll
        for (int r0 = 0; r0 < 16; r0 += ROWS) {
            const int r = r0 + vr;
            const int t = m0 + wm*WM + mf*16 + r;
            const int4 v = *(const int4 *) (ebuf + r*EPI_LD*EPI_ESZ + vc*16);
            if (t < T) {
                // column of this vector's first value in the output row
                const int64_t col = epi == W8A8_EPI_F32 || epi == W8A8_EPI_I32 ? n0 + wn*64 + vc*4 : (n0 + wn*64)/2 + vc*(16/EPI_ESZ);
                __stcs((int4 *) ((uint8_t *) dst + ((int64_t) t*ldd + col)*EPI_ESZ), v);
            }
        }
    }
#else
    (void) qx; (void) sy; (void) dt; (void) qw; (void) dw; (void) dst; (void) ldd; (void) T; (void) N; (void) K; (void) group;
#endif
}

// w8a8_gemm_cuda's contract (raw pointers, stream-ordered; group 0 = w8a8_default_group) for the Turing kernel
static inline cudaError_t w8a8t_gemm_cuda(const int epi, const int8_t * qx, const uint8_t * sy, const float * dt, const int8_t * qw,
                                          const float * dw, void * dst, const int64_t ldd, const int T, const int N, const int K,
                                          cudaStream_t stream, int group = 0) {
    if (group == 0) {
        group = w8a8_default_group(T, N, K);
    }
    const int ntm = (T + W8A8_BM - 1) / W8A8_BM;
    const int ntn = N / W8A8_BN;
    const dim3 grid((unsigned) (ntm * ntn));
    const int smem = w8a8t_smem_bytes();
    cudaError_t err = cudaSuccess;
    // the largest shared-memory carveout (Turing: 64 KB), once per device (function attributes are per device context)
    int device = 0;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return err;
    }
#define W8A8T_G(E) { \
        static bool attr_set[64] = {}; \
        if (device < 64 && !attr_set[device]) { \
            err = cudaFuncSetAttribute(w8a8t_gemm<E>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            if (err == cudaSuccess) { \
                err = cudaFuncSetAttribute(w8a8t_gemm<E>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            } \
            if (err != cudaSuccess) { \
                return err; \
            } \
            attr_set[device] = true; \
        } \
        w8a8t_gemm<E><<<grid, W8A8T_NTHREADS, smem, stream>>>(qx, sy, dt, qw, dw, dst, ldd, T, N, K, group); }
    switch (epi) {
        case W8A8_EPI_F32:     W8A8T_G(W8A8_EPI_F32);     break;
        case W8A8_EPI_GLU_F16: W8A8T_G(W8A8_EPI_GLU_F16); break;
        case W8A8_EPI_GLU_F32: W8A8T_G(W8A8_EPI_GLU_F32); break;
        case W8A8_EPI_I32:     W8A8T_G(W8A8_EPI_I32);     break;
        default: return cudaErrorInvalidValue;
    }
#undef W8A8T_G
    return cudaGetLastError();
}
