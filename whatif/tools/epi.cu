// Throughput of the int8 m16n8k32 IMMA mixed with the epilogue instructions of the MMQ kernels (RTX 3060, sm_86).
// Each warp runs NCH independent chains; per chain and iteration: one IMMA (fresh accumulator) and the epilogue ops on
// its 4 outputs. Reports T(int8 MAC*2)/s and cycles-equivalent per IMMA per SM sub-partition.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ void imma0(int (&c)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}

constexpr int NCH = 4;

// MODE: 0 IMMA only (accumulate into int, one IADD per output: needed to keep results alive)
//       1 ring epilogue: 4 IDP4A + 4 IMAD per IMMA, + (I2F, FMUL, FFMA) per output every 4th IMMA
//       2 async epilogue: per output I2F + FMUL + FFMA
//       3 4 IMAD per IMMA only
//       4 4 IDP4A + 4 IMAD per IMMA only (no float part)
//       5 per output I2F only (+ FADD to keep alive)
//       6 per output FMUL + FFMA from int bits (no I2F: __int_as_float)
//       7 magic conversion: per output IADD + FADD... (int + 0x4B400000 as float - 12582912) + FMUL + FFMA
//       8 ring epilogue with 2 IMAD per output (IMUL sc then IMAD sy) instead of IDP+IMAD (same count)
// MODE 13 / 14: per iteration and chain 4 IMMAs (4 sub-blocks, fresh accumulators), then the int (13: IDP4A + IMAD per
// output and IMMA, one I2F+FMUL+FFMA per output per iteration) or the float (14: I2F + FMUL + FFMA per output and IMMA)
// epilogue; no branches. 4x fewer iterations.
template <int MODE>
__global__ void __launch_bounds__(256, 1) k4(int seed, float * out, int iters) {
    int a[4][4], b[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[j][0] = seed + j; a[j][1] = seed + 1 + j; a[j][2] = seed + 2; a[j][3] = seed + 3 + j;
        b[j][0] = seed * 3 + j; b[j][1] = seed * 5 - j;
    }
    float facc[NCH][4] = {};
    unsigned scb[4] = {(unsigned) seed & 0x3F, ((unsigned) seed & 0x3F) << 8, ((unsigned) seed & 0x3F) << 16, ((unsigned) seed & 0x3F) << 24};
    unsigned sy0 = 0x80706050u + seed, sy1 = 0x60504030u + seed;
    float dx = 1.0f + threadIdx.x * 1e-7f, dy = 0.5f + seed * 1e-7f;
    float dA[4], dB[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { dA[j] = 1.0f + j * 0.1f + seed * 1e-6f; dB[j] = 0.25f + j * 0.01f; }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            int acc[4] = {0, 0, 0, 0};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c[4];
                imma0(c, a[j], b[j]);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    if (MODE == 13) {
                        const int m = (int) __dp4a(scb[j], (l & 1) ? sy1 : sy0, 0u);
                        acc[l] += m * c[l];
                    } else {
                        facc[ch][l] = fmaf(dA[j] * dB[(j + l) & 3], (float) c[l], facc[ch][l]);
                    }
                }
            }
            if (MODE == 13) {
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] = fmaf(dx * (dy + l), (float) acc[l], facc[ch][l]);
            }
            dx += 1e-9f;
        }
        a[0][0] += 1;
    }
    float r = 0;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int l = 0; l < 4; ++l) r += facc[ch][l];
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

template <int MODE>
__global__ void __launch_bounds__(256, 1) k(int seed, float * out, int iters) {
    int a[4] = {seed, seed + 1, seed + 2, seed + 3};
    int b[2] = {seed * 3, seed * 5};
    int   iacc[NCH][4] = {};
    float facc[NCH][4] = {};
    unsigned scb[4] = {(unsigned) seed & 0x3F, ((unsigned) seed & 0x3F) << 8, ((unsigned) seed & 0x3F) << 16, ((unsigned) seed & 0x3F) << 24};
    unsigned sy = 0x80706050u + seed;
    float dx = 1.0f + threadIdx.x * 1e-7f, dy = 0.5f + seed * 1e-7f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ch = 0; ch < NCH; ++ch) {
            int c[4];
            imma0(c, a, b);
            if (MODE == 0) {
#pragma unroll
                for (int l = 0; l < 4; ++l) iacc[ch][l] += c[l];
            } else if (MODE == 1 || MODE == 4 || MODE == 8) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int m;
                    if (MODE == 8) m = (int) (scb[(it + l) & 3] >> 3) * (int) (sy >> (l * 2));
                    else m = (int) __dp4a(scb[(it + ch) & 3], sy + l, 0u);
                    iacc[ch][l] += m * c[l];
                }
                if (MODE != 4 && (it & 3) == 3) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) { facc[ch][l] = fmaf(dx * dy, (float) iacc[ch][l], facc[ch][l]); iacc[ch][l] = 0; }
                }
            } else if (MODE == 2) {
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] = fmaf(dx * (dy + l), (float) c[l], facc[ch][l]);
            } else if (MODE == 3) {
#pragma unroll
                for (int l = 0; l < 4; ++l) iacc[ch][l] += (int) (scb[(it + l) & 3] + 1) * c[l];
            } else if (MODE == 5) {
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] += (float) c[l];
            } else if (MODE == 6) {
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] = fmaf(dx * (dy + l), __int_as_float(c[l]), facc[ch][l]);
            } else if (MODE == 9) { // 4 IMAD not on C (pipe contention only)
#pragma unroll
                for (int l = 0; l < 4; ++l) iacc[ch][l] = iacc[ch][l] * (int) (scb[l] | 1) + (int) sy;
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] += __int_as_float(c[l]);
            } else if (MODE == 10) { // 4 LOP3 not on C
#pragma unroll
                for (int l = 0; l < 4; ++l) iacc[ch][l] = (iacc[ch][l] ^ (int) scb[l]) & (int) sy;
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] += __int_as_float(c[l]);
            } else if (MODE == 11) { // 4 IDP4A not on C
#pragma unroll
                for (int l = 0; l < 4; ++l) iacc[ch][l] = (int) __dp4a(scb[l], sy, (unsigned) iacc[ch][l]);
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] += __int_as_float(c[l]);
            } else if (MODE == 12) { // 4 FADD on C bits only (the baseline of 9..11)
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] += __int_as_float(c[l]);
            } else if (MODE == 7) {
#pragma unroll
                for (int l = 0; l < 4; ++l) facc[ch][l] = fmaf(dx * (dy + l), __int_as_float(c[l] + 0x4B400000) - 12582912.0f, facc[ch][l]);
            }
            dx += 1e-9f;
        }
        a[0] += 1;
    }
    float r = 0;
#pragma unroll
    for (int ch = 0; ch < NCH; ++ch)
#pragma unroll
        for (int l = 0; l < 4; ++l) r += facc[ch][l] + iacc[ch][l];
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

static int g_blocks = 28;
template <int MODE>
void run(const char * name) {
    float * out;
    cudaMalloc(&out, 1 << 24);
    const int blocks = g_blocks, threads = 256, iters = 8192;
    if (MODE >= 13) k4<MODE><<<blocks, threads>>>(1, out, 16); else k<MODE><<<blocks, threads>>>(1, out, 64);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    if (MODE >= 13) k4<MODE><<<blocks, threads>>>(1, out, iters/4); else k<MODE><<<blocks, threads>>>(1, out, iters);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    int clk_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    const double mmas = (double) blocks * (threads / 32) * iters * NCH;
    const double tops = mmas * 4096 * 2 / (ms * 1e-3) / 1e12;
    // cycles per IMMA per sub-partition at the reported clock: (time * clk) / (mmas / (28 SMs * 4 SMSPs))
    const double cyc = (ms * 1e-3) * (clk_khz * 1e3) / (mmas / (28 * 4));
    printf("%-52s %8.3f ms  %6.1f T/s  %5.1f cycles/IMMA/SMSP (at %d MHz)\n", name, ms, tops, cyc, clk_khz / 1000);
    cudaFree(out);
}

int main(int argc, char ** argv) {
    if (argc > 1) g_blocks = atoi(argv[1]);
    printf("blocks %d x 256 threads\n", g_blocks);
    run<0>("0 IMMA (+ IADD per output)");
    run<3>("3 IMMA + 4 IMAD");
    run<4>("4 IMMA + 4 IDP4A + 4 IMAD");
    run<1>("1 IMMA + ring epilogue (4 IDP4A + 4 IMAD + 3/4 float)");
    run<8>("8 IMMA + 4 IMUL + 4 IMAD + 3/4 float");
    run<5>("5 IMMA + 4 I2F (+FADD)");
    run<6>("6 IMMA + 4 FMUL + 4 FFMA (no conversion)");
    run<2>("2 IMMA + async epilogue (4 I2F + 4 FMUL + 4 FFMA)");
    run<7>("7 IMMA + magic conv (4 IADD + 4 FADD) + 4 FMUL + 4 FFMA");
    run<12>("12 IMMA + 4 FADD (C bits)");
    run<9>("9 IMMA + 4 FADD + 4 IMAD independent of C");
    run<10>("10 IMMA + 4 FADD + 4 LOP3 independent of C");
    run<11>("11 IMMA + 4 FADD + 4 IDP4A independent of C");
    run<13>("13 4 IMMA + int epilogue (IDP4A+IMAD per out, 1 conv per 4)");
    run<14>("14 4 IMMA + float epilogue (I2F+FMUL+FFMA per out)");
    return 0;
}
