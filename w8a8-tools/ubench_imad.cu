// Cost of the W8 x A8-sub32 accumulate on the int8 tensor cores (sm_80+; measured on the RTX 3060 = sm_86):
// a GEMM-like warp tile of 4 x 8 m16n8k32 fragments (64 x 64 outputs, 128 int32 accumulators per thread), 2 warps
// per SM sub-partition (one 256-thread block per SM), per k32 step:
//   mode 0  plain:    acc = mma(a, b, acc)                                  (the textbook GEMM)
//   mode 1  sub32:    p = mma(a, b, 0); acc += sy(row) * p  (4 IMAD per MMA, sy constant registers)
//   mode 2  sub32 + sy from shared memory per k step (one LDS.64 per thread and k step + byte extraction)
//   mode 3  plain + ldmatrix A/B from shared memory per k step
//   mode 4  sub32 + ldmatrix + sy from shared memory (the full main-loop instruction mix minus global loads)
//   mode 5  zero-C MMA + 4 IADD (asm, so the add is not folded into the MMA): dispatch cost of a cheap ALU op
// usage: ubench_imad [blocks]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma_acc(int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_zero(int (&d)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}
__device__ __forceinline__ void ldsm_x4(int (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ int iadd_asm(int a, int b) {
    int r; asm volatile("add.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}

template <int MODE>
__global__ void __launch_bounds__(256, 1) kb(const int * __restrict__ in, int * __restrict__ out, const int iters) {
    __shared__ __align__(16) int smem[2][4096];     // 2 x 16 KB: an A and a B region for the ldmatrix variants
    __shared__ __align__(16) unsigned char sys[2][8][32][8];
    for (int i = threadIdx.x; i < 2*4096; i += 256) (&smem[0][0])[i] = in[i % 1024] + i;
    for (int i = threadIdx.x; i < 2*8*32*8; i += 256) (&sys[0][0][0][0])[i] = (unsigned char) (1 + (in[i % 1024] & 0x7F));
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int a[4][4], b[8][2];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) a[m][j] = in[(m*4 + j)*32 + lane];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) b[n][j] = in[512 + (n*2 + j)*32 + lane];
    int acc[4][8][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0;
    int sy[4][2];
#pragma unroll
    for (int m = 0; m < 4; ++m) { sy[m][0] = 1 + (in[m] & 0x7F); sy[m][1] = 1 + (in[m + 4] & 0x7F); }
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(&smem[0][0]);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            if (MODE == 3 || MODE == 4) {
                // A: 64 rows x 32 bytes per warp per k step; one x4 per m fragment; B: one x4 per 2 n fragments
                const unsigned ao = sbase + ((warp & 3)*64*64 + ks*32 + (lane % 16)*64 + (lane / 16)*16);
#pragma unroll
                for (int m = 0; m < 4; ++m) ldsm_x4(a[m], (ao + m*16*64) ^ ((lane & 7) << 4));
                const unsigned bo = sbase + 16384 + ((warp / 4)*64*64 + ks*32 + (lane % 8 + (lane / 16)*8)*64 + ((lane / 8) % 2)*16);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    int r[4]; ldsm_x4(r, (bo + n*16*64) ^ ((lane & 7) << 4));
                    b[2*n][0] = r[0]; b[2*n][1] = r[1]; b[2*n + 1][0] = r[2]; b[2*n + 1][1] = r[3];
                }
            }
            if (MODE == 2 || MODE == 4) {
                const uint2 v = *(const uint2 *) &sys[ks][warp][lane][0];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const unsigned w = m < 2 ? v.x : v.y;
                    sy[m][0] = (w >> (16*(m % 2))) & 0xFF;
                    sy[m][1] = (w >> (16*(m % 2) + 8)) & 0xFF;
                }
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    if (MODE == 0 || MODE == 3) {
                        mma_acc(acc[m][n], a[m], b[n]);
                    } else {
                        int p[4];
                        mma_zero(p, a[m], b[n]);
                        if (MODE == 5) {
#pragma unroll
                            for (int j = 0; j < 4; ++j) acc[m][n][j] = iadd_asm(acc[m][n][j], p[j]);
                        } else {
                            acc[m][n][0] += sy[m][0] * p[0];
                            acc[m][n][1] += sy[m][0] * p[1];
                            acc[m][n][2] += sy[m][1] * p[2];
                            acc[m][n][3] += sy[m][1] * p[3];
                        }
                    }
                }
            }
        }
        if (MODE != 3 && MODE != 4) a[0][0] += 1;
    }
    int r = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) r ^= acc[m][n][j] * (1 + m + n + j);
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

static int g_blocks = 28;
template <int MODE>
static void run(const char * name, const int * din, int * dout) {
    const int iters = 4096;
    kb<MODE><<<g_blocks, 256>>>(din, dout, 64);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(e0);
        kb<MODE><<<g_blocks, 256>>>(din, dout, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        best = ms < best ? ms : best;
    }
    const cudaError_t err = cudaGetLastError();
    int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const double mmas = (double) g_blocks * 8 * iters * 2 * 32;
    const double tops = mmas * 4096 * 2 / (best * 1e-3) / 1e12;
    const double cyc_per_mma_smsp = (best * 1e-3) * (clk_khz * 1e3) / (mmas / (nsm * 4.0));
    printf("%-58s %8.3f ms %6.1f T/s  %5.2f cycles/MMA/SMSP @%d MHz (nominal) %s\n", name, best, tops, cyc_per_mma_smsp,
           clk_khz / 1000, err == cudaSuccess ? "" : cudaGetErrorString(err));
}

int main(int argc, char ** argv) {
    if (argc > 1) g_blocks = atoi(argv[1]);
    int * din; int * dout;
    cudaMalloc(&din, 4096 * sizeof(int)); cudaMalloc(&dout, 1 << 22);
    static int h[4096]; for (int i = 0; i < 4096; ++i) h[i] = (int) (0x9E3779B9u * (unsigned) (i + 1));
    cudaMemcpy(din, h, sizeof(h), cudaMemcpyHostToDevice);
    printf("blocks %d x 256 threads, 64x64 warp tile, 2 k32 steps per iteration\n", g_blocks);
    for (int round = 0; round < 2; ++round) {
        run<0>("0 plain (acc in C)", din, dout);
        run<1>("1 sub32: zero-C MMA + 4 IMAD (sy in registers)", din, dout);
        run<2>("2 sub32 + sy from shared per k step", din, dout);
        run<3>("3 plain + ldmatrix A/B per k step", din, dout);
        run<4>("4 sub32 + ldmatrix + sy from shared", din, dout);
        run<5>("5 zero-C MMA + 4 IADD (asm)", din, dout);
    }
    return 0;
}
