// Issue cost of m8n8k16 int8 MMAs next to other instructions: per MMA (32 independent chains) X extra independent
// integer multiply-adds (IMAD, the FMA pipe) or adds (IADD3 / LOP3, the ALU pipe). The MMA rate stays at the tensor
// peak while the extra instructions fit the issue slots and pipes left over.
// usage: ubench_issue [blocks per SM] [warps per block]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int ITERS = 1024;

template <int X, int KIND>   // KIND 0: IMAD, 1: IADD3-like (xor/add), 2: FFMA
__global__ void kissue(int * out, const int * in, int seed) {
    const int a = in[threadIdx.x & 255] + seed, b = in[(threadIdx.x * 7) & 255] ^ seed;
    int c[16][2] = {};
    int e[8];
    float f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { e[j] = in[(threadIdx.x + j) & 255]; f[j] = (float) e[j]; }
    const int m = seed | 1;
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                         : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a), "r"(b));
#pragma unroll
            for (int x = 0; x < X; ++x) {
                const int q = (j * X + x) & 7;
                if (KIND == 0) {
                    asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(e[q]) : "r"(m), "r"(x + 1));
                } else if (KIND == 1) {
                    asm volatile("add.s32 %0, %0, %1;" : "+r"(e[q]) : "r"(m));
                } else {
                    asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(f[q]) : "f"(1.0001f));
                }
            }
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += c[j][0] + c[j][1];
#pragma unroll
    for (int j = 0; j < 8; ++j) s += e[j] + (int) f[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int X, int KIND>
static void run(int blocks, int threads, int * out, int * in) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(e0));
        kissue<X, KIND><<<blocks, threads>>>(out, in, r);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
    }
    const double macs = (double) blocks * (threads / 32) * ITERS * 16 * (8.0 * 8 * 16);
    printf("%s x %d per MMA: %7.3f ms %6.1f T/s\n", KIND == 0 ? "IMAD" : KIND == 1 ? "IADD" : "FFMA", X, best, 2 * macs / best / 1e9);
}

int main(int argc, char ** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int bps = argc > 1 ? atoi(argv[1]) : 2, wpb = argc > 2 ? atoi(argv[2]) : 4;
    const int blocks = p.multiProcessorCount * bps, threads = 32 * wpb;
    int * out; CK(cudaMalloc(&out, (size_t) blocks * threads * 4));
    int * in; CK(cudaMalloc(&in, 256 * 4)); CK(cudaMemset(in, 1, 256 * 4));
    printf("%s, %d SMs, %d blocks x %d warps\n", p.name, p.multiProcessorCount, blocks, wpb);
    run<0, 0>(blocks, threads, out, in);
    run<1, 0>(blocks, threads, out, in);
    run<2, 0>(blocks, threads, out, in);
    run<3, 0>(blocks, threads, out, in);
    run<1, 1>(blocks, threads, out, in);
    run<2, 1>(blocks, threads, out, in);
    run<3, 1>(blocks, threads, out, in);
    run<4, 1>(blocks, threads, out, in);
    run<1, 2>(blocks, threads, out, in);
    run<2, 2>(blocks, threads, out, in);
    run<3, 2>(blocks, threads, out, in);
    return 0;
}
