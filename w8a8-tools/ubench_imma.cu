// Peak int8 MMA throughput of the card for the two PTX shapes: m16n8k32 (sm_80+) and m8n8k16 (sm_75+, what the Turing
// W8A8 kernel and ggml's Turing MMQ issue), plus m8n8k16 with one IMAD per output (the zero-C + sub-scale pattern).
// Each warp runs independent accumulator chains (no memory traffic); T/s = 2 * MACs / time.
// usage: ubench_imma [blocks per SM] [warps per block]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int ITERS = 4096;
constexpr int CH = 8;   // independent chains per warp

__global__ void k16832(int * out, int seed) {
#if __CUDA_ARCH__ >= 800
    int a[4] = {seed, seed + 1, seed + 2, seed + 3}, b0 = seed * 3, b1 = seed * 5;
    int c[CH][4] = {};
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int j = 0; j < CH; ++j) {
            asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < CH; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
#endif
}
__global__ void k8816(int * out, int seed, int imad) {
    int a = seed, b = seed * 3;
    int c[4 * CH][2] = {};
    int sv = seed & 31;
    if (!imad) {
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
            for (int j = 0; j < 4 * CH; ++j) {
                asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                             : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a), "r"(b));
            }
        }
    } else {
        // the Turing W8A8 pattern: two chained MMAs from zero C per 32-value group, then sy * P into the accumulator
        for (int it = 0; it < ITERS / 2; ++it) {
#pragma unroll
            for (int j = 0; j < 4 * CH; ++j) {
                int p0, p1;
                asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%4};"
                             : "=r"(p0), "=r"(p1) : "r"(a), "r"(b), "r"(0));
                asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                             : "+r"(p0), "+r"(p1) : "r"(a + j), "r"(b));
                c[j][0] += sv * p0;
                c[j][1] += sv * p1;
            }
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < 4 * CH; ++j) s += c[j][0] + c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char ** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int bps = argc > 1 ? atoi(argv[1]) : 2, wpb = argc > 2 ? atoi(argv[2]) : 4;
    const int blocks = p.multiProcessorCount * bps, threads = 32 * wpb;
    int * out; CK(cudaMalloc(&out, (size_t) blocks * threads * 4));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int mode = 0; mode < 3; ++mode) {
        if (mode == 0) {   // m16n8k32 needs sm_80 code (a compute_75-only binary has an empty kernel)
            cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k16832));
            if (p.major < 8 || fa.ptxVersion < 80) continue;
        }
        float best = 1e9f;
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(e0));
            if (mode == 0) k16832<<<blocks, threads>>>(out, r);
            else k8816<<<blocks, threads>>>(out, r, mode == 2);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        // MACs per warp: mode 0: ITERS*CH*16*8*32; modes 1/2: ITERS*4CH*8*8*16 (mode 2: half the iterations, 2 MMAs each)
        const double macs = (double) blocks * wpb * ITERS * (mode == 0 ? CH * 16.0 * 8 * 32 : 4.0 * CH * 8 * 8 * 16);
        printf("%-34s %8.3f ms  %6.1f T/s (int8 ops)\n", mode == 0 ? "m16n8k32 (C chained)" : mode == 1 ? "m8n8k16 (C chained)" :
               "m8n8k16 x2 zero-C + IMAD per output", best, 2 * macs / best / 1e9);
    }
    printf("%s, %d SMs, %d blocks x %d warps\n", p.name, p.multiProcessorCount, blocks, wpb);
    return 0;
}
