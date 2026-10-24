// Instruction-cache knee of the int8 MMA + IMAD pattern of the Turing W8A8 kernel: the same work (m8n8k16 x2 from a zero
// C per 32-value group, then sy * P into the accumulator) with the k loop body unrolled U times (U x 32 groups = U x 128
// instructions, ~U x 2 KB of SASS). A drop in T/s as U grows = the loop no longer fits the SM partition's L0 i-cache.
// usage: ubench_icache [blocks per SM] [warps per block]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int WORK = 96;   // groups per chain per launch, in units of 32 chains (the same for every U)

template <int U>
__global__ void kloop(int * out, const int * in, int seed) {
    int a[U], b[U];
#pragma unroll
    for (int u = 0; u < U; ++u) { a[u] = in[(threadIdx.x + u) & 255] + seed; b[u] = in[(threadIdx.x + 7*u) & 255] ^ seed; }
    int c[32][2] = {};
    const int sv = seed & 31;
    for (int it = 0; it < WORK * 8 / U; ++it) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                int p0, p1;
                asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%4};"
                             : "=r"(p0), "=r"(p1) : "r"(a[u]), "r"(b[u]), "r"(0));
                asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                             : "+r"(p0), "+r"(p1) : "r"(a[u] + j), "r"(b[u]));
                c[j][0] += sv * p0;
                c[j][1] += sv * p1;
            }
        }
    }
    int s = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) s += c[j][0] + c[j][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int U>
static void run(int blocks, int threads, int * out, int * in) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(e0));
        kloop<U><<<blocks, threads>>>(out, in, r);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
    }
    const double macs = (double) blocks * (threads / 32) * (WORK * 8) * 32 * 2 * (8.0 * 8 * 16);
    printf("U %2d (loop ~%5d instr, ~%3d KB): %7.3f ms %6.1f T/s\n", U, U * 128, U * 128 * 16 / 1024, best, 2 * macs / best / 1e9);
}

int main(int argc, char ** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const int bps = argc > 1 ? atoi(argv[1]) : 2, wpb = argc > 2 ? atoi(argv[2]) : 4;
    const int blocks = p.multiProcessorCount * bps, threads = 32 * wpb;
    int * out; CK(cudaMalloc(&out, (size_t) blocks * threads * 4));
    int * in; CK(cudaMalloc(&in, 256 * 4)); CK(cudaMemset(in, 1, 256 * 4));
    printf("%s, %d SMs, %d blocks x %d warps\n", p.name, p.multiProcessorCount, blocks, wpb);
    run<1>(blocks, threads, out, in);
    run<2>(blocks, threads, out, in);
    run<3>(blocks, threads, out, in);
    run<4>(blocks, threads, out, in);
    run<6>(blocks, threads, out, in);
    run<8>(blocks, threads, out, in);
    run<12>(blocks, threads, out, in);
    return 0;
}
