// Shared-memory load throughput on the GTX 10xx for the patterns of the INT8 flash-attention score loop
// (fattn-tile-dp4a.cu): how many 16-byte shared loads a warp can issue per clock when the lanes read distinct
// chunks, when they all read the same chunk (broadcast), or when 4 / 8 lanes share one, against the dp4a issue rate.
// If a broadcast LDS.128 still costs 4 clocks (one per quarter warp), the score loop - 9 LDS.128 per 32 dp4a - is
// shared-memory bound 4.5 : 1 on Pascal and a register-tiled score loop is the lever.
//   nvcc -O3 -arch=sm_61 -o lds_bench lds_bench.cu && ./lds_bench
// GTX 1070, 2026-10-24: dp4a 3.4 / clock / SM; LDS.128 distinct 4.7 clocks, broadcast (whole warp, per 4, per 8 lanes)
// 2.2; LDS.64 per 8 lanes 0.55; LDS.32 per 8 lanes 0.28, distinct 0.8; the score mix 1.34 dp4a / clock (39 %). The
// "LDS.32, one word for the whole warp" row is NOT trustworthy (it reports more than the issue limit; something is
// optimised away there). The patterns 10 / 11 (narrower key loads) measure nothing new: ptxas merges adjacent
// ld.shared.b32 / .v2.b32 back into LDS.128.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

static constexpr int NT    = 256;       // threads per block (8 warps)
static constexpr int ITERS = 1 << 14;   // loop iterations, 8 loads each
static constexpr int SMEM  = 16384;     // bytes of shared memory used

// pattern: 0 distinct chunk per lane, 1 one chunk for all lanes, 2 one chunk per 4 lanes, 3 one chunk per 8 lanes,
// 4 the score loop's mix (1 distinct + 8 per-4-lanes loads, then 32 dp4a), 5 dp4a only, 6 LDS.32 broadcast
template <int PATTERN>
__global__ void __launch_bounds__(NT) lds_kernel(int * out, const int salt) {
    __shared__ __align__(16) int smem[SMEM / 4];
    for (int i = threadIdx.x; i < SMEM / 4; i += NT) {
        smem[i] = i * 2654435761u + salt;
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int off;
    if (PATTERN == 0 || PATTERN == 4) off = lane * 16;
    else if (PATTERN == 1 || PATTERN == 6) off = 0;
    else if (PATTERN == 2) off = (lane / 4) * 16;
    else off = (lane / 8) * 16;
    const int koff = (lane / 4) * 16;                       // the keys' pattern in the mix
    const int wbase = 0; (void) warp;                       // every warp reads the same lines (conflicts are per warp)
    int acc = salt, a0 = salt, a1 = salt + 1, a2 = salt + 2, a3 = salt + 3, a4 = salt + 4, a5 = salt + 5, a6 = salt + 6, a7 = salt + 7;
    const int q = salt * 7919 + lane;
    const int c0 = salt * 3 + 1, c1 = salt * 5 + 2, c2 = salt * 7 + 3, c3 = salt * 11 + 4, c4 = salt * 13 + 5, c5 = salt * 17 + 6, c6 = salt * 19 + 7, c7 = salt * 23 + 8;
    for (int it = 0; it < ITERS; it++) {
        asm volatile("" ::: "memory");                      // the shared memory may have changed: reload
        const int step = (it * 512) & (SMEM / 2 - 1);       // a different line every iteration
        if (PATTERN == 5) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                a0 = __dp4a(q, c0, a0); a1 = __dp4a(q, c1, a1); a2 = __dp4a(q, c2, a2); a3 = __dp4a(q, c3, a3);
                a4 = __dp4a(q, c4, a4); a5 = __dp4a(q, c5, a5); a6 = __dp4a(q, c6, a6); a7 = __dp4a(q, c7, a7);
            }
        } else if (PATTERN == 6) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                acc ^= smem[(wbase + off + ((step + k * 64) & 1023)) / 4];
            }
        } else if (PATTERN == 4) {
            const int4 qv = *(const int4 *) &smem[(wbase + off + (step & 1023)) / 4];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int4 kv = *(const int4 *) &smem[(wbase + koff + ((step + 64 * k + 128) & 1023)) / 4];
                int & a = k == 0 ? a0 : k == 1 ? a1 : k == 2 ? a2 : k == 3 ? a3 : k == 4 ? a4 : k == 5 ? a5 : k == 6 ? a6 : a7;
                a = __dp4a(qv.x, kv.x, a); a = __dp4a(qv.y, kv.y, a); a = __dp4a(qv.z, kv.z, a); a = __dp4a(qv.w, kv.w, a);
            }
        } else if (PATTERN == 10 || PATTERN == 11) {         // the score mix with the keys' chunks as 4 x 32-bit / 2 x 64-bit loads
            const int4 qv = *(const int4 *) &smem[(wbase + off + (step & 1023)) / 4];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int a = (wbase + koff + ((step + 64 * k + 128) & 1023));
                const unsigned sa = (unsigned) __cvta_generic_to_shared(&smem[a / 4]);
                int kx, ky, kz, kw;
                if (PATTERN == 10) {
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(kx) : "r"(sa));
                    asm volatile("ld.shared.b32 %0, [%1+4];" : "=r"(ky) : "r"(sa));
                    asm volatile("ld.shared.b32 %0, [%1+8];" : "=r"(kz) : "r"(sa));
                    asm volatile("ld.shared.b32 %0, [%1+12];" : "=r"(kw) : "r"(sa));
                } else {
                    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(kx), "=r"(ky) : "r"(sa));
                    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2+8];" : "=r"(kz), "=r"(kw) : "r"(sa));
                }
                int & acc_k = k == 0 ? a0 : k == 1 ? a1 : k == 2 ? a2 : k == 3 ? a3 : k == 4 ? a4 : k == 5 ? a5 : k == 6 ? a6 : a7;
                acc_k = __dp4a(qv.x, kx, acc_k); acc_k = __dp4a(qv.y, ky, acc_k); acc_k = __dp4a(qv.z, kz, acc_k); acc_k = __dp4a(qv.w, kw, acc_k);
            }
        } else if (PATTERN == 7) {                          // LDS.64, one chunk per 8 lanes
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int2 v = *(const int2 *) &smem[((lane / 8) * 8 + ((step + k * 64) & 1023)) / 4];
                a0 = __dp4a(v.x, c0, a0); a1 = __dp4a(v.y, c1, a1);
            }
        } else if (PATTERN == 8) {                          // LDS.32, one word per 8 lanes
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int v = smem[((lane / 8) * 4 + ((step + k * 64) & 1023)) / 4];
                a0 = __dp4a(v, c0, a0);
            }
        } else if (PATTERN == 9) {                          // LDS.32, a distinct word per lane
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int v = smem[(lane * 4 + ((step + k * 128) & 2047)) / 4];
                a0 = __dp4a(v, c0, a0);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int4 v = *(const int4 *) &smem[(wbase + off + ((step + k * 64) & 1023)) / 4];
                a0 = __dp4a(v.x, c0, a0); a1 = __dp4a(v.y, c1, a1); a2 = __dp4a(v.z, c2, a2); a3 = __dp4a(v.w, c3, a3);
            }
        }
    }
    if (acc + a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0x7fffffff) out[0] = acc;   // keep everything live
}

template <int PATTERN>
static float run(int * d_out, int blocks, const char * name, double clock_hz, int sms, double * per_clock) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    lds_kernel<PATTERN><<<blocks, NT>>>(d_out, 1);           // warm-up
    cudaEventRecord(e0);
    lds_kernel<PATTERN><<<blocks, NT>>>(d_out, 2);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double warps = (double) blocks * NT / 32;
    double instr;                                           // warp instructions of the measured kind
    if (PATTERN == 5)      instr = warps * ITERS * 32;      // dp4a
    else if (PATTERN == 4 || PATTERN == 10 || PATTERN == 11) instr = warps * ITERS * 32;   // dp4a (plus the loads)
    else                   instr = warps * ITERS * 8;       // loads
    const double clocks = ms * 1e-3 * clock_hz;
    *per_clock = instr / clocks / sms;
    printf("%-44s %8.2f ms  %6.2f warp-instr / clock / SM", name, ms, *per_clock);
    return ms;
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    const double clock_hz = p.clockRate * 1e3;
    printf("%s: %d SMs, %.0f MHz (reported), smem/block %zu\n", p.name, p.multiProcessorCount, clock_hz / 1e6, p.sharedMemPerBlock);
    int * d_out;
    CHECK(cudaMalloc(&d_out, 4));
    const int blocks = p.multiProcessorCount * 2 * 16;      // 2 blocks per SM resident (16 warps), 16 waves
    double r;
    run<5>(d_out, blocks, "dp4a only (register operands)", clock_hz, p.multiProcessorCount, &r);
    printf("   (4.0 = full rate)\n");
    run<0>(d_out, blocks, "LDS.128, a distinct chunk per lane", clock_hz, p.multiProcessorCount, &r);
    printf("   (0.25 = 4 clocks each)\n");
    run<1>(d_out, blocks, "LDS.128, one chunk for the whole warp", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<2>(d_out, blocks, "LDS.128, one chunk per 4 lanes (8 distinct)", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<3>(d_out, blocks, "LDS.128, one chunk per 8 lanes (4 distinct)", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<6>(d_out, blocks, "LDS.32, one word for the whole warp", clock_hz, p.multiProcessorCount, &r);
    printf("   (1.0 = 1 clock each)\n");
    run<7>(d_out, blocks, "LDS.64, one chunk per 8 lanes", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<8>(d_out, blocks, "LDS.32, one word per 8 lanes", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<9>(d_out, blocks, "LDS.32, a distinct word per lane", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<4>(d_out, blocks, "score mix: 1 + 8 LDS.128 then 32 dp4a", clock_hz, p.multiProcessorCount, &r);
    printf("   (dp4a rate; 4.0 would be compute bound)\n");
    run<10>(d_out, blocks, "score mix, keys as 4 x LDS.32", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    run<11>(d_out, blocks, "score mix, keys as 2 x LDS.64", clock_hz, p.multiProcessorCount, &r);
    printf("\n");
    return 0;
}
