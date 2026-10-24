// Register banks on Pascal (GTX 10xx): a register-only 8x8 dp4a outer product - 8 a's, 8 b's, 64 accumulators, 512 dp4a
// per loop iteration - measured in warp-instructions per clock per SM with clock64 (one wave: 2 blocks of 256 threads
// per SM), so the result does not depend on the clock the power cap sets. ptxas's own code runs at 99.5 % of the dp4a
// issue rate; sass_banks.py synth makes variants of its loop with chosen registers / reuse flags (NOTES-w8a8-dp4a.md,
// "Register banks").
// build (Linux; Windows: the same nvcc line under vcvars64):
//   nvcc -O3 -arch=sm_61 bank_bench.cu -o bank_bench && nvcc -O3 -arch=sm_61 -cubin bank_bench.cu -o bank_bench.cubin
//   python sass_banks.py synth bank_bench.cubin -o banks/
// usage: bank_bench [cubin|-] [iters]   (a cubin: a variant of this program's kernel, loaded with cudaLibraryLoadData)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ void __launch_bounds__(256, 2) outer8(int * out, long long * clk, int iters, const int * __restrict__ src) {
    const int tid = threadIdx.x;
    int a[8], b[8], p[64];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = src[(tid + i) & 1023]; b[i] = src[(tid * 3 + i + 300) & 1023]; }
#pragma unroll
    for (int i = 0; i < 64; ++i) p[i] = src[(tid * 5 + i + 600) & 1023];
    __syncthreads();
    const long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int rep = 0; rep < 8; ++rep)
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) p[i * 8 + j] = __dp4a(a[i], b[j], p[i * 8 + j]);
    }
    const long long t1 = clock64();
    int s = 0;
#pragma unroll
    for (int i = 0; i < 64; ++i) s ^= p[i] * (i + 1);
    out[blockIdx.x * blockDim.x + tid] = s;
    if (tid % 32 == 0) clk[(blockIdx.x * blockDim.x + tid) / 32] = t1 - t0;
}

int main(int argc, char ** argv) {
    const char * cub = argc > 1 && argv[1][0] != '-' ? argv[1] : nullptr;
    const int iters = argc > 2 ? atoi(argv[2]) : 4000;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount, blocks = 2 * sms, threads = 256, warps = blocks * threads / 32;
    int * out; long long * clk; int * src;
    CK(cudaMalloc(&out, blocks * threads * 4)); CK(cudaMalloc(&clk, warps * 8)); CK(cudaMalloc(&src, 4096));
    std::vector<int> h(1024);
    for (int i = 0; i < 1024; ++i) h[i] = (int) (0x9E3779B9u * (i + 1));
    CK(cudaMemcpy(src, h.data(), 4096, cudaMemcpyHostToDevice));
    cudaKernel_t kern = nullptr;
    if (cub) {
        FILE * f = fopen(cub, "rb"); if (!f) { printf("no %s\n", cub); return 1; }
        std::vector<char> d; char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n); fclose(f);
        cudaLibrary_t lib;
        CK(cudaLibraryLoadData(&lib, d.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        CK(cudaLibraryGetKernel(&kern, lib, "outer8"));
    }
    std::vector<double> ipc;
    unsigned ref = 0;
    for (int r = 0; r < 7; ++r) {
        void * args[] = {&out, &clk, (void *) &iters, &src};
        if (kern) CK(cudaLaunchKernel((const void *) kern, blocks, threads, args, 0, 0));
        else CK(cudaLaunchKernel((const void *) outer8, blocks, threads, args, 0, 0));
        CK(cudaDeviceSynchronize());
        std::vector<long long> c(warps); CK(cudaMemcpy(c.data(), clk, warps * 8, cudaMemcpyDeviceToHost));
        std::vector<int> o(blocks * threads); CK(cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost));
        unsigned hsh = 0; for (int v : o) hsh = hsh * 31 + (unsigned) v;
        if (r == 0) ref = hsh; else if (hsh != ref) printf("results differ between runs!\n");
        const long long cmax = *std::max_element(c.begin(), c.end());
        // per SM: 16 warps x 512 dp4a x iters in cmax clocks (the slowest warp)
        ipc.push_back(16.0 * 512 * iters / (double) cmax);
    }
    std::sort(ipc.begin(), ipc.end());
    printf("%s: dp4a warp-instr/clk/SM median %.3f (%.1f %% of 4), min %.3f max %.3f, result hash %08x\n", cub ? cub : "compiled",
           ipc[ipc.size() / 2], ipc[ipc.size() / 2] / 4 * 100, ipc.front(), ipc.back(), ref);
    return 0;
}
