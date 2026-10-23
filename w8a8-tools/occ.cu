// occupancy of the W8A8 GEMM kernels (cudaOccupancyMaxActiveBlocksPerMultiprocessor) and their register use
#include "w8a8.cuh"
#include <cstdio>
template <int E> static void report(const char * name) {
    cudaFuncSetAttribute(w8a8_gemm<E>, cudaFuncAttributeMaxDynamicSharedMemorySize, w8a8_smem_bytes());
    cudaFuncSetAttribute(w8a8_gemm<E>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, w8a8_gemm<E>);
    int nb = -1; cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, w8a8_gemm<E>, W8A8_NTHREADS, w8a8_smem_bytes());
    printf("%-10s regs %d, static smem %zu, dynamic smem %d, threads %d: %d block(s) per SM (%s)\n", name, a.numRegs, a.sharedSizeBytes,
           w8a8_smem_bytes(), W8A8_NTHREADS, nb, cudaGetErrorString(e));
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s: regs/SM %d, smem/SM %zu, smem/block optin %zu, reserved/block %zu\n", p.name, p.regsPerMultiprocessor,
           p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.reservedSharedMemPerBlock);
    report<W8A8_EPI_F32>("f32"); report<W8A8_EPI_GLU_F16>("glu f16"); report<W8A8_EPI_GLU_F32>("glu f32");
    return 0;
}
