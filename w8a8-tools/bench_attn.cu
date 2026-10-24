// bench_attn.cu: the dp4a flash-attention kernels of fattn-tile-dp4a.cu alone, on synthetic Qwen-size inputs
// (4096 queries x 4145 keys x 32 heads by default, V with outlier and sink tokens so that some blocks take the F32 P.V):
// the kernel time (cudaEvent, median of the repetitions) and the raw output, for A/B builds of the kernel file (byte
// identity with `cmp`, speed). The kernel file is #included (its kernels are static), so a variant is just another copy
// of fattn-tile-dp4a.cu; the binary is ~1.5 MB (quick to upload to a pod) and links the engine's libggml-cuda.so /
// libggml-base.so only for the unused host functions of that file. Linux build (ggml's nvcc flags for sm_61):
//   nvcc -DGGML_BACKEND_BUILD -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_MAX_NAME=160
//     -DGGML_SCHED_MAX_COPIES=4 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -I<ggml>/src/ggml-cuda -I<ggml>/src -I<ggml>/include
//     -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550 -O3 -DNDEBUG -std=c++17
//     "--generate-code=arch=compute_61,code=[compute_61,sm_61]" -use_fast_math -extended-lambda
//     -DKERNEL_FILE='"<path>/fattn-tile-dp4a.cu"' bench_attn.cu -o bench_x -L<build>/bin -lggml-cuda -lggml-base
//     -Xlinker -rpath -Xlinker <engine dir> -Xlinker --allow-shlib-undefined     (one command line)
//   bench_x [--mode f32|int8|int8-f32|int8-all] [--mask] [--nq N] [--nk N] [--nh N] [--gqa R] [--reps N] [--out file]
// (check with cuobjdump -sass that the bench's kernel SASS equals the library's: same flags, same code.)
#include KERNEL_FILE

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char ** argv) {
    int nq = 4096, nk = 4145, nh = 32, gqa = 1, reps = 20, mode = FDP4A_PV_INT8;
    bool use_mask = false;
    const char * out = nullptr;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!strcmp(a, "--mask")) use_mask = true;
        else if (!strcmp(a, "--nq") && i + 1 < argc) nq = atoi(argv[++i]);
        else if (!strcmp(a, "--nk") && i + 1 < argc) nk = atoi(argv[++i]);
        else if (!strcmp(a, "--nh") && i + 1 < argc) nh = atoi(argv[++i]);
        else if (!strcmp(a, "--gqa") && i + 1 < argc) gqa = atoi(argv[++i]);
        else if (!strcmp(a, "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(a, "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(a, "--mode") && i + 1 < argc) {
            const char * m = argv[++i];
            mode = !strcmp(m, "f32") ? FDP4A_PV_F32 : !strcmp(m, "int8-f32") ? FDP4A_PV_INT8_F32 :
                   !strcmp(m, "int8-all") ? FDP4A_PV_INT8_ALL : FDP4A_PV_INT8;
        } else { std::printf("unknown argument %s\n", a); return 1; }
    }
    const int D = FDP4A_D, nkh = nh / gqa;
    const int nq_pad = (nq + FDP4A_NCOLS - 1) / FDP4A_NCOLS * FDP4A_NCOLS;
    const int nk_pad = (nk + FDP4A_NBATCH - 1) / FDP4A_NBATCH * FDP4A_NBATCH;
    const int nblk = nk_pad / FDP4A_PV_BLK;

    // inputs: Q [nh][nq][D] f32, K / V [nkh][nk][D] f16
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hQ(size_t(nh) * nq * D);
    std::vector<half>  hK(size_t(nkh) * nk * D), hV(size_t(nkh) * nk * D);
    for (size_t i = 0; i < hQ.size(); ++i) {
        const int ch = int(i % D);
        hQ[i] = nd(rng) * 2.0f * (ch == 5 || ch == 77 ? 5.0f : 1.0f);
    }
    for (size_t i = 0; i < hK.size(); ++i) {
        const int ch = int(i % D);
        hK[i] = __float2half(nd(rng) * (ch == 5 || ch == 77 ? 4.0f : 1.0f) + 0.5f * float(ch % 7));
    }
    for (size_t i = 0; i < hV.size(); ++i) {
        const int t = int((i / D) % nk);
        const float s = (t == 1 || t == 14 || t == 30) ? 10.0f : (t >= 44 && t <= 48) ? 1.0f / 30.0f : 1.0f;
        hV[i] = __float2half(nd(rng) * s);
    }
    std::vector<half> hM;
    if (use_mask) {                                        // [nq][nk]: 0 or -inf (10 %)
        hM.resize(size_t(nq) * nk);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (auto & m : hM) m = __float2half(u(rng) < 0.1f ? -INFINITY : 0.0f);
    }

    float *Q, *Qs, *Ks, *Km, *Vsc, *Vstat, *dst; half *K, *V, *M = nullptr; int *Q8, *K8, *V8, *Vflag;
    CK(cudaMalloc(&Q, hQ.size() * 4)); CK(cudaMalloc(&K, hK.size() * 2)); CK(cudaMalloc(&V, hV.size() * 2));
    CK(cudaMalloc(&Q8, size_t(nh) * nq_pad * D)); CK(cudaMalloc(&Qs, size_t(nh) * nq_pad * 4));
    CK(cudaMalloc(&K8, size_t(nkh) * nk_pad * D)); CK(cudaMalloc(&Ks, size_t(nkh) * nk_pad * 4));
    CK(cudaMalloc(&Km, size_t(nkh) * D * 4));
    CK(cudaMalloc(&V8, size_t(nkh) * nk_pad * D)); CK(cudaMalloc(&Vsc, size_t(nkh) * nblk * D * 4));
    CK(cudaMalloc(&Vstat, size_t(nkh) * nblk * 3 * 4)); CK(cudaMalloc(&Vflag, size_t(nkh) * nblk * 4));
    CK(cudaMalloc(&dst, size_t(nq) * nh * D * 4));
    CK(cudaMemcpy(Q, hQ.data(), hQ.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(K, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(V, hV.data(), hV.size() * 2, cudaMemcpyHostToDevice));
    if (use_mask) {
        CK(cudaMalloc(&M, hM.size() * 2));
        CK(cudaMemcpy(M, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(dst, 0, size_t(nq) * nh * D * 4));

    const int64_t kb1 = D * 2, kb2 = int64_t(nk) * D * 2, kb3 = int64_t(nkh) * nk * D * 2;
    const int64_t qb1 = D * 4, qb2 = int64_t(nq) * D * 4, qb3 = int64_t(nh) * nq * D * 4;
    const float scale = 1.0f / std::sqrt(float(D));
    flash_attn_dp4a_key_mean<half><<<nkh, 4 * D / 2>>>((const char *) K, Km, nk, nkh, kb1, kb2, kb3);
    flash_attn_dp4a_quantize<half><<<dim3(nk_pad / 8, nkh), 256>>>((const char *) K, Km, K8, Ks, nk, nk_pad, nkh, kb1, kb2, kb3, 1.0f);
    flash_attn_dp4a_quantize<float><<<dim3(nq_pad / 8, nh), 256>>>((const char *) Q, nullptr, Q8, Qs, nq, nq_pad, nh, qb1, qb2, qb3, scale * FDP4A_LOG2E);
    const dim3 block(WARP_SIZE, FDP4A_NWARPS, 1);
    if (mode != FDP4A_PV_F32) {
        flash_attn_dp4a_quantize_v<half><<<dim3(nblk, nkh), block>>>((const char *) V, V8, Vsc, Vstat, nk, nkh, kb1, kb2, kb3);
        flash_attn_dp4a_v_flags<<<nkh, 256>>>(Vstat, Vflag, nblk, nk, 4.0f, mode);
    }
    CK(cudaDeviceSynchronize());
    std::vector<int> hF(size_t(nkh) * nblk);
    if (mode != FDP4A_PV_F32) {
        CK(cudaMemcpy(hF.data(), Vflag, hF.size() * 4, cudaMemcpyDeviceToHost));
    }
    int nflag = 0;
    for (int f : hF) nflag += f;

    const dim3 grid(nq_pad / FDP4A_NCOLS, 1, nh);
    const int64_t sm = use_mask ? nk : 0;
    auto launch = [&]() {
        if (mode == FDP4A_PV_F32) {
            if (use_mask) flash_attn_tile_dp4a<true><<<grid, block>>>(Q8, Qs, K8, Ks, (const char *) V, M, nullptr, dst,
                nq, nq_pad, nk, nk_pad, nh, gqa, kb1, kb2, kb3, sm, 0, nq, 1, 0.0f, 1.0f, 1.0f, 32);
            else flash_attn_tile_dp4a<false><<<grid, block>>>(Q8, Qs, K8, Ks, (const char *) V, nullptr, nullptr, dst,
                nq, nq_pad, nk, nk_pad, nh, gqa, kb1, kb2, kb3, 0, 0, 1, 1, 0.0f, 1.0f, 1.0f, 32);
        } else {
            if (use_mask) flash_attn_tile_dp4a_pv8<true, half><<<grid, block>>>(Q8, Qs, K8, Ks, V8, Vsc, Vflag,
                (const char *) V, M, nullptr, dst, nq, nq_pad, nk, nk_pad, nh, gqa, kb1, kb2, kb3, sm, 0, nq, 1,
                0.0f, 1.0f, 1.0f, 32);
            else flash_attn_tile_dp4a_pv8<false, half><<<grid, block>>>(Q8, Qs, K8, Ks, V8, Vsc, Vflag,
                (const char *) V, nullptr, nullptr, dst, nq, nq_pad, nk, nk_pad, nh, gqa, kb1, kb2, kb3, 0, 0, 1, 1,
                0.0f, 1.0f, 1.0f, 32);
        }
    };
    launch();
    launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t = 0.0f;
        cudaEventElapsedTime(&t, e0, e1);
        ms.push_back(t);
    }
    CK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    std::vector<float> hO(size_t(nq) * nh * D);
    CK(cudaMemcpy(hO.data(), dst, hO.size() * 4, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    int nonfinite = 0;
    for (float v : hO) { if (std::isfinite(v)) sum += v; else ++nonfinite; }
    std::printf("mode %d mask %d nq %d nk %d nh %d gqa %d: flagged blocks %d of %d; median %.3f ms min %.3f max %.3f; checksum %.6e nonfinite %d\n",
        mode, int(use_mask), nq, nk, nh, gqa, nflag, nkh * nblk, ms[ms.size() / 2], ms.front(), ms.back(), sum, nonfinite);
    if (out) {
        FILE * f = std::fopen(out, "wb");
        std::fwrite(hO.data(), 4, hO.size(), f);
        std::fclose(f);
    }
    return 0;
}
