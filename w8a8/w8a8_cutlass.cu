// CUTLASS 2.x int8 tensor-op GEMM probes for the W8A8 route on sm_86:
//   D[m][n] = float(sum_k X[m][k] * W[n][k]) * s_tok[m] * s_row[n]   (X: M x K int8 row-major, W: N x K int8 row-major
//   = the GGUF layout, D: M x N f32 row-major)
// variants: EVT (the scaling fused into the epilogue: accumulator * row-broadcast s_row * column-broadcast s_tok),
//           LC   (LinearCombination alpha = 1 to f32, then the scaling kernel of common.cuh),
//           I32  (int32 output, the GEMM alone, for the comparison with cuBLASLt).
// usage: w8a8_cutlass [M K N ...]   (default: the three Qwen-Image shapes)
#include "common.cuh"
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

using namespace cute;

template <int TBM, int TBN, int TBK, int WM, int WN, int Stages, int Swz = 8>
struct Cfg {
    using ElementA = int8_t;  using LayoutA = cutlass::layout::RowMajor;    // X: M x K
    using ElementB = int8_t;  using LayoutB = cutlass::layout::ColumnMajor; // W: N x K row-major = K x N column-major
    using ElementC = float;   using LayoutC = cutlass::layout::RowMajor;    // D: M x N
    static constexpr int AlignA = 16, AlignB = 16, AlignC = 4;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WS = cutlass::gemm::GemmShape<WM, WN, TBK>;
    using IS = cutlass::gemm::GemmShape<16, 8, 32>;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>;

    // fused epilogue: D = (acc * s_row[n]) * s_tok[m]
    using TileMap  = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, WS, ElementC, AlignC, 1>;
    using Accum    = cutlass::epilogue::threadblock::VisitorAccFetch;
    using RowScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<TileMap, float, Stride<_0, _1, int32_t>>;
    using ColScale = cutlass::epilogue::threadblock::VisitorColBroadcast<TileMap, float, Stride<_1, _0, int32_t>>;
    using Mul      = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float,
                                                                    cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0     = cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, RowScale>;
    using EVT1     = cutlass::epilogue::threadblock::Sm80EVT<Mul, EVT0, ColScale>;
    using Store    = cutlass::epilogue::threadblock::VisitorAuxStore<TileMap, float, cutlass::FloatRoundStyle::round_to_nearest,
                                                                     Stride<int64_t, _1, int64_t>>;
    using EVTD     = cutlass::epilogue::threadblock::Sm80EVT<Store, EVT1>;
    using EvtKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC, int32_t, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS, EVTD, Swizzle, Stages,
        cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using EvtGemm = cutlass::gemm::device::GemmUniversalAdapter<EvtKernel>;

    // plain: alpha * acc to f32 / to int32
    using LcGemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, float, LayoutC, int32_t,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS,
        cutlass::epilogue::thread::LinearCombination<float, 4, int32_t, float>, Swizzle, Stages, AlignA, AlignB, false,
        cutlass::arch::OpMultiplyAddSaturate>;
    using I32Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, int32_t, LayoutC, int32_t,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS,
        cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, float>, Swizzle, Stages, AlignA, AlignB, false,
        cutlass::arch::OpMultiplyAddSaturate>;
};

struct Bufs {
    int M, K, N;
    int8_t * X; int8_t * W; float * D; int32_t * Di; float * s_tok; float * s_row;
    std::vector<float> h_tok, h_row;
};

#define CS(x) do { cutlass::Status s_ = (x); if (s_ != cutlass::Status::kSuccess) { \
    fprintf(stderr, "CUTLASS error %s at %s:%d\n", cutlassGetStatusString(s_), __FILE__, __LINE__); return false; } } while (0)

template <class C>
static bool run_cfg(const char * name, Bufs & b, bool do_lc) {
    const double flops = 2.0 * b.M * (double) b.N * b.K;
    const cutlass::gemm::GemmCoord ps(b.M, b.N, b.K);
    char * ws = nullptr;

    // EVT
    {
        typename C::EVTD::Arguments cb{
            {
                {
                    {},                                                // Accum
                    {b.s_row, 0.0f, {_0{}, _1{}, int32_t(b.N)}},       // RowScale (along N)
                    {}                                                 // Mul
                },
                {b.s_tok, 0.0f, {_1{}, _0{}, int32_t(b.M)}},           // ColScale (along M)
                {}                                                     // Mul
            },
            {b.D, {int64_t(b.N), _1{}, int64_t(b.M) * b.N}}            // Store
        };
        typename C::EvtGemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, ps, 1, cb,
            b.X, b.W, nullptr, nullptr,
            int64_t(b.M) * b.K, int64_t(b.N) * b.K, 0, 0,
            int64_t(b.K), int64_t(b.K), 0, 0);
        typename C::EvtGemm gemm;
        CS(gemm.can_implement(args));
        const size_t wsz = C::EvtGemm::get_workspace_size(args);
        if (wsz) CK(cudaMalloc(&ws, wsz));
        CS(gemm.initialize(args, ws));
        CK(cudaMemset(b.D, 0, sizeof(float) * (size_t) b.M * b.N));
        CS(gemm.run());
        CK(cudaDeviceSynchronize());
        const double err = check_sample(b.X, b.W, b.D, b.h_tok, b.h_row, b.M, b.N, b.K);
        const float ms = time_median([&] { gemm.run(); });
        printf("  %-34s EVT (fused scale)  %8.3f ms  %6.1f T/s  %5.1f %% of 102  maxrel %.1e\n", name, ms, flops/ms/1e9,
               flops/ms/1e9/102*100, err);
        if (ws) { CK(cudaFree(ws)); ws = nullptr; }
    }
    if (!do_lc) return true;
    // LC to f32 + scaling kernel
    {
        typename C::LcGemm::Arguments args(ps, {b.X, b.K}, {b.W, b.K}, {b.D, b.N}, {b.D, b.N}, {1.0f, 0.0f});
        typename C::LcGemm gemm;
        CS(gemm.can_implement(args));
        CS(gemm.initialize(args, nullptr));
        CS(gemm.run());
        launch_scale<float>(b.D, b.D, b.s_tok, b.s_row, b.M, b.N);
        CK(cudaDeviceSynchronize());
        const double err = check_sample(b.X, b.W, b.D, b.h_tok, b.h_row, b.M, b.N, b.K);
        const float ms_g = time_median([&] { gemm.run(); });
        const float ms_s = time_median([&] { launch_scale<float>(b.D, b.D, b.s_tok, b.s_row, b.M, b.N); });
        const float ms_b = time_median([&] { gemm.run(); launch_scale<float>(b.D, b.D, b.s_tok, b.s_row, b.M, b.N); });
        printf("  %-34s LC f32 GEMM alone  %8.3f ms  %6.1f T/s | + scale kernel %.3f ms = %.3f ms %6.1f T/s  maxrel %.1e\n",
               name, ms_g, flops/ms_g/1e9, ms_s, ms_b, flops/ms_b/1e9, err);
    }
    // int32 out (GEMM alone)
    {
        typename C::I32Gemm::Arguments args(ps, {b.X, b.K}, {b.W, b.K}, {b.Di, b.N}, {b.Di, b.N}, {1.0f, 0.0f});
        typename C::I32Gemm gemm;
        CS(gemm.can_implement(args));
        CS(gemm.initialize(args, nullptr));
        const float ms = time_median([&] { gemm.run(); });
        const float ms_b = time_median([&] { gemm.run(); launch_scale<int32_t>(b.Di, b.D, b.s_tok, b.s_row, b.M, b.N); });
        launch_scale<int32_t>(b.Di, b.D, b.s_tok, b.s_row, b.M, b.N);
        CK(cudaDeviceSynchronize());
        const double err = check_sample(b.X, b.W, b.D, b.h_tok, b.h_row, b.M, b.N, b.K);
        printf("  %-34s I32 GEMM alone     %8.3f ms  %6.1f T/s | + scale kernel = %.3f ms %6.1f T/s  maxrel %.1e\n",
               name, ms, flops/ms/1e9, ms_b, flops/ms_b/1e9, err);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::vector<int> shapes = {4096, 4096, 4096, 4096, 4096, 24576, 4096, 12288, 4096};
    if (argc > 3) { shapes.clear(); for (int i = 1; i + 2 < argc; i += 3) { shapes.push_back(atoi(argv[i])); shapes.push_back(atoi(argv[i+1])); shapes.push_back(atoi(argv[i+2])); } }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs, clock %d MHz\n", prop.name, prop.multiProcessorCount, prop.clockRate/1000);
    for (size_t s = 0; s < shapes.size(); s += 3) {
        Bufs b; b.M = shapes[s]; b.K = shapes[s+1]; b.N = shapes[s+2];
        std::vector<int8_t> hx((size_t) b.M*b.K), hw((size_t) b.N*b.K);
        fill_i8(hx, 1); fill_i8(hw, 2);
        b.h_tok.resize(b.M); b.h_row.resize(b.N); fill_scale(b.h_tok, 3); fill_scale(b.h_row, 4);
        CK(cudaMalloc(&b.X, hx.size())); CK(cudaMalloc(&b.W, hw.size()));
        CK(cudaMalloc(&b.D, sizeof(float)*(size_t) b.M*b.N)); CK(cudaMalloc(&b.Di, sizeof(int32_t)*(size_t) b.M*b.N));
        CK(cudaMalloc(&b.s_tok, sizeof(float)*b.M)); CK(cudaMalloc(&b.s_row, sizeof(float)*b.N));
        CK(cudaMemcpy(b.X, hx.data(), hx.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.W, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.s_tok, b.h_tok.data(), sizeof(float)*b.M, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.s_row, b.h_row.data(), sizeof(float)*b.N, cudaMemcpyHostToDevice));
        printf("M %d K %d N %d (2MNK = %.3g)\n", b.M, b.K, b.N, 2.0*b.M*(double) b.N*b.K);
        run_cfg<Cfg<128, 128, 64, 64, 64, 3>>("128x128x64 w64x64 s3", b, true);
        run_cfg<Cfg<128, 128, 64, 64, 64, 4>>("128x128x64 w64x64 s4", b, false);
        run_cfg<Cfg<128, 256, 64, 64, 64, 3>>("128x256x64 w64x64 s3", b, true);
        run_cfg<Cfg<256, 128, 64, 64, 64, 3>>("256x128x64 w64x64 s3", b, false);
        run_cfg<Cfg<128, 128, 128, 64, 64, 3>>("128x128x128 w64x64 s3", b, false);
        run_cfg<Cfg<128, 256, 64, 64, 64, 4>>("128x256x64 w64x64 s4", b, false);
        CK(cudaFree(b.X)); CK(cudaFree(b.W)); CK(cudaFree(b.D)); CK(cudaFree(b.Di)); CK(cudaFree(b.s_tok)); CK(cudaFree(b.s_row));
    }
    return 0;
}
