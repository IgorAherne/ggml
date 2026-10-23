# NOTES-w8a8: the "W8 x A8-sub32" int8 tensor-core matmul for Qwen-Image 2.1 (worktree spellcast-w8a8)

Brief: scratchpad qs4/brief_w8a8.md. Worktree C:\_myDrive\repos\spellcast3d\temp_code\ggml-w8a8, branch spellcast-w8a8
from spellcast-perf 6d8cc278. Build: build-w8a8.cmd (copy of ggml-mmq3/build-mmq3.cmd: standalone ggml, sm_86) ->
build-sm86\bin. Stand-alone tests: w8a8-tools/ (build_tool.cmd <name> [out]; extra nvcc flags in NVCC_EXTRA; the same
nvcc flags as ggml-cuda: -O3 -use_fast_math -arch=sm_86). Every card run through the GPU lock
(python scratchpad/qs3/lockrun.py "<purpose>" <min> -- <cmd>).
Coordinator update 2026-10-23: the W8g256 variant (section 6 step 4) is dropped; the format stays section 1 exactly.

## Files
- src/ggml-cuda/w8a8.cuh: the kernels (self-contained, CUDA headers only; the tests include it): weight conversion
  (Q4_K/Q5_K -> int8 rows + dW, the row max computed in the same kernel: no cache), activation quantizer (f32/f16 in,
  SCALE pre-op), the GEMM (zero-C m16n8k32 MMA + IMAD of the sub-scale, int32 accumulators, f32 / SwiGLU-f16 / int32
  epilogues).
- w8a8-tools/test_gemm.cu: GEMM vs an exact int64 reference (all N of 48 rows incl. the first/last tile rows, all T of
  48 columns incl. the first/last), plus `bench T N K reps` (median of CUDA-event times; W8A8_EPI=1: SwiGLU f16 out).
- w8a8-tools/cmp.sh: 3 interleaved rounds of several builds on the three shapes, median of the rounds (the card
  varies +-5..10 % between single runs: power cap; interleaving makes comparisons repeatable to ~1 %).
- w8a8-tools/ubench_imad.cu: IMMA + IMAD micro-benchmark; census.py (SASS opcode census of the K loop, from mmq3).

## Measurements (RTX 3060, power-capped at ~1.81-1.84 GHz during the GEMMs: nvidia-smi reports throttle reason
## 0x4 SW power cap, ~139 W; int8 dense peak at 1.807 GHz = 103.6 T/s)

### Micro-benchmark (ubench_imad, 28 blocks x 256 threads, 64x64 warp tile, 2 k32 steps per iteration)
0 plain (acc in C) 114.1 / 109.5 T/s | 1 zero-C + 4 IMAD (sy in registers) 106.1 / 104.0 | 3 plain + ldmatrix 103.8 |
5 zero-C + 4 IADD 101.0 | (modes 2 and 4 spilled: 72.8 / 76.7, not meaningful).
-> in isolation the IMAD accumulate costs ~7 % (the clock here is above nominal: short kernel, not power-capped).

### CUTLASS baseline re-measured this session (ggml-mmq3/w8a8/w8a8_cutlass.exe, EVT fused scale, 256x128x64 s3)
4096^3 88.2 T/s | 4096x4096x24576 86.7 | 4096x12288x4096 93.2   (128x256x64 s3: 88.1 / 83.5 / 92.6)

### GEMM versions (test_gemm bench, T = 4096; [0] = N 4096 K 4096, [1] = N 24576 K 4096, [2] = N 4096 K 12288)
v1 (single-buffered fragments, barrier at the stage start): 64.7 / 57.8 / 59.4 T/s, exact on all test shapes.
  what-ifs: plain MMA 72.6/62.2/65.9; constant sy 69.4/60.1/63.7; no global loads 71.7/75.3/75.8.
v2 (CUTLASS order: A+B double-buffered, copy + barrier inside k step 0): plain what-if 86.6/70.2/88.9, but the IMAD
  version spills (255 regs + 120 B) -> 50.2/47.5/38.8.
v3 (A double-buffered, B one pair of n fragments ahead): 68.6/66.1/71.1 (32-48 B spills).
v4 (one fragment set: B one pair ahead, the next k step's A fragments loaded during the last pair, each right after its
  last MMA; one barrier per stage after every read of the slot; S slots resident): no spills in the f32 kernel,
  419 instr per 64 IMMA (IMAD 270, LDSM 16, sy extraction 26, LDGSTS 7). 68.8/61.2/66.3 (GROUP_M 4).
  what-ifs (same session): no global loads 76.7/77.7/78.9; no barrier 75.9/65.0/62.3; no epilogue stores
  76.6/74.6/73.9; plain + no loads 99.1/101.8/104.8; plain with loads 77.8/64.7/75.5.
  IMAD vs alternatives (no global loads; cmp-like same session): IADD3 instead of IMAD 84.5..93/88.5..94/97;
  FADD 85.5/86.5/88.5; FFMA 78.5/80.4/82.2 (= IMAD); FADD+FFMA 69.2/70.5/71.7; IMAD 78.1/78.8/80.8.
  -> the 3-operand multiply-add (IMAD or FFMA) next to the IMMA is what costs, not the instruction count; the float
     route (magic-number conversion through the MMA's C operand + FFMA) is no cheaper, and not exact.
  L2::128B prefetch hint on the copies: no change. Tile order: GROUP_M 2 / GROUP_N 2..16 ~ +3..15 % on [1] vs GROUP_M 4.
v4 + staged epilogue (per m fragment through shared memory, 16-byte streaming stores) + tile options, cmp.sh:
  b256 (256x128 tile, 8 warps, 1 block/SM) GROUP_N 4: 74.1/71.4/73.8; GROUP_M 2: 74.3/71.8/73.8
  b128 (128x128 tile, 4 warps, 2 blocks/SM) GROUP_N 4: 79.6/65.8/75.6; GROUP_M 2: 78.5/73.6/74.4
  ([1] with the SwiGLU f16 epilogue, the real use: GLU_EPI=1)
  what-ifs b256 (no global loads): IMAD 81.5/82.5/85.0; constant sy 87.4/86.7/90.0; plain 100.0/101.6/105.3;
  IADD 91.7/94.1/97.2; no fragment loads (spills) 75.3/78.6/79.5.
  S=4: no gain. PRMT sy extraction: slower (spills). Warp tile 32x64 (16 warps / 128 regs): spills 92-120 B.

NOTE (found 2026-10-23 ~04:00): the lock wrapper started `bash` = WSL's bash (python's subprocess on Windows), which
does not inherit Windows environment variables: every "GLU_EPI=1" cmp run above timed [1] with the f32 epilogue, not
the SwiGLU f16 one. Fixed with w8a8-tools/lk.sh (Git's bash). All [1] numbers above are f32-epilogue numbers.

### Tile order sweep, 128x128 tile (BM 128, 4 warps, 2 blocks/SM), run-time group (raster.sh, 3 interleaved rounds,
### ms per round; [1] with the SwiGLU f16 epilogue = the real gate|up call)
group      4096^3 (ms)          gate|up 4096x24576x4096 GLU   down 4096x4096x12288
 4         1.713 1.720 1.725    10.465 10.553 10.606          5.389 5.413 5.414
 3         1.719 1.720 1.747    10.468 10.546 10.554          5.415 5.437 5.455
 2         1.731 1.731 1.744    10.493 10.496 10.544          5.450 5.509 5.536
 1         2.044 2.066 2.086    10.955 10.981 10.982          6.268 6.298 6.364
-2         1.740 1.747 1.816    10.504 10.550 10.647          5.498 5.520 5.540
-4         1.714 1.715 1.728    10.558 10.567 10.644          5.342 5.387 5.439
-8         1.940 1.940 1.952    11.118 11.227 11.275          5.844 5.845 5.887
-> default: group 4 for N >= 16384, -4 otherwise: 4096^3 1.714 ms = 80.2 T/s; gate|up GLU 10.465 ms = 78.8 T/s;
   down 5.342 ms = 77.2 T/s (GEMM kernels alone, random int8 data, CUDA events).

## Step 1: the format kernels, bit-identical to the emulation (w8a8-tools/test_formats.cu; verbatim copies of the
## emulation's w8a8_rowquant (instrumented with extra stores only) and of ggml's dequantize_q4_K / _q5_K, same flags)
Command: build_tool.cmd test_formats; lk.sh "..." 3 "./test_formats.exe <dumps>" (dumps: qs4/dump_knight/000252
(to_q input, T 260 K 4096), 000255 (to_out input), 000256 (gate_up input), 000257 (img_mlp.out input = SwiGLU output,
K 12288), dump_kback/000377 (block 20 img_mlp.out, example-picture run), 005036 (block 25 to_v, last step)).
- weights Q4_K 4096x4096, 512x12288, Q5_K 4096x4096, 512x12288 (random blocks, realistic f16 d / dmin, a zero row):
  our dequantized values = ggml's on every value (0 differ of 16.8M / 6.3M); int8 0 differ from the emulation; dW 0 differ;
  the glu-interleaved conversion (row n -> 2n / 2(n-N/2)+1) 0 differ.
- activations: int8, sy and d_t 0 differ in every case: synthetic (gaussian + 8 outlier channels x50, a zero row, rows
  x1e-30 and x1e4, groups x1e-4) T 300 / 1811, K 4096 / 12288; the SCALE pre-op (x/32 and 0.7x+0.1, against scale_f32
  then the emulation); the f16 input path (against the emulation of the f16-rounded values); the 6 real dumps, each also
  with the SCALE pre-op and the f16 path. Padding rows of the last token tile: q 0, d_t 0.

## Step 3: integration (src/ggml-cuda/w8a8-mm.cu/.cuh, ggml-cuda.cu)
- ggml_cuda_mul_mat: after the research emulation hook (w8a8-emu.cu copied from the main tree, inactive unless
  GGML_CUDA_W8A8 is set; when it is set the W8A8 path is off), before MMVF/MMF/MMVQ/MMQ: ggml_cuda_w8a8_supported ->
  ggml_cuda_w8a8_mul_mat (quantize src1, convert the weight into pool memory, GEMM).
- Conditions: Q4_K/Q5_K weight in a weight buffer, f32 src1/dst, no batch dims, K % 1024 == 0 and K <= 16384, N % 128 == 0,
  T >= GGML_CUDA_W8A8_MIN_TOKENS (1024), the weight name contains GGML_CUDA_W8A8_NAMES ("transformer_blocks" by default:
  the scope the format was judged on; the Qwen3-VL text encoder with an example picture has 1,119-token Q4_K/Q5_K products
  (qs4/dump_kback/index.txt) that the emulation did not take either), 16-byte aligned rows, an sm_80+ card whose kernels
  were compiled for sm_80+ (ggml_cuda_highest_compiled_arch), GGML_CUDA_W8A8_KERNEL != off.
- ggml_cuda_mul_mat_uses_mmq_q8_1 returns false for these products, so no q8_1 fusion takes them; the path's own
  fusions (ggml_cuda_try_fuse_w8a8, before the q8_1 ones): SCALE + MUL_MAT (to_out's x/32 applied in the quantizer),
  2..4 MUL_MATs of one src1 (q/k/v: one quantization), [gate|up] MUL_MAT -> SWIGLU -> MUL_MAT (the GLU chain: weights
  converted with gate/up rows interleaved, the epilogue writes silu(g)*u into the SWIGLU node's own buffer as f16 or f32
  (GGML_CUDA_W8A8_GLU), the down product quantizes that). The mul+add+norm+mul q8_1 writer falls back to its f32 form.
- The SwiGLU intermediate lives in glu->data (allocated by the graph allocator for the f32 SWIGLU output anyway): no
  extra memory for f16 or f32, so the brief's memory argument for f16 does not apply here; f16 halves the bytes.
- Pool memory per product: int8 weights N*K + 4N bytes (the [gate|up] layer 100.7 MB), activations T_pad*K*(1+1/32) +
  4*T_pad (17.3 MB at T 4096, K 4096; 51.9 MB for the down input K 12288). LIFO frees (the VMM pool is a stack).

## ggml-level verification against the emulation in the same DLL (w8a8-tools/verify1.sh, w8a8bench.exe, compare.py;
## emu = GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:8; k = the path (fusion on); kn = the path, fusion off)
rel L2 of the output vs the emulation | max |diff| / rms | max rel diff where |ref| > 0.1 rms:
  mm q4_K 4096x4096 T 4096  k = kn: 1.144e-06 | 1.68e-05 | 6.4e-05      MMQ: 1.517e-02 | 8.9e-02 | 0.71
  mm q4_K T 8360            1.144e-06 | 1.71e-05 | 6.5e-05             MMQ 1.521e-02
  mm q4_K T 4145            8.14e-07 | 1.01e-05 | 9.2e-05              MMQ 1.515e-02
  mm q4_K T 1811            1.144e-06 | 1.68e-05 | 5.6e-05             MMQ 1.514e-02
  mm q5_K 4096x4096 T 4096  1.144e-06 | 1.71e-05 | 6.3e-05             MMQ 1.524e-02
  mm q4_K K 12288 T 4096    1.971e-06 | 2.67e-05 | 8.2e-05             MMQ 1.557e-02
  SCALE x/32 + mm q5_K T 4145  8.14e-07 | 9.3e-06 | 5.5e-05           MMQ 1.523e-02
  3 products of one src1 (q4_K, T 4145)  8.14e-07 | 1.01e-05 | 9.2e-05
  glu q4_K 4096 24576 T 4096: f16 intermediate (k) 2.297e-03 | 1.55e-02 | 0.10; f32 (k with GLU=f32) = kn (fusion off,
     ggml's SWIGLU op) BIT-IDENTICAL (hash 322c7c88e5c9ead0): 2.123e-04 | 5.7e-03 | 3.7e-02;  MMQ 3.218e-02
  glu T 4145: f16 2.297e-03; f32 = kn 2.125e-04; MMQ 3.218e-02
  real activations (dump 000252 to_q input rows, mm q5_K T 4145): 8.29e-07 | 8.8e-06 | 5.2e-05; MMQ 2.200e-02
  real (000257, down input, K 12288 T 4145): 1.366e-06 | 3.1e-05 | 1.0e-04; MMQ 2.264e-02
  real glu (000256 gate_up input, T 4145): f16 2.377e-03; f32 = kn 2.273e-04; MMQ 3.847e-02
Reading: the products equal the emulation to its own f32 accumulation rounding (our accumulation is exact int32, the
emulation's cuBLAS SGEMM rounds its K-long sums); the GLU chain's 2e-4 is that rounding of the [gate|up] product re-rounded
by the down product's int8 quantizer (a value near a rounding edge moves one step); f16 storage of the SwiGLU output adds
2.3e-3 (an f16 ulp is 1/1024 of the value) - still 14x below the format's own difference from today (MMQ 3.2e-2).
ggml-level times (w8a8bench, incl. weight conversion + activation quantization, median of 5): mm q4_K 4096^3 2.171 ms
63.3 T/s (MMQ 3.067, 44.8); T 8360 4.311 (MMQ 6.270); T 4145 2.225 (MMQ 3.319); T 1811 1.109 (MMQ 1.556); q5_K 2.187 (MMQ
3.291); K 12288 6.391 64.5 T/s (MMQ 8.916); SCALE+mm 2.239 (MMQ 3.531); 3 x mm one src1 8.459 (MMQ 11.747); glu 4096
18.613 66.5 T/s (f32 intermediate 18.736; fusion off 21.425; MMQ 25.523).
