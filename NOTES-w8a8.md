# NOTES-w8a8: the "W8 x A8-sub32" int8 tensor-core matmul for Qwen-Image 2.1 (worktree spellcast-w8a8)

Brief: scratchpad qs4/brief_w8a8.md. Worktree C:\_myDrive\repos\spellcast3d\temp_code\ggml-w8a8, branch spellcast-w8a8
from spellcast-perf 6d8cc278. Build: build-w8a8.cmd (copy of ggml-mmq3/build-mmq3.cmd: standalone ggml, sm_86) ->
build-sm86\bin. Stand-alone tests: w8a8-tools/ (build_tool.cmd <name> [out]; extra nvcc flags in NVCC_EXTRA; the same
nvcc flags as ggml-cuda: -O3 -use_fast_math -arch=sm_86). Every card run through the GPU lock
(python scratchpad/qs3/lockrun.py "<purpose>" <min> -- <cmd>).
Coordinator update 2026-10-23: the W8g256 variant (section 6 step 4) is dropped; the format stays section 1 exactly.
FORMAT CHANGE (review, 2026-10-23 afternoon, task 3 at the end): the sub-scales are 5 bits now, M = W8A8_SY_MAX = 31
(sy = max(1, min(31, ceil(31*g/amax))), d_g = d_t*sy/31, the epilogue's d_t/31): with 8-bit sub-scales the int32
accumulator came within 3.2x of 2^31 on real to_out activations. Everything above the "Task 3" section was measured
with M = 255 / GGML_CUDA_W8A8=sub:8 (history); the current reference is GGML_CUDA_W8A8=sub:5.

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

## Kernel-only times (nsys, w8a8-tools/kt.py on w8a8bench, 11 calls each, medians; the path's real inputs: Q4_K/Q5_K
## weights from N(0, 0.02), activations N(0, 1); MMQ = GGML_CUDA_W8A8_KERNEL=off on the same build)
| product (T = 4096)               | W8A8 GEMM          | + conversion | + quantizer      | MMQ kernel       | + q8_1 quantize |
| mm q4_K 4096x4096                | 1799.3 us 76.4 T/s | 108.2 us     | 253.0 us (f32)   | 2803.8 us 49.0   | 257.4 |
| mm q5_K 4096x4096                | 1798.0 us 76.4     | 123.6        | 253.2            | 3027.4 us 45.4   | 257.2 |
| mm q4_K K 12288 N 4096           | 5251.3 us 78.5     | 358.4        | 756.3 (f32)      | 8103.9 us 50.9   | 802.2 |
| glu: gate|up N 24576 (SwiGLU f16)| 10201.2 us 80.8    | 609.9 (rows interleaved) | 253.1 | 17156.1 (GLU kernel, q8_1 out) | 258.1 |
|      down K 12288 of it          | 5386.0 us 76.6     | 359.5        | 457.0 (f16 in)   | 8114.0           | (in the GLU kernel) |
| SCALE x/32 + mm q5_K             | 1798.1             | 123.6        | 252.9 (pre-op)   |                  |       |
GEMM T/s = 2 T N K / kernel time. Per call incl. conversion and quantization: 4096^2 q4_K 2.160 vs 3.061 ms (-29 %);
q5_K 2.175 vs 3.284 (-34 %); down 6.365 vs 8.906 (-29 %); GLU chain 17.27 vs 25.53 ms (-32 %).
(Random uniform int8 in test_gemm gives 1.714 ms for the 4096^3 GEMM: the data changes the card's power / clock.)

## Release architectures (w8a8-tools/relcheck.py: the changed files w8a8-mm.cu, ggml-cuda.cu, w8a8-emu.cu compiled with
## build-sm86's exact ggml-cuda flags and --generate-code=arch=compute_XX,code=[compute_XX,sm_XX] for each of
## 61 70 75 80 86 89 90 100 120)
All 27 compiles OK. cuobjdump of w8a8-mm per arch: sm_61 / 70 / 75: every w8a8 kernel a stub (6-8 instructions: the
__CUDA_ARCH__ >= 800 guard); sm_80 / 86 / 89 / 90 / 100 / 120: the real kernels (GEMM 1960-2184 instructions, 64 IMMA in
the loop). The host side takes the path only when ggml_cuda_highest_compiled_arch(cc) >= 800 and cc >= 800, so a
Pascal / Volta / Turing card keeps MMQ.

## Pictures (code/devtools/qwen-speed/suite.py, --extra=--sage-attn, fusion on, bin = sd.cpp build-sm86 bin with this
## branch's ggml-cuda.dll; tags w8a8k_f16 (default) and w8a8k_f32 (GGML_CUDA_W8A8_GLU=f32))
PSNR rgb dB (SSIM, LPIPS)    chest            knight           potion           fox              knight_back      ork_right
w8a8k_f16 vs sub8_nofuse     35.04 (.971 .033) 44.07 (.997 .003) 36.28 (.993 .005) 32.78 (.987 .010) 35.49 (.993 .005) 30.88 (.971 .016)
w8a8k_f32 vs sub8_nofuse     38.23 (.984 .020) 43.42 (.997 .003) 35.94 (.991 .008) 25.97 (.961 .030) 28.82 (.972 .018) 39.80 (.996 .002)
w8a8k_f16 vs w8a8k_f32       38.82            43.21            32.67            25.00            28.67            30.26
w8a8k_f16 vs ctrl_nofuse     31.02 (.958 .042) 39.78 (.994 .007) 37.34 (.993 .006) 19.55 (.883 .098) 24.87 (.943 .050) 31.54 (.973 .014)
w8a8k_f32 vs ctrl_nofuse     31.29            38.44            34.29            19.12            23.68            31.46
sub8_nofuse vs ctrl_nofuse   31.79 (.962 .037) 38.81 (.993 .007) 37.00 (.993 .006) 19.54 (.883 .098) 25.15 (.945 .049) 32.36 (.975 .013)
(the emulation's own numbers, for the last three rows; the rounding-only yardstick of round 3: chest 35.15, fox 23.58)
Steady step (sd-cli, 1024x1024, 25 steps): w8a8k_f16 1.11-1.12 s/it text only, 1.31-1.32 with an example picture;
w8a8k_f32 1.16 / 1.34-1.35; today's engine (release2 tag of the coordinator, fusion on) 1.44-1.47; ctrl_nofuse 1.68-1.72.

## ggml-level verification, more token counts (w8a8-tools/verify2.sh; same columns as verify1)
  glu T 8360: f16 2.299e-03; f32 = fusion off (hash 6fb89991c41834f1) 2.087e-04; MMQ 3.226e-02
  glu T 1811: f16 2.290e-03; f32 = fusion off (hash 97bd0a545896db9e) 2.120e-04; MMQ 3.209e-02
  mm q4_K K 12288 T 8360: 1.970e-06 | 2.84e-05 | 1.1e-04;  T 1811: 1.970e-06;  q5_K K 12288 T 4145: 1.396e-06
  times (ggml level): glu T 8360 35.09 ms 72.0 T/s (f32 37.48, fusion off 43.32, MMQ 52.43); K 12288 T 8360 12.74 ms
  (MMQ 18.31); T 1811 3.27 (MMQ 4.52); glu T 1811 11.12 (MMQ 14.60).
Emulation sanity: GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:8 through THIS branch's DLL renders chest pixel-identical
to the coordinator's sub8_nofuse (PSNR inf, suite tag sub8_redo): the same-DLL comparisons compare with the very
emulation the pictures were judged on.

## Token-count crossover (w8a8-tools/thresh.sh: w8a8bench medians, GGML_CUDA_W8A8_MIN_TOKENS=1 vs KERNEL=off)
T      mm q4_K 4096x4096 (W8A8 / MMQ ms)   mm q4_K K 12288 N 4096
128    0.233 / 0.134                         0.696 / 0.319
256    0.290 / 0.231                         0.865 / 0.597
384    0.348 / 0.325                         1.033 / 0.872
512    0.406 / 0.427                         1.197 / 1.212
768    0.523 / 0.597                         1.532 / 1.704
1024   0.676 / 0.831                         1.992 / 2.398
2048   1.177 / 1.576                         3.455 / 4.569
4096   2.065 / 3.069                         6.384 / 8.915
-> W8A8 wins from ~512 tokens (the per-call weight conversion is amortized); the default stays 1024 (clearly ahead,
   and the DiT at 1024x1024 has 4096+ tokens; at 512x512, 1024). (The glu rows of this run were erratic below 1024
   tokens for both paths - e.g. MMQ 28.7 ms at 384 and 5.8 at 768 - and are left out.)

## Device memory (thresh.sh: the device's used memory after the runs minus before the tensors = the tensors + the
## pool's peak + lazily loaded modules; the same tensors for both paths)
                       W8A8      MMQ       difference
glu T 4096             904 MiB   872 MiB   +32 MiB
mm K 12288 T 4096      384       352       +32
3 x mm one src1 T 4096 640       636       +4
glu T 8360             1706      1678      +28
mm K 12288 T 8360      704       674       +30
3 x mm T 8360          1258      1254      +4
Reading: MMQ needs pool memory too (its q8_1 activations, T*K*1.125 bytes, and the GLU kernel's q8_1 output of the
SwiGLU, T*12288*1.125: 56 MB at 4096, 116 MB at 8360); the W8A8 pool peak is the [gate|up] layer's int8 weights
(100.7 MB) + the activations (17 / 36 MB), or the down product's int8 input (52 / 106 MB) + its weights (50 MB): about
the same size, so the route's extra VRAM is ~30 MiB at 4096 and at 8360 tokens. The SwiGLU intermediate lives in the
graph's own (already allocated) SWIGLU tensor.

## The hub on the 8 GB-card emulation (code/devtools/qwen-speed/bench_pair.py w8a8_8gb "--runs 2 --card-gb 8 --verbose"
## <bin_w8a8> "--attention sage --prefix-keep on": the installed engine spellcast-perf-9fcb5c2 (release build), then
## this branch's ggml-cuda.dll in sd.cpp's build-sm86 bin folder, through the hub's own launch)
installed: run 1 40.6 s (prompt 1.15, drawing 37.33, finishing 1.95); run 2 (re-roll) 38.9 s (drawing 36.96)
W8A8:      run 1 35.4 s (prompt 2.79, drawing 29.45, finishing 2.62); run 2 30.2 s (drawing 28.31, finishing 1.81)
-> no memory problem on the 8 GB plan; a re-roll -22 % (drawing -23 %: 1.48 -> 1.13 s a step). Run 1's prompt (2.79
   vs 1.15 s) and finishing (2.62 vs 1.95 s) are not the path's work (the text encoder and the VAE have no
   transformer_blocks products); NOT investigated (a dev build against the release build; run 2's finishing is equal).

## Per drawing step (code/devtools/qwen-speed/stepprof.py: nsys of sd-cli at 2 and 6 steps, kernel totals subtracted
## / 4; 1024x1024 text only, --sage-attn, fusion on; out: w8a8-tools/out/stepprof_*.txt)
today's engine (sd.cpp build-sm86 bin as is, MMQ): 1458.4 ms of kernels per step
  mul_mat_q_glu_q8_1 Q4_K 545.30 | mul_mat_q Q4_K 531.73 | mul_mat_q Q5_K 87.27 | quantize_mmq_q8_1 (SCALE pre-op) 8.38
  + 0.54 + 0.78 -> the matmul route 1173.2 ms (+ the q8_1 writer inside mul_add_norm_mul_f32<1024,1> 46.79)
W8A8 (this branch): 1120.1 ms of kernels per step
  w8a8_gemm f32 (126 4096^2 products + 32 downs) 405.09 | w8a8_gemm GLU (32 gate|up) 330.61 | conversion 22.16 + 19.79
  (gate|up interleaved) + 3.65 (Q5_K) | quantizers 16.79 (f32: 66 calls = q/k/v once per block + gate|up + 2) + 14.71
  (f16 down inputs) + 8.17 (SCALE pre-op, to_out) -> the route 821.0 ms (+ mul_add_norm_mul_f32 f32 output 54.12, +5.6 ms
  against the q8_1 writer: the norm-writer fusion into this format is the brief's "later optimisation")
-> matmul route 1173.2 -> 821.0 ms (-352 ms, -30 %); the step's kernels 1458.4 -> 1120.1 ms (-23 %); wall: the suite's
   steady 1.44-1.47 -> 1.11-1.12 s/it, the hub's 8 GB re-roll drawing 36.96 -> 28.31 s.
   (The brief's yardstick "1,160 ms of MMQ kernels" was the mmq3 agent's per-call list; measured here 1173.2.)
With an example picture (bench_pair.py w8a8_8gb_ref ... --references knight_front.png; step 0 = 8,360 tokens):
installed: run 1 44.7 s (references 0.71, prompt 1.36, drawing 40.15, finishing 1.82); run 2 41.5 s (drawing 38.88)
W8A8:      run 1 35.8 s (references 0.65, prompt 1.28, drawing 31.51, finishing 1.81); run 2 32.9 s (drawing 30.33)
-> -20 % / -21 %, the 8 GB plan holds at 8,360 tokens.

## Rounding-only control of the EMULATION itself (this branch's w8a8-emu.cu: GGML_CUDA_W8A8_EMU_KSPLIT=2 computes the
## same f32 SGEMM as two half-K GEMMs accumulated in dst: the same values, another f32 summation order; suite tag
## sub8_ksplit2, GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:8, 6 pictures)
PSNR rgb dB                  chest  knight potion fox    knight_back ork_right
sub8_ksplit2 vs sub8_nofuse  36.38  43.84  41.21  33.76  28.72       36.27     <- the emulation against itself
w8a8k_f16 vs sub8_nofuse     35.04  44.07  36.28  32.78  35.49       30.88
w8a8k_f32 vs sub8_nofuse     38.23  43.42  35.94  25.97  28.82       39.80
w8a8k_f16 vs sub8_ksplit2    42.31  44.38  37.31  36.04  28.79       32.43
w8a8k_f32 vs sub8_ksplit2    41.99  43.93  34.83  25.31  34.78       35.04
sub8_ksplit2 vs ctrl_nofuse  31.09  38.97  37.18  19.49  23.93       33.44     (sub8_nofuse vs ctrl: 31.79 38.81 37.00 19.54 25.15 32.36)
Reading: a summation-order change inside the emulation moves its own pictures by 28.7..43.8 dB, the same class as the
kernel's 26.0..44.1 dB against it: the brief's "> 40 dB vs sub8_nofuse" is not reachable by any implementation that
does not repeat cuBLAS's exact f32 summation order (ours sums exactly in int32). The kernel's pictures differ from the
emulation's by rounding only, and reproduce its distance to today's pictures (rows "vs ctrl_nofuse" above).

## Final DLL = the measured one: after removing the what-if branches and the per-device attribute fix, the SASS of all 48
## w8a8 kernels in ggml-cuda.dll is identical (cuobjdump, addresses stripped: md5 76bba4c77b77b171540b9612c35fa05d both).

## Final regression with the final DLL (2026-10-23 ~04:30; out/final_*.log)
test_gemm: 0 differences from the int64 reference on all 9 shapes (incl. T 4145 / 1811 / 8360 / 100); test_formats: 0
differences (weights Q4_K/Q5_K, activations synthetic / SCALE / f16 / 2 real dumps); verify1: the same numbers as
before to the last digit (rel L2 vs the emulation 8.1e-07 .. 1.97e-06 on the products, GLU f32 = fusion off 2.1e-04,
f16 2.3e-03). Pictures with the final DLL (tag w8a8k_final: chest, fox): pixel-identical to w8a8k_f16 (PSNR inf) - the
path is deterministic from run to run.

## Status (for the hand-over)
Verified: format bit-identity (weights, activations, sub-scales, row scales, SCALE pre-op, f16 input, GLU interleave)
against verbatim copies of the emulation and ggml's dequantizers; GEMM exactness against int64; ggml-level outputs
vs the emulation in the same DLL at T 1811 / 4096 / 4145 / 8360 on all shapes, the GLU chain, the SCALE fusion, the
shared-src1 fusion, real DiT activations; kernel-only speed (nsys) >= 76 T/s on the three shapes; per-step kernels
(stepprof) 1458 -> 1120 ms; pictures (rounding class against the emulation, the emulation's own summation-order
control in the same class; the emulation's distance to today reproduced); memory (+~30 MiB; the hub's 8 GB plan
text-only and with an example picture); compile for sm_61/70/75/80/86/89/90/100/120 (changed files per arch; the full
ggml-cuda target with the release GPU list: see below).
Not verified / not done: other GPUs than the RTX 3060 (sm_80/89/90/100/120 only compiled, never run); the release build
of stable-diffusion.cpp itself (only this ggml's ggml-cuda target); the norm-writer fusion into this format (the brief's
later optimisation: ~22 ms a step); W8g256 (dropped by the coordinator).
Full ggml-cuda target with the release GPU list (build-relcheck.cmd: CMAKE_CUDA_ARCHITECTURES 61;70;75;80;86;89;90;100;120,
the release's CUDA flags): BUILD OK in 13 min 14 s, no errors (build-relcheck.log); ggml-cuda.dll 341.9 MB with cubins
sm_61 .. sm_120.

## Merge of spellcast-perf (2026-10-23 ~04:40)
spellcast-perf gained the coordinator's research hooks (e43fa740: w8a8-emu.cu/.cuh + GGML_CUDA_ACT_DUMP) after this
branch started; merged into this branch (d47db4e9): the emulation hook once, the W8A8 dispatch right after it, the act
dump hook in the MMQ branch as there; w8a8-emu.cu = spellcast-perf's + the K-split control. Against spellcast-perf the
branch now adds only: ggml-cuda.cu +104 lines, w8a8-emu.cu +15/-2, w8a8.cuh, w8a8-mm.cu/.cuh, notes, tools.
After the merge (rebuilt build-sm86): w8a8bench hashes unchanged - mm q4_K 4096^3 e65cf1a2d9d23f7d, glu 4096
beb8864db98f4545, SCALE+mm q5_K T 4145 8eb416686829375f, 3 x mm T 4145 579847c482e3ba17 (= the verified run).
The sd-cli folder used for every picture / bench: scratchpad w8a8\bin_w8a8 (sd.cpp build-sm86 bin + this ggml-cuda.dll).

# Task 2 (2026-10-23): the activation quantization fused into the norm writer
Spec (coordinator): when the mul + add + norm + mul chain's modulated output feeds only W8A8 products (q/k/v of a block,
or the [gate|up] head of the GLU chain), the fused norm kernel writes q_x / sy / d_t itself with exactly
w8a8_quantize_act's arithmetic, bit-identical to the unfused chain, the f32 output not written.
## What changed (commit b3c1389c)
- src/ggml-cuda/w8a8-format.cuh (new): the tile constants, w8a8_sy_pos, w8a8_pad_tokens (moved from w8a8.cuh) and
  w8a8_act_ptrs, shared by w8a8.cuh and norm.cu. All 48 w8a8 kernels' SASS unchanged (w8a8-tools/sass_cmp.py, old DLL
  of 568feaa7 vs new: 48 equal).
- norm.cu: mul_add_norm_mul_f32<1024, NORM_QLAYOUT_W8A8 = 100> + norm_store_w8a8: the q8_1 variant's pinned
  normalization, the row max over the block, then sy / d / inv / q with the quantizer's expressions (M = 255 passed at
  run time, as the quantizer gets it). The kernel got two trailing parameters (w8a8_act_ptrs, M): the SASS of all 36
  existing norm.cu kernels is unchanged (sass_cmp.py on norm.cu.obj, the new parameters normalized out of the names);
  the new variant uses 40 registers, no stack (sm_86; 40-50 on sm_61..sm_120).
- ggml-cuda.cu: ggml_cuda_w8a8_consumers (ggml_cuda_quantized_consumers' gating: 1..4 MUL_MATs right after the node,
  all of its uses, each taken by the W8A8 path; the tensor F32, contiguous, one channel / sample, not an output, not a
  view) beside the q8_1 test in the mul_add_norm_mul fusion; the GLU-chain test made a function
  (ggml_cuda_w8a8_glu_chain) for both fusions. Products then take the pre-quantized activations as the shared-src1
  fusion does. If anything else reads the norm's output, the fusion does not apply (f32 + quantizer as before).
- w8a8-mm.cu/.cuh: ggml_cuda_w8a8_act_alloc (the last token tile's padding rows zeroed: q_x, d_t and that tile's sy
  block by cudaMemsetAsync before the writer fills rows < T), GGML_CUDA_W8A8_NORM=on|off (new switch, default on),
  GGML_CUDA_W8A8_CHECK_NORM=1 (research: the chain also run unfused - ggml's NORM and MUL ops into their own buffers,
  then the quantizer - and the two activation sets compared byte for byte, padding included; synchronous).
  The existing switches and formats unchanged.
## Bit-identity (w8a8-tools/verify_norm.sh, verify_norm_glu32.sh; w8a8bench graphs normmm3 = h = norm(x + a*b)*m then
## q/k/v of h built one after the other, normglu = h into [gate|up] + SwiGLU + down; random and real activations)
output hashes, fused | GGML_CUDA_W8A8_NORM=off | GGML_CUDA_DISABLE_FUSION=1, and the byte check:
  normmm3 q4_K random T 4096 / 4145 / 8360 / 1811: 3f646541dbf777b4, 65b4b5df86ddbb42, 01628cd2885248e9, dc8eb0310b73c14f
     - all three modes identical, 0 bytes differ
  normmm3 q5_K real (dump 000252) T 4096 / 4145 / 8360 / 1811: 8bde4988bb2702a7, 36cbd7c0ec643778, 5221ceaa24e4d5af,
     1613f08266dc4d27 - identical, 0 bytes
  normglu q4_K (SwiGLU f16): fused = norm off at every T (random d1810745475d66ce, f863ddd3e001e170, d233895a04e75cd9,
     d78b0ce5ee14586e; real 3d31bed76301cbd0, 28c1e11491bd11fd, 98a4122ed0dcc63e, 0e94fc4eba3bb0d2), 0 bytes; fusion
     off differs as expected (the GLU chain then runs unfused with ggml's f32 SWIGLU, not the f16 intermediate);
     with GGML_CUDA_W8A8_GLU=f32 in all modes: all three identical at every T (random 836a7b55343d7c86, e95fbae594035c64,
     feaff4cd81089e02, f595a9494d5c9dea; real bbd18e8212a1ea1b, 808e0ca1ec04ee42, 2225f33ce0754d8a, 612a3e1d9169fe62)
## Speed
nsys per call (kt.py), the norm kernel(s) only: T 4096: fused writer 850.3 / 852.0 us vs f32 writer 854.0 / 849.2 +
quantizer 254.1 / 253.5 us; T 8360: 1727.9 / 1761.5 vs 1721.5 / 1737.1 + 514.9 / 515.1 us -> ~250 us (4096) / ~500 us
(8360) saved per call; the writer itself within -0.4 .. +1.4 % of the f32 writer.
Per drawing step (stepprof.py, same DLL, the switch only): norm writer off 1113.4 ms of kernels (mul_add_norm_mul_f32
<1024,-1> 54.60 + quantize_act<4,float,0> 16.26 ms) -> on 1092.2 ms (mul_add_norm_mul_f32<1024,100> 52.12 + the
unfused remainder <1024,-1> 1.94 + quantize_act 0.73): -21.2 ms a step (the targeted kernels 70.9 -> 54.8 ms, -16.1).
Suite steady (6 prompts): on 1.10-1.11 s/it text only, 1.29-1.30 with an example picture; off 1.13 / 1.31-1.32.
## Pictures: suite tags w8a8k_norm (default) and w8a8k_normoff: all six pixel-identical to w8a8k_f16 (PSNR inf).
## Compile: relcheck.py with norm.cu added: w8a8-mm.cu, ggml-cuda.cu, w8a8-emu.cu, norm.cu x sm_61/70/75/80/86/89/90/
## 100/120: all 36 OK. The new norm variant is plain CUDA (no sm_80 instructions), compiled everywhere; the host launches
## it only when every consumer passes ggml_cuda_w8a8_supported, which requires an sm_80+ card with sm_80+ kernels.
## In-engine byte check on the real model (suite tag w8a8k_normcheck, GGML_CUDA_W8A8_CHECK_NORM=1, chest and
## knight_back): every fused call of the picture compared with the unfused chain (ggml's NORM + MUL, then the quantizer):
chest: 1,409+ calls checked (the last report "check 1409 ... 0 bytes differ (total 0)"); knight_back (example picture):
the same, total 0. Both pictures pixel-identical to w8a8k_f16. (Step 0 with the prefix in the STORE graph slices and
concatenates the modulated tokens, so its norms are not this pattern; the later steps' 64 calls a step are fused.)

# Task 3 (2026-10-23 afternoon): the adversarial review's fixes (commit 43111a85)
1. Accumulator margin: on the real dumps the 8-bit sub-scales brought the int32 accumulator to 669,048,290 = 2^31/3.2
   (to_out, the residual stream's massive channels; the brief's "impossible" bound was wrong). Now 5-bit sub-scales,
   W8A8_SY_MAX = 31 in w8a8-format.cuh, used by the quantizer launcher (w8a8.cuh, a run-time kernel argument as before),
   the norm writer (norm.cu, likewise) and the GEMM epilogue (d_t / 31): |acc| <= 31*127^2*K = 2,047,995,904 < 2^31 at
   K 4096 (q/k/v/out/gate_up: no wrap possible); the down product (K 12288) keeps a 41x measured margin (the review's
   number). The emulation's exact reference: GGML_CUDA_W8A8=sub:5.
   SASS (sass_cmp.py, d38ac9d2's DLL vs the new one, 1476 functions w8a8_ / mul_add_norm / mul_mat_q): only the three
   GEMM epilogue variants differ (w8a8_gemm<0, 1, 2>: the 1/31 constant, the clamp); the quantizer, the converter, the
   norm writer and every MMQ kernel are unchanged (M is a run-time argument).
2. The f16 SwiGLU intermediate clamped to +-65504 before __float2half (a step-0 text-token channel reaches ~3,990;
   an inf there would become a NaN token through the prefix cache). Note: a NaN input becomes -65504 (fminf / fmaxf).
3. The q8_1 shared-src1 fusion asks mmq_get_q8_1_ds_layout only after ggml_cuda_mul_mat_uses_mmq_q8_1 (it aborts on
   F16 / BF16 / F32 weights, e.g. a runtime LoRA's product sharing the input). MMQ results unchanged (verify1's MMQ hashes
   = the first runs': c06cfba24dc30fd0, efadb8b54f9628c3, ...).
4. The W8A8 dispatch admits only the K the quantizer has a kernel for (w8a8_quantize_act_supported: K/1024 in
   1 2 3 4 5 6 8 12 16), instead of every K % 1024 == 0 <= 16384 (7168, 9216, ... would have aborted). Checked:
   mm q4_K K 7168 and K 9216 (T 2048, transformer_blocks names) run MMQ: hashes a7e4dd4b6467fdaa / 303097100b0bda73 =
   the GGML_CUDA_W8A8_KERNEL=off runs', no abort.
5. Comments: "two blocks per SM" is TRUE and stays (the review said one block): w8a8-tools/occ.cu,
   cudaOccupancyMaxActiveBlocksPerMultiprocessor on the RTX 3060 = 2 for all three GEMM variants (255 registers x 128
   threads = 32,768 = half the 65,536-register file; 2 x (49,920 + 1,024) = 101,888 <= 102,400 bytes of shared memory);
   every sm_80+ target has 64K registers per SM and >= 100 KB of shared memory, and the GEMM uses 255 registers on all
   of them (relcheck res-usage). The comments now say why; the stale "512 bytes" (BM 256's sy stage) fixed; the
   carveout comment says what it is for; 8-bit / 255 / sub:8 texts in w8a8.cuh, w8a8-mm.cuh replaced.
## Re-verification (5-bit sub-scales)
- test_formats (bits = 5 in the emulation's verbatim quantizer): 38 of 38 lines with 0 differences (weights Q4_K/Q5_K;
  activations synthetic f32 / SCALE / f16 / 6 real dumps). test_gemm (sy 1..31): 0 differences on all 9 shapes.
- verify1 / verify2 against GGML_CUDA_W8A8=sub:5 (same DLL): products rel L2 8.0e-07 .. 1.97e-06 (mm q4_K 4096^2
  T 4096 1.137e-06, T 8360 1.138e-06, T 4145 8.05e-07, T 1811 1.138e-06, q5_K 1.137e-06, K 12288 1.967e-06, SCALE
  8.05e-07, 3 x mm 8.05e-07, real to_q 8.21e-07, real down 1.356e-06, K 12288 T 8360 / 1811 1.966e-06, q5_K K 12288
  1.39e-06); GLU f16 2.07e-03, GLU f32 = fusion off (bit-identical) 1.88e-04 (real 1.78e-04); MMQ vs sub:5 1.59e-02 ..
  4.0e-02. Hashes (fusion on = off): mm4096 7a9b31ec0f5211c9, mm8360 526839cf860f3fc1, mm4145 6eb456aac43e4fae, mm1811
  1b76f55a1f55be66, q5_K e3edb9d64bf26024, K 12288 b411cd044aed1b46, SCALE 9bf45f4b580af1d9, 3 x mm 0b91937729d117fc,
  glu4096 f16 f8d3242a490c36aa (f32 = fusion off f85e0b0448a13456).
- verify_norm (the fused norm writer, 5-bit): normmm3 fused = norm off = no fusion at T 4096 / 4145 / 8360 / 1811,
  random and real (7825f4d13bcd90f4, 241475e86d5ac296, 2ed4064f70a0ccce, 4d203f1c52ff9520; real 0c6004ccb788a2ae,
  52d744bdccfb4e7e, 21b42be420c87a9e, 0a2bdec1a5fa20d2), 0 bytes differ; normglu fused = norm off, and with
  GGML_CUDA_W8A8_GLU=f32 all three modes identical at every T (random 799d560ca3c8ff1b, aefc030429d91433,
  0c8db733ba5d4c3d, 17b8aa1c7fb199a1; real 8e40232aa34fcb3b, 7d8a06505e3df471, b3cfdae904fac809, 1bb4b45b49ac9f86).
- Release architectures (relcheck.py): w8a8-mm.cu, ggml-cuda.cu, w8a8-emu.cu, norm.cu x 9 archs: 36 OK; the GEMM 255
  registers on sm_80 .. sm_120 (stacks 0-32 B as before), stubs on sm_61 / 70 / 75.

# Turing port, 2026-10-24 (RTX 3060 emulation)
Brief: scratchpad brief_turing.md (session befd2abf). The W8A8 GEMM for cc 7.5 (RTX 20xx / GTX 16xx): sm_75 has int8
tensor cores but neither cp.async nor the m16n8k32 int8 MMA, so w8a8_gemm compiles to an empty stub there and Turing
kept MMQ. No Turing card was available: everything below ran on the RTX 3060 - the kernel's compute_75 PTX through the
driver's JIT (a compute_75-only build) with GGML_CUDA_FAKE_CC=750 for the host dispatch (ours and MMQ's).
CAVEAT: this measures the Turing CODE on Ampere hardware (the same int8 tensor rate per SM per clock as Turing, but
twice the FP32 lanes, a 128 KB L1/shared (100 KB shared) instead of 96 KB (64 KB), a 170 W power cap). It validates
correctness fully and gives a speed PROXY; the comparisons are against MMQ's own Turing path under the same emulation.

## Files
- src/ggml-cuda/w8a8-turing.cuh (new): w8a8t_gemm<epi> + w8a8t_gemm_cuda (w8a8_gemm_cuda's signature). Device code
  only for __CUDA_ARCH__ == 750 (W8A8T_DEVICE_OK; stubs elsewhere); W8A8T_ANY_ARCH (tests only) compiles it for every
  arch >= 7.5 so that one sm_86 binary runs both kernels. w8a8.cuh is unchanged.
- src/ggml-cuda/w8a8-mm.cu / .cuh: w8a8_turing_device(cc) = cc == 750 (NVIDIA) and ggml_cuda_highest_compiled_arch(cc)
  == 750 (sm_75 / compute_75 code in the build) and GGML_CUDA_W8A8_TURING != 0; w8a8_device_ok takes it with the
  shared-memory check (smpbo >= 49,152); w8a8_product launches w8a8t_gemm_cuda; the log line says "int8 tensor cores,
  sm_75 kernel". Default: `constexpr bool W8A8_TURING_DEFAULT = true;` (one line, w8a8-mm.cu). GGML_CUDA_W8A8_KERNEL=off
  still turns the whole path off.
- w8a8-tools/test_gemm.cu: W8A8_KERNEL=turing selects the Turing kernel (check / bench / hash); `cmp` = both kernels
  on the same inputs, every epilogue (f32, GLU f16, GLU f32, i32) x tile orders 0 4 -4 1 3 -2 -8 16, outputs compared
  byte for byte (+ a position-dependent hash; realistic d_t / dW incl. a zero row and large dW for the f16 clamp);
  `hash` = the selected kernel's hashes for comparing binaries. build_tool.cmd: NVCC_ARCH overrides -arch=sm_86.
- w8a8-tools/ubench_imma.cu (m8n8k16 / m16n8k32 peaks), ubench_issue.cu (MMA next to IMAD / IADD / FFMA),
  ubench_icache.cu (loop-size knee).
- stable-diffusion.cpp/build-sm75.cmd: build-sm61.cmd with CMAKE_CUDA_ARCHITECTURES=75, build-sm75, -j 12.

## Design (final: W8A8T_WM 32, 3 slots, 2 pieces, gap 1, L2::128B)
- MMA: per m16n8k32 of the sm_80 kernel (same ldmatrix x4 fragments) four mma.sync.aligned.m8n8k16.row.col.satfinite.
  s32.s8.s8.s32 (exists from sm_75; ggml's mma.cuh maps it the same way): rows gid (a0, a2) and gid+8 (a1, a3), k 0-15
  with b0 and a zero C, then k 16-31 with b1 through C. The 32-value product P is the same exact int32 (|P| <=
  32*127^2, so satfinite can never trigger, as with the sm_80 kernel's zero-C MMA), then the same IMAD c += sy * P.
- Tile: block 128 x 128 x 64 (the sy layout is per 128-token tile), 8 warps of 32 tokens x 64 channels (4 x 2), 256
  threads, 64 accumulators a thread, __launch_bounds__(256, 1). A warp slice's 2*MF = 4 sub-scales of a group are
  consecutive bytes in w8a8_sy_pos order (read as one u32 at w8a8_sy_pos(wm*32 + gid)), the serpentine m order and
  fragment schedule are w8a8_gemm's, the epilogue's expressions are verbatim (MF 2, WM 32).
- Pipeline without cp.async: 3 shared slots of 16 KB (A 8 KB + B 8 KB swizzled as in w8a8_gemm; 49,152 B dynamic,
  the epilogue's 8 x 4,608 B staging reuses it); each thread copies 4 16-byte chunks a stage (w8a8_gemm's chunk layout:
  row (c%2)*64 + tid/4, chunk tid%4) with ld.global.nc.L2::128B into 16 staging registers: stage kt+2 is loaded during
  stage kt (A at pair q 1, B at q 5) and stored during stage kt+1 (q 0 / 4), one barrier per stage at q 7 (before the
  next stage's first fragment loads, w8a8_gemm's place), so a load has a whole stage to arrive and the barrier never
  waits for one. The sub-scales: ld.global.nc.u32 one k step ahead (no shared stage for them).
- Resources (cuobjdump -res-usage of an sm_75 compile, w8a8-tools/test_gemm built with NVCC_ARCH=-arch=sm_75):
  w8a8t_gemm<0..3> REG 234-235, STACK 0, LOCAL 0 (ptxas: 0 bytes spill); k loop (one stage, 2 k steps) 319-323
  instructions = 5 KB: 128 IMMA.8816.S8.S8.SAT, 128-135 IMAD, 12 LDSM, 6 LDG, 4 STS, 1 BAR, 1-7 MOV.
  Occupancy for cc 7.5 (cuda_occupancy.h with 64 KB shared / 64K registers per SM, no per-block reserve - an
  assumption, Ampere reserves 1 KB): 1 block of 8 warps per SM (registers and shared memory both at the limit). The
  64 x 64 alternative (W8A8T_WM=64: 4 warps, 2 slots of 16 KB, sub-scales from global memory) fits 2 blocks (8 warps);
  the sm_80 kernel's own 3 x 16.6 KB would allow 1 block of 4 warps.

## How it got there (test_gemm bench, cmp.sh 3 interleaved rounds, the compute_75 binary through the JIT; T/s of
## [0] 4096^3 f32 / [1] 4096x24576x4096 SwiGLU f16 / [2] 4096x4096x12288 f32; the sm_80 kernel: 80.2 / 78.2 / 76.5)
- First port (the sm_80 kernel's 64 x 64 warp tile, 2 slots, a row per thread copied): 47.3 / 48.6 / 41.2, exact on
  the first try; 255 registers + 32-88 B stack, 45-100 MOVs a stage in the loop (accumulators shuffled).
- w8a8_gemm's chunk layout for the copies (coalesced): 56.6 / 51.3 / 56.4; + the L2::128B hint 59.4 / 53.2 / 59.6;
  prefetch.global.L2 of the next stages 22-38 (rejected); 2 or 8 pieces instead of 4: 38-40 (spills / latency).
- The cost of m8n8k16 itself: w8a8_gemm (cp.async) with its MMA replaced by 4 m8n8k16 drops to 45.4 / 47.8 / 30.8
  (24 LDL + 16 STL a stage). Not the MMA rate: ubench_imma: m8n8k16 102.4 T/s = m16n8k32 102.7 on the 3060; the
  zero-C pair + 1 IMAD per output 96.4. ubench_issue: next to each m8n8k16, 1 IMAD -10 %, 2 -20 %, 3 -42 %; IADD free
  up to 3; FFMA like IMAD. ubench_icache: a 12-16 KB loop -7..12 % against 8 KB (mild). Reading: ptxas hoists the many
  independent m8n8k16 and their products fill the registers -> spills and moves at 128 accumulators a thread.
- 32 x 64 warp tile (64 accumulators): 0 spills, 1-7 MOVs; 2 slots: 53.5 / 59.9 / 55.2 (whole stage per piece);
  3 slots, 2 pieces: 58.5 / 59.3 / 59.3; store-to-load gap 1 pair: 58.6 / 60.2 / 61.2 (final; repeat 57.9 / 60.2 /
  61.2). Load kinds (nc, nc.L2::128B, cg, cg.L2::128B, default, L1::no_allocate): all within +-1.5 %.
- What-ifs with the real data kept in shared memory (no copies in the k loop): 75.3 / 80.0 / 82.1 at 1,927 MHz; no
  epilogue stores: 64.4 / 61.8 / 62.6 (the epilogue does not overlap another block: one block per SM); stores of the
  prologue's data without the loads: 67.9 / 80.1 / 70.8. So the global -> register -> shared copy costs ~25 % on the
  3060. TRAP: a what-if that fills shared memory with constants runs the card cooler and faster (1,965 MHz at ~74 W vs
  1,815 MHz at the 170 W cap): it overstates the ceiling (80.5 T/s at K 12288); keep real data in such what-ifs.
- Tile order (W8A8_GROUP 4 2 1 8 -2 -4 -8 16 -16): 4, 8, -2, -4, -8 within +-2 %; 1 -15..-23 %; w8a8_default_group
  kept (4 for N >= 16384, else -4).

## Exactness (all on the RTX 3060; out files in the session scratch turing/)
- test_gemm against the int64 reference, 15 shapes (T 4096 / 4145 / 1811 / 8360 / 100 / 1, N 4096 / 12288 / 24576 /
  384, K 4096 / 12288 / 1024): 0 differences - the sm_86 binary's Turing kernel, the compute_75 binary's (JIT, the
  engine's code), and w8a8_gemm.
- test_gemm cmp (sm_86 binary, both kernels): 15 shapes x 4 epilogues x 8 tile orders = 480 runs, 0 bytes differ,
  hashes equal. The 64 x 64 variant: 5 shapes x 32 runs, 0 bytes differ.
- Across binaries: the 480 output hashes of the compute_75 binary's Turing kernel = the sm_86 binary's w8a8_gemm's.
- ggml level (w8a8bench through the engine DLLs: weight conversion + quantizer + GEMM + fusions; build-sm75 with
  GGML_CUDA_FAKE_CC=750 vs an sm_86 build of the same sources): 16 cases, all hashes identical - mm q4_K 4096^2 at T
  4096 / 8360 / 4145 / 1811 (7a9b31ec0f5211c9, 526839cf860f3fc1, 6eb456aac43e4fae, 1b76f55a1f55be66), q5_K
  e3edb9d64bf26024, K 12288 b411cd044aed1b46, SCALE + mm 9bf45f4b580af1d9, 3 x mm 0b91937729d117fc, GLU f16
  f8d3242a490c36aa (T 4145 338a1513960e0618), GLU f32 f85e0b0448a13456, normmm3 241475e86d5ac296, normglu
  d81ebe63abc78400, real activations to_q 3fae212fb11cf919, down d28a8ca030879537, GLU a7dd9bb63e9a61dc (the first
  ten = this file's Task 3 hashes).
- The sm_86 SASS of all 53 w8a8_* / mul_add_norm_mul kernels of the sm_86 build with these changes = the earlier
  build-sm86 DLL's (sass_cmp.py): the Ampere path is untouched.
- Release architectures (relcheck.py's procedure on w8a8-mm.cu, the build's flags, 61 70 75 80 86 89 90 100 120):
  9 OK; w8a8t_gemm real only on sm_75 (128 IMMA, 952-1,344 instructions, 0 LDL), 6-16-instruction stubs elsewhere;
  w8a8_gemm real on sm_80+ and stubs below, as before.
- Not verified: on Turing HARDWARE the GLU epilogues' expf and division (fast-math MUFU approximations) might round
  differently from Ampere's; the f32 and i32 epilogues are plain IEEE multiplies (identical by construction).

## Speed under the emulation
Kernel only (nsys, w8a8bench, medians of 12 calls; MMQ = GGML_CUDA_W8A8_KERNEL=off, the same build and FAKE_CC):
  mm q4_K 4096^3: w8a8t_gemm 2,438.9 us (56.4 T/s) vs mul_mat_q 4,007.3 (34.3) -39 %; q5_K 2,439 vs 4,160 -41 %;
  K 12288: 6,792.7 vs 12,220.9 -44 %; GLU chain: gate|up 14,009 (SwiGLU f16 epilogue) vs 23,516 -40 %, down 6,795
  vs 11,615 -41 %; conversion 108-609 us and quantizers 253-756 us as on Ampere; MMQ's q8_1 quantizers 258-1,486 us.
ggml level (w8a8bench median ms incl. conversion + quantization; Turing / MMQ75 / [sm_86 W8A8 / sm_86 MMQ]):
  mm 4096^2 T 4096 2.814 / 3.970 -29 % [2.173 / 2.915]; T 8360 5.609 / 8.749 -36 %; T 4145 2.872 / 4.075 -30 %;
  T 1811 1.407 / 2.080 -32 %; q5_K 2.826 / 4.422 -36 %; K 12288 7.917 / 12.667 -38 %; SCALE + mm 2.811 / 4.219 -33 %;
  3 x mm 10.381 / 14.964 -31 %; GLU T 4096 22.645 / 37.586 -40 %; T 4145 23.452 / 38.518 -39 %; GLU f32 22.188;
  normmm3 12.485 / 16.828 -26 %; normglu 24.725 / 39.968 -38 %; real to_q 2.714 / 4.240 -36 %, down 8.112 / 12.997
  -38 %, GLU 23.384 / 38.547 -39 %. (TRAP: the first call of a kernel after the DLL changed includes its JIT: one
  MMQ q5_K reading of 30 ms was that; a warm cache gives 4.42.)
Per drawing step (stepprof.py, nsys of sd-cli at 2 and 6 steps, chest 1024^2, FAKE_CC=750; tags turing75_w8a8 /
turing75_mmq): kernels 2,686.0 -> 2,093.6 ms (-22 %); the matmul route mul_mat_q 1,441.4 + 116.1 + q8_1 quantizers
55.3 = 1,612.8 ms -> w8a8t_gemm 503.2 + 427.3 + conversion 45.2 + quantizers 23.6 = 999.3 ms (-38 %); the norm writer
46.7 -> 49.4 (its W8A8 form). The flash attention (Turing path) is now half the step: 994 ms.
sd-cli (run_sd.py, chest seed 4242, 4 steps, twice each): steady steps 2.06-2.07 s/it (W8A8) vs 2.69-2.72 (MMQ) -23 %;
sampling 16.90 / 16.99 s vs 19.47 / 19.67 s (step 1 includes ~8 s of loading). The same runs on the native sm_86
build: 1.43-1.44 (W8A8) vs 1.78 (MMQ).

## Pictures (imgcmp: PSNR rgb / alpha; the Turing emulation vs the sm_86 build differ in other kernels too - the
## flash attention takes its Turing path - so pixel identity is not expected)
- 2 steps: W8A8 Turing-emulated vs W8A8 sm_86: 39.02 / 31.80 dB (max abs 255, 52 % of pixels); control MMQ emulated vs
  MMQ sm_86: 41.21 / 32.58; W8A8 vs MMQ under the emulation 36.33 / 29.25 (the format change, as on Ampere).
- 4 steps: W8A8 Turing-emulated vs sm_86 45.10 / 46.84 (max 30); control MMQ vs MMQ 45.36 / 45.85; two runs of each
  emulated configuration pixel-identical (deterministic).

## GTX 16xx (TU116 / TU117: cc 7.5 without the RTX cards' tensor cores)
ggml dispatches by compute capability only: turing_mma_available(cc) is true for every cc 7.5 card, so MMQ runs its
int8 MMA (m8n8k16) path on a GTX 1650 / 1660 too; ggml_cuda_init merely logs "suboptimal performance due to a lack of
tensor cores" for names "NVIDIA GeForce GTX 16*", MX450 and MX550. The Turing W8A8 kernel keeps parity: every cc 7.5
card takes it (same MMA instruction class as MMQ there). Not measured on a GTX 16xx.

## Not verified
Speed and power on a real Turing card (RTX 2060 Super / 2070 / 2080 Ti); a GTX 16xx; the Linux (gcc) compile of the
new code (no Linux nvcc on this host; the sdlinux container was off-limits to this task - the code uses no macros in
#pragma, only standard C++17); the GLU epilogues' MUFU rounding on Turing hardware; the hub's 8 GB plan on Turing (the
path's memory is the Ampere path's: the GEMM needs no device memory beyond shared memory).
TRAP met: another agent's work in progress in w8a8-dp4a.cuh (included by w8a8-mm.cu) broke the build for an hour
(a generated header that did not exist yet); the measured DLLs were built with -DW8A8D_NO_SASS (the dp4a kernel is not
used on cc 7.5 / 8.6). One source tree per concurrent agent avoids this.

## Follow-up: 30 minutes on a rented RTX 2070 / 2080 Ti (Linux pod; build locally in Docker, never on the pod)
Local (Docker, the CUDA 12.8 devel image, a tree of its own - not another agent's container):
  cmake ... -DCMAKE_CUDA_ARCHITECTURES=75 (the pod README's engine build), then in the same container
  nvcc -O3 -std=c++17 -arch=sm_75 -use_fast_math -extended-lambda -I ggml/src/ggml-cuda ggml/w8a8-tools/test_gemm.cu -o test_gemm
  (the same with -DW8A8T_WM=64 -o test_gemm_wm64, and ubench_imma.cu / ubench_issue.cu with -arch=sm_75), w8a8bench.cpp
  against the build's libggml; scp the engine to /workspace/sdcpp-turing, the tools + cmp.sh + the 3060's hash file
  (scratch turing/h_amp86.txt) to /workspace/w8a8t.
On the pod (cd /workspace/w8a8t; CUDA_VISIBLE_DEVICES=0):
  W8A8_KERNEL=turing ./test_gemm                                   # 0 differences on the 9 default shapes, timings
  W8A8_KERNEL=turing ./test_gemm hash 4096 4096 4096 4096 24576 4096 4096 4096 12288 4145 4096 4096 1811 4096 4096 \
      8360 4096 4096 4145 24576 4096 1811 4096 12288 100 384 1024 1 384 1024 1 4096 4096 4145 12288 4096 \
      8360 24576 4096 4096 384 1024 8360 4096 12288 > hash_real.txt  # compare with h_amp86.txt (f32 / i32 must match;
                                                                    # GLU differences = MUFU rounding of the card)
  export W8A8_KERNEL=turing; ROUNDS=3 GLU_EPI=1 ./cmp.sh test_gemm test_gemm_wm64   # the two designs on the card
  ./ubench_imma 1 8; ./ubench_issue 1 8                            # the card's m8n8k16 peak, IMAD interference
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -lms 200 (beside a long bench: the power cap)
  python run_sd.py --bin /workspace/sdcpp-turing --out t/chest4.png --steps 4 [--env GGML_CUDA_W8A8_TURING=0]
      (twice each, alternating; steady s/it; the log must say "int8 tensor cores, sm_75 kernel"; --extra
      "--vae-tiling --vae-tile-size 512" on an 8 GB card), then --steps 25 once each for the drawing time.
Decision rule: keep W8A8_TURING_DEFAULT = true if the drawing step is faster than with GGML_CUDA_W8A8_TURING=0; if the
64 x 64 binary wins on the card, set W8A8T_WM 64 (its defaults follow).
