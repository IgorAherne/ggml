# NOTES-mmq3: faster int8 pipeline for the Q4_K / Q5_K MMA kernel (worktree spellcast-mmq3)

Brief: scratchpad qs3/brief_mmq3.md. Worktree C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3, branch spellcast-mmq3
from spellcast-perf 6d8cc278. Build: build-mmq3.cmd (standalone ggml, sm_86, flags of build-sm86.cmd; ninja from the
pixal3d venv - vcvars alone does not put ninja on PATH here) -> build-sm86\bin (+ cudart/cublas DLLs copied).

Main shapes (coordinator's correction, 2026-10-22: hidden size 4096): mm q4_K 4096 4096 4096, glu q4_K 4096 24576 4096,
mm q4_K 12288 4096 4096, mm q5_K 4096 4096 4096. The 3072 ones are optional; "mm q4_K 3072 3072 4096" hash
191a301da1996ef6 = the setup check.

## Log
- setup: worktree created; first build running.
- setup check: worktree build = build-sm86 hashes on all 5 lines (191a301da1996ef6 for mm q4_K 3072 3072 4096 etc.).
- tooling (scratchpad mmq3/): bench.sh (mb.py on the main shapes), kt.py (nsys kernel-only times under the GPU lock),
  census.py (SASS opcode counts of a kernel's inner K loop), pcmp.py (pipes off/async/ring on one binary: times +
  output diffs), tool/mmq3bench.exe (mmbench + MMBENCH_DUMP_IN input dumps), ref.py / refrun.py (numpy reference:
  exact float64 product, DS4 emulation, DSY emulation, compared to the kernels' dumps).
- NOTE: mmbench's mm time includes the activation quantize kernel (0.26 ms for K=4096, 0.84 ms for K=12288, T=4096).

## Baseline (worktree build, async path), 2026-10-22 evening
mmbench median (incl. quantize) / nsys kernel-only median:
| shape                      | mmbench          | kernel us |
| mm q4_K 4096 4096 4096     | 3.068 ms 44.8 T/s | 2803.5 (49.0 T/s) |
| glu q4_K 4096 24576 4096   | 24.90 ms 49.7 T/s | gate|up 16725.9 + down 7905.5 |
| mm q4_K 12288 4096 4096    | 8.958 ms 46.0 T/s | 8103.7 |
| mm q5_K 4096 4096 4096     | 3.275 ms 42.0 T/s | 3026.1 |
| mm q4_K 3072 3072 4096     | 1.837 ms 42.1 T/s | 1620.7 |

## What-ifs W1..W5 (timing only, wrong results; patch: whatif/W1-W5-whatifs.patch), nsys kernel-only medians, us
| variant                                  | q4K 4096^3 | q4K 12288x4096 | q5K 4096^3 | q4K 3072 | regs |
| W0 async baseline                        | 2803.5 | 8103.7 | 3026.1 | 1620.7 | 255 |
| W1 no epilogue (C accumulates in place)  | 2246.0 (-19.9%) | 6228.6 (-23.1%) | 2417.2 (-20.1%) | 1314.0 (-18.9%) | 194 |
| W2 W1 + constant B / scale fragments     | 2394.4 (W1 +6.6%) | 6521.2 | 2559.8 | 1394.9 | 206 |
| W3 W1 + no global loads after prologue   | 2044.0 (W1 -9.0%) | 5794.8 (-7.0%) | 2164.9 (-10.4%) | 1195.2 (-9.0%) | 186 |
| W4 W1 + 3-deep y ring, 1 barrier/half    | 2201.4 (W1 -2.0%) | 6323.6 (+1.5%) | (Q5_K n/a) | 1276.9 (-2.8%) | 210 |
| W5 W1 + 64x32 warp tile                  | 2274.5 (W1 +1.3%) | 6303.1 (+1.2%) | 2534.5 (+4.9%) | 1329.8 (+1.2%) | 204 |
glu numbers in kt-log.txt; they vary +-5 % run to run under the what-ifs (garbage data changes the card's power/clock).
Reading: the epilogue (12 float ops + 2 scale ops per MMA) costs ~20 %; global loads ~9 % (the deeper ring recovers
only ~2 % of it); constant B fragments are SLOWER than loaded ones (surprise, scheduling); the 64x32 tile is no gain.
W1 census: 4.97 instructions per IMMA in the K loop and still only 61 T/s kernel-only (60 % of peak): the kernel is
latency/stall bound, not issue bound (no Nsight counters here: ERR_NVGPUCTRPERM).

## Design change (coordinator, 2026-10-22): per-128 activations OUT; integer sub-scales (DSY layout)
d128 + 8-bit sy per 32 values (+ the per-32 sums). Epilogue per MMA: 4 IDP4A (sc*sy) + 4 IMAD; per 128 block 3 float
ops per element. Overflow: unsigned Q4_K with 8-bit sy reaches 3.9e9 > 2^31 -> weights centred (q - 8) in the MMA, the
offset 8*d*sc moves into the f16 min-term MMA (m' = 8*d*sc - dmin*m, rounded once to f16). Q5_K: centre 16 and G = 2
(int accumulation over 2 sub-blocks). Static bound checks in mmq-ring.cuh. Unpack cost of centring: 7 ops per raw
register instead of 3 (+1 op per MMA).
Numerics (ref.py, mm q4_K 4096^3, first 256 tokens, all rows): ring vs its DSY emulation rel L2 3.3e-5 (kernel exact
to f32/f16-s rounding); async vs its DS4 emulation 2.3e-5; vs the EXACT float64 product: ring 5.67e-3, async/off
1.24e-2 -> the ring path is 2.2x closer to the exact product (the centring: the sub-block offsets go through the
exact f32 sums).

## The ring kernel (src/ggml-cuda/mmq-ring.cuh), committed 97746b08 as an OPT-IN path
GGML_CUDA_MMQ_PIPE = off | async (default, bit-identical to before on all lines) | ring. DSY layout produced by
quantize_mmq_q8_1 / norm_store_q8_1 (helper ggml_cuda_q8_1_dsy_quantize in mmq.cuh); the layout decision is one
function (ggml_cuda_mmq_y_layout: ring env + Q4_K/Q5_K + J_best == 128 + ring available + x aligned) used by every
producer and by the MUL_MAT (asserts in launch_mul_mat_q / mul_mat_q_switch_J). d128 stored as f32 (the header has
the room: s01, s23, d128, sy) - more exact than the half the coordinator emulated. No GLU-epilogue fusion with DSY
(ggml_cuda_mul_mat_q_glu_supported returns false -> unfused mm + swiglu + mm).
Steps and measurements (mm q4_K 4096^3, mmbench total incl. 0.26 ms quantize; async = 3.069 ms):
- v1 256 thr, 32x64 warp tiles, s pairs in registers across the barrier: 4.73 ms, 255 regs + 184 B stack. Bugs found
  on the way: ldmatrix with a 32-bit shared address but no .shared (illegal address); __dp4a(int,int) is SIGNED
  (sy > 127 negative) -> must be the unsigned overload.
- DSY_MAX 127 (uncentred Q4_K): 4.50 ms, still 232 B stack. B ldmatrix with .shared + register address: same.
- 512 thr, 16x64 tiles: 3.65 ms (128 regs, 72 B stack, 20.9 instr/IMMA).
- min term s pairs via shared memory (ds_h0) instead of registers: 16x128 @256 thr 3.41 ms (16.65 instr/IMMA),
  16x64 @512 3.45, 32x64 @256 3.74. Q5_K needs ds_h0 over the ids area (ring = no MoE): 101,376 B exactly.
- final (16x128, 256 thr): 3.49 ms (ptxas: 255 regs, 56 B stack/spill non-fallback, 88/96 B fallback; Q5_K 160/176;
  shared 93,184 B Q4_K, 101,376 B Q5_K). Census: 16.6 instr per IMMA (IMAD 4.2, IDP 4.0, FMUL/I2FP/FFMA 1.06 each).
Ring what-if decomposition (nsys kernel-only, us, 4096^3 | 12288x4096): R1 no epilogue 2282 | -; R2 + no min term
2134 | 5931; R3 + no barriers/waits 2064 | 4551; R4 + no global loads 1507 | 4234; full ring 3135..3225 | 9048..9365.
async W1 (no epilogue) 2246 vs ring R1 2282: the ring STRUCTURE (3-ring, 1 barrier/half, register scales, ldmatrix
B) is no better than async's; the int epilogue costs ~870-940 us vs async's float epilogue 557 us.
"per-128" int epilogue what-if (m = sc, no IDP; with R4's removals): 2426 us vs R4 1507 -> no cheaper (spills again).

## Micro-benchmark (scratchpad mmq3/ubench/epi.cu, 28 blocks x 256 thr = 2 warps/SMSP, cycles per IMMA per SMSP)
IMMA alone 16.7 (99 T/s); + 4 FADD on C 17.7; + 4 independent IMAD 20.0, LOP3 19.5, IDP4A 19.6;
4 IMMA (sub-blocks) + int epilogue (IDP4A + IMAD per output, 1 conversion per 4) 22.8;
same structure + float epilogue (I2F + FMUL + FFMA per output) 22.1; at 4 warps/SMSP 22.2 vs 22.1.
-> the integer sub-scale epilogue costs the same as the float one: with per-32 activation scales every output of every
MMA needs its own factor (sc_row,j * sy_col,j), i.e. >= 2 ops per output per MMA either way. The int epilogue only
pays off with per-128 activations (factor sc alone), which the picture test rejected.

## Accuracy study (numpy, scratchpad mmq3/refq.py; synthetic N(0,1) activations, Q4_K N(0,0.02) weights, 256 tokens)
rel L2 vs exact float64:      uncentred   centred (q - 8 in the int dot, 8*d*sc - dmin*m via the f16 sums)
  per-32 (DS4, today)          1.28e-2     5.61e-3
  per-64                       1.31e-2     6.20e-3
  per-128                      1.44e-2     6.76e-3
  DSY 8-bit sy                 1.30e-2     5.68e-3
  DSY 7-bit sy                 1.26e-2     5.67e-3
-> centring halves the error of every scheme; per-128 + centring is below today's per-32 error on this data (real
activations have outliers: only a picture test can tell).

## Final tables: see the report (bench-log.txt "FINAL", kt-log.txt "FINAL kernel times")

## FINAL tables (2026-10-22 night; pcmp.py 3 interleaved rounds, mmbench medians incl. the quantize kernel)
| shape                       | off               | async (default)   | ring              | ring vs async |
| mm q4_K 4096 4096 4096      | 3.943 ms 34.9 T/s | 3.069 ms 44.8 T/s | 3.489 ms 39.4 T/s | +13.7 % |
| glu q4_K 4096 24576 4096    | 34.66 ms 35.7     | 25.51 ms 48.5     | 30.24 ms 40.9     | +18.6 % (no GLU fusion with DSY) |
| mm q4_K 12288 4096 4096     | 11.707 ms 35.2    | 8.962 ms 46.0     | 10.227 ms 40.3    | +14.1 % |
| mm q5_K 4096 4096 4096      | 4.112 ms 33.4     | 3.290 ms 41.8     | 3.819 ms 36.0     | +16.1 % |
| mm q4_K 3072 3072 4096      | 2.316 ms 33.4     | 1.836 ms 42.1     | 2.076 ms 37.2     | +13.1 % |
| mm q4_K 3072 12288 4096     | 8.553 ms 36.2     | 6.578 ms 47.0     | 7.508 ms 41.2     | +14.1 % |
| mm q4_K 12288 3072 4096     | 9.062 ms 34.1     | 6.981 ms 44.3     | 7.941 ms 38.9     | +13.8 % |
| glu q4_K 3072 24576 4096    | 26.67 ms 34.8     | 19.48 ms 47.6     | 23.39 ms 39.7     | +20.1 % |
| mm q5_K 3072 3072 4096      | 2.398 ms 32.2     | 1.961 ms 39.4     | 2.287 ms 33.8     | +16.6 % |
async hashes = the baseline's on every line (default path unchanged). Kernel-only (nsys, us) async -> ring:
4096^3 2803.8 -> 3225.5 (+15.0 %); 12288x4096 8104.3 -> 9365.4 (+15.6 %); q5_K 4096^3 3026.9 -> 3551.3 (+17.3 %);
3072^3 1621.2 -> 1860.3 (+14.7 %).
Errors (ref.py / refglu.py; rel L2): vs exact float64: off = async 1.26e-2..1.30e-2, ring 5.60e-3..5.68e-3 (glu chain:
2.77e-2 vs 1.26e-2); ring vs its own DSY emulation 2.1e-5..3.5e-5 (async vs its DS4 emulation 3.7e-5..5.4e-5) on
q4_K/q5_K 4096^3, 4096x3080x4100 (fallback rows, partial column tile), 640x640 (stream-k + fixup), 12288x4096x512;
T = 100 (J < 128): all three pipes give the same hash (the ring/DSY is not selected). off vs async 3.1e-7..7.2e-7.

## ringf: the ring pipeline with async's per-32 activations and float epilogue (whatif/ringf_float_epilogue_on_the_ring.patch, NOT committed)
GGML_CUDA_MMQ_PIPE=ringf: outputs BIT-IDENTICAL to async (hash c06cfba24dc30fd0 q4_K 4096^3, 13b791ed018d1697 q5_K):
an independent check of the ring's pipeline (slots, x double buffer, ds_h0, min term). Times (mm q4_K 4096^3, async
3.069 ms): 16x128 @256 thr 3.790 (248 B stack), 16x64 @512 3.544 (128 regs, 80 B stack), 32x64 @256 4.148. Its
column-tile loop (n outer, 4 sub-blocks inner, all 4 B fragments and 8 d_y live) was built for the int accumulation
chain; async's order (sub-block outer, n inner) keeps fewer registers live - not tried on the ring.
Commits: 97746b08 (pipe selector + ring + DSY quantizer, opt-in), 8100c1f5 (sm_120 static shared memory margin).
Compiled the changed files for sm_61/75/80/89/120 (CUDA 12.8): no errors; the ring kernels exist on sm_80/89/120
(sm_120 shows 1 KB static shared memory in every Q4_K mul_mat_q -> margin commit).

## Conclusions (for the hand-over)
1. With per-32 activation precision every output of every int8 MMA needs its own (row, column, sub-block) factor:
   >= 2 ops per output per MMA in either arithmetic. Measured in isolation: int (IDP4A+IMAD) 22.8 vs float
   (I2F+FMUL+FFMA) 22.1 cycles per IMMA (pure IMMA 16.7). In the kernel the int form costs MORE (register pressure ->
   spills at 255 regs). The integer sub-scale design cannot beat today's float epilogue on GA106.
2. The pipeline changes of the brief (3-deep ring, one barrier per half, register scales, ldmatrix B) are worth
   ~0 here: no-epilogue ring 2282 us vs no-epilogue async 2246 us; W4 (ring on async) -2 %.
3. Where the async kernel's 2804 us (4096^3) go, from the what-ifs: ~1390 pure IMMA (99 T/s), ~560 epilogue,
   ~200-550 global memory (W3: -9 % with barriers, -27 % in the barrier-free ring), ~150 min-term HMMA (12.5 % of
   the tensor pipe on GeForce: f16 x f16 -> f32 at 1/4 the int8 rate), ~70 barriers.
4. Weight centring (q - 8 in the int8 dot, 8*d*sc - dmin*m through the exact f32 sums) halves the matmul's error
   (5.7e-3 vs 1.28e-2 rel L2 vs exact) at ~+1 int op per MMA: a quality lever, and possibly what makes coarser
   (per-64/128) activations acceptable - numpy on N(0,1) data: per-128 + centring 6.8e-3 < today's 1.28e-2;
   needs a picture test with real activations (outliers).
Tools copied to whatif/tools/ (they were in the session scratchpad): epi.cu (the IMMA + epilogue micro-benchmark;
nvcc -O3 -arch=sm_86), ref.py / refrun.py (kernel outputs vs exact and vs DS4/DSY emulations), refq.py (scheme x
centring error study), refglu.py, pcmp.py (pipes on one binary), kt.py (nsys kernel times), census.py (SASS opcode
census of the K loop), mmq3bench.cpp (mmbench + MMBENCH_DUMP_IN). Paths inside point at the old scratchpad.

# W8A8 route: speed ceiling on the RTX 3060 (2026-10-22/23 night; programs in w8a8/, all card runs under the GPU lock)
Route: int8 weights per output channel (s_row[n] = max|w[n,:]|/127), dequantized from Q4_K/Q5_K per layer at run
time; int8 activations per token (s_tok[m] = max|x[m,:]|/127); D[m][n] = float(sum_k X[m][k] W[n][k]) * s_tok[m] *
s_row[n] in f32. X: M x K row-major, W: N x K row-major (GGUF layout), D: M x N row-major.
Programs (build: cmd //c "C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3\w8a8\build.cmd" [cutlass|cublaslt|quant]):
w8a8_cutlass.cu (CUTLASS 4.2.1 headers, 2.x API, int8 TN, int32 accumulation, OpMultiplyAddSaturate; EVT epilogue
= accumulator x row-broadcast s_row x column-broadcast s_tok -> f32; plus LinearCombination f32 / int32 + the scale
kernel), w8a8_cublaslt.cu (cublasLtMatmul R_8I x R_8I -> R_32I, COMPUTE_32I, all heuristic algos timed),
w8a8_quant.cu (dequantizers, activation quantizer, SwiGLU + quantizer). Medians of 25-30 runs after warm-up (CUDA
events). Every variant checked on 2048 sampled outputs against an exact int64 reference (max rel err ~1e-7) or, for the
quantizers, every value of sampled rows against a host reference (0 mismatches). Raw outputs: w8a8/*_run*.txt.

## 1. GEMM (ms, T/s = 2MNK/t, share of 102 TOPS)
| M K N               | CUTLASS EVT fused scale (best tile)   | cuBLASLt TN-W GEMM alone | + scale kernel | cuBLASLt TN-X | torch._int_mm (given) |
| 4096 4096 4096      | 1.536 ms 89.5 T/s 87.7 % (256x128x64 s3) | 1.603 ms 85.8 T/s | 2.025 ms 67.9 T/s | 1.543 ms 89.1 T/s | 83.2 T/s |
| 4096 4096 24576     | 9.595 ms 85.9 T/s 84.3 % (256x128x64 s3) | 10.340 ms 79.7 T/s | 12.542 ms 65.8 T/s | 9.555 ms 86.3 T/s | 76.7 T/s |
| 4096 12288 4096     | 4.475 ms 92.1 T/s 90.3 % (256x128x64 / 128x256x64 s3) | 4.467 ms 92.3 T/s | 4.824 ms 85.5 T/s | 4.446 ms 92.7 T/s | 90.2 T/s |
| 4096 4096 16384     | 6.296 ms 87.3 T/s 85.6 % (256x128x64 s3) | 6.389 ms 86.1 T/s | 7.927 ms 69.3 T/s | 6.332 ms 86.8 T/s | - |
- The fused scaling costs nothing measurable: CUTLASS LinearCombination f32 / int32 GEMM alone at the same tile is
  within +-3 % of the EVT one; the separate scale kernel adds 0.41 ms (4096x4096 out) .. 2.45 ms (4096x24576 out) at
  ~320 GB/s - i.e. 20-25 % on the [gate|up] shape. Two runs reproduce within 1 %.
- Tile sweep (EVT): 128x128x64 s3 is the slowest (80 / 65 / 78 T/s), s4 helps it; 128x256x64 s3 and 256x128x64 s3 are
  best; 128x256x64 s4 is bad on the large-N shape (63 T/s); 128x128x128 s3 in between.
- cuBLASLt returns 4 heuristic algos; the first is the fastest each time. Its TN-W layout (output = the row-major M x N
  the model wants) is 4-8 % slower than TN-X (transposed output) on 4096^3 / 24576.

## 2. Per-layer weight dequantizer (Q4_K / Q5_K -> int8 per row; one warp or half-warp per block)
| weights        | exact div | x reciprocal | 16-lane, 16-byte loads (rcp) | row amax (once per model load) |
| Q4_K 4096x4096 | 0.137 ms  | 0.119 ms | 0.117 ms 225 GB/s | 0.263 ms |
| Q4_K 24576x4096| 0.783     | 0.678    | 0.663 ms 237 GB/s | 1.529 |
| Q4_K 4096x12288| 0.373     | 0.336    | 0.335 ms 235 GB/s | 0.691 |
| Q5_K 4096x4096 | 0.134     | 0.124    | 0.122 ms 232 GB/s | 0.273 |
| Q5_K 16384x4096| 0.514     | 0.477    | 0.469 ms 241 GB/s | 1.059 |
| Q5_K 4096x12288| 0.388     | 0.358    | 0.353 ms 240 GB/s | 0.792 |
(GB/s = (Q bytes in + int8 bytes out) / t; ~2/3 of the 360 GB/s DRAM peak, the activation quantizer reaches 330.)
The reciprocal variant matches the host reference of rint(w * (1/s_row)); the exact-division one rint(w / s_row).

## 3. Activation quantizers (one 256-thread block per token, row in registers)
- per-token int8 of 4096 x 4096 f32: 0.255 ms (329 GB/s); 4096 x 12288: 0.752 ms (334 GB/s). (Today's
  quantize_mmq_q8_1: 0.257 / 0.827 ms - the same cost, so it cancels in a comparison.)
- SwiGLU of a [gate|up] f32 output 4096 x (2 x 12288) + per-token int8 of the 4096 x 12288 product: 1.358 ms
  (334 GB/s): the step the W8A8 route needs between [gate|up] and down (today's GLU kernel fuses it).

## Per step (the Q4_K_M call list: Q4_K 32 x [gate|up] K4096 N24576, 100 x 4096^2, 31 x down K12288 N4096; Q5_K
## 26 x 4096^2, 1 x K4096 N16384, 1 x K12288 N4096; M = 4096 tokens everywhere)
| part                                                      | ms per step |
| dequant per layer (16-lane, rcp)                          | 47.3 |
| GEMM with fused scaling (CUTLASS 256x128x64 s3)           | 650.4 (gate|up 307.4, 4096^2 Q4_K 153.6, down 138.7, Q5_K 50.7) |
| SwiGLU + quant between gate|up and down (32 calls)        | 43.5 |
| = W8A8 route                                              | 741.2 |
| activation quantizers, one per other matmul input (159)   | <= 40.5 (fewer with shared inputs; today pays the same 0.257 ms each) |
| (cuBLASLt TN-W + scale kernel instead of CUTLASS fused)   | 818.8 + 47.3 + 43.5 = 909.6 |
Today's MMQ kernels for the SAME call list, measured the same way (nsys kernel-only, async default, mmbench data):
32 x 16.737 (GLU kernel) + 100 x 2.803 + 31 x 7.912 + 26 x 3.027 + 11.797 (Q5_K N16384) + 8.752 (Q5_K K12288)
= 1160.4 ms -> W8A8 741.2 ms = -36 % (-419 ms per step); against the round-1 profile's 1,325 ms: -44 %.
Extra VRAM for the route: a ring of 2 layer buffers of int8 weights = 2 x 100.7 MB (the [gate|up] layer) + the
per-row scales (4 B per row); the gate|up f32 output 4096 x 24576 (403 MB) is written once per call (included in
the GEMM times above) and read back once by the SwiGLU + quant kernel (in its 1.358 ms).
Not covered: real-model token counts other than 4096 (text stream), power/clock differences of real data vs random
int8, and the attention / other ops (unchanged by the route).
