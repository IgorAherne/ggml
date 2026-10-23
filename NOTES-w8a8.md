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
