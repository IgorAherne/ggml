# The F32 implicit-GEMM convolution for the GTX 10xx (2026-10-24)

`src/ggml-cuda/conv2d.cu`, an Opus agent's kernel, built locally (Docker `sdlinux`, sm_61) and measured on a rented
GTX 1070 (GPU 0, `flock /workspace/gpu.lock`); test program `w8a8-tools/test_conv_simt.cu` (includes `conv2d.cu` with
`GGML_CUDA_CONV2D_KERNELS_ONLY` for its kernels, runs the old SIMT branch and the new kernels on the same inputs with
the engine's split_k policy and reduce kernel, compares every 32-bit word).

## Why
The Wan VAE decode of a 1024x1024 Qwen-Image picture on a GTX 1070 took 8.3 s, 6.3 s of it the 46 convolutions, which
ran at 2.2-3.0 TFLOPS on a card whose F32 FFMA peak is 6.45 TFLOPS: the SIMT branch of `conv2d_implicit_gemm_f16`
(written for MooreThreads, enabled for NVIDIA cards before Volta in `spellcast-perf-838b2f2`) converts F16 operands
to F32 one element at a time inside its inner loop with scalar 16-bit shared loads (12 loads + 12 conversions per 32
FFMA) and has a 64x64x64 tile with 4x8 outputs a thread.

## What
- Gate: inside the existing `use_simt` path, NVIDIA cards only (so compute capability < 7.0; MooreThreads keeps the
  old kernel); `GGML_CUDA_CONV2D_PASCAL=0` takes them back to the old branch (the control: same output bit for bit).
- Tiling: 256 threads, 2 blocks per SM; a block 144 output channels x 128 positions (144 divides every Wan channel
  count; 128 channels when that wastes less), a thread 9 channels x 8 positions; 72 FFMA per k from two 16-byte
  shared loads of the weights, one 4-byte load for the 9th channel and two 16-byte loads of the input.
- The weights are transposed once per call into contiguous k-rows per block of channels (kept F16; an F32 copy was
  4 % slower: twice the bytes); F16 -> F32 once when a tile is staged into shared memory; double-buffered shared
  memory with one barrier per tile.
- Input staging modes: 3x3 stride 1 with the padded width a multiple of 64 (one input channel per tile, each value
  loaded and converted once and stored at its 3 kx shifts); 1x1 (8-byte row loads); any other shape (a per-row
  gather); the decoder's last conv (4 output channels) a narrow kernel (4 channels x 16 positions a thread); other
  shapes with <= 16 output channels keep the old kernel.
- Same output bit for bit: the same `c += a*b` FFMA chain in k order from 0, one accumulator per output, the split_k
  policy and `conv2d_reduce_split_k` unchanged; the one subtlety - under `-use_fast_math` (FTZ) the old kernel's
  zero-padded k tail turns a -0 sum into +0, which the new epilogue repeats whenever the padded range exceeds K.

## Measured (GTX 1070, `/workspace/tests/conv/run_final2.txt`; "new" includes the weight transpose)
| output, OC <- IC, kernel | calls | old ms / TFLOPS | new ms / TFLOPS | x |
|---|---|---|---|---|
| 1024^2, 144 <- 144, 3x3 | 5 | 177.2 / 2.21 | 76.1 / 5.14 | 2.33 |
| 1024^2, 144 <- 288, 3x3 | 1 | 348.8 / 2.24 | 151.2 / 5.18 | 2.31 |
| 1024^2, 144 <- 288, 1x1 | 1 | 40.1 / 2.17 | 19.6 / 4.44 | 2.04 |
| 1024^2, 288 <- 288, 3x3 | 1 | 586.2 / 2.67 | 304.8 / 5.14 | 1.92 |
| 1024^2, 4 <- 144, 3x3 (narrow) | 1 | 55.8 / 0.19 | 2.7 / 3.99 | 20.4 |
| 512^2, 288 <- 288, 3x3 | 5 | 146.2 / 2.68 | 76.1 / 5.14 | 1.92 |
| 512^2, 288 <- 576, 3x3 / 1x1 | 1 / 1 | 287.7 / 29.8 | 152.9 / 18.5 | 1.88 / 1.61 |
| 512^2, 576 <- 576, 3x3 | 1 | 525.4 / 2.98 | 306.4 / 5.11 | 1.71 |
| 256^2, 576 <- 576, 3x3 | 5 | 130.4 / 3.00 | 77.6 / 5.04 | 1.68 |
| 256^2, 576 <- 1152, 3x3 / 1x1 | 1 / 1 | 261.2 / 26.9 | 154.5 / 18.3 | 1.69 / 1.47 |
| 256^2, 1152 <- 1152, 3x3 | 1 | 525.6 / 2.98 | 309.9 / 5.05 | 1.70 |
| 128^2, 1152 <- 1152, 3x3 | 7 | 130.6 / 3.00 | 78.7 / 4.97 | 1.66 |
| 64^2, 1152 <- 1152, 3x3 / 1x1 | 12 / 1 | 33.4 / 3.47 | 19.8 / 2.41 | 1.68 / 1.44 |
| 64^2, 1152 <- 64, 3x3; 3456 <- 1152, 1x1 | 1 / 1 | 2.01 / 10.24 | 1.20 / 7.02 | 1.67 / 1.46 |

The decoder's convolutions 6,287 -> 3,387 ms (1.86x); the 3x3 layers at 5.0-5.2 TFLOPS, 78-80 % of the nominal
peak (the pod's card ran at 1.70-1.76 GHz under a 130 W power cap: ~77 % of its actual clock). Tail / odd shapes
(37x53, 129x127, K 117 / 333 / 175, M 4 / 40 / 70 / 100, split_k 2-10, batch 2, stride 2, dilation 2, 5x5): 1.04-1.85x
through the gather mode; the narrow kernel's extra shapes (M 3-8, batch 2, width 200) 3.8-5.3x. Every one of the 34
shapes: 0 differing words (and the per-split partials with split_k > 1). compute-sanitizer memcheck / racecheck /
synccheck: 0 errors on 13 shapes covering every mode. sm_61 (`cuobjdump -res-usage`): the main kernels 123-128
registers, 2 blocks per SM, 25.6-26.8 KB of shared memory; the 3x3 kernel spills 32 B (2 local loads a tile, ~1 %),
the 1x1 rows kernel 40 B, the gather 1x1 48 B; the narrow kernel 123 registers, weights in dynamic shared memory
(20.7 KB at 144 input channels); the weight transpose 15 registers.

In the engine (sd-cli, 1024x1024, `--vae-conv-direct`, the pod's GTX 1070): the decode 8.3 -> 5.6 s; pictures
pixel-identical to the published engine's (2 steps, and with `GGML_CUDA_CONV2D_PASCAL=0` as the control; checked
again by the main session: a 2-step knight with an example picture - the VAE encode goes through the same kernels -
and the 25-step chest, both pixel-identical; VRAM peak unchanged at 7,251 MiB with an example picture). Compiles with
MSVC 2022 + nvcc 12.8 for every release architecture.

## Traps (the agent's)
- Loads sunk by the compiler: a conditional prefetch, or a load whose only use sits in a branch, is moved down to its
  use after the math and exposes its latency. Unconditional clamped loads (the last tile reloaded), an empty
  `asm volatile("" :: "r"(v))` (guarded off for HIP / MUSA) and branch-free stores into guard cells took the kernel
  from 4.7 to 5.1 TFLOPS.
- 9-warp blocks (288 threads, 2 per SM) hit the per-partition register file: ptxas caps at 96 registers and spills;
  `__maxnreg__(112)` drops to 1 block per SM. Dropped.
- What-if timings with zero-filled weights run at higher clocks under the power cap (inflated); final numbers use
  real data.
- Register bank conflicts: 48-70 % of the FFMAs by static analysis, from ptxas's allocation; reordering the loops made
  it worse, no fix from CUDA C.

## Not verified
Other Pascal cards (GTX 1070 Ti / 1080 / 1060 / 1050; the P100, cc 6.0, takes the path too but the release carries no
sm_60 code), the `GGML_CUDA_FAKE_CC` emulation, MooreThreads (its path is unchanged).
