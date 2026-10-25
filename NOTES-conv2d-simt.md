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

## int8 convolutions, 2026-10-25

`src/ggml-cuda/conv2d.cu` (`conv2d_implicit_gemm_i8`, its passes `conv2d_i8_*`, the gate `conv2d_i8_input_ok`), an
Opus agent's work; test program `w8a8-tools/test_conv_i8.cu`. Everything below was measured on the RTX 3060 (native
sm_86 code, and the compute_61 PTX that the `GGML_CUDA_FAKE_CC=610` emulation runs); the GTX 1070's speed is not
measured yet.

### What
- The F32 kernel's tiling (144 / 128 output channels x 128 positions a block, 9 / 8 x 8 a thread, the SEG3 and ROWS
  staging, double-buffered shared memory, one barrier a tile) on int32 words of 4 int8 values, dp4a in place of FFMA.
  k runs over (input channel group of 4, ky, kx) and a word holds the 4 channels of a group at one position, so a word
  of the input is still one 32-bit value of the map and SEG3 stores it at its 3 kx shifts as before. k tiles of 9 words
  (SEG3: one channel group) / 8 words (ROWS: 32 channels). The int32 sums are exact (|sum| <= 127^2 K < 2^31 for
  K < 133,000; the decoder's largest K is 10,368).
- Passes at each call (nothing cached between calls): the per-channel maxima of |x| (chunks of 16,384 values, no
  atomics), the scales (one block), the weights' per-row scale, the weights quantized and transposed into the kernel's
  [M tile][k4][BM] words through shared memory, the input quantized and padded into [B][IC/4][PH][PW] words. They take
  the place of the F32 path's F16 pad copy and weight transpose; the scratch is about half of that path's (int8 input
  words instead of F16).
- Number format: s_c = amax_c^alpha * S with S = max_c(amax_c^(1-alpha)) / 127, x_q = clamp(rint(x / s_c), +-127);
  the activations' scales are folded into the weights (w' = w s_c, d_o = max|w'[o]| / 127, w_q = rint(w' / d_o)), so
  the epilogue is y = d_o * sum (one float per output channel) and the inner loop is dp4a and shared loads only.
  alpha 0 = one scale for the tensor, 1 = one a channel, between = SmoothQuant's migration; the engine uses 0.5.
- Gate: NVIDIA cc 6.1 / 6.2 (dp4a) before Volta, inside the Pascal F32 path (`GGML_CUDA_CONV2D_PASCAL=0` turns both
  off); the shapes of the SEG3 / ROWS modes (3x3 stride 1 with width % 64, or 1x1 stride 1 unpadded; map % 128),
  IC % 4 == 0, OC > 8 (the last conv keeps the F32 narrow kernel), split_k 1; and the convolution's input must be a
  SiLU's output (seen through reshapes / views / copies / conts / scales). `GGML_CUDA_CONV2D_INT8=0`: everything F32;
  `=2`: every int8 shape whatever feeds it (research only, it ruins the pictures); `GGML_CUDA_CONV2D_INT8_ALPHA=a`.
  Turing and newer are untouched (their convolutions run on the F16 tensor cores).

### Why only the SiLU-fed convolutions (the measurement that decided it)
Int8 on all 45 int8-shaped decoder convolutions ruined the picture (chest, 4 steps, against the F32 path: alpha 0
12.7 dB, 0.5 13.7, 1 14.1; colours gone). Fake quantization through the exact F32 kernel (an experiment build, removed)
separated the two operands: int8 weights per row with exact activations 51.5 dB; int8 activations per tensor 13.2 dB,
per channel 17.7 dB with exact weights. Per-channel activations one layer at a time: every convolution fed by
norm + SiLU 61-68 dB, but the 1x1 shortcuts (input an ADD; 512^2 37.0 dB, 1024^2 16.9 dB) and the 1024^2 upsampling
convolution (input an UPSCALE; 23.2 dB; the 512^2 one 55.7) ruin it alone: they carry the residual stream, whose
outliers (max |x| up to 140 x its rms; e.g. the 1024^2 shortcut's input: rms 37, max 5,100, a channel's median max 82)
leave the rest of its values a few int8 steps. The graph shows the pattern exactly: every conv's input is
SCALE(RESHAPE(x)) (the VAE's conv scale 1/128), x = SILU for the 34 residual-block convolutions, ADD for the shortcuts
and the first conv, UPSCALE for the 4 upsampling convolutions, MUL (the norm) / the attention for the attention block.
Tried for the stream convolutions, not adopted: a second int8 plane of the input's rounding residual (x / s = q +
lo / 254, the kernel run twice, ~15-bit activations; exact in the test program). One layer alone it is fine only with
per-channel scales (the 1024^2 shortcut 59.2 dB, the upsampling conv 64.2; per tensor 44.9 / 60.0), but per-channel
scales must be folded into the weights, and on all stream convolutions that cost more than it saved: 47.9 dB at alpha
0.5, 42.2 at 0, 40.1 at 1, against 50.8 dB with them on F32. So they stay F32 (about 1.0 s of the 1070's 3.4 s of
convolutions: the four upsampling convolutions 79 + 310 + 306 + 305 ms, the three shortcuts ~60 ms).

### Quality (the RTX 3060 under the Pascal emulation; int8 against GGML_CUDA_CONV2D_INT8=0 in the same build, so the
sampling is identical and the difference is the decode alone)
| picture | RGB PSNR / mean abs diff / max | alpha PSNR | over grey (what one sees) | LPIPS |
|---|---|---|---|---|
| chest, 4 steps, alpha 0.5 (alpha 0 / 0.3 / 0.4 / 0.6 / 0.7 / 1) | 50.80 dB / 0.38 / 11 (42.7 / 48.0 / 49.5 / 47.7 / 49.1 / 39.0) | 56.4 | 58.3 dB | 0.0009 |
| chest, 25 steps | 52.70 / 0.20 / 26 | 62.8 | 57.8 | 0.0004 |
| knight, 25 steps | 46.07 / 0.62 / 24 | 59.5 | 59.0 | 0.0020 |
| fox, 25 steps | 48.69 / 0.46 / 21 | 56.2 | 56.3 | 0.0012 |
| knight_back (an example picture: the VAE encode through int8 too), 25 steps | 48.88 / 0.41 / 31 | 54.9 | 56.6 | 0.0011 |

The RGB differences concentrate where alpha is 0 (invisible); over grey the pictures differ by 0.07-0.13 levels on
average. At 100 % (x16 difference maps, the knight's chest plate, the fox's face and eyes): +-1 level noise on flat areas
and along edges, no banding in the gradients, nothing visible in the dark areas. `GGML_CUDA_CONV2D_INT8=0` gives the
engine's pictures before this change bit for bit (chest, 4 steps, against the HEAD build: pixel-identical, twice).

### Exactness and checks
- `test_conv_i8` (19 decoder shapes + odd ones: OC 200 tile tail, IC 4, a 64 x 2 map, batch 2, 1x1 K 20 padded to 8
  words, OC 16; shapes the plan must refuse: odd maps, width 100, IC 142), alpha 0 / 0.5 / 1: the int32 sums equal an
  int64 reference over the operands the kernel read, every output (0 of up to 302 M differ); the quantized input and
  weight words equal a host recomputation from the F32 input / F16 weights and the passes' scales, every word; the
  scales within 3e-7 of a double-precision recomputation. Both as sm_86 SASS and as compute_61 PTX JIT-compiled on the
  RTX 3060 (the emulation's code).
- compute-sanitizer memcheck / racecheck / synccheck: 0 errors / 0 hazards on 5 shapes covering SEG3 and ROWS, 144 and
  128 channel tiles, batch 2, the tails.
- sm_61 (`-Xptxas -v`, cuobjdump): SEG3 tm 9 128 registers, no spill, 2 blocks per SM, 26.8 KB shared; its tile loop
  737 instructions, 648 of them IDP (88 %), 7,856 bytes - under the ~8 KB instruction-cache limit (the F32 kernel's:
  830 instructions, 648 FFMA = 78 %, 8,848 bytes). ROWS tm 9: 128 registers, 8 bytes of spill outside the loop, loop
  677 instructions / 576 IDP / 7,216 bytes; tm 8: 126-127 registers, no spill.
- The release build's exact nvcc command for conv2d.cu (all of 61;70;75;80;86;89;90;100;120, MSVC 2022 + nvcc 12.8)
  compiles.

### Speed on the RTX 3060 (a proxy: its dp4a issues at half its FFMA rate, the GTX 1070's at the full rate)
`test_conv_i8` (compute_61 PTX), ms: int8 kernel alone / int8 path with its passes / F32 path (pad + transpose + kernel):
| shape | int8 kernel | int8 path | F32 path | x |
|---|---|---|---|---|
| 1024^2 144 <- 144 3x3 | 14.6 (26.8 T/s) | 18.6 | 45.9 | 2.47 |
| 1024^2 144 <- 288 3x3 | 31.6 | 36.9 | 91.6 | 2.48 |
| 1024^2 288 <- 288 3x3 | 57.7 (27.1 T/s) | 66.0 | 177.4 | 2.69 |
| 512^2 288 <- 288 3x3 | 14.5 | 16.6 | 44.5 | 2.68 |
| 256^2 1152 <- 1152 3x3 | 58.3 | 60.6 | 173.6 | 2.87 |
| 128^2 1152 <- 1152 3x3 | 14.7 | 15.4 | 43.8 | 2.84 |
| 64^2 1152 <- 1152 3x3 | 3.9 | 4.4 | 11.7 | 2.69 |
| 1x1 shapes (512^2 288 <- 576 / 256^2 576 <- 1152 / 64^2 3456 <- 1152) | 3.3 / 3.4 / 1.3 | 7.4 / 5.4 / 1.5 | 12.8 / 11.3 / 3.8 | 1.7 / 2.1 / 2.5 |
The kernel runs at 24.8-27.1 T/s on the 3x3 shapes: up to what 28 SMs x 64 dp4a lanes x 8 ops x ~1.9 GHz allow (GA106 issues dp4a at half its
FFMA rate, an inference from this number), so there it is bound by the integer pipe; the F32 kernel at 8.5-9.0 TFLOPS (~65 % of the FFMA peak). The passes cost 4 ms
at 1024^2 x 144 (amax read + quantize read/write of the F32 input: memory-bound).
In the engine (sd-cli, chest 1024^2, the decode graph, same build): F32 3.43 s (5 runs, 3.43-3.44) -> int8 2.57-2.68 s
(-0.8 s, -24 %), the per-shape sum predicting -0.83 s. (Some runs of either kind took 5.4-5.8 s: a ~4 s pause with the
card idle inside the decode, seen with nvidia-smi sampling; host-side, not the kernels.)
Expected on the GTX 1070 (an estimate, not measured): the SiLU-fed convolutions are 2.28 s of its 3.39 s; dp4a issues
there at the FFMA rate, so the kernel should run ~3.5-4x the F32 kernel (88 % dp4a in the loop vs 78 % FFMA, x4 the
products an instruction), less the power cap and the memory-bound passes (256 GB/s): ~0.7-0.8 s instead of 2.28 s, the
decode 5.5 -> ~4.0 s.

### To measure on the pod (the GTX 1070)
- The kernels: build `w8a8-tools/test_conv_i8.cu` in the Linux build container (`nvcc -O3 -std=c++17 -arch=sm_61
  -use_fast_math -extended-lambda -I<ggml>/src/ggml-cuda -I<ggml>/src -I<ggml>/include -o test_conv_i8
  test_conv_i8.cu`), run `CONV_ALPHA=0.5 ./test_conv_i8` (exactness + ms / T/s per shape); sample
  `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -lms 100` meanwhile to compare per clock.
- The engine: `python run_sd.py --bin <bin> --out <png> --prompt chest --seed 4242 --steps 4 [--env
  GGML_CUDA_CONV2D_INT8=0] "--extra=--vae-conv-direct"`, twice each, "computing vae decode graph completed, taking";
  per kernel with nvprof's timeline (the pod's `/workspace/cuda/bin/nvprof --print-gpu-summary --csv`):
  `conv2d_implicit_gemm_i8` + `conv2d_i8_*` against `conv2d_implicit_gemm_simt_f32` + `conv2d_simt_weights` +
  `conv2d_pad_f16`.

### Not done / open
- Not measured on a real GTX 10xx (speed, power-capped clock, per clock); the emulation proves the code and the
  pictures, not the 1070's speed.
- The VAE encode: its stride-2 downsampling convolutions stay F32 (gather mode); its SiLU-fed stride-1 3x3 take int8.
- Larger pictures with VAE tiling: tiles whose widths are not multiples of 64 fall back to F32 (gather).
- Levers left: asymmetric (zero-point) activations for the SiLU outputs (they span ~[-0.28 a, a]: symmetric int8 wastes
  almost half its range; a per-channel zero point folds into one int32 constant per output channel; untried); fusing
  the maxima pass into the preceding SiLU (the passes are memory-bound: ~20 % of the int8 path at 1024^2).
