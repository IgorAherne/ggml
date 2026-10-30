# The F16 tensor-core convolution for Turing and newer (2026-10-29/30)

`src/ggml-cuda/conv2d-mma.cuh` (kernels and launch helpers), `conv2d.cu` (`ggml_cuda_conv2d_mma_supported`,
`ggml_cuda_conv2d_mma_fused`, the dispatch in `ggml_cuda_op_conv2d`), `ggml-cuda.cu` (the graph fusions), `norm.cu`
(the read-once channel RMS norm, also writing the convolution's input copy), `upscale.cu` (nearest by integer factors);
the sd.cpp side on its branch `spellcast-vaeconv` (`wan_vae.hpp`: DupUp3D for one frame). Test program
`w8a8-tools/test_conv_mma.cu`; test-backend-ops cases `test_conv_2d_vae_chain` and the decoder's channel counts. An Opus
agent's work, measured on the RTX 3060 (sm_86).

## Why
The Qwen-Image 2.1 VAE decode of a 1024x1024 picture (Wan 2.2-style decoder, 144 .. 1152 channels, 46 CONV_2D nodes:
40 3x3 + 6 1x1, about 17 TFLOP - not 3.5: the 1024^2 level alone has seven 391-1566 GFLOP layers) took 1.9-2.9 s with
`--vae-conv-direct` on the RTX 3060, 1.2-2.0 s of it in `conv2d_implicit_gemm_f16<3,3,true>` (64x64x64 tiles, one
shared stage with two barriers a tile, 16-bit gathers of a padded F16 copy), the rest in copies around it (the padded
input, DupUp3D's concats and permute copies, the norms, the scale / bias passes). The im2col path (the 12 GB
launcher's) spends ~1 s in cuBLAS F16-accumulate GEMMs plus 0.33 s of im2col and needs a 7.8 GB compute buffer.

The card's F16 x F16 -> F32 tensor rate (GeForce Ampere: half the F16-accumulate rate) is 28 SMs x 512 FLOP/clk x
~1.97 GHz = ~28 TFLOPS: 17 TFLOP cannot go below ~0.6 s with F32 accumulation (the brief's 0.2-0.3 s assumed 3.5 TFLOP).

## What
### The kernel (`conv2d_mma_f16<CFG, ASYNC, ACC16>`)
- Implicit GEMM: rows = output positions (b, oy, ox) flattened, columns = output channels, k = (ky*KW + kx)*IC + ci.
- The input is copied to F16 channels last once per call (`conv2d_mma_nhwc`: 64 positions x 32 channels through shared
  memory, coalesced F32 reads, 16-byte F16 writes), so 8 consecutive k of a row are one 16-byte load (IC % 8 == 0).
  Out-of-map taps (the padding) and k past K are zero-filled by the loads: no padded copy. Any kernel size, stride,
  dilation, padding, batch.
- Weights `[OC][IC][KH][KW]` reordered once per call to `[OC][KH][KW][IC]` (`conv2d_mma_weights`, a block per output
  channel through shared memory; 1x1: none).
- Tiles: 128 positions x 144 channels (144 divides every channel count of this VAE), 128 x 128, 128 x 64, 256 x 8 (<= 8
  channels: the last conv, 4); the one that pads the fewest channels. 8 warps, warp tile 32 x 72 (2 m16 x 9 n8 MMAs a
  k16), k tiles of 32, shared rows padded to 80 bytes (ldmatrix without bank conflicts).
- cc >= 8.0: two-slot cp.async pipeline (16-byte copies with zero-fill), one barrier a tile, `mma.sync.m16n8k16` F16 x F16
  -> F32, two blocks a SM (`__launch_bounds__(256, 2)`: <= 128 registers, 43.5 KB shared each). Against one block a
  SM with four slots (85 KB): 23-24 vs 19-20 TFLOPS - the other block's math hides a block's epilogue and pipeline
  fill. `cp.async.ca` (L1) was slower than `.cg`.
- Turing (cc 7.5; no cp.async, no m16n8k16): the same tiles, global -> registers -> the other of two shared slots (the
  next tile's loads issued before the current tile's math), two `m16n8k8` per k16, one block a SM (Turing has 64 KB).
  It also runs on newer cards (`GGML_CUDA_CONV2D_MMA=2`); on the RTX 3060 its outputs are bit-identical to the
  m16n8k16 kernel's on every shape tested.
- Epilogue: F32 `[b][oc][oh][ow]`, scalar stores (a warp writes 32-byte runs per channel), with what the fusion hands it:
  `v = fma(scale, acc, sbias)` (the graph's SCALE after the conv: scale_f32's FFMA), `+ bias[oc]`, `+ z` (residual), the
  unfused kernels' arithmetic, so a fused chain gives the bits of the unfused one.
- `GGML_CUDA_CONV2D_MMA=0`: the old kernels (conv2d_implicit_gemm_f16, cuBLAS for 1x1). Volta, the GTX 10xx (their F32
  SIMT and int8 kernels), AMD and MooreThreads keep their paths.
- `GGML_CUDA_CONV2D_MMA_ACC=16` (research, off): F16 accumulation (`mma ... .f16.f16.f16.f16`), see below.

### Fusions (ggml-cuda.cu)
- `ggml_cuda_try_fuse_conv2d`, at a SCALE (or a nearest 2x UPSCALE + SCALE) whose only reader is the CONV_2D, or at the
  CONV_2D: the input SCALE (sd.cpp's 1/128 against F16 overflow) and the UPSCALE (the Resample's) go into the F16 copy;
  `ggml_cuda_conv2d_fuse_output` folds SCALE(128) -> ADD(bias) -> ADD(residual, through Conv2dBut3d's reshape: a view of
  an in-place ADD's view has the root tensor as view_src) into the epilogue. Both of sd.cpp's SCALE nodes around every
  convolution are gone from the direct path.
- `ggml_cuda_try_fuse_rms_norm_cf`: when the norm chain (cont(permute) -> RMS_NORM -> MUL -> cont(permute) -> SILU ->
  SCALE) feeds only a CONV_2D on this path, the norm writes the convolution's F16 channels-last input copy directly
  (`ggml_cuda_op_rms_norm_cf_fused_nhwc_f16`) and the convolution follows with its output side fused: the F32 norm
  result is never written or read.
- `GGML_CUDA_CONV2D_FUSE=0`: none of these fusions (the nodes run one by one; the same pixels).

### Every architecture (the GTX 10xx included)
- The read-once norm (`rms_norm_cf_tiled_f32`: plain CUDA, <= 48 KB shared): a block stages its tile of pixels'
  channels in shared memory and replays rms_norm_cf_f32's chains and butterflies from there: the same bits, 2 tensor
  sizes of traffic instead of 3. `GGML_CUDA_RMS_CF_TILED=0`: the old kernel.
- `upscale_f32_nearest_int` (upscale.cu): nearest by integer factors in the first two dims, four outputs a thread with
  one 16-byte store, the same source indices (test-backend-ops UPSCALE 22/22).
- sd.cpp (`spellcast-vaeconv`, wan_vae.hpp): DupUp3D (the decoder's upsampling shortcut) for one frame:
  out[c][Y][X] = x[cc / repeats][Y / fs][X / fs], cc = ((c ft + ft - 1) fs + Y % fs) fs + X % fs - for this decoder
  either every step-th channel (x itself twice, then the odd channels) or the rows from two channels in turn (the last
  level): the strided channel view copied contiguous (a quarter / half of the output), then one nearest UPSCALE. It
  replaces 18 concat_cont + 8 cpy_scalar (permute copies at ~40-90 GB/s). `SD_VAE_DUPUP=0` keeps the general graph.
  The contiguous copy is there for the graph allocator: the UPSCALE straight from the view gave the same values but a
  4,718 MB compute buffer instead of 3,889 (the shortcut as the ADD's first operand: 4,430; even dropping the general
  graph's no-op third copy: 4,897) - the allocator's placement, not the live set, sets the high-water mark here.

## Measured (RTX 3060 12 GB, sm_86; the card is shared: minima, quiet runs, nsys GPU time)

### Per shape (`w8a8-tools/test_conv_mma.exe`, `CONV_SET=decoder CONV_REPS=7 CONV_TUR16=1`, 1.97-2.02 GHz)
ms per call (median of 7) and TFLOPS. "old" = conv2d_pad_f16 + conv2d_implicit_gemm_f16<.., true> (+ split-k reduce) as
ggml_cuda_op_conv2d ran it; "new" = the kernel alone (its F16 copy adds 0.1-6 ms, folded into the norm for most of the
engine's calls); "acc16" = the F16-accumulate variant.

| output, OC <- IC, kernel (calls) | old | new | acc16 |
|---|---|---|---|
| 64^2 1152 <- 1152 3x3 (10) | 6.79 (14.4) | 4.81 (20.3) | 3.08 (31.8) |
| 64^2 3456 <- 1152 1x1 (1) | 2.08 (15.7) | 1.41 (23.1) | 1.11 (29.4) |
| 128^2 1152 <- 1152 3x3 (7) | 24.5 (16.0) | 16.6 (23.6) | 14.9 (26.3) |
| 256^2 1152 <- 1152 3x3 (1) | 97.5 (16.1) | 67.4 (23.2) | 61.2 (25.6) |
| 256^2 576 <- 1152 3x3 (1) | 49.7 (15.8) | 33.0 (23.8) | 30.6 (25.6) |
| 256^2 576 <- 576 3x3 (5) | 24.9 (15.7) | 16.4 (23.9) | 11.2 (34.9) |
| 512^2 576 <- 576 3x3 (1) | 100.3 (15.6) | 66.0 (23.7) | 46.1 (34.0) |
| 512^2 288 <- 576 3x3 (1) | 57.0 (13.7) | 33.4 (23.5) | 22.8 (34.3) |
| 512^2 288 <- 288 3x3 (5) | 28.9 (13.5) | 16.4 (23.8) | 11.0 (35.6) |
| 1024^2 288 <- 288 3x3 (1) | 118.5 (13.2) | 66.1 (23.7) | 44.1 (35.5) |
| 1024^2 144 <- 288 3x3 (1) | 73.7 (10.6) | 33.7 (23.2) | 22.5 (34.7) |
| 1024^2 144 <- 144 3x3 (5) | 36.6 (10.7) | 17.1 (22.9) | 11.6 (33.6) |
| 1024^2 144 <- 288 1x1 (1) | 15.2 (5.7) | 4.37 (19.9) | 4.38 (19.9) |
| 1024^2 4 <- 144 3x3 (1) | 13.4 (0.8) | 6.97 (1.6) | 6.92 (1.6) |

Weighted by the decode's calls: old 1,237 ms, new 734 ms, acc16 550 ms. The F32 kernel holds 22.6-23.9 TFLOPS on every
large shape, ~85 % of the card's F16 x F16 -> F32 rate at that clock; the old kernel 10.6-16 T here (in the engine
29.5 ms a call on average under nsys: 1.18 s for its 40 calls).
Variants on the way (same harness; ms for K = 1296 / 288 / 10368 / 5184 shapes): four cp.async slots at one block a SM
21.2 / 5.9 / 79.4 / 79.8; three slots 20.7 / 6.0 / 77.2 / 79.5; two slots at two blocks a SM 17.0 / 4.4 / 69.4 / 68.0
(taken); `cp.async.ca` 24.3 / 7.7 / 89.5 / 92.2 against `.cg` 24.8 / 6.5 / 80.6 / 80.6 in that run; without the
epilogue's stores (a what-if, one block a SM) K = 1296 21.2 instead of 24.8 ms.

### Exactness (the same test, every output of every decoder shape; outputs ~N(0, 1), weights 1/sqrt(K))
- new vs old: 1x1 bit-identical (the same k order and MMA sequence); 3x3 max |d| 9.8e-6 .. 6.5e-5, max |d| / max |old|
  1.9e-6 .. 1.25e-5, relative RMS 1.4e-6 .. 1.1e-5 (k order (ky, kx, ci) instead of (ci, ky, kx): summation order).
- vs an F64 sum of the same F16 operands (1,000 sampled outputs a shape), |err| / sum |w x|: new 1.5e-7 .. 9.5e-7, old
  1.5e-7 .. 1.06e-6 (the same class); acc16 3.5e-4 .. 6.8e-4.
- The Turing kernel (register-staged, m16n8k8) on the RTX 3060: bit-identical to the m16n8k16 kernel on all 34 shapes
  (decoder + odd ones: tile tails, IC 8-48, OC 1-300, stride 2, dilation 2, 5x5, padding 0 / 2, batch 2-3); compute_75
  PTX through the JIT (`test_conv_mma_c75.exe`, `CONV_ASYNC=0`): the odd shapes all OK.
- compute-sanitizer memcheck / racecheck / synccheck: 0 errors / 0 hazards on 6 shapes (37x53 100<-32, 50x50 9<-16 b2,
  33x31 300<-24 s2, 64x40 144<-144, 70x30 4<-8 1x1 b3, 40x24 64<-64 5x5 d2).
- test-backend-ops (CUDA0 vs CPU): CONV_2D 2010/2010 (the decoder's channel counts added), UPSCALE 22/22, the VAE chains
  (`test_conv_2d_vae_chain`, 21 cases: SCALE / UPSCALE+SCALE / norm chain -> CONV_2D -> SCALE -> ADD [-> ADD]) 21/21
  fused, with `GGML_CUDA_CONV2D_FUSE=0 GGML_CUDA_RMS_CF_TILED=0`, with `GGML_CUDA_CONV2D_MMA=2` (the Turing kernel) and
  with `GGML_CUDA_CONV2D_MMA_ACC=16`.

### The decode in the engine (sd-cli, Turbo Q4_K_M, 1 step, chest seed 4242, `--sage-attn --vae-conv-direct`)
nsys, the decode phase (`scratchpad/prof/phases.py`): old direct GPU busy 1.733 s (span 1.885 s); the ggml changes with
the general DupUp3D graph 1.092 s (span 1.221); with the one-frame DupUp3D 0.937 s (span 1.050). That run's kernels:
convolutions 742 + 7 ms (45 calls), the norms writing the convolutions' F16 input 82 + 16 ms, the DupUp3D ADDs 20 ms,
the remaining F16 copies 11 + 6 ms (the 1x1 shortcuts, the first convs, the four upsampling convs with the UPSCALE
folded in), upscale 9.5 ms, one permute copy 8 ms (the last level's channel view), weight reorders 4.4 ms. Gone:
conv2d_pad_f16 (68 ms), the SCALE before the convs (32 ms), scale_add (75 ms), cpy_scalar (88 ms), concat_cont (47 ms).
"computing vae decode graph completed" (wall: host work included, and this machine's intermittent multi-second stalls,
seen with the old kernels too: 9.1 s, 5.8 s): new direct 1.33 / 1.35 / 2.34 s (knight 1.54), old direct 2.16 / 3.66 /
9.12 s, im2col 1.77 / 2.38 / 2.87 s. Compute buffer: 3,889 MB for both direct paths (im2col 7,777 MB).
The encode with an example picture (knight_front.png): 0.75 s (old direct 0.94, im2col 0.80); its buffer 1,612 MB as
before.

### Pictures (`code/devtools/qwen-speed/compare.py`, RGB / alpha PSNR)
- new vs old direct: chest 72.36 / 77.93 dB (max 1 level, mean 0.003), knight 72.07 / 76.36 dB.
- new vs im2col: chest 57.26 / 61.14, knight 57.56 / 59.94 - what old direct vs im2col gives (57.26 / 61.14,
  57.56 / 59.95).
- With an example picture (the encode's differences go through the sampler): new vs old direct 44.53 / 42.02 dB, new
  vs im2col 43.04 / 41.39, old direct vs im2col 42.73 / 40.47: the same class.
- `GGML_CUDA_CONV2D_FUSE=0 GGML_CUDA_RMS_CF_TILED=0`: pixel-identical to the default. The one-frame DupUp3D with the old
  kernels (`GGML_CUDA_CONV2D_MMA=0`): pixel-identical to the base engine; `SD_VAE_DUPUP=0`: pixel-identical. Repeated
  runs: identical.
- Turing (`GGML_CUDA_FAKE_CC=750`, a compute_75-only build through the JIT, no --sage-attn): new vs the old kernel
  72.25 / 77.90 dB (chest), 47.53 / 46.73 (example picture); decode 2.01-2.14 s vs 2.66 s, encode 0.76 vs 0.97 s (a
  proxy: an RTX 20xx has 64 KB of shared memory, one block a SM there).
- GTX 10xx paths (`GGML_CUDA_FAKE_CC=610`, a compute_61-only build): the generic changes (tiled norm, DupUp3D)
  pixel-identical to `SD_VAE_DUPUP=0 GGML_CUDA_RMS_CF_TILED=0`.
- F16 accumulation (`GGML_CUDA_CONV2D_MMA_ACC=16`): 55.82 / 56.53 dB vs the F32 kernel (knight 55.75 / 55.12), 54.91 /
  55.80 vs im2col (itself F16-accumulate in cuBLAS); decode 1.35-1.38 s (its ~185 ms of GPU time did not show in these
  wall times). Not the default.

### Compile
Every release architecture (61 70 75 80 86 89 90 100 120, SASS + PTX each, the build's flags:
`RELCHECK_FILES=conv2d.cu,norm.cu,ggml-cuda.cu python w8a8-tools/relcheck.py`) compiles. `cuobjdump -res-usage`: the
cp.async kernels 62-128 registers without stack on sm_80-120 (the 144-channel tile 128, at the two-blocks cap; 126 on
sm_100 / sm_120); the Turing kernels 80-187 registers without stack (the 144-channel tile 170 on sm_75); sm_61 / sm_70
stubs only.

## Not verified / open
- A real RTX 20xx (the Turing kernel's speed), RTX 40xx / 50xx, data-centre cards; a real GTX 10xx for the generic
  parts (the coordinator's pod).
- The brief's target (0.2-0.3 s of convolutions) assumed 3.5 TFLOP; this decoder is 17 TFLOP: ~0.6 s is the floor with
  F32 accumulation on an RTX 3060, 0.75 s is reached; F16 accumulation gives ~0.55 s (measured above, research only).
- The encoder is unchanged apart from the kernel (its AvgDown3D shortcut copies remain).
- Levers left: the narrow last conv (4 channels, 7 ms: each input value read 9 times from L2); the DupUp3D ADDs (20 ms)
  into the conv epilogue (needs the shortcut computed first, which moves the allocator's high-water mark); the four
  remaining F32 -> F16 copies (17 ms) into their producers.
