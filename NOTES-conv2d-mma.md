# The F16 tensor-core convolution for Turing and newer (2026-10-29)

`src/ggml-cuda/conv2d-mma.cuh` (kernels and launch helpers), `conv2d.cu` (`ggml_cuda_conv2d_mma_supported`,
`ggml_cuda_conv2d_mma_fused`, the dispatch in `ggml_cuda_op_conv2d`), `ggml-cuda.cu` (the graph fusions), `norm.cu`
(the read-once channel RMS norm, also writing the convolution's input copy); the sd.cpp side on its branch
`spellcast-vaeconv` (`wan_vae.hpp`: DupUp3D for one frame). Test program `w8a8-tools/test_conv_mma.cu`; test-backend-ops
cases `test_conv_2d_vae_chain` and the decoder's channel counts. An Opus agent's work, measured on the RTX 3060 (sm_86).

## Why
The Qwen-Image 2.1 VAE decode of a 1024x1024 picture (Wan 2.2-style decoder, 144 .. 1152 channels, 46 CONV_2D nodes:
40 3x3 + 6 1x1, about 17 TFLOP - not 3.5: the 1024^2 level alone has seven 391-1566 GFLOP layers) took 2.2-2.9 s with
`--vae-conv-direct` on the RTX 3060, 2.0 s of it in `conv2d_implicit_gemm_f16<3,3,true>` (64x64x64 tiles, one shared
stage with two barriers a tile, 16-bit gathers of a padded F16 copy: ~8.5 TFLOPS on average, 4.6-7 on the large maps),
the rest in copies around it (the padded input, DupUp3D's concats and permute copies, the norms, the scale/bias passes).
The im2col path (the 12 GB launcher's) spends ~1 s in cuBLAS F16-accumulate GEMMs plus 0.33 s of im2col and needs a
7.8 GB compute buffer.

The card's F16 x F16 -> F32 tensor rate (GeForce Ampere: half the F16-accumulate rate) is 28 SMs x 256 FMA/clk x 2 x
~1.95 GHz = ~28 TFLOPS: 17 TFLOP cannot go below ~0.6 s with F32 accumulation (the brief's 0.2-0.3 s assumed 3.5 TFLOP).

## What
### The kernel (`conv2d_mma_f16<CFG, ASYNC>`)
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
  -> F32, two blocks a SM (`__launch_bounds__(256, 2)`: <= 128 registers, 43.5 KB shared each).
  Measured against one block a SM with four slots (85 KB): 23-24 vs 19-20 TFLOPS - the other block's math hides a
  block's epilogue and pipeline fill. `cp.async.ca` (L1) was slower than `.cg`.
- Turing (cc 7.5; no cp.async, no m16n8k16): the same tiles, global -> registers -> the other of two shared slots (the
  next tile's loads issued before the current tile's math), two `m16n8k8` per k16, one block a SM (Turing has 64 KB).
  It also runs on newer cards (`GGML_CUDA_CONV2D_MMA=2`; the test program compares it there): on the RTX 3060 its outputs
  are bit-identical to the m16n8k16 kernel's on every shape tested.
- Epilogue: F32 `[b][oc][oh][ow]`, scalar stores (a warp writes 32-byte runs per channel), with what the fusion hands it:
  `v = fma(scale, acc, sbias)` (the graph's SCALE after the conv: scale_f32's FFMA), `+ bias[oc]`, `+ z` (residual), the
  unfused kernels' arithmetic, so a fused chain gives the bits of the unfused one.

### Fusions (ggml-cuda.cu)
- `ggml_cuda_try_fuse_conv2d`, at a SCALE (or a nearest 2x UPSCALE + SCALE) whose only reader is the CONV_2D, or at the
  CONV_2D: the input SCALE (sd.cpp's 1/128 against F16 overflow) and the UPSCALE (the Resample's) go into the F16 copy;
  `ggml_cuda_conv2d_fuse_output` folds the SCALE(128) -> ADD(bias) -> ADD(residual, through Conv2dBut3d's reshape) into
  the epilogue. Both of sd.cpp's SCALE nodes around every convolution are gone from the direct path.
- `ggml_cuda_try_fuse_rms_norm_cf`: when the norm chain (cont(permute) -> RMS_NORM -> MUL -> cont(permute) -> SILU ->
  SCALE) feeds only a CONV_2D on this path, the norm writes the convolution's F16 channels-last input copy directly
  (`ggml_cuda_op_rms_norm_cf_fused_nhwc_f16`) and the convolution follows with its output side fused: the F32 norm
  result is never written or read.
- `GGML_CUDA_CONV2D_FUSE=0`: none of these fusions (the nodes run one by one, the same bits).
- The read-once norm (`rms_norm_cf_tiled_f32`, every architecture: plain CUDA, <= 48 KB shared): a block stages its tile
  of pixels' channels in shared memory and replays rms_norm_cf_f32's chains and butterflies from there: the same bits,
  2 tensor sizes of traffic instead of 3. `GGML_CUDA_RMS_CF_TILED=0`: the old kernel.

### sd.cpp (`spellcast-vaeconv`, wan_vae.hpp)
DupUp3D (the decoder's upsampling shortcut) for one frame: out[c][Y][X] = x[cc / repeats][Y / fs][X / fs],
cc = ((c ft + ft - 1) fs + Y % fs) fs + X % fs - for this decoder either every step-th channel (levels 1-3: x itself,
x itself, the odd channels) or the rows from two channels in turn (level 4), each one nearest UPSCALE of a strided view.
It replaces 18 concat_cont + 8 cpy_scalar (the permute copies at ~40 GB/s). Architecture-generic (a graph change).

## Measured
(filled in below)

## Not verified / open
(filled in below)
