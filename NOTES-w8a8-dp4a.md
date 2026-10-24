# The W8A8 matmul path on the GTX 10xx: the dp4a GEMM (2026-10-24)

`src/ggml-cuda/w8a8-dp4a.cuh`, dispatched from `w8a8-mm.cu` for cc 6.1 / 6.2 (Pascal with dp4a: the GTX 10xx; Volta and
Turing keep MMQ). The same buffers, number formats, fusions and epilogues as the tensor-core kernel `w8a8_gemm`
(`w8a8.cuh`, `NOTES-w8a8.md`); only the GEMM's inner products differ. Test program: `w8a8-tools/test_dp4a.cu`.

## Result in short

- GTX 1070 Ti (a rented Linux pod): the card's dp4a peak measured at **30.5 T/s** (119 dp4a / clock / SM at 1,683 MHz,
  i.e. full rate; FFMA 7.6 TFLOPS). ggml's MMQ (q4_K, m 4096, n 512, k 14336, test-backend-ops) runs at **10.9 T/s**.
  The dp4a kernel: **19.3 T/s** at 4096^3, 18.6-19.0 T/s in the model (nsys: F32 epilogue 1,659 ms, GLU 1,390 ms a
  1024x1024 step).
- Qwen-Image 2.1 1024x1024, 25 steps, sd-cli on the GTX 1070 Ti: sampling **176.1 -> 122.8 s** (7.04 -> 4.91 s a step);
  `GGML_CUDA_W8A8_KERNEL=off` in the same build gives the old engine's picture pixel for pixel (178.8 s).
- Pictures: chest 30.8 dB / LPIPS 0.040 against the MMQ engine, the class of the RTX 30xx switch from MMQ to this
  format (31.2 dB on the same chest); same composition and materials, the bistable lock hasp flips.
- Exactness (RTX 3060, `W8A8D_SCALE_F32=0`): the int32 accumulators equal an int64 reference, and the f32 and the
  SwiGLU f16 outputs are byte-identical to `w8a8_gemm`'s at the Qwen shapes and odd token counts (4096 / 4145 / 1811 /
  8360 / 100 tokens; 0 of up to 101 M values differ). The default f32 mode: max relative error 7e-7 .. 1e-6 against
  the exact sums at K 4096; on outputs near zero more (2.5e-4 here, 1.38e-3 in the review's run at T 4145, N 384,
  K 12288: cancellation of 384 rounded additions), all far below the int8 formats' own error.

## Design

- Tile 128 tokens x 64 channels (`W8A8D_BN`, 128 also built), 128 threads, two blocks per SM (228 registers, 25 KB of
  shared memory), K stages of 64 values (two sub-scale groups), each thread 8 tokens x 8 channels.
- Shared memory holds the stage transposed into int32 words: `As[k4][token]`, `Bs[k4][channel]` (4 int8 values of K
  per word), so that a thread reads its 8 tokens' and 8 channels' words of one k4 step with two 16-byte loads each and
  issues 64 dp4a. Columns are XOR-swizzled (bit 2 by bit 5): the copies (a warp = 32 consecutive rows of one 16-byte
  chunk) and the reads (a warp = 4 token groups x 8 channel groups) are both free of bank conflicts.
- The thread's 8 tokens are tile rows `wm*64 + gid + 8i` (i = 0..7): their 5-bit sub-scales are positions
  `wm*64 + gid*8 + i` of the token tile's sy block (`w8a8_sy_pos`), 8 consecutive bytes, one 8-byte load per group;
  the A tile is stored in that position order. So the quantizer and the fused norm writer stay as they are.
- Per 32-value group the 64 dot products (8 k4 steps, the first starting from a zero accumulator) are scaled by the
  tokens' sub-scales into the accumulators: f32 (`I2F + FFMA`, default) or exact int32 (`W8A8D_SCALE_F32=0`). Pascal
  has no 32-bit integer multiply-add: `acc += sy * P` is three XMADs (ptxas does not use the 16-bit range of sy; the
  two-XMAD split needs two accumulators per output, too many registers), so the exact mode is 10 % slower there.
- Pipeline: no cp.async on Pascal; the next stage goes global -> registers before the current stage's math and
  registers -> the other shared buffer after it; one barrier per stage.
- Epilogue: `y = acc * d_t/31 * dW_n` in w8a8_gemm's order; F32 rows (2 x 16-byte streaming stores per token), or the
  interleaved [gate | up] pairs to ggml's SwiGLU, f16 (clamped to +-65504) or f32.

## What mattered (GTX 1070 Ti, test_dp4a bench, T/s at 4096^3 / glu 4096x24576x4096 / 4096x4096x12288)

| variant | T/s |
|---|---|
| first version: 128x128 tile, 256 threads, everything unrolled, exact | 13.2 / 12.9 / 11.2 |
| f32 group sums | 13.6 / 13.3 / 12.5 |
| 64-channel tile, two blocks per SM | 12.5-13.1 (no change) |
| next k4 fragments prefetched | no change |
| no scaling at all (what-if, wrong results) | 15.2 |
| no shared loads in the k4 loop (what-if) | 15.5 (the loads were not the limit) |
| **the group loop NOT unrolled** (~7 KB loop instead of ~17 KB): f32, 128 channels | 18.5 / 18.9 / 18.5 |
| the same, 64 channels (default) | **19.3 / 19.4 / 18.8** |
| the same, exact int32 | 17.7 / 17.4 / 16.3 |
| no scaling (what-if) | 20.1 |
| k4 loop unrolled by 2 / 1 (zeroing MOVs, loop overhead) | 15.6 / 14.1 |
| half the tokens' sums per pass (to fit 128 registers) | spills: 1.7 |

The instruction cache was the limit: a register-only 8x8 dp4a outer product (no memory at all) reaches 73 % of the
peak at 4, 8 or 16 warps per SM, while the fully unrolled stage (~1,600 instructions, 17 KB with the control words)
stalled at 43 % whatever was changed around it; the same code as a ~690-instruction loop runs at 63 %. Static stall
counts (decoded from the control words, `ctrl.py` in the session's scratch) were all 1 except 13-cycle waits after a
few shared loads, so the schedule itself was fine. Register bank conflicts: 15 % of the dp4a (157 of 1,024).

## Not done / open

- Volta (cc 7.0, dp4a, no int8 tensor cores) would likely gain too; it was never run, so it keeps MMQ.
- The 8 GB / 16 GB fit on the real card: the path's buffers equal the tensor-core path's (int8 weights of the current
  layer ~100 MB, activations), measured fine in the picture runs (no out-of-memory), not stress-tested.
- Beyond 19.3 T/s: the per-group scaling is 2 instructions per output per 8 dp4a (an 80 % ceiling); the rest is
  scheduling. A maxas-style SASS kernel would be the next step, not worth it here.

## Review (an Opus agent, 2026-10-24)

No wrong result, crash, out-of-bounds access or race found (compute-sanitizer racecheck / synccheck / memcheck clean;
exact mode byte-identical to the tensor-core kernel at T 1 / 100 / 129 / 257 / 1153 / 4223 / 4145, N 128 / 384 / 640 /
16384 / 24576, K 1024 / 3072 / 5120 / 12288, also as compute_61 code; sm_61: 241-243 registers, no spills, two blocks
per SM). Fixed after it: the tiled SIMT convolution only before Volta (it had also taken Volta, never measured); the
dp4a header guarded for research builds with other W8A8 tiles (its static_assert broke them).
