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
- The six-prompt suite under the Pascal emulation (a compute_61-only build + `GGML_CUDA_FAKE_CC=610` on the RTX 3060;
  the GTX 1070 Ti pod's CUDA had failed), against the same build with `GGML_CUDA_W8A8_KERNEL=off` (MMQ), PSNR dB rgb
  chest / knight / potion / fox / knight_back / ork_right: **37.0 / 38.5 / 33.2 / 22.2 / 23.8 / 24.7** (LPIPS 0.011 /
  0.007 / 0.013 / 0.068 / 0.047 / 0.056); the rounding-only control (the text encoder on the CPU) 30.0 / 30.5 / 33.6 /
  18.3 / 21.0 / 30.0 (LPIPS 0.047 / 0.026 / 0.012 / 0.110 / 0.082 / 0.019). Closer than the control on five of six; the
  orc's cybernetic forearm sits a little lower (a pose shift, same detail and sharpness at 100 %).
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
(Corrected in "Register banks" below: that outer product's 73 % came from its own extra instructions - it runs at 95 %
of its issue ceiling - and `ctrl.py` misread the dual-issue pairs; the GEMM's bank conflicts cost nothing measurable.)

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

## Shared-memory load costs on Pascal (2026-10-24 afternoon, a real GTX 1070, `w8a8-tools/lds_bench.cu`)

Clocks of the SM's one shared-memory pipe per warp instruction, 16 warps resident (the number the kernels above run
with); dp4a issues at 3.4 warp instructions per clock per SM (4 = the limit):

| pattern | clocks per warp instruction |
|---|---|
| LDS.128, every lane its own 16-byte chunk | 4.7 |
| LDS.128, one chunk for the whole warp, or one per 4 or 8 lanes (broadcast) | 2.2 |
| LDS.64, one chunk per 8 lanes | 0.55 |
| LDS.32, one word per 8 lanes | 0.28 |
| LDS.32, every lane its own word | 0.8 |

So a 16-byte shared load costs about two clocks even when the lanes share it, while 4- and 8-byte broadcasts are
nearly free - but ptxas re-merges adjacent `ld.shared.b32` / `.v2.b32` into LDS.128 (asm volatile does not stop it),
so the way to pay less is fewer loads per dp4a (bigger register tiles), not narrower loads. What this says about the
kernels: the dp4a GEMM's inner step (4 broadcast LDS.128 per 64 dp4a: ~9 clocks against 18.7 of issue) is not bound
by its loads, as the what-if above found; the INT8 attention's score loop (1 + 8 LDS.128 per 32 dp4a: ~20 clocks
against 9.4) is - the benchmark's copy of that mix runs at 39 % of the dp4a rate, and the real kernel's attention
call on this card (33.4 ms) sits at ~35 % of the card's peak. A 2-query x 4-key lane tile (6 loads per 64 dp4a)
is the fix being made (see `NOTES-fattn-dp4a.md`).

Also measured the same day: the per-node timing dump (`SD_EVAL_DUMP`) computes nodes one at a time, outside the
graph-level fusions, so it shows the VAE's permute copies around every RMS norm (1.5 s of a 7.7 s decode) that the
real graph never runs (the fork fuses cont(permute) -> RMS_NORM -> MUL -> cont(permute) into one kernel). A tiled
2D-transpose copy added to `cpy.cu` for permutes that merge into one transpose (bit-identical, ggml's CPY / CONT
tests pass, the dump's copies 1,542 -> 557 ms) therefore changed the real decode by nothing measurable (8.5 s both).

## Register banks, 2026-10-24 (GTX 1070 pod)

The question: the GEMM runs at ~60 % of the card's dp4a peak; is the loss beyond its issue mix ptxas's register
allocation on Pascal's 4 register banks (bank = register mod 4; operands of one instruction in one bank are read one
after the other unless one comes from the reuse cache)? **No.** The plan was to rename the registers of the sm_61
cubin and load it at runtime; the measurements below stopped it at the validation step, so nothing of it ships.
Tools: `w8a8-tools/sass_banks.py` (analyze: bank statistics, stall counts; whatif: NOPs / dropped waits / bank pairs
removed, timing only; synth: the patterns below), `w8a8-tools/bank_bench.cu` (the synthetic kernel, clock64-timed, so
its figures do not depend on the clock), `test_dp4a` with `W8A8D_CUBIN=<cubin>` (runs that cubin's kernels in place of
the compiled ones; the check mode compares their outputs with the compiled kernels' byte for byte).

**Synthetic: a register-only 8x8 dp4a outer product** (`bank_bench`, 512 dp4a per iteration, 16 warps per SM; % of
the dp4a issue rate, 4 warp-instructions per clock per SM; the same instructions and control words, only registers /
reuse flags edited):

| variant | % |
|---|---|
| ptxas's code (a reused across each 8-dp4a run; 33 same-bank pairs per 512 statically) | **99.5** |
| conflict-free with a reused / conflict-free without any reuse | 98.8 / 99.5 |
| the conflict-free registers without the reuse flags (a read every time, often in c's bank) | 74.5 |
| every dp4a: a and b in one bank (the same bank 8 times in a row) / a, b and c in one bank | 59.5 / 39.3 |
| a/b in one bank on the first dp4a of each run only (a then reused) / b and c in one bank, a reused | 92.5 / 91.4 |
| every 2nd dp4a a same-bank pair, the banks otherwise free / every dp4a, always the same bank | 97.9 / 50.0 |
| one bank read 3 times per 2 instructions | 66.6 |

So a same-bank pair inside one instruction is nearly free when the next instructions read other banks (0.03-0.09
clock); what costs is a bank asked for more than one read per instruction over several instructions in a row. Reuse
cache (probe = a dp4a whose a operand shares b's bank): a flagged operand serves a later reader of the same slot also
across an instruction that reads another register there unflagged (99.4 % vs 97.6 % without the flag), across a flagged
one (99.4), across a MOV (99.5 vs 97.6); a hit without the flag releases it (the next read misses: 71.6 %). The earlier
"73 % of the peak" of the outer product (`p6/outer.cu`) came from its own 38 non-dp4a instructions per 128 dp4a: a
77 % issue ceiling, of which it reached 95 %. The earlier bank count (157 of 1,024) and stall decode used a parser that
took the second instruction of a dual-issue pair (cuobjdump prints its encoding on the next line) for a control word.

**The GEMM's sm_61 code** (the Linux build of 1b054db): inner (group) loop 700 instructions = 512 IDP, 72 I2F, 64 FFMA,
33 LDS, 19 other; stall counts 106 dual-issue pairs, 538 x 1, 43 x 2, 6 x 4, 1 x 5, 6 x 13 = 731 clocks per 700
instructions of one warp, not binding with 4 warps per scheduler; 43 instructions wait on a scoreboard (IDP / FFMA
after shared loads). Same-bank operand reads (reuse counted conservatively): `w8a8d_gemm<0>` 63 per group loop (IDP 55,
all between the two LDS.128 fragment quads - a_i and b_j with i % 4 == j % 4 on the first dp4a of a run: the
4-register alignment fixes their banks, no renaming can move them; FFMA 8), `<1>` 50 (IDP 45, FFMA 5); outer (stage)
loop 11 (STS / IADD between quads). ptxas already keeps the 64 accumulators out of the b operands' banks.

What-if on the real kernel (`sass_banks.py whatif --fix-banks`: 81 % of those pairs removed by reading another lane,
wrong results, everything else unchanged), T/s medians of 4 runs at 4096^3 / glu 4096x24576x4096 / 4096x4096x12288:
compiled 14.07 / 14.62 / 14.31, without the pairs 14.04 / 14.59 / 14.32; per clock (SM clock sampled every 50 ms,
median while above 90 W, 3 runs) 8.92 / 9.15 / 9.07 vs 8.95 / 9.18 / 9.13 T/s per GHz: +0.3..0.7 %, the noise.
Nothing to gain.

**Where the time goes instead** (SASS what-ifs on the same code, results wrong; the T/s ranges are medians of 3 or 4
runs in separate sessions, per GHz = T/s at the median SM clock):

| what-if (stage = the K-stage loop) | T/s 4096^3 / glu / down | SM clock | T/s per GHz |
|---|---|---|---|
| none (the compiled kernel) | 14.1-14.3 / 14.4-14.6 / 14.1-14.3 | 1544-1582 MHz | 8.92 / 9.15 / 9.07 |
| the 7 LDG.128 of a stage NOPed | +11..17 %: 15.6-16.2 / 16.4-16.6 / 16.4-16.6 | 1771 MHz | 9.06 / 9.28 / 9.32 (+1.4..2.8 %) |
| (a build whose stage loop skips the global loads, a temporary source edit: 234 registers) | 15.6 / 16.3 / 16.2 | | |
| the 33 shared loads of the group loop NOPed | +7..15 %: 16.1-16.2 / 16.4 / 15.1-15.3 | 1645-1657 MHz | 9.78 / 9.85 / 9.20 (+1.4..9.6 %) |
| no wait for the global loads before the stores (DEPBAR SB5 and waits dropped) | -2..0 % | | |
| only the sub-scale load NOPed (the DEPBAR on its address registers at the loop top) | +-0 | | |
| no barrier / no shared stores | 0..+2 % / +1..3 % | | |
| tile order `W8A8_GROUP` -2, -8, -16, +2 / +8, +16 | within 1.5 % / 4..28 % slower | | |

The GTX 1070 runs this GEMM at its 130 W cap (~1.55-1.6 GHz). The global loads' latency is hidden (dropping the waits
changes nothing); their cost is energy: without them the clock rises 12-15 % at nearly the same work per clock - the
L2-to-SM traffic of the 128x64 tile (12 KB per block and 64-value stage, 1.5 bytes per output per 64 values of K).
DRAM is not it (the tile order changes nothing). The shared loads cost both: up to ~10 % of the clocks (the scoreboard
waits) and power. Per clock the kernel issues dp4a at 59 % of the card's peak (9.1 of 15.36 T/s per GHz) against an
issue ceiling of 69 % (1,024 dp4a per 1,477 instructions of a stage): 86 % of its ceiling. The levers left are
therefore less traffic per output (a bigger register or block tile: fewer global and shared loads per dp4a; the
128-channel block tile was 4 % slower on the GTX 1070 Ti, 18.5 vs 19.3 T/s, and was not measured on this power-capped
card) and fewer instructions per dp4a (the scaling), not the register allocation or the schedule's stall counts.
On a power-capped card compare variants per clock as well as in T/s: a variant that does less work per output can
win by its clock alone.

Exactness, re-checked with the test's default shapes (100 x 24576 x 4096 and 4145 x 4096 x 12288 added): exact mode,
0 of the sampled accumulators differ from the int64 reference at all 11 shapes; f32 mode, max relative error
1.6e-7 .. 1.0e-6 at K <= 4096 and 2e-4 .. 2.3e-3 at K 12288 (outputs near zero, as in the review). Traps: cuobjdump
puts the encoding of a dual-issue pair's second instruction on the next line; nvdisasm rejects a NOP whose control
field still sets a scoreboard (clear the barrier bits); the kernel names differ between Linux and Windows builds
(int64_t is `long` / `long long`: `..._Pvliiii` / `..._Pvxiiii`).

The 128-channel block tile on this card (test_dp4a built with `-DW8A8D_BN=128`: 256 threads, one block per SM, <0> /
<1> 231 / 233 registers, no spills, 33,280 B shared; exact mode 0 differences from the int64 reference at 4096^3 and
4145 x 4096 x 12288), medians of 4 runs interleaved with the default tile, 4096^3 / glu / down: **13.53 / 14.23 /
13.48 T/s at 1680 / 1651 / 1632 MHz = 8.05 / 8.62 / 8.26 T/s per GHz**, against BN 64 (242 / 243 registers) 14.29 /
14.54 / 14.23 T/s at 1600 / 1588 / 1569 MHz = 8.94 / 9.13 / 9.09 per GHz.
So a third less L2-to-SM traffic per output buys 4-5 % of clock, but one block per SM (8 warps, no second block to
fill a block's barrier and load phase) loses 6-10 % per clock: 2-5 % slower overall; the default stays.
