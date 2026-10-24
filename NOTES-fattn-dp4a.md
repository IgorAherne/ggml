# FlashAttention with INT8 (dp4a) Q.K^T scores for GTX 10xx: notes (2026-10-24)

Committed as c6e96c02 (in the engine release spellcast-perf-6a82e8d). The sections below up to "Real GTX 1070 Ti"
are the writing agent's, measured on the RTX 3060 emulation (compute_61 code, `GGML_CUDA_FAKE_CC=610`); the real
card's figures are at the end. `build-sm61` (compute_61 only) holds a build of this state, with the path on by default.

**2026-10-24, later: the P.V product in INT8 as well (branch `spellcast-pv8`): see "INT8 P.V" at the end.**

## Result in short

- **Speed** (RTX 3060 running the compute_61 code with `GGML_CUDA_FAKE_CC=610`; relative numbers only):
  - Qwen-Image 2.1's big attention call (4096 queries x 4145 keys x 32 heads, head 128) drops from 50.0 to 26.1 ms
    (per-node eval dump, median of 32 calls): **-48 %, 1.91x**.
  - Isolated op bench: 51.4 to 26.6 ms.
  - Per step: the 32 big calls take 1603 to 839 ms; the DiT section 6.68 to 5.85 s.
  - 25-step sampling: 153.6 to 133.1 s (-13 %).
  - A real GTX 1070 Ti is not measured (see "Speed" below for why the ratio may differ there).
- **Accuracy**:
  - With INT8-exact inputs the kernel matches an F64 reference to 1e-6, which checks the indexing.
  - With realistic synthetic Q/K (row norms 12-36, outlier channels, a shared K channel offset), the per-row error
    is below a SageAttention-granularity INT8 emulation: median 1.18e-2 vs 1.69e-2, p99 0.22 vs 0.29.
- **Pictures** (vs the F32 tile kernel, same binary, `GGML_CUDA_FATTN_DP4A=off`):
  - chest 41.95 dB RGB / 41.00 dB alpha.
  - knight 34.43 dB RGB / 28.28 dB alpha.
  - A rounding-only control lands in the same class: the same F32 kernel with Q merely rounded to F16 gives chest
    34.51 / 31.92 and knight 35.14 / 29.61.
  - The shipped sage path, natively on the 3060, gives chest 35.40 / 28.24 and knight 43.58 / 38.67 against its own
    flash attention.
  - No object or shape changes. At 100 % the knight's breastplate sculpting and a small dent on one pauldron differ,
    and the chest's two lock rivets shade differently. Details below.

## What was built

### `ggml/src/ggml-cuda/fattn-tile-dp4a.cu` / `.cuh` (new)

`ggml_cuda_flash_attn_ext_tile_dp4a_use(device, dst)` is the gate. It returns true only when **all** of these hold:
- The `GGML_CUDA_FATTN_DP4A` environment variable is not `off`. It is read once; any other value or unset means on.
- The card is NVIDIA with cc >= 6.1 and < 7.0, and the build has dp4a code
  (`ggml_cuda_highest_compiled_arch(cc) >= 610`). That means Pascal with dp4a: not P100 (6.0, no dp4a), no Volta+.
- Head size 128 for Q, K and V, and Q has **>= 1024 columns**.
- No attention sinks and no logit softcap (both fall back, see "Not handled").
- Q is F32; K and V are F16 or F32; any mask is F16 with `ne[2] == 1`.
- `Q->ne[2] % K->ne[2] == 0`, matching `ne[3]`s, and `Q->ne[2]*Q->ne[3] <= 65535`.
- Rows are aligned for 4-element loads per lane: Q 16 B; K and V 8 B (F16) or 16 B (F32).
- A non-F16 V is not a view of K.

`ggml_cuda_flash_attn_ext_tile_dp4a(ctx, dst)` runs the op in these steps:
1. **V to F16** if needed. This is the same conversion `launch_fattn` does, into the op's F16 extra space, which
   `ggml_cuda_flash_attn_ext_get_alloc_size` already reserves for TILE (unchanged).
2. **Prologue** into one pool buffer:
   - The mean of every K channel over the keys, per head.
   - K minus that mean as INT8 per row (per key and head), with one F32 scale per row (row absmax / 127).
   - Q times the softmax scale times log2(e), as INT8 per row with one F32 scale.

   Subtracting the key mean is SageAttention's "smooth K". It adds one constant to all scores of a query, which the
   softmax removes exactly, but only without sinks/softcap; hence the gate.
   For Qwen the buffer is about 34.9 MB per call. The F32 tile path allocates 128 MB of `dst_tmp` here, because
   `launch_fattn` picks parallel_blocks = 2 for this shape.
3. **Attention kernel** `flash_attn_tile_dp4a<use_mask>`:
   - A block is 8 warps on 64 queries of one head, 32 keys per iteration; 2 blocks per SM.
   - Scores: each lane takes 1 query and 8 keys; one score is 32 `__dp4a` (via `ggml_cuda_dp4a`) over 16-byte
     shared loads, times the two scales.
   - Online softmax in F32, with exp2 and the tile kernel's max offset (3 in log2 units).
   - P.V in F32: V converted to F32 in shared memory, each lane accumulating 8 queries x 4 channels.
   - It writes the normalised output directly: no parallel blocks over the keys, no combine kernel.
   - Global loads of the next V tile (1 iteration ahead) and K tile (2 ahead) go through registers; 2
     `__syncthreads` per iteration.
   - Masks are supported: mask x ALiBi slope added to the scores. With a mask, `flash_attn_mask_to_KV_max<64>` skips
     fully masked key tiles when `mask->ne[3] == Q->ne[3]` and the mask has at least ceil(nq/64)*64 rows; otherwise
     all keys are visited.
   - GQA works by indexing (K head = Q head / ratio); multiple sequences work too.

Resources: sm_61 uses 127 registers, **0 spill** (unmasked variant) and 39.5 KB static shared memory. The SASS uses
`IDP.4A`, not an emulation. The masked variant spills 72 B on sm_61; Qwen-Image passes no mask.

**All release architectures compile** (61;70;75;80;86;89;90;100;120a, checked with the release gencode list). The
kernel is only dispatched on 6.x.

### `ggml/src/ggml-cuda/fattn.cu` (modified, +7/-1 lines)

The include, and in `ggml_cuda_flash_attn_ext`, `case BEST_FATTN_KERNEL_TILE:` calls the dp4a path when the gate
says so, else `ggml_cuda_flash_attn_ext_tile` as before. The rest of the backend is untouched:
- `ggml_cuda_get_best_fattn_kernel` and `get_alloc_size` are unchanged.
- The tile kernel itself is unchanged; it was temporarily edited for one control experiment, then restored with
  `git checkout`.

**Important for the next build:** ggml-cuda's `*.cu` files are globbed at CMake **configure** time. An existing build
directory must be re-configured first (`cmake -S . -B build-release`, or the pod's equivalent). Otherwise
`fattn-tile-dp4a.cu` is not compiled and `ggml-cuda.dll` fails to link: unresolved
`ggml_cuda_flash_attn_ext_tile_dp4a_use` / `ggml_cuda_flash_attn_ext_tile_dp4a`. `build-sm61` was re-configured.

The release build that was running when I started (00:10, `build-release`) predates the new file, and its
`fattn.cu.obj` was up to date at that build's start. So it does not contain this change.

### `ggml/tests/test_fattn_dp4a.cpp` (new, standalone; not wired into CMake)

It builds one `ggml_flash_attn_ext` on the CUDA backend and compares the output with an F64 CPU reference on sampled
rows. It also reports two CPU emulations: Q/K rounded to F16, and SageAttention-granularity INT8 (Q per 16 rows, K
smoothed and per 64 keys).

Options:
- `--exact`: INT8-exact integer inputs.
- `--mask`, `--causal`, `--alibi B`, `--softcap C`, `--nseq S`, `--f32kv`, `--bias B`, `--rows N`.
- `--out file`: raw output.
- `--bench N`: time the op.

Build line, against build-sm61's import libs:
`cl /O2 /EHsc /std:c++17 /MD /I ggml\include ggml\tests\test_fattn_dp4a.cpp /link build-sm61\ggml\src\ggml.lib build-sm61\ggml\src\ggml-base.lib`.

Run from `build-sm61\bin` with `GGML_CUDA_FAKE_CC=610`, or on a real cc 6.1 card:
`test_fattn_dp4a.exe <bin dir> 4096 4145 32 32`. Add `GGML_CUDA_FATTN_DP4A=off` for the F32 kernel.

A built copy (links build-sm61's `ggml.dll` / `ggml-base.dll`, so run it with `build-sm61\bin` on PATH) is
`C:\Users\Igor\AppData\Local\Temp\claude\C---myDrive-repos-spellcast3d\28dae9df-ec87-4365-a233-4b80e1acef34\scratchpad\fadp4a\test_fattn_dp4a.exe`.

## Numeric check (test_fattn_dp4a, 3060 with FAKE_CC=610)

Relative L2 error per output row (one query of one head) against F64:

| case | kernel | max | p99 | median |
|---|---|---|---|---|
| Qwen size 4096 x 4145 x 32h, realistic (1056 rows) | **dp4a** | 3.59e-1 | 2.24e-1 | 1.18e-2 |
|  | F32 tile kernel | 1.69e-5 | 1.11e-5 | 5.5e-7 |
|  | CPU: Q, K as F16 | 7.8e-3 | 3.5e-3 | 1.5e-4 |
|  | CPU: sage-granularity INT8 | 4.40e-1 | 2.95e-1 | 1.69e-2 |
| same, K channel offset 0 / 4 (bias) | dp4a | 3.66e-1 / 3.61e-1 | 2.15e-1 / 2.25e-1 | 1.15e-2 / 1.16e-2 |
| 2048 x 2100 x 8h, 2 sequences | dp4a / sage-gran. | 4.77e-1 / 4.37e-1 | 2.46e-1 / 2.81e-1 | 1.01e-2 / 1.30e-2 |
| 1024 x 77 keys x 8h | dp4a / sage-gran. | 2.39e-1 / 3.54e-1 | 1.46e-1 / 2.08e-1 | 3.1e-3 / 5.8e-3 |
| 1100 x 1000 x 4h, GQA 2:1 | dp4a / sage-gran. | 2.10e-1 / 4.67e-1 | 1.81e-1 / 2.79e-1 | 7.3e-3 / 1.16e-2 |

- The synthetic rows are deliberately peaked: norms 12-36 and 4 outlier channels at 5x. A few rows are dominated by
  one or two keys, so a 0.1 error in a score moves the output a lot; hence the large maxima, for the sage
  granularity too.
- Per-row INT8 beats the sage granularity in median and p99 everywhere. The max is better in 3 of 4 cases.
- The dp4a kernel run with sage's granularity (a temporary experiment, removed) reproduced the CPU sage-style numbers
  to every printed digit (4.396e-1 / 2.947e-1 / 1.691e-2). That cross-checks both the kernel and the emulation.

Exactness checks: integer Q/K with row absmax 127 and K pairs of opposite sign, so the key mean is 0 and INT8 is
exact. The dp4a kernel must then match F64 to F32 rounding:

| case | dp4a max rel | F32 tile max rel |
|---|---|---|
| 1100 q x 1001 keys (odd), 4 heads, GQA 2:1 | 9.3e-7 | 4.7e-6 |
| 1100 x 1024, GQA 4:1, block mask | 8.4e-7 | 4.9e-6 |
| 1280 x 1280, causal (KV_max skip active), GQA 2:1 | 1.0e-6 | 5.0e-6 |
| 1280 x 1280, causal + ALiBi 8, 8 heads | 9.8e-7 | 5.5e-6 |
| 1280 x 1280, causal + ALiBi, 3 sequences, GQA 2:1 | 9.6e-7 | 5.2e-6 |
| 1100 x 1001, 2 sequences | 9.0e-7 | 5.0e-6 |
| 1100 x 1001, K and V F32 (V converted to F16 as the tile kernel does) | 2.52e-4 | 2.52e-4 (same V rounding) |
| softcap 20: dp4a declines | output bit-identical with `=off` | |

## Pictures (Qwen-Image 2.1, seed 4242, 25 steps, 1024 px, VAE tiles 512, emulation on the 3060)

All 25-step pictures are in the scratchpad folder
`C:\Users\Igor\AppData\Local\Temp\claude\C---myDrive-repos-spellcast3d\28dae9df-ec87-4365-a233-4b80e1acef34\scratchpad\fadp4a\`:
- `v2\` holds the dp4a / f32 pairs, 100 % crops and diff maps.
- `ref\` holds the f32 pictures from the unmodified build. Those match `v2\*_off.png` to 74.5 / 75.3 dB (max |d| 1),
  so the reference is stable.
- `sage\` holds the native 3060 FA vs sage pictures.
- `ctrlA\` and `ctrlB\` hold the controls.

PSNR (RGB / alpha), SSIM, LPIPS (compare.py), each against the F32 tile kernel of the same build:

| picture | dp4a (this path) | control A: dp4a with sage granularity | control B: F32 kernel, Q rounded to F16 |
|---|---|---|---|
| chest | **41.95 / 41.00**, SSIM 0.9945, LPIPS 0.0058 | 36.47 / 36.77, SSIM 0.9821, LPIPS 0.0169 | 34.51 / 31.92, SSIM 0.9753, LPIPS 0.0200 |
| knight | **34.43 / 28.28**, SSIM 0.9817, LPIPS 0.0147 | 35.10 / 30.28, SSIM 0.9837, LPIPS 0.0130 | 35.14 / 29.61, SSIM 0.9861, LPIPS 0.0117 |

Yardstick: the shipped sage path vs flash attention, natively on the 3060 (build-sm86, same prompts and seed):
- chest 35.40 / 28.24 (SSIM 0.9879, LPIPS 0.0095).
- knight 43.58 / 38.67 (SSIM 0.9965, LPIPS 0.0028).
- The native and emulated F32 pictures themselves differ by about 25 dB (knight 24.7 dB): different trajectories.

What I make of it:
- Both pictures are chaotic in their details on the emulation trajectory. A perturbation about 100x smaller than
  INT8 (Q rounded to F16) already moves them to about 35 dB.
- Which picture is the sensitive one depends on the trajectory: chest for native sage, knight for the emulation.
- The knight's alpha at 28.3 dB is below the "30 dB" line. But the rounding-only control gives 29.6 there, and the
  shipped sage path gives 28.2 on the native chest. I read this as the noise floor of that picture, not a bug.

By eye at 100 %, flipping between the images:
- **Chest:** same object and texture. The two small rivets on the lock plate shade differently (one highlight
  brighter in dp4a). The silhouette alpha edge shifts sub-pixel.
- **Knight:** same character, pose, armor pieces, strap, buckles and rivet collar. The breastplate sculpting differs
  (pronounced pecs in F32, a smoother plate in dp4a), one pauldron gains a small dent/hole, and the outline shifts by
  about a pixel, which is the alpha halo in the diff map. These are noticeable in a flip test, but of the same kind
  as control B.
- Crops: `v2\crop_chest_lock.png`, `v2\crop_chest_maxdiff.png`, `v2\crop_knight_chest.png`.
- Diff maps (x4): `v2\diffmap_chest.png`, `v2\diffmap_knight.png`.

## Speed (RTX 3060 executing the compute_61 PTX, FAKE_CC=610)

| measure | F32 tile kernel | dp4a | change |
|---|---|---|---|
| big FA call, eval dump, median of 32 (min..max) | 50.0 ms (49.3..53.5) | 26.1 ms (25.9..28.2) | -48 % |
| 32 big FA calls of one step | 1603 ms | 839 ms | -48 % |
| 32 small FA calls (49 text queries, tile kernel both times) | 13.6 ms | 13.0 ms | unchanged path |
| DiT section of one step (eval dump) | 6.68 s | 5.85 s | -12 % |
| isolated op (test --bench) | 51.4 ms | 26.6 ms | -48 % |
| 25-step sampling (chest / knight) | 153.7 / 153.5 s | 133.1 / 133.2 s | -13 % |
| s/it in the run | 6.06-6.15 | 5.28 | |

Breakdown of the dp4a op on the 3060, measured by switching parts off in an earlier variant (approximate):
- prologue (key mean + two quantizers): 0.54 ms
- score phase: ~6.2 ms, about 85-90 % of the GA106's dp4a rate
- P.V: ~15.5 ms, about 70 % of the F32 FMA peak
- the rest: ~6.7 ms (softmax, tile stores, the 2 barriers per 32 keys, latency)

Occupancy is 2 blocks x 8 warps per SM, limited by registers (127). Tried and dropped, each within about 1.5 % or
worse:
- 4 queries x 8 channels per lane in P.V (slower: 128-bit loads still take 4 wavefronts).
- Other unroll depths.
- A later prefetch, which spilled on sm_61.

Kept: skipping the accumulator rescale when no query's maximum moved (exact; about 1 %).

Why the 1070 Ti ratio may differ:
- On GP104, `dp4a` runs at 4x the FP32 rate; on GA106 it is about 2x. So the score phase should weigh less on Pascal.
- P.V is plain FP32 FFMA on both.
- What I can't verify here: Pascal's cost for 16-byte shared loads where 4-8 lanes share an address. The score loop
  issues 72 such loads per warp per 32 keys, against 256 dp4a. If Pascal took 4 wavefronts for each, the score phase
  would become shared-memory bound.

A rough expectation for the attention node on the 1070 Ti is 1.9-2.3x, but please measure. The per-node dump command
is the same as here, without FAKE_CC.

## Not handled (falls back to the unchanged F32 tile kernel)

- Head sizes other than 128.
- Q with < 1024 columns. This includes Qwen-Image's 49-query text calls.
- Logit softcap and attention sinks: the key-mean shift is not softmax-invariant with them. Verified bit-identical
  fallback for softcap.
- K or V in quantized types (q4_0, q8_0, bf16 ...).
- A non-F16 V that is a view of K.
- Unaligned rows.
- Cards other than cc 6.1-6.x.
- Masks are supported, but:
  - No sparse-gather path.
  - The KV_max skip is only active when `mask->ne[3] == Q->ne[3]` and the mask has >= ceil(nq/64)*64 rows. Otherwise
    every key tile is visited (correct, just no skipping).
  - The masked kernel variant spills 72 B on sm_61.
- GQA: correct, but without the tile kernel's GQA packing (ncols2), irrelevant for >= 1024 queries.
- V is kept F16/F32, never INT8, as asked.

## Unsure / to verify

- Real-Pascal speed and the shared-memory broadcast question above. I tuned only on the 3060 emulation.
- Real Qwen Q/K were not dumped. The numeric check uses synthetic data; the pictures are the real-data evidence.
- An all-masked query gives NaN (0/0), as the tile kernel does.
- The CUDA-graph path was not exercised separately. Launches are plain `<<<>>>` on `ctx.stream()` with a pool buffer,
  as in `launch_fattn`; sd-cli ran fine.
- The pool buffer (~35 MB per call for Qwen) replaces the tile path's 128 MB `dst_tmp`. I derived the parallel_blocks
  = 2 choice of `launch_fattn` by reading the code (occupancy 2), not by instrumenting it.

## Real GTX 1070 Ti (2026-10-24, a rented vast.ai pod, the Linux build of the same commits)

Seed 4242 (potion: 7), 25 steps, 1024x1024, 512-pixel VAE tiles, sd-cli through `code/devtools/qwen-speed/run_sd.py`
of the hub; on / off = `GGML_CUDA_FATTN_DP4A`.

| measure | F32 tile kernel | dp4a |
|---|---|---|
| 25-step sampling, chest / knight / potion | 216.4 / 216.4 / 216.5 s | 174.1 / 178.7 / 178.4 s (-18..-20 %) |
| the hub's own launcher (qwen_bench, text only, one picture) | 228.4 s (drawing 213.6) | 188.1 s (drawing 173.2) |
| a drawing step | ~8.5 s | ~6.9 s |

Per step before the change (per-node dump with times only, 1024², F32 kernel): the quantized matmuls 5.6 s (the dp4a
MMQ near the card's int8 peak for this 20B model), the attention 2.8 s, everything else ~0.6 s. So the attention went
~2.8 -> ~1.1 s, about the 1.9-2.3x expected; what is left of a step is the matmuls at the hardware limit.

Pictures against the F32 tile kernel of the same binary (PSNR RGB / alpha), with rounding-only controls on the same
card (the F32 kernel unchanged; only the attention as plain matmuls, or only the text encoder on the CPU):

| picture | dp4a | control: no flash attention | control: text encoder on the CPU |
|---|---|---|---|
| chest | 34.69 / 33.84 dB | 39.31 dB RGB (*) | 33.19 dB RGB (*) |
| knight | 30.34 / 28.28 dB | 31.44 / 23.70 dB | 29.69 / 23.73 dB |
| potion | 38.03 / 33.59 dB | - | - |

(*) The chest controls were drawn before the tile fix (decoded in 256-px tiles): they are measured against the shipped
build's picture with the same decode; that decode alone differs from the 512-tile one by 40.8 dB.

The real card diverges more than the emulation (chest 42 -> 35 dB): a 25-step sampler amplifies any rounding change,
and the rounding-only controls move these pictures by as much or more. By eye: the same objects and quality; bistable
details flip - the chest's lock gains a raised hasp (the RTX 3060's own picture has it too; the text encoder on the
CPU gives a third, flat variant), the knight's strap ring grows, while both controls also lengthen the tabard strip
and add a sword hilt. The release keeps the path on for cc 6.x.

## INT8 P.V (2026-10-24, branch spellcast-pv8, an Opus agent; measured on the RTX 3060 emulation only)

The P.V product of the dp4a kernel in INT8 too: `GGML_CUDA_FATTN_DP4A_PV` = `int8` (default) | `f32` (the kernel
above, unchanged: bit-identical pictures) | `int8-all` / `int8-f32` (research: no outlier blocks / the new kernel's F32
path for every block). Commits on `spellcast-pv8` (from `spellcast-dp4a` 421b2cb1): 6a31623e (first version), 41e1ab0e
(numerator rounding without F2I, unroll macros, test), b1158241 (score sums read without I2F), plus these notes.

### Format (chosen on real Q/K/V of the model)

- P: the online softmax's numerators of one query over one 32-key block (one iteration) as **u8 relative to the
  block's largest**: q = rint(255 * exp2(s - s_max_block)), scale exp2(s_max_block - m)/255. The block's largest key is
  exact (255); a block of small numerators keeps its own relative precision. The denominator sums the same dequantised
  numerators (scale * sum q), so the weights stay normalised (an exact-p denominator was measured slightly worse).
- V: **int8 per (channel, 32-key block)**, scale = the block's |max| of the channel / 127, from a prologue kernel.
- **Outlier blocks keep the F32 P.V**: a block with a key whose V (|max| over the channels) is above 4x or below 1/4x
  the mean over the head's keys. In Qwen-Image these are the text tokens: attention sinks (keys 44-48 at step 0 take
  ~98 % of the weight in most heads, with V norms ~1/30 of the typical) and outliers 7-20x the typical (keys 1, 14, 30,
  33, 37). Measured: 2 of 130 blocks per head (1.3-1.5 %); `GGML_CUDA_FATTN_DP4A_PV_RATIO` changes the 4.

Study (`scratchpad pv8/study.py` on dumps of 8 calls of the chest picture, steps 0 / 12-13 / 24, 512 queries x 32
heads each, against F64 attention of the same inputs; numbers = Frobenius error of the whole call relative to its
norm):

| call (step, block) | today: INT8 scores, F32 P.V | P.V only: u8 P, V per channel | per (channel, block) | per token folded into P | V 16 bit (2 dp4a digits) | **chosen: per (ch, block) + outlier blocks** |
|---|---|---|---|---|---|---|
| 0 (0, 0) | 9.79e-3 | 1.54e-1 | 8.13e-2 | 6.65e-3 | 4.82e-3 | **2.52e-3** |
| 13 (0, 13) | 8.96e-3 | 8.60e-3 | 5.61e-3 | 3.59e-3 | 3.32e-3 | **2.77e-3** |
| 31 (0, 31) | 4.29e-3 | 2.40e-3 | 1.58e-3 | 4.31e-3 | 1.04e-3 | **1.17e-3** |
| 400 (12, 16) | 8.63e-3 | 7.87e-3 | 4.72e-3 | 4.02e-3 | 3.03e-3 | **3.35e-3** |
| 415 (12, 31) | 4.87e-3 | 4.87e-3 | 2.99e-3 | 3.56e-3 | 1.42e-3 | **2.79e-3** |
| 431 (13, 15) | 8.23e-3 | 7.39e-3 | 4.79e-3 | 3.76e-3 | 2.80e-3 | **3.04e-3** |
| 780 (24, 12) | 8.32e-3 | 9.99e-3 | 5.80e-3 | 3.09e-3 | 3.27e-3 | **2.51e-3** |
| 799 (24, 31) | 6.25e-3 | 6.25e-3 | 3.74e-3 | 4.16e-3 | 1.53e-3 | **3.23e-3** |

- The chosen format's P.V error is 24-57 % of what the INT8 scores already add; both together are 3-16 % above the
  scores alone (they add in quadrature). The F16 P.V of the tensor-core kernels: 2e-5..1.7e-4.
- Per-channel V scales fail at step 0 (15 %): the sink's tiny V row sits under the channel's scale. Per-token scales
  folded into P fix that but break on the outlier tokens (one row at 12 % in call 415). 16-bit V (two dp4a passes)
  is as good as the chosen format but costs twice the dp4a.
- The kernel's real output (a second dump, of the INT8 P.V kernel itself): 1.01e-2, 9.36e-3, 4.94e-3, 9.32e-3,
  5.53e-3, 8.95e-3, 8.70e-3, 7.09e-3 for the same calls (today's format on those inputs: 9.79e-3, 8.91e-3, 4.80e-3,
  8.69e-3, 4.81e-3, 8.40e-3, 8.34e-3, 6.20e-3); kernel vs the Python emulation of the format 1e-5..1.3e-4 (rounding
  flips).

### Kernel (`flash_attn_tile_dp4a_pv8`, `flash_attn_dp4a_quantize_v`, `flash_attn_dp4a_v_flags`)

- Prologue: one CTA per (head, 32-key block) quantizes V straight from its type (F32 strided view or F16: the F16
  conversion of V is skipped in this mode) into words of 4 keys of one channel ([head][block][key/4][128]), the
  channel scales, and per block the min / max / sum of the keys' |V| max; one CTA per head sums the blocks in a fixed
  order (deterministic) and writes the outlier flags.
- Main kernel: the F32 kernel's structure (8 warps, 64 queries, 32 keys an iteration, INT8 scores unchanged). After the
  softmax a lane writes its 8 numerators as bytes into [key/4][query] words (8 STS.U8, as many stores as the F32
  numerators); q is the low byte of fma(255, e, 1.5*2^23) (no F2I, a quarter-rate instruction on Pascal).
- P.V: a lane still owns 8 queries x 4 channels; per word of 4 keys one LDS.128 of V words + two broadcast LDS.128 of
  P words, then 32 `dp4a.u32.s32` (SASS IDP.4A.U8.S8). The int32 sums start at the bits of 1.5*2^23 (|sum| <= 32 x 255
  x 127 < 2^22), so one FADD reads them as floats (no I2F); then acc += (p_scale * v_scale) * sum. Per lane and 32 keys:
  256 dp4a + 96 FADD/FMUL/FFMA + 27 LDS.128, against 1024 FFMA + 96 LDS.128 in the F32 kernel.
- The same trick for the score sums (|sum| <= 128 x 127 x 127 < 2^22): bit-identical, -3 % on the 3060.
- Outlier blocks: F32 values loaded straight from V into the shared tile (no prefetch, rare), the F32 numerators and
  the F32 P.V loop of the old kernel (unrolled 2).
- sm_61: 128 registers, no spill in the unmasked main loop (the masked variant spills 48 B, as the old one 72 B),
  41.9 KB static shared memory (+2.3 KB), 2 blocks per SM. Main loop 13.8 KB of SASS (old kernel 13.2 KB with its 6 KB
  P.V loop run twice per iteration). `FDP4A_PV8_SC_UNROLL` / `_KW_UNROLL` / `_F32_UNROLL` shrink it (2/2: 9.7 KB,
  +2 % on the 3060) if the real card shows an instruction-cache problem.

### Checks

- `test_fattn_dp4a --pvexact --scale 1e-4` (new): Q INT8-exact, all keys of a 32-key block identical (so every
  unmasked key has numerator 255), a random 0/-inf mask, V integers with |max| 127 in every (channel, block), every 9th
  block with one key at 1/8 (an outlier block). GPU vs F64: max 3.4e-7 (GQA 2:1; 2 sequences; F32 K/V), the F32 path
  for every block (`int8-f32`) 7.3e-7 = the old kernel's 7.3e-7.
- `--emu` (new): a CPU emulation of the format (the kernel's quantizer roundings, the key mean summed in its order).
  GPU vs emulation, median 1.3e-7..1.6e-7 per row; a few rows up to 1.8e-3 (rounding-boundary flips: the fast-math
  reciprocal in the quantizers, ex2.approx); the emulation vs F64 equals the GPU's error to 3 digits. Cases: Qwen size,
  block mask, causal + ALiBi + 3 sequences + GQA, F32 K/V, `int8-all`.
- softcap declines to the F32 tile kernel as before; `GGML_CUDA_FATTN_DP4A_PV=f32` reproduces today's chest picture
  bit for bit.

### Speed (RTX 3060 executing the compute_61 code, `GGML_CUDA_FAKE_CC=610`; relative numbers only)

| measure | F32 P.V (today) | INT8 P.V |
|---|---|---|
| Qwen-size call (4096 q x 4145 keys x 32 heads), isolated op, `test_fattn_dp4a --bench 30` | 26.85 ms | **16.09 ms (-40 %)** |
| the same in sd-cli, per-node eval dump (`SD_EVAL_DUMP`, `SD_EVAL_DUMP_STATS=0`), median of 64 calls | 26.77 ms | **16.67 ms (-38 %)** |
| 32 big calls of a step | 857 ms | 534 ms |
| 25 steps, 1024x1024 (suite, steady s/it; sampling), text-only cases | 3.89 s/it; 99.1 s | 3.57 s/it; 91.0 s (*) |
| the same, cases with a reference image (knight_back / ork_right) | 4.77 / 4.72 s/it; 131.3 / 123.9 s | 4.12 / 4.10 s/it; 114.9 / 107.9 s (*) |

(*) measured with 41e1ab0e; b1158241 takes another ~0.45 ms a call (~15 ms a step) off, bit-identical outputs.

On the 3060 a dp4a issues at half the FFMA rate; on Pascal at the same rate, so the P.V's share should shrink more
there: if the F32 P.V took ~26 of the ~40 ms a call on the GTX 1070 Ti, the call should land around 20-24 ms (the
attention ~1.3 -> ~0.7 s a step, ~15 s off a 25-step picture). **Not measured**: the rented GTX 1070 Ti pod's CUDA was
broken the whole session (cuInit -> CUDA_ERROR_NOT_INITIALIZED at 02:30 and 03:26 UTC). Two Pascal-only unknowns to
check first there: the instruction cache (main loop 13.8 KB, the old kernel's 13.2 KB; try `FDP4A_PV8_SC_UNROLL` /
`FDP4A_PV8_KW_UNROLL` 4 or 2) and the cost of the broadcast LDS.128 of the numerator words.

### Pictures (Qwen-Image 2.1, the six suite cases, 25 steps, 1024 px, 512-px VAE tiles, emulation on the 3060)

All against today's kernel (`pv8_base`: the same engine, 421b2cb1 + the inactive dump hook) through the same
compute_61 engine; PSNR RGB / alpha, LPIPS (compare.py). Controls of today's kernel: the text encoder on the CPU
(`--backend te=cpu`) and no flash attention (`run_sd.py --no-fa`: the attention as plain F32/F16 matmuls, i.e. without
the INT8 scores). Last column: two builds of the INT8 kernel that differ only in rare rounding ties of the numerators
(6a31623e vs 41e1ab0e), the sampler's own noise floor for each case.

| case | INT8 P.V (b1158241 = 41e1ab0e) | control: TE on CPU | control: no FA | INT8 tie-rounding variant vs INT8 |
|---|---|---|---|---|
| chest | **44.99 / 40.06, 0.0025** | 31.11 / 26.61, 0.0366 | 44.45 / 39.69, 0.0048 | 45.45 / 41.50 |
| knight | **44.75 / 41.27, 0.0015** | 30.94 / 25.72, 0.0225 | 40.96 / 40.85, 0.0032 | 45.15 / 48.43 |
| potion | **41.34 / 35.97, 0.0020** | 34.58 / 32.17, 0.0094 | 40.04 / 38.65, 0.0036 | 41.69 / 36.79 |
| fox | **27.40 / 27.32, 0.0237** | 21.99 / 24.14, 0.0647 | 23.97 / 23.89, 0.0439 | 26.15 / 26.69 |
| knight_back | **24.73 / 20.93, 0.0363** | 25.81 / 26.78, 0.0423 | 30.23 / 29.24, 0.0146 | 26.87 / 22.87 |
| ork_right | **26.62 / 22.64, 0.0381** | 37.09 / 34.17, 0.0071 | 27.55 / 23.49, 0.0310 | 31.95 / 29.36 |

- Closer to today's picture than both controls in chest, knight, potion and fox; knight_back and ork_right sit at the
  controls' level and at the level of the tie-rounding variant (these two flip under any perturbation).
- By eye at 100 % (`scratchpad pv8/view/full100_<case>.png`: today | INT8 P.V | TE on CPU | no FA): the same objects,
  style and detail class; no streaks, halos, noise or banding. Bistable details flip: the orc's mechanical hand
  (the knuckle lights and claw arrangement, the knee guard) - the no-FA control flips it the same way; the
  knight_back's cape outline and right-hand pose shift by a few pixels (the alpha numbers), while TE on CPU turns its
  tassets into mail; the fox's backpack is unchanged (TE on CPU changes it and the tail).
- `GGML_CUDA_FATTN_DP4A_PV=f32` on the new engine reproduces today's chest bit for bit; the 2-step chest of 41e1ab0e
  and b1158241 is bit-identical.

### Traps met

- `subprocess.call(["bash", ...])` from a Windows Python runs WSL's bash: environment variables such as
  `GGML_CUDA_FAKE_CC` do not reach the Windows test program, which then measures the F32 tile kernel on the real
  cc 8.6 and looks perfect. Call Git's `C:\Program Files\Git\usr\bin\bash.exe` by its full path.
- Python string patches of C sources eat `\` + newline (macro continuations) and turn `\n` into a real newline.
- From step 1 on Qwen-Image's V arrives as F16 (the prefix cache), at step 0 as the F32 strided view: test both.

## Score loop register tile, 2026-10-24 (GTX 1070 pod)

An Opus agent; built locally (Docker `sdlinux`, sm_61), measured on the rented GTX 1070 pod (GPU 0, under
`flock /workspace/gpu.lock`). The INT8 P.V kernel's score loop is shared-memory bound on Pascal: a 16-byte shared load
costs ~2.2 clocks of the SM's one shared-memory pipe whether 1, 4 or 8 distinct chunks are read (`lds_bench.cu`), and
the 1 x 8 tile issued 9 LDS.128 per 32 dp4a (20 clocks of loads against 9.4 of dp4a).

**Change** (`flash_attn_tile_dp4a_pv8` only; the comment block above its lane setup has the details):
- A lane computes 2 queries x 4 keys: lane = 8*g + c, c = 2*qp + h; queries 8*w + 2*qp and + 1, keys 8*j + 4*h + g
  (j = 0..3). Per 16-byte chunk 2 + 4 LDS.128 for 32 dp4a (not 64: the planning arithmetic was off by 2), conflict-free
  (query chunks at banks 8*qp (+4), key chunks at 16*h + 4*g). 8 scores a lane are fixed by the block shape (64
  queries x 32 keys, 256 threads); 2 x 4 and 4 x 2 both need 6 loads for them, the minimum, so the loop is still
  shared-memory bound, 13.2 : 9.4 clocks (was 2.1 : 1).
- Bit-identical by construction: exact int32 scores times the same two scales in the same expression; the maxima are
  fmaxf (order-free); lane (c, g) still keeps query 8*w + c's denominator over its keys 4*i + g: the tile holds half of
  them (i = 2*j + h), lane ^ 1 the other half - the u8 codes' integer sum is completed with one shuffle (exact), an
  outlier block's F32 numerators are written to shared memory and summed back in today's order (8 LDS.32, rare);
  the numerators land in the same bytes / floats. The SASS keeps today's FMUL + FFMA shapes of the denominator update.
- The INT8 numerator rows padded to 72 words (`FDP4A_P8S`): the new byte stores were 2-way bank conflicts at 64.
- The F32 P.V kernel (`GGML_CUDA_FATTN_DP4A_PV=f32`) is untouched: its sm_61 SASS is identical before / after (both
  mask variants, `cuobjdump -sass`).
- sm_61: 128 registers, 0 spill (unmasked; the masked variant 16 B of stack, was 40), 42,112 B static shared memory
  (+256), 2 blocks per SM. Compiles for 61;70;75;80;86;89;90;100;120a (Linux nvcc 12.8) and with MSVC-hosted nvcc 12.8
  (build-sm61's command line; its sm_61 SASS equals the Linux build's).

**Measured** (GTX 1070, 15 SMs):

| measure | before (838b2f2's kernel) | 2 x 4 tile |
|---|---|---|
| sd-cli per-node dump (`SD_EVAL_DUMP`, stats off), 2nd step, 32 attention calls: median per call (two runs) | 34.39 / 33.86 ms | **27.94 / 27.83 ms (-18 %)** |
| the same, sum of the 32 calls; the whole step | 1104 / 1085 ms; 5.77 / 5.69 s | 896 / 895 ms; 5.49 / 5.45 s |
| kernel alone (`w8a8-tools/bench_attn.cu`, synthetic Qwen size, median of 30, several runs) | 29.7 - 31.1 ms | 25.2 - 26.7 ms (-15 %) |
| the same, masked variant (10 % -inf) | 37.8 ms | 30.1 ms |
| 25-step chest (seed 4242, `--vae-conv-direct`), `sampling completed` | 141.64 s (the published engine's run: 142.30) | **135.61 s** |

Where the kernel's time goes now (bench, 25.2 ms): without the score loop 16.1 ms (the loop ~9.1 ms; its dp4a alone
~6.3 ms at the measured peak, its 48 LDS.128 at 2.2 clocks ~8.9 ms - the loads still set its pace), without the INT8 P.V
dp4a loop 18.4 ms (P.V ~6.8 ms, near its dp4a bound of ~6.3), the rest ~9.3 ms (softmax, tile traffic, the 2 barriers
per 32 keys, outlier blocks).

Tried and dropped (bench): the score loop unrolled 4 instead of 8 (+0.9 - 1.3 ms); the 16-byte chunks as two LDS.64
(+0.9 ms, an 8 B spill; plain 8-byte loads are re-merged into LDS.128 by ptxas even across the unrolled chunks, so
they need `ld.volatile.shared`). A further lever would be one barrier per iteration with double-buffered K / V / P
tiles; the shared-memory budget (the outlier blocks' F32 tiles) is the obstacle. Not tried. (Tried in the next round:
slower, see "Fixed costs per iteration".)

**Identity checks** (all byte / pixel comparisons, the unmodified kernel built in the same tree as the control):
- the attention dump hook (`GGML_CUDA_FATTN_DUMP`, every query) of calls 0, 17 (step 0, V an F32 strided view), 32,
  63 (step 1, V F16): inputs and outputs byte-identical;
- the bench, 9 cases: INT8 P.V (with / without mask), `int8-f32` (every block on the F32 path), `int8-all`, the F32
  P.V kernel (with / without mask), 1100 queries x 1001 keys with GQA 2:1 (with / without mask), `int8-f32` odd
  sizes: outputs byte-identical;
- pictures: 2 steps pixel-identical (and identical to the published engine's `b2_838b2f2.png`), 2 steps with
  `GGML_CUDA_FATTN_DP4A_PV=f32` pixel-identical, 25 steps pixel-identical to the published engine's `new_direct.png`.
Not checked: a real-data masked call (Qwen-Image passes no mask; the bench's synthetic masked cases are identical),
other Pascal cards (GTX 1070 Ti / 1080 / 1060: same sm_61 code).

## Fixed costs per iteration, 2026-10-24 (GTX 1070 pod)

An Opus agent; built locally (Docker `sdlinux`, sm_61), measured on the rented GTX 1070 pod (GPU 0, under
`flock /workspace/gpu.lock`). The question: the INT8 P.V kernel spends a third of its time outside its two dp4a loops;
where, and what of it can go with a bit-identical output. Result: the "rest" has no dominant fixed cost; one change
(the iteration's branch layout + the maxima as two shuffle chains) takes a call **41.9 -> 40.1 Mclk (-4.3 %)**, in
sd-cli **27.05 -> 25.85 ms (-4.4 %, nvprof)**, bit-identical.

### Method

- What-ifs on the real kernel in `w8a8-tools/bench_attn.cu` (synthetic Qwen size: 4096 queries x 4145 keys x 32 heads,
  64 of 4160 blocks outliers), each a copy of the kernel file with parts removed (wrong results, timing only).
- **Per clock**: the GTX 1070 runs this kernel at its 130 W cap and its clock moves between runs (1,585 - 1,751 MHz
  within one run of 30 calls), and a faster variant draws more power and runs at a lower clock. The kernel file has a
  compiled-out timing hook (`-DFDP4A_BENCH_CLK`, bench builds only): thread 0 of every block writes clock64 and
  %globaltimer at its start and end and its SM; the bench prints the call in clocks (per SM the span from its first
  block's start to its last block's end in that SM's clock64; the slowest SM; median of the repetitions) and the SM
  clock (clocks / ns). Medians of 3 rounds of 30 calls (the final pair: 4 rounds); the clock figures repeat to
  +-0.1 % between rounds and sessions, the milliseconds to +-1.5 %. The hook costs nothing measurable (24.73 vs
  24.78 ms, one run each).
- **Occupancy what-if without touching the kernel**: `bench_attn --blocker` starts one 48 KB-shared-memory block per SM
  that waits beside the kernel (a chain of dependent L2 loads, a mapped host flag read once per 1000 of them), so only
  one attention block fits per SM.
- The model of the call: 2048 blocks of 8 warps, 130 iterations of 32 keys, 2 blocks per SM on 15 SMs = 8,875 "rounds"
  of 16 warps per SM. The INT8 path of an iteration is 930 warp instructions (SASS census), 512 of them IDP.4A: an
  issue bound of 930 x 16 / 4 x 8,875 = 33.0 Mclk and a dp4a bound of 18.2 Mclk, against 42.0 Mclk measured.

### What-if table (the published kernel of 1b054db + the hook; Mclk a call)

| what-if (wrong results) | Mclk | change |
|---|---|---|
| none (the published kernel) | 41.99 | |
| (e) outlier path never taken (`--mode int8-all`) | 41.29 | -0.69 (-1.7 %) |
| (e2) outlier code compiled out (int8-all) | 39.56 | -1.73 vs int8-all (-4.2 %) |
| (a) no barriers (both `__syncthreads` of the loop) | 38.97 | -3.02 (-7.2 %) |
| (a1) / (a2) only the first / only the second barrier removed (int8-all) | 40.17 / 40.11 | -1.10 / -1.16 vs int8-all |
| (b) no K / V tile staging (stale tiles; int8-all) | 40.99 | -0.29 vs int8-all (-0.7 %) |
| (b1) no global loads in the loop (stale registers stored; int8-all) | 41.21 | -0.07 vs int8-all |
| (a)+(b) no barriers, no staging (int8-all) | 38.07 | -3.21 vs int8-all |
| (c) no softmax math (scores as raw bits, no maxima shuffles, no exp2) | 40.07 | -1.92 (-4.6 %) |
| (c2) no maxima shuffles (each lane's own partial maxima) | 40.63 | -1.36 (-3.2 %) |
| (c3) no exp2 (no MUFU / RRO) | 41.71 | -0.27 (-0.7 %) |
| (c1) no score scaling / bounds tests | 43.70 | +1.71 (*) |
| (d) no numerator stores (u8 codes, scales) | 41.84 | -0.14 (-0.3 %) |
| no rescale of the accumulators | 40.69 | -1.30 (-3.1 %) |
| P.V epilogue (2 scales, FADD, FFMA) as one FADD | 39.58 | -2.41 (-5.7 %) |
| score loop's shared loads hoisted (6 LDS.128 an iteration instead of 48) | 38.94 | -3.05 (-7.3 %) |
| P.V loop's shared loads hoisted (3 instead of 24) | 38.95 | -3.04 (-7.2 %) |
| no score loop | 31.40 | -10.58 (-25 %) |
| no INT8 P.V loop | 32.24 | -9.75 (-23 %) |
| only the two dp4a loops left (everything above removed, int8-all) | 18.22 | -23.77 (-57 %) |
| (f) one block per SM (`--blocker`): 8 warps instead of 16 | 60.59 | +18.60 (+44 %); a block 441 instead of 608 kclk |

(*) raw-bit scores are huge: the running maximum moves at every block and the rescale runs every iteration.

The barriers, on the intermediate one-branch layout (int8-all 39.83 Mclk): no barriers -2.02; the barriers as per-warp
named barriers (BAR.SYNC of 32 threads + its MEMBAR.CTA, no waiting for other warps) -0.99; as half-block barriers
(4 warps) -0.17. So about half of the barriers' cost is the BAR.SYNC + MEMBAR.CTA pair itself (ptxas adds a MEMBAR.CTA
after every bar.sync on sm_61, named barriers too) and half the waiting, which does not shrink with 4-warp groups.

What it says:
- The two dp4a loops alone run at the card's dp4a issue rate (18.22 Mclk = the 18.2 bound): their shared loads cost
  nothing by themselves; they cost by competing with the rest.
- 57 % of the call is the rest, and no single fixed cost dominates it: the barriers (7 %), the P.V epilogue (5.7 %),
  the maxima shuffles (3.2 %), the rescale (3.1 %), the outlier code (1.7 % taken, 4.2 % merely present), the shared
  loads inside both loops (7 % each); staging (0.7 %), the numerator stores (0.3 %) and exp2 (0.7 %) are small. The
  instruction stream as a whole runs at 79 % of its issue bound (33.0 of 42.0 Mclk); the non-dp4a instructions (418 of
  930 an iteration) are mostly the necessary ones (score scaling, u8 quantization, the P.V epilogue: their count is
  set by the format, their order by the bit-identity).
- 16 warps per SM are far from enough to hide this kernel's latencies (one block per SM: +44 %), so whatever stalls
  a block (a barrier) is only partly covered by the other one.
- The 2.2-clock price of a broadcast LDS.128 (`lds_bench`) does not bind here: removing 42 of the score loop's loads
  saved 0.5 clock each, 21 of the P.V loop's 1.0 clock each, and narrower loads made it slower (below).

### What was changed (bit-identical; `flash_attn_tile_dp4a_pv8` only)

1. **The iteration's branch layout.** An iteration tested the block's outlier flag three times (numerators, value
   tile, P.V); ptxas placed the outlier code first each time, so the common INT8 path branched around it three times an
   iteration. Now the INT8 blocks run in an inner loop that an outlier block leaves with a `break`; the outlier block's
   code (the same F32 numerators, value tile and P.V as before) follows the loop, and the INT8 loop resumes. The
   common path has one branch to it. (ptxas still places the outlier code before the INT8 code: neither the `if/else`
   order nor `__builtin_expect` changes that.) One `if/else` per iteration measured 40.54 Mclk; this loop shape 40.09,
   with the same instruction count (932 vs 931 an iteration; a different schedule).
2. **The maxima as two chains**: both tile queries' maxima over g (xor 8, 16), then over h (xor 1): 6 shuffles 3 deep
   instead of 4 shuffles 4 deep (-0.31 Mclk alone). fmaxf is order-free: the same maxima.

Why the output is unchanged: both paths compute the same expressions in the same order as before (the scores, the u8
codes, the denominator updates, the rescale, the P.V products; the outlier code is moved, not changed), and the maxima
do not depend on the order of fmaxf. Checked by byte / pixel identity below. The F32 P.V kernel
(`GGML_CUDA_FATTN_DP4A_PV=f32`) is untouched: its sm_61 SASS is identical before / after (both mask variants); of the
library's 6,490 kernels only the four `flash_attn_tile_dp4a_pv8` variants differ from the published library's.

sm_61 resources (Linux build): unmasked variants 127 registers, 0 spill (was 128, 0); masked variants 128 registers,
8 B stack (was 16); 42,112 B static shared memory (unchanged); 2 blocks per SM.

### Measured

| measure | published (1b054db) | this change |
|---|---|---|
| bench, Mclk a call (with the timing hook; median of 4 x 30) | 41.90 (1,649 MHz) | **40.09 (-4.3 %)** (1,634 MHz) |
| the same, `--mode int8-all` / masked variant (10 % -inf) | 41.27 / 47.22 | 39.45 (-4.4 %) / 42.56 (-9.9 %) |
| bench, ms a call (no hook; median of 5 x 30, interleaved) | 25.46 | 24.72 (-2.9 %: the faster code runs at a lower clock) |
| the same, masked variant | 28.96 | 26.72 (-7.7 %) |
| sd-cli, nvprof timeline of a 2-step chest: avg ms a call (32 calls each), V as F16 / as the F32 view, two runs | 27.00 / 26.92; 27.10 / 27.19 | **25.72 / 25.77; 26.05 / 25.86** |
| 25-step chest (seed 4242, `--vae-conv-direct`), `sampling completed`, same session, one run each | 137.79 s | 135.50 s |

The per-call gain in sd-cli (-1.2 ms, -4.4 %) is ~1 s of a 25-step picture (32 calls a step x 25 steps x 1.2 ms =
0.96 s); the 25-step pair differs by 2.3 s, the rest of it is the card's clock.

### Tried and dropped (all measured; the two one-barrier variants also ran the 12 identity cases: byte-identical)

| variant | Mclk vs its base | why it was dropped |
|---|---|---|
| one barrier an iteration: K / V tiles double-buffered, the outlier path without shared tiles (numerators by shuffles, V straight from global) | +4.17 vs published (int8-all +2.27) | 24 B spill; an outlier block 4x dearer; 30.5 KB shared |
| the same with the outlier numerators in shared memory and its F32 values in half tiles (47.9 KB shared) | +2.20 vs the one-if/else layout | 24 B spill; its single barrier costs 2.6 Mclk, more than the two barriers of the one-if/else layout (2.0) |
| the next tiles' global loads after the second barrier | +0.07 | the MEMBAR.CTA after the barrier does not wait for them (or it does not matter) |
| the global loads after the P.V loop (registers free during it) | +0.68 | 8 B spill |
| the rescale after the second barrier | +2.27 | 8 B spill |
| the numerator words of the P.V loop as 4 volatile LDS.64 / 8 LDS.32 (not 2 LDS.128) | +0.24 / +3.37 | more instructions; the broadcast-load price does not bind |
| the rescale skipped per query whose maximum did not move | +0.06 | |
| the outlier P.V not unrolled (and its staging) | -0.33 / -0.14 (old layout); -0.11 on the one-if/else one | noise level |
| 32-bit element offsets for the tile loads | +0.37 (masked -1.29) | the unmasked variant is the one used |

### Identity checks (all run)

- bench (`bench_attn`, the published kernel file vs this one, same flags), 12 cases byte-identical: INT8 P.V with /
  without mask, `int8-f32` with / without mask, `int8-all`, the F32 P.V kernel with / without mask, 1100 x 1001 GQA 2:1
  with / without mask, `int8-f32` 1100 x 1001, 1024 queries x 77 keys, `int8-f32` 1024 x 40 keys.
- engine (`/workspace/sdcpp-attn2` = a copy of the published engine with this `libggml-cuda.so`): the 2-step chest
  pixel-identical to the published engine's 2-step chest (and to the previous round's `b2_v1.png`); the 2-step chest
  with `GGML_CUDA_FATTN_DP4A_PV=f32` pixel-identical to the published engine's (and to `f32_v1.png`); the 25-step chest
  pixel-identical to `/workspace/tests/v2/new_direct.png` (and to the published engine's 25-step chest of the same
  session).
- Compiles for 61;70;75;80;86;89;90;100;120a with Linux nvcc 12.8 (GCC host) and with MSVC-hosted nvcc 12.8
  (build-sm61's command line; sm_61 alone and all architectures).

### Traps

- **MSVC-hosted nvcc is not deterministic for these kernels.** Of 8 Windows compiles (sm_61) of this file and its
  variants, 3 produced one kernel each with another register allocation and schedule (once `pv8<0, half>`, once each
  F32 P.V kernel variant, whose source did not change); the other 5, among them 3 in a row of the same file and the
  final file, matched the Linux build kernel for kernel, and the Linux builds always repeated. A Windows-vs-Linux SASS
  comparison from one compile proves nothing either way. The other `pv8<0, half>` (its sm_61 cubin run on the pod
  through the driver API, a scratch variant of the bench): byte-identical output; 24.33 - 24.66 against 24.26 -
  24.59 ms for the Linux code, interleaved (not measured in clocks).
- `--blocker`: polling the mapped flag in a tight loop made the attention kernel 10x slower; CUDA's lazy loading
  deadlocks when the attention kernel is first loaded while the blocker spins (launch it once before it, or run with
  `CUDA_MODULE_LOADING=EAGER`); `cudaDeviceSynchronize` waits for the blocker (sync the stream instead).
- Small register changes move this kernel by several percent (127 vs 128 registers, an 8 B spill): measure every
  variant, per clock.
- On this power-capped card a faster kernel runs at a lower clock: -4.3 % in clocks was -2.9 % in ms in the bench.

### Not verified

- The Windows build on a real GTX 10xx (its sm_61 code runs on the Linux pod; see Traps for the compile variance).
- Other Pascal cards (GTX 1060 / 1070 Ti / 1080: the same sm_61 code); a real-data masked call (Qwen-Image passes no
  mask; the bench's masked cases are byte-identical).
- The 25-step time beyond one run each.
