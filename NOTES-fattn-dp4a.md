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
