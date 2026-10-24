# FlashAttention with INT8 (dp4a) Q.K^T scores for GTX 10xx: notes (2026-10-24)

Committed as c6e96c02 (in the engine release spellcast-perf-6a82e8d). The sections below up to "Real GTX 1070 Ti"
are the writing agent's, measured on the RTX 3060 emulation (compute_61 code, `GGML_CUDA_FAKE_CC=610`); the real
card's figures are at the end. `build-sm61` (compute_61 only) holds a build of this state, with the path on by default.

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
| chest | 34.69 / 33.84 dB | 37.09 dB RGB (vs the shipped build) | 32.34 dB RGB |
| knight | 30.34 / 28.28 dB | 31.44 / 23.70 dB | 29.69 / 23.73 dB |
| potion | 38.03 / 33.59 dB | - | - |

The real card diverges more than the emulation (chest 42 -> 35 dB): a 25-step sampler amplifies any rounding change,
and the rounding-only controls move these pictures by as much or more. By eye: the same objects and quality; bistable
details flip - the chest's lock gains a raised hasp (the RTX 3060's own picture has it too; the text encoder on the
CPU gives a third, flat variant), the knight's strap ring grows, while both controls also lengthen the tabard strip
and add a sword hilt. The release keeps the path on for cc 6.x.
