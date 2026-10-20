# SageAttention CUDA device code

Adapted from [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention),
revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, under Apache-2.0 (see LICENSE).

`attn.cuh` contains the device kernel from
`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`; `attn_fp8.cuh` adapts
`csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`. The remaining headers come from `csrc/`
and `csrc/qattn/attn_utils.cuh`. PyTorch includes and host wrappers were removed;
include paths and guards were adjusted, device helpers were placed in the
`sageattention` namespace, and `div_ceil` was renamed to avoid macro collisions.
Unused `<cuda/pipeline>` includes were removed so multi-architecture builds can
also compile targets below SM70; the backend rejects SageAttention below SM80.
Predicated loads now zero unused FP16 V lanes: masked probabilities cannot
suppress NaNs left in shared memory by previous kernels.

The ggml adapter in `../sage-attn.cu` supplies FP32 K smoothing and native CUDA
per-thread INT8 quantization, matching the token grouping in upstream
`sageattention/triton/quant_per_thread.py`. V is quantized per channel and
transposed, padded with zeros, and permuted according to
`csrc/fused/fused.cu`. Its FP8 range is 448 for SageAttention2 and 2.25 for
SageAttention2++. Zero channels use a positive scale to avoid division by zero.
SageAttention2 uses an FP32 instruction buffer; 2++ uses an FP16 instruction
buffer added to FP32 accumulators. The SM80 compatibility path retains per-warp
INT8 Q/K and FP16 V with FP32 accumulation.

The adapter owns allocation and stream selection; it has no Python, PyTorch,
or Triton dependency. The public output is converted back to FP32.

## Spellcast3D changes (github.com/IgorAherne/ggml, branch spellcast-perf2)

Spellcast3D runs Qwen-Image 2.1 (head 128, 32 heads, 4,096 picture tokens and 56+ prompt tokens at 1024x1024)
through this op. Its changes, measured on an RTX 3060 12 GB (sm_86, 28 SMs, CUDA 12.8):

- `attn.cuh`: the kernel template takes `bool tuned` (default `false`: the kernel compiles exactly as before).
  `tuned` is the launch the Spellcast3D round J study chose for the per-warp INT8 / FP16 + FP32 head-128 kernel
  (github.com/IgorAherne/SageAttention, README-round-j.md; -13 % a call there, the same bits as the untuned kernel):
  `__launch_bounds__(256, 2)` (two blocks an SM, 128 registers) and the P.V step one 16-column output tile at a time
  (`sage_tuned_sv`). The kernel also writes an F32 output (`DTypeOut = float`) straight from its FP32 accumulators,
  without the FP16 rounding and the shared-memory pass of the half / bf16 outputs.
- `../sage-attn.cu`: `GGML_SAGE_ATTN_AUTO` runs that kernel (`SAGE_PATH_TUNED`: Q in INT8 per 16 rows, K in INT8 per
  64 keys after the key mean is subtracted, V in FP16, P.V accumulated in FP16 over each 64-key block and then in
  FP32, F32 output) for head 128 on every card from cc 8.6 on - the RTX 30 series and newer, where an FP16 MMA with
  FP16 accumulation runs at twice the rate of one with FP32 accumulation - including the RTX 40 / 50 series, where
  upstream chose the FP8 kernels (`attn_fp8.cuh`): those were not checked in pictures (no such card here), the INT8 /
  FP16 one was. Head 64 and cc 8.0 - 8.5 keep upstream's SM80 path. `GGML_CUDA_SAGE=tuned|sm80|fp8|fp8pp` in the
  environment picks a path where the card has it (for comparisons). Q and K are taken in F16 as well as F32, V as any
  F16 view whose rows are contiguous (read through its strides: e.g. a permuted [D, heads, tokens] tensor, without a
  copy); the scratch (`SageWorkspace`) holds only what the chosen path uses.

Measured, one call at Qwen-Image 2.1's shape (4,096 queries x 4,152 keys x 32 heads x 128, random inputs; medians):
flash attention (`ggml_flash_attn_ext`, F16 K/V) 18.0 ms; upstream's SM80 path 7.62 ms; the tuned path 5.79 ms
(the attention kernel ~5.0 ms, the key mean and the Q / K quantization the rest). Output against flash attention's:
relative L2 1.28e-2 (upstream's SM80 path 1.30e-2), the smallest cosine of an output row 0.9997. In Qwen-Image 2.1
(25 steps, 1024x1024, Q4_K_M, 2026-10-21) a step took 1.97-2.00 s with flash attention, 1.76-1.78 s with upstream's
path and 1.61 s with the tuned one; with one 1024x1024 example picture (8,250 keys) 2.47-2.56 s -> 1.78-1.83 s.
Pictures of six prompts (four text-only, two with an example picture) against flash attention's: PSNR 27-42 dB,
LPIPS 0.004-0.034 - the same compositions, colours and sharpness, small details placed differently (the sampler
amplifies any rounding change over its 25 steps; the example-picture cases most). Deterministic: two runs, the same
bits.
