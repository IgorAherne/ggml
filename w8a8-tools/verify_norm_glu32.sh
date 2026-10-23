#!/bin/bash
# the GLU cases of verify_norm.sh with GGML_CUDA_W8A8_GLU=f32 in all three modes (the fused GLU chain with an f32
# intermediate equals ggml's own SWIGLU, so "no fusion" is a reference for the whole chain there)
export GGML_CUDA_W8A8_GLU=f32
source <(sed -n '/^BIN=/,/^}/p' verify_norm.sh)
for T in 4096 4145 8360 1811; do
  case_run random "" normglu q4_K 4096 24576 $T
  case_run real "$D/dump_knight/000256.bin" normglu q4_K 4096 24576 $T
done
