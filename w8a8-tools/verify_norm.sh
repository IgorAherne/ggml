#!/bin/bash
# verify_norm.sh: the norm writer fused into the W8A8 activations against the unfused chain, same DLL:
#   fused (default) | GGML_CUDA_W8A8_NORM=off (norm f32, then the quantizer) | GGML_CUDA_DISABLE_FUSION=1 (every op alone)
#   | GGML_CUDA_W8A8_CHECK_NORM=1 (the byte check of q_x / sy / d_t inside the fusion); output hashes must be equal
BIN=../build-sm86/bin
export PATH="$(cd $BIN && pwd):$PATH"
D=/c/Users/Igor/AppData/Local/Temp/claude/C---myDrive-repos-spellcast3d/28dae9df-ec87-4365-a233-4b80e1acef34/scratchpad/qs4
case_run() { # label act op type K N T
  local label=$1 act=$2; shift 2
  local h1 h2 h3 chk
  h1=$(W8A8BENCH_ACT=$act ./w8a8bench.exe $BIN $@ 5 | grep median)
  h2=$(W8A8BENCH_ACT=$act GGML_CUDA_W8A8_NORM=off ./w8a8bench.exe $BIN $@ 5 | grep median)
  h3=$(W8A8BENCH_ACT=$act GGML_CUDA_DISABLE_FUSION=1 ./w8a8bench.exe $BIN $@ 5 | grep median)
  chk=$(W8A8BENCH_ACT=$act GGML_CUDA_W8A8_CHECK_NORM=1 ./w8a8bench.exe $BIN $@ 2 2>&1 | grep "norm writer check" | tail -1 | sed -E 's/.*: ([0-9]+ bytes differ \(total [0-9]+\)).*/\1/')
  x1=$(echo "$h1" | sed -E 's/.*hash ([0-9a-f]+).*/\1/'); x2=$(echo "$h2" | sed -E 's/.*hash ([0-9a-f]+).*/\1/'); x3=$(echo "$h3" | sed -E 's/.*hash ([0-9a-f]+).*/\1/')
  t1=$(echo "$h1" | sed -E 's/.*median ([0-9.]+) ms.*/\1/'); t2=$(echo "$h2" | sed -E 's/.*median ([0-9.]+) ms.*/\1/'); t3=$(echo "$h3" | sed -E 's/.*median ([0-9.]+) ms.*/\1/')
  same=$([ "$x1" = "$x2" ] && [ "$x1" = "$x3" ] && echo IDENTICAL || echo DIFFERENT)
  echo "$label $*: fused $x1 ($t1 ms) | norm off $x2 ($t2 ms) | no fusion $x3 ($t3 ms) -> $same | check: $chk"
}
for T in 4096 4145 8360 1811; do
  case_run random "" normmm3 q4_K 4096 4096 $T
  case_run real "$D/dump_knight/000252.bin" normmm3 q5_K 4096 4096 $T
  case_run random "" normglu q4_K 4096 24576 $T
  case_run real "$D/dump_knight/000256.bin" normglu q4_K 4096 24576 $T
done
