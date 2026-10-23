#!/bin/bash
# verify1.sh: ggml-level outputs of the W8A8 path against the research emulation in the same DLL (and MMQ for scale)
# out/<case>_{emu,k,kn,kf32,mmq}.bin; compare.py on each pair
BIN=../build-sm86/bin
export PATH="$(cd $BIN && pwd):$PATH"
D=/c/Users/Igor/AppData/Local/Temp/claude/C---myDrive-repos-spellcast3d/28dae9df-ec87-4365-a233-4b80e1acef34/scratchpad/qs4
run() { # name op type K N T [act dump]
  local name=$1; shift
  local act=""; [ -n "$6" ] && act="$6"
  W8A8BENCH_ACT=$act GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:5 MMBENCH_DUMP=out/${name}_emu.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 2 | sed "s/^/emu   /"
  W8A8BENCH_ACT=$act MMBENCH_DUMP=out/${name}_k.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/w8a8  /"
  W8A8BENCH_ACT=$act GGML_CUDA_DISABLE_FUSION=1 MMBENCH_DUMP=out/${name}_kn.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/nofuse/"
  W8A8BENCH_ACT=$act GGML_CUDA_W8A8_KERNEL=off MMBENCH_DUMP=out/${name}_mmq.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/mmq   /"
  if [ $1 = glu ]; then
    W8A8BENCH_ACT=$act GGML_CUDA_W8A8_GLU=f32 MMBENCH_DUMP=out/${name}_kf32.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/gluf32/"
  fi
}
run glu8360   glu q4_K 4096 24576 8360
run glu1811   glu q4_K 4096 24576 1811
run mmk12_8360 mm q4_K 12288 4096 8360
run mmk12_1811 mm q4_K 12288 4096 1811
run mmq5k12   mm q5_K 12288 4096 4145
for c in glu8360 glu1811 mmk12_8360 mmk12_1811 mmq5k12; do
  for v in k kn kf32 mmq; do
    [ -f out/${c}_$v.bin ] && /c/_myDrive/repos/spellcast3d/code/venvs/pixal3d/Scripts/python.exe compare.py out/${c}_emu.bin out/${c}_$v.bin
  done
done
