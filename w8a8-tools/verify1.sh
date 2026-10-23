#!/bin/bash
# verify1.sh: ggml-level outputs of the W8A8 path against the research emulation in the same DLL (and MMQ for scale)
# out/<case>_{emu,k,kn,kf32,mmq}.bin; compare.py on each pair
BIN=../build-sm86/bin
export PATH="$(cd $BIN && pwd):$PATH"
D=/c/Users/Igor/AppData/Local/Temp/claude/C---myDrive-repos-spellcast3d/28dae9df-ec87-4365-a233-4b80e1acef34/scratchpad/qs4
run() { # name op type K N T [act dump]
  local name=$1; shift
  local act=""; [ -n "$6" ] && act="$6"
  W8A8BENCH_ACT=$act GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:8 MMBENCH_DUMP=out/${name}_emu.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 2 | sed "s/^/emu   /"
  W8A8BENCH_ACT=$act MMBENCH_DUMP=out/${name}_k.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/w8a8  /"
  W8A8BENCH_ACT=$act GGML_CUDA_DISABLE_FUSION=1 MMBENCH_DUMP=out/${name}_kn.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/nofuse/"
  W8A8BENCH_ACT=$act GGML_CUDA_W8A8_KERNEL=off MMBENCH_DUMP=out/${name}_mmq.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/mmq   /"
  if [ $1 = glu ]; then
    W8A8BENCH_ACT=$act GGML_CUDA_W8A8_GLU=f32 MMBENCH_DUMP=out/${name}_kf32.bin ./w8a8bench.exe $BIN $1 $2 $3 $4 $5 5 | sed "s/^/gluf32/"
  fi
}
run mm4096   mm q4_K 4096 4096 4096
run mm8360   mm q4_K 4096 4096 8360
run mm4145   mm q4_K 4096 4096 4145
run mm1811   mm q4_K 4096 4096 1811
run mmq5     mm q5_K 4096 4096 4096
run mmk12    mm q4_K 12288 4096 4096
run scale    scalemm q5_K 4096 4096 4145
run mm3      mm3 q4_K 4096 4096 4145
run glu4096  glu q4_K 4096 24576 4096
run glu4145  glu q4_K 4096 24576 4145
run real_q   mm q5_K 4096 4096 4145 $D/dump_knight/000252.bin
run real_dn  mm q4_K 12288 4096 4145 $D/dump_knight/000257.bin
run real_glu glu q4_K 4096 24576 4145 $D/dump_knight/000256.bin
for c in mm4096 mm8360 mm4145 mm1811 mmq5 mmk12 scale mm3 glu4096 glu4145 real_q real_dn real_glu; do
  for v in k kn kf32 mmq; do
    [ -f out/${c}_$v.bin ] && /c/_myDrive/repos/spellcast3d/code/venvs/pixal3d/Scripts/python.exe compare.py out/${c}_emu.bin out/${c}_$v.bin
  done
done
