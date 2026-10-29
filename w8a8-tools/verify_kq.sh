#!/bin/bash
# verify_kq.sh: the W8A8 path with Q6_K / Q8_0 / Q3_K / Q2_K weights (branch spellcast-w8a8-kquants), w8a8bench level.
#   verify_kq.sh regress <old backend dir> [<new backend dir>]
#       Q4_K / Q5_K output hashes (verify1's and verify_norm's cases, fusion on) of two builds: must be identical
#   verify_kq.sh emu [part]
#       the new types against the research emulation in the same DLL (GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:5),
#       the path fused (k) and unfused (kn), MMQ (GGML_CUDA_W8A8_KERNEL=off) for scale; out/kq_<case>_*.bin, compare.py
#   verify_kq.sh norm [T ...]
#       mixed-type q/k/v and GLU chains behind the fused norm writer: fused = norm off = no fusion, the byte check
# A backend dir holds ggml-cuda.dll (+ ggml-cpu-*.dll); ggml.dll / ggml-base.dll / the CUDA runtime come from
# ../build-sm86/bin (PATH). Run each mode under the GPU lock.
cd "$(dirname "$0")"
BIN=${BIN:-../build-sm86/bin}
export PATH="$(cd ../build-sm86/bin && pwd):$PATH"
D=/c/Users/Igor/AppData/Local/Temp/claude/C---myDrive-repos-spellcast3d/28dae9df-ec87-4365-a233-4b80e1acef34/scratchpad/qs4
PY=/c/_myDrive/repos/spellcast3d/code/venvs/pixal3d/Scripts/python.exe
mkdir -p out
hash_of() { sed -nE 's/.*hash ([0-9a-f]+).*/\1/p' | head -1; }

regress() {
  local old=$1 new=${2:-$BIN} n=0 bad=0
  while read -r label act types args; do
    [ -z "$label" ] && continue
    [ "$act" = "-" ] && act=""
    [ "$types" = "-" ] && types=""
    h1=$(W8A8BENCH_ACT=$act W8A8BENCH_TYPES=$types ./w8a8bench.exe $old $args 2 | hash_of)
    h2=$(W8A8BENCH_ACT=$act W8A8BENCH_TYPES=$types ./w8a8bench.exe $new $args 2 | hash_of)
    n=$((n+1)); [ "$h1" = "$h2" ] && s=IDENTICAL || { s=DIFFERENT; bad=$((bad+1)); }
    echo "$label $args: old $h1 new $h2 $s"
  done << EOF
mm4096 - - mm q4_K 4096 4096 4096
mm8360 - - mm q4_K 4096 4096 8360
mm4145 - - mm q4_K 4096 4096 4145
mm1811 - - mm q4_K 4096 4096 1811
mmq5 - - mm q5_K 4096 4096 4096
mmk12 - - mm q4_K 12288 4096 4096
mmq5k12 - - mm q5_K 12288 4096 4145
scale - - scalemm q5_K 4096 4096 4145
mm3 - - mm3 q4_K 4096 4096 4145
glu4096 - - glu q4_K 4096 24576 4096
glu4145 - - glu q4_K 4096 24576 4145
real_q $D/dump_knight/000252.bin - mm q5_K 4096 4096 4145
real_dn $D/dump_knight/000257.bin - mm q4_K 12288 4096 4145
real_glu $D/dump_knight/000256.bin - glu q4_K 4096 24576 4145
normmm3 - - normmm3 q4_K 4096 4096 4145
normmm3_real $D/dump_knight/000252.bin - normmm3 q5_K 4096 4096 4145
normglu - - normglu q4_K 4096 24576 4145
normglu_real $D/dump_knight/000256.bin - normglu q4_K 4096 24576 4145
mixed45 - q5_K,q4_K,q5_K mm3 q4_K 4096 4096 4145
EOF
  echo "regress: $n cases, $bad different"
}

emu_case() { # name act types op type K N T
  local name=$1 act=$2 types=$3; shift 3
  [ "$act" = "-" ] && act=""
  [ "$types" = "-" ] && types=""
  local E="W8A8BENCH_ACT=$act W8A8BENCH_TYPES=$types"
  env $E GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_W8A8=sub:5 MMBENCH_DUMP=out/kq_${name}_emu.bin ./w8a8bench.exe $BIN "$@" 2 | sed "s/^/emu    $name /" | grep median
  env $E MMBENCH_DUMP=out/kq_${name}_k.bin ./w8a8bench.exe $BIN "$@" 5 | sed "s/^/w8a8   $name /" | grep median
  env $E GGML_CUDA_DISABLE_FUSION=1 MMBENCH_DUMP=out/kq_${name}_kn.bin ./w8a8bench.exe $BIN "$@" 5 | sed "s/^/nofuse $name /" | grep median
  env $E GGML_CUDA_W8A8_KERNEL=off MMBENCH_DUMP=out/kq_${name}_mmq.bin ./w8a8bench.exe $BIN "$@" 5 | sed "s/^/mmq    $name /" | grep median
  if [ "$1" = glu ] || [ "$1" = normglu ]; then
    env $E GGML_CUDA_W8A8_GLU=f32 MMBENCH_DUMP=out/kq_${name}_kf32.bin ./w8a8bench.exe $BIN "$@" 5 | sed "s/^/gluf32 $name /" | grep median
  fi
  for v in k kn kf32 mmq; do
    [ -f out/kq_${name}_$v.bin ] && $PY compare.py out/kq_${name}_emu.bin out/kq_${name}_$v.bin
  done
  rm -f out/kq_${name}_*.bin
}

emu() {
  local part=${1:-all}
  if [ $part = all ] || [ $part = 1 ]; then
    emu_case q6_mm4096   - - mm q6_K 4096 4096 4096
    emu_case q6_mm4145   - - mm q6_K 4096 4096 4145
    emu_case q6_mm1811   - - mm q6_K 4096 4096 1811
    emu_case q6_mm8360   - - mm q6_K 4096 4096 8360
    emu_case q6_k12      - - mm q6_K 12288 4096 4145
    emu_case q8_mm4096   - - mm q8_0 4096 4096 4096
    emu_case q8_mm4145   - - mm q8_0 4096 4096 4145
    emu_case q8_mm8360   - - mm q8_0 4096 4096 8360
    emu_case q8_k12      - - mm q8_0 12288 4096 4145
  fi
  if [ $part = all ] || [ $part = 2 ]; then
    emu_case q3_mm4145   - - mm q3_K 4096 4096 4145
    emu_case q3_k12      - - mm q3_K 12288 4096 4145
    emu_case q2_mm4145   - - mm q2_K 4096 4096 4145
    emu_case q6_scale    - - scalemm q6_K 4096 4096 4145
    emu_case q8_scale    - - scalemm q8_0 4096 4096 4145
    emu_case mix_mm3     - q5_K,q8_0 mm3 q5_K 4096 4096 4145
    emu_case real_q_q8   $D/dump_knight/000252.bin - mm q8_0 4096 4096 4145
    emu_case real_q_q6   $D/dump_knight/000252.bin - mm q6_K 4096 4096 4145
    emu_case real_dn_q6  $D/dump_knight/000257.bin - mm q6_K 12288 4096 4145
  fi
  if [ $part = all ] || [ $part = 3 ]; then
    emu_case q6_glu4145  - - glu q6_K 4096 24576 4145
    emu_case q6q5_glu    - -,-,q5_K glu q6_K 4096 24576 4145
    emu_case q8_glu4145  - - glu q8_0 4096 24576 4145
    emu_case real_glu_q6 $D/dump_knight/000256.bin - glu q6_K 4096 24576 4145
  fi
}

norm_case() { # label act types op type K N T
  local label=$1 act=$2 types=$3; shift 3
  [ "$act" = "-" ] && act=""
  [ "$types" = "-" ] && types=""
  local E="W8A8BENCH_ACT=$act W8A8BENCH_TYPES=$types"
  local h1 h2 h3 chk
  h1=$(env $E ./w8a8bench.exe $BIN "$@" 3 | grep median)
  h2=$(env $E GGML_CUDA_W8A8_NORM=off ./w8a8bench.exe $BIN "$@" 3 | grep median)
  h3=$(env $E GGML_CUDA_DISABLE_FUSION=1 ./w8a8bench.exe $BIN "$@" 3 | grep median)
  chk=$(env $E GGML_CUDA_W8A8_CHECK_NORM=1 ./w8a8bench.exe $BIN "$@" 2 2>&1 | grep "norm writer check" | tail -1 | sed -E 's/.*: ([0-9]+ bytes differ \(total [0-9]+\)).*/\1/')
  x1=$(echo "$h1" | hash_of); x2=$(echo "$h2" | hash_of); x3=$(echo "$h3" | hash_of)
  t1=$(echo "$h1" | sed -E 's/.*median ([0-9.]+) ms.*/\1/'); t2=$(echo "$h2" | sed -E 's/.*median ([0-9.]+) ms.*/\1/'); t3=$(echo "$h3" | sed -E 's/.*median ([0-9.]+) ms.*/\1/')
  if [ "$x1" = "$x2" ] && [ "$x1" = "$x3" ]; then same="all three IDENTICAL"; elif [ "$x1" = "$x2" ]; then same="fused = norm off"; else same=DIFFERENT; fi
  echo "$label [$types] $*: fused $x1 ($t1 ms) | norm off $x2 ($t2 ms) | no fusion $x3 ($t3 ms) -> $same | check: $chk"
}

norm() {  # [T ...] (default 4145 1811); the GLU f32 case with T 4145
  for T in ${@:-4145 1811}; do
    norm_case random - q5_K,q8_0 normmm3 q5_K 4096 4096 $T
    norm_case random - q6_K,q8_0 normmm3 q6_K 4096 4096 $T
    norm_case real $D/dump_knight/000252.bin q5_K,q8_0 normmm3 q5_K 4096 4096 $T
    norm_case random - -,-,q5_K normglu q6_K 4096 24576 $T
    norm_case real $D/dump_knight/000256.bin -,-,q6_K normglu q6_K 4096 24576 $T
  done
  case " ${*:-4145} " in *" 4145 "*) GGML_CUDA_W8A8_GLU=f32 norm_case glu_f32 - -,-,q5_K normglu q6_K 4096 24576 4145;; esac
}

mode=$1; shift
$mode "$@"
