#!/bin/sh
# bench.sh <label> [extra shapes...]: the main shapes on the worktree build, appended to the bench log
L=/c/_myDrive/repos/spellcast3d/temp_code/ggml-mmq3/bench-log.txt
label=$1; shift
cd /c/_myDrive/repos/spellcast3d/code/devtools/qwen-speed/bench/
echo "== $label  $(date '+%F %T')  PIPE=${GGML_CUDA_MMQ_PIPE:-unset}" >> $L
/c/_myDrive/repos/spellcast3d/code/venvs/pixal3d/Scripts/python.exe mb.py 'C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3\build-sm86\bin' \
  "mm q4_K 4096 4096 4096" "glu q4_K 4096 24576 4096" "mm q4_K 12288 4096 4096" "mm q5_K 4096 4096 4096" "mm q4_K 3072 3072 4096" "$@" 2>&1 | tee -a $L
