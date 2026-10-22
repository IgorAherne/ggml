#!/bin/sh
# res.sh <instance name e.g. q4_k> [grep pattern]: register/stack/shared usage of the instance's kernels
B=/c/_myDrive/repos/spellcast3d/temp_code/ggml-mmq3/build-sm86/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances
CUO="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cuobjdump.exe"
"$CUO" -res-usage "$B/mmq-instance-$1.cu.obj" 2>&1 | grep -A1 "Function" | grep -v "^--" | paste - - | sed 's/Function \([^ ]*\):/\1/' | awk '{print $1, $2, $3, $4, $6}' | grep "${2:-Li128}"
