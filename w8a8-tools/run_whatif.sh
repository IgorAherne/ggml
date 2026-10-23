#!/bin/bash
# times the what-if builds on the three Qwen shapes (T = 4096)
for exe in "$@"; do
  for s in "4096 4096 4096" "4096 24576 4096" "4096 4096 12288"; do
    echo -n "$exe: "; ./$exe bench $s 20 | tail -1
  done
done
