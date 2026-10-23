#!/bin/bash
# cmp.sh exe1 exe2 ... : ROUNDS (3) interleaved rounds of every exe on the three Qwen shapes (T = 4096); prints per
# exe and shape the median ms of the rounds and the T/s of that median (each round = the median of 20 timed runs).
# GLU_EPI=1 times the [gate|up] shape (N = 24576) with the SwiGLU f16 epilogue (the real use) instead of f32.
ROUNDS=${ROUNDS:-3}
SHAPES=("4096 4096 4096" "4096 24576 4096" "4096 4096 12288")
declare -A R
for r in $(seq $ROUNDS); do
  for exe in "$@"; do
    for i in 0 1 2; do
      epi=0
      if [ $i = 1 ]; then epi=${GLU_EPI:-0}; fi
      ms=$(W8A8_EPI=$epi ./$exe bench ${SHAPES[$i]} 20 | tail -1 | sed -E 's/.*epilogue ([0-9.]+) ms.*/\1/')
      R[$exe,$i]="${R[$exe,$i]} $ms"
    done
  done
done
for exe in "$@"; do
  line="$exe:"
  for i in 0 1 2; do
    set -- ${SHAPES[$i]}
    med=$(echo ${R[$exe,$i]} | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    tops=$(awk -v t=$1 -v n=$2 -v k=$3 -v ms=$med 'BEGIN {printf "%.1f", 2*t*n*k/ms/1e9}')
    line="$line  [$i] $med ms $tops T/s ($(echo ${R[$exe,$i]}))"
  done
  echo "$line"
done
