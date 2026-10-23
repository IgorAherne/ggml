#!/bin/bash
# raster.sh exe: interleaved rounds of the tile orders (W8A8_GROUP) on the three shapes ([1] with the GLU f16 epilogue)
for r in 1 2 3; do
  for g in 1 2 3 4 -2 -4 -8; do
    for s in "4096 4096 4096 0" "4096 24576 4096 1" "4096 4096 12288 0"; do
      set -- $s
      echo "$g $1 $2 $3 $(W8A8_GROUP=$g W8A8_EPI=$4 ./$EXE bench $1 $2 $3 20 | tail -1 | sed -E 's/.*epilogue ([0-9.]+) ms.*/\1/')"
    done
  done
done | sort -k1,1n -k3,3n -k4,4n | awk '{k=$1" "$2" "$3" "$4; v[k]=v[k]" "$5} END {for (k in v) print k, v[k]}' | sort -k3,3n -k4,4n -k1,1n
