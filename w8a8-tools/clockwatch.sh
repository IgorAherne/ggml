#!/bin/bash
# runs the given command while sampling the SM clock / power / temperature every 100 ms; prints min/median/max
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 100 > clk.log &
P=$!
"$@"
kill $P
sort -n clk.log | awk -F, '{c[NR]=$1; p[NR]=$2; t[NR]=$3} END {print "samples", NR, "clock min/med/max", c[1], c[int(NR/2)], c[NR]}'
awk -F, '{print $4}' clk.log | sort | uniq -c
awk -F, '{s+=$2; if ($3>m) m=$3} END {print "mean power", s/NR, "max temp", m}' clk.log
