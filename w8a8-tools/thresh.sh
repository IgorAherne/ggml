#!/bin/bash
# thresh.sh: the W8A8 path (GGML_CUDA_W8A8_MIN_TOKENS=1) against MMQ (GGML_CUDA_W8A8_KERNEL=off) per token count, and the
# device memory of both paths (glu and mm at T 4096 / 8360); ggml-level medians (w8a8bench)
BIN=../build-sm86/bin
export PATH="$(cd $BIN && pwd):$PATH"
for T in 128 256 384 512 768 1024 1536 2048 4096; do
  for s in "mm q4_K 4096 4096" "mm q4_K 12288 4096" "glu q4_K 4096 24576"; do
    a=$(GGML_CUDA_W8A8_MIN_TOKENS=1 ./w8a8bench.exe $BIN $s $T 10 | grep median | sed -E 's/.*median ([0-9.]+) ms.*/\1/')
    b=$(GGML_CUDA_W8A8_KERNEL=off ./w8a8bench.exe $BIN $s $T 10 | grep median | sed -E 's/.*median ([0-9.]+) ms.*/\1/')
    echo "T $T $s: w8a8 $a ms, mmq $b ms"
  done
done
for T in 4096 8360; do
  for s in "glu q4_K 4096 24576" "mm q4_K 12288 4096" "mm3 q4_K 4096 4096"; do
    echo "T $T $s w8a8: $(./w8a8bench.exe $BIN $s $T 3 | grep 'device memory')"
    echo "T $T $s mmq:  $(GGML_CUDA_W8A8_KERNEL=off ./w8a8bench.exe $BIN $s $T 3 | grep 'device memory')"
  done
done
