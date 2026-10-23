"""compare.py a.bin b.bin [more pairs]: f32 output dumps (MMBENCH_DUMP) compared: relative L2 difference, the largest
difference relative to the output RMS, the largest relative difference among outputs with |ref| > 0.1 RMS (non-cancelling),
the share of bit-identical values. a = reference."""
import sys
import numpy as np

args = sys.argv[1:]
for i in range(0, len(args) - 1, 2):
    a = np.fromfile(args[i], dtype=np.float32).astype(np.float64)
    b = np.fromfile(args[i + 1], dtype=np.float32).astype(np.float64)
    if a.shape != b.shape:
        print(f"{args[i]} vs {args[i + 1]}: sizes differ {a.shape} {b.shape}")
        continue
    d = b - a
    rms = np.sqrt(np.mean(a * a))
    big = np.abs(a) > 0.1 * rms
    rel_big = np.max(np.abs(d[big]) / np.abs(a[big])) if big.any() else 0.0
    print(f"{args[i + 1]} vs {args[i]}: n {a.size}, rel L2 {np.linalg.norm(d) / np.linalg.norm(a):.3e}, "
          f"max |diff|/rms {np.max(np.abs(d)) / rms:.3e}, max rel (|ref| > 0.1 rms) {rel_big:.3e}, "
          f"identical {np.mean(a == b) * 100:.2f} %, nan/inf {int((~np.isfinite(b)).sum())}")
