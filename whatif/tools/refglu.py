"""refglu.py <type> <K> <N> <T> <prefix> <out dumps...> [--stride s]: glu (gate|up -> swiglu -> down) outputs vs the exact
float64 computation from the dumped inputs (W [N, K], X [T, K], Wd [K, N/2])."""
import sys
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
stride = 16
for i, a in enumerate(sys.argv):
    if a == "--stride": stride = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
typ, K, N, T, prefix = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
dumps = args[5:]
bs = 144 if typ == "q4_K" else 176
def deq(raw, rows, cols):
    W = raw.reshape(rows, cols // 256, bs)
    d = W[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    dmin = W[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
    sc8 = W[:, :, 4:16].astype(np.int64)
    sc = np.zeros((rows, cols // 256, 8), np.int64); m = np.zeros_like(sc)
    for j in range(8):
        if j < 4: sc[..., j] = sc8[..., j] & 63; m[..., j] = sc8[..., j + 4] & 63
        else:
            sc[..., j] = (sc8[..., j + 4] & 0xF) | ((sc8[..., j - 4] >> 6) << 4)
            m[..., j] = (sc8[..., j + 4] >> 4) | ((sc8[..., j] >> 6) << 4)
    if typ == "q4_K": qs = W[:, :, 16:144].astype(np.int64); qh = None
    else: qh = W[:, :, 16:48].astype(np.int64); qs = W[:, :, 48:176].astype(np.int64)
    q = np.zeros((rows, cols // 256, 8, 32), np.int64)
    for j2 in range(4):
        lo = qs[:, :, 32 * j2:32 * j2 + 32] & 0xF; hi = qs[:, :, 32 * j2:32 * j2 + 32] >> 4
        if qh is not None: lo = lo + ((qh >> (2 * j2)) & 1) * 16; hi = hi + ((qh >> (2 * j2 + 1)) & 1) * 16
        q[:, :, 2 * j2] = lo; q[:, :, 2 * j2 + 1] = hi
    return (d[..., None, None] * sc[..., None] * q - dmin[..., None, None] * m[..., None]).reshape(rows, cols)
W = deq(np.fromfile(prefix + "_in0.bin", np.uint8), N, K)
X = np.fromfile(prefix + "_in1.bin", np.float32).reshape(T, K)[::stride].astype(np.float64)
Wd = deq(np.fromfile(prefix + "_in2.bin", np.uint8), K, N // 2)
h = X @ W.T
g, u = h[:, :N // 2], h[:, N // 2:]
a = g / (1.0 + np.exp(-g)) * u
exact = a @ Wd.T
rn = np.linalg.norm(exact); rms = np.sqrt((exact ** 2).mean())
print(f"glu {typ} K={K} N={N} T={T} ({X.shape[0]} tokens): vs exact float64")
for f in dumps:
    o = np.fromfile(f, np.float32).reshape(T, K)[::stride].astype(np.float64)
    dd = o - exact
    print(f"  {f.split(chr(92))[-1]:14s} rel L2 {np.linalg.norm(dd)/rn:.3e}  max|d|/rms {np.abs(dd).max()/rms:.3e}")
