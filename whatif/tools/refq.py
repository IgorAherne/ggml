"""refq.py <type> <K> <N> <T> <prefix> [--tokens n]: error vs the exact product of activation quantization schemes x
weight centring (numpy emulation on the inputs dumped by mmq3bench). Schemes: per-G activation scale (G = 32, 64, 128)
with the f16-rounded s sums for the min term; DSY (d128 + 8/7-bit sy per 32); weights uncentred (min term -dmin*m) or
centred (q - c in the int dot, c*d*sc - dmin*m in the min term)."""
import sys
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ntok = 256
for i, a in enumerate(sys.argv):
    if a == "--tokens": ntok = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
typ, K, N, T, prefix = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
bs = 144 if typ == "q4_K" else 176
W = np.fromfile(prefix + "_in0.bin", np.uint8).reshape(N, K // 256, bs)
X = np.fromfile(prefix + "_in1.bin", np.float32).reshape(T, K)[:: max(1, T // ntok)][:ntok]
def f16(a): return np.asarray(a, np.float32).astype(np.float16).astype(np.float64)
d = W[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
dmin = W[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
sc8 = W[:, :, 4:16].astype(np.int64)
sc = np.zeros((N, K // 256, 8), np.int64); m = np.zeros_like(sc)
for j in range(8):
    if j < 4: sc[..., j] = sc8[..., j] & 63; m[..., j] = sc8[..., j + 4] & 63
    else:
        sc[..., j] = (sc8[..., j + 4] & 0xF) | ((sc8[..., j - 4] >> 6) << 4)
        m[..., j] = (sc8[..., j + 4] >> 4) | ((sc8[..., j] >> 6) << 4)
if typ == "q4_K": qs = W[:, :, 16:144].astype(np.int64); qh = None
else: qh = W[:, :, 16:48].astype(np.int64); qs = W[:, :, 48:176].astype(np.int64)
q = np.zeros((N, K // 256, 8, 32), np.int64)
for j2 in range(4):
    lo = qs[:, :, 32 * j2:32 * j2 + 32] & 0xF; hi = qs[:, :, 32 * j2:32 * j2 + 32] >> 4
    if qh is not None: lo = lo + ((qh >> (2 * j2)) & 1) * 16; hi = hi + ((qh >> (2 * j2 + 1)) & 1) * 16
    q[:, :, 2 * j2] = lo; q[:, :, 2 * j2 + 1] = hi
cen = 8 if typ == "q4_K" else 16
dsc = (d[..., None] * sc).reshape(N, K // 32)                        # exact d*sc per sub-block
dm = (dmin[..., None] * m).reshape(N, K // 32)
qf = q.reshape(N, K // 32, 32).astype(np.float64)
Wf = (dsc[..., None] * qf - dm[..., None]).reshape(N, K)
exact = X.astype(np.float64) @ Wf.T
s32 = f16(X.reshape(ntok, K // 32, 32).astype(np.float32).sum(-1))  # [ntok, K/32]

def quant_G(G):  # per-G scale, values as float (dequantized), float32 arithmetic like quantize_mmq_q8_1
    XG = X.reshape(ntok, K // G, G)
    am = np.abs(XG).max(-1).astype(np.float32)
    with np.errstate(divide="ignore"):
        dinv = (np.float32(127.0) / am).astype(np.float32)
    qy = np.round(XG * dinv[..., None])
    return (qy * f16(np.float32(1.0) / dinv)[..., None]).reshape(ntok, K)   # f16 d as the DS4 header stores it
def quant_dsy(MAX):
    X128 = X.reshape(ntok, K // 128, 4, 32)
    am = np.abs(X128).max(-1).astype(np.float32); am128 = am.max(-1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sy = np.where(am128[..., None] > 0, np.minimum(MAX, np.ceil((np.float32(MAX) * am / am128[..., None]).astype(np.float32))), 0)
        amp = (am128[..., None] * sy.astype(np.float32) / np.float32(MAX)).astype(np.float32)
        dinv = np.where(amp > 0, (np.float32(127.0) / amp).astype(np.float32), np.float32(0))
    qy = np.clip(np.round(X128 * dinv[..., None]), -127, 127)
    scale = (am128 / np.float32(127.0)).astype(np.float64)[..., None] * sy / MAX
    return (qy * scale[..., None]).reshape(ntok, K)
def product(Xq, centred):
    # main term with the quantized activations; min term with the exact-ish f16 sums s32
    if centred:
        Wc = (dsc[..., None] * (qf - cen)).reshape(N, K)
        cst = f16(cen * dsc - dm)                                        # [N, K/32] rounded once to f16
        return Xq @ Wc.T + s32 @ cst.T
    Wq = (dsc[..., None] * qf).reshape(N, K)
    return Xq @ Wq.T + s32 @ f16(-dm).T
rn = np.linalg.norm(exact)
print(f"{typ} K={K} N={N} ({ntok} tokens): rel L2 error vs the exact float64 product")
for name, Xq in (("per-32 (DS4, today)", quant_G(32)), ("per-64", quant_G(64)), ("per-128", quant_G(128)),
                 ("DSY 8-bit sy", quant_dsy(255)), ("DSY 7-bit sy", quant_dsy(127))):
    e0 = np.linalg.norm(product(Xq, False) - exact) / rn
    e1 = np.linalg.norm(product(Xq, True) - exact) / rn
    print(f"  {name:22s} uncentred {e0:.3e}   centred {e1:.3e}")
