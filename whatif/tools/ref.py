"""ref.py <type q4_K|q5_K> <K> <N> <T> <prefix> <out dump> [<out dump> ...] [--tokens n]
Reference checks of mm outputs from the inputs dumped by mmq3bench (MMBENCH_DUMP_IN=<prefix>):
 exact  = dequantized weights x float activations (float64)
 ds4    = emulation of the per-32 (d, s) path (async / default): f16(d*sc) * f16(d_y) * C + f16(-dmin*m) * f16(s)
 dsy    = emulation of the ring path: DSY activations, weights centred (q - centre), min term f16(centre*d*sc - dmin*m) * f16(s)
prints rel L2 / max|d|/rms of each dump against each emulation, on the first n tokens."""
import sys
import numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ntok, stride = 256, 1
for i, a in enumerate(sys.argv):
    if a == "--tokens": ntok = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
    if a == "--stride": stride = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
typ, K, N, T, prefix = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
dumps = args[5:]
DSY_MAX = 255
bs = 144 if typ == "q4_K" else 176
W = np.fromfile(prefix + "_in0.bin", np.uint8).reshape(N, K // 256, bs)
X = np.fromfile(prefix + "_in1.bin", np.float32).reshape(T, K)[::stride][:ntok]
ntok = X.shape[0]

def f16(a): return np.asarray(a, np.float32).astype(np.float16).astype(np.float64)

d = W[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float64)       # [N, nb]
dmin = W[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
sc8 = W[:, :, 4:16].astype(np.int64)                                       # [N, nb, 12]
sc = np.zeros((N, K // 256, 8), np.int64); m = np.zeros_like(sc)
for j in range(8):
    if j < 4:
        sc[..., j] = sc8[..., j] & 63; m[..., j] = sc8[..., j + 4] & 63
    else:
        sc[..., j] = (sc8[..., j + 4] & 0xF) | ((sc8[..., j - 4] >> 6) << 4)
        m[..., j] = (sc8[..., j + 4] >> 4) | ((sc8[..., j] >> 6) << 4)
if typ == "q4_K":
    qs = W[:, :, 16:144].astype(np.int64); qh = None
else:
    qh = W[:, :, 16:48].astype(np.int64); qs = W[:, :, 48:176].astype(np.int64)
q = np.zeros((N, K // 256, 8, 32), np.int64)                                # value index within sub-block
for j2 in range(4):
    lo = qs[:, :, 32 * j2:32 * j2 + 32] & 0xF; hi = qs[:, :, 32 * j2:32 * j2 + 32] >> 4
    if qh is not None:
        lo = lo + ((qh >> (2 * j2)) & 1) * 16; hi = hi + ((qh >> (2 * j2 + 1)) & 1) * 16
    q[:, :, 2 * j2] = lo; q[:, :, 2 * j2 + 1] = hi
centre = 8 if typ == "q4_K" else 16
Wf = (d[..., None, None] * sc[..., None] * q - dmin[..., None, None] * m[..., None]).reshape(N, K)
exact = X.astype(np.float64) @ Wf.T                                          # [ntok, N]

# activations: [ntok, K/32, 32]
X32 = X.reshape(ntok, K // 32, 32)
# DS4 (per 32): d = amax/127 (float32), q = round(x*127/amax), s = f16(sum)
amax = np.abs(X32).max(-1).astype(np.float32)
dinv = (np.float32(127.0) / amax).astype(np.float32)
qy4 = np.round(X32 * dinv[..., None]).astype(np.int64)                      # roundf vs rint: ties negligible
dy4 = f16(np.float32(1.0) / dinv)
s32 = f16(X32.astype(np.float32).sum(-1))
# DSY (per 128 + sy)
X128 = X.reshape(ntok, K // 128, 4, 32)
am = np.abs(X128).max(-1).astype(np.float32)                                 # [ntok, K/128, 4]
am128 = am.max(-1).astype(np.float32)
with np.errstate(divide="ignore", invalid="ignore"):
    sy = np.where(am128[..., None] > 0, np.minimum(DSY_MAX, np.ceil((np.float32(DSY_MAX) * am / am128[..., None]).astype(np.float32))), 0).astype(np.int64)
    amp = (am128[..., None] * sy.astype(np.float32) / np.float32(DSY_MAX)).astype(np.float32)
    dinvy = np.where(amp > 0, (np.float32(127.0) / amp).astype(np.float32), np.float32(0))
qyy = np.clip(np.round(X128 * dinvy[..., None]), -127, 127).astype(np.int64)
d128 = (am128 / np.float32(127.0)).astype(np.float64)  # f32 in the header

nb = K // 256
# per 256-block, per sub-block: integer dots C[t, n, b, j] = sum_k q * y
qw = q.reshape(N, nb * 8, 32)
def emulate(kind):
    out = np.zeros((ntok, N))
    for b in range(nb):
        for j in range(8):
            jj = b * 8 + j
            if kind == "ds4":
                C = qy4[:, jj, :] @ qw[:, jj, :].T                                 # [ntok, N]
                out += C * (dy4[:, jj][:, None] * f16(d[:, b] * sc[:, b, j])[None, :])
                out += s32[:, jj][:, None] * f16(-dmin[:, b] * m[:, b, j])[None, :]
            else:
                blk, sj = jj // 4, jj % 4
                C = qyy[:, blk, sj, :] @ (qw[:, jj, :] - centre).T
                out += C * sc[:, b, j][None, :] * sy[:, blk, sj][:, None] * (d128[:, blk][:, None] / DSY_MAX) * d[:, b][None, :]
                cj = f16(d[:, b] * centre * sc[:, b, j] - dmin[:, b] * m[:, b, j])
                out += s32[:, jj][:, None] * cj[None, :]
    return out
refs = {"exact": exact, "ds4": emulate("ds4"), "dsy": emulate("dsy")}
rms = np.sqrt((exact ** 2).mean())
def cmp(a, b): d_ = a - b; return f"rel L2 {np.linalg.norm(d_)/np.linalg.norm(b):.3e} max|d|/rms {np.abs(d_).max()/rms:.3e}"
print(f"{typ} K={K} N={N} T={T} ({ntok} tokens, every {stride}th)")
print(f"  ds4 emu vs exact: {cmp(refs['ds4'], exact)}")
print(f"  dsy emu vs exact: {cmp(refs['dsy'], exact)}")
for f in dumps:
    o = np.fromfile(f, np.float32).reshape(T, N)[::stride][:ntok].astype(np.float64)
    print(f"  {f.split(chr(92))[-1]:18s} " + " | ".join(f"vs {k}: {cmp(o, v)}" for k, v in refs.items()))
