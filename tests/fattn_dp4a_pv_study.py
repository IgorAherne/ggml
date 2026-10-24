"""Format study for an INT8 P.V in the GTX 10xx dp4a flash attention, on real Q/K/V dumped by the engine
(GGML_CUDA_FATTN_DUMP, fattn-tile-dp4a.cu). Per output row (one query of one head) the relative L2 error against an
F64 reference; the variants:

  kernel   : the dumped output of today's kernel (INT8 scores, F32 P.V)
  cur      : emulation of today's kernel (INT8 Q per row, key-mean-smoothed K INT8 per row, F64 softmax and P.V)
  f16pv    : exact scores, P (relative to the row max) and V rounded to F16 (the tensor-core kernels' P.V class)
  pv:<fmt> : exact scores, P.V in the format <fmt>
  all:<fmt>: INT8 scores as today + P.V in <fmt>

<fmt> = [H<r>:]P<bits>[f].V<gran>[m][.lexact]: P as unsigned <bits>-bit integers with one scale per (query, 32-key
block) = the block's max p / (2^bits - 1); V int8 with one scale per channel (Vc), per (channel, 32-key block) (Vb),
per key (Vt), per key after a per-channel normalisation (Vct), 16 bits per channel (V2c) or per key (V2t), or exact
(Vx); f = the per-key V scale folded into P before its quantisation; m = V minus its per-channel key mean (added back
after the normalisation: exact algebra); .lexact = the denominator from the unquantised p; H<r>: = blocks with a key
whose |V| max is above r x or below 1/r x the mean over the keys stay exact (the kernel's outlier blocks).
The kernel (GGML_CUDA_FATTN_DP4A_PV=int8) is H4:P8.Vb; "kernel-vs-emu" compares a dump of it with this emulation.

python fattn_dp4a_pv_study.py <fa####.bin> [...] [--heads=0-31] [--fmts a,b] [--all a,b]   (numpy; ~1 min a dump)
Dumps: sd-cli with GGML_CUDA_FAKE_CC=610 (or a real cc 6.1 card), GGML_CUDA_FATTN_DUMP=<dir>,
GGML_CUDA_FATTN_DUMP_CALLS=0,13,31,400 (calls of the dp4a kernel; Qwen-Image 2.1: 32 a step), see fattn-tile-dp4a.cu.
"""
import sys
import numpy as np

LOG2E = 1.4426950408889634


def load(path):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(36), dtype=np.int32)
        magic, D, nqs, nk, nh, nkh, qstride, kt, vt = [int(x) for x in hdr]
        assert magic == 0x34414446, hex(magic)
        scale = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
        Q = np.frombuffer(f.read(nh * nqs * D * 4), dtype=np.float32).reshape(nh, nqs, D)
        kdt = np.float16 if kt == 1 else np.float32
        vdt = np.float16 if vt == 1 else np.float32
        K = np.frombuffer(f.read(nkh * nk * D * np.dtype(kdt).itemsize), dtype=kdt).reshape(nkh, nk, D)
        V = np.frombuffer(f.read(nkh * nk * D * np.dtype(vdt).itemsize), dtype=vdt).reshape(nkh, nk, D)
        rest = f.read()
        O = np.frombuffer(rest, dtype=np.float32).reshape(nh, nqs, D) if len(rest) == nh * nqs * D * 4 else None
    return dict(D=D, nqs=nqs, nk=nk, nh=nh, nkh=nkh, qstride=qstride, scale=scale, Q=Q, K=K, V=V, O=O)


def q8_rows(x):
    """INT8 per row (last axis), scale = absmax/127; returns (int values as f64, scales)."""
    amax = np.abs(x).max(axis=-1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 0.0)
    q = np.where(s > 0, np.clip(np.rint(x / np.where(s > 0, s, 1.0)), -127, 127), 0.0)
    return q, s


def scores_exact(Qh, Kh, scale):
    return (Qh.astype(np.float64) @ Kh.astype(np.float64).T) * (scale * LOG2E)   # log2 units


def scores_int8(Qh, Kh, scale):
    Qs = Qh.astype(np.float32) * np.float32(scale * LOG2E)                      # the kernel scales Q in f32 first
    q, qs = q8_rows(Qs.astype(np.float64))
    Kd = Kh.astype(np.float64)
    Km = Kd - Kd.mean(axis=0, keepdims=True)
    k, ks = q8_rows(Km)
    return (q @ k.T) * qs * ks.T


def blocks(nk, nb=32):
    return [(b, min(b + nb, nk)) for b in range(0, nk, nb)]


def pv_f64(S, V):
    m = S.max(axis=1, keepdims=True)
    P = np.exp2(S - m)
    return (P @ V) / P.sum(axis=1, keepdims=True)


def pv_f16(S, V):
    m = S.max(axis=1, keepdims=True)
    P = np.exp2(S - m).astype(np.float16).astype(np.float64)
    return (P @ V.astype(np.float16).astype(np.float64)) / P.sum(axis=1, keepdims=True)


def quant_v(V, gran, mean, nk):
    """V (f64 [nk][D]) -> dequantised V8 (f64) and the mean to add back."""
    mu = V.mean(axis=0, keepdims=True) if mean else np.zeros((1, V.shape[1]))
    X = V - mu
    if gran == "c":
        s = np.abs(X).max(axis=0, keepdims=True) / 127.0
        s = np.where(s > 0, s, 1.0)
        return np.clip(np.rint(X / s), -127, 127) * s, mu
    if gran == "t":
        s = np.abs(X).max(axis=1, keepdims=True) / 127.0
        s = np.where(s > 0, s, 1.0)
        return np.clip(np.rint(X / s), -127, 127) * s, mu
    assert gran == "b"
    out = np.empty_like(X)
    for b0, b1 in blocks(nk):
        s = np.abs(X[b0:b1]).max(axis=0, keepdims=True) / 127.0
        s = np.where(s > 0, s, 1.0)
        out[b0:b1] = np.clip(np.rint(X[b0:b1] / s), -127, 127) * s
    return out, mu


def pv_int(S, V, fmt):
    """fmt: P<bits>[f].V<gran>[m][.lexact]; P..f = a per-token V scale folded into P (gran t); V2c = V as 16 bits
    (two int8 digits) per channel; Vx = V unquantised"""
    hyb = None
    if fmt.startswith("H"):
        hyb, fmt = fmt[1:].split(":")
        hyb = float(hyb)
    parts = fmt.split(".")
    pspec = parts[0]
    fold = pspec.endswith("f")
    pbits = int(pspec[1:].rstrip("f"))
    vg = parts[1][1:].rstrip("m")
    vmean = parts[1].endswith("m")
    lexact = "lexact" in parts
    pmax_int = (1 << pbits) - 1
    nq, nk = S.shape
    if vg == "x":
        Vq, mu = V, np.zeros((1, V.shape[1]))
    elif vg == "2c":
        mu = np.zeros((1, V.shape[1]))
        s = np.abs(V).max(axis=0, keepdims=True) / 32767.0
        s = np.where(s > 0, s, 1.0)
        Vq = np.clip(np.rint(V / s), -32767, 32767) * s
    elif fold:
        mu = V.mean(axis=0, keepdims=True) if vmean else np.zeros((1, V.shape[1]))
    else:
        Vq, mu = quant_v(V, vg, vmean, nk)
    if fold:
        # per-token V scale folded into P; "ct": V first divided by its per-channel absmax over the keys (applied back
        # to the output); "2t": V as 16 bits per token (two int8 digits)
        assert vg in ("t", "ct", "2t", "2ct")
        X = V - mu
        cd = np.ones((1, V.shape[1]))
        if "c" in vg:
            cd = np.abs(X).max(axis=0, keepdims=True)
            cd = np.where(cd > 0, cd, 1.0)
        Xn = X / cd
        lv = 32767.0 if vg.startswith("2") else 127.0
        vs = np.abs(Xn).max(axis=1) / lv
        vs = np.where(vs > 0, vs, 1e-30)
        V8 = np.clip(np.rint(Xn / vs[:, None]), -lv, lv) * cd
    M = S.max(axis=1, keepdims=True)
    flagged = set()
    if hyb is not None:
        ta = np.abs(V).max(axis=1)
        med = np.mean(ta)                       # the kernel's rule: the mean over the keys
        for bi, (b0, b1) in enumerate(blocks(nk)):
            if ta[b0:b1].max() > hyb * med or ta[b0:b1].min() < med / hyb:
                flagged.add(b0)
        HYB_STATS.append(len(flagged) / len(blocks(nk)))
    num = np.zeros((nq, V.shape[1]))
    den = np.zeros((nq, 1))
    den_exact = np.zeros((nq, 1))
    for b0, b1 in blocks(nk):
        Sb = S[:, b0:b1]
        if b0 in flagged:
            Pb = np.exp2(Sb - M)
            num += Pb @ (V[b0:b1] - mu)
            den += Pb.sum(axis=1, keepdims=True)
        elif fold:
            W = np.exp2(Sb - M) * vs[b0:b1]                          # p * the key's V scale
            wmax = W.max(axis=1, keepdims=True)
            wmax = np.where(wmax > 0, wmax, 1.0)
            qp = np.rint(pmax_int * W / wmax)
            sc = wmax / pmax_int
            num += sc * (qp @ V8[b0:b1])
            den += sc * (qp / vs[b0:b1]).sum(axis=1, keepdims=True)  # the implied p, consistent with the numerator
        else:
            smax = Sb.max(axis=1, keepdims=True)
            qp = np.rint(pmax_int * np.exp2(Sb - smax))             # 0 .. pmax_int, the block max exactly pmax_int
            sc = np.exp2(smax - M) / pmax_int
            num += sc * (qp @ Vq[b0:b1])
            den += sc * qp.sum(axis=1, keepdims=True)
        den_exact += np.exp2(Sb - M).sum(axis=1, keepdims=True)
    o = num / (den_exact if lexact else den)
    return o + mu * (den / (den_exact if lexact else den))


HYB_STATS = []


def rel_rows(o, ref):
    return np.linalg.norm(o - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)


def main():
    args = [a for a in sys.argv[1:] if a.endswith(".bin")]
    heads = None
    for a in sys.argv[1:]:
        if a.startswith("--heads="):
            lo, hi = a.split("=")[1].split("-")
            heads = range(int(lo), int(hi) + 1)
    fmts = [f for f in (sys.argv[sys.argv.index("--fmts") + 1].split(",") if "--fmts" in sys.argv else
            ["P8.Vx", "P8.Vc", "P8.Vb", "P8.Vt", "P8f.Vt", "P8f.Vtm", "P8.V2c", "P8.Vc.lexact"])]
    allf = [f for f in (sys.argv[sys.argv.index("--all") + 1].split(",") if "--all" in sys.argv else ["P8.Vc", "P8f.Vt", "P8.V2c"])]
    for path in args:
        d = load(path)
        nh, nkh, nk = d["nh"], d["nkh"], d["nk"]
        hs = heads or range(nh)
        acc = {}
        absr = {}
        refn = []
        fro = {}
        vstats = []
        for h in hs:
            kh = h // (nh // nkh)
            Qh, Kh, Vh = d["Q"][h], d["K"][kh], d["V"][kh].astype(np.float64)
            Se = scores_exact(Qh, Kh, d["scale"])
            Si = scores_int8(Qh, Kh, d["scale"])
            ref = pv_f64(Se, Vh)
            refn.append(np.linalg.norm(ref, axis=1))
            res = {"cur": pv_f64(Si, Vh), "f16pv": pv_f16(Se, Vh)}
            if d["O"] is not None:
                res["kernel"] = d["O"][h].astype(np.float64)
                res["kernel-vs-cur"] = None
            for f in fmts:
                res["pv:" + f] = pv_int(Se, Vh, f)
            for f in allf:
                res["all:" + f] = pv_int(Si, Vh, f)
            if d["O"] is not None and "all:H4:P8.Vb" in res:
                res["kernel-vs-emu"] = None
            for k, o in res.items():
                if k in ("kernel-vs-cur", "kernel-vs-emu"):
                    other = res["cur"] if k == "kernel-vs-cur" else res["all:H4:P8.Vb"]
                    e = rel_rows(res["kernel"], other)
                    nrm = (np.linalg.norm(res["kernel"] - other), np.linalg.norm(other))
                else:
                    e = rel_rows(o, ref)
                    nrm = (np.linalg.norm(o - ref), np.linalg.norm(ref))
                acc.setdefault(k, []).append(e)
                ab = np.linalg.norm((res["kernel"] - (res["cur"] if k == "kernel-vs-cur" else res["all:H4:P8.Vb"]))
                                    if k in ("kernel-vs-cur", "kernel-vs-emu") else (o - ref), axis=1)
                absr.setdefault(k, []).append(ab)
                a = fro.setdefault(k, [0.0, 0.0])
                a[0] += nrm[0] ** 2
                a[1] += nrm[1] ** 2
            # V statistics: channel absmax over keys vs channel RMS; token norm spread; the softmax's spread
            ch_amax = np.abs(Vh).max(axis=0)
            ch_rms = np.sqrt((Vh ** 2).mean(axis=0))
            tok = np.linalg.norm(Vh, axis=1)
            P = np.exp2(Se - Se.max(axis=1, keepdims=True))
            eff = (P.sum(axis=1) ** 2) / (P ** 2).sum(axis=1)                 # effective number of keys
            top = P.max(axis=1) / P.sum(axis=1)
            vstats.append((np.median(ch_amax / ch_rms), np.max(ch_amax / ch_rms), np.max(tok) / np.median(tok),
                           np.argmax(tok), np.median(eff), np.percentile(eff, 1), np.median(top), np.max(top)))
        print(f"== {path}: nq_sub {d['nqs']} (every {d['qstride']}th query), nk {nk}, heads {len(hs)}/{nh}, scale {d['scale']:.6g}")
        vs = np.array(vstats)
        print(f"   V: channel absmax/rms median {np.median(vs[:,0]):.1f} max {vs[:,1].max():.1f}; token norm max/median "
              f"median {np.median(vs[:,2]):.1f} max {vs[:,2].max():.1f} (argmax tokens {sorted(set(vs[:,3].astype(int)))[:8]})")
        print(f"   softmax: effective keys median {np.median(vs[:,4]):.0f}, 1st pct {np.min(vs[:,5]):.1f}; top weight median "
              f"{np.median(vs[:,6]):.3f} max {vs[:,7].max():.3f}")
        rms = np.sqrt(np.mean(np.concatenate(refn) ** 2))
        print(f"   {'variant':22s} {'rel/row: median':>15s} {'p90':>9s} {'p99':>9s} {'max':>9s} | {'frob':>9s} | "
              f"{'err/rms-row: p50':>16s} {'p99':>9s} {'max':>9s}")
        for k, es in acc.items():
            e = np.concatenate(es)
            a = np.concatenate(absr[k]) / rms
            fr = np.sqrt(fro[k][0] / fro[k][1])
            print(f"   {k:22s} {np.median(e):15.2e} {np.percentile(e, 90):9.2e} {np.percentile(e, 99):9.2e} {e.max():9.2e} | {fr:9.2e} | "
                  f"{np.median(a):16.2e} {np.percentile(a, 99):9.2e} {a.max():9.2e}")
        if HYB_STATS:
            print(f"   hybrid: flagged block fraction mean {np.mean(HYB_STATS):.3f} max {np.max(HYB_STATS):.3f}")
            HYB_STATS.clear()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
