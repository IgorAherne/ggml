"""pcmp.py "<mmbench args>" ...: for each shape, run mmbench on the worktree build with GGML_CUDA_MMQ_PIPE = off, async,
ring (interleaved, 3 rounds, medians of mmbench's own medians), dump the outputs and compare them (rel L2, max|d|/rms)."""
import os, subprocess, sys, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\_myDrive\repos\spellcast3d\code\spellcast\profiling")
import gpu_lock
MB = r"C:\_myDrive\repos\spellcast3d\code\devtools\qwen-speed\bench\mmbench.exe"
BIN = os.environ.get("MMQ3_BIN", r"C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3\build-sm86\bin")
OUT = Path(r"C:\Users\Igor\AppData\Local\Temp\claude\C---myDrive-repos-spellcast3d\28dae9df-ec87-4365-a233-4b80e1acef34\scratchpad\mmq3\dumps")
OUT.mkdir(exist_ok=True)
PIPES = os.environ.get("MMQ3_PIPES", "off,async,ring").split(",")
ROUNDS = int(os.environ.get("MMQ3_ROUNDS", "3"))
def run(pipe, args, dump=None):
    env = dict(os.environ, PATH=BIN + os.pathsep + os.environ["PATH"], GGML_CUDA_MMQ_PIPE=pipe)
    if dump: env["MMBENCH_DUMP"] = str(dump)
    p = subprocess.run([MB, BIN, *args], env=env, cwd=BIN, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if "median" in l]
    if not lines:
        raise RuntimeError(p.stdout[-800:] + p.stderr[-1500:])
    return float(lines[0].split("median ")[1].split(" ms")[0]), lines[0]
with gpu_lock.gpu_lock("QS", "mmq3 pcmp", expected_minutes=5):
    for a in sys.argv[1:]:
        args = a.split()
        t = {p: [] for p in PIPES}
        for r in range(ROUNDS):
            order = PIPES if r % 2 == 0 else PIPES[::-1]
            for p in order:
                t[p].append(run(p, args + ["10"])[0])
        outs, lines = {}, {}
        for p in PIPES:
            f = OUT / f"{p}.bin"
            lines[p] = run(p, args + ["2"], f)[1]
            outs[p] = np.fromfile(f, np.float32).astype(np.float64)
        flops = 2.0 * float(args[2]) * float(args[3]) * float(args[4]) * (1.5 if args[0] == "glu" else 1.0)
        print(f"{a}:")
        for p in PIPES:
            ms = statistics.median(t[p])
            print(f"   {p:6s} {ms:8.3f} ms  {flops/ms/1e9:6.1f} T/s   ({' '.join(f'{x:.3f}' for x in t[p])})  {lines[p].split('hash ')[1].split(',')[0]}")
        ref = outs[PIPES[0]]
        rms = np.sqrt((ref*ref).mean())
        for i, p in enumerate(PIPES):
            for q in PIPES[i+1:]:
                d = outs[q] - outs[p]
                print(f"   {q} vs {p}: rel L2 {np.linalg.norm(d)/np.linalg.norm(outs[p]):.3e}  max|d|/rms {np.abs(d).max()/rms:.3e}  nan {int(np.isnan(outs[q]).sum())}")
