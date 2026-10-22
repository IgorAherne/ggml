"""refrun.py "<type K N T>" ...: run mmq3bench (worktree build) per pipe with input/output dumps, then ref.py."""
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, r"C:\_myDrive\repos\spellcast3d\code\spellcast\profiling")
import gpu_lock
HERE = Path(__file__).resolve().parent
EXE = str(HERE / "tool" / "mmq3bench.exe")
BIN = os.environ.get("MMQ3_BIN", r"C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3\build-sm86\bin")
D = HERE / "dumps"; D.mkdir(exist_ok=True)
PIPES = os.environ.get("MMQ3_PIPES", "off,async,ring").split(",")
for a in sys.argv[1:]:
    typ, K, N, T = a.split()
    outs = []
    with gpu_lock.gpu_lock("QS", "mmq3 ref", expected_minutes=2):
        for p in PIPES:
            env = dict(os.environ, PATH=BIN + os.pathsep + os.environ["PATH"], GGML_CUDA_MMQ_PIPE=p,
                       MMBENCH_DUMP=str(D / f"o_{p}.bin"))
            if p == PIPES[0]: env["MMBENCH_DUMP_IN"] = str(D / "in")
            r = subprocess.run([EXE, BIN, "mm", typ, K, N, T, "2"], env=env, cwd=BIN, capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if "median" in l]
            print(p, line[0] if line else r.stdout[-500:] + r.stderr[-800:])
            outs.append(str(D / f"o_{p}.bin"))
    subprocess.run([sys.executable, str(HERE / "ref.py"), typ, K, N, T, str(D / "in"), *outs] + os.environ.get("MMQ3_REFARGS", "").split())
