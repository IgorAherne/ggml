"""kt.py <bin dir> "<w8a8bench args after the bin dir>" ...: per-kernel GPU times (nsys CUDA trace) of w8a8bench runs
under the GPU lock (after the mmq3 agent's kt.py). Environment variables of this process pass through (e.g.
GGML_CUDA_W8A8_KERNEL=off for MMQ). Prints the bench line and, per kernel, the median / min time and the count."""
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, r"C:\_myDrive\repos\spellcast3d\code\spellcast\profiling")
import gpu_lock  # noqa: E402

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe"
HERE = Path(__file__).resolve().parent
EXE = HERE / "w8a8bench.exe"
OUT = HERE / "out" / "nsys"
OUT.mkdir(parents=True, exist_ok=True)
bin_dir = str(Path(sys.argv[1]).resolve())
tag = os.environ.get("KT_TAG", "r")
env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
with gpu_lock.gpu_lock("QS4", "w8a8 nsys kernel times", expected_minutes=5):
    for k, a in enumerate(sys.argv[2:]):
        rep = OUT / f"{tag}{k}"
        for ext in (".nsys-rep", ".sqlite"):
            if (OUT / f"{tag}{k}{ext}").exists():
                (OUT / f"{tag}{k}{ext}").unlink()
        p = subprocess.run([NSYS, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "-o", str(rep), str(EXE), bin_dir,
                            *a.split()], env=env, cwd=bin_dir, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if "median" in l]
        print(a, "|", line[0] if line else p.stdout[-500:] + p.stderr[-1500:], flush=True)
        s = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(rep) + ".nsys-rep"],
                           capture_output=True, text=True)
        txt = s.stdout[s.stdout.find('"Time (%)"') if '"Time (%)"' in s.stdout else s.stdout.find("Time (%)"):]
        for row in csv.DictReader(io.StringIO(txt)):
            name = row["Name"][:110]
            print(f"   {float(row['Med (ns)'])/1e3:9.1f} us med  {float(row['Min (ns)'])/1e3:9.1f} min  x{row['Instances']:>4}  {name}",
                  flush=True)
