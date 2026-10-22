"""kt.py <bin dir> "<mmbench args>" ...: per-kernel GPU times (nsys CUDA trace) of mmbench runs, under the GPU lock."""
import os, subprocess, sys, csv, io
from pathlib import Path
sys.path.insert(0, r"C:\_myDrive\repos\spellcast3d\code\spellcast\profiling")
import gpu_lock
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe"
MB = r"C:\_myDrive\repos\spellcast3d\code\devtools\qwen-speed\bench\mmbench.exe"
OUT = Path(r"C:\Users\Igor\AppData\Local\Temp\claude\C---myDrive-repos-spellcast3d\28dae9df-ec87-4365-a233-4b80e1acef34\scratchpad\mmq3\nsys")
OUT.mkdir(exist_ok=True)
bin_dir = sys.argv[1]
env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"])
with gpu_lock.gpu_lock("QS", "mmq3 nsys", expected_minutes=3):
    for k, a in enumerate(sys.argv[2:]):
        rep = OUT / f"r{k}"
        for ext in (".nsys-rep", ".sqlite"):
            if (OUT / f"r{k}{ext}").exists(): (OUT / f"r{k}{ext}").unlink()
        p = subprocess.run([NSYS, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "-o", str(rep), MB, bin_dir, *a.split()],
                           env=env, cwd=bin_dir, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if "median" in l]
        print(a, "|", line[0] if line else p.stdout[-500:] + p.stderr[-1500:])
        s = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(rep) + ".nsys-rep"],
                           capture_output=True, text=True)
        txt = s.stdout[s.stdout.find('"Time (%)"') if '"Time (%)"' in s.stdout else s.stdout.find("Time (%)"):]
        for row in csv.DictReader(io.StringIO(txt)):
            name = row["Name"][:110]
            print(f"   {float(row['Med (ns)'])/1e3:9.1f} us med  {float(row['Min (ns)'])/1e3:9.1f} min  x{row['Instances']:>4}  {name}")
