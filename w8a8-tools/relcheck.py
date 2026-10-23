"""relcheck.py [arch ...]: compiles the files this branch changed or added (w8a8-mm.cu, ggml-cuda.cu, w8a8-emu.cu) with the
ggml-cuda flags of the build (build-sm86/build.ninja) for each architecture of the release build (61 70 75 80 86 89 90 100
120: SASS + PTX each, as stable-diffusion.cpp's build-release.cmd), then lists the W8A8 kernels in each w8a8-mm object
(cuobjdump: instruction counts; sm_61..75 must be empty stubs, sm_80+ the real kernels)."""
import concurrent.futures as cf
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NINJA = ROOT / "build-sm86" / "build.ninja"
OUT = Path(__file__).resolve().parent / "out" / "relcheck"
OUT.mkdir(parents=True, exist_ok=True)
NVCC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
CUOBJDUMP = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cuobjdump.exe"
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
FILES = ["w8a8-mm.cu", "ggml-cuda.cu", "w8a8-emu.cu"]
ARCHS = [a for a in sys.argv[1:]] or ["61", "70", "75", "80", "86", "89", "90", "100", "120"]

text = NINJA.read_text()
blk = text[text.index("w8a8-mm.cu.obj: CUDA_COMPILER"):]
defines = re.search(r"\n  DEFINES = (.*)", blk).group(1)
includes = re.search(r"\n  INCLUDES = (.*)", blk).group(1)
flags = re.search(r"\n  FLAGS = (.*)", blk).group(1)
flags = re.sub(r'"--generate-code=arch=compute_86,code=\[compute_86,sm_86\]"', "{GEN}", flags)
assert "{GEN}" in flags


def job(arch: str, f: str):
    obj = OUT / f"{Path(f).stem}.sm{arch}.obj"
    gen = f'"--generate-code=arch=compute_{arch},code=[compute_{arch},sm_{arch}]"'
    script = OUT / f"job_{arch}_{Path(f).stem}.cmd"
    lines = ["@echo off", f'call "{VCVARS}" >nul',
             f'"{NVCC}" -forward-unknown-to-host-compiler {defines} {includes} {flags.replace("{GEN}", gen)} -x cu -c '
             f'"{ROOT / "src" / "ggml-cuda" / f}" -o "{obj}"']
    script.write_text(chr(10).join(lines) + chr(10))  # text mode: CRLF on Windows
    p = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True)
    errs = [l for l in (p.stdout + p.stderr).splitlines() if " error" in l or "error:" in l]
    return arch, f, p.returncode, errs[:5], obj


with cf.ThreadPoolExecutor(max_workers=12) as ex:
    res = list(ex.map(lambda af: job(*af), [(a, f) for a in ARCHS for f in FILES]))
for arch, f, rc, errs, obj in res:
    print(f"sm_{arch:>3} {f:14} {'OK' if rc == 0 else 'FAILED rc ' + str(rc)} {' | '.join(errs)}")
for arch in ARCHS:
    obj = OUT / f"w8a8-mm.sm{arch}.obj"
    if not obj.exists():
        continue
    sass = subprocess.run([CUOBJDUMP, "-sass", str(obj)], capture_output=True, text=True).stdout
    funcs = re.split(r"\n\s*Function : ", sass)[1:]
    summary = []
    for fn in funcs:
        name = fn.split("\n", 1)[0].strip()
        if "w8a8" not in name:
            continue
        n_ins = len(re.findall(r"/\*[0-9a-f]{4,}\*/\s+[A-Z@]", fn))
        n_mma = fn.count("IMMA")
        short = re.sub(r"^_Z\d+", "", name)[:40]
        summary.append(f"{short}: {n_ins} instr, {n_mma} IMMA")
    print(f"sm_{arch}: " + ("; ".join(summary) if summary else "no w8a8 kernels in the SASS"))
