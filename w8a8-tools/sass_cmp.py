"""sass_cmp.py <old .obj/.dll> <new .obj/.dll> <name substring> [...]: the SASS of every function whose (mangled) name
contains one of the substrings, compared between two binaries (instruction addresses stripped): per function equal /
different / only in one.
SASS_MAP="old=new,..." renames the OLD binary's functions first (substring replacement), for kernels whose template
parameters changed type (e.g. w8a8_convert_weights<bool q5, ...> -> <int wt, ...>: ILb1ELb0E=ILi1ELb0E)."""
import hashlib
import os
import re
import subprocess
import sys

CUO = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cuobjdump.exe"


def funcs(path: str, pats: list[str]) -> dict[str, str]:
    if path.endswith(".txt"):  # a saved `cuobjdump -sass` listing (a large DLL takes minutes to disassemble)
        txt = open(path).read()
    else:
        txt = subprocess.run([CUO, "-sass", path], capture_output=True, text=True).stdout
    out = {}
    for f in re.split(r"\n\s*Function : ", txt)[1:]:
        name = f.split("\n", 1)[0].strip()
        if any(p in name for p in pats):
            # the cubin's last function also carries the listing's next headers ("Fatbin ptx code: ...")
            body = re.sub(r"/\*[0-9a-f]{4,}\*/", "", f.split("\n", 1)[1].split("\nFatbin ")[0].rstrip())
            body = re.sub(r"\.section.*|\.headerflags.*", "", body)
            key = name.replace("13w8a8_act_ptrsi", "")  # the norm kernel's added parameters
            out[key] = out.get(key, "") + hashlib.md5(body.encode()).hexdigest()
    return out


a, b, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
fa, fb = funcs(a, pats), funcs(b, pats)
for pair in filter(None, os.environ.get("SASS_MAP", "").split(",")):
    old, new = pair.split("=", 1)
    fa = {k.replace(old, new): v for k, v in fa.items()}
same = [k for k in fa if k in fb and fa[k] == fb[k]]
diff = [k for k in fa if k in fb and fa[k] != fb[k]]
print(f"{len(fa)} / {len(fb)} functions; equal {len(same)}, different {len(diff)}, only old {len(set(fa) - set(fb))}, "
      f"only new {len(set(fb) - set(fa))}")
for k in diff:
    print("  DIFFERENT", k[:150])
for k in sorted(set(fb) - set(fa)):
    print("  only new ", k[:150])
for k in sorted(set(fa) - set(fb)):
    print("  only old ", k[:150])
