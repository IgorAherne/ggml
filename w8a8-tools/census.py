"""census.py <obj> <function substring> [out.sass]: opcode counts of a kernel's SASS (whole function and the biggest loop body)."""
import re, subprocess, sys, collections
CUO = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cuobjdump.exe"
obj, pat = sys.argv[1], sys.argv[2]
txt = subprocess.run([CUO, "-sass", obj], capture_output=True, text=True).stdout
funcs = re.split(r"\n\s*Function : ", txt)
sel = [f for f in funcs if f.split("\n", 1)[0].strip().find(pat) >= 0]
if not sel:
    print("no function matches"); sys.exit(1)
f = sel[0]
print(f.split("\n", 1)[0].strip()[:150], f"({len(sel)} matches)")
if len(sys.argv) > 3: open(sys.argv[3], "w").write(f)
ins = []
for line in f.splitlines():
    m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if m: ins.append((int(m.group(1), 16), m.group(3), line))
# loops: backward branches
loops = []
for addr, op, line in ins:
    if op.startswith("BRA"):
        t = re.search(r"0x([0-9a-f]+)", line.split("BRA", 1)[1])
        if t and int(t.group(1), 16) < addr: loops.append((int(t.group(1), 16), addr))
def count(lo, hi):
    c = collections.Counter()
    for addr, op, _ in ins:
        if lo <= addr <= hi: c[op.split(".")[0]] += 1
    return c
tot = count(0, 1 << 40)
print("whole function:", sum(tot.values()), "instr")
for lo, hi in sorted(set(loops)):
    c = count(lo, hi)
    print(f"  loop {lo:#x}-{hi:#x}: {sum(c.values())} instr, IMMA {c.get('IMMA', 0)}")
cand = [l for l in set(loops) if count(*l).get("IMMA", 0) > 0]
if cand:
    lo, hi = min(cand, key=lambda l: l[1] - l[0])
    c = count(lo, hi)
    n = sum(c.values()); mma = c.get("IMMA", 0)
    print(f"biggest loop {lo:#x}-{hi:#x}: {n} instr, IMMA {mma}, per IMMA {n/max(mma,1):.2f}")
    print("  " + "  ".join(f"{k} {v}" for k, v in c.most_common()))
