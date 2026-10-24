#!/usr/bin/env python3
"""Register banks and other SASS-level experiments on the dp4a GEMM's sm_61 code (w8a8d_gemm, src/ggml-cuda/w8a8-dp4a.cuh).

Maxwell / Pascal read an instruction's register operands from 4 banks (bank = register number mod 4); two operands in
one bank are read one after the other unless one comes from the operand reuse cache (the .reuse flags of the control
words). This tool counts those reads in the kernels' loops and makes timing experiments by editing the machine code:

    python sass_banks.py analyze <libggml-cuda.so | ggml-cuda.dll | cubin>
        per kernel: the loops' same-bank operand reads by instruction class, stall counts, scoreboard waits
    python sass_banks.py whatif <lib | cubin> -o <out.cubin> [--nop OPS] [--region group|stage] [--clear-wait N]
                                [--fix-banks]
        the named instructions of a loop replaced by NOPs (OPS: opcodes, "@!P1 LDG.E.CI.128" with its predicate, or
        "=text prefix"), the waits on scoreboard N dropped, or (--fix-banks) every same-bank read pair of the inner
        loop's IDP / FFMA removed by reading another register in one unflagged operand. THE RESULTS ARE WRONG: timing
        only, run with test_dp4a (W8A8D_CUBIN=<out.cubin> test_dp4a bench ...)
    python sass_banks.py synth <bank_bench.cubin> -o <dir>
        the register-bank patterns of bank_bench.cu (512 dp4a with chosen registers and reuse flags, the same
        instructions and control words otherwise): run them with bank_bench <dir>/<name>.cubin

The library: the sm_61 cubin holding the w8a8d_gemm kernels is taken out with cuobjdump -xelf. Needs the CUDA
toolkit's nvdisasm and cuobjdump (PATH, or the environment variables NVDISASM / CUOBJDUMP). The register fields of every
instruction are learned from nvdisasm itself (bits flipped, the printed register checked); the control word layout
(Maxwell / Pascal: one 64-bit word per 3 instructions, 21 bits each: stall 4, yield 1, write barrier 3, read barrier
3, wait mask 6, reuse 4 (bit 0 = operand a, bits 8-15; bit 1 = b, bits 20-27; bit 2 = c, bits 39-46)) is the only
encoding knowledge built in. Findings: NOTES-w8a8-dp4a.md, "Register banks".
"""
import argparse, collections, glob, json, os, re, shutil, struct, subprocess, tempfile

NVDISASM = os.environ.get('NVDISASM') or shutil.which('nvdisasm') or 'nvdisasm'
CUOBJDUMP = os.environ.get('CUOBJDUMP') or shutil.which('cuobjdump') or 'cuobjdump'
KERNEL_RE = re.compile(r'^_Z10w8a8d_gemmILi(\d)EEv')


# ---------------------------------------------------------------- ELF
class Elf:
    def __init__(self, data):
        self.data = bytearray(data)
        d = self.data
        if d[:4] != b'\x7fELF' or d[4] != 2 or d[5] != 1:
            raise ValueError('not a little-endian ELF64 cubin')
        shoff, = struct.unpack_from('<Q', d, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from('<HHH', d, 0x3a)
        self.sections = []
        for i in range(shnum):
            name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from('<IIQQQQIIQQ', d, shoff + i * shentsize)
            self.sections.append(dict(name_off=name, type=typ, offset=off, size=size, link=link, info=info))
        stro = self.sections[shstrndx]['offset']
        for s in self.sections:
            s['name'] = d[stro + s['name_off']:d.index(0, stro + s['name_off'])].decode()

    def section(self, name):
        for s in self.sections:
            if s['name'] == name:
                return s
        raise KeyError(name)

    def kernels(self):
        names = [s['name'][6:] for s in self.sections if s['name'].startswith('.text.') and KERNEL_RE.match(s['name'][6:])]
        return sorted(names, key=lambda n: int(KERNEL_RE.match(n).group(1)))

    def text(self, fname):
        s = self.section('.text.' + fname)
        return [struct.unpack_from('<Q', self.data, s['offset'] + 8 * i)[0] for i in range(s['size'] // 8)]

    def set_text(self, fname, words):
        s = self.section('.text.' + fname)
        assert len(words) * 8 == s['size']
        for i, w in enumerate(words):
            struct.pack_into('<Q', self.data, s['offset'] + 8 * i, w)

    def regcount(self, fname):
        return self.section('.text.' + fname)['info'] >> 24


def run(args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f'{os.path.basename(args[0])} failed: {r.stderr.strip()[:2000]}')
    return r.stdout


def find_cubin(path, tmpdir):
    """the sm_61 cubin with the w8a8d_gemm kernels: the file itself, or taken out of a library"""
    data = open(path, 'rb').read()
    try:
        if Elf(data).kernels():
            return path
    except (ValueError, KeyError, struct.error):
        pass
    run([CUOBJDUMP, '-xelf', 'all', os.path.abspath(path)], cwd=tmpdir)
    hits = [f for f in sorted(glob.glob(os.path.join(tmpdir, '*sm_61*.cubin'))) if Elf(open(f, 'rb').read()).kernels()]
    if len(hits) != 1:
        raise RuntimeError(f'{path}: {len(hits)} sm_61 cubins with w8a8d_gemm kernels (expected 1)')
    return hits[0]


# ---------------------------------------------------------------- disassembly, control words, register fields
def disasm_json(path, raw=False):
    d = json.loads(run([NVDISASM, '-json', '-c', '-novliw'] + (['-b', 'SM61'] if raw else []) + [path]))
    return {f.get('function-name', '<raw>'): f['sass-instructions'] for f in d[1]}


REG = re.compile(r'(?<![A-Za-z0-9_])R(\d+|Z)(?![0-9])')


def reg_tokens(operands):
    return [255 if m.group(1) == 'Z' else int(m.group(1)) for m in REG.finditer(operands)]


def ctrl_field(cw, k):
    f = (cw >> (21 * k)) & 0x1FFFFF
    return dict(stall=f & 0xF, yield_=(f >> 4) & 1, wbar=(f >> 5) & 7, rbar=(f >> 8) & 7, wait=(f >> 11) & 0x3F,
                reuse=(f >> 17) & 0xF)


def set_reuse(cw, k, bits):
    sh = 21 * k + 17
    return (cw & ~(0xF << sh)) | ((bits & 0xF) << sh)


NOP_WORD = 0x50b0000000070f00
NEUTRAL_CTRL = 0x7e1 | (0x7e1 << 21) | (0x7e1 << 42)   # stall 1, no barriers, no reuse


def _raw_disasm1(words, tmpdir):
    buf = bytearray()
    for k in range(0, len(words), 3):
        grp = list(words[k:k + 3]) + [NOP_WORD] * (3 - len(words[k:k + 3]))
        buf += struct.pack('<Q', NEUTRAL_CTRL) + b''.join(struct.pack('<Q', w) for w in grp)
    p = os.path.join(tmpdir, 'raw.bin')
    open(p, 'wb').write(buf)
    ents = next(iter(disasm_json(p, raw=True).values()))
    return [(e.get('predicate', ''), e['opcode'], e.get('operands', '')) for i, e in enumerate(ents) if i % 4][:len(words)]


def raw_disasm(words, tmpdir, chunk=600):
    """disassemble instruction words; nvdisasm stops at an illegal encoding: it becomes a NOP (reported as illegal)
    and the chunk is disassembled again"""
    out = []
    for c0 in range(0, len(words), chunk):
        part = list(words[c0:c0 + chunk])
        bad = set()
        while True:
            try:
                res = _raw_disasm1([NOP_WORD if i in bad else w for i, w in enumerate(part)], tmpdir)
                break
            except RuntimeError as e:
                m = re.search(r'at address 0x([0-9a-fA-F]+)', str(e))
                i = (int(m.group(1), 16) // 32) * 3 + (int(m.group(1), 16) % 32) // 8 - 1 if m else -1
                if not m or i in bad or not 0 <= i < len(part):
                    raise
                bad.add(i)
        out += [('', '<illegal>', '') if i in bad else r for i, r in enumerate(res)]
    return out


CAND = (0, 8, 20, 28, 39)


def learn_fields(words, tmpdir):
    """{word: [field bit position of each register token]}: an 8-bit register field at position p is recognised by
    flipping its bit 4, then its bit 2: nvdisasm must print exactly that token changed by 16 (4) and nothing else"""
    uniq = sorted(set(words))
    base = raw_disasm(uniq, tmpdir)
    vd = raw_disasm([w ^ (flip << p) for w in uniq for p in CAND for flip in (0x10, 0x04)], tmpdir)
    res = {}
    k = 0
    for wi, w in enumerate(uniq):
        pr0, op0, ops0 = base[wi]
        toks0 = reg_tokens(ops0)
        fmap = {}
        for p in CAND:
            hits = []
            for flip in (0x10, 0x04):
                pr, op, ops = vd[k]
                k += 1
                toks = reg_tokens(ops)
                if op != op0 or pr != pr0 or REG.sub('R#', ops) != REG.sub('R#', ops0) or len(toks) != len(toks0):
                    hits.append(None)
                    continue
                diff = [t for t in range(len(toks)) if toks[t] != toks0[t]]
                hits.append(diff[0] if len(diff) == 1 and toks[diff[0]] == toks0[diff[0]] ^ flip else None)
            if hits[0] is not None and hits[0] == hits[1]:
                if hits[0] in fmap:
                    raise RuntimeError(f'two fields for one register: {op0} {ops0}')
                fmap[hits[0]] = p
        if len(fmap) != len(toks0):
            raise RuntimeError(f'register operands without a field: {op0} {ops0} {fmap}')
        res[w] = [fmap[t] for t in range(len(toks0))]
    return res


# ---------------------------------------------------------------- the kernel model
SLOT = {8: 0, 20: 1, 39: 2}      # operand slots a, b, c = reuse flag bits 0, 1, 2
MEM_OPS = ('LDS', 'LDG', 'LD', 'LDL', 'LDC', 'STS', 'STG', 'ST', 'STL', 'ATOM', 'ATOMS', 'RED')
STORES = ('STS', 'STG', 'ST', 'STL', 'RED')


class Ins:
    pass


def width_of(op, field):
    base, mods = op.split('.')[0], op.split('.')[1:]
    if base not in MEM_OPS:
        if base.startswith('D') or any(m in ('F64', '64') for m in mods):
            raise RuntimeError(f'64-bit ALU operands are not handled: {op}')
        return 1
    if field == 8:
        return 2 if 'E' in mods and base not in ('LDS', 'STS', 'LDC') else 1
    if field == 0:
        return 4 if '128' in mods else (2 if '64' in mods else 1)
    raise RuntimeError(f'unexpected register field {field} of {op}')


def load_kernel(elf, fname, js, fields):
    words = elf.text(fname)
    ents = js[fname]
    ins = []
    for i, w in enumerate(words):
        if i % 4 == 0:
            if ents[i]['opcode'] != '':
                raise RuntimeError(f'{fname}: word {i} is not a control word')
            continue
        e = ents[i]
        x = Ins()
        x.idx, x.off, x.word, x.ctrl, x.k = i, 8 * i, w, i - i % 4, i % 4 - 1
        x.op, x.pred, x.operands = e['opcode'], e.get('predicate', ''), e.get('operands', '')
        x.text = (x.pred + ' ' + x.op + ' ' + x.operands).strip()
        x.opnds = []     # (field, register, width, is_read)
        for f, r in zip(fields[w], reg_tokens(x.operands)):
            x.opnds.append((f, r, width_of(x.op, f), f != 0 or x.op.split('.')[0] in STORES))
        c = ctrl_field(words[x.ctrl], x.k)
        x.stall, x.reuse, x.wait = c['stall'], c['reuse'], c['wait']
        ins.append(x)
    return words, ins


def load_all(path, tmpdir):
    cub = find_cubin(path, tmpdir)
    elf = Elf(open(cub, 'rb').read())
    js = disasm_json(cub)
    fields = learn_fields([w for fn in elf.kernels() for i, w in enumerate(elf.text(fn)) if i % 4], tmpdir)
    return elf, js, fields


def loops(ins):
    """(first offset, branch offset) of the backward branches (not the branch to itself after EXIT), innermost first"""
    res = []
    for x in ins:
        if x.op == 'BRA' and re.fullmatch(r'0x[0-9a-f]+', x.operands) and int(x.operands, 16) < x.off:
            res.append((int(x.operands, 16), x.off))
    return sorted(res, key=lambda t: t[1] - t[0])


def bank_reads(ins):
    """per instruction: the registers it reads from the banks; an operand comes from the reuse cache when the previous
    instruction reading that slot read the same register with the reuse flag (no write in between). Conservative: the
    hardware also keeps a flagged value across instructions that read another register unflagged (bank_bench p2)."""
    cache = {}
    res = []
    for x in ins:
        regs = []
        for f, r, wd, rd in x.opnds:
            if rd and r != 255:
                s = SLOT.get(f) if not (f == 0 and x.op.split('.')[0] in STORES) else None
                if s is None or cache.get(s) != r:
                    regs += [r + k for k in range(wd)]
        res.append(sorted(set(regs)))
        for f, r, wd, rd in x.opnds:
            if rd and f in SLOT and r != 255:
                cache[SLOT[f]] = r if x.reuse >> SLOT[f] & 1 else None
        for f, r, wd, rd in x.opnds:
            if not rd:
                for s in list(cache):
                    if cache[s] is not None and r <= cache[s] < r + wd:
                        cache[s] = None
        if x.op in ('BRA', 'SYNC', 'SSY', 'EXIT') or x.op.startswith('BAR'):
            cache = {}
    return res


# ---------------------------------------------------------------- commands
def cmd_analyze(a):
    with tempfile.TemporaryDirectory() as td:
        elf, js, fields = load_all(a.input, td)
        for fn in elf.kernels():
            words, ins = load_kernel(elf, fn, js, fields)
            lp = loops(ins)
            print(f'{fn}: {elf.regcount(fn)} registers, {len(ins)} instructions, loops {[(hex(p), hex(q)) for p, q in lp]}')
            blk = set()          # registers of 64- / 128-bit operands: their bank is fixed by the alignment
            for x in ins:
                for f, r, wd, rd in x.opnds:
                    if wd > 1:
                        blk.update(range(r, r + wd))
            rds = bank_reads(ins)
            for li, (lo, hi) in enumerate(lp[:2]):
                body = [(x, rs) for x, rs in zip(ins, rds) if lo <= x.off <= hi and not (li and lp[0][0] <= x.off <= lp[0][1])]
                per, kinds, st, waits = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
                for x, rs in body:
                    k = x.op.split('.')[0]
                    per[k] += 1
                    byb = collections.defaultdict(list)
                    for r in rs:
                        byb[r % 4].append(r)
                    for regs in byb.values():
                        if len(regs) > 1:
                            kinds[k + ' ' + '+'.join(sorted('quad' if r in blk else 'single' for r in regs))] += len(regs) - 1
                    st[x.stall] += 1
                    if x.wait:
                        waits[k] += 1
                print(f'  {"inner loop" if li == 0 else "outer loop without the inner one"}: {len(body)} instructions ('
                      + ', '.join(f'{k} {v}' for k, v in per.most_common(6)) + ', ...)')
                print(f'    same-bank extra operand reads {sum(kinds.values())}: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common()))
                print(f'    stall counts {dict(sorted(st.items()))} (sum {sum(k * v for k, v in st.items())} cycles; 0 = dual issue), '
                      f'instructions waiting on a scoreboard {dict(waits)}')


def cmd_whatif(a):
    ops = [o.strip() for o in (a.nop or '').split(',') if o.strip()]
    with tempfile.TemporaryDirectory() as td:
        elf, js, fields = load_all(a.input, td)
        new = Elf(bytes(elf.data))
        for fn in elf.kernels():
            words, ins = load_kernel(elf, fn, js, fields)
            lp = loops(ins)
            lo, hi = lp[0] if a.region == 'group' else lp[-1]
            ws = list(words)
            nnop = nfix = nleft = 0
            for x in ins:
                if not lo <= x.off <= hi:
                    continue
                if a.clear_wait is not None:
                    ws[x.ctrl] &= ~((1 << (11 + a.clear_wait)) << (21 * x.k))
                if any(x.op == o or x.op.startswith(o + '.') or (o.startswith('@') and x.pred + ' ' + x.op == o) or
                       (o.startswith('=') and x.text.startswith(o[1:])) for o in ops):
                    ws[x.idx] = NOP_WORD
                    ws[x.ctrl] |= (0x3F << 5) << (21 * x.k)    # no write / read barrier: a NOP may not set one
                    nnop += 1
            if a.fix_banks:
                for x, rs in zip(ins, bank_reads(ins)):
                    if not lp[0][0] <= x.off <= lp[0][1] or x.op.split('.')[0] not in ('IDP', 'FFMA'):
                        continue
                    banks = collections.Counter(r % 4 for r in rs)
                    if all(v == 1 for v in banks.values()):
                        continue
                    done = False
                    for f, r, wd, rd in x.opnds:   # an unflagged operand in a doubly read bank: another lane / register
                        s = SLOT.get(f)
                        if done or s is None or not rd or r == 255 or x.reuse >> s & 1 or banks[r % 4] < 2 or r not in rs:
                            continue
                        used = {q % 4 for q in rs if q != r}
                        for d in (1, 2, 3, -1, -2, -3):
                            r2 = (r & ~3) | ((r + d) & 3)
                            if r2 % 4 not in used:
                                ws[x.idx] = (ws[x.idx] & ~(0xFF << f)) | (r2 << f)
                                done = True
                                break
                    nfix += done
                    nleft += not done
            new.set_text(fn, ws)
            print(f'{fn}: {nnop} instructions -> NOP' +
                  (f', {nfix} same-bank read pairs removed ({nleft} left: both operands flagged)' if a.fix_banks else ''))
        open(a.output, 'wb').write(new.data)
        run([NVDISASM, '-c', a.output])   # still a valid cubin
        print(f'wrote {a.output} (timing only: the results are wrong)')


def cmd_synth(a):
    """bank_bench patterns: the 512 IDPs of bank_bench's loop with chosen registers; garbage values, timing only"""
    with tempfile.TemporaryDirectory() as td:
        elf = Elf(open(a.input, 'rb').read())
        js = disasm_json(a.input)
        fields = learn_fields([w for i, w in enumerate(elf.text('outer8')) if i % 4], td)
        words, ins = load_kernel(elf, 'outer8', js, fields)
    loop = [x for x in ins if x.op.startswith('IDP')]
    assert len(loop) == 512, len(loop)
    ACC = sorted({x.opnds[0][1] for x in loop})
    A, B = list(range(2, 10)), list(range(10, 18))   # ptxas's a and b registers in the build this was written for
    assert len(ACC) == 64 and {x.opnds[1][1] for x in loop} == set(A) and {x.opnds[2][1] for x in loop} == set(B), \
        'bank_bench was compiled to other registers: adapt A / B'
    bank = lambda r: r % 4
    os.makedirs(a.output, exist_ok=True)

    def setf(w, pos, v):
        return (w & ~(0xFF << pos)) | ((v & 0xFF) << pos)

    def emit(name, seq):
        """seq: 512 x (kind, d, a, b, c, reuse bits)"""
        new = Elf(bytes(elf.data))
        ws = list(words)
        for x, (kind, rd, ra, rb, rc, ru) in zip(loop, seq):
            if kind == 'IDP':
                ws[x.idx] = setf(setf(setf(setf(x.word, 0, rd), 8, ra), 20, rb), 39, rc)
            else:   # MOV rd, rb (reads operand b only)
                ws[x.idx] = setf(setf(0x5c98078000070000, 0, rd), 20, rb)
            ws[x.ctrl] = set_reuse(ws[x.ctrl], x.k, ru)
        new.set_text('outer8', ws)
        open(os.path.join(a.output, name + '.cubin'), 'wb').write(new.data)
        print('  ' + name)

    ALL = {0, 1, 2, 3}

    def assign_acc(allowed):
        """(i, j) -> an accumulator register in an allowed bank (a matching of the 64 roles to the 64 registers)"""
        cap = collections.Counter(bank(r) for r in ACC)
        slots = [(b, k) for b in range(4) for k in range(cap[b])]
        match = {}

        def try_k(k, seen):
            for si, (b, _) in enumerate(slots):
                if b in allowed[k] and si not in seen:
                    seen.add(si)
                    if si not in match or try_k(match[si], seen):
                        match[si] = k
                        return True
            return False
        for k in list(allowed):
            if not try_k(k, set()):
                allowed[k] = ALL
                assert try_k(k, set())
        pools = {b: [r for r in ACC if bank(r) == b] for b in range(4)}
        return {k: pools[slots[si][0]].pop() for si, k in match.items()}

    def build(name, order_j, a_of, b_of, allowed, reuse_a):
        acc = assign_acc({(i, j): allowed(i, j) for i in range(8) for j in range(8)})
        seq = []
        for rep in range(8):
            for i in range(8):
                for n, j in enumerate(order_j(i)):
                    seq.append(('IDP', acc[(i, j)], a_of(i, n), b_of(i, j), acc[(i, j)], 1 if reuse_a and n < 7 else 0))
        emit(name, seq)

    emit('orig', [('IDP', x.opnds[0][1], x.opnds[1][1], x.opnds[2][1], x.opnds[3][1], x.reuse) for x in loop])
    rot = lambda i: [(j + (1 if bank(A[i]) == bank(B[0]) else 0)) % 8 for j in range(8)]
    first = lambda i, j: {bank(A[i])} if j == rot(i)[0] else set()
    build('free_reuse', rot, lambda i, n: A[i], lambda i, j: B[j], lambda i, j: ALL - {bank(B[j])} - first(i, j), True)
    build('free_reuse_noflags', rot, lambda i, n: A[i], lambda i, j: B[j], lambda i, j: ALL - {bank(B[j])} - first(i, j), False)
    a01, b23 = [4, 5, 8, 9, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11, 14, 15]
    build('free_noreuse', lambda i: list(range(8)), lambda i, n: a01[i], lambda i, j: b23[j],
          lambda i, j: ALL - {bank(a01[i]), bank(b23[j])}, False)
    bsame = lambda i, j: [r for r in B if bank(r) == bank(A[i])][j % 2]
    build('conf_ab', lambda i: list(range(8)), lambda i, n: A[i], bsame, lambda i, j: ALL - {bank(A[i])}, False)
    build('conf_ab_reuse', lambda i: list(range(8)), lambda i, n: A[i], bsame, lambda i, j: ALL - {bank(A[i])}, True)
    build('conf_bc_reuse', rot, lambda i, n: A[i], lambda i, j: B[j], lambda i, j: {bank(B[j])} - first(i, j) or {bank(B[j])}, True)
    build('conf_abc', lambda i: list(range(8)), lambda i, n: A[i], bsame, lambda i, j: {bank(A[i])}, False)

    # short periodic patterns (a, b, bank of c, reuse bits[, kind]): reuse cache semantics and the cost of a conflict
    pool = {b: [r for r in ACC if bank(r) == b] for b in range(4)}
    ptr = collections.Counter()

    def acc_in(b):
        r = pool[b][ptr[b] % len(pool[b])]
        ptr[b] += 1
        return r

    def pattern(name, pat):
        seq = []
        while len(seq) < 512:
            for it in pat:
                if len(seq) < 512:
                    c = acc_in(it[2])
                    seq.append((it[4] if len(it) > 4 else 'IDP', c, it[0], it[1], c, it[3]))
        emit(name, seq)
    X, Y, Z = 4, 5, 9            # banks 0, 1, 1
    B3, B0 = 11, 12              # banks 3, 0
    pattern('c_free', [(Y, B3, 2, 0), (Y, B0, 2, 0)])
    pattern('c_probe_miss', [(Y, B3, 2, 0), (X, B0, 3, 0)])
    pattern('c_probe_all', [(X, B0, 3, 0), (X, B0, 2, 0)])
    pattern('p1_next', [(X, B3, 2, 1), (X, B0, 3, 0)])
    pattern('p2_gap_noflag', [(X, B3, 2, 1), (Y, B3, 2, 0), (X, B0, 3, 0)])
    pattern('p2c_gap_noflag_miss', [(X, B3, 2, 0), (Y, B3, 2, 0), (X, B0, 3, 0)])
    pattern('p3_gap_flag', [(X, B3, 2, 1), (Y, B3, 2, 1), (X, B0, 3, 0)])
    pattern('p4_gap_mov', [(X, B3, 2, 1), (0, B3, 2, 0, 'MOV'), (X, B0, 3, 0)])
    pattern('p4c_gap_mov_miss', [(X, B3, 2, 0), (0, B3, 2, 0, 'MOV'), (X, B0, 3, 0)])
    pattern('p5_hit_then', [(X, B3, 2, 1), (X, B0, 3, 0), (X, B0, 3, 0)])
    pattern('p6_slot_b_next', [(Y, B0, 2, 2), (Z, B0, 0, 0)])
    pattern('p6c_slot_b_miss', [(Y, B0, 2, 0), (Z, B0, 0, 0)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    q = sp.add_parser('analyze')
    q.add_argument('input')
    q = sp.add_parser('whatif')
    q.add_argument('input')
    q.add_argument('-o', '--output', required=True)
    q.add_argument('--nop', help='comma-separated: OPCODE[.MOD], "@!P1 OPCODE.MODS" or "=text prefix"')
    q.add_argument('--region', choices=('group', 'stage'), default='group', help='the inner (group) or the outer (stage) loop')
    q.add_argument('--clear-wait', type=int, help='drop the waits on this scoreboard (0..5) in the region')
    q.add_argument('--fix-banks', action='store_true')
    q = sp.add_parser('synth')
    q.add_argument('input')
    q.add_argument('-o', '--output', required=True)
    a = p.parse_args()
    {'analyze': cmd_analyze, 'whatif': cmd_whatif, 'synth': cmd_synth}[a.cmd](a)


if __name__ == '__main__':
    main()
