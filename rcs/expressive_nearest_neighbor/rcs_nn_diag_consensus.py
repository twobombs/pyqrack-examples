# Patch ACE prob_perm consensus:
#   Two patch circuits (diagonal and anti-diagonal cuts) run the same nn circuit.
#   prob_perm queried over full 2^n Hilbert space on each patch pair.
#   Separable joint probability = product of subsystem prob_perms.
#   Average over H and V patches.
#
# XEB and HOG vs full ideal simulator (small scale only).
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while ((width // row_len) * row_len) != width:
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    return row_len, col_len


def make_patches_general(width, membership):
    patch     = np.asarray(membership, dtype=np.int32)
    local_idx = np.empty(width, dtype=np.int32)
    ctr = [0, 0]
    for i in range(width):
        local_idx[i] = ctr[patch[i]]
        ctr[patch[i]] += 1
    return patch, local_idx


def make_diagonal_patches(width, row_len, col_len):
    """Diagonal cut: patch 0 if row+col < row_len, patch 1 otherwise."""
    membership = []
    for q in range(width):
        row = q // col_len; col = q % col_len
        membership.append(0 if (row + col) < row_len else 1)
    return make_patches_general(width, membership)


def make_antidiag_patches(width, row_len, col_len):
    """Anti-diagonal cut: patch 0 if row+(col_len-1-col) < row_len."""
    membership = []
    for q in range(width):
        row = q // col_len; col = q % col_len
        membership.append(0 if (row + (col_len - 1 - col)) < row_len else 1)
    return make_patches_general(width, membership)


# ---------------------------------------------------------------------------
# Gate shadow machinery
# ---------------------------------------------------------------------------

def _x(sim,q,p,l):   sim[p[q]].x(l[q])
def _z(sim,q,p,l):   sim[p[q]].z(l[q])
def _h(sim,q,p,l):   sim[p[q]].h(l[q])
def _s(sim,q,p,l):   sim[p[q]].s(l[q])
def _adjs(sim,q,p,l):sim[p[q]].adjs(l[q])

def ct_pair_prob(sim,q1,q2,p,l):
    p1=sim[p[q1]].prob(l[q1]); p2=sim[p[q2]].prob(l[q2])
    return (p2,q1) if p1<p2 else (p1,q2)

def cz_shadow(sim,q1,q2,p,l,anti=False):
    if anti: _x(sim,q1,p,l)
    prob_max,t=ct_pair_prob(sim,q1,q2,p,l)
    if prob_max>0.5: _z(sim,t,p,l)
    if anti: _x(sim,q1,p,l)

def cx_shadow(sim,c,t,p,l,anti=False):
    _h(sim,t,p,l); cz_shadow(sim,c,t,p,l,anti); _h(sim,t,p,l)

def cy_shadow(sim,c,t,p,l,anti=False):
    _adjs(sim,t,p,l); cx_shadow(sim,c,t,p,l,anti); _s(sim,t,p,l)

def swap_shadow(sim,q1,q2,p,l):
    cx_shadow(sim,q1,q2,p,l); cx_shadow(sim,q2,q1,p,l); cx_shadow(sim,q1,q2,p,l)

def _same(q1,q2,p): return p[q1]==p[q2]
def _lq(q,l): return int(l[q])
def _pp(q,p): return int(p[q])

def cx(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcx([_lq(q1,l)],_lq(q2,l))
def cy(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcy([_lq(q1,l)],_lq(q2,l))
def cz(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].mcz([_lq(q1,l)],_lq(q2,l))
def acx(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cx_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macx([_lq(q1,l)],_lq(q2,l))
def acy(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cy_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macy([_lq(q1,l)],_lq(q2,l))
def acz(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l,True)
    else: sim[_pp(q1,p)].macz([_lq(q1,l)],_lq(q2,l))
def swap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].swap(_lq(q1,l),_lq(q2,l))
def iswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
        _s(sim,q1,p,l); _s(sim,q2,p,l)
    else: sim[_pp(q1,p)].iswap(_lq(q1,l),_lq(q2,l))
def iiswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        _s(sim,q1,p,l); _s(sim,q2,p,l)
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else: sim[_pp(q1,p)].adjiswap(_lq(q1,l),_lq(q2,l))
def pswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2)
def mswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p): swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)
def nswap(sim,q1,q2,p,l):
    if not _same(q1,q2,p):
        cz_shadow(sim,q1,q2,p,l); swap_shadow(sim,q1,q2,p,l); cz_shadow(sim,q1,q2,p,l)
    else:
        pp=_pp(q1,p); l1=_lq(q1,l); l2=_lq(q2,l)
        sim[pp].mcz([l1],l2); sim[pp].swap(l1,l2); sim[pp].mcz([l1],l2)

TWO_BIT_GATES = swap,pswap,mswap,nswap,iswap,iiswap,cx,cy,cz,acx,acy,acz


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
    exp_mixed = (1.0 - model) * exp_probs + model * u_u
    p_c   = ideal_probs - u_u
    q_c   = exp_probs   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, chi=None):
    row_len, col_len = factor_width(width)
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    rng_state     = random.getstate()
    gateSequence0 = [0,3,2,1,2,1,0,3]

    t_circ    = time.perf_counter()
    qc        = QuantumCircuit(width)

    random.setstate(rng_state)
    gateSequence = gateSequence0.copy()
    for _ in range(depth):
        for i in range(width):
            th, ph, lm = (random.uniform(0,2*math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.pi + 2 * th * abs(math.cos(2 * th))
            qc.u(th, ph, lm, i)
        gate = gateSequence.pop(0); gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                tr=row+(1 if(gate&2)else -1); tc=col+(1 if(gate&1)else 0)
                if tr<0: tr+=row_len
                if tc<0: tc+=col_len
                if tr>=row_len: tr-=row_len
                if tc>=col_len: tc-=col_len
                b1=row*row_len+col; b2=tr*row_len+tc
                if (b1==b2)or(b1>=width)or(b2>=width): continue
                g_name = random.choice(['cx','cy','cz','swap','iswap',
                                        'iiswap','pswap','mswap','nswap',
                                        'acx','acy','acz'])
                # Apply to Qiskit circuit
                if g_name == 'cx':      qc.cx(b1,b2)
                elif g_name == 'cy':    qc.cy(b1,b2)
                elif g_name == 'cz':    qc.cz(b1,b2)
                elif g_name == 'swap':  qc.swap(b1,b2)
                elif g_name == 'iswap': qc.iswap(b1,b2)
                elif g_name == 'iiswap':qc.iswap(b2,b1)
                elif g_name == 'pswap': qc.cz(b1,b2); qc.swap(b1,b2)
                elif g_name == 'mswap': qc.swap(b1,b2); qc.cz(b1,b2)
                elif g_name == 'nswap': qc.cz(b1,b2); qc.swap(b1,b2); qc.cz(b1,b2)
                elif g_name == 'acx':   qc.x(b1); qc.cx(b1,b2); qc.x(b1)
                elif g_name == 'acy':   qc.x(b1); qc.cy(b1,b2); qc.x(b1)
                elif g_name == 'acz':   qc.x(b1); qc.cz(b1,b2); qc.x(b1)

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    random.setstate(rng_state)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"ideal seconds: {t_ideal - t_circ}")

    patch_h, local_h = make_diagonal_patches(width, row_len, col_len)
    patch_v, local_v = make_antidiag_patches(width, row_len, col_len)

    max_patch = max(int(np.sum(patch_h==0)), int(np.sum(patch_h==1)),
                    int(np.sum(patch_v==0)), int(np.sum(patch_v==1)))

    print(f"Grid: {row_len} x {col_len}, largest patch: {max_patch} qubits")

    if np.array_equal(patch_h, patch_v):
        print("WARNING: diagonal and anti-diagonal splits are identical!")
        return

    # -----------------------------------------------------------------------
    # Patch ACE prob_perm over full Hilbert space.
    # Two patch circuits (H and V); separable joint probability via prob_perm
    # product on each subsystem. Average over H and V patches.
    # -----------------------------------------------------------------------
    def run_patch_sim(patch, local_idx):
        random.setstate(rng_state)
        n0 = int(np.sum(patch==0)); n1 = int(np.sum(patch==1))
        sim = [QrackSimulator(n0), QrackSimulator(n1)]
        gateSequence = gateSequence0.copy()
        for _ in range(depth):
            for i in range(width):
                th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
                # Keep it Haar-random towards the poles:
                th = math.pi + 2 * th * abs(math.cos(2 * th))
                sim[patch[i]].u(local_idx[i],th,ph,lm)
            gate=gateSequence.pop(0); gateSequence.append(gate)
            for row in range(1,row_len,2):
                for col in range(col_len):
                    tr=row+(1 if(gate&2)else -1); tc=col+(1 if(gate&1)else 0)
                    if tr<0: tr+=row_len
                    if tc<0: tc+=col_len
                    if tr>=row_len: tr-=row_len
                    if tc>=col_len: tc-=col_len
                    b1=row*row_len+col; b2=tr*row_len+tc
                    if (b1==b2)or(b1>=width)or(b2>=width): continue
                    g=random.choice(TWO_BIT_GATES); g(sim,b1,b2,patch,local_idx)
        return sim

    sim_d  = run_patch_sim(patch_h, local_h)   # diagonal
    sim_ad = run_patch_sim(patch_v, local_v)   # anti-diagonal

    def patch_prob(sim_pair, patch, local_idx, outcome):
        q0=[int(local_idx[q]) for q in range(width) if patch[q]==0]
        c0=[(outcome>>q)&1    for q in range(width) if patch[q]==0]
        q1=[int(local_idx[q]) for q in range(width) if patch[q]==1]
        c1=[(outcome>>q)&1    for q in range(width) if patch[q]==1]
        p0=sim_pair[0].prob_perm(q0,c0) if q0 else 1.0
        p1=sim_pair[1].prob_perm(q1,c1) if q1 else 1.0
        return p0 * p1

    ace_probs = np.empty(n_pow, dtype=np.float64)
    for outcome in range(n_pow):
        ace_probs[outcome] = 0.5 * patch_prob(sim_d,  patch_h, local_h, outcome) + \
                             0.5 * patch_prob(sim_ad, patch_v, local_v, outcome)
    for s in sim_d:  del s
    for s in sim_ad: del s

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_probs, n_pow)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":         width,
        "depth":         depth,
        "xeb_ace":       xeb_ace,
        "hog_ace":       hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_diag_consensus.py [width] [depth]\n"
            "Recommended widths: 20 (4x5), 30 (5x6), 42 (6x7)")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    result = bench_qrack(width, depth)
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
