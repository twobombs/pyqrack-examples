# Nearest-neighbor RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


# ---------------------------------------------------------------------------
# Gate wrappers
# ---------------------------------------------------------------------------

def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs, n_pow):
    u_u   = 1.0 / n_pow
    p_c   = ideal_probs - u_u
    q_c   = exp_probs   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_probs[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 4
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc      = [[] for _ in range(n_inst)]

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(width)

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.pi + 2 * th * abs(math.cos(th))
            for c in qc:
                c.append((u, i, th, ph, lm))

        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        cl = []
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                cl.append((g, b1, b2))

        for c in qc:
            random.shuffle(cl)
            for g in cl:
                c.append(g)

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc[0])
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"qrack_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Method: ACE prob_perm over full Hilbert space.
    # Since the ideal simulation is already materialized for ground truth,
    # we can afford to walk all 2^n permutations with prob_perm — giving
    # the complete ACE probability distribution, not just sampled candidates.
    # Two ACE instances (sequential + stride); average their prob_perm values.
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        run_circuit(sim, qc[inst])
        ace_sims.append(sim)

    ace_probs = np.zeros(n_pow, dtype=np.float64)
    for inst in ace_sims:
        ace_probs += np.array(inst.out_probs(), dtype=np.float64)
    ace_probs /= n_inst

    # q_bits = list(range(width))
    # for outcome in range(n_pow):
        # It's possible to access every amplitude without materializing the state vector:
        # bits  = [(outcome >> b) & 1 for b in range(width)]
        # ace_probs[outcome] = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_probs, n_pow)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":         width,
        "depth":         depth,
        "sdrp":          sdrp,
        "xeb_ace":       xeb_ace,
        "hog_ace":       hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace.py [width] [depth] [sdrp=0.1464466]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else ((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
