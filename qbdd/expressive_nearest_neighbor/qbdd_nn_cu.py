# Fully-connected RCS: Automatic circuit elision
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
# Global Gate API
# ---------------------------------------------------------------------------

def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cu(sim, c, t, th, ph, lm, gm):
    sim.mcu([c], t, th, ph, lm, gm)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 1
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)

    # -----------------------------------------------------------------------
    # Build n_inst independent random circuits (same single-qubit angles,
    # different coupler orderings — identical to fc_ace_consensus.py)
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = [[] for _ in range(n_inst)]

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
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

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                if random.randint(0, 1):
                    b1, b2 = b2, b1

                cl.append(((b1, b2), [random.uniform(0, 2*math.pi) for _ in range(4)]))

        for c in qc:
            random.shuffle(cl)
            for g in cl:
                b, p = g
                c.append((cu, b[0], b[1], p[0], p[1], p[2], p[3]))

    t_build = time.perf_counter()
    print(f"circuit_build_seconds: {t_build - t_circ}")

    # -----------------------------------------------------------------------
    # ACE consensus instances
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        sim = QrackSimulator(width, is_binary_decision_tree=True)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        run_circuit(sim, qc[inst])
        ace_sims.append(sim)

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_build}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace_consensus.py [width] [depth] [trials=1] [sdrp=0.0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    bench_qrack(width, depth, sdrp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
