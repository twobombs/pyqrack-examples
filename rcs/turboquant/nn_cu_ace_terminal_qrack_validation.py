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


def cu(sim, b1, b2, th, ph, lm, gm):
    sim.mcu([b1], b2, th, ph, lm, gm)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0, p=6):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc     = []

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.append((u, i, th, ph, lm))

        gate = gateSequence.pop(0)
        gateSequence.append(gate)
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

                th, ph, lm, gm = (random.uniform(-math.pi, math.pi) for _ in range(4))
                # Keep it Haar-random towards the poles:
                th = math.asin(th / math.pi)
                qc.append((cu, b1, b2, th, ph, lm, gm))

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    run_circuit(sim_ideal, qc)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"qrack_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Method: ACE prob_perm over full Hilbert space (compressed)
    # -----------------------------------------------------------------------
    sim = QrackSimulator(width)
    if sdrp > 0.0:
        sim.set_sdrp(sdrp)
    sim.set_ace_max_qb((width + 1) >> 1)
    run_circuit(sim, qc)

    sim.lossy_out_to_file("nn.svtq", p=p)
    sim.lossy_in_from_file("nn.svtq")

    out_probs = np.array(sim.out_probs())

    del sim

    xeb_ace, hog_ace = calc_stats(ideal_probs, out_probs, n_pow)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":         width,
        "depth":         depth,
        "sdrp":          sdrp,
        "xeb_ace_tq":    xeb_ace,
        "hog_ace_tq":    hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace.py [width] [depth] [sdrp=0.1464466] [p=6]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else ((1 - 1 / math.sqrt(2)) / 2)
    p = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    result = bench_qrack(width, depth, sdrp, p)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
