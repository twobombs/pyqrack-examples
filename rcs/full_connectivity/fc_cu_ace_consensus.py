# Fully-connected RCS: Automatic circuit elision
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

def bench_qrack(width, depth, sdrp=0.0, trials=1):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 4
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    results = {
        "width":         width,
        "depth":         depth,
        "sdrp":          sdrp,
        "trials":        trials,
        "xeb_ace":       0.0,
        "hog_ace":       0.0,
    }

    for _ in range(trials):
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
            shuffled = all_bits[:]
            random.shuffle(shuffled)
            cl = []
            while len(shuffled) > 1:
                th, ph, lm, gm = (random.uniform(-math.pi, math.pi) for _ in range(4))
                # Keep it Haar-random towards the poles:
                th = math.asin(th / math.pi)
                cl.append(((shuffled.pop(), shuffled.pop()), [th, ph, lm, gm]))
            for c in qc:
                random.shuffle(cl)
                for g in cl:
                    b, p = g
                    c.append((cu, b[0], b[1], p[0], p[1], p[2], p[3]))

        t_build = time.perf_counter()
        if trials == 1:
            print(f"circuit_build_seconds: {t_build - t_circ}")

        # -----------------------------------------------------------------------
        # Ideal ground truth (full state vector via Qrack)
        # -----------------------------------------------------------------------
        sim_ideal = QrackSimulator(width)
        run_circuit(sim_ideal, qc[0])
        ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
        del sim_ideal


        t_ideal = time.perf_counter()
        if trials == 1:
            print(f"ideal_seconds: {t_ideal - t_build}")

        # -----------------------------------------------------------------------
        # ACE consensus instances
        # -----------------------------------------------------------------------
        ace_sims = []
        for inst in range(n_inst):
            sim = QrackSimulator(width)
            if sdrp > 0.0:
                sim.set_sdrp(sdrp)
            sim.set_ace_max_qb((width + 1) >> 1)
            run_circuit(sim, qc[inst])
            ace_sims.append(sim)

        t_ace = time.perf_counter()
        if trials == 1:
            print(f"ace_seconds: {t_ace - t_ideal}")

        # -----------------------------------------------------------------------
        # Route into heavy / light and compute statistics
        # -----------------------------------------------------------------------
        ace_probs = np.empty(n_pow, dtype=np.float64)
        for inst in ace_sims:
            ace_probs += np.array(inst.out_probs())
        ace_probs /= n_inst

        for s in ace_sims:
            del s

        xeb, hog = calc_stats(ideal_probs, ace_probs, n_pow)

        t_elapsed = time.perf_counter() - t_ideal

        if math.isfinite(xeb) and math.isfinite(hog):
            results["xeb_ace"] += xeb
            results["hog_ace"] += hog
        else:
            trials -= 1

    if trials == 0:
        raise ZeroDivisionError("ERROR: No trials produced finite XEB and HOG!")

    results["xeb_ace"] /= trials
    results["hog_ace"] /= trials

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_cu_ace_consensus.py [width] [depth] [trials=100] [sdrp=0.146466]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    sdrp  = float(sys.argv[4]) if len(sys.argv) > 4 else ((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, sdrp, trials)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
