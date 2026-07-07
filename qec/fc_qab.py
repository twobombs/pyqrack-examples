# Fully-connected RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import statistics
import sys
import time

import numpy as np
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.qrack.backends import AceQasmSimulator


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom
    return xeb, hog_prob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(8, width + 2)

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = QuantumCircuit(width, width)

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            th = math.asin(th / math.pi)   # Haar-uniform on Bloch sphere
            qc.u(th, ph, lm, i)
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        while len(shuffled) > 1:
            c, t = shuffled.pop(), shuffled.pop()
            qc.cx(c, t)

    qc.measure(all_bits, all_bits)

    # -----------------------------------------------------------------------
    # Method: Qiskit provider QrackAceBackend
    # Pass target= directly to avoid any legacy basis-gate / noise-model
    # collision that occurs when backend= is used with older provider installs.
    # -----------------------------------------------------------------------
    sim = AceQasmSimulator(n_qubits=width)

    # transpile against the backend's Target explicitly
    qc_t = transpile(qc, target=sim.target, optimization_level=1)

    job = sim.run(qc_t, shots=shots)
    raw_counts = job.result().get_counts()

    # get_counts() returns {'bitstring': count} keyed by binary strings.
    # Convert to integer keys for calc_stats.
    ace_counts = {int(k, 2): v for k, v in raw_counts.items()}

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_circ:.4f}")

    # -----------------------------------------------------------------------
    # Ideal ground truth via QrackSimulator
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc_t, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()
    print(f"ideal_seconds: {t_ideal - t_ace:.4f}")

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shots)

    return {
        "width":    width,
        "depth":    depth,
        "xeb_ace":  xeb_ace,
        "hog_ace":  hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 fc_qab.py [width] [depth]")
    width  = int(sys.argv[1])
    depth  = int(sys.argv[2])
    result = bench_qrack(width, depth)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
