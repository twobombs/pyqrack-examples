# Quantum volume protocol certification

import math
import random
import statistics
import sys
import time

from pyqrack import QrackSimulator

from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def rand_u3(circ, q):
    th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
    # Keep it Haar-random towards the poles:
    th = math.asin(th / math.pi)
    circ.u(th, ph, lm, q)


def coupler(circ, q1, q2):
    circ.cx(q1, q2)


def bench_qrack(n, backend, shots):
    # This is a "quantum volume" (random) circuit.
    circ = QuantumCircuit(n)

    lcv_range = range(n)
    all_bits = list(lcv_range)

    for d in range(n):
        # Single-qubit gates
        for i in lcv_range:
            rand_u3(circ, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            coupler(circ, c, t)

    start = time.perf_counter()
    sim = QrackSimulator(n)
    sim.run_qiskit_circuit(circ, shots=0)
    ideal_probs = sim.out_probs()
    del sim
    sim_interval = time.perf_counter() - start

    circ.measure_all()

    device = (
        Aer.get_backend(backend)
        if len(Aer.backends(backend)) > 0
        else QiskitRuntimeService().backend(backend)
    )
    circ = transpile(circ, device, layout_method="noise_adaptive")

    result = device.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    interval = result.time_taken

    return (ideal_probs, counts, interval, sim_interval)


def calc_stats(ideal_probs, counts, interval, sim_interval, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        b = (bin(i)[2:]).zfill(n)

        count = counts[b] if b in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    return {
        "qubits": n,
        "seconds": interval,
        "xeb": xeb,
        "hog_prob": hog_prob,
        "pass": hog_prob >= 2 / 3,
        "clops": (n * shots) / interval,
        "sim_clops": (n * shots) / sim_interval,
        "eplg": (1 - (xeb ** (1 / depth))) if xeb < 1 else 0,
    }


def main():
    n = 20
    shots = 1 << n
    backend = "qasm_simulator"
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        shots = int(sys.argv[2])
    else:
        shots = 1 << n
    if len(sys.argv) > 3:
        backend = sys.argv[3]
    if len(sys.argv) > 4:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum", token=sys.argv[4], set_as_default=True
        )

    results = bench_qrack(n, backend, shots)

    ideal_probs = results[0]
    counts = results[1]
    interval = results[2]
    sim_interval = results[3]

    print(calc_stats(ideal_probs, counts, interval, sim_interval, shots))

    return 0


if __name__ == "__main__":
    sys.exit(main())
