# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackSimulator


def bench_qrack(width, depth, p, sdrp):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 1 << min(8, (width + 2))

    lcv_range = range(width)
    all_bits = list(lcv_range)

    control = QrackSimulator(width)
    experiment = QrackSimulator(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            control.u(i, th, ph, lm)
            experiment.u(i, th, ph, lm)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            control.mcx([c], t)
            experiment.mcx([c], t)

        # if sdrp > 0:
        #     experiment.set_sdrp(sdrp)
        # experiment.run_qiskit_circuit(circ)

        # The point is to test whether XEB survives with a TurboQuant-based compression approach
        control_probs = control.out_probs()
        experiment.lossy_out_to_file("fc.svtq", p=p)
        experiment.lossy_in_from_file("fc.svtq")
        experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))

        print(calc_stats(control_probs, experiment_counts, d + 1, shots))


def calc_stats(ideal_probs, counts, depth, shots):
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
        count = counts[i] if i in counts else 0
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
        "depth": depth,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [compression block size power] [sdrp]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    p = 6
    sdrp = 0
    if len(sys.argv) > 3:
        p = int(sys.argv[3])
    if len(sys.argv) > 4:
        sdrp = float(sys.argv[4])

    # Run the benchmarks
    bench_qrack(width, depth, p, sdrp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
