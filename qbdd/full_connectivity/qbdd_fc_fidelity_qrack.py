# Validates (with Qiskit) the use of "Quantum Binary Decision Diagram" (QBDD) with lght-cone (nearest-neighbor)

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        print("(Prime - skipped)")
        return

    control = QrackSimulator(width, is_binary_decision_tree=False)
    experiment = QrackSimulator(width, is_binary_decision_tree=True)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
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

        experiment_ket = experiment.out_ket()
        control_ket = control.out_ket()

        overall_fidelity = np.abs(
            sum([np.conj(x) * y for x, y in zip(experiment_ket, control_ket)])
        )

        print(
            "Depth="
            + str(d + 1)
            + ", overall fidelity="
            + str(overall_fidelity)
        )


def main():
    # Run the benchmarks
    for i in range(1, 21):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == "__main__":
    sys.exit(main())
