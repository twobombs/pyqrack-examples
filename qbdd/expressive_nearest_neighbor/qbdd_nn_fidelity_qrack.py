# Validates (with Qiskit) the use of "Quantum Binary Decision Diagram" (QBDD) with lght-cone (nearest-neighbor)

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator


def rand_u3(sim, q):
    th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
    # Keep it Haar-random towards the poles:
    th = math.asin(th / math.pi)
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


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    circ = []

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        print("(Prime - skipped)")
        return

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            circ.append((rand_u3, i))

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if (
                    (temp_row < 0)
                    or (temp_col < 0)
                    or (temp_row >= row_len)
                    or (temp_col >= col_len)
                ):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                circ.append((g, b1, b2))

        experiment = QrackSimulator(width, is_binary_decision_tree=True)
        run_circuit(experiment, circ)
        experiment = experiment.out_ket()
        control = QrackSimulator(width, is_binary_decision_tree=False)
        run_circuit(control, circ)
        control = control.out_ket()

        overall_fidelity = np.abs(
            sum([np.conj(x) * y for x, y in zip(experiment, control)])
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
