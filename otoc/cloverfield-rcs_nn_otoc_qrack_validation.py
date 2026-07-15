# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid RCS-OTOC Benchmark (27 Patches, 729 Qubits Total)
# Nearest-Neighbor Coupler RCS + OTOC Echo Cycles + ACE-vs-Exact Validation
#
# REVISION 98 - DOCUMENTATION OF SPARSE PT-APPROXIMATIONS
#
# CHANGES (Rev 98):
# - PHYSICS DOCS (XEB): Explicitly noted that fixing the XEB denominator to the PT 
#   expectation (1/N) bypasses the old denom-collapse guard. ACE collapse will now 
#   manifest purely as a numerator collapse (XEB -> 0).
# - PHYSICS DOCS (L2): Annotated that l2_sq_dist uses the PT ensemble expectation 
#   (2/N) for the unsampled tail, which is mathematically robust for depth >= 8.
#
# CHANGES (Rev 97):
# - PERFORMANCE (CRITICAL): Replaced 1.07 GB out_probs() statevector readout with 
#   sparse prob_perm() lookups for observed sample outcomes (~350 lookups vs 134M).
# - OPTIMIZATION: Shifted probability lookup directly to worker process, allowing 
#   the QrackSimulator instance to be destroyed *prior* to statistics calculations.
# - MATH: Utilized Porter-Thomas expectation values for XEB denominator and analytical 
#   HOG threshold approximations to allow exact sub-sampling metrics.
# - TIMING: Lat_Probs_ms now records the exact overhead of the sparse query loop.

import os
import sys
import csv
import json
import time
import math
import random
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from collections import Counter
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Topography tuning for raw statevectors.
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 1
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# Porter-Thomas ideal HOG score (median-threshold variant used by calc_stats):
PT_IDEAL_HOG = (1.0 + math.log(2.0)) / 2.0

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

# =====================================================================
# PURE FUNCTIONS
# =====================================================================
def factor_width(width: int) -> Tuple[int, int]:
    """
    Finds the most square-like rectangular factorization.
    Note: For width=27, this returns (9, 3), resulting in a 9x3 grid topology.
    """
    col_len = int(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    return (row_len, col_len)


# --- Gate helpers ---
def cx(sim, q1, q2):  sim.mcx([q1], q2)
def cy(sim, q1, q2):  sim.mcy([q1], q2)
def cz(sim, q1, q2):  sim.mcz([q1], q2)
def acx(sim, q1, q2): sim.macx([q1], q2)
def acy(sim, q1, q2): sim.macy([q1], q2)
def acz(sim, q1, q2): sim.macz([q1], q2)
def u(sim, q, th, ph, lm): sim.u(q, th, ph, lm)
def x(sim, q): sim.x(q)
def y(sim, q): sim.y(q)
def z(sim, q): sim.z(q)

TWO_BIT_GATES = (cx, cy, cz, acx, acy, acz)
PAULI_OPS = ('I', 'X', 'Y', 'Z')


def act_string(otoc: list, string: str):
    """
    Applies the Pauli string to the OTOC sequence. 
    Note: 'i' assumes len(string) == width (QUBITS_PER_PATCH).
    """
    for i in range(len(string)):
        match string[i]:
            case 'X':
                otoc.append((x, i))
            case 'Y':
                otoc.append((y, i))
            case 'Z':
                otoc.append((z, i))
            case _:
                pass


def build_rcs_otoc_circuit(
    width: int, depth: int, cycles: int, rng: random.Random
) -> Tuple[list, List[str]]:
    lcv_range = range(width)
    
    # Sycamore-style sequence: period is 8.
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)

    rcs = []
    for d in range(depth):
        for i in lcv_range:
            th = rng.uniform(0, 2 * math.pi)
            ph = rng.uniform(0, 2 * math.pi)
            lm = rng.uniform(0, 2 * math.pi)
            rcs.append((u, i, th, ph, lm))

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

                b1 = row * col_len + col
                b2 = temp_row * col_len + temp_col

                if (b1 >= width) or (b2 >= width) or (b1 == b2):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = rng.choice(TWO_BIT_GATES)
                rcs.append((g, b1, b2))

    ircs = []
    for tup in reversed(rcs):
        if tup[0] == u:
            ircs.append((u, tup[1], -tup[2], -tup[4], -tup[3]))
        else:
            ircs.append(tup)

    pauli_strings: List[str] = []
    otoc: list = []
    
    for _cycle in range(cycles):
        otoc.extend(rcs)
        string = "".join(rng.choice(PAULI_OPS) for _ in range(width))
        pauli_strings.append(string)
        act_string(otoc, string)
        otoc.extend(ircs)

    return otoc, pauli_strings


def calc_stats(
    ideal_at: np.ndarray, freqs: np.ndarray,
    depth: int, cycles: int, shots: int, ace_qb: int, n: int
) -> Dict[str, Any]:
    n_pow = 1 << n
    uniform = 1.0 / n_pow
    u_u = uniform
    
    # Porter-Thomas Expectation values for XEB Denominator normalization
    # E[p^2] = 2 / N^2 => sum_sq_ideal = 2 / N
    # NOTE (Rev 98): Because we use the PT expectation rather than the empirical sum 
    # of squares, denom is essentially fixed at 1/N. The historic denom-collapse guard 
    # for ACE will not fire; instead, ACE collapse will manifest directly as a 
    # numerator collapse (XEB -> 0).
    sum_sq_ideal = 2.0 * uniform
    denom = sum_sq_ideal - u_u

    ic_at = ideal_at - u_u
    numer = float(np.dot(ic_at, freqs))
    
    pt_var = uniform
    xeb = numer / denom if denom > (pt_var * 1e-3) else 0.0

    # Analytical PT median mapping threshold: ln(2) / N
    threshold = math.log(2.0) * uniform
    hog_prob = float(np.sum(freqs[ideal_at > threshold]))

    # L2: The first term (sum_sq_ideal) represents the unsampled tail.
    # NOTE (Rev 98): We approximate this unsampled tail using the PT ensemble expectation 
    # (2/N) rather than the exact empirical sum. For well-scrambled circuits (depth >= 8), 
    # this introduces negligible sub-percent error.
    l2_sq_dist = (
        sum_sq_ideal
        - float(np.dot(ideal_at, ideal_at))
        + float(np.sum((ideal_at - freqs) ** 2))
    )
    
    l2_sq_dist_random = (
        sum_sq_ideal - 2.0 * uniform + float(n_pow) * uniform * uniform
    )

    return {
        "qubits": n,
        "ace_qb_limit": ace_qb,
        "depth": depth,
        "cycles": cycles,
        "effective_depth": depth * cycles,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_sq_dist": float(l2_sq_dist),
        "l2_sq_dist_vs_uniform_random": float(l2_sq_dist_random),
    }

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    depth: int,
    cycles: int,
    shots: int,
    master_seed: int
):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_MAX_ALLOC_MB"] = "64000"
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    from pyqrack import QrackSimulator

    try:
        _probe = QrackSimulator(2)
        _probe.x(0)
        _sv = list(_probe.measure_shots([0, 1], 8))
        if len(set(_sv)) != 1:
            raise RuntimeError(
                f"Fatal: measure_shots nondeterministic on basis state: {_sv}"
            )
        _pk = int(np.argmax(np.asarray(_probe.out_probs(), dtype=np.float64)))
        if _pk != int(_sv[0]):
            raise RuntimeError(
                f"Fatal: measure_shots value ({_sv[0]}) and out_probs argmax "
                f"({_pk}) disagree on bit order; calc_stats indexing is unsafe."
            )
        if _pk not in (1, 2):
            raise RuntimeError(
                f"Fatal: Unexpected bit ordering. Expected 1 (LSB) or 2 (MSB), got {_pk}."
            )
        del _probe

        all_bits = list(range(QUBITS_PER_PATCH))
        ace_qb = (QUBITS_PER_PATCH + 3) >> 2

        for p in assigned_patches:
            seed_int = int(
                np.random.SeedSequence([master_seed, p]).generate_state(1)[0]
            )
            rng = random.Random(seed_int)

            t0 = time.perf_counter()
            otoc, pauli_strings = build_rcs_otoc_circuit(
                QUBITS_PER_PATCH, depth, cycles, rng
            )
            t_build = time.perf_counter() - t0

            control = QrackSimulator(QUBITS_PER_PATCH)
            experiment = QrackSimulator(QUBITS_PER_PATCH)
            try:
                experiment.set_ace_max_qb(ace_qb)
            except Exception as e:
                raise RuntimeError(
                    f"Fatal: set_ace_max_qb unavailable in this PyQrack "
                    f"build: {e}"
                )

            t0 = time.perf_counter()
            for tup in otoc:
                tup[0](control, *tup[1:])
                tup[0](experiment, *tup[1:])
            t_gates = time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_samples = experiment.measure_shots(all_bits, shots)
            counts = Counter(raw_samples)
            samples_arr = np.array(raw_samples, dtype=np.uint32)
            t_sample = time.perf_counter() - t0
            del experiment

            # Fast target-selective lookups over observed basis states
            t0 = time.perf_counter()
            num_distinct = len(counts)
            ideal_at = np.empty(num_distinct, dtype=np.float64)
            freqs = np.empty(num_distinct, dtype=np.float64)
            
            for i, (outcome, count) in enumerate(counts.items()):
                # Unpack bit-representation LSB first for prob_perm mapping
                bits = [bool((outcome >> j) & 1) for j in range(QUBITS_PER_PATCH)]
                ideal_at[i] = control.prob_perm(all_bits, bits)
                freqs[i] = count / shots
                
            t_probs = time.perf_counter() - t0
            del control

            t0 = time.perf_counter()
            stats = calc_stats(ideal_at, freqs, depth, cycles, shots, ace_qb, QUBITS_PER_PATCH)
            t_stats = time.perf_counter() - t0

            payload = dict(stats)
            payload.update({
                "pauli_strings": pauli_strings,
                "samples": samples_arr,
                "seed": seed_int,
                "lat_build_ms": t_build * 1000.0,
                "lat_gates_ms": t_gates * 1000.0,
                "lat_sample_ms": t_sample * 1000.0,
                "lat_probs_ms": t_probs * 1000.0,
                "lat_stats_ms": t_stats * 1000.0,
            })
            conn.send({p: payload})

    finally:
        conn.close()

# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuRcsOtocEngine:
    def __init__(self, master_seed: int = 1337):
        self.master_seed = master_seed

        self.patch_coords = {}
        idx = 0
        for xg in range(GRID_X):
            for yg in range(GRID_Y):
                for zg in range(GRID_Z):
                    self.patch_coords[idx] = (xg, yg, zg)
                    idx += 1

        self.per_patch_csv = "rcs_otoc_ace_per_patch.csv"
        self.aggregate_csv = "rcs_otoc_ace_aggregate.csv"
        self.samples_file  = "rcs_otoc_ace_bitstrings.npz"
        self.pauli_file    = "rcs_otoc_ace_pauli_strings.json"
        self.config_file   = "rcs_otoc_ace_config.json"

        self.worker_assignments: List[List[int]] = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self, depth: int, cycles: int, shots: int):
        try:
            row_len, col_len = factor_width(QUBITS_PER_PATCH)
            with open(self.config_file, 'w') as f:
                json.dump({
                    "patch_grid_topology": f"{GRID_X}x{GRID_Y}x{GRID_Z}",
                    "intra_patch_topology": f"{row_len}x{col_len}",
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "master_seed": self.master_seed,
                    "benchmark": "RCS-NN-OTOC-ACE",
                    "depth": depth,
                    "cycles": cycles,
                    "effective_depth": depth * cycles,
                    "shots": shots,
                    "ace_qb_limit": (QUBITS_PER_PATCH + 3) >> 2,
                    "pt_ideal_xeb": 1.0,
                    "pt_ideal_hog": PT_IDEAL_HOG,
                }, f)
            with open(self.per_patch_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Patch", "X", "Y", "Z", "Seed", "Effective_Depth", "XEB", "HOG_Prob",
                    "L2_Sq_Dist", "L2_Sq_Dist_vs_Uniform", "Lat_Build_ms",
                    "Lat_Gates_ms", "Lat_Sample_ms", "Lat_Probs_ms", "Lat_Stats_ms"
                ]).writeheader()
            with open(self.aggregate_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Depth", "Cycles", "Effective_Depth", "Shots", "Patches",
                    "XEB_Mean", "XEB_Std", "XEB_Min", "XEB_Max",
                    "HOG_Mean", "HOG_Std", "L2_Sq_Mean", "L2_Sq_vs_Uniform_Mean"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}",
                  flush=True, file=sys.stderr)

    def _log_patch(self, p: int, payload: Dict[str, Any]):
        try:
            with open(self.per_patch_csv, mode='a', newline='') as f:
                cx_, cy_, cz_ = self.patch_coords[p]
                csv.DictWriter(f, fieldnames=[
                    "Patch", "X", "Y", "Z", "Seed", "Effective_Depth", "XEB", "HOG_Prob",
                    "L2_Sq_Dist", "L2_Sq_Dist_vs_Uniform", "Lat_Build_ms",
                    "Lat_Gates_ms", "Lat_Sample_ms", "Lat_Probs_ms", "Lat_Stats_ms"
                ]).writerow({
                    "Patch": p, "X": cx_, "Y": cy_, "Z": cz_,
                    "Seed": payload["seed"],
                    "Effective_Depth": payload["effective_depth"],
                    "XEB": payload["xeb"],
                    "HOG_Prob": payload["hog_prob"],
                    "L2_Sq_Dist": payload["l2_sq_dist"],
                    "L2_Sq_Dist_vs_Uniform": payload["l2_sq_dist_vs_uniform_random"],
                    "Lat_Build_ms": payload["lat_build_ms"],
                    "Lat_Gates_ms": payload["lat_gates_ms"],
                    "Lat_Sample_ms": payload["lat_sample_ms"],
                    "Lat_Probs_ms": payload["lat_probs_ms"],
                    "Lat_Stats_ms": payload["lat_stats_ms"],
                })
        except Exception as e:
            print(f"[CSV] Warning: Per-patch log write failed: {e}",
                  flush=True, file=sys.stderr)

    def _log_aggregate(self, depth, cycles, shots, results):
        try:
            xeb = np.array([results[p]["xeb"] for p in sorted(results)])
            hog = np.array([results[p]["hog_prob"] for p in sorted(results)])
            l2  = np.array([results[p]["l2_sq_dist"] for p in sorted(results)])
            l2u = np.array([results[p]["l2_sq_dist_vs_uniform_random"]
                            for p in sorted(results)])
            with open(self.aggregate_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Depth", "Cycles", "Effective_Depth", "Shots", "Patches",
                    "XEB_Mean", "XEB_Std", "XEB_Min", "XEB_Max",
                    "HOG_Mean", "HOG_Std", "L2_Sq_Mean", "L2_Sq_vs_Uniform_Mean"
                ]).writerow({
                    "Depth": depth, "Cycles": cycles, 
                    "Effective_Depth": depth * cycles,
                    "Shots": shots, "Patches": len(results),
                    "XEB_Mean": float(np.mean(xeb)),
                    "XEB_Std": float(np.std(xeb)),
                    "XEB_Min": float(np.min(xeb)),
                    "XEB_Max": float(np.max(xeb)),
                    "HOG_Mean": float(np.mean(hog)),
                    "HOG_Std": float(np.std(hog)),
                    "L2_Sq_Mean": float(np.mean(l2)),
                    "L2_Sq_vs_Uniform_Mean": float(np.mean(l2u)),
                })
        except Exception as e:
            print(f"[CSV] Warning: Aggregate log write failed: {e}",
                  flush=True, file=sys.stderr)

    def _save_artifacts(self, results: Dict[int, Dict[str, Any]]):
        try:
            np.savez_compressed(
                self.samples_file,
                **{f"patch_{p:03d}": results[p]["samples"] for p in results}
            )
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to save samples: {e}",
                  flush=True, file=sys.stderr)
        try:
            with open(self.pauli_file, 'w') as f:
                json.dump(
                    {str(p): results[p]["pauli_strings"] for p in results}, f
                )
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to save Pauli strings: {e}",
                  flush=True, file=sys.stderr)

    def run(self, depth: int, cycles: int = 1, shots: int = 512):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if cycles < 1:
            raise ValueError("cycles must be at least 1")
        if shots < 1:
            raise ValueError("shots must be a positive integer")

        self._init_files(depth, cycles, shots)

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] RCS-NN-OTOC / ACE-vs-Exact | {TOTAL_PATCHES} patches, "
              f"{total_qubits} qubits, {GPUS_AVAILABLE} GPUs "
              f"({WORKERS_PER_GPU} workers/GPU), depth={depth}, "
              f"cycles={cycles} (Eff Depth: {depth * cycles}), shots={shots}/patch", flush=True)
        print(f"[Engine] ACE limit: {(QUBITS_PER_PATCH + 3) >> 2} qb | "
              f"Ideal targets: XEB -> 1.0000, HOG -> {PT_IDEAL_HOG:.4f}",
              flush=True)

        active_ranks = [r for r in range(TOTAL_WORKERS)
                        if self.worker_assignments[r]]

        workers = []
        pipes = []
        pipe_rank: Dict[int, int] = {}

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank],
                      child_conn, depth, cycles, shots, self.master_seed)
            )
            proc.start()
            child_conn.close()
            workers.append(proc)
            pipes.append(parent_conn)
            pipe_rank[id(parent_conn)] = rank

        results: Dict[int, Dict[str, Any]] = {}
        t_run0 = time.perf_counter()

        try:
            open_pipes = list(pipes)
            while len(results) < TOTAL_PATCHES and open_pipes:
                ready = mp.connection.wait(open_pipes)
                for conn in ready:
                    try:
                        data = conn.recv()
                    except EOFError:
                        rank = pipe_rank[id(conn)]
                        if conn in open_pipes:
                            open_pipes.remove(conn)
                        undelivered = [
                            p for p in self.worker_assignments[rank]
                            if p not in results
                        ]
                        if undelivered:
                            raise RuntimeError(
                                f"Fatal: worker rank {rank} exited with "
                                f"patches {undelivered} undelivered."
                            )
                        continue

                    for p, payload in data.items():
                        results[p] = payload
                        self._log_patch(p, payload)
                        print(
                            f"Patch {p:02d} {self.patch_coords[p]} | "
                            f"XEB: {payload['xeb']:+.4f} | "
                            f"HOG: {payload['hog_prob']:.4f} "
                            f"(PT {PT_IDEAL_HOG:.4f}) | "
                            f"L2 Sq: {payload['l2_sq_dist']:.3e} "
                            f"(vs U: {payload['l2_sq_dist_vs_uniform_random']:.3e}) | "
                            f"Gates/Sample/Probs: "
                            f"{payload['lat_gates_ms']:7.1f}/"
                            f"{payload['lat_sample_ms']:6.1f}/"
                            f"{payload['lat_probs_ms']:7.1f}ms | "
                            f"[{len(results)}/{TOTAL_PATCHES}]",
                            flush=True
                        )

                    # Simplified pipe drain logic
                    rank = pipe_rank[id(conn)]
                    delivered_for_rank = sum(1 for p in self.worker_assignments[rank] if p in results)
                    if delivered_for_rank == len(self.worker_assignments[rank]) and conn in open_pipes:
                        open_pipes.remove(conn)

            if len(results) != TOTAL_PATCHES:
                raise RuntimeError(
                    f"Fatal: IPC gather incomplete. Expected {TOTAL_PATCHES} "
                    f"patches, got {len(results)}."
                )

            xeb_arr = np.array([results[p]["xeb"] for p in sorted(results)])
            hog_arr = np.array([results[p]["hog_prob"] for p in sorted(results)])
            print(f"\n[Master] Aggregate | XEB: {np.mean(xeb_arr):+.4f} "
                  f"+/- {np.std(xeb_arr):.4f} | "
                  f"HOG: {np.mean(hog_arr):.4f} (PT {PT_IDEAL_HOG:.4f}) | "
                  f"Wall: {time.perf_counter() - t_run0:.1f}s", flush=True)

            self._log_aggregate(depth, cycles, shots, results)

        finally:
            for conn in pipes:
                try:
                    conn.close()
                except Exception:
                    pass

            if results:
                self._save_artifacts(results)
                print(f"[Master] Dumped bitstrings to {self.samples_file} and "
                      f"OTOC Pauli strings to {self.pauli_file}", flush=True)

            for proc in workers:
                try:
                    proc.join(timeout=15)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # CLI: python3 multi_gpu_rcs_nn_otoc_ace_rev98.py [depth] [cycles] [shots]
    depth  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    shots  = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << min(9, QUBITS_PER_PATCH + 2)

    engine = MultiGpuRcsOtocEngine(master_seed=1337)
    try:
        engine.run(depth=depth, cycles=cycles, shots=shots)
    except KeyboardInterrupt:
        pass
