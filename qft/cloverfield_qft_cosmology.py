# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with Statistical Variance Injection
#
# REVISION 89.1 - SEPARABLE-QUBIT SEMICLASSICAL QFT SAMPLER MERGE + ROBUSTNESS
#
# NEW (Rev 89.1):
# - EXPLICIT SPATIAL TRAVERSAL: The QFT sampler now enforces a strict 
#   lexicographical (X, Y, Z) ordering to ensure the spectral content of the 
#   output bitstring is topologically stable for downstream analysis.
# - CLOSURE GUARD: Added explicit None-check for qft_sim when enable_qft=False.
#
# NEW (Rev 89):
# - QFT PROBE: Merged the "entropic single separable qubit QFT" sampler
#   (semiclassical / Griffiths-Niu style QFT, see arXiv:1702.06959; concept
#   credit Dan Strano) as a per-patch spectral readout at every measure step.
#   Physics note: each patch is internally entangled, so its single-qubit
#   marginals are mixed. The sampler draws from the PRODUCT-OF-MARGINALS
#   (mean-field) shadow of the patch -- exactly the same object the engine
#   already uses for inter-patch coupling, so the probe is self-consistent
#   with the mean-field boundary approximation. Mixed marginals are handled
#   rigorously by sampling the pure-state ensemble {+r_hat, -r_hat} with
#   weights (1 +/- |r|)/2, which reproduces exact single-qubit statistics.
# - DETERMINISTIC SAMPLING: measurement outcomes are drawn from prob(0) with
#   a provenance-seeded RNG stream [QFT_SEED_TAG, master_seed, step, patch]
#   instead of qsim.m(), so bit strings are bit-exact reproducible.
# - WORKER-SIDE EXECUTION: the sampler runs inside each worker on a
#   persistent 1-qubit CPU simulator (is_gpu=False). mtrx() is safe here:
#   the rusticl JIT stack overflow only affects mtrx on large GPU
#   statevectors. The master process stays pyqrack-free.
# - NEW CSV: separable_qft_samples_multi.csv (Step, Patch, BitString, ms).
# - STANDALONE BENCH: original random-prep benchmark preserved via
#   `python <script> --qft-bench [n]`.
#
# BUGFIXES (Rev 88):
# - STOCHASTIC SCALING: Removed `measure_every` from the `scale` calculation
#   (now `np.sqrt(dt / effective_shots)`). Since the boundary field payload is
#   applied at every intermediate Trotter step, baking the lumped measurement
#   interval variance into the continuous field artificially inflated the noise.
#
# BUGFIXES (Rev 87):
# - CONTINUOUS COUPLING: Removed the non-measure payload reset in the worker loop.
# - INTEGRATION SCALING: Dropped the `m_every` multiplier from `apply_kicks`.
# - FIDELITY WARNING: Added a one-time stderr alert for missing PyQrack fidelity bindings.

import os
import sys
import gc
import csv
import json
import time
import math
import cmath
import random
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# Topography tuning for raw statevectors
GPUS_AVAILABLE = 1
WORKERS_PER_GPU = 3  # Adjust >1 for decisive experiment comparison
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# Distinct RNG stream tag for the QFT sampler so it never collides with the
# master's boundary-noise streams ([master_seed, t, p]).
QFT_SEED_TAG = 0x51F7

# =====================================================================
# ENVIRONMENT - set before pyqrack import
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

# =====================================================================
# PURE FUNCTIONS (Math & Topology)
# =====================================================================
def generate_27q_lattice_subvolume() -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    lx, ly, lz = 3, 3, 3
    edges: List[Tuple[int, int]] = []
    boundaries: Dict[str, List[int]] = {
        "+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []
    }

    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                idx = x * (ly * lz) + y * lz + z
                if x < lx - 1: edges.append((idx, (x + 1) * (ly * lz) + y * lz + z))
                if y < ly - 1: edges.append((idx, x * (ly * lz) + (y + 1) * lz + z))
                if z < lz - 1: edges.append((idx, x * (ly * lz) + y * lz + (z + 1)))

                if x == 0: boundaries["-X"].append(idx)
                if x == lx - 1: boundaries["+X"].append(idx)
                if y == 0: boundaries["-Y"].append(idx)
                if y == ly - 1: boundaries["+Y"].append(idx)
                if z == 0: boundaries["-Z"].append(idx)
                if z == lz - 1: boundaries["+Z"].append(idx)

    return edges, boundaries

# =====================================================================
# STANDALONE BENCHMARK (original Strano script, random separable prep)
# Invoke with: python <script> --qft-bench [n]
# =====================================================================
def bench_qrack(n):
    # Discrete Fourier transform after initializing all qubits randomly
    # but separably. (See https://arxiv.org/abs/1702.06959)
    from pyqrack import QrackSimulator

    start = time.perf_counter()

    qsim = QrackSimulator(1, is_schmidt_decompose=False,
                          is_stabilizer_hybrid=False, is_gpu=False)

    result_bits = []
    for c in range(n):
        qsim.u(
            0,
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        )
        qsim.h(0)
        phase_factor = cmath.exp(1j * math.pi)
        for t in range(c):
            phase_factor = cmath.sqrt(phase_factor)
            # FIX: Iterate backward against measurement history for correct QFT phasing
            if result_bits[c - 1 - t]:
                qsim.mtrx([1.0, 0.0, 0.0, phase_factor], 0)
        b = qsim.m(0)
        result_bits.append(b)
        if b:
            qsim.x(0)

    bit_string = "".join("1" if b else "0" for b in result_bits)
    return (time.perf_counter() - start, bit_string)

# =====================================================================
# WORKER PROCESS LOGIC
# =====================================================================
def gpu_worker_process(
    rank: int,
    workers_per_gpu: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    dt: float,
    total_steps: int,
    initial_hx: float,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int,
    master_seed: int,
    enable_qft: bool
):
    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    # Map multiple ranks to the same physical GPU device index
    physical_gpu_index = rank // workers_per_gpu
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)

    # Bind QPager to the assigned device to enable driver-level PCIe paging
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    # Unleash VRAM allocations, proportionally capping by worker density
    alloc_mb = 64000 // workers_per_gpu
    os.environ["QRACK_MAX_ALLOC_MB"] = str(alloc_mb)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}
    qft_sim = None

    try:
        # --- PAULI CODE AUTODETECT ---
        _THRESH = 0.5

        # Z Probe
        _probe_z = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        vals0_z = {}
        for _code in range(8):
            try: vals0_z[_code] = _probe_z.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_z.x(0)
        PZ, SIGN_Z = None, None
        for _code, v0 in vals0_z.items():
            try: v1 = _probe_z.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PZ = _code
                SIGN_Z = 1.0 if v0 > 0 else -1.0
                break
        if PZ is None:
            raise RuntimeError("Fatal: could not autodetect PZ code")
        del _probe_z

        # X Probe
        _probe_x = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _probe_x.h(0)
        vals0_x = {}
        for _code in range(8):
            if _code == PZ: continue
            try: vals0_x[_code] = _probe_x.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_x.z(0)
        PX, SIGN_X = None, None
        for _code, v0 in vals0_x.items():
            try: v1 = _probe_x.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PX = _code
                SIGN_X = 1.0 if v0 > 0 else -1.0
                break
        if PX is None:
            raise RuntimeError("Fatal: could not autodetect PX code")
        del _probe_x

        # Y Probe
        _probe_y = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _c, _s = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        _probe_y.mtrx([complex(_c, 0), complex(0, _s), complex(0, _s), complex(_c, 0)], 0)
        vals0_y = {}
        for _code in range(8):
            if _code in (PX, PZ): continue
            try: vals0_y[_code] = _probe_y.pauli_expectation([0], [_code])
            except Exception: pass
        _probe_y.z(0)
        PY, SIGN_Y = None, None
        for _code, v0 in vals0_y.items():
            try: v1 = _probe_y.pauli_expectation([0], [_code])
            except Exception: continue
            if abs(v0) > _THRESH and abs(v1) > _THRESH and (v0 * v1) < 0:
                PY = _code
                SIGN_Y = 1.0 if v0 > 0 else -1.0
                break
        if PY is None:
            raise RuntimeError("Fatal: could not autodetect PY code")
        del _probe_y
        # ----------------------------

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        mag_check = _sim_mag.pauli_expectation([0], [PZ])
        _corrected = SIGN_Z * mag_check
        if abs(_corrected + 1.0) < 0.1:
            ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1:
            ANGLE_SCALE = 0.5
        else:
            raise RuntimeError(
                f"Fatal: r(PX,pi) returned ambiguous SIGN_Z*<Z> = {_corrected:.6f}; "
                f"expected ~+1.0 or ~-1.0"
            )
        del _sim_mag
        # -------------------------------------------------------

        def apply_h(sim, q): sim.h(q)

        def apply_rx(sim, theta, q):
            sim.r(PX, float(theta) * ANGLE_SCALE, q)

        def apply_ry(sim, theta, q):
            sim.r(PY, float(theta) * ANGLE_SCALE, q)

        def apply_rz(sim, theta, q):
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, intra_edges, J, hx, hz, dt_local):
            dt_half = dt_local / 2.0

            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)
            for q1, q2 in intra_edges: apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim, qubits):
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])

        def x_means(sim, qubits):
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])

        def y_means(sim, qubits):
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])

        def zz_means_meanfield(z_exp, edges):
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks, dt_local):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)

                # Continuous evolution across all steps: m_every multiplier removed
                coef = -2.0 * dt_local

                theta_x = kx * coef
                theta_y = ky * coef
                theta_z = kz * coef

                if abs(theta_x) > 1e-12: apply_rx(sim, theta_x, q)
                if abs(theta_y) > 1e-12: apply_ry(sim, theta_y, q)
                if abs(theta_z) > 1e-12: apply_rz(sim, theta_z, q)

        # =============================================================
        # SEPARABLE-QUBIT SEMICLASSICAL QFT SAMPLER (Rev 89.1)
        # Semiclassical QFT over the patch's single-qubit marginals.
        # Standard gates only (u/h/mtrx/prob) -> no Pauli-code or angle
        # convention dependency. 1-qubit CPU sim; mtrx is safe here.
        # =============================================================
        if enable_qft:
            qft_sim = QrackSimulator(qubit_count=1, is_schmidt_decompose=False,
                                     is_stabilizer_hybrid=False, is_gpu=False)

        _QFT_EPS = 1e-12

        def separable_qft_sample(bx, by, bz, rng):
            if qft_sim is None:
                return "", 0.0
                
            t_start = time.perf_counter()
            n = len(bz)
            result_bits = []
            for c in range(n):
                qft_sim.reset_all()

                # Mixed marginal rho = (I + r.sigma)/2 sampled as the pure
                # ensemble {+r_hat, -r_hat} with weights (1 +/- |r|)/2.
                rx_, ry_, rz_ = float(bx[c]), float(by[c]), float(bz[c])
                norm = math.sqrt(rx_ * rx_ + ry_ * ry_ + rz_ * rz_)
                if norm < _QFT_EPS:
                    ux, uy, uz = 0.0, 0.0, 1.0
                    p_plus = 0.5
                else:
                    r_len = min(norm, 1.0)  # clip tomography noise overshoot
                    ux, uy, uz = rx_ / norm, ry_ / norm, rz_ / norm
                    p_plus = 0.5 * (1.0 + r_len)
                if rng.random() >= p_plus:
                    ux, uy, uz = -ux, -uy, -uz

                theta = math.acos(max(-1.0, min(1.0, uz)))
                phi = math.atan2(uy, ux)
                qft_sim.u(0, theta, phi, 0.0)

                qft_sim.h(0)
                phase_factor = cmath.exp(1j * math.pi)
                for tq in range(c):
                    phase_factor = cmath.sqrt(phase_factor)
                    # FIX: Iterate backward against measurement history for correct QFT phasing
                    if result_bits[c - 1 - tq]:
                        qft_sim.mtrx([1.0, 0.0, 0.0, phase_factor], 0)

                # Deterministic, provenance-seeded measurement: sample from
                # prob(0) with our RNG instead of qsim.m()'s internal RNG.
                p1 = float(qft_sim.prob(0))
                b = 1 if rng.random() < p1 else 0
                result_bits.append(b)

            bit_string = "".join("1" if b else "0" for b in result_bits)
            return bit_string, (time.perf_counter() - t_start)

        intra_edges, boundaries = generate_27q_lattice_subvolume()
        
        # Explicit canonical spatial traversal mapping for QFT stability
        spatial_order = [x * 9 + y * 3 + z for x in range(3) for y in range(3) for z in range(3)]

        for p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[p] = sim

            # --- VRAM PAGING SMOKE TEST ---
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM/PCIe paging allocation failed on patch {p}. Driver error: {e}")

        kick_payloads = {p: {} for p in assigned_patches}
        _warned_fidelity = False

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            patch_data_to_master = {}

            for p in assigned_patches:
                sim = sims[p]
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt)

                t_start_trotter = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                t_lat_trotter = time.perf_counter() - t_start_trotter

                t_lat_tomo = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
                    all_q = list(range(QUBITS_PER_PATCH))
                    state = {
                        "Z": z_means(sim, all_q),
                        "X": x_means(sim, all_q),
                        "Y": y_means(sim, all_q),
                    }
                    zz_exp = zz_means_meanfield(state["Z"], intra_edges)
                    bulk_e = (-current_hz * float(np.sum(state["Z"]))
                              - current_J  * float(np.sum(zz_exp))
                              - current_hx * float(np.sum(state["X"])))
                    t_lat_tomo = time.perf_counter() - t_start_tomo

                    try:
                        fidelity = float(sim.get_unitary_fidelity())
                    except AttributeError:
                        fidelity = 1.0  # Fallback if API lacks binding
                        if not _warned_fidelity:
                            print(f"[Worker {rank}] Warning: sim.get_unitary_fidelity() not found. Upgrade PyQrack for true fidelity tracking.", file=sys.stderr)
                            _warned_fidelity = True

                    # --- SEMICLASSICAL QFT PROBE (Rev 89.1) ---
                    qft_bits, qft_lat = "", 0.0
                    if enable_qft:
                        qft_rng = np.random.default_rng(
                            [QFT_SEED_TAG, master_seed, t, p])
                        qft_bits, qft_lat = separable_qft_sample(
                            state["X"][spatial_order], 
                            state["Y"][spatial_order], 
                            state["Z"][spatial_order], 
                            qft_rng
                        )

                    patch_data_to_master[p] = {
                        "state": state,
                        "meanfield_bulk_energy": bulk_e,
                        "lat_trotter_ms": t_lat_trotter * 1000.0,
                        "lat_tomo_ms": t_lat_tomo * 1000.0,
                        "unitary_fidelity": fidelity,
                        "qft_bits": qft_bits,
                        "lat_qft_ms": qft_lat * 1000.0
                    }

            if is_measure:
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()

            # The 'else: kick_payloads = {}' block was removed here to
            # retain the boundary field across non-measure intermediate steps.

    finally:
        for p in list(sims.keys()):
            _s = sims.pop(p)
            del _s
        sims.clear()
        if qft_sim is not None:
            del qft_sim
        gc.collect()
        conn.close()

# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337):
        self.master_seed = master_seed
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.all_boundary_qubits = sorted(set(q for face in self.boundaries.values() for q in face))
        self._bq_to_idx = {q: i for i, q in enumerate(self.all_boundary_qubits)}
        self._bq_arr    = np.array(self.all_boundary_qubits, dtype=np.intp)

        self.patch_coords = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.lattice_history  = []
        self.energy_csv       = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.qft_csv          = "separable_qft_samples_multi.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._init_files()

        self.worker_assignments = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                           "num_patches": TOTAL_PATCHES,
                           "qubits_per_patch": QUBITS_PER_PATCH,
                           "qft_seed_tag": QFT_SEED_TAG,
                           "master_seed": self.master_seed}, f)
            with open(self.energy_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity"
                ]).writeheader()
            with open(self.profiles_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ]).writeheader()
            with open(self.qft_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "QFT_BitString", "QFT_ms"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step, anneal, bulk, bound, total, min_fidelity,
                  patch_profiles, qft_samples):
        try:
            with open(self.energy_csv, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity"
                ]).writerow({"Step": step, "Anneal_Percent": anneal,
                             "MeanField_Bulk_Energy": bulk, "MeanField_Boundary_Energy": bound,
                             "MeanField_Total_Energy": total,
                             "Min_Unitary_Fidelity": min_fidelity})

            with open(self.profiles_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"
                ])
                for p, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": p, "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})

            if qft_samples:
                with open(self.qft_csv, mode='a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "Step", "Patch", "QFT_BitString", "QFT_ms"
                    ])
                    for p in sorted(qft_samples.keys()):
                        bits, lat_ms = qft_samples[p]
                        w.writerow({"Step": step, "Patch": p,
                                    "QFT_BitString": bits,
                                    "QFT_ms": f"{lat_ms:.3f}"})
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float, target_g_face: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0,
            enable_qft_sampling: bool = True):

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] {TOTAL_PATCHES} patches, {total_qubits} qubits, {GPUS_AVAILABLE} GPUs ({WORKERS_PER_GPU} workers/GPU), {total_steps} steps"
              + (" | QFT probe: ON" if enable_qft_sampling else " | QFT probe: OFF"))

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=gpu_worker_process,
                args=(rank, WORKERS_PER_GPU, self.worker_assignments[rank], child_conn,
                      dt, total_steps, initial_hx, target_J, target_hx, target_hz,
                      measure_every, self.master_seed, enable_qft_sampling)
            )
            p.start()
            child_conn.close()
            workers.append(p)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s = t / max(1, (total_steps - 1))
                current_g_face = s * target_g_face
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)

                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                patch_full_states = {}
                qft_samples = {}
                bulk_energy = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo = 0.0
                max_lat_qft = 0.0
                min_fidelity = 1.0

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                        min_fidelity = min(min_fidelity, payload.get("unitary_fidelity", 1.0))
                        if payload.get("qft_bits"):
                            qft_samples[p] = (payload["qft_bits"],
                                              payload.get("lat_qft_ms", 0.0))
                            max_lat_qft = max(max_lat_qft, payload.get("lat_qft_ms", 0.0))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(patch_full_states)}."
                    )

                # --- BUILD PROFILES ---
                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles = {}
                bq = self._bq_arr

                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    patch_profiles[p] = {
                        "means": {
                            "X": state["X"][bq],
                            "Y": state["Y"][bq],
                            "Z": state["Z"][bq],
                        },
                        "vars": {
                            "X": np.clip(1.0 - state["X"][bq]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"][bq]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"][bq]**2, 0.0, 1.0),
                        }
                    }

                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Failed to save: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & BOUNDARY ENERGY ---
                next_kick_payloads = {p: {} for p in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0

                scale = np.sqrt(dt / effective_shots)
                stochastic_noise = {}
                n_b = len(self.all_boundary_qubits)

                for p in range(TOTAL_PATCHES):
                    prof = patch_profiles[p]
                    rng_p = np.random.default_rng([self.master_seed, t, p])

                    xn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Z"]) * scale

                    stochastic_noise[p] = {
                        q: (xn[i], yn[i], zn[i])
                        for i, q in enumerate(self.all_boundary_qubits)
                    }

                for p1, coord1 in self.patch_coords.items():
                    x1, y1, z1 = coord1
                    neighbors = {
                        "+X": (x1+1, y1,   z1  ), "-X": (x1-1, y1,   z1  ),
                        "+Y": (x1,   y1+1, z1  ), "-Y": (x1,   y1-1, z1  ),
                        "+Z": (x1,   y1,   z1+1), "-Z": (x1,   y1,   z1-1),
                    }

                    for dir1, coord2 in neighbors.items():
                        p2 = self.coord_to_patch.get(coord2)
                        if p2 is None or p1 >= p2: continue

                        dir2    = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
                        face1_q = self.boundaries[dir1]
                        face2_q = self.boundaries[dir2]
                        prof1, noise1 = patch_profiles[p1], stochastic_noise[p1]
                        prof2, noise2 = patch_profiles[p2], stochastic_noise[p2]

                        ax2 = np.mean([prof2["means"]["X"][self._bq_to_idx[q]] + noise2[q][0] for q in face2_q])
                        ay2 = np.mean([prof2["means"]["Y"][self._bq_to_idx[q]] + noise2[q][1] for q in face2_q])
                        az2 = np.mean([prof2["means"]["Z"][self._bq_to_idx[q]] + noise2[q][2] for q in face2_q])

                        ax1 = np.mean([prof1["means"]["X"][self._bq_to_idx[q]] + noise1[q][0] for q in face1_q])
                        ay1 = np.mean([prof1["means"]["Y"][self._bq_to_idx[q]] + noise1[q][1] for q in face1_q])
                        az1 = np.mean([prof1["means"]["Z"][self._bq_to_idx[q]] + noise1[q][2] for q in face1_q])

                        macroscopic_boundary_energy += (
                            -current_g_face
                            * (ax1*ax2 + ay1*ay2 + az1*az2)
                            * ((len(face1_q) + len(face2_q)) / 2.0)
                        )

                        for q1f in face1_q:
                            k = next_kick_payloads[p1].get(q1f, (0., 0., 0.))
                            next_kick_payloads[p1][q1f] = (
                                k[0] + current_g_face * ax2,
                                k[1] + current_g_face * ay2,
                                k[2] + current_g_face * az2,
                            )
                        for q2f in face2_q:
                            k = next_kick_payloads[p2].get(q2f, (0., 0., 0.))
                            next_kick_payloads[p2][q2f] = (
                                k[0] + current_g_face * ax1,
                                k[1] + current_g_face * ay1,
                                k[2] + current_g_face * az1,
                            )

                total_energy = bulk_energy + macroscopic_boundary_energy
                print(f"Step {t:03d} | E: {total_energy:+.4f} | Lat(Trot/Tomo/QFT): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}/{max_lat_qft:5.1f}ms | Fid: {min_fidelity:.5f} | {time.perf_counter() - t0:.2f}s")
                self._log_csvs(t, s * 100, bulk_energy,
                               macroscopic_boundary_energy, total_energy, min_fidelity,
                               patch_profiles, qft_samples)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {p: next_kick_payloads[p]
                                      for p in self.worker_assignments[w_rank]}
                    pipes[i].send(worker_payload)

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass

            if self.lattice_history:
                try:
                    np.save(self.state_dump_file, np.array(self.lattice_history))
                    print(f"\n[Master] Dumped history matrix to {self.state_dump_file}")
                except Exception as e:
                    print(f"\n[Master] Failed to save lattice history: {e}", file=sys.stderr)

            for proc in workers:
                proc.join(timeout=15)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
                    if proc.is_alive():
                        try: proc.kill()
                        except Exception: pass


if __name__ == "__main__":
    # Standalone benchmark mode (original Strano script behavior):
    #   python <script> --qft-bench [n]
    if len(sys.argv) > 1 and sys.argv[1] == "--qft-bench":
        max_qb = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        bench_qrack(1)  # warm-up
        elapsed, bits = bench_qrack(max_qb)
        print(max_qb, "qubits,", elapsed, "seconds,", bits, "measurement result")
        sys.exit(0)

    mp.set_start_method('spawn', force=True)

    engine = MultiGpuHadronEngine(master_seed=1337)
    try:
        engine.run(
            total_steps=100,
            dt=0.04,
            initial_hx=3.0,
            target_g_face=0.15,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0,
            enable_qft_sampling=True
        )
    except KeyboardInterrupt:
        pass
