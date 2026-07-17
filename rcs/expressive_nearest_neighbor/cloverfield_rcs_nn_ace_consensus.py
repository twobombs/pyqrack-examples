# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with Statistical Variance Injection
# + In-Place RCS Layer with Inverse-Circuit Restoration (Rev 90.7)
#
# REVISION 90.7 - RCS ON TROTTER (Option 4)
#
# ARCHITECTURE:
# - RCS LAYER: At each RCS validation step, a random circuit of depth
#   `RCS_DEPTH` is applied IN-PLACE to the exact simulator. The circuit
#   is recorded as an ordered gate list so the exact inverse can be
#   applied immediately after sampling, restoring the Trotter state
#   without any clone, out_ket, or in_ket. Zero additional VRAM.
#
# - GATE SET: u(theta, phi, lambda) for single-qubit rotations + iswap
#   for entangling gates. Both have confirmed adjoints in the PyQrack API:
#   u adjoint = u(-theta, -lambda, -phi); iswap adjoint = adjiswap.
#
# - XEB: Standard linear XEB = 2^n * mean(p_ideal(samples)) - 1.
#   After the random layer the output distribution approaches Porter-Thomas,
#   making this a genuine RCS XEB score.
#   Samples are drawn from the post-RCS exact sim.
#   p_ideal is queried via prob_perm on the post-RCS exact sim for each
#   sampled bitstring.
#
# - INVERSE RESTORATION: After sampling and prob_perm, the gate list is 
#   reversed and each gate replaced by its adjoint, restoring the exact 
#   Trotter state mathematically (no approximation).
#
# BUGFIXES & UPGRADES (Rev 90.7):
# - GPUS_AVAILABLE restored to 6 for the production hardware topography.
# - measure_shots runtime smoke test upgraded to 27 qubits to guarantee
#   the check traverses the identical multi-qubit code path as the main sim.
# - Trotter step upgraded to True Strang Splitting: 
#   Rx(1/2) -> Rz(1/2) -> ZZ(full) -> Rz(1/2) -> Rx(1/2).

import os
import sys
import gc
import csv
import json
import time
import math
import random
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# 6-GPU Symmetrical Topography (AMD Radeon Pro V340 x6)
GPUS_AVAILABLE = 6
WORKERS_PER_GPU = 1
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# --- RCS CONFIGURATION ---
RCS_VALIDATION_ENABLED = True
RCS_DEPTH = 20                # Random circuit depth (layers of u + iswap)
RCS_SHOTS = 256               # Samples drawn from post-RCS state per probe patch
RCS_VALIDATE_EVERY = 5        # In units of measure steps
RCS_PROBE_PATCHES = None      # None = all patches; [13] = center only

# Bit-ordering convention for prob_perm bitmask construction.
# False = LSB-first (qubit 0 = bit 0 of the integer outcome).
RCS_MSB_FIRST = False

# =====================================================================
# ENVIRONMENT
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

# =====================================================================
# PURE FUNCTIONS
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
                if x == 0:      boundaries["-X"].append(idx)
                if x == lx - 1: boundaries["+X"].append(idx)
                if y == 0:      boundaries["-Y"].append(idx)
                if y == ly - 1: boundaries["+Y"].append(idx)
                if z == 0:      boundaries["-Z"].append(idx)
                if z == lz - 1: boundaries["+Z"].append(idx)
    return edges, boundaries


def calc_xeb(ideal_probs: np.ndarray, width: int) -> Tuple[float, float]:
    """Standard linear XEB and HOG fraction.

    XEB   = 2^n * <p_ideal(sample)> - 1
    HOG   = fraction of samples where p_ideal > ln(2)/2^n
    """
    n_pow = float(1 << width)
    if ideal_probs.size == 0:
        return 0.0, 0.0
    xeb = n_pow * float(np.mean(ideal_probs)) - 1.0
    hog = float(np.mean(ideal_probs > (math.log(2.0) / n_pow)))
    return xeb, hog


# =====================================================================
# RCS LAYER: APPLY AND INVERSE
# =====================================================================
def apply_rcs_layer(sim, num_qubits: int, edges: List[Tuple[int, int]],
                    depth: int, rng: random.Random) -> List[tuple]:
    """Apply `depth` layers of random u + iswap gates to `sim`."""
    gate_list: List[tuple] = []

    for _ in range(depth):
        # --- Single-qubit layer: random u on every qubit ---
        for q in range(num_qubits):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            lam   = rng.uniform(0.0, 2.0 * math.pi)
            sim.u(q, theta, phi, lam)
            gate_list.append(('u', q, theta, phi, lam))

        # --- Entangling layer: random non-overlapping iswap pairs ---
        shuffled = list(edges)
        rng.shuffle(shuffled)
        used: set = set()
        for q1, q2 in shuffled:
            if q1 not in used and q2 not in used:
                sim.iswap(q1, q2)
                gate_list.append(('iswap', q1, q2))
                used.add(q1)
                used.add(q2)

    return gate_list


def apply_rcs_layer_inverse(sim, gate_list: List[tuple]) -> None:
    """Apply the exact inverse of an RCS gate list to `sim`.
    
    Mathematically exact adjoints based on PyQrack definitions:
    u(theta, phi, lambda)† = u(-theta, -lambda, -phi)
    iswap† = adjiswap
    """
    for gate in reversed(gate_list):
        if gate[0] == 'u':
            _, q, theta, phi, lam = gate
            sim.u(q, -theta, -lam, -phi)
        elif gate[0] == 'iswap':
            _, q1, q2 = gate
            sim.adjiswap(q1, q2)


# =====================================================================
# WORKER PROCESS
# =====================================================================
def gpu_worker_process(
    rank: int,
    assigned_patches: List[int],
    conn: mp.connection.Connection,
    dt: float,
    total_steps: int,
    initial_hx: float,
    target_J: float,
    target_hx: float,
    target_hz: float,
    measure_every: int,
    rcs_cfg: Dict[str, Any],
) -> None:
    import time as _t
    stagger_time = rank * 10.0
    if stagger_time > 0:
        _t.sleep(stagger_time)
    print(f"[Worker {rank}] Awakening after {stagger_time:.0f}s stagger...", flush=True)

    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    physical_gpu_index = rank // WORKERS_PER_GPU
    os.environ["QRACK_OCL_DEFAULT_DEVICE"]    = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"]        = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"]    = str(physical_gpu_index)
    os.environ["QRACK_MAX_ALLOC_MB"]          = str(8000 // WORKERS_PER_GPU)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    from pyqrack import QrackSimulator

    sims: Dict[int, Any] = {}

    try:
        # ----------------------------------------------------------------
        # PAULI CODE AUTODETECT
        # ----------------------------------------------------------------
        _THRESH = 0.5

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
                PZ = _code; SIGN_Z = 1.0 if v0 > 0 else -1.0; break
        if PZ is None: raise RuntimeError("Fatal: could not autodetect PZ code")
        del _probe_z

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
                PX = _code; SIGN_X = 1.0 if v0 > 0 else -1.0; break
        if PX is None: raise RuntimeError("Fatal: could not autodetect PX code")
        del _probe_x

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
                PY = _code; SIGN_Y = 1.0 if v0 > 0 else -1.0; break
        if PY is None: raise RuntimeError("Fatal: could not autodetect PY code")
        del _probe_y

        # ----------------------------------------------------------------
        # ANGLE CONVENTION AUTODETECT
        # ----------------------------------------------------------------
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        _corrected = SIGN_Z * _sim_mag.pauli_expectation([0], [PZ])
        if   abs(_corrected + 1.0) < 0.1: ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1: ANGLE_SCALE = 0.5
        else: raise RuntimeError(f"Fatal: ambiguous ANGLE_SCALE, SIGN_Z*<Z>={_corrected:.6f}")
        del _sim_mag

        # ----------------------------------------------------------------
        # MEASURE_SHOTS DESTRUCTIVE SMOKE TEST
        # ----------------------------------------------------------------
        # Scaled to full QUBITS_PER_PATCH to force the exact multi-qubit codepaths
        # that will be used during the RCS sequences.
        _probe_nd = QrackSimulator(
            qubit_count=QUBITS_PER_PATCH,
            is_binary_decision_tree=False,
            is_stabilizer_hybrid=False,
            is_gpu=True,
        )
        for _q in range(QUBITS_PER_PATCH): _probe_nd.h(_q)
        p_before = _probe_nd.prob(0)
        _ = _probe_nd.measure_shots(list(range(QUBITS_PER_PATCH)), 64)
        p_after = _probe_nd.prob(0)
        if abs(p_before - p_after) > 0.01:
            raise RuntimeError("Fatal: measure_shots is destructive — RCS XEB invalid.")
        del _probe_nd

        # ----------------------------------------------------------------
        # GATE HELPERS
        # ----------------------------------------------------------------
        def apply_rx(sim, theta, q): sim.r(PX, float(theta) * ANGLE_SCALE, q)
        def apply_ry(sim, theta, q): sim.r(PY, float(theta) * ANGLE_SCALE, q)
        def apply_rz(sim, theta, q): sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2); apply_rz(sim, 2.0 * theta, q2); sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, edge_list, J, hx, hz, dt_local):
            dt_half = dt_local / 2.0
            theta_x      = -2.0 * hx * dt_half
            theta_z_half = -2.0 * hz * dt_half
            theta_zz     = -J * dt_local
            
            # True Strang Splitting: Rx(1/2) -> Rz(1/2) -> ZZ(full) -> Rz(1/2) -> Rx(1/2)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z_half, q)
            for q1, q2 in edge_list:    apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits): apply_rz(sim, theta_z_half, q)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim, qubits):
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])
        def x_means(sim, qubits):
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])
        def y_means(sim, qubits):
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])
        def zz_means_mf(z_exp, edges):
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks, time_delta):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                coef = -2.0 * time_delta
                if abs(kx * coef) > 1e-12: apply_rx(sim, kx * coef, q)
                if abs(ky * coef) > 1e-12: apply_ry(sim, ky * coef, q)
                if abs(kz * coef) > 1e-12: apply_rz(sim, kz * coef, q)

        # ----------------------------------------------------------------
        # TOPOLOGY
        # ----------------------------------------------------------------
        intra_edges, boundaries = generate_27q_lattice_subvolume()
        all_q = list(range(QUBITS_PER_PATCH))

        # ----------------------------------------------------------------
        # RCS CONFIG
        # ----------------------------------------------------------------
        rcs_enabled = rcs_cfg.get("enabled", False)
        if rcs_cfg.get("probe_patches") is None:
            rcs_probe_set = set(assigned_patches)
        else:
            rcs_probe_set = set(rcs_cfg["probe_patches"]) & set(assigned_patches)

        master_seed = rcs_cfg.get("master_seed", 1337)

        _warned_fidelity = False

        # ----------------------------------------------------------------
        # SIMULATOR ALLOCATION
        # ----------------------------------------------------------------
        for patch_id in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): sim.h(q)
            sims[patch_id] = sim
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM smoke test failed on patch {patch_id}: {e}")

        print(f"[Worker {rank}] {len(assigned_patches)} patches allocated.", flush=True)

        # ----------------------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------------------
        kick_payloads = {patch_id: {} for patch_id in assigned_patches}
        meas_count    = 0
        time_delta    = dt * measure_every

        for t in range(total_steps):
            s          = t / max(1, total_steps - 1)
            current_hx = (1.0 - s) * initial_hx + s * rcs_cfg["target_hx"]
            current_J  = s * target_J
            current_hz = s * rcs_cfg["target_hz"]
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)
            do_rcs     = (
                rcs_enabled and is_measure and
                ((meas_count % max(1, rcs_cfg["validate_every"]) == 0)
                 or (t == total_steps - 1))
            )

            patch_data = {}

            for patch_id in assigned_patches:
                sim = sims[patch_id]

                # Boundary kicks (lump-sum, measure steps only)
                if is_measure and kick_payloads[patch_id]:
                    apply_kicks(sim, kick_payloads[patch_id], time_delta)

                # Trotter step
                t0_tr = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                lat_trotter = (time.perf_counter() - t0_tr) * 1000.0

                if not is_measure:
                    continue

                # Tomography
                t0_tomo = time.perf_counter()
                state = {
                    "Z": z_means(sim, all_q),
                    "X": x_means(sim, all_q),
                    "Y": y_means(sim, all_q),
                }
                zz_exp = zz_means_mf(state["Z"], intra_edges)
                bulk_e = (
                    -current_hz * float(np.sum(state["Z"]))
                    - current_J  * float(np.sum(zz_exp))
                    - current_hx * float(np.sum(state["X"]))
                )
                lat_tomo = (time.perf_counter() - t0_tomo) * 1000.0

                try:
                    # Will return constant 1.0 for exact simulation.
                    # Retained as telemetry placeholder for future runs using SDRP approximate sim.
                    fidelity = float(sim.get_unitary_fidelity())
                except AttributeError:
                    fidelity = 1.0
                    if not _warned_fidelity:
                        print(f"[Worker {rank}] Warning: get_unitary_fidelity() unavailable.",
                              file=sys.stderr)
                        _warned_fidelity = True

                # ----------------------------------------------------
                # RCS VALIDATION (in-place + inverse restoration)
                # ----------------------------------------------------
                rcs_xeb, rcs_hog = None, None
                lat_rcs = 0.0

                if do_rcs and patch_id in rcs_probe_set:
                    t0_rcs = time.perf_counter()
                    try:
                        depth      = rcs_cfg["depth"]
                        shots      = rcs_cfg["shots"]
                        msb_first  = rcs_cfg["msb_first"]

                        # Deterministic RNG: same seed -> same circuit
                        rng = random.Random((master_seed << 32) ^ (patch_id << 16) ^ t)

                        # 1. Apply random circuit in-place
                        gate_list = apply_rcs_layer(
                            sim, QUBITS_PER_PATCH, intra_edges, depth, rng
                        )

                        # 2. Draw samples from post-RCS state.
                        # Protected by runtime smoke test verifying this is non-destructive
                        # despite upstream documentation classifying it identically to m().
                        samples = sim.measure_shots(all_q, shots)

                        # 3. Query p_ideal for each sampled bitstring on the intact post-RCS state
                        ideal_p_list = []
                        for o in samples:
                            if msb_first:
                                bitmask = [bool((int(o) >> (QUBITS_PER_PATCH - 1 - b)) & 1)
                                           for b in range(QUBITS_PER_PATCH)]
                            else:
                                bitmask = [bool((int(o) >> b) & 1)
                                           for b in range(QUBITS_PER_PATCH)]
                            ideal_p_list.append(sim.prob_perm(all_q, bitmask))

                        # 4. Final inverse: Restore Trotter State via exact adjoints
                        apply_rcs_layer_inverse(sim, gate_list)

                        ideal_p  = np.array(ideal_p_list, dtype=np.float64)
                        rcs_xeb, rcs_hog = calc_xeb(ideal_p, QUBITS_PER_PATCH)

                    except Exception as e:
                        print(f"[Worker {rank}] RCS validation error (patch {patch_id}): {e}",
                              file=sys.stderr)
                        rcs_xeb, rcs_hog = None, None

                    lat_rcs = (time.perf_counter() - t0_rcs) * 1000.0

                patch_data[patch_id] = {
                    "state":                state,
                    "meanfield_bulk_energy": bulk_e,
                    "lat_trotter_ms":        lat_trotter,
                    "lat_tomo_ms":           lat_tomo,
                    "lat_rcs_ms":            lat_rcs,
                    "unitary_fidelity":      fidelity,
                    "rcs_xeb":               rcs_xeb,
                    "rcs_hog":               rcs_hog,
                }

            if is_measure:
                meas_count += 1
                conn.send(patch_data)
                kick_payloads = conn.recv()

    finally:
        for patch_id in list(sims.keys()):
            s = sims.pop(patch_id); del s
        sims.clear()
        gc.collect()
        conn.close()


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337) -> None:
        self.master_seed = master_seed
        self.intra_edges, self.boundaries = generate_27q_lattice_subvolume()
        self.all_boundary_qubits = sorted(
            set(q for face in self.boundaries.values() for q in face)
        )
        self._bq_to_idx = {q: i for i, q in enumerate(self.all_boundary_qubits)}
        self._bq_arr    = np.array(self.all_boundary_qubits, dtype=np.intp)

        self.patch_coords: Dict[int, Tuple[int, int, int]] = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.lattice_history: List[np.ndarray] = []
        self.energy_csv    = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv  = "boundary_profiles_multi.csv"
        self.rcs_csv       = "rcs_validation_multi.csv"
        self.state_dump    = "macroscopic_lattice_states.npy"
        self.config_file   = "lattice_config.json"

        self._init_files()

        self.worker_assignments: List[List[int]] = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    # ------------------------------------------------------------------
    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "rcs_depth": RCS_DEPTH,
                    "rcs_shots": RCS_SHOTS,
                    "rcs_validate_every": RCS_VALIDATE_EVERY,
                    "rcs_probe_patches": RCS_PROBE_PATCHES,
                    "rcs_msb_first": RCS_MSB_FIRST,
                }, f)
            with open(self.energy_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity",
                ]).writeheader()
            with open(self.profiles_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean",
                ]).writeheader()
            with open(self.rcs_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Patch", "RCS_Depth",
                    "XEB_RCS", "HOG_RCS",
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: init failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    def _log_energy(self, step: int, anneal: float, bulk: float,
                    bound: float, total: float, min_fid: float) -> None:
        try:
            with open(self.energy_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
                    "MeanField_Boundary_Energy", "MeanField_Total_Energy",
                    "Min_Unitary_Fidelity",
                ]).writerow({
                    "Step": step, "Anneal_Percent": anneal,
                    "MeanField_Bulk_Energy": bulk,
                    "MeanField_Boundary_Energy": bound,
                    "MeanField_Total_Energy": total,
                    "Min_Unitary_Fidelity": min_fid,
                })
        except Exception as e:
            print(f"[CSV] Energy log error: {e}", file=sys.stderr)

    def _log_profiles(self, step: int, patch_profiles: dict) -> None:
        try:
            with open(self.profiles_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean",
                ])
                for patch_id, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": patch_id, "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception as e:
            print(f"[CSV] Profile log error: {e}", file=sys.stderr)

    def _log_rcs(self, step: int, anneal: float,
                 rcs_records: List[Tuple[int, float, float]]) -> None:
        if not rcs_records:
            return
        try:
            with open(self.rcs_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Patch", "RCS_Depth",
                    "XEB_RCS", "HOG_RCS",
                ])
                for patch_id, xeb, hog in rcs_records:
                    w.writerow({
                        "Step": step, "Anneal_Percent": anneal,
                        "Patch": patch_id, "RCS_Depth": RCS_DEPTH,
                        "XEB_RCS": xeb, "HOG_RCS": hog,
                    })
        except Exception as e:
            print(f"[CSV] RCS log error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_face: float, target_J: float,
            target_hx: float, target_hz: float,
            measure_every: int = 1,
            effective_shots: float = 512.0) -> None:

        if total_steps < 1:  raise ValueError("total_steps must be >= 1")
        if measure_every < 1: raise ValueError("measure_every must be >= 1")

        rcs_cfg = {
            "enabled":        RCS_VALIDATION_ENABLED,
            "depth":          RCS_DEPTH,
            "shots":          RCS_SHOTS,
            "validate_every": RCS_VALIDATE_EVERY,
            "probe_patches":  RCS_PROBE_PATCHES,
            "msb_first":      RCS_MSB_FIRST,
            "target_hx":      target_hx,
            "target_hz":      target_hz,
            "master_seed":    self.master_seed,
        }

        n_probe = TOTAL_PATCHES if RCS_PROBE_PATCHES is None else len(RCS_PROBE_PATCHES)
        print(f"[Engine] {TOTAL_PATCHES} patches, {TOTAL_PATCHES * QUBITS_PER_PATCH} qubits, "
              f"{GPUS_AVAILABLE} GPUs, {total_steps} steps")
        print(f"[Master] Staggered init — expect ~{TOTAL_WORKERS * 10}s before first output.",
              flush=True)
        if RCS_VALIDATION_ENABLED:
            print(f"[Engine] RCS ON: depth={RCS_DEPTH}, shots={RCS_SHOTS}, "
                  f"{n_probe} probe patches, every {RCS_VALIDATE_EVERY} measure steps")
            print(f"[Engine] XEB interpretation: 0=uniform, 1=Porter-Thomas, >1=concentrated")

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]
        workers, pipes = [], []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=gpu_worker_process,
                args=(rank, self.worker_assignments[rank], child_conn,
                      dt, total_steps, initial_hx, target_J,
                      target_hx, target_hz, measure_every, rcs_cfg),
            )
            proc.start()
            child_conn.close()
            workers.append(proc)
            pipes.append(parent_conn)

        try:
            for t in range(total_steps):
                s              = t / max(1, total_steps - 1)
                current_g_face = s * target_g_face
                is_measure     = (t % measure_every == 0) or (t == total_steps - 1)
                if not is_measure:
                    continue

                t0 = time.perf_counter()

                # --- GATHER ---
                patch_full_states: Dict[int, dict] = {}
                bulk_energy = 0.0
                max_lat_trotter = max_lat_tomo = max_lat_rcs = 0.0
                min_fidelity = 1.0
                rcs_records: List[Tuple[int, float, float]] = []

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for patch_id, payload in data.items():
                        patch_full_states[patch_id] = payload["state"]
                        bulk_energy     += payload["meanfield_bulk_energy"]
                        max_lat_trotter  = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo     = max(max_lat_tomo,    payload["lat_tomo_ms"])
                        max_lat_rcs      = max(max_lat_rcs,     payload.get("lat_rcs_ms", 0.0))
                        min_fidelity     = min(min_fidelity,    payload.get("unitary_fidelity", 1.0))
                        if payload.get("rcs_xeb") is not None:
                            rcs_records.append((patch_id, payload["rcs_xeb"], payload["rcs_hog"]))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"IPC gather incomplete: got {len(patch_full_states)}/{TOTAL_PATCHES}")

                # --- BUILD PROFILES ---
                step_state   = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles: Dict[int, dict] = {}
                bq = self._bq_arr

                for patch_id, state in patch_full_states.items():
                    step_state[patch_id, :, 0] = state["X"]
                    step_state[patch_id, :, 1] = state["Y"]
                    step_state[patch_id, :, 2] = state["Z"]
                    patch_profiles[patch_id] = {
                        "means": {
                            "X": state["X"][bq],
                            "Y": state["Y"][bq],
                            "Z": state["Z"][bq],
                        },
                        "vars": {
                            "X": np.clip(1.0 - state["X"][bq]**2, 0.0, 1.0),
                            "Y": np.clip(1.0 - state["Y"][bq]**2, 0.0, 1.0),
                            "Z": np.clip(1.0 - state["Z"][bq]**2, 0.0, 1.0),
                        },
                    }

                self.lattice_history.append(step_state.copy())
                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Failed: {e}", file=sys.stderr)

                # --- BOUNDARY KICKS & ENERGY ---
                next_kick_payloads = {pid: {} for pid in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0
                scale = np.sqrt(dt / effective_shots)
                n_b   = len(self.all_boundary_qubits)
                stochastic_noise: Dict[int, dict] = {}

                for patch_id in range(TOTAL_PATCHES):
                    prof  = patch_profiles[patch_id]
                    rng_p = np.random.default_rng([self.master_seed, t, patch_id])
                    xn = rng_p.normal(0, 1, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = rng_p.normal(0, 1, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = rng_p.normal(0, 1, n_b) * np.sqrt(prof["vars"]["Z"]) * scale
                    stochastic_noise[patch_id] = {
                        q: (xn[i], yn[i], zn[i])
                        for i, q in enumerate(self.all_boundary_qubits)
                    }

                for patch_id_1, (x1, y1, z1) in self.patch_coords.items():
                    neighbors = {
                        "+X": (x1+1, y1,   z1  ), "-X": (x1-1, y1,   z1  ),
                        "+Y": (x1,   y1+1, z1  ), "-Y": (x1,   y1-1, z1  ),
                        "+Z": (x1,   y1,   z1+1), "-Z": (x1,   y1,   z1-1),
                    }
                    for dir1, coord2 in neighbors.items():
                        patch_id_2 = self.coord_to_patch.get(coord2)
                        if patch_id_2 is None or patch_id_1 >= patch_id_2: continue

                        dir2    = dir1.replace("+","temp").replace("-","+").replace("temp","-")
                        face1_q = self.boundaries[dir1]
                        face2_q = self.boundaries[dir2]
                        prof1, noise1 = patch_profiles[patch_id_1], stochastic_noise[patch_id_1]
                        prof2, noise2 = patch_profiles[patch_id_2], stochastic_noise[patch_id_2]

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
                            k = next_kick_payloads[patch_id_1].get(q1f, (0., 0., 0.))
                            next_kick_payloads[patch_id_1][q1f] = (
                                k[0] + current_g_face * ax2,
                                k[1] + current_g_face * ay2,
                                k[2] + current_g_face * az2,
                            )
                        for q2f in face2_q:
                            k = next_kick_payloads[patch_id_2].get(q2f, (0., 0., 0.))
                            next_kick_payloads[patch_id_2][q2f] = (
                                k[0] + current_g_face * ax1,
                                k[1] + current_g_face * ay1,
                                k[2] + current_g_face * az1,
                            )

                total_energy = bulk_energy + macroscopic_boundary_energy
                status = (
                    f"Step {t:03d} | E: {total_energy:+.4f} | "
                    f"Lat(Trot/Tomo/RCS): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}/{max_lat_rcs:6.1f}ms | "
                    f"Fid: {min_fidelity:.5f}"
                )
                if rcs_records:
                    mean_xeb = float(np.mean([r[1] for r in rcs_records]))
                    mean_hog = float(np.mean([r[2] for r in rcs_records]))
                    status += f" | XEB(RCS): {mean_xeb:+.4f} | HOG(RCS): {mean_hog:.3f}"
                status += f" | {time.perf_counter() - t0:.2f}s"
                print(status)

                self._log_energy(t, s * 100, bulk_energy,
                                 macroscopic_boundary_energy, total_energy, min_fidelity)
                self._log_profiles(t, patch_profiles)
                self._log_rcs(t, s * 100, rcs_records)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    pipes[i].send({
                        pid: next_kick_payloads[pid]
                        for pid in self.worker_assignments[w_rank]
                    })

        finally:
            for conn in pipes:
                try: conn.close()
                except Exception: pass
            if self.lattice_history:
                try:
                    np.save(self.state_dump, np.array(self.lattice_history))
                    print(f"\n[Master] State history saved to {self.state_dump}")
                except Exception as e:
                    print(f"\n[Master] State save failed: {e}", file=sys.stderr)
            for proc in workers:
                proc.join(timeout=15)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
                    if proc.is_alive():
                        try: proc.kill()
                        except Exception: pass


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
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
        )
    except KeyboardInterrupt:
        pass
