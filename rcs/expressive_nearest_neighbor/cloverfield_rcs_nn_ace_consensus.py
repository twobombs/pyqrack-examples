# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with Statistical Variance Injection
# + Integrated ACE Cross-Validation (from Dan Strano's fc_ace.py NN-RCS harness)
#
# REVISION 89.27 - OBSERVABLE DRIFT (NON-DESTRUCTIVE TELEMETRY)
#
# BUGFIXES (Rev 89.27):
# - C++ METADATA CORRUPTION (CRITICAL): `in_ket()` parsing a 1.07GB Python list takes 
#   12+ minutes and permanently corrupts the ACE Schmidt-decomposition tree metadata, 
#   causing fatal `Apply2x2` bounds-check crashes on subsequent Trotter steps. 
# - ARCHITECTURAL PIVOT: Abandoned mid-run destructive sampling and state vector 
#   reloads entirely. Continuous validation is now performed by calculating the 
#   Root Mean Square Error (RMSE) of the local Pauli observables (Z, X, Y) between 
#   the ACE replicas and the exact simulator. 
# - FINAL STEP XEB: Actual destructive `measure_shots` sampling and `prob_perm` XEB 
#   scoring is reserved exclusively for the final Trotter step (t == 99), providing 
#   the golden RCS metric without ruining the mid-run state.
#
# BUGFIXES (Rev 89.26 & older):
# - 6-GPU TOPOLOGY: Uniform saturation of 6x AMD Radeon Pro V340s.
# - ALLOCATION STORM: Staggered worker startup by `rank * 10.0` seconds to prevent
#   NUMA memory bus deadlock.

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

# 6-GPU Symmetrical Topography
GPUS_AVAILABLE = 6
WORKERS_PER_GPU = 1  # 6 workers total (1 per AMD Radeon Pro V340)
TOTAL_WORKERS = GPUS_AVAILABLE * WORKERS_PER_GPU

# --- ACE CROSS-VALIDATION CONFIGURATION (ported from fc_ace.py) ---
ACE_VALIDATION_ENABLED = True
ACE_N_INST = 2                                   # ACE replicas per probed patch
ACE_SDRP = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0    # fc_ace.py default (~0.1464466)
ACE_SHOTS = 256                                  # samples pooled across replicas per validation
ACE_VALIDATE_EVERY = 5                           # in units of *measure* steps
ACE_PROBE_PATCHES = [17]                         # Routes exclusively to Worker 5
ACE_MSB_FIRST = False                            # Toggle based on specific PyQrack build bit-ordering

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


def calc_sparse_stats(ideal_probs_of_samples: np.ndarray, width: int) -> Tuple[float, float]:
    n_pow = float(1 << width)
    if ideal_probs_of_samples.size == 0:
        return 0.0, 0.0
    xeb = n_pow * float(np.mean(ideal_probs_of_samples)) - 1.0
    hog = float(np.mean(ideal_probs_of_samples > (math.log(2.0) / n_pow)))
    return xeb, hog

# =====================================================================
# WORKER PROCESS LOGIC
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
    ace_cfg: Dict[str, Any]
) -> None:
    # Stagger worker startup to prevent NUMA memory bus contention 
    # during simultaneous statevector allocations across 6 GPUs.
    import time as _t
    stagger_time = rank * 10.0
    if stagger_time > 0:
        _t.sleep(stagger_time)
    print(f"[Worker {rank}] Awakening after {stagger_time}s stagger... allocating statevectors.", flush=True)

    os.environ["PYQRACK_SHARED_LIB_PATH"] = "/usr/local/lib/qrack/libqrack_pinvoke.so"
    os.environ["OCL_ICD_PLATFORM_SORT"] = "none"

    # Map multiple ranks to the same physical GPU device index
    physical_gpu_index = rank // WORKERS_PER_GPU
    os.environ["QRACK_OCL_DEFAULT_DEVICE"] = str(physical_gpu_index)

    # Bind QPager to the assigned device to enable driver-level PCIe paging
    os.environ["QRACK_QPAGER_DEVICES"] = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"] = str(physical_gpu_index)

    # Strict VRAM caps based on physical 8GB per AMD Pro V340
    alloc_mb = 8000 // WORKERS_PER_GPU
    os.environ["QRACK_MAX_ALLOC_MB"] = str(alloc_mb)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    import pyqrack
    from pyqrack import QrackSimulator

    sims = {}
    ace_sims: Dict[int, List[Any]] = {}
    ace_rngs: Dict[int, List[random.Random]] = {}

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

        def trotter_step_body(sim, num_qubits, edge_list, J, hx, hz, dt_local):
            dt_half = dt_local / 2.0
            theta_x  = -2.0 * hx * dt_half
            theta_z  = -2.0 * hz * dt_local
            theta_zz = -J * dt_local

            for q in range(num_qubits): apply_rx(sim, theta_x, q)
            for q in range(num_qubits): apply_rz(sim, theta_z, q)
            for q1, q2 in edge_list: apply_zz(sim, theta_zz, q1, q2)
            for q in range(num_qubits): apply_rx(sim, theta_x, q)

        def z_means(sim, qubits):
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])

        def x_means(sim, qubits):
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])

        def y_means(sim, qubits):
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])

        def zz_means_meanfield(z_exp, edges):
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks, time_delta):
            if not kicks: return
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                coef = -2.0 * time_delta
                
                theta_x = kx * coef
                theta_y = ky * coef
                theta_z = kz * coef

                if abs(theta_x) > 1e-12: apply_rx(sim, theta_x, q)
                if abs(theta_y) > 1e-12: apply_ry(sim, theta_y, q)
                if abs(theta_z) > 1e-12: apply_rz(sim, theta_z, q)

        intra_edges, boundaries = generate_27q_lattice_subvolume()
        
        # Hoisted out of the measurement loop
        all_q = list(range(QUBITS_PER_PATCH))

        # --- ACE availability + probe set ---
        ace_enabled = bool(ace_cfg.get("enabled", False)) and ace_cfg.get("n_inst", 0) > 0
        if ace_cfg.get("probe_patches") is None:
            probe_set = set(assigned_patches)
        else:
            probe_set = set(ace_cfg["probe_patches"]) & set(assigned_patches)
        _warned_ace = False

        for patch_id in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): apply_h(sim, q)
            sims[patch_id] = sim

            # --- VRAM PAGING SMOKE TEST ---
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM/PCIe paging allocation failed on patch {patch_id}. Driver error: {e}")

            # --- ACE REPLICA SETUP ---
            if ace_enabled and patch_id in probe_set:
                replicas, rngs = [], []
                try:
                    for inst in range(ace_cfg["n_inst"]):
                        a = QrackSimulator(
                            qubit_count=QUBITS_PER_PATCH,
                            is_binary_decision_tree=False,
                            is_stabilizer_hybrid=False,
                            is_gpu=True,
                        )
                        if ace_cfg["sdrp"] > 0.0:
                            a.set_sdrp(ace_cfg["sdrp"])
                        a.set_ace_max_qb((QUBITS_PER_PATCH + 1) >> 1)
                        for q in range(QUBITS_PER_PATCH): apply_h(a, q)
                        replicas.append(a)
                        rngs.append(random.Random((rank << 24) ^ (patch_id << 8) ^ inst))
                except AttributeError as e:
                    if not _warned_ace:
                        print(f"[Worker {rank}] Warning: ACE bindings unavailable ({e}). "
                              f"Upgrade PyQrack for set_ace_max_qb/set_sdrp. "
                              f"Disabling ACE cross-validation.", file=sys.stderr)
                        _warned_ace = True
                    ace_enabled = False
                    for a in replicas: del a
                else:
                    ace_sims[patch_id] = replicas
                    ace_rngs[patch_id] = rngs

        kick_payloads = {patch_id: {} for patch_id in assigned_patches}
        _warned_fidelity = False
        meas_count = 0
        
        # Loop invariant hoist
        time_delta = dt * measure_every

        for t in range(total_steps):
            s = t / max(1, (total_steps - 1))
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)
            
            do_validate = (
                ace_enabled and is_measure and
                ((meas_count % max(1, ace_cfg["validate_every"]) == 0) or (t == total_steps - 1))
            )

            patch_data_to_master = {}

            for patch_id in assigned_patches:
                sim = sims[patch_id]
                
                # NOTE ON BOUNDARY KICK TIMING:
                # Kicks are computed by the master AFTER the gather step and sent to workers for 
                # the NEXT measure window. This implies an inherent 1-window lag (i.e. 'measure_every' 
                # Trotter steps). The boundary field is treated as constant over this window, hence 
                # scaled by `time_delta = dt * measure_every` as a lump sum.
                if is_measure and kick_payloads[patch_id]:
                    apply_kicks(sim, kick_payloads[patch_id], time_delta)
                    
                    # ACE replicas track the identical boundary trajectory
                    for a in ace_sims.get(patch_id, []):
                        apply_kicks(a, kick_payloads[patch_id], time_delta)

                t_start_trotter = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                t_lat_trotter = time.perf_counter() - t_start_trotter

                # --- ACE REPLICA EVOLUTION ---
                t_lat_ace_evol = 0.0
                if patch_id in ace_sims:
                    t_start_ace_evol = time.perf_counter()
                    for a, rng_a in zip(ace_sims[patch_id], ace_rngs[patch_id]):
                        shuffled_edges = list(intra_edges)
                        rng_a.shuffle(shuffled_edges)
                        trotter_step_body(a, QUBITS_PER_PATCH, shuffled_edges,
                                          current_J, current_hx, current_hz, dt)
                    t_lat_ace_evol = time.perf_counter() - t_start_ace_evol

                t_lat_tomo = 0.0
                t_lat_ace_val = 0.0
                if is_measure:
                    t_start_tomo = time.perf_counter()
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

                    # --- ACE SPARSE XEB/HOG & OBSERVABLE DRIFT VALIDATION ---
                    ace_xeb, ace_hog = None, None
                    ace_z_rmse, ace_x_rmse, ace_y_rmse = None, None, None
                    
                    if do_validate and patch_id in ace_sims:
                        t_start_ace_val = time.perf_counter()
                        try:
                            num_replicas = len(ace_sims[patch_id])
                            
                            # 1. CONTINUOUS OBSERVABLE DRIFT (Non-Destructive)
                            # To bypass destructive measurement and heavy `in_ket` Python list bottlenecks, 
                            # we query the 1-local Pauli expectations directly and track the Root Mean Square 
                            # Error (RMSE) of the ACE state's divergence from the exact state. 
                            z_ace_avg = np.zeros(QUBITS_PER_PATCH)
                            x_ace_avg = np.zeros(QUBITS_PER_PATCH)
                            y_ace_avg = np.zeros(QUBITS_PER_PATCH)
                            
                            for a in ace_sims[patch_id]:
                                z_ace_avg += z_means(a, all_q)
                                x_ace_avg += x_means(a, all_q)
                                y_ace_avg += y_means(a, all_q)
                                
                            z_ace_avg /= num_replicas
                            x_ace_avg /= num_replicas
                            y_ace_avg /= num_replicas
                            
                            ace_z_rmse = float(np.sqrt(np.mean((state["Z"] - z_ace_avg)**2)))
                            ace_x_rmse = float(np.sqrt(np.mean((state["X"] - x_ace_avg)**2)))
                            ace_y_rmse = float(np.sqrt(np.mean((state["Y"] - y_ace_avg)**2)))
                            
                            # 2. FINAL STEP DESTRUCTIVE XEB
                            # Perform the true overlap integral sampling exclusively on the final Trotter 
                            # step, providing the golden RCS metric right before safe shutdown.
                            if t == total_steps - 1:
                                shots_per = max(1, ace_cfg["shots"] // num_replicas)
                                samples = []
                                
                                for a in ace_sims[patch_id]:
                                    samples.extend(a.measure_shots(all_q, shots_per))
                                    
                                ideal_p_list = []
                                for o in samples:
                                    if ace_cfg["msb_first"]:
                                        bitmask = [bool((int(o) >> (QUBITS_PER_PATCH - 1 - b)) & 1) for b in range(QUBITS_PER_PATCH)]
                                    else:
                                        bitmask = [bool((int(o) >> b) & 1) for b in range(QUBITS_PER_PATCH)]
                                    
                                    p_val = sim.prob_perm(all_q, bitmask)
                                    ideal_p_list.append(p_val)
                                    
                                ideal_p = np.array(ideal_p_list, dtype=np.float64)
                                ace_xeb, ace_hog = calc_sparse_stats(ideal_p, QUBITS_PER_PATCH)
                            
                        except (AttributeError, TypeError, RuntimeError) as e:
                            if not _warned_ace:
                                print(f"[Worker {rank}] Warning: sparse validation bindings "
                                      f"unavailable, signature mismatched, or allocation failed ({e}). "
                                      f"Disabling ACE cross-validation.",
                                      file=sys.stderr)
                                _warned_ace = True
                            ace_enabled = False
                            for pp in list(ace_sims.keys()):
                                for a in ace_sims.pop(pp):
                                    del a
                        t_lat_ace_val = time.perf_counter() - t_start_ace_val

                    patch_data_to_master[patch_id] = {
                        "state": state,
                        "meanfield_bulk_energy": bulk_e,
                        "lat_trotter_ms": t_lat_trotter * 1000.0,
                        "lat_tomo_ms": t_lat_tomo * 1000.0,
                        "lat_ace_evol_ms": t_lat_ace_evol * 1000.0,
                        "lat_ace_val_ms": t_lat_ace_val * 1000.0,
                        "unitary_fidelity": fidelity,
                        "ace_xeb": ace_xeb,
                        "ace_hog": ace_hog,
                        "ace_z_rmse": ace_z_rmse,
                        "ace_x_rmse": ace_x_rmse,
                        "ace_y_rmse": ace_y_rmse,
                    }

            if is_measure:
                meas_count += 1
                conn.send(patch_data_to_master)
                kick_payloads = conn.recv()

    finally:
        for patch_id in list(ace_sims.keys()):
            for a in ace_sims.pop(patch_id):
                del a
        ace_sims.clear()
        ace_rngs.clear()
        for patch_id in list(sims.keys()):
            _s = sims.pop(patch_id)
            del _s
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

        self.lattice_history: List[np.ndarray] = []
        self.energy_csv       = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv     = "boundary_profiles_multi.csv"
        self.ace_csv          = "ace_validation_multi.csv"
        self.state_dump_file  = "macroscopic_lattice_states.npy"
        self.config_file      = "lattice_config.json"

        self._init_files()

        # Symmetric Load Balancing across 6 identical GPUs
        self.worker_assignments: List[List[int]] = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                           "num_patches": TOTAL_PATCHES,
                           "qubits_per_patch": QUBITS_PER_PATCH,
                           "ace_validation": ACE_VALIDATION_ENABLED,
                           "ace_n_inst": ACE_N_INST,
                           "ace_sdrp": ACE_SDRP,
                           "ace_shots": ACE_SHOTS,
                           "msb_first": ACE_MSB_FIRST}, f)
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
            with open(self.ace_csv, mode='w', newline='') as f:
                csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "XEB_ACE", "HOG_ACE", "Z_RMSE", "X_RMSE", "Y_RMSE"
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: Setup configuration write failed: {e}", file=sys.stderr)

    def _log_csvs(self, step: int, anneal: float, bulk: float, bound: float, total: float, min_fidelity: float, patch_profiles: dict) -> None:
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
                for patch_id, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]] for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]] for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]] for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": patch_id, "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception as e:
            print(f"[CSV] Warning: Log write failed: {e}", file=sys.stderr)

    def _log_ace_csv(self, step: int, ace_records: List[Tuple]) -> None:
        if not ace_records:
            return
        try:
            with open(self.ace_csv, mode='a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["Step", "Patch", "XEB_ACE", "HOG_ACE", "Z_RMSE", "X_RMSE", "Y_RMSE"])
                for patch_id, xeb, hog, z_rmse, x_rmse, y_rmse in ace_records:
                    w.writerow({
                        "Step": step, 
                        "Patch": patch_id,
                        "XEB_ACE": xeb if xeb is not None else "", 
                        "HOG_ACE": hog if hog is not None else "",
                        "Z_RMSE": z_rmse if z_rmse is not None else "",
                        "X_RMSE": x_rmse if x_rmse is not None else "",
                        "Y_RMSE": y_rmse if y_rmse is not None else ""
                    })
        except Exception as e:
            print(f"[CSV] Warning: ACE log write failed: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float, target_g_face: float,
            target_J: float, target_hx: float, target_hz: float,
            measure_every: int = 1, effective_shots: float = 512.0) -> None:

        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if measure_every < 1:
            raise ValueError("measure_every must be a positive integer")

        ace_cfg = {
            "enabled": ACE_VALIDATION_ENABLED,
            "n_inst": ACE_N_INST,
            "sdrp": ACE_SDRP,
            "shots": ACE_SHOTS,
            "validate_every": ACE_VALIDATE_EVERY,
            "probe_patches": ACE_PROBE_PATCHES,
            "msb_first": ACE_MSB_FIRST,
        }

        total_qubits = TOTAL_PATCHES * QUBITS_PER_PATCH
        print(f"[Engine] {TOTAL_PATCHES} patches, {total_qubits} qubits, {GPUS_AVAILABLE} GPUs ({TOTAL_WORKERS} workers total), {total_steps} steps")
        
        # Add a master notice about the expected boot delay so the user doesn't assume a hang.
        print(f"[Master] Waiting for staggered worker initialization (expect ~{TOTAL_WORKERS * 10}s delay)...", flush=True)
        
        if ACE_VALIDATION_ENABLED:
            n_probe = TOTAL_PATCHES if ACE_PROBE_PATCHES is None else len(ACE_PROBE_PATCHES)
            print(f"[Engine] ACE cross-validation ON: {ACE_N_INST} replicas x {n_probe} patches, "
                  f"sdrp={ACE_SDRP:.7f}, {ACE_SHOTS} shots, every {ACE_VALIDATE_EVERY} measure steps "
                  f"(Note: ACE_val latency will read 0.0 except on validation steps)")
            
            # Guard against depletion
            if ACE_SHOTS // max(1, ACE_N_INST) < 1:
                print(f"[Engine] Warning: ACE_SHOTS ({ACE_SHOTS}) is less than ACE_N_INST ({ACE_N_INST}). Sampling budget is depleted per replica.", file=sys.stderr)

        active_ranks = [r for r in range(TOTAL_WORKERS) if self.worker_assignments[r]]

        workers = []
        pipes   = []

        for rank in active_ranks:
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=gpu_worker_process,
                args=(rank, self.worker_assignments[rank], child_conn,
                      dt, total_steps, initial_hx, target_J, target_hx, target_hz,
                      measure_every, ace_cfg)
            )
            proc.start()
            child_conn.close()
            workers.append(proc)
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
                bulk_energy = 0.0
                max_lat_trotter = 0.0
                max_lat_tomo = 0.0
                max_lat_ace_evol = 0.0
                max_lat_ace_val = 0.0
                min_fidelity = 1.0
                ace_records = []

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for patch_id, payload in data.items():
                        patch_full_states[patch_id] = payload["state"]
                        bulk_energy += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo = max(max_lat_tomo, payload["lat_tomo_ms"])
                        max_lat_ace_evol = max(max_lat_ace_evol, payload.get("lat_ace_evol_ms", 0.0))
                        max_lat_ace_val = max(max_lat_ace_val, payload.get("lat_ace_val_ms", 0.0))
                        min_fidelity = min(min_fidelity, payload.get("unitary_fidelity", 1.0))
                        
                        if payload.get("ace_z_rmse") is not None or payload.get("ace_xeb") is not None:
                            ace_records.append((
                                patch_id, 
                                payload.get("ace_xeb"), 
                                payload.get("ace_hog"),
                                payload.get("ace_z_rmse"),
                                payload.get("ace_x_rmse"),
                                payload.get("ace_y_rmse")
                            ))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"Fatal: IPC gather incomplete. "
                        f"Expected {TOTAL_PATCHES} patches, got {len(patch_full_states)}."
                    )

                # --- BUILD PROFILES ---
                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles = {}
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
                        }
                    }

                self.lattice_history.append(step_state.copy())

                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump_file, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Warning: Failed to save: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & BOUNDARY ENERGY ---
                next_kick_payloads = {patch_id: {} for patch_id in range(TOTAL_PATCHES)}
                macroscopic_boundary_energy = 0.0

                scale = np.sqrt(dt / effective_shots)
                stochastic_noise = {}
                n_b = len(self.all_boundary_qubits)

                for patch_id in range(TOTAL_PATCHES):
                    prof = patch_profiles[patch_id]
                    rng_p = np.random.default_rng([self.master_seed, t, patch_id])

                    xn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["X"]) * scale
                    yn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Y"]) * scale
                    zn = rng_p.normal(0.0, 1.0, n_b) * np.sqrt(prof["vars"]["Z"]) * scale

                    stochastic_noise[patch_id] = {
                        q: (xn[i], yn[i], zn[i])
                        for i, q in enumerate(self.all_boundary_qubits)
                    }

                for patch_id_1, coord1 in self.patch_coords.items():
                    x1, y1, z1 = coord1
                    neighbors = {
                        "+X": (x1+1, y1,   z1  ), "-X": (x1-1, y1,   z1  ),
                        "+Y": (x1,   y1+1, z1  ), "-Y": (x1,   y1-1, z1  ),
                        "+Z": (x1,   y1,   z1+1), "-Z": (x1,   y1,   z1-1),
                    }

                    for dir1, coord2 in neighbors.items():
                        patch_id_2 = self.coord_to_patch.get(coord2)
                        if patch_id_2 is None or patch_id_1 >= patch_id_2: continue

                        dir2    = dir1.replace("+", "temp").replace("-", "+").replace("temp", "-")
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
                status = (f"Step {t:03d} | E: {total_energy:+.4f} | "
                          f"Lat(Trot/Tomo/ACE_ev/ACE_val): {max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}/{max_lat_ace_evol:5.1f}/{max_lat_ace_val:5.1f}ms | "
                          f"Fid: {min_fidelity:.5f}")
                          
                if ace_records:
                    valid_xebs = [r[1] for r in ace_records if r[1] is not None]
                    valid_hogs = [r[2] for r in ace_records if r[2] is not None]
                    valid_z_rmse = [r[3] for r in ace_records if r[3] is not None]
                    
                    if valid_z_rmse:
                        mean_z_rmse = float(np.mean(valid_z_rmse))
                        status += f" | Z-RMSE(ACE): {mean_z_rmse:.4f}"
                        
                    if valid_xebs:
                        mean_xeb = float(np.mean(valid_xebs))
                        mean_hog = float(np.mean(valid_hogs))
                        status += f" | XEB(ACE): {mean_xeb:+.4f} | HOG*(ACE): {mean_hog:.3f}"
                        
                status += f" | {time.perf_counter() - t0:.2f}s"
                print(status)

                self._log_csvs(t, s * 100, bulk_energy,
                               macroscopic_boundary_energy, total_energy, min_fidelity, patch_profiles)
                self._log_ace_csv(t, ace_records)

                # --- SCATTER ---
                for i, w_rank in enumerate(active_ranks):
                    worker_payload = {patch_id: next_kick_payloads[patch_id]
                                      for patch_id in self.worker_assignments[w_rank]}
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
            effective_shots=512.0
        )
    except KeyboardInterrupt:
        pass
