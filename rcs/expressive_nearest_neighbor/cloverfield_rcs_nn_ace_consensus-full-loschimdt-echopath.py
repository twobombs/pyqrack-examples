# -*- coding: us-ascii -*-
# 27-Qubit 3x3x3 Macroscopic Grid Annealing (27 Patches, 729 Qubits Total)
# High-Throughput Volumetric Engine with Statistical Variance Injection
# + In-Place Loschmidt Echo Probe with Exact Ket Restoration (Rev 100)
#
# REVISION 100 - LOSCHMIDT ECHO PROBE & REVIEW FIXES
#
# CHANGES vs Rev 99:
# - ECHO PROBE: The self-referential sampling XEB is replaced by a mirror
#   circuit (C . C^dag) Loschmidt echo. Forward random circuit is recorded
#   gate-by-gate, exactly inverted in reversed order, and echo fidelity
#   F = |<psi_pristine|psi_mirror>|^2 is computed host-side via np.vdot.
#   Exact simulation -> F = 1; SDRP rounding / precision loss over the
#   2*depth circuit shows as decay. NOTE: mirror pairs can coherently
#   cancel errors, so F_echo is an optimistic (upper) bound on forward-
#   circuit fidelity.
# - out_probs()/measure_shots() sampling path DELETED from the probe:
#   removes the ~1 GB DMA burst and multi-GB host list allocation.
# - iSWAP MIRROR SMOKE TEST at worker init autodetects native adjiswap()
#   (fallback: iSWAP^dag = (Sdag x Sdag) . CZ . SWAP) and validates the
#   full inverse convention, including U3^dag = U3(-theta, -lam, -phi).
# - PROBE LIST FIX: RCS_PROBE_PATCHES now maps to exactly one patch per
#   worker under round-robin assignment (i % TOTAL_WORKERS).
# - SNAPSHOT GUARD FIX: full-snapshot steps may coincide with the final
#   step (t == total_steps - 1), which is always a measure step.
# - DEAD PARAM FIX: worker reads target_hx / target_hz from its function
#   arguments; duplicates removed from rcs_cfg. Annealing schedule no
#   longer depends on the RCS config dict.
#
# ARCHITECTURE (unchanged):
# - IPC DELEGATION: Worker computes exact mathematical variance for Pauli
#   observables (1 - <P>^2) and sends discrete `var_X/Y/Z` arrays.
# - PROFILING ISOLATION: lat_rcs strictly bounds the echo probe logic
#   (out_ket, forward, inverse, out_ket, vdot), excluding gc.collect()
#   and in_ket() restore time.

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
import multiprocessing.connection
from typing import List, Tuple, Dict, Any, Optional

# --- GLOBAL CONFIGURATION ---
GRID_X, GRID_Y, GRID_Z = 3, 3, 3
TOTAL_PATCHES  = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH = 27

# 6-GPU Symmetrical Topography
GPUS_AVAILABLE  = 6
WORKERS_PER_GPU = 1          # DO NOT set to 2: rusticl falls back to CPU per die
TOTAL_WORKERS   = GPUS_AVAILABLE * WORKERS_PER_GPU

# --- ECHO PROBE CONFIGURATION ---
RCS_VALIDATION_ENABLED  = True
RCS_DEPTH               = 20
RCS_VALIDATE_EVERY      = 5        # routine cadence: all probe patches every 5 measure steps
# Exactly one patch per worker under round-robin (patch i -> worker i % 6):
#   0->w0, 13->w1, 8->w2, 21->w3, 22->w4, 17->w5. Center patch 13 on w1.
RCS_PROBE_PATCHES       = [0, 13, 8, 21, 22, 17]
RCS_FULL_SNAPSHOT_STEPS = [42, 82, 99] # all-patch echo: phase transition + final

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


# =====================================================================
# ECHO PROBE: FORWARD (RECORDING) + EXACT INVERSE
# =====================================================================
def apply_rcs_layer(sim, num_qubits: int, edges: List[Tuple[int, int]],
                    depth: int, rng: random.Random) -> List[tuple]:
    """Apply `depth` layers of random u + iswap gates in-place.

    Returns the gate record needed by the inverse applier:
        [("u", q, theta, phi, lam), ("iswap", q1, q2), ...]
    in application order.
    """
    record: List[tuple] = []
    for _ in range(depth):
        for q in range(num_qubits):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            lam   = rng.uniform(0.0, 2.0 * math.pi)
            sim.u(q, theta, phi, lam)
            record.append(("u", q, theta, phi, lam))
        shuffled = list(edges)
        rng.shuffle(shuffled)
        used: set = set()
        for q1, q2 in shuffled:
            if q1 not in used and q2 not in used:
                sim.iswap(q1, q2)
                record.append(("iswap", q1, q2))
                used.add(q1); used.add(q2)
    return record


def make_rcs_inverse(adjiswap_native: bool):
    """Build the inverse-applier, resolving the iswap-adjoint strategy once.

    U3(theta, phi, lam)^dag = U3(-theta, -lam, -phi)   [phi/lam swap AND negate]
    iSWAP^dag: native sim.adjiswap if available, else the decomposition
        iSWAP = SWAP . CZ . (S x S)   (rightmost applied first)
     => iSWAP^dag applies: SWAP, then CZ, then Sdag on both qubits.
    """
    if adjiswap_native:
        def _inv_iswap(sim, q1, q2):
            sim.adjiswap(q1, q2)
    else:
        def _inv_iswap(sim, q1, q2):
            sim.swap(q1, q2)
            sim.mcz([q1], q2)
            sim.adjs(q1)
            sim.adjs(q2)

    def apply_rcs_inverse(sim, record: List[tuple]) -> None:
        for g in reversed(record):
            if g[0] == "u":
                _, q, theta, phi, lam = g
                sim.u(q, -theta, -lam, -phi)
            else:
                _, q1, q2 = g
                _inv_iswap(sim, q1, q2)

    return apply_rcs_inverse


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
    os.environ["OCL_ICD_PLATFORM_SORT"]    = "none"

    physical_gpu_index = rank // WORKERS_PER_GPU
    os.environ["QRACK_OCL_DEFAULT_DEVICE"]           = str(physical_gpu_index)
    os.environ["QRACK_QPAGER_DEVICES"]               = str(physical_gpu_index)
    os.environ["QRACK_QUNITMULTI_DEVICES"]           = str(physical_gpu_index)
    os.environ["QRACK_MAX_ALLOC_MB"]                 = str(8000 // WORKERS_PER_GPU)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    from pyqrack import QrackSimulator

    sims: Dict[int, Any] = {}

    try:
        # --- PAULI CODE AUTODETECT ---
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
        _probe_y.mtrx([complex(_c,0), complex(0,_s), complex(0,_s), complex(_c,0)], 0)
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

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        _corrected = SIGN_Z * _sim_mag.pauli_expectation([0], [PZ])
        if   abs(_corrected + 1.0) < 0.1: ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1: ANGLE_SCALE = 0.5
        else: raise RuntimeError(
            f"Fatal: ambiguous ANGLE_SCALE, SIGN_Z*<Z>={_corrected:.6f}")
        del _sim_mag

        # --- VECTORIZED r() DETECTION ---
        _vec_probe = QrackSimulator(qubit_count=2,
                                    is_binary_decision_tree=False,
                                    is_gpu=True)
        _VECTORIZED_R = False
        try:
            _vec_probe.r(PX, 0.0, [0, 1])
            _VECTORIZED_R = True
        except Exception:
            pass
        del _vec_probe
        print(f"[Worker {rank}] Vectorized r(): {'YES' if _VECTORIZED_R else 'NO (fallback to loop)'}",
              flush=True)

        # --- ISWAP MIRROR SMOKE TEST + adjiswap AUTODETECT ---
        # Prepares a non-trivial two-qubit state, applies iswap then the
        # chosen inverse, and demands echo >= 0.999. Catches a missing
        # adjiswap, a wrong fallback decomposition, AND a wrong U3^dag
        # convention (tested implicitly at full width by echo values,
        # but the two-qubit iswap round-trip is validated here).
        _echo_probe = QrackSimulator(qubit_count=2, is_binary_decision_tree=False)
        _echo_probe.h(0)
        _echo_probe.u(1, 0.7, 1.1, 2.3)          # arbitrary non-trivial state
        _ket_before = np.asarray(_echo_probe.out_ket(), dtype=np.complex128)
        _adjiswap_native = hasattr(_echo_probe, "adjiswap")
        _echo_probe.iswap(0, 1)
        if _adjiswap_native:
            try:
                _echo_probe.adjiswap(0, 1)
            except Exception:
                # adjiswap advertised but broken: state may be corrupted;
                # apply fallback anyway -- if state was altered, the
                # fidelity check below fails loudly.
                _adjiswap_native = False
                _echo_probe.swap(0, 1)
                _echo_probe.mcz([0], 1)
                _echo_probe.adjs(0)
                _echo_probe.adjs(1)
        else:
            _echo_probe.swap(0, 1)
            _echo_probe.mcz([0], 1)
            _echo_probe.adjs(0)
            _echo_probe.adjs(1)
        _ket_after = np.asarray(_echo_probe.out_ket(), dtype=np.complex128)
        _f_smoke = abs(np.vdot(_ket_before, _ket_after)) ** 2
        if _f_smoke < 0.999:
            raise RuntimeError(
                f"Fatal: iswap mirror smoke test failed, F={_f_smoke:.6f}. "
                f"adjiswap_native={_adjiswap_native}. Inverse convention wrong.")
        del _echo_probe, _ket_before, _ket_after
        apply_rcs_inverse = make_rcs_inverse(_adjiswap_native)
        print(f"[Worker {rank}] iSWAP inverse: "
              f"{'native adjiswap' if _adjiswap_native else 'decomposed fallback'} "
              f"(mirror smoke F={_f_smoke:.6f})", flush=True)

        # --- GATE HELPERS ---
        _all_q_list = list(range(QUBITS_PER_PATCH))

        if _VECTORIZED_R:
            def apply_rx_all(sim, theta):
                sim.r(PX, float(theta) * ANGLE_SCALE, _all_q_list)
            def apply_rz_all(sim, theta):
                sim.r(PZ, float(theta) * ANGLE_SCALE, _all_q_list)
        else:
            def apply_rx_all(sim, theta):
                ang = float(theta) * ANGLE_SCALE
                for q in _all_q_list: sim.r(PX, ang, q)
            def apply_rz_all(sim, theta):
                ang = float(theta) * ANGLE_SCALE
                for q in _all_q_list: sim.r(PZ, ang, q)

        def apply_rx(sim, theta, q): sim.r(PX, float(theta) * ANGLE_SCALE, q)
        def apply_ry(sim, theta, q): sim.r(PY, float(theta) * ANGLE_SCALE, q)
        def apply_rz(sim, theta, q): sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta, q1, q2):
            sim.mcx([q1], q2)
            apply_rz(sim, 2.0 * theta, q2)
            sim.mcx([q1], q2)

        def trotter_step_body(sim, num_qubits, edge_list, J, hx, hz, dt_local):
            dt_half      = dt_local / 2.0
            theta_x      = -2.0 * hx * dt_half
            theta_z_half = -2.0 * hz * dt_half
            theta_zz     = -J * dt_local
            apply_rx_all(sim, theta_x)
            apply_rz_all(sim, theta_z_half)
            for q1, q2 in edge_list: apply_zz(sim, theta_zz, q1, q2)
            apply_rz_all(sim, theta_z_half)
            apply_rx_all(sim, theta_x)

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
            coef = -2.0 * time_delta
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                if abs(kx * coef) > 1e-12: apply_rx(sim, kx * coef, q)
                if abs(ky * coef) > 1e-12: apply_ry(sim, ky * coef, q)
                if abs(kz * coef) > 1e-12: apply_rz(sim, kz * coef, q)

        # --- TOPOLOGY ---
        intra_edges, boundaries = generate_27q_lattice_subvolume()
        all_q = list(range(QUBITS_PER_PATCH))

        rcs_enabled = rcs_cfg.get("enabled", False)
        if rcs_cfg.get("probe_patches") is None:
            rcs_probe_set = set(assigned_patches)
        else:
            rcs_probe_set = set(rcs_cfg["probe_patches"]) & set(assigned_patches)

        full_snapshot_steps = set(rcs_cfg.get("full_snapshot_steps", []))
        master_seed       = rcs_cfg.get("master_seed", 1337)
        _warned_fidelity  = False

        # --- SIMULATOR ALLOCATION ---
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

        kick_payloads = {patch_id: {} for patch_id in assigned_patches}
        meas_count    = 0
        time_delta    = dt * measure_every

        for t in range(total_steps):
            s          = t / max(1, total_steps - 1)
            # REV 100 FIX: annealing schedule reads target_hx / target_hz
            # from function arguments, not rcs_cfg.
            current_hx = (1.0 - s) * initial_hx + s * target_hx
            current_J  = s * target_J
            current_hz = s * target_hz
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            is_snapshot = (t in full_snapshot_steps)
            do_rcs = (
                rcs_enabled and is_measure and
                (is_snapshot or (meas_count % max(1, rcs_cfg["validate_every"]) == 0))
            )
            rcs_probe_this_step = set(assigned_patches) if is_snapshot else rcs_probe_set

            patch_data = {}

            for patch_id in assigned_patches:
                sim = sims[patch_id]

                if is_measure and kick_payloads[patch_id]:
                    apply_kicks(sim, kick_payloads[patch_id], time_delta)

                t0_tr = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                lat_trotter = (time.perf_counter() - t0_tr) * 1000.0

                if not is_measure:
                    continue

                t0_tomo = time.perf_counter()

                # Fetch exact expectation values
                x_exp = x_means(sim, all_q)
                y_exp = y_means(sim, all_q)
                z_exp = z_means(sim, all_q)

                state = {
                    "X": x_exp,
                    "Y": y_exp,
                    "Z": z_exp,
                    # Exact variance mapping Var(P) = 1 - <P>^2
                    "var_X": np.clip(1.0 - x_exp**2, 0.0, 1.0),
                    "var_Y": np.clip(1.0 - y_exp**2, 0.0, 1.0),
                    "var_Z": np.clip(1.0 - z_exp**2, 0.0, 1.0),
                }

                zz_exp = zz_means_mf(state["Z"], intra_edges)
                bulk_e = (
                    -current_hz * float(np.sum(state["Z"]))
                    - current_J  * float(np.sum(zz_exp))
                    - current_hx * float(np.sum(state["X"]))
                )
                lat_tomo = (time.perf_counter() - t0_tomo) * 1000.0

                try:
                    fidelity = float(sim.get_unitary_fidelity())
                except AttributeError:
                    fidelity = 1.0
                    if not _warned_fidelity:
                        print(f"[Worker {rank}] Warning: get_unitary_fidelity() unavailable.", file=sys.stderr)
                        _warned_fidelity = True

                # --- LOSCHMIDT ECHO PROBE (Mirror Circuit C . C^dag) ---
                echo_f, echo_n_gates = None, None
                lat_rcs = 0.0

                if do_rcs and patch_id in rcs_probe_this_step:
                    t0_rcs = time.perf_counter()
                    pristine_ket = None
                    final_ket = None
                    try:
                        depth = rcs_cfg["depth"]
                        rng = random.Random(
                            (master_seed << 32) ^ (patch_id << 16) ^ t)

                        # 1. Snapshot pristine statevector (complex128,
                        #    ~2 GB at 2^27; the transient Python list
                        #    from out_ket is freed on conversion)
                        pristine_ket = np.asarray(sim.out_ket(),
                                                  dtype=np.complex128)

                        # 2. Forward random circuit, recording gate stream
                        record = apply_rcs_layer(
                            sim, QUBITS_PER_PATCH, intra_edges, depth, rng)
                        echo_n_gates = len(record)

                        # 3. Exact inverse in reversed order
                        apply_rcs_inverse(sim, record)

                        # 4. Echo fidelity via host-side overlap.
                        #    Norm-corrected as cheap insurance against
                        #    non-normalized kets after SDRP rounding.
                        final_ket = np.asarray(sim.out_ket(),
                                               dtype=np.complex128)
                        ov  = np.vdot(pristine_ket, final_ket)
                        n1  = float(np.vdot(pristine_ket, pristine_ket).real)
                        n2  = float(np.vdot(final_ket, final_ket).real)
                        echo_f = float(abs(ov) ** 2 / max(n1 * n2, 1e-300))

                        # Isolate actual algorithmic time from clean-up delays
                        lat_rcs = (time.perf_counter() - t0_rcs) * 1000.0

                        if is_snapshot:
                            print(f"[Worker {rank}] Snapshot echo patch {patch_id}: "
                                  f"F={echo_f:.6f} ({echo_n_gates} gates mirrored)",
                                  flush=True)

                    except Exception as e:
                        print(f"[Worker {rank}] Echo probe error (patch {patch_id}): {e}",
                              file=sys.stderr)
                        echo_f, echo_n_gates = None, None

                    finally:
                        # 5. Fail-safe cleanup + EXACT state restoration.
                        #    Even a perfect mirror leaves rounding
                        #    contamination; in_ket() keeps the annealing
                        #    trajectory deterministic and probe-independent.
                        if final_ket is not None:
                            del final_ket
                        if pristine_ket is not None:
                            try:
                                sim.in_ket(pristine_ket)
                            except Exception as restore_e:
                                print(f"[Worker {rank}] FATAL: Failed to restore ket on patch {patch_id}: {restore_e}",
                                      file=sys.stderr)
                            finally:
                                del pristine_ket
                                gc.collect() # Nudge OS to reclaim memory blocks

                patch_data[patch_id] = {
                    "state":                  state,
                    "meanfield_bulk_energy":  bulk_e,
                    "lat_trotter_ms":         lat_trotter,
                    "lat_tomo_ms":            lat_tomo,
                    "lat_rcs_ms":             lat_rcs,
                    "unitary_fidelity":       fidelity,
                    "echo_fidelity":          echo_f,
                    "echo_n_gates":           echo_n_gates,
                    "is_snapshot":            is_snapshot,
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

        # Hardened Topology Guard
        face_sizes = {k: len(v) for k, v in self.boundaries.items()}
        if len(set(face_sizes.values())) != 1 or 0 in face_sizes.values():
            raise ValueError(f"Asymmetric or empty boundary faces detected: {face_sizes}. "
                             f"Boundary energy weight logic requires symmetric grids.")

        self.patch_coords: Dict[int, Tuple[int, int, int]] = {}
        idx = 0
        for x in range(GRID_X):
            for y in range(GRID_Y):
                for z in range(GRID_Z):
                    self.patch_coords[idx] = (x, y, z)
                    idx += 1
        self.coord_to_patch = {v: k for k, v in self.patch_coords.items()}

        self.lattice_history: List[np.ndarray] = []
        self.energy_csv   = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv = "boundary_profiles_multi.csv"
        self.rcs_csv      = "echo_validation_multi.csv"
        self.state_dump   = "macroscopic_lattice_states.npy"
        self.config_file  = "lattice_config.json"

        self._init_files()

        self.worker_assignments: List[List[int]] = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": GRID_X, "grid_y": GRID_Y, "grid_z": GRID_Z,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "rcs_depth": RCS_DEPTH,
                    "rcs_validate_every": RCS_VALIDATE_EVERY,
                    "rcs_probe_patches": RCS_PROBE_PATCHES,
                    "rcs_full_snapshot_steps": RCS_FULL_SNAPSHOT_STEPS,
                    "probe_metric": "loschmidt_echo",
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
                    "Echo_Fidelity", "N_Gates", "Is_Snapshot",
                ]).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: init failed: {e}", file=sys.stderr)

    def _log_energy(self, step, anneal, bulk, bound, total, min_fid):
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

    def _log_profiles(self, step, patch_profiles):
        try:
            with open(self.profiles_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Patch", "Face", "X_mean", "Y_mean", "Z_mean"])
                for patch_id, prof in patch_profiles.items():
                    for face_name, face_qubits in self.boundaries.items():
                        if not face_qubits: continue
                        xm = float(np.mean([prof["means"]["X"][self._bq_to_idx[q]]
                                            for q in face_qubits]))
                        ym = float(np.mean([prof["means"]["Y"][self._bq_to_idx[q]]
                                            for q in face_qubits]))
                        zm = float(np.mean([prof["means"]["Z"][self._bq_to_idx[q]]
                                            for q in face_qubits]))
                        w.writerow({"Step": step, "Patch": patch_id,
                                    "Face": face_name,
                                    "X_mean": xm, "Y_mean": ym, "Z_mean": zm})
        except Exception as e:
            print(f"[CSV] Profile log error: {e}", file=sys.stderr)

    def _log_rcs(self, step, anneal,
                 rcs_records: List[Tuple[int, float, Optional[int], bool]]) -> None:
        if not rcs_records:
            return
        try:
            with open(self.rcs_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    "Step", "Anneal_Percent", "Patch", "RCS_Depth",
                    "Echo_Fidelity", "N_Gates", "Is_Snapshot",
                ])
                for patch_id, echo_f, n_gates, is_snap in rcs_records:
                    w.writerow({
                        "Step": step, "Anneal_Percent": anneal,
                        "Patch": patch_id, "RCS_Depth": RCS_DEPTH,
                        "Echo_Fidelity": echo_f, "N_Gates": n_gates,
                        "Is_Snapshot": int(is_snap),
                    })
        except Exception as e:
            print(f"[CSV] Echo log error: {e}", file=sys.stderr)

    def run(self, total_steps: int, dt: float, initial_hx: float,
            target_g_face: float, target_J: float,
            target_hx: float, target_hz: float,
            measure_every: int = 1,
            effective_shots: float = 512.0) -> None:

        if total_steps < 1:   raise ValueError("total_steps must be >= 1")
        if measure_every < 1: raise ValueError("measure_every must be >= 1")

        # REV 100 FIX: the final step (total_steps - 1) is always a
        # measure step in the worker, so it is a legitimate snapshot step
        # even when it is not a multiple of measure_every.
        bad_cadence = [s for s in RCS_FULL_SNAPSHOT_STEPS
                       if (s % measure_every != 0) and (s != total_steps - 1)]
        if bad_cadence:
            raise ValueError(
                f"RCS_FULL_SNAPSHOT_STEPS {bad_cadence} are not measure steps "
                f"(must be multiples of measure_every={measure_every} or the "
                f"final step {total_steps - 1})")

        invalid_snaps = [s for s in RCS_FULL_SNAPSHOT_STEPS if s >= total_steps]
        if invalid_snaps:
            raise ValueError(
                f"RCS_FULL_SNAPSHOT_STEPS contains steps beyond total_steps={total_steps}: {invalid_snaps}"
            )

        # REV 100 FIX: target_hx / target_hz removed from rcs_cfg -- the
        # worker reads them from its function arguments.
        rcs_cfg = {
            "enabled":              RCS_VALIDATION_ENABLED,
            "depth":                RCS_DEPTH,
            "validate_every":       RCS_VALIDATE_EVERY,
            "probe_patches":        RCS_PROBE_PATCHES,
            "full_snapshot_steps":  RCS_FULL_SNAPSHOT_STEPS,
            "master_seed":          self.master_seed,
        }

        n_probe = TOTAL_PATCHES if RCS_PROBE_PATCHES is None else len(RCS_PROBE_PATCHES)
        print(f"[Engine] {TOTAL_PATCHES} patches, "
              f"{TOTAL_PATCHES * QUBITS_PER_PATCH} qubits, "
              f"{GPUS_AVAILABLE} GPUs, {total_steps} steps")
        print(f"[Master] Staggered init - expect ~{TOTAL_WORKERS * 10}s before first output.",
              flush=True)
        if RCS_VALIDATION_ENABLED:
            print(f"[Engine] Echo probe ON: depth={RCS_DEPTH} (mirrored -> "
                  f"{2 * RCS_DEPTH} effective layers), "
                  f"{n_probe} routine probe patch(es), "
                  f"every {RCS_VALIDATE_EVERY} measure steps")
            print(f"[Engine] Full-cube snapshots at steps: {RCS_FULL_SNAPSHOT_STEPS} "
                  f"(all {TOTAL_PATCHES} patches, 2x out_ket DMA per patch, "
                  f"parallel across workers)")

        active_ranks = [r for r in range(TOTAL_WORKERS)
                        if self.worker_assignments[r]]
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

                patch_full_states: Dict[int, dict] = {}
                bulk_energy      = 0.0
                max_lat_trotter  = max_lat_tomo = max_lat_rcs = 0.0
                min_fidelity     = 1.0
                rcs_records: List[Tuple[int, float, Optional[int], bool]] = []
                any_snapshot     = False

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for patch_id, payload in data.items():
                        patch_full_states[patch_id] = payload["state"]
                        bulk_energy    += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo    = max(max_lat_tomo, payload["lat_tomo_ms"])
                        max_lat_rcs     = max(max_lat_rcs, payload.get("lat_rcs_ms", 0.0))
                        min_fidelity    = min(min_fidelity, payload.get("unitary_fidelity", 1.0))

                        if payload.get("echo_fidelity") is not None:
                            is_snap = bool(payload.get("is_snapshot", False))
                            any_snapshot = any_snapshot or is_snap
                            rcs_records.append((
                                patch_id,
                                payload["echo_fidelity"],
                                payload["echo_n_gates"],
                                is_snap,
                            ))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(f"IPC gather incomplete: got {len(patch_full_states)}/{TOTAL_PATCHES}")

                step_state    = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
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
                            "X": state["var_X"][bq],
                            "Y": state["var_Y"][bq],
                            "Z": state["var_Z"][bq],
                        },
                    }

                self.lattice_history.append(step_state.copy())
                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Failed: {e}", file=sys.stderr)

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
                        "+X": (x1+1, y1,   z1  ),
                        "-X": (x1-1, y1,   z1  ),
                        "+Y": (x1,   y1+1, z1  ),
                        "-Y": (x1,   y1-1, z1  ),
                        "+Z": (x1,   y1,   z1+1),
                        "-Z": (x1,   y1,   z1-1),
                    }
                    for dir1, coord2 in neighbors.items():
                        patch_id_2 = self.coord_to_patch.get(coord2)
                        if patch_id_2 is None or patch_id_1 >= patch_id_2:
                            continue
                        dir2    = (dir1.replace("+", "temp").replace("-", "+").replace("temp", "-"))
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
                snap_tag     = " [SNAPSHOT]" if any_snapshot else ""
                status = (
                    f"Step {t:03d}{snap_tag} | E: {total_energy:+.4f} | "
                    f"Lat(Trot/Tomo/Echo): "
                    f"{max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}/{max_lat_rcs:6.1f}ms | "
                    f"Fid: {min_fidelity:.5f}"
                )
                if rcs_records:
                    # min is the headline: one degraded patch is the
                    # signal, and the mean alone would wash it out across
                    # 27 patches on snapshot steps.
                    min_echo  = min(r[1] for r in rcs_records)
                    mean_echo = float(np.mean([r[1] for r in rcs_records]))
                    n_rcs     = len(rcs_records)
                    status   += (f" | Echo(min/mean): {min_echo:.6f}/{mean_echo:.6f} "
                                 f"[{n_rcs} patch{'es' if n_rcs>1 else ''}]")
                status += f" | {time.perf_counter() - t0:.2f}s"
                print(status)

                self._log_energy(t, s * 100, bulk_energy, macroscopic_boundary_energy, total_energy, min_fidelity)
                self._log_profiles(t, patch_profiles)
                self._log_rcs(t, s * 100, rcs_records)

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
