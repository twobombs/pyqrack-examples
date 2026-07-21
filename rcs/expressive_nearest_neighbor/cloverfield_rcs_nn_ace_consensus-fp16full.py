# -*- coding: us-ascii -*-
# 16-Qubit 4x4 Brane Tiles -> 4x4x4 Brane-Stack Blocks -> 4x4x4 Block Lattice
# (256 Patches, 4096 Qubits Total)
#
# DEV BUILD - Fast fp16-capable Trotter engine for development iteration
#
# INTEGRATION NOTES (cloverfield_rcs_nn_ace_consensus-full.py -> this file):
#
#   SOURCE TOPOLOGY  : 4x4x4 brane block lattice (Rev 88-C)
#     256 patches, 16 qubits/patch (fp16-capable), 3 coupling kinds
#     (Z_INTRA / Z_INTER / XY), site-resolved kicks, continuous kick
#     application every Trotter step.
#
#   DONATED COMPONENTS from cloverfield (Rev 99):
#     1. STRANG SPLITTING  - trotter_step_body now uses correct symmetric
#        splitting: Rx(half) -> Rz(half) -> ZZ(full) -> Rz(half) -> Rx(half).
#        Rev 88-C had an asymmetric form (theta_z applied at full dt, not half).
#     2. VECTORIZED r()    - startup probe detects batch r(axis, angle, [q0..qN])
#        support; falls back to loop if unavailable.
#     3. MEASURE_SHOTS LSB - smoke-test verifying out_probs() / measure_shots()
#        indexing convention before first RCS probe.
#     4. STARTUP STAGGER   - rank * 10 s delay to avoid simultaneous JIT
#        compilation storms across workers sharing GPU memory.
#     5. RCS LAYER         - optional per-step random-circuit probe with
#        out_ket() / in_ket() snapshot restoration, XEB + HOG scoring,
#        rcs_validation.csv output. All RCS config exposed at top of file.
#     6. var_X/Y/Z IPC     - workers ship exact Pauli variances (1 - <P>^2)
#        alongside means; master uses them for stochastic kick scaling.
#
#   RETAINED from Rev 88-C (unchanged):
#     - 4x4x4 block lattice topology, build_interfaces(), 3 coupling kinds.
#     - Continuous kick application on every Trotter step (no is_measure gate).
#     - Site-resolved numpy kick accumulators -> sparse dict payloads.
#     - Per-class energy logging (E_Z_Intra, E_Z_Inter, E_XY).
#     - GPUS_AVAILABLE / WORKERS_PER_GPU tuning knobs.
#     - PERIODIC_X/Y/Z boundary flags.

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

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
BLOCK_GRID_X     = 4
BLOCK_GRID_Y     = 4
BLOCK_GRID_Z     = 4
BRANES_PER_BLOCK = 4
GLOBAL_Z         = BLOCK_GRID_Z * BRANES_PER_BLOCK   # 16 brane layers

TOTAL_PATCHES    = BLOCK_GRID_X * BLOCK_GRID_Y * GLOBAL_Z  # 256
QUBITS_PER_PATCH = 16
TOTAL_QUBITS     = TOTAL_PATCHES * QUBITS_PER_PATCH         # 4096

PERIODIC_X = False
PERIODIC_Y = False
PERIODIC_Z = False

# GPU topology - tune for your hardware
GPUS_AVAILABLE  = 6
WORKERS_PER_GPU = 4    # 256 patches / 24 workers = ~10-11 patches/worker
TOTAL_WORKERS   = GPUS_AVAILABLE * WORKERS_PER_GPU  # 24

# =====================================================================
# RCS CONFIGURATION (donated from cloverfield Rev 99)
# =====================================================================
RCS_VALIDATION_ENABLED  = True
RCS_DEPTH               = 12          # shallower than 27q version; 16q circuit
RCS_SHOTS               = 128
RCS_VALIDATE_EVERY      = 5
RCS_PROBE_PATCHES       = None        # None = one probe per worker's assigned set
RCS_FULL_SNAPSHOT_STEPS = [42, 82, 99]

# =====================================================================
# ENVIRONMENT
# =====================================================================
os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"


# =====================================================================
# PURE FUNCTIONS - Topology
# =====================================================================
def generate_16q_brane_tile() -> Tuple[List[Tuple[int, int]], List[int]]:
    """4x4 planar square lattice. idx = x*4 + y (row-major in x).
    Returns (intra_edges, brane_sites). Every site is a Z-interface site."""
    lx, ly = 4, 4
    edges: List[Tuple[int, int]] = []
    for x in range(lx):
        for y in range(ly):
            idx = x * ly + y
            if x < lx - 1: edges.append((idx, (x + 1) * ly + y))
            if y < ly - 1: edges.append((idx, x * ly + (y + 1)))
    return edges, list(range(lx * ly))


def patch_id(tx: int, ty: int, z: int) -> int:
    return (tx * BLOCK_GRID_Y + ty) * GLOBAL_Z + z


def patch_coords(p: int) -> Tuple[int, int, int]:
    z    = p % GLOBAL_Z
    rest = p // GLOBAL_Z
    return rest // BLOCK_GRID_Y, rest % BLOCK_GRID_Y, z


def build_interfaces() -> List[Tuple[int, int, np.ndarray, np.ndarray, str]]:
    """All coupled seams as (p1, p2, idx1, idx2, kind).
    Z_INTRA: adjacent branes within a block.
    Z_INTER: block-boundary brane pairs along Z.
    XY:      lateral tile-edge seams between in-plane neighbor blocks.
    """
    z_i1 = np.arange(QUBITS_PER_PATCH)
    z_i2 = z_i1.copy()
    x_i1 = np.array([12 + y for y in range(4)])   # +X face of p1
    x_i2 = np.array([y for y in range(4)])        # -X face of p2
    y_i1 = np.array([x * 4 + 3 for x in range(4)]) # +Y face of p1
    y_i2 = np.array([x * 4 for x in range(4)])     # -Y face of p2

    interfaces: List[Tuple[int, int, np.ndarray, np.ndarray, str]] = []
    for tx in range(BLOCK_GRID_X):
        for ty in range(BLOCK_GRID_Y):
            for z in range(GLOBAL_Z):
                p1 = patch_id(tx, ty, z)

                # Z neighbor
                if z < GLOBAL_Z - 1:
                    kind = "Z_INTRA" if (z + 1) % BRANES_PER_BLOCK != 0 else "Z_INTER"
                    interfaces.append((p1, patch_id(tx, ty, z + 1), z_i1, z_i2, kind))
                elif PERIODIC_Z and GLOBAL_Z > 2:
                    interfaces.append((p1, patch_id(tx, ty, 0), z_i1, z_i2, "Z_INTER"))

                # X neighbor
                if tx < BLOCK_GRID_X - 1:
                    interfaces.append((p1, patch_id(tx + 1, ty, z), x_i1, x_i2, "XY"))
                elif PERIODIC_X and BLOCK_GRID_X > 2:
                    interfaces.append((p1, patch_id(0, ty, z), x_i1, x_i2, "XY"))

                # Y neighbor
                if ty < BLOCK_GRID_Y - 1:
                    interfaces.append((p1, patch_id(tx, ty + 1, z), y_i1, y_i2, "XY"))
                elif PERIODIC_Y and BLOCK_GRID_Y > 2:
                    interfaces.append((p1, patch_id(tx, 0, z), y_i1, y_i2, "XY"))

    return interfaces


# =====================================================================
# PURE FUNCTIONS - RCS (donated from cloverfield Rev 99)
# =====================================================================
def calc_xeb(ideal_probs: np.ndarray, width: int) -> Tuple[float, float]:
    n_pow = float(1 << width)
    if ideal_probs.size == 0:
        return 0.0, 0.0
    xeb = n_pow * float(np.mean(ideal_probs)) - 1.0
    hog = float(np.mean(ideal_probs > (math.log(2.0) / n_pow)))
    return xeb, hog


def apply_rcs_layer(sim, num_qubits: int, edges: List[Tuple[int, int]],
                    depth: int, rng: random.Random) -> None:
    """Apply `depth` layers of random u + iswap gates in-place."""
    for _ in range(depth):
        for q in range(num_qubits):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            lam   = rng.uniform(0.0, 2.0 * math.pi)
            sim.u(q, theta, phi, lam)
        shuffled = list(edges)
        rng.shuffle(shuffled)
        used: set = set()
        for q1, q2 in shuffled:
            if q1 not in used and q2 not in used:
                sim.iswap(q1, q2)
                used.add(q1)
                used.add(q2)


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

    # Startup stagger: prevents simultaneous JIT storms (donated from cloverfield)
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
    os.environ["QRACK_MAX_ALLOC_MB"]                 = str(64000 // WORKERS_PER_GPU)
    os.environ["QRACK_DISABLE_QUNIT_FIDELITY_GUARD"] = "1"

    from pyqrack import QrackSimulator

    sims: Dict[int, Any] = {}

    try:
        # --- PAULI CODE AUTODETECT (identical to cloverfield Rev 99) ---
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

        # --- ANGLE CONVENTION AUTODETECT ---
        _sim_mag = QrackSimulator(qubit_count=1, is_binary_decision_tree=False)
        _sim_mag.r(PX, math.pi, 0)
        _corrected = SIGN_Z * _sim_mag.pauli_expectation([0], [PZ])
        if   abs(_corrected + 1.0) < 0.1: ANGLE_SCALE = 1.0
        elif abs(_corrected - 1.0) < 0.1: ANGLE_SCALE = 0.5
        else: raise RuntimeError(
            f"Fatal: ambiguous ANGLE_SCALE, SIGN_Z*<Z>={_corrected:.6f}")
        del _sim_mag

        # --- VECTORIZED r() DETECTION (donated from cloverfield Rev 99) ---
        _vec_probe = QrackSimulator(qubit_count=2, is_binary_decision_tree=False,
                                    is_gpu=True)
        _VECTORIZED_R = False
        try:
            _vec_probe.r(PX, 0.0, [0, 1])
            _VECTORIZED_R = True
        except Exception:
            pass
        del _vec_probe
        print(f"[Worker {rank}] Vectorized r(): {'YES' if _VECTORIZED_R else 'NO (loop fallback)'}",
              flush=True)

        # --- MEASURE_SHOTS LSB/MSB SMOKE TEST (donated from cloverfield Rev 99) ---
        _sim_ms = QrackSimulator(qubit_count=5, is_binary_decision_tree=False, is_gpu=True)
        _sim_ms.x(0)
        _sim_ms.x(2)   # Prepare |00101> -> index 5 in LSB
        _probs_ms = np.array(_sim_ms.out_probs())
        _shot_ms  = _sim_ms.measure_shots(list(range(5)), 1)[0]
        if _probs_ms[int(_shot_ms)] < 0.9:
            raise RuntimeError(
                f"Fatal: measure_shots LSB/MSB mismatch. "
                f"Prepared |5>, got shot {_shot_ms}, prob {_probs_ms[int(_shot_ms)]:.4f}.")
        del _sim_ms

        # --- GATE HELPERS ---
        _all_q_list = list(range(QUBITS_PER_PATCH))

        if _VECTORIZED_R:
            def apply_rx_all(sim, theta: float) -> None:
                sim.r(PX, float(theta) * ANGLE_SCALE, _all_q_list)
            def apply_rz_all(sim, theta: float) -> None:
                sim.r(PZ, float(theta) * ANGLE_SCALE, _all_q_list)
        else:
            def apply_rx_all(sim, theta: float) -> None:
                ang = float(theta) * ANGLE_SCALE
                for q in _all_q_list: sim.r(PX, ang, q)
            def apply_rz_all(sim, theta: float) -> None:
                ang = float(theta) * ANGLE_SCALE
                for q in _all_q_list: sim.r(PZ, ang, q)

        def apply_rx(sim, theta: float, q: int) -> None:
            sim.r(PX, float(theta) * ANGLE_SCALE, q)
        def apply_ry(sim, theta: float, q: int) -> None:
            sim.r(PY, float(theta) * ANGLE_SCALE, q)
        def apply_rz(sim, theta: float, q: int) -> None:
            sim.r(PZ, float(theta) * ANGLE_SCALE, q)

        def apply_zz(sim, theta: float, q1: int, q2: int) -> None:
            sim.mcx([q1], q2)
            apply_rz(sim, 2.0 * theta, q2)
            sim.mcx([q1], q2)

        # STRANG SPLITTING (donated from cloverfield Rev 99)
        # Fixes Rev 88-C asymmetric form where theta_z used dt (not dt/2).
        # Order: Rx(half) -> Rz(half) -> ZZ(full) -> Rz(half) -> Rx(half)
        def trotter_step_body(sim, num_qubits: int, edge_list: List[Tuple[int, int]],
                              J: float, hx: float, hz: float, dt_local: float) -> None:
            dt_half      = dt_local / 2.0
            theta_x      = -2.0 * hx * dt_half
            theta_z_half = -2.0 * hz * dt_half
            theta_zz     = -J * dt_local
            apply_rx_all(sim, theta_x)
            apply_rz_all(sim, theta_z_half)
            for q1, q2 in edge_list: apply_zz(sim, theta_zz, q1, q2)
            apply_rz_all(sim, theta_z_half)
            apply_rx_all(sim, theta_x)

        def z_means(sim, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Z * float(sim.pauli_expectation([q], [PZ])) for q in qubits])
        def x_means(sim, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_X * float(sim.pauli_expectation([q], [PX])) for q in qubits])
        def y_means(sim, qubits: List[int]) -> np.ndarray:
            return np.array([SIGN_Y * float(sim.pauli_expectation([q], [PY])) for q in qubits])
        def zz_means_mf(z_exp: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
            return np.array([z_exp[q1] * z_exp[q2] for q1, q2 in edges])

        def apply_kicks(sim, kicks: Dict[int, Tuple[float, float, float]],
                        dt_local: float) -> None:
            if not kicks: return
            coef = -2.0 * dt_local
            for raw_q, (kx, ky, kz) in kicks.items():
                q = int(raw_q)
                if abs(kx * coef) > 1e-12: apply_rx(sim, kx * coef, q)
                if abs(ky * coef) > 1e-12: apply_ry(sim, ky * coef, q)
                if abs(kz * coef) > 1e-12: apply_rz(sim, kz * coef, q)

        # --- TOPOLOGY ---
        intra_edges, _brane_sites = generate_16q_brane_tile()
        all_q = list(range(QUBITS_PER_PATCH))

        # --- RCS CONFIG ---
        rcs_enabled = rcs_cfg.get("enabled", False)
        if rcs_cfg.get("probe_patches") is None:
            rcs_probe_set = set(assigned_patches)
        else:
            rcs_probe_set = set(rcs_cfg["probe_patches"]) & set(assigned_patches)
        full_snapshot_steps = set(rcs_cfg.get("full_snapshot_steps", []))
        master_seed        = rcs_cfg.get("master_seed", 1337)
        _warned_fidelity   = False

        # --- SIMULATOR ALLOCATION ---
        for patch_id_p in assigned_patches:
            sim = QrackSimulator(
                qubit_count=QUBITS_PER_PATCH,
                is_binary_decision_tree=False,
                is_stabilizer_hybrid=False,
                is_gpu=True,
            )
            for q in range(QUBITS_PER_PATCH): sim.h(q)
            sims[patch_id_p] = sim
            try:
                _ = sim.pauli_expectation([0], [PZ])
            except Exception as e:
                raise RuntimeError(f"Fatal: VRAM smoke test failed on patch {patch_id_p}: {e}")

        print(f"[Worker {rank}] {len(assigned_patches)} patches allocated.", flush=True)

        kick_payloads = {p: {} for p in assigned_patches}
        meas_count    = 0

        for t in range(total_steps):
            s          = t / max(1, total_steps - 1)
            current_hx = (1.0 - s) * initial_hx + s * rcs_cfg["target_hx"]
            current_J  = s * target_J
            current_hz = s * rcs_cfg["target_hz"]
            is_measure = (t % measure_every == 0) or (t == total_steps - 1)

            is_snapshot = (t in full_snapshot_steps)
            do_rcs = (
                rcs_enabled and is_measure and
                (is_snapshot or (meas_count % max(1, rcs_cfg["validate_every"]) == 0))
            )
            rcs_probe_this_step = set(assigned_patches) if is_snapshot else rcs_probe_set

            patch_data: Dict[int, dict] = {}

            for p in assigned_patches:
                sim = sims[p]

                # Continuous kick application (every step - Rev 88-C behavior retained)
                if kick_payloads[p]:
                    apply_kicks(sim, kick_payloads[p], dt)

                t0_tr = time.perf_counter()
                trotter_step_body(sim, QUBITS_PER_PATCH, intra_edges,
                                  current_J, current_hx, current_hz, dt)
                lat_trotter = (time.perf_counter() - t0_tr) * 1000.0

                if not is_measure:
                    continue

                t0_tomo = time.perf_counter()
                x_exp = x_means(sim, all_q)
                y_exp = y_means(sim, all_q)
                z_exp = z_means(sim, all_q)

                state = {
                    "X": x_exp,
                    "Y": y_exp,
                    "Z": z_exp,
                    # Exact Pauli variance (donated from cloverfield Rev 99)
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
                        print(f"[Worker {rank}] Warning: get_unitary_fidelity() unavailable.",
                              file=sys.stderr)
                        _warned_fidelity = True

                # --- RCS VALIDATION (donated from cloverfield Rev 99) ---
                rcs_xeb, rcs_hog = None, None
                lat_rcs = 0.0

                if do_rcs and p in rcs_probe_this_step:
                    t0_rcs = time.perf_counter()
                    pristine_ket = None
                    try:
                        depth = rcs_cfg["depth"]
                        shots = rcs_cfg["shots"]
                        rng   = random.Random((master_seed << 32) ^ (p << 16) ^ t)

                        pristine_ket = sim.out_ket()
                        apply_rcs_layer(sim, QUBITS_PER_PATCH, intra_edges, depth, rng)
                        
                        # [88D-04] FIX: Use sparse prob_perm to avoid out_probs() 512KB DMA burst
                        # (Also replaced 's' shadow variable with 'outcome')
                        samples    = sim.measure_shots(all_q, shots)
                        unique_out = list(set(int(o) for o in samples))
                        bit_lists  = [[((outcome >> q) & 1) for q in all_q] for outcome in unique_out]
                        sparse_p   = {outcome: sim.prob_perm(all_q, bl) for outcome, bl in zip(unique_out, bit_lists)}
                        ideal_p    = np.array([sparse_p[int(o)] for o in samples], dtype=np.float64)
                        
                        rcs_xeb, rcs_hog = calc_xeb(ideal_p, QUBITS_PER_PATCH)
                        lat_rcs = (time.perf_counter() - t0_rcs) * 1000.0

                        if is_snapshot:
                            print(f"[Worker {rank}] Snapshot RCS patch {p}: "
                                  f"XEB={rcs_xeb:+.4f} HOG={rcs_hog:.3f}", flush=True)

                    except Exception as e:
                        print(f"[Worker {rank}] RCS error (patch {p}): {e}", file=sys.stderr)
                        rcs_xeb, rcs_hog = None, None

                    finally:
                        if pristine_ket is not None:
                            try:
                                sim.in_ket(pristine_ket)
                            except Exception as restore_e:
                                print(f"[Worker {rank}] FATAL: ket restore failed on "
                                      f"patch {p}: {restore_e}", file=sys.stderr)
                            finally:
                                del pristine_ket

                patch_data[p] = {
                    "state":                 state,
                    "meanfield_bulk_energy": bulk_e,
                    "lat_trotter_ms":        lat_trotter,
                    "lat_tomo_ms":           lat_tomo,
                    "lat_rcs_ms":            lat_rcs,
                    "unitary_fidelity":      fidelity,
                    "rcs_xeb":               rcs_xeb,
                    "rcs_hog":               rcs_hog,
                    "is_snapshot":           is_snapshot,
                }

            if is_measure:
                meas_count += 1
                conn.send(patch_data)
                kick_payloads = conn.recv()

    finally:
        for p in list(sims.keys()):
            s = sims.pop(p); del s
        sims.clear()
        gc.collect()
        conn.close()


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================
class MultiGpuHadronEngine:
    def __init__(self, master_seed: int = 1337) -> None:
        self.master_seed  = master_seed
        self.intra_edges, self.brane_sites = generate_16q_brane_tile()
        self.n_sites      = len(self.brane_sites)  # 16
        self.patch_coords = {p: patch_coords(p) for p in range(TOTAL_PATCHES)}
        self.interfaces   = build_interfaces()

        n_by_kind = {"Z_INTRA": 0, "Z_INTER": 0, "XY": 0}
        for _, _, _, _, kind in self.interfaces:
            n_by_kind[kind] += 1
        self.n_by_kind = n_by_kind

        self.lattice_history: List[np.ndarray] = []
        self.energy_csv  = "meanfield_ground_state_energy_curve_multi.csv"
        self.profiles_csv = "boundary_profiles_multi.csv"
        self.rcs_csv     = "rcs_validation_multi.csv"
        self.state_dump  = "macroscopic_lattice_states.npy"
        self.config_file = "lattice_config.json"

        self._energy_fields = [
            "Step", "Anneal_Percent", "MeanField_Bulk_Energy",
            "E_Z_Intra", "E_Z_Inter", "E_XY",
            "MeanField_Boundary_Energy", "MeanField_Total_Energy",
            "Min_Unitary_Fidelity",
        ]
        self._profile_fields = [
            "Step", "Patch", "Tx", "Ty", "Z", "Block_Z", "Layer",
            "Face", "X_mean", "Y_mean", "Z_mean",
        ]
        self._rcs_fields = [
            "Step", "Anneal_Percent", "Patch", "RCS_Depth",
            "XEB_RCS", "HOG_RCS", "Is_Snapshot",
        ]

        self._init_files()

        self.worker_assignments: List[List[int]] = [[] for _ in range(TOTAL_WORKERS)]
        for i in range(TOTAL_PATCHES):
            self.worker_assignments[i % TOTAL_WORKERS].append(i)

    def _init_files(self) -> None:
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "grid_x": BLOCK_GRID_X, "grid_y": BLOCK_GRID_Y,
                    "grid_z": GLOBAL_Z,
                    "block_grid": [BLOCK_GRID_X, BLOCK_GRID_Y, BLOCK_GRID_Z],
                    "branes_per_block": BRANES_PER_BLOCK,
                    "num_patches": TOTAL_PATCHES,
                    "qubits_per_patch": QUBITS_PER_PATCH,
                    "total_qubits": TOTAL_QUBITS,
                    "periodic": [PERIODIC_X, PERIODIC_Y, PERIODIC_Z],
                    "interfaces_by_kind": self.n_by_kind,
                    "rcs_depth": RCS_DEPTH,
                    "rcs_shots": RCS_SHOTS,
                    "rcs_validate_every": RCS_VALIDATE_EVERY,
                    "rcs_full_snapshot_steps": RCS_FULL_SNAPSHOT_STEPS,
                }, f, indent=2)
            with open(self.energy_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._energy_fields).writeheader()
            with open(self.profiles_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._profile_fields).writeheader()
            with open(self.rcs_csv, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self._rcs_fields).writeheader()
        except Exception as e:
            print(f"[CSV] Warning: init failed: {e}", file=sys.stderr)

    def _log_energy(self, step: int, anneal: float, bulk: float,
                    e_by_kind: Dict[str, float], bound: float, total: float,
                    min_fid: float) -> None:
        try:
            with open(self.energy_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self._energy_fields).writerow({
                    "Step": step, "Anneal_Percent": anneal,
                    "MeanField_Bulk_Energy": bulk,
                    "E_Z_Intra": e_by_kind["Z_INTRA"],
                    "E_Z_Inter": e_by_kind["Z_INTER"],
                    "E_XY": e_by_kind["XY"],
                    "MeanField_Boundary_Energy": bound,
                    "MeanField_Total_Energy": total,
                    "Min_Unitary_Fidelity": min_fid,
                })
        except Exception as e:
            print(f"[CSV] Energy log error: {e}", file=sys.stderr)

    def _log_profiles(self, step: int, patch_profiles: Dict[int, dict]) -> None:
        try:
            with open(self.profiles_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._profile_fields)
                for p, prof in patch_profiles.items():
                    tx, ty, z = self.patch_coords[p]
                    w.writerow({
                        "Step": step, "Patch": p,
                        "Tx": tx, "Ty": ty, "Z": z,
                        "Block_Z": z // BRANES_PER_BLOCK,
                        "Layer": z % BRANES_PER_BLOCK,
                        "Face": "BRANE",
                        "X_mean": float(np.mean(prof["means"]["X"])),
                        "Y_mean": float(np.mean(prof["means"]["Y"])),
                        "Z_mean": float(np.mean(prof["means"]["Z"])),
                    })
        except Exception as e:
            print(f"[CSV] Profile log error: {e}", file=sys.stderr)

    def _log_rcs(self, step: int, anneal: float,
                 rcs_records: List[Tuple[int, float, float, bool]]) -> None:
        if not rcs_records:
            return
        try:
            with open(self.rcs_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self._rcs_fields)
                for patch_id_r, xeb, hog, is_snap in rcs_records:
                    w.writerow({
                        "Step": step, "Anneal_Percent": anneal,
                        "Patch": patch_id_r, "RCS_Depth": RCS_DEPTH,
                        "XEB_RCS": xeb, "HOG_RCS": hog,
                        "Is_Snapshot": int(is_snap),
                    })
        except Exception as e:
            print(f"[CSV] RCS log error: {e}", file=sys.stderr)

    def run(self,
            total_steps: int,
            dt: float,
            initial_hx: float,
            target_g_intra_z: float,
            target_g_inter_z: float,
            target_g_xy: float,
            target_J: float,
            target_hx: float,
            target_hz: float,
            measure_every: int = 1,
            effective_shots: float = 512.0) -> None:

        if total_steps < 1:   raise ValueError("total_steps must be >= 1")
        if measure_every < 1: raise ValueError("measure_every must be >= 1")

        if not all(s % measure_every == 0 for s in RCS_FULL_SNAPSHOT_STEPS):
            raise ValueError(
                f"All RCS_FULL_SNAPSHOT_STEPS must be multiples of measure_every ({measure_every})")
        invalid_snaps = [s for s in RCS_FULL_SNAPSHOT_STEPS if s >= total_steps]
        if invalid_snaps:
            raise ValueError(
                f"RCS_FULL_SNAPSHOT_STEPS contains steps beyond total_steps={total_steps}: {invalid_snaps}")

        rcs_cfg = {
            "enabled":             RCS_VALIDATION_ENABLED,
            "depth":               RCS_DEPTH,
            "shots":               RCS_SHOTS,
            "validate_every":      RCS_VALIDATE_EVERY,
            "probe_patches":       RCS_PROBE_PATCHES,
            "full_snapshot_steps": RCS_FULL_SNAPSHOT_STEPS,
            "target_hx":           target_hx,
            "target_hz":           target_hz,
            "master_seed":         self.master_seed,
        }

        print(f"[Engine] {BLOCK_GRID_X}x{BLOCK_GRID_Y}x{BLOCK_GRID_Z} block lattice "
              f"x {BRANES_PER_BLOCK} branes/block = {TOTAL_PATCHES} patches, "
              f"{TOTAL_QUBITS} qubits | interfaces: "
              f"{self.n_by_kind['Z_INTRA']} Z-intra, "
              f"{self.n_by_kind['Z_INTER']} Z-inter, "
              f"{self.n_by_kind['XY']} XY | "
              f"{GPUS_AVAILABLE} GPUs ({WORKERS_PER_GPU} workers/GPU), {total_steps} steps")
        print(f"[Master] Staggered init - expect ~{TOTAL_WORKERS * 10}s before first output.",
              flush=True)
        if RCS_VALIDATION_ENABLED:
            probe_desc = ("all patches" if RCS_PROBE_PATCHES is None
                          else f"{len(RCS_PROBE_PATCHES)} patch(es)")
            print(f"[Engine] RCS ON: depth={RCS_DEPTH}, shots={RCS_SHOTS}, "
                  f"{probe_desc} per worker, every {RCS_VALIDATE_EVERY} measure steps")
            print(f"[Engine] Full-cube snapshots at steps: {RCS_FULL_SNAPSHOT_STEPS}")

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
                s = t / max(1, total_steps - 1)
                g_now = {
                    "Z_INTRA": s * target_g_intra_z,
                    "Z_INTER": s * target_g_inter_z,
                    "XY":      s * target_g_xy,
                }
                is_measure = (t % measure_every == 0) or (t == total_steps - 1)
                if not is_measure:
                    continue

                t0 = time.perf_counter()

                patch_full_states: Dict[int, dict] = {}
                bulk_energy       = 0.0
                max_lat_trotter   = max_lat_tomo = max_lat_rcs = 0.0
                min_fidelity      = 1.0
                rcs_records: List[Tuple[int, float, float, bool]] = []
                any_snapshot      = False

                for conn in pipes:
                    try:
                        data = conn.recv()
                    except EOFError:
                        raise RuntimeError("Worker IPC connection lost.")
                    for p, payload in data.items():
                        patch_full_states[p] = payload["state"]
                        bulk_energy    += payload["meanfield_bulk_energy"]
                        max_lat_trotter = max(max_lat_trotter, payload["lat_trotter_ms"])
                        max_lat_tomo    = max(max_lat_tomo, payload["lat_tomo_ms"])
                        max_lat_rcs     = max(max_lat_rcs, payload.get("lat_rcs_ms", 0.0))
                        min_fidelity    = min(min_fidelity, payload.get("unitary_fidelity", 1.0))
                        if payload.get("rcs_xeb") is not None:
                            is_snap      = bool(payload.get("is_snapshot", False))
                            any_snapshot = any_snapshot or is_snap
                            rcs_records.append((p, payload["rcs_xeb"],
                                                payload["rcs_hog"], is_snap))

                if len(patch_full_states) != TOTAL_PATCHES:
                    raise RuntimeError(
                        f"IPC gather incomplete: got {len(patch_full_states)}/{TOTAL_PATCHES}")

                # --- BUILD PROFILES ---
                step_state = np.zeros((TOTAL_PATCHES, QUBITS_PER_PATCH, 3))
                patch_profiles: Dict[int, dict] = {}

                for p, state in patch_full_states.items():
                    step_state[p, :, 0] = state["X"]
                    step_state[p, :, 1] = state["Y"]
                    step_state[p, :, 2] = state["Z"]
                    patch_profiles[p] = {
                        "means": {
                            "X": state["X"].copy(),
                            "Y": state["Y"].copy(),
                            "Z": state["Z"].copy(),
                        },
                        # Use worker-computed exact variances (donated from cloverfield)
                        "vars": {
                            "X": state["var_X"].copy(),
                            "Y": state["var_Y"].copy(),
                            "Z": state["var_Z"].copy(),
                        },
                    }

                self.lattice_history.append(step_state.copy())
                if len(self.lattice_history) % 10 == 0:
                    try:
                        np.save(self.state_dump, np.array(self.lattice_history))
                    except Exception as e:
                        print(f"[Checkpoint] Failed: {e}", file=sys.stderr)

                # --- COMPUTE KICKS & INTERFACE ENERGY (site-resolved, Rev 88-C) ---
                scale = np.sqrt(dt / effective_shots)
                n_s   = self.n_sites
                AXES  = ("X", "Y", "Z")

                noisy_field: Dict[int, Dict[str, np.ndarray]] = {}
                for p in range(TOTAL_PATCHES):
                    prof    = patch_profiles[p]
                    rng_p   = np.random.default_rng([self.master_seed, t, p])
                    noisy_field[p] = {
                        ax: prof["means"][ax]
                            + rng_p.normal(0.0, 1.0, n_s)
                            * np.sqrt(prof["vars"][ax]) * scale
                        for ax in AXES
                    }

                kick_acc = {p: np.zeros((n_s, 3)) for p in range(TOTAL_PATCHES)}
                e_by_kind = {"Z_INTRA": 0.0, "Z_INTER": 0.0, "XY": 0.0}

                for p1, p2, i1, i2, kind in self.interfaces:
                    g = g_now[kind]
                    if g == 0.0:
                        continue
                    f1 = noisy_field[p1]
                    f2 = noisy_field[p2]
                    m1 = patch_profiles[p1]["means"]
                    m2 = patch_profiles[p2]["means"]

                    dot = 0.0
                    for a, ax in enumerate(AXES):
                        dot += float(np.sum(m1[ax][i1] * m2[ax][i2]))
                        kick_acc[p1][i1, a] += g * f2[ax][i2]
                        kick_acc[p2][i2, a] += g * f1[ax][i1]
                    e_by_kind[kind] += -g * dot

                macroscopic_boundary_energy = sum(e_by_kind.values())

                next_kick_payloads: Dict[int, dict] = {}
                for p in range(TOTAL_PATCHES):
                    acc = kick_acc[p]
                    next_kick_payloads[p] = {
                        q: (float(acc[q, 0]), float(acc[q, 1]), float(acc[q, 2]))
                        for q in range(n_s) if np.any(np.abs(acc[q]) > 0.0)
                    }

                total_energy = bulk_energy + macroscopic_boundary_energy
                snap_tag     = " [SNAPSHOT]" if any_snapshot else ""
                status = (
                    f"Step {t:03d}{snap_tag} | E: {total_energy:+.4f} "
                    f"(Zi {e_by_kind['Z_INTRA']:+.3f} / "
                    f"Ze {e_by_kind['Z_INTER']:+.3f} / "
                    f"XY {e_by_kind['XY']:+.3f}) | "
                    f"Lat(Trot/Tomo/RCS): "
                    f"{max_lat_trotter:5.1f}/{max_lat_tomo:5.1f}/{max_lat_rcs:6.1f}ms | "
                    f"Fid: {min_fidelity:.5f}"
                )
                if rcs_records:
                    mean_xeb = float(np.mean([r[1] for r in rcs_records]))
                    mean_hog = float(np.mean([r[2] for r in rcs_records]))
                    n_rcs    = len(rcs_records)
                    status  += (f" | XEB(RCS): {mean_xeb:+.4f} HOG: {mean_hog:.3f} "
                                f"[{n_rcs} patch{'es' if n_rcs > 1 else ''}]")
                status += f" | {time.perf_counter() - t0:.2f}s"
                print(status)

                self._log_energy(t, s * 100, bulk_energy, e_by_kind,
                                 macroscopic_boundary_energy, total_energy, min_fidelity)
                self._log_profiles(t, patch_profiles)
                self._log_rcs(t, s * 100, rcs_records)

                for i, w_rank in enumerate(active_ranks):
                    pipes[i].send({p: next_kick_payloads[p]
                                   for p in self.worker_assignments[w_rank]})

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
            # Hierarchy: g_intra_z binds branes within a block (tight),
            # g_inter_z and g_xy are weaker inter-block glue.
            # Set all three equal to recover a uniform mean-field 16x16x16 cube.
            target_g_intra_z=0.12,
            target_g_inter_z=0.06,
            target_g_xy=0.06,
            target_J=1.0,
            target_hx=0.5,
            target_hz=0.2,
            measure_every=1,
            effective_shots=512.0,
        )
    except KeyboardInterrupt:
        pass
