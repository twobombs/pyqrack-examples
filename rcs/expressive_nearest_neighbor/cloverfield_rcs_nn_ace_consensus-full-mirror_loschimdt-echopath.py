# -*- coding: us-ascii -*-
# REV 100 PATCH: LOSCHMIDT ECHO PROBE (replaces self-referential XEB)
#
# Changed sections only. Everything not shown is unchanged from Rev 99.
#
# SUMMARY OF CHANGES:
#   1. apply_rcs_layer() now records the gate stream; new apply_rcs_inverse()
#      replays it in reverse with exact gate inverses (mirror circuit C -> C^dag).
#   2. Probe metric is now echo fidelity F = |<psi_pristine|psi_mirror>|^2,
#      computed on host via np.vdot over the two kets. Sensitive to SDRP
#      rounding / precision loss over the 2*depth circuit; exact sim -> 1.0.
#   3. out_probs() + measure_shots() sampling path DELETED from the probe.
#      Removes the ~1 GB DMA burst and the multi-GB host list allocation
#      (Rev 99 review finding #2 resolved).
#   4. adjiswap availability smoke-tested at worker init, with a verified
#      decomposed fallback: iSWAP^dag = (Sdag x Sdag) . CZ . SWAP.
#   5. Config: RCS_SHOTS removed (unused). CSV schema: XEB_RCS/HOG_RCS ->
#      Echo_Fidelity/N_Gates. IPC payload: rcs_xeb/rcs_hog -> echo_fidelity/
#      echo_n_gates.
#   6. The measure_shots LSB/MSB smoke test from Rev 99 is now dead weight
#      for the probe path; retained here commented-out in case other tooling
#      relies on it, safe to delete.
#
# KNOWN LIMITATION (by design): mirror pairs can coherently cancel errors
# between C and C^dag, so F_echo is an OPTIMISTIC fidelity bound. Future
# hardening: insert a random Pauli layer at the midpoint and conjugate the
# inverse through it (Pauli-frame randomized mirror benchmarking).

# --- GLOBAL CONFIGURATION (changed lines) ---
RCS_VALIDATION_ENABLED  = True
RCS_DEPTH               = 20
# RCS_SHOTS deleted -- no sampling in the echo probe
RCS_VALIDATE_EVERY      = 5
RCS_PROBE_PATCHES       = [0, 13, 8, 21, 22, 17]   # exactly one per worker
                                                    # (w0..w5 under i % 6),
                                                    # center patch 13 on w1
RCS_FULL_SNAPSHOT_STEPS = [42, 82, 99]


# =====================================================================
# RCS LAYER: APPLY (RECORDING) + EXACT INVERSE
# =====================================================================
def apply_rcs_layer(sim, num_qubits, edges, depth, rng):
    """Apply `depth` layers of random u + iswap gates in-place.

    Returns the gate record needed by apply_rcs_inverse():
        [("u", q, theta, phi, lam), ("iswap", q1, q2), ...]
    in application order.
    """
    record = []
    for _ in range(depth):
        for q in range(num_qubits):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phi   = rng.uniform(0.0, 2.0 * math.pi)
            lam   = rng.uniform(0.0, 2.0 * math.pi)
            sim.u(q, theta, phi, lam)
            record.append(("u", q, theta, phi, lam))
        shuffled = list(edges)
        rng.shuffle(shuffled)
        used = set()
        for q1, q2 in shuffled:
            if q1 not in used and q2 not in used:
                sim.iswap(q1, q2)
                record.append(("iswap", q1, q2))
                used.add(q1); used.add(q2)
    return record


def make_rcs_inverse(adjiswap_native):
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

    def apply_rcs_inverse(sim, record):
        for g in reversed(record):
            if g[0] == "u":
                _, q, theta, phi, lam = g
                sim.u(q, -theta, -lam, -phi)
            else:
                _, q1, q2 = g
                _inv_iswap(sim, q1, q2)

    return apply_rcs_inverse


# =====================================================================
# WORKER INIT: ADD AFTER THE VECTORIZED r() DETECTION BLOCK
# (replaces the Rev 99 measure_shots LSB/MSB smoke test, which the echo
#  probe no longer needs)
# =====================================================================
        # --- ISWAP MIRROR SMOKE TEST + adjiswap AUTODETECT ---
        # Verifies that our chosen inverse actually mirrors iswap to
        # identity, catching both a missing adjiswap AND any convention
        # mismatch in the fallback decomposition.
        _adjiswap_native = hasattr(QrackSimulator(qubit_count=1), "adjiswap")
        _echo_probe = QrackSimulator(qubit_count=2,
                                     is_binary_decision_tree=False)
        _echo_probe.h(0)
        _echo_probe.u(1, 0.7, 1.1, 2.3)          # arbitrary non-trivial state
        _ket_before = np.asarray(_echo_probe.out_ket(), dtype=np.complex128)
        _echo_probe.iswap(0, 1)
        if _adjiswap_native:
            try:
                _echo_probe.adjiswap(0, 1)
            except Exception:
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


# =====================================================================
# WORKER LOOP: REPLACE THE ENTIRE "RCS VALIDATION" BLOCK
# =====================================================================
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
                        #    ~2 GB; the transient Python list from out_ket
                        #    is freed on conversion)
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

                        # Algorithmic time only: excludes cleanup + restore
                        lat_rcs = (time.perf_counter() - t0_rcs) * 1000.0

                        if is_snapshot:
                            print(f"[Worker {rank}] Snapshot echo patch "
                                  f"{patch_id}: F={echo_f:.6f} "
                                  f"({echo_n_gates} gates mirrored)",
                                  flush=True)

                    except Exception as e:
                        print(f"[Worker {rank}] Echo probe error "
                              f"(patch {patch_id}): {e}", file=sys.stderr)
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
                                print(f"[Worker {rank}] FATAL: Failed to "
                                      f"restore ket on patch {patch_id}: "
                                      f"{restore_e}", file=sys.stderr)
                            finally:
                                del pristine_ket
                                gc.collect()

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


# =====================================================================
# MASTER: rcs_cfg (drop "shots"), CSV SCHEMA, GATHER, STATUS LINE
# =====================================================================
# rcs_cfg: delete the "shots" entry.

# _init_files(): rcs_csv fieldnames become:
#     ["Step", "Anneal_Percent", "Patch", "RCS_Depth",
#      "Echo_Fidelity", "N_Gates", "Is_Snapshot"]

# _log_rcs(): records are (patch_id, echo_f, n_gates, is_snap); writerow:
#     {"Step": step, "Anneal_Percent": anneal, "Patch": patch_id,
#      "RCS_Depth": RCS_DEPTH, "Echo_Fidelity": echo_f,
#      "N_Gates": n_gates, "Is_Snapshot": int(is_snap)}

# Gather loop: replace the rcs_xeb collection with:
#
#                         if payload.get("echo_fidelity") is not None:
#                             is_snap = bool(payload.get("is_snapshot", False))
#                             any_snapshot = any_snapshot or is_snap
#                             rcs_records.append((
#                                 patch_id,
#                                 payload["echo_fidelity"],
#                                 payload["echo_n_gates"],
#                                 is_snap,
#                             ))

# Status line: replace the XEB/HOG segment with:
#
#                 if rcs_records:
#                     min_echo = min(r[1] for r in rcs_records)
#                     mean_echo = float(np.mean([r[1] for r in rcs_records]))
#                     n_rcs = len(rcs_records)
#                     status += (f" | Echo(min/mean): "
#                                f"{min_echo:.6f}/{mean_echo:.6f} "
#                                f"[{n_rcs} patch{'es' if n_rcs > 1 else ''}]")
#
# min is the headline number: one degraded patch is the signal, and the
# mean alone would wash it out across 27 patches on snapshot steps.
