# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_104.py
# Dashboard for MultiGpuHadronEngine Rev 104+
# (27-Qubit 3x3x3 Macroscopic Grid Annealing, 729 Qubits Total)
#
# PORTED FROM Rev 91 DASHBOARD - SCHEMA CHANGES:
# - RCS CSV columns: XEB_RCS/HOG_RCS -> Echo_Fidelity/N_Gates
# - probe_metric: "loschmidt_echo" or "observable_mae" (both handled)
# - No rcs_shots field in config (removed along with sampling path)
# - Snapshot panel: per-patch Echo_Fidelity heatmap replaces XEB cube
# - RCS time-series panel: Echo Fidelity trace replaces XEB/HOG dual-axis
# - N_Gates trace added as secondary axis on RCS panel
# - Echo value semantics:
#     Rev 100-103 (true Loschmidt): F in [0,1], 1.0 = perfect round-trip
#     Rev 104+ (observable MAE proxy): 1 - MAE, ~[0,1], 1.0 = zero drift
#   Dashboard handles both transparently via probe_metric in config.
# - SPACEBAR: save high-resolution PNG screenshot (300 DPI) to disk.
# - Snapshot steps marked with vertical gold bands on all time-series panels.
# - Full-cube Echo Fidelity heatmap rendered for snapshot steps.

import sys
import csv
import json
import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
DATA_FILE      = "macroscopic_lattice_states.npy"
CONFIG_FILE    = "lattice_config.json"
ENERGY_FILE    = "meanfield_ground_state_energy_curve_multi.csv"
RCS_FILE       = "echo_validation_multi.csv"    # Rev 104 filename
SAVE_FILE      = "cloverfield_dashboard.mp4"
SCREENSHOT_DPI = 300

PATCH_LX, PATCH_LY, PATCH_LZ = 3, 3, 3
GAP_X, GAP_Y, GAP_Z          = 1.5, 1.5, 1.5
QUIVER_STRIDE                 = 1

KIND_COLORS    = {"X_SEAM": "#e64550", "Y_SEAM": "#45b0e6", "Z_SEAM": "#e6b422"}
SNAPSHOT_COLOR = "#f5c518"   # gold band for snapshot steps
# ---------------------


def load_energy_data(num_steps, log_prefix=""):
    energies = {'Total': [], 'Bulk': [], 'Boundary': [], 'MinFid': []}
    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
                energies['MinFid'].append(float(row.get("Min_Unitary_Fidelity", 1.0)))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE}: {e}")
    for key in energies:
        if len(energies[key]) < num_steps:
            energies[key].extend([np.nan] * (num_steps - len(energies[key])))
        energies[key] = energies[key][:num_steps]
    return energies


def load_echo_data(num_steps, log_prefix=""):
    """Load Loschmidt echo data from Rev 104+ CSV schema.

    CSV columns: Step, Anneal_Percent, Patch, RCS_Depth,
                 Echo_Fidelity, N_Gates, Is_Snapshot

    Returns:
        echo_mean      : (num_steps,) forward-filled mean Echo_Fidelity
        echo_min       : (num_steps,) forward-filled min Echo_Fidelity
        ngates_mean    : (num_steps,) forward-filled mean N_Gates
        snapshot_echo  : dict {step -> np.ndarray shape (27,)} per-patch
                         Echo_Fidelity, only for Is_Snapshot=1 steps
        all_probed     : sorted list of all patch IDs seen in CSV
    """
    step_echo:    dict = {}   # step -> [echo_f, ...]
    step_ngates:  dict = {}   # step -> [n_gates, ...]
    step_patches: dict = {}
    snap_echo:    dict = {}   # step -> {patch_id: echo_f}

    try:
        with open(RCS_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step    = int(row["Step"])
                echo_f  = float(row["Echo_Fidelity"])
                n_gates = int(row["N_Gates"]) if row.get("N_Gates") else 0
                patch   = int(row["Patch"])
                is_snap = int(row.get("Is_Snapshot", 0))
                step_echo.setdefault(step, []).append(echo_f)
                step_ngates.setdefault(step, []).append(n_gates)
                step_patches.setdefault(step, set()).add(patch)
                if is_snap:
                    snap_echo.setdefault(step, {})[patch] = echo_f
    except Exception as e:
        print(f"{log_prefix}Notice: No echo data yet ({e}).")

    echo_mean  = np.full(num_steps, np.nan)
    echo_min   = np.full(num_steps, np.nan)
    ngates_mean = np.full(num_steps, np.nan)

    for step, vals in step_echo.items():
        if step < num_steps:
            echo_mean[step] = float(np.mean(vals))
            echo_min[step]  = float(np.min(vals))
    for step, vals in step_ngates.items():
        if step < num_steps:
            ngates_mean[step] = float(np.mean(vals))

    # Forward-fill so the traces are continuous
    echo_mean   = pd.Series(echo_mean).ffill().values
    echo_min    = pd.Series(echo_min).ffill().values
    ngates_mean = pd.Series(ngates_mean).ffill().values

    # Build per-patch echo arrays for snapshot steps
    snapshot_echo: dict = {}
    for step, patch_dict in snap_echo.items():
        arr = np.full(27, np.nan)
        for pid, ef in patch_dict.items():
            if 0 <= pid < 27:
                arr[pid] = ef
        snapshot_echo[step] = arr

    all_probed = sorted(
        set(p for patches in step_patches.values() for p in patches))

    return echo_mean, echo_min, ngates_mean, snapshot_echo, all_probed


def _safe_gradient(arr_with_nans):
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask    = ~np.isnan(arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        out[valid_indices] = np.gradient(arr[valid_indices], valid_indices)
    elif len(valid_indices) == 1:
        out[valid_indices[0]] = 0.0
    return out


def build_interfaces(grid_x, grid_y, grid_z):
    x_i1 = np.array([18 + y*3 + z for y in range(3) for z in range(3)])
    x_i2 = np.array([y*3 + z       for y in range(3) for z in range(3)])
    y_i1 = np.array([x*9 + 6 + z  for x in range(3) for z in range(3)])
    y_i2 = np.array([x*9 + z       for x in range(3) for z in range(3)])
    z_i1 = np.array([x*9 + y*3 + 2 for x in range(3) for y in range(3)])
    z_i2 = np.array([x*9 + y*3     for x in range(3) for y in range(3)])
    interfaces = []
    def patch_id(tx, ty, tz):
        return (tx * grid_y + ty) * grid_z + tz
    for tx in range(grid_x):
        for ty in range(grid_y):
            for tz in range(grid_z):
                p1 = patch_id(tx, ty, tz)
                if tx < grid_x - 1:
                    interfaces.append(
                        (p1, patch_id(tx+1, ty, tz), x_i1, x_i2, "X_SEAM"))
                if ty < grid_y - 1:
                    interfaces.append(
                        (p1, patch_id(tx, ty+1, tz), y_i1, y_i2, "Y_SEAM"))
                if tz < grid_z - 1:
                    interfaces.append(
                        (p1, patch_id(tx, ty, tz+1), z_i1, z_i2, "Z_SEAM"))
    return interfaces


def run_dashboard(mode="interactive"):
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    grid_x         = config.get("grid_x", 3)
    grid_y         = config.get("grid_y", 3)
    grid_z         = config.get("grid_z", 3)
    cfg_qpp        = config.get("qubits_per_patch", 27)
    cfg_rcs_prob   = config.get("rcs_probe_patches", None)
    cfg_rcs_dep    = config.get("rcs_depth", "?")
    cfg_rcs_ev     = config.get("rcs_validate_every", "?")
    cfg_snap_steps = set(config.get("rcs_full_snapshot_steps", []))
    # Rev 104 uses "observable_mae" or "loschmidt_echo"
    probe_metric   = config.get("probe_metric", "loschmidt_echo")

    # Human-readable metric label for axis/title use
    if probe_metric == "observable_mae":
        echo_label     = "Echo Obs-F"
        echo_label_long = "Observable MAE Proxy  (1 - mean|dP|)"
        echo_vmin, echo_vmax = 0.0, 1.0
    else:
        echo_label     = "Echo Fidelity"
        echo_label_long = "Loschmidt Echo Fidelity  F = |<psi|C_dag C|psi>|^2"
        echo_vmin, echo_vmax = 0.0, 1.0

    if cfg_rcs_prob is None:
        rcs_probe_label = "all patches"
    elif isinstance(cfg_rcs_prob, list) and len(cfg_rcs_prob) == 1:
        rcs_probe_label = f"patch {cfg_rcs_prob[0]} only"
    elif isinstance(cfg_rcs_prob, list):
        rcs_probe_label = f"patches {cfg_rcs_prob}"
    else:
        rcs_probe_label = str(cfg_rcs_prob)

    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches, qpp, _ = history.shape
    total_qubits = num_patches * qpp

    def patch_coords(p):
        tz   = p % grid_z
        rest = p // grid_z
        ty   = rest % grid_y
        tx   = rest // grid_y
        return tx, ty, tz

    energies = load_energy_data(num_steps, prefix)
    (echo_mean, echo_min,
     ngates_mean, snapshot_echo,
     probed_patches) = load_echo_data(num_steps, prefix)
    interfaces = build_interfaces(grid_x, grid_y, grid_z)

    snap_steps_in_csv = sorted(snapshot_echo.keys())
    if probed_patches:
        print(f"{prefix}Echo-probed patches: {probed_patches}")
    if snap_steps_in_csv:
        print(f"{prefix}Full-cube snapshot steps in CSV: {snap_steps_in_csv}")

    kinds_present = ["X_SEAM", "Y_SEAM", "Z_SEAM"]
    n_ifaces      = max(len(interfaces), 1)
    disagreements = np.zeros((num_steps, n_ifaces))
    iface_kind    = []
    if interfaces:
        hist_arr = np.asarray(history)
        for i, (p1, p2, i1, i2, kind) in enumerate(interfaces):
            diff = hist_arr[:, p1, i1, :] - hist_arr[:, p2, i2, :]
            disagreements[:, i] = np.mean(
                np.linalg.norm(diff, axis=2), axis=1)
            iface_kind.append(kind)
    iface_kind       = (np.array(iface_kind) if iface_kind
                        else np.array([], dtype=str))
    avg_disagreement = (np.mean(disagreements, axis=1)
                        if interfaces else np.zeros(num_steps))

    dE_dt   = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

    # 3-D layout
    pitch_x = PATCH_LX - 1 + GAP_X + 1
    pitch_y = PATCH_LY - 1 + GAP_Y + 1
    pitch_z = PATCH_LZ - 1 + GAP_Z + 1
    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        tx, ty, tz = patch_coords(p)
        for q in range(qpp):
            qx = q // 9; qy = (q % 9) // 3; qz = q % 3
            global_X.append(tx * pitch_x + float(qx))
            global_Y.append(ty * pitch_y + float(qy))
            global_Z.append(tz * pitch_z + float(qz))
    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)
    x_max = (grid_x - 1) * pitch_x + PATCH_LX - 1
    y_max = (grid_y - 1) * pitch_y + PATCH_LY - 1
    z_max = (grid_z - 1) * pitch_z + PATCH_LZ - 1

    stride   = max(1, int(QUIVER_STRIDE))
    draw_idx = np.arange(0, num_patches * qpp, stride)
    qX, qY, qZ = global_X[draw_idx], global_Y[draw_idx], global_Z[draw_idx]

    # ----------------------------------------------------------------
    # Figure layout (identical column structure to Rev 91):
    # Left  : 3D quiver (wide)
    # Middle: 6-row time-series stack
    #   0  Energy + Min Unitary Fidelity
    #   1  Echo Fidelity time series (mean + min) + N_Gates secondary axis
    #   2  Derivatives
    #   3  Heatmap <X>
    #   4  Heatmap <Y>
    #   5  Heatmap <Z>
    # Right : 3x3x3 Echo Fidelity cube (snapshot steps only)
    # ----------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1, 0.7], wspace=0.12)

    ax3d     = fig.add_subplot(gs[0], projection='3d')
    gs_mid   = gridspec.GridSpecFromSubplotSpec(
        6, 1, subplot_spec=gs[1], hspace=0.80)

    ax_energy  = fig.add_subplot(gs_mid[0])
    ax_echo_ts = fig.add_subplot(gs_mid[1])
    ax_deriv   = fig.add_subplot(gs_mid[2])
    ax_hmap_x  = fig.add_subplot(gs_mid[3])
    ax_hmap_y  = fig.add_subplot(gs_mid[4])
    ax_hmap_z  = fig.add_subplot(gs_mid[5])

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12)

    # ----------------------------------------------------------------
    # 3x3x3 Echo Fidelity cube (right column)
    # One tiny axis per patch; Z=2 at top row, Z=0 at bottom.
    # Each row of 3 mini-axes = one (X, Z-slice) pair, columns = Y index.
    # ----------------------------------------------------------------
    gs2_bbox = gs[2].get_position(fig)
    gx0, gy0, gx1, gy1 = (gs2_bbox.x0, gs2_bbox.y0,
                            gs2_bbox.x1, gs2_bbox.y1)
    cell_w = (gx1 - gx0) / 3.0
    cell_h = (gy1 - gy0) / 9.2

    cube_ax_list = []
    for iz in range(3):
        layer = []
        for ix in range(3):
            row_axes = []
            for iy in range(3):
                disp_iz   = 2 - iz
                slice_row = disp_iz * 3 + ix
                left  = gx0 + iy * cell_w
                bot   = gy1 - (slice_row + 1) * cell_h
                ax    = fig.add_axes([left, bot, cell_w * 0.90, cell_h * 0.85])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor('#111111')
                for sp in ax.spines.values():
                    sp.set_edgecolor('#333333'); sp.set_linewidth(0.5)
                row_axes.append(ax)
            layer.append(row_axes)
        cube_ax_list.append(layer)

    cube_patch_axes = {}
    for iz in range(3):
        for ix in range(3):
            for iy in range(3):
                pid = (ix * grid_y + iy) * grid_z + iz
                cube_patch_axes[pid] = cube_ax_list[iz][ix][iy]

    # Z-layer labels
    for iz in range(3):
        disp_iz = 2 - iz
        mid_row = disp_iz * 3 + 1
        label_y = gy1 - (mid_row + 0.5) * cell_h
        fig.text(gx0 - 0.005, label_y, f"Z={iz}",
                 va='center', ha='right', fontsize=6,
                 color='#aaaaaa', rotation=90)

    # Column headers (Y index)
    for iy in range(3):
        fig.text(gx0 + (iy + 0.45) * cell_w, gy1 + 0.005,
                 f"Y={iy}", ha='center', va='bottom',
                 fontsize=6, color='#aaaaaa')

    # Row labels (X index)
    for ix in range(3):
        for iz in range(3):
            disp_iz = 2 - iz
            row_idx = disp_iz * 3 + ix
            label_y = gy1 - (row_idx + 0.5) * cell_h
            fig.text(gx0 - 0.012, label_y, f"X={ix}",
                     va='center', ha='right', fontsize=5,
                     color='#777777')

    # Echo fidelity colormap: blue (0) -> dark (0.5) -> green (1.0)
    echo_cmap = mcolors.LinearSegmentedColormap.from_list(
        "echo_cube",
        [(0.60, 0.05, 0.05, 1.0),   # red: badly degraded echo (F ~ 0)
         (0.10, 0.10, 0.10, 1.0),   # near-black: mid-range
         (0.00, 0.85, 0.40, 1.0),   # green: good echo (F ~ 1)
         ])
    echo_cube_norm = mcolors.Normalize(vmin=echo_vmin, vmax=echo_vmax)

    _nan_data = np.array([[np.nan]])
    cube_imgs = {}
    for pid, ax in cube_patch_axes.items():
        img = ax.imshow(_nan_data, cmap=echo_cmap, norm=echo_cube_norm,
                        aspect='auto', interpolation='nearest')
        cube_imgs[pid] = img

    cube_title_obj = fig.text(
        (gx0 + gx1) / 2, gy1 + 0.022,
        "Echo Cube (no snapshot yet)",
        ha='center', va='bottom', fontsize=8,
        color='#cccccc', fontweight='bold')

    # ----------------------------------------------------------------
    # 3D quiver
    # ----------------------------------------------------------------
    spin_norm   = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vector_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ghost_vectors",
        [(0.15, 0.35, 0.85, 0.85),
         (0.85, 0.85, 0.85, 0.45),
         (0.85, 0.15, 0.25, 0.85)])

    def _quiver_colors(w_flat):
        return vector_cmap(spin_norm(w_flat))

    def get_vector_data(step_idx):
        return (history[step_idx, :, :, 0].ravel()[draw_idx],
                history[step_idx, :, :, 1].ravel()[draw_idx],
                history[step_idx, :, :, 2].ravel()[draw_idx])

    U, V, W    = get_vector_data(0)
    quiver_obj = [ax3d.quiver(
        qX, qY, qZ, U, V, W,
        length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3)]

    ax_cbar = fig.add_axes([0.02, 0.25, 0.012, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Bloch Z Component', fontsize=9)

    energy_text   = ax3d.text2D(0.04, 0.96, "", transform=ax3d.transAxes,
                                color='lightgreen', fontsize=12, fontweight='bold')
    echo_mean_txt = ax3d.text2D(0.04, 0.92, "", transform=ax3d.transAxes,
                                color='cyan', fontsize=11, fontweight='bold')
    echo_min_txt  = ax3d.text2D(0.04, 0.88, "", transform=ax3d.transAxes,
                                color='#88ddff', fontsize=10)
    rcs_src_text  = ax3d.text2D(
        0.04, 0.84, f"[probe: {rcs_probe_label}]",
        transform=ax3d.transAxes, color='#aaaaaa', fontsize=8)
    snap_badge    = ax3d.text2D(
        0.04, 0.80, "",
        transform=ax3d.transAxes,
        color=SNAPSHOT_COLOR, fontsize=10, fontweight='bold')

    def _set_3d_title(frame, is_snap):
        tag = "  *** FULL-CUBE SNAPSHOT ***" if is_snap else ""
        ax3d.set_title(
            f"Volumetric Lattice Annealing Rev 104  "
            f"({grid_x}x{grid_y}x{grid_z} | "
            f"{num_patches} patches | {total_qubits} qubits)\n"
            f"Trotter Step: {frame}/{num_steps-1}{tag}",
            fontsize=12, pad=10)

    _set_3d_title(0, False)
    ax3d.set_xlim(-0.5, x_max + 0.5)
    ax3d.set_ylim(-0.5, y_max + 0.5)
    ax3d.set_zlim(-0.5, z_max + 0.5)
    try:
        ax3d.set_box_aspect((x_max + 1.0, y_max + 1.0, z_max + 1.0))
    except AttributeError:
        pass
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor('none')
        except AttributeError:
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis.line.set_linewidth(0)
        axis.set_ticklabels([]); axis.set_ticks([])
    ax3d.grid(False); ax3d.set_axis_off()

    # ----------------------------------------------------------------
    # Energy panel -- now also shows Min_Unitary_Fidelity on secondary axis
    # ----------------------------------------------------------------
    ax_energy.plot(energies['Total'],    label='Total', color='lightgreen')
    ax_energy.plot(energies['Bulk'],     label='Bulk',  color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Bndry', color='orange')
    ax_energy_r = ax_energy.twinx()
    ax_energy_r.plot(energies['MinFid'], label='Min Fid',
                     color='#ffaa44', linewidth=0.8, linestyle=':')
    ax_energy_r.set_ylabel("Min Unitary Fid.", fontsize=7, color='#ffaa44')
    ax_energy_r.set_ylim(0.0, 1.05)
    for ss in cfg_snap_steps:
        if ss < num_steps:
            ax_energy.axvline(ss, color=SNAPSHOT_COLOR,
                              linewidth=0.8, alpha=0.5, linestyle=':')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.set_ylabel("Energy (a.u.)", fontsize=8)
    lns_e = (ax_energy.get_lines() + ax_energy_r.get_lines())
    ax_energy.legend(lns_e, [l.get_label() for l in lns_e],
                     fontsize=6, loc='upper left', ncol=4)
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ----------------------------------------------------------------
    # Echo time-series panel
    # Primary axis : Echo_Fidelity mean (cyan) + min (dashed cyan)
    # Secondary axis: N_Gates mean (dim magenta) -- sanity check that
    #                 gate counts are stable across the run
    # ----------------------------------------------------------------
    has_echo_data = not np.all(np.isnan(echo_mean))
    ax_echo_r = ax_echo_ts.twinx()

    if has_echo_data:
        ax_echo_ts.plot(echo_mean, color='cyan', linewidth=1.4,
                        label=f'{echo_label} (mean)')
        ax_echo_ts.plot(echo_min,  color='cyan', linewidth=0.8,
                        linestyle='--', alpha=0.6, label=f'{echo_label} (min)')
        ax_echo_ts.axhline(1.0, color='cyan',  linewidth=0.5,
                           alpha=0.3, linestyle=':')
        ax_echo_ts.axhline(0.0, color='white', linewidth=0.5,
                           alpha=0.3, linestyle=':')
        ax_echo_r.plot(ngates_mean, color='#cc88ff', linewidth=0.7,
                       linestyle=':', label='N_Gates (mean)')
        for ss in cfg_snap_steps:
            if ss < num_steps:
                ax_echo_ts.axvline(ss, color=SNAPSHOT_COLOR,
                                   linewidth=1.2, alpha=0.7, linestyle=':')
        lns_rcs = ax_echo_ts.get_lines() + ax_echo_r.get_lines()
        ax_echo_ts.legend(lns_rcs, [l.get_label() for l in lns_rcs],
                          fontsize=6, loc='lower left')
    else:
        ax_echo_ts.set_title(f"{echo_label} (awaiting data)", fontsize=9)

    ax_echo_ts.set_title(
        f"{echo_label_long}\n"
        f"[{rcs_probe_label}, depth={cfg_rcs_dep}, every {cfg_rcs_ev} steps]",
        fontsize=8)
    ax_echo_ts.set_ylabel(echo_label, fontsize=8, color='cyan')
    ax_echo_ts.set_ylim(-0.05, 1.10)
    ax_echo_r.set_ylabel("N_Gates", fontsize=7, color='#cc88ff')
    ax_echo_ts.grid(True, alpha=0.2)
    vline_rcs = ax_echo_ts.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ----------------------------------------------------------------
    # Derivative panel
    # ----------------------------------------------------------------
    l1 = ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("dE/dt", fontsize=8)
    ax_deriv_r = ax_deriv.twinx()
    l2 = ax_deriv_r.plot(dRes_dt, label='d||ds||/dt', color='crimson')
    ax_deriv_r.set_ylabel("d||ds||/dt", fontsize=8)
    lns2 = l1 + l2
    ax_deriv.legend(lns2, [l.get_label() for l in lns2],
                    loc='upper left', fontsize=7)
    ax_deriv.set_title("Convergence Rate", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    for ss in cfg_snap_steps:
        if ss < num_steps:
            ax_deriv.axvline(ss, color=SNAPSHOT_COLOR,
                             linewidth=0.8, alpha=0.5, linestyle=':')
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ----------------------------------------------------------------
    # Heatmap panels
    # ----------------------------------------------------------------
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap",
        [(0.15, 0.35, 0.85, 1.0),
         (0.10, 0.10, 0.10, 1.0),
         (0.85, 0.15, 0.25, 1.0)])
    y_ticks  = np.arange(num_patches)
    y_labels = [f"P{p}" for p in range(num_patches)]

    def _init_heatmap(ax, data, cmap, norm, label,
                      show_ylabel=False, show_xlabel=False):
        img = ax.imshow(data, cmap=cmap, norm=norm,
                        aspect='auto', interpolation='nearest')
        for r in range(1, num_patches):
            ax.axhline(r - 0.5, color='white', linewidth=0.3, alpha=0.35)
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels, fontsize=4)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel(f"Qubit (0-{qpp-1})", fontsize=7)
            ax.tick_params(axis='x', which='major', labelsize=6)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                            heatmap_cmap, spin_norm, "<X>", show_ylabel=True)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                            heatmap_cmap, spin_norm, "<Y>", show_ylabel=True)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                            heatmap_cmap, spin_norm, "<Z>",
                            show_ylabel=True, show_xlabel=True)

    # ----------------------------------------------------------------
    # Controls
    # ----------------------------------------------------------------
    ax_slider = fig.add_axes([0.12, 0.04, 0.55, 0.02])
    slider    = Slider(ax=ax_slider, label='Trotter Step',
                       valmin=0, valmax=num_steps - 1,
                       valinit=0, valstep=1, color='#4a90e2')
    ax_play   = fig.add_axes([0.70, 0.028, 0.07, 0.04])
    btn_play  = Button(ax_play, 'Pause',
                       color='#333333', hovercolor='#555555')
    is_playing  = True
    _cur_frame  = [0]

    # ----------------------------------------------------------------
    # Echo cube updater
    # ----------------------------------------------------------------
    _last_snap_step = [None]

    def _update_echo_cube(frame):
        """Find the most recent snapshot at or before this frame and render."""
        best = None
        for ss in sorted(snapshot_echo.keys()):
            if ss <= frame:
                best = ss
        if best is None:
            for pid, img in cube_imgs.items():
                img.set_data(_nan_data)
            cube_title_obj.set_text("Echo Cube (no snapshot yet)")
            _last_snap_step[0] = None
            return
        if best == _last_snap_step[0]:
            return
        _last_snap_step[0] = best
        echo_arr = snapshot_echo[best]   # shape (27,)
        for pid, img in cube_imgs.items():
            val = echo_arr[pid] if pid < len(echo_arr) else np.nan
            img.set_data(np.array([[val]]))
            img.set_clim(echo_vmin, echo_vmax)
            for sp in cube_patch_axes[pid].spines.values():
                sp.set_edgecolor(SNAPSHOT_COLOR if not np.isnan(val)
                                 else '#333333')
                sp.set_linewidth(1.0 if not np.isnan(val) else 0.5)
        mean_ef = float(np.nanmean(echo_arr))
        min_ef  = float(np.nanmin(echo_arr))
        cube_title_obj.set_text(
            f"Echo Cube -- step {best}  "
            f"mean={mean_ef:.4f}  min={min_ef:.4f}")

    # ----------------------------------------------------------------
    # Screenshot
    # ----------------------------------------------------------------
    def _save_screenshot(frame):
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"screenshot_step{frame:03d}_{ts}.png"
        fig.savefig(fname, dpi=SCREENSHOT_DPI, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        print(f"[Screenshot] Saved {fname} ({SCREENSHOT_DPI} DPI)", flush=True)

    # ----------------------------------------------------------------
    # Scroll zoom
    # ----------------------------------------------------------------
    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        if not hasattr(ax3d, 'custom_zoom'):
            ax3d.custom_zoom = 1.0
        ax3d.custom_zoom *= 0.9 if event.button == 'down' else 1.1
        try:
            ca = ax3d.get_box_aspect()
            if ca is None:
                ca = (x_max + 1.0, y_max + 1.0, z_max + 1.0)
            ax3d.set_box_aspect(ca, zoom=ax3d.custom_zoom)
        except TypeError:
            ax3d.dist *= 1.1 if event.button == 'down' else 0.9
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key(event):
        if event.key == ' ':
            _save_screenshot(_cur_frame[0])

    fig.canvas.mpl_connect('key_press_event', on_key)

    _from_animation = [False]

    # ----------------------------------------------------------------
    # Per-frame update
    # ----------------------------------------------------------------
    def update(frame):
        frame = int(frame)
        _cur_frame[0] = frame
        is_snap = frame in snapshot_echo

        U, V, W = get_vector_data(frame)
        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            qX, qY, qZ, U, V, W,
            length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3)

        _set_3d_title(frame, is_snap)

        # Energy overlay
        e_list = energies['Total']
        if e_list and frame < len(e_list):
            e_val = e_list[frame]
            energy_text.set_text(
                f"Total Energy: {e_val:.4f}" if not np.isnan(e_val) else "")
        else:
            energy_text.set_text("")

        # Echo overlays
        em_val  = echo_mean[frame]  if frame < len(echo_mean)  else np.nan
        emin_val = echo_min[frame]  if frame < len(echo_min)   else np.nan
        ng_val  = ngates_mean[frame] if frame < len(ngates_mean) else np.nan

        echo_mean_txt.set_text(
            f"{echo_label}: {em_val:.6f}" if not np.isnan(em_val) else "")
        echo_min_txt.set_text(
            f"Echo min: {emin_val:.6f}  |  Gates: {ng_val:.0f}"
            if (not np.isnan(emin_val) and not np.isnan(ng_val)) else "")
        snap_badge.set_text("[ FULL-CUBE SNAPSHOT ]" if is_snap else "")

        vline_e.set_xdata([frame])
        vline_rcs.set_xdata([frame])
        vline_deriv.set_xdata([frame])

        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        _update_echo_cube(frame)

        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return (quiver_obj[0], energy_text,
                echo_mean_txt, echo_min_txt, snap_badge,
                hmap_x, hmap_y, hmap_z)

    def _animation_update(frame):
        _from_animation[0] = True
        result = update(frame)
        _from_animation[0] = False
        return result

    def on_slider_update(val):
        _from_animation[0] = False
        update(val)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_update)

    def toggle_play(event):
        nonlocal is_playing
        if is_playing:
            ani.event_source.stop()
            btn_play.label.set_text('Play')
        else:
            ani.event_source.start()
            btn_play.label.set_text('Pause')
        is_playing = not is_playing
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    ani = animation.FuncAnimation(
        fig, _animation_update, frames=num_steps, interval=150, blit=False)

    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save. ffmpeg installed? Error: {e}")
    else:
        print(f"{prefix}Opening GUI.  SPACEBAR = screenshot ({SCREENSHOT_DPI} DPI)")
        plt.show()


def main():
    mp.set_start_method('spawn', force=True)
    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()
    run_dashboard(mode="interactive")
    if render_process.is_alive():
        print("\nInteractive viewer closed. "
              "Waiting for background 4K render...")
        render_process.join()
    print("All processes terminated.")


if __name__ == "__main__":
    main()
