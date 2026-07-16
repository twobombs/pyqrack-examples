# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_v7.py
#
# Changes vs v6:
#   - QFT CSV INTEGRATION: Loads separable_qft_samples_multi.csv and computes
#     two new per-step scalar metrics:
#       * Mean bitstring Shannon entropy across patches (probe of spectral spread)
#       * Mean pairwise inter-patch Hamming distance (probe of patch-to-patch
#         spectral divergence; a proxy for spatial correlations in the QFT output)
#   - FIDELITY PANEL: Min_Unitary_Fidelity (already in energy CSV, previously
#     ignored) is now plotted, sharing a row with mean QFT latency as twinx.
#   - QFT ANALYTICS PANEL: entropy and mean Hamming distance on a new row,
#     twinx layout.
#   - RIGHT COLUMN: expanded from 6 rows to 8 rows to accommodate the two new
#     panels above the heatmaps.
#   - ANNEAL PERCENT: read from energy CSV and used to annotate the energy panel
#     x-axis so step numbers carry physical meaning.
#
# Changes vs v5 (carried through from v6):
#   - Removed the fourth heatmap panel (total polarization sum).
#   - GridSpec right column reduced to 6 rows.
#   - Enabled y-axis (XYZ coordinates) labels on hmap_y and hmap_z.
#   - Shifted x-axis label (Local Qubit Index) to hmap_z since it is now the bottom panel.
#
# FIXED (Rev 85 alignment, carried through):
#   - ENERGY_FILE: corrected filename to "meanfield_ground_state_energy_curve_multi.csv"
#   - CSV column names aligned to MultiGpuHadronEngine._log_csvs() fieldnames.

import sys
import csv
import json
import math
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
DATA_FILE     = "macroscopic_lattice_states.npy"
CONFIG_FILE   = "lattice_config.json"
ENERGY_FILE   = "meanfield_ground_state_energy_curve_multi.csv"
PROFILES_FILE = "boundary_profiles_multi.csv"
QFT_FILE      = "separable_qft_samples_multi.csv"
SAVE_FILE     = "macroscopic_lattice_dash.mp4"
# ---------------------


def _bitstring_entropy(s):
    """Shannon entropy (bits) of a bitstring. Returns NaN for empty strings."""
    n = len(s)
    if n == 0:
        return float('nan')
    ones = s.count('1')
    zeros = n - ones
    h = 0.0
    for count in (ones, zeros):
        if count > 0:
            p = count / n
            h -= p * math.log2(p)
    return h


def _mean_pairwise_hamming(bitstrings):
    """
    Mean pairwise Hamming distance across a list of equal-length bitstrings.
    Uses a vectorized bit-count approach: converts each string to a numpy
    uint8 array of 0/1, stacks into a matrix, then computes all pairwise
    XOR sums in one matrix operation.  O(P^2 * n) but P=27 and n=27 so
    the inner product is tiny (27x27 matrix multiply).
    Returns NaN if fewer than 2 valid bitstrings are present.
    """
    valid = [s for s in bitstrings if s and len(s) > 0]
    if len(valid) < 2:
        return float('nan')
    n = len(valid[0])
    mat = np.array([[int(c) for c in s[:n]] for s in valid], dtype=np.uint8)
    # pairwise XOR sum via: sum_ij ||row_i - row_j||_1
    # = 2 * sum_k [ p_k * (1 - p_k) ] * P  where p_k = column mean
    # This is exact for binary {0,1} entries.
    col_sums = mat.sum(axis=0).astype(float)   # shape (n,)
    P = len(valid)
    # Number of (i,j) pairs with i<j where bit k differs = col_sums[k] * (P - col_sums[k])
    pair_diffs = col_sums * (P - col_sums)     # shape (n,)
    total_hamming = pair_diffs.sum()
    num_pairs = P * (P - 1) / 2
    return float(total_hamming / num_pairs)


def load_analytics_data(num_steps, log_prefix=""):
    """
    Loads all CSV outputs from the Rev 89.4 engine.

    Returns
    -------
    energies   : dict with keys Total, Bulk, Boundary, Fidelity -- each a list
                 of length num_steps (NaN-padded if run was aborted early).
    anneal_pct : list of length num_steps (anneal % 0-100, NaN-padded).
    profiles   : dict[step][patch][face] -> np.array([X_mean, Y_mean, Z_mean])
    qft_entropy : np.array shape (num_steps,) -- mean bitstring Shannon entropy
                  across patches at each step (NaN where no QFT data).
    qft_hamming : np.array shape (num_steps,) -- mean pairwise inter-patch
                  Hamming distance at each step (NaN where <2 patches).
    qft_lat_ms  : np.array shape (num_steps,) -- mean QFT sampler latency in ms.
    """
    energies = {'Total': [], 'Bulk': [], 'Boundary': [], 'Fidelity': []}
    anneal_pct = []
    profiles = {}

    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
                energies['Fidelity'].append(float(row["Min_Unitary_Fidelity"]))
                anneal_pct.append(float(row["Anneal_Percent"]))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE}: {e}")

    # NaN-pad all series to num_steps
    for key in energies:
        arr = energies[key]
        if len(arr) < num_steps:
            arr.extend([float('nan')] * (num_steps - len(arr)))
        energies[key] = arr[:num_steps]

    if len(anneal_pct) < num_steps:
        anneal_pct.extend([float('nan')] * (num_steps - len(anneal_pct)))
    anneal_pct = anneal_pct[:num_steps]

    try:
        with open(PROFILES_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = int(row["Step"])
                p = int(row["Patch"])
                f_name = row["Face"]
                if s not in profiles:
                    profiles[s] = {}
                if p not in profiles[s]:
                    profiles[s][p] = {}
                profiles[s][p][f_name] = np.array([
                    float(row["X_mean"]), float(row["Y_mean"]), float(row["Z_mean"])
                ])
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {PROFILES_FILE}: {e}")

    # QFT CSV: collect per-step lists of bitstrings and latencies
    qft_step_bits = {}   # step -> list of bitstrings
    qft_step_lats = {}   # step -> list of latencies (ms)

    try:
        with open(QFT_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = int(row["Step"])
                bits = row["QFT_BitString"].strip()
                lat  = float(row["QFT_ms"])
                if s not in qft_step_bits:
                    qft_step_bits[s] = []
                    qft_step_lats[s] = []
                qft_step_bits[s].append(bits)
                qft_step_lats[s].append(lat)
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {QFT_FILE}: {e}")

    qft_entropy  = np.full(num_steps, float('nan'))
    qft_hamming  = np.full(num_steps, float('nan'))
    qft_lat_ms   = np.full(num_steps, float('nan'))

    for s in range(num_steps):
        if s in qft_step_bits and qft_step_bits[s]:
            bits_list = qft_step_bits[s]
            lats_list = qft_step_lats[s]
            entropies = [_bitstring_entropy(b) for b in bits_list if b]
            if entropies:
                qft_entropy[s] = float(np.nanmean(entropies))
            qft_hamming[s] = _mean_pairwise_hamming(bits_list)
            if lats_list:
                qft_lat_ms[s] = float(np.mean(lats_list))

    return energies, anneal_pct, profiles, qft_entropy, qft_hamming, qft_lat_ms


def _safe_gradient(arr_with_nans):
    """
    Compute np.gradient only over the leading valid (non-NaN) slice.
    Returns a full-length array with NaN in positions where input was NaN.
    Prevents gradient NaN bleed at the boundary caused by trailing NaN padding.
    """
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_count = int(np.sum(valid_mask))
    if valid_count > 1:
        out[:valid_count] = np.gradient(arr[:valid_count])
    elif valid_count == 1:
        out[:1] = 0.0
    return out


def run_dashboard(mode="interactive"):
    """
    Main visualization routine.
    mode: "interactive" (opens UI) or "save" (renders to disk headless).
    """
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    # 1. Grid configuration
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            grid_x = config.get("grid_x", 1)
            grid_y = config.get("grid_y", 1)
            grid_z = config.get("grid_z", 1)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    # 2. State history
    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches = history.shape[0], history.shape[1]
    total_qubits = num_patches * 27

    # 3. Analytics
    (energies, anneal_pct, profiles,
     qft_entropy, qft_hamming, qft_lat_ms) = load_analytics_data(num_steps, prefix)

    patch_coords = {}
    coord_to_patch = {}
    idx = 0
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                patch_coords[idx] = (x, y, z)
                coord_to_patch[(x, y, z)] = idx
                idx += 1

    interfaces = []
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                p1 = coord_to_patch[(x, y, z)]
                if x < grid_x - 1:
                    p2 = coord_to_patch[(x + 1, y, z)]
                    interfaces.append((p1, p2, "+X", "-X", x * 3 + 2.5, y * 3 + 1.0, z * 3 + 1.0))
                if y < grid_y - 1:
                    p2 = coord_to_patch[(x, y + 1, z)]
                    interfaces.append((p1, p2, "+Y", "-Y", x * 3 + 1.0, y * 3 + 2.5, z * 3 + 1.0))
                if z < grid_z - 1:
                    p2 = coord_to_patch[(x, y, z + 1)]
                    interfaces.append((p1, p2, "+Z", "-Z", x * 3 + 1.0, y * 3 + 1.0, z * 3 + 2.5))

    disagreements = np.zeros((num_steps, max(len(interfaces), 1)))
    for s in range(num_steps):
        if s in profiles:
            for i, (p1, p2, f1, f2, _, _, _) in enumerate(interfaces):
                try:
                    v1 = profiles[s][p1][f1]
                    v2 = profiles[s][p2][f2]
                    disagreements[s, i] = np.linalg.norm(v1 - v2)
                except KeyError:
                    pass

    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_steps)

    dE_dt   = _safe_gradient(energies['Total'])
    dRes_dt = _safe_gradient(avg_disagreement)

    # 4. Global 3D qubit coordinates
    q_coords = {}
    for x in range(3):
        for y in range(3):
            for z in range(3):
                q_coords[x * 9 + y * 3 + z] = (x, y, z)

    global_X, global_Y, global_Z = [], [], []
    for p in range(num_patches):
        px, py, pz = patch_coords[p]
        for q in range(27):
            qx, qy, qz = q_coords[q]
            global_X.append(px * 3 + qx)
            global_Y.append(py * 3 + qy)
            global_Z.append(pz * 3 + qz)

    global_X = np.array(global_X)
    global_Y = np.array(global_Y)
    global_Z = np.array(global_Z)

    # 5. Layout: 8 rows in right column
    #   0: energy components
    #   1: inter-patch boundary coupling error
    #   2: derivatives (dE/dt + d_disagreement/dt)
    #   3: fidelity + QFT mean latency  [NEW]
    #   4: QFT entropy + Hamming        [NEW]
    #   5: heatmap <X>
    #   6: heatmap <Y>
    #   7: heatmap <Z>
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    gs_right = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=gs[1], hspace=1.10)
    ax_energy   = fig.add_subplot(gs_right[0])
    ax_dis      = fig.add_subplot(gs_right[1])
    ax_deriv    = fig.add_subplot(gs_right[2])
    ax_fid      = fig.add_subplot(gs_right[3])   # NEW: fidelity + QFT latency
    ax_qft      = fig.add_subplot(gs_right[4])   # NEW: entropy + Hamming
    ax_hmap_x   = fig.add_subplot(gs_right[5])
    ax_hmap_y   = fig.add_subplot(gs_right[6])
    ax_hmap_z   = fig.add_subplot(gs_right[7])

    # --- Shared colour infrastructure ---
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vector_colors = [
        (0.15, 0.35, 0.85, 0.85),
        (0.85, 0.85, 0.85, 0.45),
        (0.85, 0.15, 0.25, 0.85),
    ]
    vector_cmap = mcolors.LinearSegmentedColormap.from_list("ghost_vectors", vector_colors)

    def get_vector_data(step_idx):
        return (
            history[step_idx, :, :, 0].flatten(),
            history[step_idx, :, :, 1].flatten(),
            history[step_idx, :, :, 2].flatten(),
        )

    def _quiver_colors(w_flat):
        return vector_cmap(spin_norm(w_flat))

    U, V, W = get_vector_data(0)
    quiver_obj = [ax3d.quiver(
        global_X, global_Y, global_Z, U, V, W,
        length=0.75, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('<sigma_Z> Bloch Component (Mean-field)', fontsize=10)

    energy_text = ax3d.text2D(
        0.04, 0.96, "", transform=ax3d.transAxes,
        color='lightgreen', fontsize=12, fontweight='bold'
    )

    ax3d.set_title(
        f"Cloverfield QFT: 3D TFIM Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
        f"{num_patches} Patches | {total_qubits} Qubits)\nTrotter Step: 0/{num_steps-1}",
        fontsize=14, pad=10
    )
    ax3d.set_xlim(-0.5, grid_x * 3 - 0.5)
    ax3d.set_ylim(-0.5, grid_y * 3 - 0.5)
    ax3d.set_zlim(-0.5, grid_z * 3 - 0.5)
    try:
        ax3d.set_box_aspect((grid_x, grid_y, max(1, grid_z)))
    except AttributeError:
        pass

    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # --- Panel 0: Energy components ---
    ax_energy.plot(energies['Total'],    label='Total',    color='lightgreen')
    ax_energy.plot(energies['Bulk'],     label='Bulk',     color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary', color='orange')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.legend(fontsize=7, loc='upper left')
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # --- Panel 1: Inter-patch boundary coupling error ---
    vline_d = None
    if interfaces:
        ax_dis.plot(avg_disagreement, color='crimson',
                    label='Mean ||d<sigma>|| across interfaces')
        ax_dis.set_title("Inter-patch Boundary Coupling Error\n"
                         "mean_ifaces( ||<sigma>+face - <sigma>-face|| )", fontsize=9)
        ax_dis.legend(fontsize=7, loc='upper left')
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # --- Panel 2: Derivatives ---
    ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("Energy Delta", fontsize=7)
    ax_deriv.legend(loc='upper left', fontsize=7)
    ax_deriv_r = ax_deriv.twinx()
    ax_deriv_r.plot(dRes_dt, label='d||d<sigma>||/dt', color='crimson')
    ax_deriv_r.set_ylabel("d<sigma> Coupling dt", fontsize=7)
    ax_deriv_r.legend(loc='upper right', fontsize=7)
    ax_deriv.set_title("Derivatives (Convergence Rate)", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # --- Panel 3: Fidelity + QFT mean latency (NEW) ---
    # Fidelity = Min_Unitary_Fidelity across all patches at each step.
    # QFT latency = mean QFT sampler wall time in ms (right axis).
    # Physical interpretation: fidelity degradation indicates ACE approximation
    # budget is being consumed; QFT latency tracks sampler performance.
    fid_arr = np.array(energies['Fidelity'], dtype=float)
    ax_fid.plot(fid_arr, color='gold', label='Min ACE Fidelity (across patches)', linewidth=1.2)
    ax_fid.set_ylim(bottom=0.0, top=1.05)
    ax_fid.set_ylabel("ACE Fidelity", fontsize=7, color='gold')
    ax_fid.tick_params(axis='y', labelcolor='gold', labelsize=6)
    ax_fid.set_title("ACE Unitary Fidelity + QFT Sampler Latency", fontsize=10)
    ax_fid.legend(loc='lower left', fontsize=7)
    ax_fid.grid(True, alpha=0.2)
    vline_fid = ax_fid.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    ax_fid_r = ax_fid.twinx()
    # Only plot latency where we have QFT data
    qft_lat_valid = np.where(np.isfinite(qft_lat_ms), qft_lat_ms, np.nan)
    ax_fid_r.plot(qft_lat_valid, color='mediumpurple', label='Mean QFT ms',
                  linewidth=1.0, alpha=0.8)
    ax_fid_r.set_ylabel("QFT latency (ms)", fontsize=7, color='mediumpurple')
    ax_fid_r.tick_params(axis='y', labelcolor='mediumpurple', labelsize=6)
    ax_fid_r.legend(loc='upper right', fontsize=7)

    # --- Panel 4: QFT bitstring entropy + mean pairwise Hamming distance (NEW) ---
    # Shannon entropy of per-patch bitstrings, averaged across patches.
    # Max = 1.0 bit (uniform distribution over 0/1 at each position).
    # Early annealing: patches near |+> -> high equatorial spread -> high entropy.
    # Late annealing: patches polarize toward |Z> -> phase feedback shrinks -> lower entropy.
    # Mean pairwise Hamming distance: how different are the QFT bitstrings across patches?
    # High Hamming = patches have divergent spectral signatures (spatially heterogeneous).
    # Low Hamming = patches have converged to similar spectral structure (ordered phase).
    qft_ent_valid = np.where(np.isfinite(qft_entropy), qft_entropy, np.nan)
    qft_ham_valid = np.where(np.isfinite(qft_hamming), qft_hamming, np.nan)

    ax_qft.plot(qft_ent_valid, color='cyan', label='Mean QFT Entropy (bits)', linewidth=1.2)
    ax_qft.set_ylim(bottom=0.0, top=1.05)
    ax_qft.set_ylabel("Entropy (bits)", fontsize=7, color='cyan')
    ax_qft.tick_params(axis='y', labelcolor='cyan', labelsize=6)
    ax_qft.set_title("QFT Bitstring Entropy + Inter-patch Hamming", fontsize=10)
    ax_qft.legend(loc='lower left', fontsize=7)
    ax_qft.grid(True, alpha=0.2)
    vline_qft = ax_qft.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    ax_qft_r = ax_qft.twinx()
    ax_qft_r.plot(qft_ham_valid, color='tomato', label='Mean Hamming dist',
                  linewidth=1.0, alpha=0.8)
    ax_qft_r.set_ylabel("Hamming dist (bits)", fontsize=7, color='tomato')
    ax_qft_r.tick_params(axis='y', labelcolor='tomato', labelsize=6)
    ax_qft_r.legend(loc='upper right', fontsize=7)

    # --- Panels 5-7: Heatmaps (unchanged from v6) ---
    heatmap_colors = [
        (0.15, 0.35, 0.85, 1.0),
        (0.10, 0.10, 0.10, 1.0),
        (0.85, 0.15, 0.25, 1.0),
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list("heatmap_cmap", heatmap_colors)

    y_ticks = []
    y_labels = []
    if num_patches <= 32:
        y_ticks = list(np.arange(num_patches))
        y_labels = [
            f"X:{patch_coords[i][0]} Y:{patch_coords[i][1]} Z:{patch_coords[i][2]}"
            for i in y_ticks
        ]
    else:
        z_mid = grid_z // 2
        for i in range(num_patches):
            if patch_coords[i][2] == z_mid:
                y_ticks.append(i)
                y_labels.append(
                    f"X:{patch_coords[i][0]} Y:{patch_coords[i][1]} Z:{patch_coords[i][2]}"
                )

    def _init_heatmap(ax, data, cmap, norm, label, show_ylabel=False, show_xlabel=False):
        img = ax.imshow(
            data, cmap=cmap, norm=norm,
            aspect='auto', interpolation='nearest'
        )
        for i in range(1, num_patches):
            if patch_coords[i][0] != patch_coords[i - 1][0]:
                ax.axhline(i - 0.5, color='white', linewidth=1.2, alpha=1.0)
            elif patch_coords[i][1] != patch_coords[i - 1][1]:
                ax.axhline(i - 0.5, color='#aaaaaa', linewidth=0.7, alpha=0.7, linestyle=':')
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=6)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel("Patch Qubit Index (spatial order, X-Y-Z)", fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0],
                            heatmap_cmap, spin_norm, "Bloch <X> (QFT probe input)",
                            show_ylabel=True, show_xlabel=False)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1],
                            heatmap_cmap, spin_norm, "Bloch <Y> (QFT probe input)",
                            show_ylabel=True, show_xlabel=False)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2],
                            heatmap_cmap, spin_norm, "Bloch <Z> (QFT phase weight)",
                            show_ylabel=True, show_xlabel=True)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    ax_slider = fig.add_axes([0.15, 0.04, 0.60, 0.02])
    slider = Slider(
        ax=ax_slider, label='Trotter Step (Anneal Step)',
        valmin=0, valmax=num_steps - 1, valinit=0, valstep=1, color='#4a90e2'
    )

    ax_play = fig.add_axes([0.80, 0.025, 0.08, 0.04])
    btn_play = Button(ax_play, 'Pause', color='#333333', hovercolor='#555555')
    is_playing = True

    def on_scroll(event):
        if event.inaxes != ax3d:
            return
        if not hasattr(ax3d, 'custom_zoom'):
            ax3d.custom_zoom = 1.0
        ax3d.custom_zoom *= 0.9 if event.button == 'down' else 1.1
        try:
            current_aspect = ax3d.get_box_aspect()
            if current_aspect is None:
                current_aspect = (grid_x, grid_y, max(1, grid_z))
            ax3d.set_box_aspect(current_aspect, zoom=ax3d.custom_zoom)
        except TypeError:
            ax3d.dist *= 0.9 if event.button == 'down' else 1.1
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    def on_key_press(event):
        if event.key == ' ':
            current_step = int(slider.val)
            e_list = energies['Total']
            e_val = e_list[current_step] if current_step < len(e_list) else float('nan')
            e_str = f"{e_val:.4f}" if (isinstance(e_val, float) and not np.isnan(e_val)) else "NaN"
            ap = anneal_pct[current_step] if current_step < len(anneal_pct) else float('nan')
            ap_str = f"{ap:.1f}" if not np.isnan(ap) else "NaN"
            filename = (
                f"dash_snapshot_{grid_x}x{grid_y}x{grid_z}"
                f"_step{current_step}_anneal{ap_str}pct_E{e_str}.png"
            )
            print(f"{prefix}Saving screenshot to {filename}...")
            fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"{prefix}Screenshot saved.")

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    _from_animation = [False]

    all_vlines = [vline_e, vline_d, vline_deriv, vline_fid, vline_qft]

    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)

        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            global_X, global_Y, global_Z, U, V, W,
            length=0.75, colors=_quiver_colors(W), arrow_length_ratio=0.3
        )

        ap = anneal_pct[frame] if frame < len(anneal_pct) and not np.isnan(anneal_pct[frame]) else None
        ap_str = f" | Anneal: {ap:.1f}%" if ap is not None else ""
        ax3d.set_title(
            f"Cloverfield QFT: 3D TFIM Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
            f"{num_patches} Patches | {total_qubits} Qubits)\n"
            f"Trotter Step: {frame}/{num_steps-1}{ap_str}",
            fontsize=14, pad=10
        )

        e_list = energies['Total']
        if e_list and frame < len(e_list) and not np.isnan(e_list[frame]):
            energy_text.set_text(f"Mean-field Energy: {e_list[frame]:.4f}")
        else:
            energy_text.set_text("")

        for vl in all_vlines:
            if vl is not None:
                vl.set_xdata([frame, frame])

        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return quiver_obj[0], energy_text, hmap_x, hmap_y, hmap_z

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
        fig, _animation_update, frames=num_steps, interval=150, blit=False
    )

    if mode == "save":
        print(f"{prefix}Commencing 4K FFmpeg render to '{SAVE_FILE}'...")
        try:
            ani.save(SAVE_FILE, writer='ffmpeg', fps=10, dpi=216)
            print(f"{prefix}Save complete.")
        except Exception as e:
            print(f"{prefix}Failed to save. Is ffmpeg installed? Error: {e}")
    else:
        print(f"{prefix}Opening GUI...")
        plt.show()


def main():
    mp.set_start_method('spawn', force=True)

    print("Forking 4K render to background process...")
    render_process = mp.Process(target=run_dashboard, args=("save",))
    render_process.start()

    run_dashboard(mode="interactive")

    if render_process.is_alive():
        print("\nInteractive viewer closed. Waiting for the background 4K render to finish...")
        render_process.join()

    print("All processes terminated.")


if __name__ == "__main__":
    main()
