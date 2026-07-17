# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_90.py
# Dashboard for MultiGpuHadronEngine Rev 90.x
# (27-Qubit 3x3x3 Macroscopic Grid Annealing, 729 Qubits Total).
# Reads:  macroscopic_lattice_states.npy
#         lattice_config.json
#         meanfield_ground_state_energy_curve_multi.csv
#         rcs_validation_multi.csv   <-- Rev 90 output (XEB_RCS / HOG_RCS)

import sys
import csv
import json
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
RCS_FILE       = "rcs_validation_multi.csv"   # Rev 90 output
SAVE_FILE      = "cloverfield_dashboard.mp4"

PATCH_LX, PATCH_LY, PATCH_LZ = 3, 3, 3   # Internal patch dimensions
GAP_X, GAP_Y, GAP_Z          = 1.5, 1.5, 1.5
QUIVER_STRIDE                 = 1

KIND_COLORS = {"X_SEAM": "#e64550", "Y_SEAM": "#45b0e6", "Z_SEAM": "#e6b422"}
# ---------------------


def load_energy_data(num_steps, log_prefix=""):
    """Load total / bulk / boundary energy from the engine CSV."""
    energies = {'Total': [], 'Bulk': [], 'Boundary': []}
    try:
        with open(ENERGY_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                energies['Total'].append(float(row["MeanField_Total_Energy"]))
                energies['Bulk'].append(float(row["MeanField_Bulk_Energy"]))
                energies['Boundary'].append(float(row["MeanField_Boundary_Energy"]))
    except Exception as e:
        print(f"{log_prefix}Warning: Could not parse {ENERGY_FILE}: {e}")

    for key in energies:
        if len(energies[key]) < num_steps:
            energies[key].extend([np.nan] * (num_steps - len(energies[key])))
        energies[key] = energies[key][:num_steps]

    return energies


def load_rcs_data(num_steps, log_prefix=""):
    """Load per-step mean XEB and HOG from rcs_validation_multi.csv.

    The RCS CSV has one row per *patch* per validation step, so we
    aggregate (mean) across patches before returning per-step arrays.
    Gaps between validation events are forward-filled.
    """
    # Accumulate lists keyed by step
    step_xeb: dict = {}
    step_hog: dict = {}

    try:
        with open(RCS_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["Step"])
                xeb  = float(row["XEB_RCS"])
                hog  = float(row["HOG_RCS"])
                step_xeb.setdefault(step, []).append(xeb)
                step_hog.setdefault(step, []).append(hog)
    except Exception as e:
        print(f"{log_prefix}Notice: No RCS data found or unreadable ({e}).")

    rcs_xeb = np.full(num_steps, np.nan)
    rcs_hog = np.full(num_steps, np.nan)
    for step, vals in step_xeb.items():
        if step < num_steps:
            rcs_xeb[step] = float(np.mean(vals))
    for step, vals in step_hog.items():
        if step < num_steps:
            rcs_hog[step] = float(np.mean(vals))

    # Forward-fill: validation doesn't fire every step
    rcs_xeb = pd.Series(rcs_xeb).ffill().values
    rcs_hog = pd.Series(rcs_hog).ffill().values

    return rcs_xeb, rcs_hog


def _safe_gradient(arr_with_nans):
    arr = np.asarray(arr_with_nans, dtype=float)
    out = np.full_like(arr, np.nan)
    valid_mask = ~np.isnan(arr)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        valid_data = arr[valid_indices]
        grad = np.gradient(valid_data, valid_indices)
        out[valid_indices] = grad
    elif len(valid_indices) == 1:
        out[valid_indices[0]] = 0.0
    return out


def build_interfaces(grid_x, grid_y, grid_z):
    """Boundary-qubit index arrays for every inter-patch seam.
    Local qubit index: idx = x*9 + y*3 + z  (matches Rev 90 engine).
    """
    x_i1 = np.array([18 + y*3 + z for y in range(3) for z in range(3)])  # +X (x=2)
    x_i2 = np.array([y*3 + z       for y in range(3) for z in range(3)]) # -X (x=0)
    y_i1 = np.array([x*9 + 6 + z  for x in range(3) for z in range(3)]) # +Y (y=2)
    y_i2 = np.array([x*9 + z       for x in range(3) for z in range(3)]) # -Y (y=0)
    z_i1 = np.array([x*9 + y*3 + 2 for x in range(3) for y in range(3)])# +Z (z=2)
    z_i2 = np.array([x*9 + y*3     for x in range(3) for y in range(3)])# -Z (z=0)

    interfaces = []

    def patch_id(tx, ty, tz):
        return (tx * grid_y + ty) * grid_z + tz

    for tx in range(grid_x):
        for ty in range(grid_y):
            for tz in range(grid_z):
                p1 = patch_id(tx, ty, tz)
                if tx < grid_x - 1:
                    interfaces.append((p1, patch_id(tx+1, ty, tz), x_i1, x_i2, "X_SEAM"))
                if ty < grid_y - 1:
                    interfaces.append((p1, patch_id(tx, ty+1, tz), y_i1, y_i2, "Y_SEAM"))
                if tz < grid_z - 1:
                    interfaces.append((p1, patch_id(tx, ty, tz+1), z_i1, z_i2, "Z_SEAM"))

    return interfaces


def run_dashboard(mode="interactive"):
    prefix = "[Background Render] " if mode == "save" else "[Interactive Viewer] "

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{prefix}Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    grid_x  = config.get("grid_x", 3)
    grid_y  = config.get("grid_y", 3)
    grid_z  = config.get("grid_z", 3)
    cfg_qpp = config.get("qubits_per_patch", PATCH_LX * PATCH_LY * PATCH_LZ)

    mmap = 'r' if mode == "save" else None
    try:
        history = np.load(DATA_FILE, mmap_mode=mmap)
    except FileNotFoundError:
        print(f"{prefix}Error: {DATA_FILE} not found.")
        sys.exit(1)

    num_steps, num_patches, qpp = history.shape[0], history.shape[1], history.shape[2]
    total_qubits = num_patches * qpp

    def patch_coords(p):
        tz   = p % grid_z
        rest = p // grid_z
        ty   = rest % grid_y
        tx   = rest // grid_y
        return tx, ty, tz

    energies           = load_energy_data(num_steps, prefix)
    rcs_xeb, rcs_hog   = load_rcs_data(num_steps, prefix)
    interfaces         = build_interfaces(grid_x, grid_y, grid_z)

    kinds_present = ["X_SEAM", "Y_SEAM", "Z_SEAM"]
    n_by_kind = {k: sum(1 for itf in interfaces if itf[4] == k) for k in kinds_present}
    print(f"{prefix}Interfaces: {n_by_kind['X_SEAM']} X-seam, "
          f"{n_by_kind['Y_SEAM']} Y-seam, {n_by_kind['Z_SEAM']} Z-seam")

    n_ifaces     = max(len(interfaces), 1)
    disagreements = np.zeros((num_steps, n_ifaces))
    iface_kind   = []

    if interfaces:
        hist_arr = np.asarray(history)
        for i, (p1, p2, i1, i2, kind) in enumerate(interfaces):
            diff = hist_arr[:, p1, i1, :] - hist_arr[:, p2, i2, :]
            disagreements[:, i] = np.mean(np.linalg.norm(diff, axis=2), axis=1)
            iface_kind.append(kind)

    iface_kind       = np.array(iface_kind) if iface_kind else np.array([], dtype=str)
    avg_disagreement = np.mean(disagreements, axis=1) if interfaces else np.zeros(num_steps)
    dis_by_kind      = {}
    for k in kinds_present:
        mask = (iface_kind == k)
        if np.any(mask):
            dis_by_kind[k] = np.mean(disagreements[:, mask], axis=1)

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
            qx = q // 9
            qy = (q % 9) // 3
            qz = q % 3
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
    # Figure layout
    # ----------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.1)

    ax3d     = fig.add_subplot(gs[0], projection='3d')
    gs_right = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[1], hspace=0.80)

    ax_energy = fig.add_subplot(gs_right[0])
    ax_dis    = fig.add_subplot(gs_right[1])
    ax_deriv  = fig.add_subplot(gs_right[2])
    ax_hmap_x = fig.add_subplot(gs_right[3])
    ax_hmap_y = fig.add_subplot(gs_right[4])
    ax_hmap_z = fig.add_subplot(gs_right[5])

    def get_vector_data(step_idx):
        return (
            history[step_idx, :, :, 0].ravel()[draw_idx],
            history[step_idx, :, :, 1].ravel()[draw_idx],
            history[step_idx, :, :, 2].ravel()[draw_idx],
        )

    U, V, W  = get_vector_data(0)
    spin_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    vector_colors = [
        (0.15, 0.35, 0.85, 0.85),
        (0.85, 0.85, 0.85, 0.45),
        (0.85, 0.15, 0.25, 0.85),
    ]
    vector_cmap = mcolors.LinearSegmentedColormap.from_list("ghost_vectors", vector_colors)

    def _quiver_colors(w_flat):
        return vector_cmap(spin_norm(w_flat))

    quiver_obj = [ax3d.quiver(
        qX, qY, qZ, U, V, W,
        length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3
    )]

    ax_cbar = fig.add_axes([0.02, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vector_cmap, norm=spin_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Bloch Vector Component (Spin State)', fontsize=10)

    energy_text = ax3d.text2D(0.04, 0.96, "", transform=ax3d.transAxes,
                              color='lightgreen', fontsize=12, fontweight='bold')
    # Two lines for RCS readout: XEB on top, HOG below
    rcs_xeb_text = ax3d.text2D(0.04, 0.92, "", transform=ax3d.transAxes,
                               color='cyan', fontsize=11, fontweight='bold')
    rcs_hog_text = ax3d.text2D(0.04, 0.88, "", transform=ax3d.transAxes,
                               color='#88ddff', fontsize=10)

    ax3d.set_title(
        f"Volumetric Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
        f"{num_patches} Patches | {total_qubits} Qubits)\nTrotter Step: 0/{num_steps-1}",
        fontsize=14, pad=10
    )
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
        axis.set_ticklabels([])
        axis.set_ticks([])
    ax3d.grid(False)
    ax3d.set_axis_off()

    # ---- Energy panel ----
    ax_energy.plot(energies['Total'],    label='Total Energy', color='lightgreen')
    ax_energy.plot(energies['Bulk'],     label='Bulk',         color='dodgerblue')
    ax_energy.plot(energies['Boundary'], label='Boundary',     color='orange')
    ax_energy.set_title("Energy Components", fontsize=10)
    ax_energy.set_ylabel("Energy (a.u.)", fontsize=8)
    ax_energy.legend(fontsize=6, loc='upper left', ncol=2)
    ax_energy.grid(True, alpha=0.2)
    vline_e = ax_energy.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ---- Seam disagreement panel ----
    vline_d = None
    if not interfaces:
        ax_dis.set_visible(False)
    else:
        ax_dis.plot(avg_disagreement, color='crimson', linewidth=1.6, label='Mean (all seams)')
        for k, lbl in (("X_SEAM", "Mean X"), ("Y_SEAM", "Mean Y"), ("Z_SEAM", "Mean Z")):
            if k in dis_by_kind:
                ax_dis.plot(dis_by_kind[k], color=KIND_COLORS[k],
                            linewidth=0.9, alpha=0.9, label=lbl)
        ax_dis.set_title("Coupling Error by Seam Class", fontsize=9)
        ax_dis.set_ylabel("Error (L2 Norm)", fontsize=8)
        ax_dis.legend(fontsize=6, loc='upper left', ncol=2)
        ax_dis.grid(True, alpha=0.2)
        vline_d = ax_dis.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ---- Derivative panel ----
    l1 = ax_deriv.plot(dE_dt, label='dE/dt', color='lightgreen')
    ax_deriv.set_ylabel("dE/dt (per step)", fontsize=8)
    ax_deriv_r = ax_deriv.twinx()
    l2 = ax_deriv_r.plot(dRes_dt, label='d||ds||/dt', color='crimson')
    ax_deriv_r.set_ylabel("d||ds||/dt (per step)", fontsize=8)
    lines  = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_deriv.legend(lines, labels, loc='upper left', fontsize=8)
    ax_deriv.set_title("Derivatives (Convergence Rate)", fontsize=10)
    ax_deriv.grid(True, alpha=0.2)
    vline_deriv = ax_deriv.axvline(x=0, color='white', linestyle='--', alpha=0.7)

    # ---- Heatmap panels ----
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
        "heatmap_cmap",
        [(0.15, 0.35, 0.85, 1.0), (0.10, 0.10, 0.10, 1.0), (0.85, 0.15, 0.25, 1.0)],
    )
    y_ticks = np.arange(num_patches)
    y_labels = [f"P{p}" for p in range(num_patches)]

    def _init_heatmap(ax, data, cmap, norm, label,
                      show_ylabel=False, show_xlabel=False):
        img = ax.imshow(data, cmap=cmap, norm=norm,
                        aspect='auto', interpolation='nearest')
        for r in range(1, num_patches):
            ax.axhline(r - 0.5, color='white', linewidth=0.3, alpha=0.35)
        ax.set_title(label, fontsize=9)
        if show_ylabel:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=4)
        else:
            ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel(f"Local Qubit Index (0-{qpp - 1})", fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
        else:
            ax.set_xticks([])
        return img

    hmap_x = _init_heatmap(ax_hmap_x, history[0, :, :, 0], heatmap_cmap, spin_norm,
                           "Polarization <X>", show_ylabel=True)
    hmap_y = _init_heatmap(ax_hmap_y, history[0, :, :, 1], heatmap_cmap, spin_norm,
                           "Polarization <Y>", show_ylabel=True)
    hmap_z = _init_heatmap(ax_hmap_z, history[0, :, :, 2], heatmap_cmap, spin_norm,
                           "Polarization <Z>", show_ylabel=True, show_xlabel=True)

    # ---- Controls ----
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.15)
    ax_slider = fig.add_axes([0.15, 0.05, 0.60, 0.02])
    slider = Slider(ax=ax_slider, label='Trotter Step',
                    valmin=0, valmax=num_steps - 1, valinit=0,
                    valstep=1, color='#4a90e2')

    ax_play  = fig.add_axes([0.80, 0.035, 0.08, 0.04])
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
                current_aspect = (x_max + 1.0, y_max + 1.0, z_max + 1.0)
            ax3d.set_box_aspect(current_aspect, zoom=ax3d.custom_zoom)
        except TypeError:
            ax3d.dist *= 1.1 if event.button == 'down' else 0.9
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    _from_animation = [False]

    # ----------------------------------------------------------------
    # Per-frame update
    # ----------------------------------------------------------------
    def update(frame):
        frame = int(frame)
        U, V, W = get_vector_data(frame)

        quiver_obj[0].remove()
        quiver_obj[0] = ax3d.quiver(
            qX, qY, qZ, U, V, W,
            length=0.6, colors=_quiver_colors(W), arrow_length_ratio=0.3,
        )
        ax3d.set_title(
            f"Volumetric Lattice Annealing ({grid_x}x{grid_y}x{grid_z} Grid | "
            f"{num_patches} Patches | {total_qubits} Qubits)\n"
            f"Trotter Step: {frame}/{num_steps-1}",
            fontsize=14, pad=10,
        )

        # Energy overlay
        e_list = energies['Total']
        if e_list and frame < len(e_list):
            e_val = e_list[frame]
            energy_text.set_text(
                f"Total Energy: {e_val:.4f}" if not np.isnan(e_val) else ""
            )
        else:
            energy_text.set_text("")

        # RCS XEB + HOG overlays
        xeb_val = rcs_xeb[frame] if frame < len(rcs_xeb) else np.nan
        hog_val = rcs_hog[frame] if frame < len(rcs_hog) else np.nan

        if not np.isnan(xeb_val):
            rcs_xeb_text.set_text(f"RCS XEB: {xeb_val:+.4f}")
        else:
            rcs_xeb_text.set_text("")

        if not np.isnan(hog_val):
            rcs_hog_text.set_text(f"HOG:     {hog_val:.3f}")
        else:
            rcs_hog_text.set_text("")

        # Timeline vlines
        vline_e.set_xdata([frame])
        if vline_d is not None:
            vline_d.set_xdata([frame])
        vline_deriv.set_xdata([frame])

        # Heatmaps
        hmap_x.set_data(history[frame, :, :, 0])
        hmap_y.set_data(history[frame, :, :, 1])
        hmap_z.set_data(history[frame, :, :, 2])

        if _from_animation[0]:
            ax3d.view_init(elev=ax3d.elev, azim=ax3d.azim + 0.3)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        return (quiver_obj[0], energy_text,
                rcs_xeb_text, rcs_hog_text,
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
        print("\nInteractive viewer closed. "
              "Waiting for the background 4K render to finish...")
        render_process.join()
    print("All processes terminated.")


if __name__ == "__main__":
    main()
