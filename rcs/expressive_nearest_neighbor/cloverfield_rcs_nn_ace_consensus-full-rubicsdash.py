# -*- coding: us-ascii -*-
# macroscopic_lattice_dash_120_desktop.py
# Desktop (matplotlib) dashboard for MultiGpuHadronEngine Rev 120
# No webserver -- opens a native Tk window directly.
#
# PANELS:
#   Top-left  : 3x3x3 Rubik's cube (Poly3DCollection, XEB heatmap per patch)
#   Top-mid   : Cross-patch |DeltaXEB| matrix (27x27, sorted by plane k)
#   Top-right : Mean XEB per diagonal plane (bar chart)
#   Mid-left  : Energy components + fidelity
#   Mid-mid   : RCS XEB time-series (routine mean + per-plane snapshot dots)
#   Mid-right : Convergence rate (dE/dt, d||ds||/dt)
#   Bot-left  : <X> heatmap (patches x qubits)
#   Bot-mid   : <Y> heatmap
#   Bot-right : <Z> heatmap
#
# CONTROLS:
#   Slider    : step scrub
#   Play/Pause button
#   SPACEBAR  : save PNG screenshot (300 DPI)
#
# USAGE:
#   python3 macroscopic_lattice_dash_120_desktop.py

import os
import glob
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================================================================
# CONFIGURATION
# =====================================================================
GRID_X, GRID_Y, GRID_Z  = 3, 3, 3
TOTAL_PATCHES            = GRID_X * GRID_Y * GRID_Z
QUBITS_PER_PATCH         = 27
SCREENSHOT_DPI           = 300
SNAPSHOT_COLOR           = "#f5c518"

# XEB colour ramp: deep-blue -> dark -> cyan -> gold -> red
XEB_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "xeb",
    [(0.05, 0.05, 0.35),   # XEB << 0
     (0.10, 0.10, 0.10),   # XEB ~ 0
     (0.00, 0.85, 0.85),   # XEB ~ 1  (Porter-Thomas)
     (1.00, 0.85, 0.00),   # XEB > 1
     (1.00, 0.27, 0.00)],  # XEB >> 1
)
XEB_NORM = mcolors.Normalize(vmin=-0.5, vmax=2.0)

PLANE_COLORS = [
    "#636EFA","#EF553B","#00CC96","#AB63FA",
    "#FFA15A","#19D3F3","#FF6692",
]

SPIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "spin", [(0.15,0.35,0.85),(0.10,0.10,0.10),(0.85,0.15,0.25)])
SPIN_NORM = mcolors.Normalize(vmin=-1.0, vmax=1.0)

# =====================================================================
# FILE DISCOVERY
# =====================================================================

def _find_config():
    hits = sorted(glob.glob("lattice_config_*.json"))
    if hits:
        return hits[-1]
    if os.path.exists("lattice_config.json"):
        return "lattice_config.json"
    return None

def _glob_csvs(pattern):
    # Anchor relative patterns to cwd so matplotlib/tk backend init
    # cannot silently shift the search root mid-startup.
    if not os.path.isabs(pattern):
        pattern = os.path.join(os.getcwd(), pattern)
    return sorted(glob.glob(pattern))

def _load_chunked_npy(prefix):
    files = sorted(glob.glob(prefix + "*.npy"))
    if files:
        return np.concatenate([np.load(f) for f in files], axis=0)
    if os.path.exists("macroscopic_lattice_states.npy"):
        return np.load("macroscopic_lattice_states.npy")
    return None

# =====================================================================
# PATCH GEOMETRY
# =====================================================================

def patch_coords(pid):
    z = pid % GRID_Z; rest = pid // GRID_Z
    y = rest % GRID_Y; x = rest // GRID_Y
    return x, y, z

def patch_plane(pid):
    x, y, z = patch_coords(pid)
    return x + y + z

# =====================================================================
# DATA LOADING
# =====================================================================

def load_config():
    cfg_file = _find_config()
    if cfg_file is None:
        print("[Config] No config file found; using defaults.")
        return {}
    with open(cfg_file) as f:
        cfg = json.load(f)
    print(f"[Config] Loaded {cfg_file}")
    return cfg

def load_history(cfg):
    prefix = cfg.get("state_file_prefix",
                     "macroscopic_lattice_states_multi_")
    hist = _load_chunked_npy(prefix)
    if hist is None:
        print("[History] No state files found.")
    else:
        print(f"[History] Loaded {hist.shape[0]} steps, "
              f"{hist.shape[1]} patches, {hist.shape[2]} qubits/patch.")
    return hist

def _discover_csvs(keywords):
    """Scan all CSVs in cwd; return those whose name contains all keywords."""
    hits = [f for f in sorted(glob.glob(os.path.join(os.getcwd(), "*.csv")))
            if all(k.lower() in os.path.basename(f).lower() for k in keywords)]
    return hits

def load_energy(cfg, num_steps):
    ts = cfg.get("run_id", "*")
    files = []
    for pat in [
        f"meanfield_energy_curve_multi_{ts}_part*.csv",
        "meanfield_energy_curve_multi_*_part*.csv",
        "meanfield_energy_curve_multi_*.csv",
        "meanfield_ground_state_energy_curve_multi.csv",
        "meanfield_ground_state_energy_curve*.csv",
        "meanfield_energy_curve*.csv",
    ]:
        files = _glob_csvs(pat)
        if files:
            break
    if not files:
        files = _discover_csvs(["energy"])
    if not files and os.path.isfile("meanfield_ground_state_energy_curve_multi.csv"):
        files = ["meanfield_ground_state_energy_curve_multi.csv"]
    print(f"[Energy] Files: {files}")
    rows = []
    for f in files:
        try:
            rows.append(pd.read_csv(f))
        except Exception as e:
            print(f"[Energy] Skipping {f}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows).sort_values("Step").drop_duplicates("Step")
    # Normalise column names across old/new schema variants
    rename = {}
    for c in df.columns:
        cs = c.strip()
        if cs == "MeanField_Total_Energy":    rename[c] = "MeanField_Total_Energy"
        elif cs == "MeanField_Bulk_Energy":   rename[c] = "MeanField_Bulk_Energy"
        elif cs == "MeanField_Boundary_Energy": rename[c] = "MeanField_Boundary_Energy"
        elif cs == "Min_Unitary_Fidelity":    rename[c] = "Min_Unitary_Fidelity"
        elif cs == "Anneal_Percent":          rename[c] = "Anneal_Percent"
    df = df.rename(columns=rename)
    print(f"[Energy] {len(df)} rows, cols: {list(df.columns)}")
    return df

def load_rcs(cfg, num_steps):
    ts = cfg.get("run_id", "*")
    files = []
    for pat in [
        f"rcs_validation_multi_{ts}_part*.csv",
        "rcs_validation_multi_*_part*.csv",
        "rcs_validation_multi_*.csv",
        "rcs_validation_multi.csv",
        "rcs_validation*.csv",
    ]:
        files = _glob_csvs(pat)
        if files:
            break
    if not files:
        files = _discover_csvs(["rcs"])
    # Final fallback: check exact known legacy name directly
    if not files and os.path.isfile("rcs_validation_multi.csv"):
        files = ["rcs_validation_multi.csv"]
    print(f"[RCS] Files: {files}")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"[RCS] Skipping {f}: {e}")
    if not frames:
        return pd.DataFrame(), {}, np.full(num_steps, np.nan), {}

    df = pd.concat(frames, ignore_index=True).sort_values(["Step","Patch"])
    df["k"] = df["Patch"].apply(patch_plane)

    snap_df = df[df["Is_Snapshot"] == 1]
    snapshot_xeb = {}
    for step, grp in snap_df.groupby("Step"):
        arr = np.full(TOTAL_PATCHES, np.nan)
        for _, row in grp.iterrows():
            pid = int(row["Patch"])
            if 0 <= pid < TOTAL_PATCHES:
                arr[pid] = row["XEB_RCS"]
        snapshot_xeb[int(step)] = arr

    routine_mean = (df.groupby("Step")["XEB_RCS"].mean()
                      .reindex(range(num_steps)).ffill().to_numpy())

    plane_mean = {}
    for step, arr in snapshot_xeb.items():
        pm = np.full(7, np.nan)
        for k in range(7):
            pids = [p for p in range(TOTAL_PATCHES) if patch_plane(p) == k]
            vals = [arr[p] for p in pids if not np.isnan(arr[p])]
            if vals:
                pm[k] = np.mean(vals)
        plane_mean[step] = pm

    return df, snapshot_xeb, routine_mean, plane_mean

def compute_disagreement(history, num_steps):
    if history is None:
        return np.zeros(num_steps)
    interfaces = []
    def pid(x,y,z): return (x*GRID_Y + y)*GRID_Z + z
    for x in range(GRID_X):
        for y in range(GRID_Y):
            for z in range(GRID_Z):
                p1 = pid(x,y,z)
                if x < GRID_X-1:
                    i1 = np.array([18+yy*3+zz for yy in range(3) for zz in range(3)])
                    i2 = np.array([yy*3+zz    for yy in range(3) for zz in range(3)])
                    interfaces.append((p1, pid(x+1,y,z), i1, i2))
                if y < GRID_Y-1:
                    i1 = np.array([x*9+6+zz for zz in range(3)])
                    i2 = np.array([x*9+zz   for zz in range(3)])
                    interfaces.append((p1, pid(x,y+1,z), i1, i2))
                if z < GRID_Z-1:
                    i1 = np.array([x*9+yy*3+2 for yy in range(3)])
                    i2 = np.array([x*9+yy*3   for yy in range(3)])
                    interfaces.append((p1, pid(x,y,z+1), i1, i2))
    if not interfaces:
        return np.zeros(num_steps)
    ns = min(history.shape[0], num_steps)
    dis = np.zeros(ns)
    for p1, p2, i1, i2 in interfaces:
        diff = history[:ns, p1][:, i1, :] - history[:ns, p2][:, i2, :]
        dis += np.mean(np.linalg.norm(diff, axis=2), axis=1)
    return dis / len(interfaces)

# =====================================================================
# RUBIK'S CUBE DRAWING
# =====================================================================
# Each patch rendered as a coloured square face on the +Z top surface of its
# cell, with thin dark side-faces to give 3-D depth.  Uses Poly3DCollection.

def _cell_faces(x, y, z, s=0.46):
    """Return (top_verts, side_verts_list) for a unit cell at (x,y,z)."""
    cx, cy, cz = float(x), float(y), float(z)
    # 8 corners
    c = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ])
    top   = [c[4], c[5], c[6], c[7]]           # +Z face
    sides = [
        [c[0],c[1],c[5],c[4]],  # front
        [c[2],c[3],c[7],c[6]],  # back
        [c[0],c[3],c[7],c[4]],  # left
        [c[1],c[2],c[6],c[5]],  # right
        [c[0],c[1],c[2],c[3]],  # bottom
    ]
    return top, sides

def draw_rubiks_cube(ax, xeb_arr):
    ax.cla()
    ax.set_facecolor('#111111')
    ax.set_axis_off()

    top_verts, top_colors = [], []
    side_verts = []

    for pid in range(TOTAL_PATCHES):
        x, y, z = patch_coords(pid)
        top, sides = _cell_faces(x, y, z)

        val = xeb_arr[pid] if xeb_arr is not None else np.nan
        if np.isnan(val):
            fc = (0.15, 0.15, 0.15, 1.0)
        else:
            fc = XEB_CMAP(XEB_NORM(val))

        top_verts.append(top)
        top_colors.append(fc)
        side_verts.extend(sides)

    # Side faces: dark grey
    sc = Poly3DCollection(side_verts, facecolor=(0.12,0.12,0.12,1.0),
                          edgecolor=(0.25,0.25,0.25,1.0), linewidth=0.4)
    ax.add_collection3d(sc)

    # Top faces (XEB coloured) -- drawn after sides so they appear on top
    tc = Poly3DCollection(top_verts, facecolors=top_colors,
                          edgecolor=(0.30,0.30,0.30,1.0), linewidth=0.5)
    ax.add_collection3d(tc)

    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.6, 2.6); ax.set_zlim(-0.6, 2.6)
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.view_init(elev=25, azim=-50)

    # Axis labels
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.set_ticklabels([])
        spine.set_ticks([])
        try:
            spine.pane.fill = False
            spine.pane.set_edgecolor('none')
        except Exception:
            pass
    ax.grid(False)

def _xeb_stats_str(xeb_arr):
    if xeb_arr is None:
        return "No snapshot yet"
    v = xeb_arr[~np.isnan(xeb_arr)]
    if len(v) == 0:
        return "No valid XEB values"
    return (f"n={len(v)}/27  mean={np.mean(v):+.4f}  "
            f"std={np.std(v):.4f}  "
            f"min={np.min(v):+.4f}(P{int(np.nanargmin(xeb_arr))})  "
            f"max={np.max(v):+.4f}(P{int(np.nanargmax(xeb_arr))})")

# =====================================================================
# DELTA MATRIX
# =====================================================================

def draw_delta_matrix(ax, xeb_arr):
    ax.cla()
    ax.set_facecolor('#1a1a1a')
    n     = TOTAL_PATCHES
    order = sorted(range(n), key=lambda p: (patch_plane(p), p))

    if xeb_arr is None:
        mat = np.full((n, n), np.nan)
    else:
        mat = np.zeros((n, n))
        for i, pi in enumerate(order):
            for j, pj in enumerate(order):
                if not (np.isnan(xeb_arr[pi]) or np.isnan(xeb_arr[pj])):
                    mat[i, j] = abs(xeb_arr[pi] - xeb_arr[pj])
                else:
                    mat[i, j] = np.nan

    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=0.5,
                   aspect='auto', interpolation='nearest')

    # Plane separator lines
    prev_k = patch_plane(order[0])
    for idx in range(1, n):
        k = patch_plane(order[idx])
        if k != prev_k:
            b = idx - 0.5
            ax.axhline(b, color=SNAPSHOT_COLOR, linewidth=0.8, linestyle=':')
            ax.axvline(b, color=SNAPSHOT_COLOR, linewidth=0.8, linestyle=':')
            prev_k = k

    labels = [f"P{order[i]}\nk={patch_plane(order[i])}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=3.5,
                                                 rotation=90, color='#aaaaaa')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=3.5,
                                                 color='#aaaaaa')
    ax.set_title("|DeltaXEB| matrix (by plane k)", fontsize=8,
                 color='#cccccc', pad=3)
    return im

# =====================================================================
# PLANE BAR CHART
# =====================================================================

def draw_plane_bar(ax, pm_arr):
    ax.cla()
    ax.set_facecolor('#1a1a1a')
    ks   = list(range(7))
    vals = [float(pm_arr[k]) if pm_arr is not None and not np.isnan(pm_arr[k])
            else 0.0 for k in ks]
    bars = ax.bar([f"k={k}" for k in ks], vals,
                  color=[PLANE_COLORS[k] for k in ks], width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:+.3f}", ha='center', va='bottom',
                fontsize=6, color='#cccccc')
    ax.axhline(1.0, color='cyan', linewidth=0.8, linestyle=':', label='PT')
    ax.axhline(0.0, color='#666666', linewidth=0.6, linestyle=':')
    ax.set_ylim(XEB_NORM.vmin, XEB_NORM.vmax + 0.3)
    ax.set_title("Mean XEB per plane", fontsize=8, color='#cccccc', pad=3)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.set_facecolor('#1a1a1a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

# =====================================================================
# TIME-SERIES PANELS
# =====================================================================

def _snap_vlines(ax, snap_steps, num_steps):
    for ss in snap_steps:
        if ss < num_steps:
            ax.axvline(ss, color=SNAPSHOT_COLOR,
                       linewidth=0.7, alpha=0.5, linestyle=':')

def draw_energy(ax, df_e, snap_steps, num_steps, step_cursor):
    ax.cla(); ax.set_facecolor('#1a1a1a')
    if not df_e.empty and "Step" in df_e.columns:
        xs = df_e["Step"].values
        def _plot_col(col, color, lw, label):
            if col in df_e.columns:
                ax.plot(xs, df_e[col].values, color=color,
                        linewidth=lw, label=label)
        _plot_col("MeanField_Total_Energy",    'lightgreen', 1.2, 'Total')
        _plot_col("MeanField_Bulk_Energy",     'dodgerblue', 1.0, 'Bulk')
        _plot_col("MeanField_Boundary_Energy", 'orange',     1.0, 'Bndry')
        if "Min_Unitary_Fidelity" in df_e.columns:
            ax2 = ax.twinx()
            ax2.plot(xs, df_e["Min_Unitary_Fidelity"].values,
                     color='#cc88ff', linewidth=0.8,
                     linestyle='--', label='Fidelity')
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(colors='#cc88ff', labelsize=6)
            ax2.set_facecolor('#1a1a1a')
            for sp in ax2.spines.values():
                sp.set_edgecolor('#333333')
    _snap_vlines(ax, snap_steps, num_steps)
    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--')
    ax.set_title("Energy components", fontsize=8, color='#cccccc', pad=3)
    ax.legend(fontsize=6, loc='upper left',
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.set_facecolor('#1a1a1a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

def draw_rcs_ts(ax, routine_mean, df_rcs, snap_steps, num_steps,
                step_cursor, cfg):
    ax.cla(); ax.set_facecolor('#1a1a1a')
    xs = list(range(num_steps))
    ax.plot(xs, routine_mean, color='cyan', linewidth=1.2, label='Mean XEB')
    ax.axhline(1.0, color='cyan', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axhline(0.0, color='#555555', linewidth=0.5, linestyle=':')

    if not df_rcs.empty:
        snap_df = df_rcs[df_rcs["Is_Snapshot"] == 1]
        for k in range(7):
            kdf = snap_df[snap_df["k"] == k]
            if kdf.empty:
                continue
            mb = kdf.groupby("Step")["XEB_RCS"].mean()
            ax.scatter(mb.index, mb.values,
                       color=PLANE_COLORS[k % len(PLANE_COLORS)],
                       s=18, marker='D', zorder=5,
                       label=f"snap k={k}")

    _snap_vlines(ax, snap_steps, num_steps)
    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--')
    depth = cfg.get("rcs_depth","?"); shots = cfg.get("rcs_shots","?")
    ev    = cfg.get("rcs_validate_every","?")
    ax.set_title(f"RCS XEB  depth={depth} shots={shots} every={ev}",
                 fontsize=8, color='#cccccc', pad=3)
    ax.legend(fontsize=5, loc='upper left', ncol=4,
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.set_facecolor('#1a1a1a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

def draw_deriv(ax, df_e, avg_dis, snap_steps, num_steps, step_cursor):
    ax.cla(); ax.set_facecolor('#1a1a1a')
    if not df_e.empty and "MeanField_Total_Energy" in df_e.columns:
        total = df_e.set_index("Step")["MeanField_Total_Energy"]\
                    .reindex(range(num_steps)).ffill().values
    else:
        total = np.zeros(num_steps)
    dE  = np.gradient(np.nan_to_num(total))
    dR  = np.gradient(avg_dis)
    xs  = list(range(num_steps))
    ax.plot(xs, dE, color='lightgreen', linewidth=1.0, label='dE/dt')
    ax2 = ax.twinx()
    ax2.plot(xs, dR, color='crimson', linewidth=1.0, label='d||ds||/dt')
    ax2.tick_params(colors='crimson', labelsize=6)
    ax2.set_facecolor('#1a1a1a')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333333')
    _snap_vlines(ax, snap_steps, num_steps)
    ax.axvline(step_cursor, color='white', linewidth=0.8, linestyle='--')
    ax.set_title("Convergence rate", fontsize=8, color='#cccccc', pad=3)
    ax.legend(fontsize=6, loc='upper left',
              facecolor='#1a1a1a', labelcolor='#cccccc', framealpha=0.5)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.set_facecolor('#1a1a1a')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

def draw_heatmap(ax, data, label, qpp):
    ax.cla()
    ax.imshow(data, cmap=SPIN_CMAP, norm=SPIN_NORM,
              aspect='auto', interpolation='nearest')
    for r in range(1, data.shape[0]):
        ax.axhline(r - 0.5, color='white', linewidth=0.2, alpha=0.3)
    ax.set_title(f"<{label}>", fontsize=8, color='#cccccc', pad=3)
    ax.set_xlabel(f"Qubit (0-{qpp-1})", fontsize=6, color='#aaaaaa')
    ax.set_ylabel("Patch", fontsize=6, color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa', labelsize=5)

# =====================================================================
# MAIN
# =====================================================================

def main():
    import sys
    # Allow overriding the working directory via command-line argument:
    #   python3 script.py /path/to/data
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        print(f"[CWD] Changing to: {target_dir}")
        os.chdir(target_dir)
    print(f"[CWD] Working directory: {os.getcwd()}")
    print(f"[CWD] CSV files: {sorted(glob.glob('*.csv'))}")
    print(f"[CWD] NPY files: {sorted(glob.glob('*.npy'))}")
    cfg        = load_config()
    history    = load_history(cfg)
    num_steps  = history.shape[0] if history is not None else 1
    qpp        = history.shape[2] if history is not None else QUBITS_PER_PATCH
    snap_steps = sorted(cfg.get("rcs_full_snapshot_steps", []))

    df_e                                         = load_energy(cfg, num_steps)
    df_rcs, snapshot_xeb, routine_mean, plane_mean = load_rcs(cfg, num_steps)
    print(f"[Snapshot] keys loaded: {sorted(snapshot_xeb.keys())}")
    avg_dis    = compute_disagreement(history, num_steps)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 13), facecolor='#111111')
    fig.suptitle(
        f"MultiGpuHadronEngine Rev 120 -- "
        f"{GRID_X}x{GRID_Y}x{GRID_Z} grid | "
        f"{TOTAL_PATCHES} patches | "
        f"{TOTAL_PATCHES * QUBITS_PER_PATCH} qubits | "
        f"{num_steps} steps",
        fontsize=12, color='#f5c518', y=0.99,
    )

    # --- Layout ---
    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.98,
                               top=0.93, bottom=0.60,
                               wspace=0.28)
    gs_mid = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.98,
                               top=0.57, bottom=0.34,
                               wspace=0.30)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.04, right=0.98,
                               top=0.31, bottom=0.12,
                               wspace=0.30)

    ax_cube   = fig.add_subplot(gs_top[0], projection='3d')
    ax_delta  = fig.add_subplot(gs_top[1])
    ax_plane  = fig.add_subplot(gs_top[2])
    ax_energy = fig.add_subplot(gs_mid[0])
    ax_rcs    = fig.add_subplot(gs_mid[1])
    ax_deriv  = fig.add_subplot(gs_mid[2])
    ax_hmx    = fig.add_subplot(gs_bot[0])
    ax_hmy    = fig.add_subplot(gs_bot[1])
    ax_hmz    = fig.add_subplot(gs_bot[2])

    # XEB cube colorbar (shared, placed manually)
    ax_cbar = fig.add_axes([0.002, 0.62, 0.008, 0.28])
    sm = plt.cm.ScalarMappable(cmap=XEB_CMAP, norm=XEB_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('XEB', fontsize=8, color='#cccccc')
    cbar.ax.tick_params(labelsize=7, colors='#aaaaaa')

    # --- Status text ---
    status_ax = fig.add_axes([0.04, 0.955, 0.92, 0.02])
    status_ax.set_axis_off()
    status_txt = status_ax.text(
        0.0, 0.5, "", transform=status_ax.transAxes,
        fontsize=8, color='#aaaaaa', va='center')

    # --- Slider ---
    ax_slider = fig.add_axes([0.10, 0.045, 0.70, 0.018])
    slider = Slider(ax_slider, 'Step', 0, num_steps - 1,
                    valinit=0, valstep=1, color='#4a90e2')
    slider.label.set_color('#cccccc')
    slider.valtext.set_color('#f5c518')

    # --- Play/Pause button ---
    ax_btn = fig.add_axes([0.83, 0.036, 0.07, 0.032])
    btn = Button(ax_btn, '> Play',
                 color='#333333', hovercolor='#555555')
    btn.label.set_color('#cccccc')

    # State
    is_playing = [False]
    cur_frame  = [0]

    # --- Helper: nearest snapshot at or before step ---
    def _snap_at(step):
        best = None
        for ss in sorted(snapshot_xeb.keys()):
            if ss <= step:
                best = ss
        return best

    # --- Full redraw for a given step ---
    def redraw(step):
        step = int(step)
        cur_frame[0] = step

        snap = _snap_at(step)
        xeb_arr = snapshot_xeb.get(snap) if snap is not None else None
        pm_arr  = plane_mean.get(snap)   if snap is not None else None

        is_snap = step in snapshot_xeb
        snap_tag = "  *** FULL-CUBE SNAPSHOT ***" if is_snap else ""
        anneal_pct = 100.0 * step / max(1, num_steps - 1)

        # Cube
        draw_rubiks_cube(ax_cube, xeb_arr)
        snap_label = (f"step {snap}  mean={np.nanmean(xeb_arr):+.3f}"
                      if xeb_arr is not None else "no snapshot yet")
        ax_cube.set_title(f"XEB Cube -- {snap_label}",
                          fontsize=8, color='#cccccc', pad=2)

        # Delta matrix
        im = draw_delta_matrix(ax_delta, xeb_arr)

        # Plane bar
        draw_plane_bar(ax_plane,
                       pm_arr if pm_arr is not None else np.full(7, np.nan))

        # Time-series
        draw_energy(ax_energy, df_e, snap_steps, num_steps, step)
        draw_rcs_ts(ax_rcs, routine_mean, df_rcs, snap_steps,
                    num_steps, step, cfg)
        draw_deriv(ax_deriv, df_e, avg_dis, snap_steps, num_steps, step)

        # Heatmaps
        blank = np.zeros((TOTAL_PATCHES, qpp))
        if history is not None and step < history.shape[0]:
            draw_heatmap(ax_hmx, history[step, :, :, 0], "X", qpp)
            draw_heatmap(ax_hmy, history[step, :, :, 1], "Y", qpp)
            draw_heatmap(ax_hmz, history[step, :, :, 2], "Z", qpp)
        else:
            draw_heatmap(ax_hmx, blank, "X", qpp)
            draw_heatmap(ax_hmy, blank, "Y", qpp)
            draw_heatmap(ax_hmz, blank, "Z", qpp)

        # Status line
        stats = _xeb_stats_str(xeb_arr)
        status_txt.set_text(
            f"Step {step}/{num_steps-1}  ({anneal_pct:.1f}% annealed)"
            f"{snap_tag}  |  {stats}")

        fig.canvas.draw_idle()

    # --- Slider callback ---
    def on_slider(val):
        redraw(val)

    slider.on_changed(on_slider)

    # --- Animation timer ---
    def _tick(frame):
        if not is_playing[0]:
            return
        next_step = (cur_frame[0] + 1) % num_steps
        slider.set_val(next_step)   # triggers on_slider -> redraw

    ani = animation.FuncAnimation(fig, _tick, interval=300, blit=False)

    # --- Play/Pause ---
    def toggle_play(event):
        is_playing[0] = not is_playing[0]
        btn.label.set_text('|| Pause' if is_playing[0] else '> Play')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_play)

    # --- Screenshot ---
    def on_key(event):
        if event.key == ' ':
            ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_step{cur_frame[0]:03d}_{ts}.png"
            fig.savefig(fname, dpi=SCREENSHOT_DPI,
                        facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"[Screenshot] Saved {fname} ({SCREENSHOT_DPI} DPI)",
                  flush=True)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial draw
    redraw(0)

    print("[Dashboard] Window open.  SPACEBAR = screenshot.  Close window to quit.")
    plt.show()


if __name__ == "__main__":
    main()
