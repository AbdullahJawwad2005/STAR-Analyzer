"""
graph_export.py — Visualization / Graph Export for STAR Analyzer
=================================================================
Creates three multi-page PDF files (generated in parallel):

  {stem}_graphs_heatmaps.pdf    — proportional zone occupancy heatmaps (1 page / animal)
  {stem}_graphs_cascade.pdf     — cascade plots: speed / speed_accel / jerk (1 page / animal)
  {stem}_graphs_distance.pdf    — inter-animal distance with proximity highlight (1 page / pair)

All plotting uses matplotlib with the Agg (non-GUI) backend so it is safe to
call from any background QThread.

Entry point: write_graphs(...)
"""

import gc
import numpy as np
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Agg backend MUST be set before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib as _mpl
import matplotlib.cm as _cm_module
import matplotlib.patheffects
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# 3x3 spatial arrangement of zones (mirrors arena: top-left origin)
_ZONE_GRID = [
    ["C1",   "W1",   "C2"],
    ["W4",   "Open", "W2"],
    ["C4",   "W3",   "C3"],
]

_ZONE_LABEL = {
    "C1":   "Top-Left\nCorner",
    "C2":   "Top-Right\nCorner",
    "C3":   "Bot-Right\nCorner",
    "C4":   "Bot-Left\nCorner",
    "W1":   "Top Wall",
    "W2":   "Right Wall",
    "W3":   "Bottom Wall",
    "W4":   "Left Wall",
    "Open": "Center\n(Open)",
}

_ALL_ZONES = ["C1", "W1", "C2", "W4", "Open", "W2", "C4", "W3", "C3"]

_NICE_LABEL = {
    "speed":             "Speed (px/frame)",
    "speed_accel":       "Speed Accel (px/frame\u00b2/s)",
    "accel":             "Accel (px/frame\u00b2)",
    "jerk":              "Jerk (px/frame\u00b3)",
    "hourglass_area":    "Hourglass Area (px\u00b2)",
    "hourglass_ratio":   "Hourglass Ratio (upper/lower)",
    "inter_animal_dist": "Inter-Animal Dist. (px)",
}

_MAX_TS_PTS = 5000  # downsample time series to this many points for display

# PDF plot style
plt.rcParams.update({
    "figure.facecolor":  "#f7f7f7",
    "axes.facecolor":    "#fdfdfd",
    "axes.edgecolor":    "#888888",
    "axes.labelcolor":   "#222222",
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "text.color":        "#222222",
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  12,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_times_and_indices(frame_map, fps):
    """Return (times_s ndarray, sleap_idxs ndarray) sorted by video frame."""
    items = sorted(frame_map.items())
    vid_frames = np.array([vf for vf, _ in items])
    sleap_idxs = np.array([si for _, si in items], dtype=int)
    times_s    = vid_frames / fps
    return times_s, sleap_idxs


def _body_prefix(node_names):
    """Return the node-name prefix (with spaces -> underscores) for the body-centre node."""
    for pat in ("center", "body", "cm", "centroid"):
        for nn in node_names:
            if pat in nn.lower():
                return nn.replace(" ", "_")
    return node_names[0].replace(" ", "_") if node_names else "body"


def _extract(track_arr, key, sleap_idxs):
    """Safely extract (n_frames,) float array from track_arr; NaN-fills if absent."""
    arr = track_arr.get(key)
    if arr is None:
        return np.full(len(sleap_idxs), np.nan)
    return arr[sleap_idxs].astype(float)


def _downsample_idx(n, max_pts):
    """Uniform-spaced index array of length <= max_pts."""
    if n <= max_pts:
        return np.arange(n)
    return np.round(np.linspace(0, n - 1, max_pts)).astype(int)


# ---------------------------------------------------------------------------
# 1. Zone Heatmaps (proportional arena shape)
# ---------------------------------------------------------------------------

def _write_heatmaps(zone_summary_df, track_names, pdf_path, status_cb,
                    arena_cm=40, strip_cm=8, arena_snapshot=None):
    """One PDF page per animal: zone occupancy overlaid on arena photograph.

    When arena_snapshot (RGB ndarray) is provided the photo is rendered as a
    desaturated background image and each zone gets a semi-transparent colour
    overlay proportional to occupancy.  Without a snapshot the figure falls back
    to opaque coloured cells (same data, no photo).
    """
    if status_cb:
        status_cb("Graphs: zone heatmaps\u2026")

    import matplotlib.patches as mpatches

    # Determine colour range across all animals for a consistent scale
    all_pcts = zone_summary_df["% of Session"].values.astype(float)
    vmax     = max(float(np.max(all_pcts)) if len(all_pcts) else 1.0, 1.0)
    cmap     = _mpl.colormaps["YlOrRd"]
    norm     = Normalize(vmin=0.0, vmax=vmax)

    # Proportional ratios: corner/wall strips vs centre
    f = max(0.02, min(strip_cm / max(arena_cm, 1), 0.45))

    # Prepare desaturated snapshot if available
    desat_img = None
    if arena_snapshot is not None:
        try:
            img = arena_snapshot.astype(np.float64) / 255.0
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            # Blend: 30% colour, 70% grey → muted background that doesn't compete
            desat_img = 0.30 * img + 0.70 * np.stack([gray]*3, axis=-1)
            desat_img = np.clip(desat_img, 0, 1)
        except Exception:
            desat_img = None

    with PdfPages(pdf_path) as pdf:
        for track_name in track_names:
            sub     = zone_summary_df[zone_summary_df["Track"] == track_name]
            pct_map = {z: 0.0 for z in _ALL_ZONES}
            pct_map.update(dict(zip(sub["Zone"].astype(str),
                                    sub["% of Session"].astype(float))))

            fig = plt.figure(figsize=(9, 9.8))

            if desat_img is not None:
                # --- Photo-overlay mode ---
                # Single axes showing the arena photo with zone rectangles overlaid
                ax_img = fig.add_axes([0.05, 0.08, 0.82, 0.82])
                ax_img.imshow(desat_img, aspect="equal", interpolation="bilinear")
                ax_img.set_xticks([]); ax_img.set_yticks([])
                for sp in ax_img.spines.values():
                    sp.set_edgecolor("#444444"); sp.set_linewidth(2.0)

                h, w = desat_img.shape[:2]
                s_px = f * w   # strip width in image pixels
                s_py = f * h

                # Zone bounding boxes in image pixel coordinates
                zone_rects = {
                    "C1":   (0,          0,          s_px,        s_py),
                    "W1":   (s_px,       0,          w - 2*s_px,  s_py),
                    "C2":   (w - s_px,   0,          s_px,        s_py),
                    "W4":   (0,          s_py,       s_px,        h - 2*s_py),
                    "Open": (s_px,       s_py,       w - 2*s_px,  h - 2*s_py),
                    "W2":   (w - s_px,   s_py,       s_px,        h - 2*s_py),
                    "C4":   (0,          h - s_py,   s_px,        s_py),
                    "W3":   (s_px,       h - s_py,   w - 2*s_px,  s_py),
                    "C3":   (w - s_px,   h - s_py,   s_px,        s_py),
                }

                for zone, (zx, zy, zw, zh) in zone_rects.items():
                    pct  = pct_map.get(zone, 0.0)
                    rgba = cmap(norm(pct))
                    # Semi-transparent colour overlay (alpha scales with occupancy)
                    alpha = 0.30 + 0.45 * min(pct / max(vmax, 1e-6), 1.0)

                    rect = mpatches.Rectangle(
                        (zx, zy), zw, zh,
                        linewidth=1.5, edgecolor="white",
                        facecolor=rgba[:3], alpha=alpha)
                    ax_img.add_patch(rect)

                    # Zone dashed border for clarity
                    rect_border = mpatches.Rectangle(
                        (zx, zy), zw, zh,
                        linewidth=1.2, edgecolor="white", linestyle="--",
                        facecolor="none", alpha=0.6)
                    ax_img.add_patch(rect_border)

                    # Text: zone label + percentage
                    cx_t = zx + zw / 2
                    cy_t = zy + zh / 2

                    # Choose text colour: white with dark outline for readability
                    ax_img.text(cx_t, cy_t - zh * 0.10,
                                _ZONE_LABEL.get(zone, zone),
                                ha="center", va="center", fontsize=10,
                                fontweight="bold", color="white",
                                multialignment="center",
                                path_effects=[
                                    matplotlib.patheffects.withStroke(
                                        linewidth=2.5, foreground="black")])
                    ax_img.text(cx_t, cy_t + zh * 0.15,
                                f"{pct:.1f}%",
                                ha="center", va="center", fontsize=14,
                                fontweight="bold", color="white",
                                path_effects=[
                                    matplotlib.patheffects.withStroke(
                                        linewidth=3, foreground="black")])

                fig.suptitle(f"Zone Occupancy  \u2014  {track_name}",
                             fontsize=14, fontweight="bold", y=0.95)

                # Colorbar
                sm = _cm_module.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cax = fig.add_axes([0.89, 0.08, 0.025, 0.82])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label("% of Session", fontsize=11)

            else:
                # --- Fallback: coloured-cell mode (no photo) ---
                ratios = [f, 1 - 2 * f, f]
                gs = gridspec.GridSpec(3, 3, figure=fig,
                                       width_ratios=ratios, height_ratios=ratios,
                                       wspace=0.05, hspace=0.05)
                fig.suptitle(f"Zone Occupancy Heatmap  \u2014  {track_name}",
                             fontsize=14, fontweight="bold")

                axes_flat = []
                for r, row_zones in enumerate(_ZONE_GRID):
                    for c, zone in enumerate(row_zones):
                        ax  = fig.add_subplot(gs[r, c])
                        axes_flat.append(ax)
                        pct = pct_map.get(zone, 0.0)
                        rgba = cmap(norm(pct))
                        ax.set_facecolor(rgba)

                        lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                        tcol = "white" if lum < 0.55 else "#222222"

                        label = _ZONE_LABEL.get(zone, zone)
                        ax.text(0.5, 0.62, label,
                                ha="center", va="center", fontsize=11,
                                fontweight="bold", transform=ax.transAxes,
                                color=tcol, multialignment="center")
                        ax.text(0.5, 0.30, f"{pct:.1f} %",
                                ha="center", va="center", fontsize=16,
                                fontweight="bold", transform=ax.transAxes,
                                color=tcol)

                        ax.set_xticks([]); ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_edgecolor("#666666"); sp.set_linewidth(1.8)

                sm = _cm_module.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes_flat, fraction=0.025, pad=0.03)
                cbar.set_label("% of Session", fontsize=11)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            gc.collect()


# ---------------------------------------------------------------------------
# 2. Cascade Plot (Speed / Speed Accel / Jerk)
# ---------------------------------------------------------------------------

def _write_cascade_plot(track_arrays, times, sleap_idxs, track_names,
                        node_names, pdf_path, status_cb):
    """One page per animal: 3 vertically stacked panels (speed, speed_accel, jerk)."""
    if status_cb:
        status_cb("Graphs: cascade plots\u2026")

    bp = _body_prefix(node_names)

    panels = [
        (f"{bp}_speed",  "Speed (px/frame)",            "#2196F3"),
        ("speed_accel",  "Speed Accel (px/frame\u00b2/s)", "#FF9800"),
        (f"{bp}_jerk",   "Jerk (px/frame\u00b3)",          "#E91E63"),
    ]

    with PdfPages(pdf_path) as pdf:
        for t_idx, tname in enumerate(track_names):
            ta = track_arrays[t_idx]

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.subplots_adjust(hspace=0.05)
            fig.suptitle(f"Cascade Plot  \u2014  {tname}",
                         fontsize=13, fontweight="bold")

            for i, (key, ylabel, color) in enumerate(panels):
                ax  = axes[i]
                arr = _extract(ta, key, sleap_idxs)
                idx = _downsample_idx(len(arr), _MAX_TS_PTS)
                t_ds = times[idx]
                v_ds = arr[idx]
                fin  = np.isfinite(v_ds)

                if np.any(fin):
                    ax.plot(t_ds[fin], v_ds[fin], linewidth=0.8,
                            color=color, alpha=0.85)

                ax.set_ylabel(ylabel, fontsize=9)
                ax.yaxis.set_major_locator(MaxNLocator(5))

                # Only bottom panel gets x-label
                if i < 2:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Time (s)", fontsize=10)
                    ax.xaxis.set_major_locator(MaxNLocator(8))

            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()


# ---------------------------------------------------------------------------
# 3. Distance Highlight Plot
# ---------------------------------------------------------------------------

def _write_distance_plot(pair_arrays, times, sleap_idxs, track_names,
                         pdf_path, status_cb, px_per_cm=1.0):
    """One page per pair: inter-animal distance with <= 3cm proximity highlighted."""
    if status_cb:
        status_cb("Graphs: distance plots\u2026")

    n_tracks = len(track_names)
    if n_tracks < 2:
        return

    with PdfPages(pdf_path) as pdf:
        for tA, tB in combinations(range(n_tracks), 2):
            pfx = f"t{tA}_t{tB}"
            key = f"{pfx}/inter_animal_dist"
            arr = pair_arrays.get(key)
            if arr is None:
                continue

            dist_px  = arr[sleap_idxs].astype(float)
            dist_cm  = dist_px / max(px_per_cm, 1e-9)

            idx  = _downsample_idx(len(dist_cm), _MAX_TS_PTS)
            t_ds = times[idx]
            d_ds = dist_cm[idx]

            nA = track_names[tA] if tA < n_tracks else f"t{tA}"
            nB = track_names[tB] if tB < n_tracks else f"t{tB}"
            pair_label = f"{nA} vs {nB}"

            fig, ax = plt.subplots(figsize=(12, 4))
            fin = np.isfinite(d_ds)

            if np.any(fin):
                ax.plot(t_ds[fin], d_ds[fin], linewidth=0.9,
                        color="#1565C0", alpha=0.85, label="Distance (cm)")
                ax.fill_between(t_ds, 0, d_ds,
                                where=(d_ds <= 3.0) & fin,
                                alpha=0.3, color="red",
                                label="Proximity (\u2264 3 cm)")

            ax.axhline(3.0, color="#B71C1C", linewidth=1.0, linestyle="--",
                       alpha=0.7, label="3 cm threshold")
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Inter-Animal Distance (cm)", fontsize=10)
            ax.set_title(f"Inter-Animal Distance  \u2014  {pair_label}",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            ax.xaxis.set_major_locator(MaxNLocator(8))
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.set_ylim(bottom=0)

            fig.tight_layout()
            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)
            gc.collect()


# ---------------------------------------------------------------------------
# Oncoplot shared helpers
# ---------------------------------------------------------------------------

_ONCO_CMAP = LinearSegmentedColormap.from_list(
    'onco', ['#2166ac', '#f7f7f7', '#b2182b'])


def _find_dist_key(keys, patterns_a, patterns_b):
    """Find a node-pair distance key matching patterns for both endpoints."""
    for k in keys:
        if not k.endswith('_dist'):
            continue
        kl = k.lower()
        has_a = any(p in kl for p in patterns_a)
        has_b = any(p in kl for p in patterns_b)
        if has_a and has_b:
            return k
    return None


def _build_10s_bins(times):
    """Return list of index arrays, one per 10-second bin."""
    bin_idx = (times // 10).astype(int)
    max_bin = int(bin_idx[-1]) if len(bin_idx) else 0
    bins = []
    for b in range(max_bin + 1):
        bins.append(np.where(bin_idx == b)[0])
    return bins


def _norm_minmax(arr, global_min, global_max):
    """Normalise to [0, 1] via global min/max."""
    rng = global_max - global_min
    if rng < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return np.clip((arr - global_min) / rng, 0.0, 1.0)


def _norm_correlation(arr):
    """Map [-1, 1] → [0, 1]."""
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def _norm_p90(chunk_means, session_p90):
    """cell = chunk_mean / session_p90, clamped [0, 1]."""
    if session_p90 < 1e-12:
        return np.full_like(chunk_means, 0.5, dtype=float)
    return np.clip(chunk_means / session_p90, 0.0, 1.0)


def _render_oncoplot(matrix, row_labels, section_breaks, times, title,
                     pdf, show_values=True):
    """Render a single oncoplot page and save to the open PdfPages."""
    n_rows, n_bins = matrix.shape
    fig_w = max(14, n_bins * 0.45)
    fig_h = max(6, n_rows * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(matrix, aspect='auto', cmap=_ONCO_CMAP, vmin=0, vmax=1,
              interpolation='nearest')

    # Section dividers — draw thick white lines between groups
    for brk in section_breaks:
        ax.axhline(brk - 0.5, color='white', linewidth=3)

    # Y-axis labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8, ha='right')

    # X-axis: bin edge labels in seconds
    bin_edges = [int(i * 10) for i in range(n_bins)]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_edges, fontsize=7, rotation=45, ha='right')
    ax.set_xlabel("Time (s)", fontsize=9)

    # Cell text (only if manageable number of bins)
    if show_values and n_bins <= 40:
        for r in range(n_rows):
            for c in range(n_bins):
                v = matrix[r, c]
                if np.isfinite(v):
                    tcol = 'white' if (v < 0.25 or v > 0.75) else '#222222'
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                            fontsize=5.5, color=tcol)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Horizontal colorbar below plot
    sm = _cm_module.ScalarMappable(cmap=_ONCO_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04,
                        pad=0.12, aspect=40)
    cbar.set_label("Normalised value", fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


# ---------------------------------------------------------------------------
# 4. Per-Animal Feature Oncoplot
# ---------------------------------------------------------------------------

def _write_oncoplot(track_arrays, times, sleap_idxs, track_names,
                    node_names, fps, pdf_path, status_cb):
    """Per-animal feature oncoplot: rows = features (4 sections), cols = 10s bins."""
    if status_cb:
        status_cb("Graphs: feature oncoplot\u2026")

    bp = _body_prefix(node_names)
    bins = _build_10s_bins(times)
    n_bins = len(bins)
    if n_bins == 0:
        return

    # Discover body-length and body-width keys
    sample_keys = list(track_arrays[0].keys()) if 0 in track_arrays else []
    body_len_key = _find_dist_key(
        sample_keys, ['nose'], ['tail_base', 'tb', 'tail'])
    body_wid_key = _find_dist_key(
        sample_keys, ['hip_l', 'hl'], ['hip_r', 'hr'])

    # ---- Build row specs: (label, key_or_func, norm_type) ----
    # norm_type: 'minmax', 'p90', 'heading', 'direct'
    section1 = [
        ('Speed',       f'{bp}_speed',   'minmax'),
        ('Speed Accel', 'speed_accel',   'minmax'),
        ('|Jerk|',      f'{bp}_jerk',    'minmax_abs'),
    ]
    section2 = [
        ('Speed p90%',       f'{bp}_speed',   'p90'),
        ('Speed Accel p90%', 'speed_accel',   'p90'),
        ('|Jerk| p90%',      f'{bp}_jerk',    'p90_abs'),
    ]
    section3 = []
    if body_len_key:
        section3.append(('Body Length', body_len_key, 'minmax'))
    if body_wid_key:
        section3.append(('Body Width', body_wid_key, 'minmax'))
    section3 += [
        ('Heading',         '__heading__',    'heading'),
        ('Hourglass Area',  'hourglass_area', 'minmax'),
        ('Hourglass Ratio', 'hourglass_ratio','minmax'),
        ('Eccentricity',    'eccentricity',   'minmax'),
        ('Circularity',     'circularity',    'minmax'),
    ]
    section4 = [
        ('Path Efficiency', 'path_efficiency', 'direct'),
        ('Elongation',      'elongation',      'minmax'),
    ]

    sections = [section1, section2, section3, section4]
    all_rows = []
    section_breaks = []
    offset = 0
    for sec in sections:
        if offset > 0:
            section_breaks.append(offset)
        all_rows.extend(sec)
        offset += len(sec)

    n_rows = len(all_rows)
    n_tracks = len(track_names)

    # ---- Pre-extract raw bin means per track per row ----
    # raw_vals[row_idx][t_idx] = (n_bins,) array of bin means
    raw_vals = [[None] * n_tracks for _ in range(n_rows)]

    for t_idx in range(n_tracks):
        ta = track_arrays[t_idx]
        for r_idx, (label, key, ntype) in enumerate(all_rows):
            if key == '__heading__':
                vx_arr = ta.get(f'{bp}_vx')
                vy_arr = ta.get(f'{bp}_vy')
                if vx_arr is not None and vy_arr is not None:
                    vx_full = vx_arr[sleap_idxs].astype(float)
                    vy_full = vy_arr[sleap_idxs].astype(float)
                    hdg = np.degrees(np.arctan2(vy_full, vx_full)) % 360.0
                else:
                    hdg = np.full(len(sleap_idxs), np.nan)
                bin_means = np.array([np.nanmean(hdg[b]) if len(b) else np.nan
                                      for b in bins])
                raw_vals[r_idx][t_idx] = bin_means
                continue

            src = ta.get(key)
            if src is None:
                raw_vals[r_idx][t_idx] = np.full(n_bins, np.nan)
                continue

            full = src[sleap_idxs].astype(float)
            if 'abs' in ntype:
                full = np.abs(full)
            bin_means = np.array([np.nanmean(full[b]) if len(b) else np.nan
                                  for b in bins])

            # For p90 types, also need session-level p90
            if ntype.startswith('p90'):
                fin = full[np.isfinite(full)]
                sess_p90 = float(np.percentile(fin, 90)) if len(fin) else 1.0
                raw_vals[r_idx][t_idx] = (bin_means, sess_p90)
            else:
                raw_vals[r_idx][t_idx] = bin_means

    # ---- Compute global min/max across all tracks for minmax rows ----
    global_stats = [None] * n_rows
    for r_idx, (label, key, ntype) in enumerate(all_rows):
        if ntype in ('minmax', 'minmax_abs'):
            all_means = np.concatenate([raw_vals[r_idx][t]
                                        for t in range(n_tracks)
                                        if raw_vals[r_idx][t] is not None])
            fin = all_means[np.isfinite(all_means)]
            if len(fin):
                global_stats[r_idx] = (float(np.min(fin)), float(np.max(fin)))
            else:
                global_stats[r_idx] = (0.0, 1.0)
        elif ntype in ('p90', 'p90_abs'):
            # Use the max session_p90 across tracks for consistency
            p90s = []
            for t in range(n_tracks):
                v = raw_vals[r_idx][t]
                if v is not None and isinstance(v, tuple):
                    p90s.append(v[1])
            global_stats[r_idx] = max(p90s) if p90s else 1.0

    # ---- Render one page per animal ----
    row_labels = [r[0] for r in all_rows]

    with PdfPages(pdf_path) as pdf_out:
        for t_idx, tname in enumerate(track_names):
            matrix = np.full((n_rows, n_bins), np.nan)

            for r_idx, (label, key, ntype) in enumerate(all_rows):
                vals = raw_vals[r_idx][t_idx]
                if vals is None:
                    continue

                if ntype in ('minmax', 'minmax_abs'):
                    gmin, gmax = global_stats[r_idx]
                    matrix[r_idx] = _norm_minmax(vals, gmin, gmax)
                elif ntype in ('p90', 'p90_abs'):
                    bm = vals[0] if isinstance(vals, tuple) else vals
                    sess_p90 = global_stats[r_idx]
                    matrix[r_idx] = _norm_p90(bm, sess_p90)
                elif ntype == 'heading':
                    # [0, 360] → [0, 1]
                    matrix[r_idx] = np.clip(vals / 360.0, 0.0, 1.0)
                elif ntype == 'direct':
                    matrix[r_idx] = np.clip(vals, 0.0, 1.0)

            _render_oncoplot(matrix, row_labels, section_breaks, times,
                             f"Feature Profile \u2014 {tname}", pdf_out,
                             show_values=(n_bins <= 40))


# ---------------------------------------------------------------------------
# 5. Synchrony Oncoplot
# ---------------------------------------------------------------------------

def _write_sync_oncoplot(pair_arrays, times, sleap_idxs, track_names,
                         fps, pdf_path, status_cb):
    """Combined synchrony oncoplot: one page with all pair metrics."""
    if status_cb:
        status_cb("Graphs: synchrony oncoplot\u2026")

    n_tracks = len(track_names)
    if n_tracks < 2:
        return

    bins = _build_10s_bins(times)
    n_bins = len(bins)
    if n_bins == 0:
        return

    # Row specs per pair: (label_template, key_suffix, norm_type)
    row_specs = [
        ('Cov X',        'pos_covariance_x',  'minmax'),
        ('Cov Y',        'pos_covariance_y',  'minmax'),
        ('Corr X',       'pos_correlation_x', 'corr'),
        ('Corr Y',       'pos_correlation_y', 'corr'),
        ('Distance',     'inter_animal_dist', 'minmax'),
        ('Vel Cos Sim',  'velocity_cos_sim',  'corr'),
    ]

    pairs = list(combinations(range(n_tracks), 2))
    n_rows_per_pair = len(row_specs)
    n_rows = len(pairs) * n_rows_per_pair

    # ---- Extract bin means ----
    raw = np.full((n_rows, n_bins), np.nan)
    row_labels = []
    row_norms = []

    r = 0
    for tA, tB in pairs:
        pfx = f't{tA}_t{tB}'
        nA = track_names[tA] if tA < n_tracks else f't{tA}'
        nB = track_names[tB] if tB < n_tracks else f't{tB}'
        pair_label = f'{nA} vs {nB}'

        for spec_label, suffix, ntype in row_specs:
            key = f'{pfx}/{suffix}'
            arr = pair_arrays.get(key)
            if arr is not None:
                full = arr[sleap_idxs].astype(float)
                raw[r] = [np.nanmean(full[b]) if len(b) else np.nan
                          for b in bins]
            row_labels.append(f'{spec_label} ({pair_label})')
            row_norms.append(ntype)
            r += 1

    # ---- Compute global min/max for minmax rows ----
    # Group by suffix to share scale across pairs
    minmax_groups = {}
    for r_idx, ntype in enumerate(row_norms):
        if ntype == 'minmax':
            # Extract suffix from label
            suffix_key = row_labels[r_idx].split(' (')[0]
            minmax_groups.setdefault(suffix_key, []).append(r_idx)

    global_ranges = {}
    for grp_key, idxs in minmax_groups.items():
        all_v = np.concatenate([raw[i] for i in idxs])
        fin = all_v[np.isfinite(all_v)]
        if len(fin):
            global_ranges[grp_key] = (float(np.min(fin)), float(np.max(fin)))
        else:
            global_ranges[grp_key] = (0.0, 1.0)

    # ---- Normalise ----
    matrix = np.full((n_rows, n_bins), np.nan)
    for r_idx in range(n_rows):
        ntype = row_norms[r_idx]
        if ntype == 'corr':
            matrix[r_idx] = _norm_correlation(raw[r_idx])
        elif ntype == 'minmax':
            grp_key = row_labels[r_idx].split(' (')[0]
            gmin, gmax = global_ranges[grp_key]
            matrix[r_idx] = _norm_minmax(raw[r_idx], gmin, gmax)

    # Section breaks between pairs
    section_breaks = [i * n_rows_per_pair for i in range(1, len(pairs))]

    with PdfPages(pdf_path) as pdf_out:
        _render_oncoplot(matrix, row_labels, section_breaks, times,
                         "Synchrony Profile", pdf_out,
                         show_values=(n_bins <= 40))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_graphs(zone_summary_df, track_arrays, pair_arrays, frame_map,
                 track_names, node_names, fps, base_path, status_cb=None,
                 arena_cm=40, strip_cm=8, px_per_cm=1.0, arena_snapshot=None):
    """
    Generate all graph PDF files (5 PDFs, generated in parallel).

    Parameters
    ----------
    zone_summary_df : pd.DataFrame
        Columns: Track, Zone, Frame Count, Time in Zone (s), % of Session
    track_arrays    : dict[int, dict[str, np.ndarray]]
        Precomputed feature arrays; track_arrays[t][feature] -> (n_sleap_frames,)
    pair_arrays     : dict[str, np.ndarray]
        Pair-feature arrays keyed 't{A}_t{B}/feature_name'
    frame_map       : dict  {video_frame_idx -> sleap_data_idx}
    track_names     : list[str]
    node_names      : list[str]
    fps             : float
    base_path       : str
        Path WITHOUT extension used as prefix for output files.
    status_cb       : callable(str) or None
    arena_cm        : int — arena size in cm
    strip_cm        : int — border strip width in cm
    px_per_cm       : float — pixels per centimetre
    arena_snapshot  : ndarray or None — RGB image of arena cropped to ROI

    Returns
    -------
    list[str]  Paths of files that were written.
    """
    bp_dir = Path(base_path)
    stem   = bp_dir.stem
    outdir = bp_dir.parent

    times, sleap_idxs = _get_times_and_indices(frame_map, fps)
    if len(times) == 0:
        return []

    def _path(suffix):
        return str(outdir / f"{stem}{suffix}")

    # Define the five PDF tasks
    tasks = {}

    p_heat = _path("_graphs_heatmaps.pdf")
    tasks[p_heat] = lambda p=p_heat: _write_heatmaps(
        zone_summary_df, track_names, p, status_cb,
        arena_cm=arena_cm, strip_cm=strip_cm, arena_snapshot=arena_snapshot)

    p_cascade = _path("_graphs_cascade.pdf")
    tasks[p_cascade] = lambda p=p_cascade: _write_cascade_plot(
        track_arrays, times, sleap_idxs, track_names, node_names, p, status_cb)

    p_dist = _path("_graphs_distance.pdf")
    tasks[p_dist] = lambda p=p_dist: _write_distance_plot(
        pair_arrays, times, sleap_idxs, track_names, p, status_cb,
        px_per_cm=px_per_cm)

    p_onco = _path("_graphs_oncoplot.pdf")
    tasks[p_onco] = lambda p=p_onco: _write_oncoplot(
        track_arrays, times, sleap_idxs, track_names, node_names,
        fps, p, status_cb)

    p_sync = _path("_graphs_sync_oncoplot.pdf")
    tasks[p_sync] = lambda p=p_sync: _write_sync_oncoplot(
        pair_arrays, times, sleap_idxs, track_names, fps, p, status_cb)

    written = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): path for path, fn in tasks.items()}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
                written.append(path)
            except Exception as exc:
                if status_cb:
                    status_cb(f"Graph error ({Path(path).stem}, skipped): {exc}")

    plt.close("all")
    gc.collect()
    return written
