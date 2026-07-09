
import os
from logging import INFO
import sys
import traceback
import colorsys
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from pathlib import Path
import ffmpeg
import math
import statistics
import csv
from itertools import zip_longest
import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import messagebox
import threading
import copy
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

class SinglePairApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MOSAIC (for SLEAP)")
        self.root.geometry("400x600")

        # File paths
        self.video_paths = []
        self.h5_paths = []

        self.mouse_id_toggle = tk.BooleanVar(value=False)  # False = default, True = swapped

        # Buttons
        tk.Button(root, text="Select Video File", command=self.select_video).pack(pady=10)
        tk.Button(root, text="Select .h5 File", command=self.select_h5).pack(pady=10)
        tk.Button(root, text="Run Process", command=self.process_files).pack(pady=20)
        toggle_btn = tk.Checkbutton(root, text="Is subject Second Mouse in Vid?", variable=self.mouse_id_toggle, onvalue=True, offvalue=False)               
        toggle_btn.pack()

        # Display path labels
        self.video_label = tk.Label(root, text="No video selected")
        self.video_label.pack()

        self.h5_label = tk.Label(root, text="No .h5 file selected")
        self.h5_label.pack()
        
        self.text_widget = tk.Text(root, height=20, width=100, bg="black", fg="white", font=("Courier", 10))
        self.text_widget.pack(padx=10, pady=10, expand=True, fill='both')

        # Redirect standard output
        sys.stdout = self
        sys.stderr = self

        print("GUI Console is ready.")
        print("Welcome to the MOSAIC Console!")
        print("All information will be printed here!")

    def write(self, message):
        try:
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
        except Exception as e:
            sys.__stdout__.write(f"[write error] {e}\n")

    def flush(self):
        pass  # Required for file-like object, but can stay empty

    def select_video(self):
        paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if paths:
            self.video_paths = list(paths)
            self.video_label.config(text=f"{len(paths)} video(s) selected")

    def select_h5(self):
        paths = filedialog.askopenfilenames(
            title="Select .h5 Files",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        if paths:
            self.h5_paths = list(paths)
            self.h5_label.config(text=f"{len(paths)} H5 file(s) selected")

            
    def update_status(self, status, add_newline=True):
        # If add_newline is True, append a newline, otherwise don't
        if add_newline:
            status += "\n"

        # Insert the status into the Text widget
        self.text_widget.insert(tk.END, status)

        # Auto-scroll to the end of the Text widget
        self.text_widget.see(tk.END)

    def _show_completion_popup(self):
        messagebox.showinfo("Processing Complete", "All selected files have been processed successfully!")

        # Reset state for re-running
        self.video_paths = []
        self.h5_paths = []
        self.video_label.config(text="No video selected")
        self.h5_label.config(text="No .h5 file selected")
        self.update_status("\nReady for new inputs.\n")


    def _run_single_analysis(self, video_path, h5_path):
        subject_track = None
        stranger_track = None

        self.update_status("Process started.")
        self.update_status("Processing files...")

        self.update_status(f"Video path: {self.video_path}")
        self.update_status(f"H5 path: {self.h5_path}")

        if self.mouse_id_toggle.get():
            subject_track = 1
            stranger_track = 0
        else:
            subject_track = 0
            stranger_track = 1

        # Define subfolder for exports
        subfolder = "exports"
        # Create subfolder if it doesn't exist
        os.makedirs(subfolder, exist_ok=True)

        #selected paths
        video_path = self.video_path
        videofile = video_path
        filename = self.h5_path
        #analysis file name gets reused for data and figure exports
        base_name = os.path.splitext(os.path.basename(self.h5_path))[0]

        with h5py.File(filename, 'r') as file:
            dset_names = list(file.keys())
            occupancy_matrix = file['track_occupancy'][:]
            tracks_matrix = file['tracks'][:]
            raw_locations = file["tracks"][:].T
            node_names = [node.decode() for node in file["node_names"][:]]
                
        frame_count, node_count, _, instance_count = raw_locations.shape

        print(occupancy_matrix.shape)
        print(tracks_matrix.shape)

        print("===filename===")
        print(filename)
        print()

        print("===HDF5 datasets===")
        print(dset_names)
        print()

        print("===locations data shape===")
        print(raw_locations.shape)
        print()

        print("===nodes===")
        for i, name in enumerate(node_names):
            print(f"{i}: {name}")
        print()

        print("frame count:", frame_count)
        print("node count:", node_count)
        print("instance count:", instance_count)

        print("locations example:\n")
        print(raw_locations)
        
        def fill_null(Y, kind="linear"): #!!!DONE!!!
            import numpy as np
            #Fills missing values independently along each dimension after the first
            initial_shape = Y.shape #store initial shape
            Y = Y.reshape((initial_shape[0], -1)) #flatten after first dim.
            # Interpolate along each slice.
            for i in range(Y.shape[-1]):
                y = Y[:, i]
                # Build interpolant.
                x = np.flatnonzero(~np.isnan(y))
                f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
                # Fill missing
                xq = np.flatnonzero(np.isnan(y))
                y[xq] = f(xq)
                # Fill leading or trailing NaNs with the nearest non-NaN values
                mask = np.isnan(y)
                y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
                # Save slice
                Y[:, i] = y
            # Restore to initial shape.
            Y = Y.reshape(initial_shape)
            return y

        def hybrid_convergent_fill(trace, fps=24, pchip_time_s=0.25):
            """
            Robust, biologically appropriate filler for 1D motion traces.

            • Uses PCHIP only when both sides of a NaN gap have valid data (true interpolation).
            • Uses Kalman smoothing for one-sided or long gaps (directional prediction).
            • Finishes with linear + edge repair so no NaNs remain.

            Parameters
            ----------
            trace : np.ndarray
                1D coordinate array containing NaNs.
            fps : float
                Frame rate; used to adapt PCHIP window length (~¼ s default).
            pchip_time_s : float
                Maximum gap duration (seconds) handled purely by PCHIP.

            Returns
            -------
            filled : np.ndarray
                Trace with all NaNs replaced, real data preserved.
            """
            trace = np.asarray(trace, dtype=float)

            # --- Early exits ---
            if np.all(np.isnan(trace)):
                return np.zeros_like(trace)
            if np.sum(np.isfinite(trace)) < 3:
                val = np.nanmean(trace)
                return np.full_like(trace, val if np.isfinite(val) else 0.0)

            filled = trace.copy()
            n = len(trace)
            x = np.arange(n)
            isnan = np.isnan(trace)
            pchip_limit = max(2, int(round(fps * pchip_time_s)))

            nan_idx = np.where(isnan)[0]
            gaps = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)

            # --- Pass 1: internal short gaps with valid flanks → PCHIP ---
            for gap in gaps:
                start, end = gap[0], gap[-1]
                gap_len = end - start + 1
                if gap_len > pchip_limit:
                    continue  # handled by Kalman later

                left_idx = start - 1 if start > 0 else None
                right_idx = end + 1 if end < n - 1 else None
                if left_idx is None or right_idx is None:
                    continue  # need both flanks for interpolation

                x_known = x[np.isfinite(filled)]
                y_known = filled[np.isfinite(filled)]

                # true interpolation between flanks
                interp = PchipInterpolator(x_known, y_known, extrapolate=False)
                filled[start:end + 1] = interp(x[start:end + 1])

            # --- Pass 2: Kalman smoother for everything else ---
            valid_idx = np.where(np.isfinite(filled))[0]
            if len(valid_idx) >= 3:
                kf = KalmanFilter(
                    transition_matrices=[[1, 1], [0, 1]],
                    observation_matrices=[[1, 0]],
                    transition_covariance=np.eye(2) * 1e-3,
                    observation_covariance=np.eye(1) * 1e-2,
                    initial_state_mean=[filled[valid_idx[0]], 0],
                )
                fwd_mean, _ = kf.filter(filled)
                bwd_mean, _ = kf.filter(filled[::-1])
                smooth = (fwd_mean[:, 0] + bwd_mean[:, 0][::-1]) / 2
                filled[isnan] = smooth[isnan]

            # --- Pass 3: global linear + edge repair (guaranteed finite) ---
            if np.any(np.isnan(filled)):
                mask = np.isfinite(filled)
                if np.sum(mask) >= 2:
                    filled[np.isnan(filled)] = np.interp(
                        x[np.isnan(filled)], x[mask], filled[mask]
                    )
                # Edge padding
                if np.isnan(filled[0]):
                    first_valid = np.flatnonzero(np.isfinite(filled))[0]
                    filled[:first_valid] = filled[first_valid]
                if np.isnan(filled[-1]):
                    last_valid = np.flatnonzero(np.isfinite(filled))[-1]
                    filled[last_valid + 1 :] = filled[last_valid]

            return filled

        def smooth_sleap_allnodes(coords, med_win=3, sg_win=5, poly=2):
            coords = np.asarray(coords)
            smoothed = np.copy(coords)
            n_frames, n_nodes, _ = coords.shape

            for axis in range(2):     # x, y
                for n in range(n_nodes):
                    d = smoothed[:, n, axis]

                    # 1) median to kill spikes
                    if med_win > 1:
                        d = median_filter(d, size=med_win)

                    # 2) light S-G to make it pretty
                    if len(d) >= sg_win:
                        d = savgol_filter(d, sg_win, poly)

                    smoothed[:, n, axis] = d

            return smoothed

        duration = None

        def get_video_duration(video_path):
            try:
                probe = ffmpeg.probe(video_path)
                duration = float(probe['format']['duration'])
                return duration
            except ffmpeg.Error as e:
                print(f"Error: {e.stderr.decode()}")
                return None

        duration = get_video_duration(video_path)
        video_file = videofile
        threshTime = 1
        gateTime = 5
        fps = frame_count / duration
        iFPS = int(math.floor(fps))
        thresh = int(math.floor(threshTime*fps))
        aaThresh = 5*int(math.floor(fps))
        gate = int(math.floor(fps*gateTime)) #first frame to consider for proximity and behaviors

        # --- Predictive NaN fill per instance track ---
        print("Applying predictive NaN fill per tracked instance...")
        import numpy as np
        instance_count = raw_locations.shape[3]
        filled_locations = np.empty_like(raw_locations, dtype=float)

        for t in range(instance_count):
            print(f"  Filling instance {t+1}/{instance_count} ...")
            track_data = raw_locations[..., t]  # (frames, nodes, 2)
            filled_track = np.copy(track_data)
            for node_idx in range(track_data.shape[1]):      # iterate over nodes
                for axis_idx in range(track_data.shape[2]):  # x=0, y=1
                    filled_track[:, node_idx, axis_idx] = hybrid_convergent_fill(
                        track_data[:, node_idx, axis_idx], fps=fps
        )
            filled_locations[..., t] = filled_track

        print("Predictive fill complete.")

        # Copy for downstream processing
        locations = np.copy(filled_locations)

        def correct_identity_switches(
                locations,
                cm_index=6,
                ns_index=1,
                er_index=3,
                el_index=4,
                tb_index=2,
                switch_penalty=80.0,
                min_switch_gain=20.0,
                window_radius=6,
                min_run_frames=8,
                majority_window=9,
                debug=False
        ):
            """
            Conservative windowed identity-switch detector/corrector for 2 tracked animals.

            Parameters
            ----------
            locations : np.ndarray
                Shape (frames, nodes, 2, 2)
            cm_index, ns_index, er_index, el_index, tb_index : int
                Node indices used to build continuity signatures.
            switch_penalty : float
                Penalty for changing assignment state. Larger => fewer switches.
            min_switch_gain : float
                Additional gain required before preferring the swapped state.
            window_radius : int
                Number of frames on each side used for local window scoring.
                Total window length = 2*window_radius + 1.
            min_run_frames : int
                Minimum duration of a swapped/original run to keep.
            majority_window : int
                Odd window size for majority-vote smoothing of the inferred state path.
                Set to 0 or 1 to disable.
            debug : bool
                If True, print diagnostics.

            Returns
            -------
            corrected : np.ndarray
                Same shape as input, with track identities corrected.
            switch_frames : list[int]
                Frames where chosen identity state flips.
            state_path : np.ndarray
                0 = original assignment, 1 = swapped assignment
            """
            import numpy as np

            corrected = np.copy(locations)
            n_frames, n_nodes, _, n_instances = corrected.shape

            if n_instances != 2:
                raise ValueError("correct_identity_switches currently supports exactly 2 instances.")

            # Build a compact pose signature for each frame/track
            sig_nodes = [cm_index, ns_index, er_index, el_index, tb_index]
            sig_nodes = [n for n in sig_nodes if 0 <= n < n_nodes]

            # Explicit indexing to avoid advanced-indexing axis weirdness
            signatures = np.zeros((n_frames, 2, len(sig_nodes) * 2), dtype=float)
            for t in range(2):
                x_part = corrected[:, sig_nodes, 0, t]  # (frames, nodes)
                y_part = corrected[:, sig_nodes, 1, t]  # (frames, nodes)
                signatures[:, t, :] = np.concatenate([x_part, y_part], axis=1)

            def safe_sqdist(a, b):
                mask = np.isfinite(a) & np.isfinite(b)
                if not np.any(mask):
                    return 1e6
                d = a[mask] - b[mask]
                return float(np.dot(d, d) / max(mask.sum(), 1))

            def get_sig(frame_idx, state):
                """
                state 0 = original ordering
                state 1 = swapped ordering
                Returns (id0_sig, id1_sig)
                """
                if state == 0:
                    return signatures[frame_idx, 0], signatures[frame_idx, 1]
                else:
                    return signatures[frame_idx, 1], signatures[frame_idx, 0]

            def local_window_cost(center_f, state):
                """
                Cost of assuming a given state in a local window around center_f.
                Lower is better.
                """
                start = max(0, center_f - window_radius)
                end = min(n_frames - 1, center_f + window_radius)

                total = 0.0
                prev0, prev1 = get_sig(start, state)

                for f in range(start + 1, end + 1):
                    cur0, cur1 = get_sig(f, state)
                    total += safe_sqdist(prev0, cur0) + safe_sqdist(prev1, cur1)
                    prev0, prev1 = cur0, cur1

                return total

            # Initial local decision per frame
            raw_state = np.zeros(n_frames, dtype=int)

            for f in range(n_frames):
                keep_cost = local_window_cost(f, 0)
                swap_cost = local_window_cost(f, 1)

                # Only accept swapped state if it's substantially better
                if swap_cost + min_switch_gain < keep_cost:
                    raw_state[f] = 1
                else:
                    raw_state[f] = 0

            # Majority-vote smoothing across nearby frames
            state_path = raw_state.copy()
            if majority_window is not None and majority_window >= 3 and majority_window % 2 == 1:
                half = majority_window // 2
                smoothed = state_path.copy()
                for f in range(n_frames):
                    s = max(0, f - half)
                    e = min(n_frames, f + half + 1)
                    votes = state_path[s:e]
                    smoothed[f] = 1 if np.sum(votes) > (len(votes) / 2.0) else 0
                state_path = smoothed

            # Remove short runs
            runs = []
            run_start = 0
            for f in range(1, n_frames):
                if state_path[f] != state_path[f - 1]:
                    runs.append((run_start, f - 1, state_path[f - 1]))
                    run_start = f
            runs.append((run_start, n_frames - 1, state_path[-1]))

            for i, (s, e, st) in enumerate(runs):
                run_len = e - s + 1
                if run_len >= min_run_frames:
                    continue

                prev_state = runs[i - 1][2] if i > 0 else None
                next_state = runs[i + 1][2] if i < len(runs) - 1 else None

                if prev_state is not None and next_state is not None and prev_state == next_state:
                    state_path[s:e + 1] = prev_state
                elif prev_state is not None and next_state is None:
                    state_path[s:e + 1] = prev_state
                elif next_state is not None and prev_state is None:
                    state_path[s:e + 1] = next_state

            # Dynamic-programming pass on the smoothed local preferences
            # This makes final switches path-consistent and expensive.
            pref_cost = np.zeros((n_frames, 2), dtype=float)
            for f in range(n_frames):
                keep_cost = local_window_cost(f, 0)
                swap_cost = local_window_cost(f, 1)

                # Bias toward the majority-smoothed local decision
                if state_path[f] == 0:
                    pref_cost[f, 0] = keep_cost
                    pref_cost[f, 1] = swap_cost + min_switch_gain
                else:
                    pref_cost[f, 0] = keep_cost + min_switch_gain
                    pref_cost[f, 1] = swap_cost

            dp = np.full((n_frames, 2), np.inf, dtype=float)
            back = np.zeros((n_frames, 2), dtype=int)

            dp[0, 0] = pref_cost[0, 0]
            dp[0, 1] = pref_cost[0, 1] + switch_penalty

            for f in range(1, n_frames):
                for cur_state in (0, 1):
                    best_cost = np.inf
                    best_prev = 0
                    for prev_state in (0, 1):
                        penalty = switch_penalty if cur_state != prev_state else 0.0
                        c = dp[f - 1, prev_state] + pref_cost[f, cur_state] + penalty
                        if c < best_cost:
                            best_cost = c
                            best_prev = prev_state
                    dp[f, cur_state] = best_cost
                    back[f, cur_state] = best_prev

            final_state = np.zeros(n_frames, dtype=int)
            final_state[-1] = int(np.argmin(dp[-1]))
            for f in range(n_frames - 2, -1, -1):
                final_state[f] = back[f + 1, final_state[f + 1]]

            # One more short-run cleanup on final path
            runs = []
            run_start = 0
            for f in range(1, n_frames):
                if final_state[f] != final_state[f - 1]:
                    runs.append((run_start, f - 1, final_state[f - 1]))
                    run_start = f
            runs.append((run_start, n_frames - 1, final_state[-1]))

            for i, (s, e, st) in enumerate(runs):
                run_len = e - s + 1
                if run_len >= min_run_frames:
                    continue

                prev_state = runs[i - 1][2] if i > 0 else None
                next_state = runs[i + 1][2] if i < len(runs) - 1 else None

                if prev_state is not None and next_state is not None and prev_state == next_state:
                    final_state[s:e + 1] = prev_state
                elif prev_state is not None and next_state is None:
                    final_state[s:e + 1] = prev_state
                elif next_state is not None and prev_state is None:
                    final_state[s:e + 1] = next_state

            switch_frames = [f for f in range(1, n_frames) if final_state[f] != final_state[f - 1]]

            # Apply swap to corrected output
            for f in range(n_frames):
                if final_state[f] == 1:
                    corrected[f, :, :, [0, 1]] = corrected[f, :, :, [1, 0]]

            if debug:
                print("Identity correction summary:")
                print("  window_radius =", window_radius)
                print("  switch_penalty =", switch_penalty)
                print("  min_switch_gain =", min_switch_gain)
                print("  min_run_frames =", min_run_frames)
                print("  total switch points =", len(switch_frames))
                if len(switch_frames) > 0:
                    print("  first switch frames =", switch_frames[:20])

            return corrected, switch_frames, final_state

        ns_index = 1
        el_index = 4
        er_index = 3
        cm_index = 6
        hl_index = 8
        hr_index = 7
        tb_index = 2
        tm_index = 5
        te_index = 0
                        
        AXIS_Y = 1
        AXIS_X = 0

        locations, switch_frames, state_path = correct_identity_switches(
            locations,
            cm_index=cm_index,
            ns_index=ns_index,
            er_index=er_index,
            el_index=el_index,
            tb_index=tb_index,
            switch_penalty=120.0,
            min_switch_gain=35.0,
            window_radius=8,
            min_run_frames=12,
            majority_window=11,
            debug=True
        )

        # --- Optional smoothing (same logic as before) ---
        print("Smoothing all nodes per track...")
        for t in range(instance_count):
            coords = locations[..., t]  # (frames, nodes, 2)
            smoothed = smooth_sleap_allnodes(coords)
            locations[..., t] = smoothed

        print("Smoothing complete.")

        print("Identity-correction switch frames:", switch_frames[:20], "..." if len(switch_frames) > 20 else "")
        print("Total identity switch corrections:", len(switch_frames))

        print("NaNs remaining:", np.isnan(locations).sum())
        print("Mean:", np.nanmean(locations))
        print("Std:", np.nanstd(locations))


        # before smoothing
        raw = raw_locations[:, node_idx, axis_idx, 0]
        # after smoothing
        sm = locations[:, node_idx, axis_idx, 0]
        import matplotlib
        matplotlib.use("Agg")  # must be BEFORE importing pyplot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(raw, alpha=0.4, label="raw")
        plt.plot(sm, linewidth=2, label="smoothed")
        plt.legend()
        plt.title("Node {} axis {}".format(node_idx, axis_idx))
        plt.tight_layout()
        plt.savefig(fname = "rawvsmoothfig", dpi=150)
        plt.close()

        cm_loc = locations[:, cm_index, :, :]        
                            
        cm_x_id1 = locations[:, cm_index, AXIS_X, subject_track]
        cm_x_id2 = locations[:, cm_index, AXIS_X, stranger_track]
        ns_x_id1 = locations[:, ns_index, AXIS_X, subject_track]
        ns_x_id2 = locations[:, ns_index, AXIS_X, stranger_track]
        tb_x_id1 = locations[:, tb_index, AXIS_X, subject_track]
        tb_x_id2 = locations[:, tb_index, AXIS_X, stranger_track]
                            
        cm_y_id1 = locations[:, cm_index, AXIS_Y, subject_track]
        cm_y_id2 = locations[:, cm_index, AXIS_Y, stranger_track]
        ns_y_id1 = locations[:, ns_index, AXIS_Y, subject_track]
        ns_y_id2 = locations[:, ns_index, AXIS_Y, stranger_track]
        tb_y_id1 = locations[:, tb_index, AXIS_Y, subject_track]
        tb_y_id2 = locations[:, tb_index, AXIS_Y, stranger_track]

        def prior_second_to_corr_frame(): #!!!DONE!!!
            pastSecondFrameChunk = []
            for i in range(0, len(finalframechunk)): 
                tempStartFrame = finalframechunk[i][0] #set the current frame equal to the frame at which proximity begins
                pastSecondFrameChunk.append([tempStartFrame - iFPS, tempStartFrame]) #set a new frame range, [initial, final] in which final is the proximity range's initial and initial is one second worth of frames prior
            return pastSecondFrameChunk

        def distance(x1, y1, x2, y2): #!!!DONE!!!
            #distance = sqrt(x^2 + y^2)
            distance = np.sqrt(((abs(x1 - x2))**2 + (abs(y1 - y2))**2))
            return distance
            
        import numpy as np

        def dynamic_sniff_range(coords=locations[:,:,:, subject_track], hl_index=8, hr_index=7, drop_percentiles=(20, 80)):
            """
            Estimate a dynamic sniff range (dsr) from body width (hip-to-hip distance).

            Parameters
            ----------
            coords : np.ndarray
                Array of shape (frames, nodes, 2) containing x,y coordinates.
            hl_index : int
                Index of the left hip node.
            hr_index : int
                Index of the right hip node.
            drop_percentiles : tuple(int, int)
                Percentile range to exclude extreme postures (default 20–80).

            Returns
            -------
            float
                Dynamic sniff range (1× median body width in pixels).
            """

            # Hip-to-hip distances across all frames
            hip_dists = np.linalg.norm(coords[:, hl_index, :] - coords[:, hr_index, :], axis=1)

            # Filter out extreme poses
            low, high = np.percentile(hip_dists, drop_percentiles)
            filtered = hip_dists[(hip_dists >= low) & (hip_dists <= high)]

            # Median of the filtered distances
            dsr = np.median(filtered)
            print("Dynamic Sniff Range (dsr):", dsr)
            return dsr

        def pixel_to_cm_scale(dsr, real_mouse_width_cm=3.0):
            """
            Estimate the real-world cm/pixel scale from the dynamic sniff range (hip width).

            Parameters
            ----------
            dsr : float
                Dynamic sniff range in pixels (median hip–hip width).
            real_mouse_width_cm : float, optional
                Estimated real-world body width of the mouse in centimeters.
                Defaults to 3.0 cm for adult C57BL/6J.

            Returns
            -------
            scale : float
                Conversion factor in cm per pixel (cm/px).
            px_per_cm : float
                Conversion factor in pixels per cm (px/cm).
            """
            cm_per_px = real_mouse_width_cm / dsr
            px_per_cm = dsr / real_mouse_width_cm
            return cm_per_px, px_per_cm

        dsr = dynamic_sniff_range()
        # --- Compute body scale (dsr in px and cm/px) ---
        cm_per_px, px_per_cm = pixel_to_cm_scale(dsr, real_mouse_width_cm=3.0)

        print(f"Dynamic Sniff Range: {dsr:.2f}px ≈ {dsr * cm_per_px:.2f} cm")
        print(f"Scale: {cm_per_px:.4f} cm/px  ({px_per_cm:.2f} px/cm)")


        def prox_frames(node_loc_0, node_loc_1): #!!!DONE!!!
            #lists [initial frame, final frame] for proximity in a multidimensional array
            near = 0
            frameChunk = []
            frameClip = []
            for i in range(gate, frame_count): #gated since arm in video tends to produce extra tracks for whatever reason. Training data consisting of 1k+ frames might not have this issue, but this is still conservative either way
                if distance(cm_x_id1[i], cm_y_id1[i], cm_x_id2[i], cm_y_id2[i]) <= (6*dsr):
                    if near == 0:
                        frameClip.append(i)
                        near = 1
                    else:
                        frameClip.append(i)
                        near = 0
                        frameChunk.append(frameClip)
                        frameClip = []            
            return frameChunk
        finalframechunk = prox_frames(cm_loc[:, :, 0], cm_loc[:, :, 1])

        timePoints = []
        frameOverlaps = []
        timesTogether = []
        sumTotalTime = 0
        for i in range(0, len(finalframechunk)): #arbitray 5 second gating for the beginning of the video to disclude mistaken tracking of arm
            frameDiff = finalframechunk[i][1] - finalframechunk[i][0] 
            frameOverlaps.append(frameDiff) #frames spent together between framepoints
            timePoints.append([finalframechunk[i][0] / fps, finalframechunk[i][1] / fps]) #timepoints during which proximity occurs
            timesTogether.append(frameDiff / fps) #Time spent between timepoints
            sumTotalTime = sumTotalTime + timesTogether[i] #sums the total time spent together within the video

        def app_angle(x1, y1, x2, y2):
            return math.degrees(math.atan2(y2 - y1, x2 - x1))

        def rel_angle(a1Theta, a2Theta):
            diff = abs(a1Theta - a2Theta) % 360
            return min(diff, 360 - diff)  # minimal angular difference

        def ps_velocity(x1, y1, x2, y2, frames): #!!!DONE!!!
            #velocity = distance / time
            velocity = distance(x1, y1, x2, y2) / frames
            return velocity

        def nose_polar_angle():
            nose_polar_angle = []
            for f in range (gate, frame_count):
                nose_polar_angle.append(app_angle(locations[f, cm_index, AXIS_X, subject_track], 
                                        locations[f, cm_index, AXIS_Y, subject_track], 
                                        locations[f, ns_index, AXIS_X, subject_track], 
                                        locations[f, ns_index, AXIS_Y, subject_track]))
            return nose_polar_angle
        nose_polar_angle = nose_polar_angle()

        def closest_to_nose(f):
            subj_nose_x = locations[f, ns_index, AXIS_X, subject_track]
            subj_nose_y = locations[f, ns_index, AXIS_Y, subject_track]

            min_dist = float('inf')
            closest_node = -1

            for node_index in range(node_count):  # typically 7 nodes
                stranger_x = locations[f, node_index, AXIS_X, stranger_track]
                stranger_y = locations[f, node_index, AXIS_Y, stranger_track]

                if not np.isnan(stranger_x) and not np.isnan(stranger_y):
                    d = distance(subj_nose_x, subj_nose_y, stranger_x, stranger_y)
                    if d < min_dist:
                        min_dist = d
                        closest_node = node_index
            return min_dist, closest_node

        import numpy as np

        def filter_behavior(arr, duration_thresh_s=1.0, between_thresh_s=0.3, fps=fps, leniency=0.1):
            """
            Post-process a binary behavior array by merging close bouts
            and removing short ones, allowing leniency for frame irregularities.

            Parameters
            ----------
            arr : array-like of int/bool
                Binary sequence (e.g., per-frame behavior flags).
            duration_thresh_s : float
                Minimum duration (in seconds) a bout must last to be kept.
            between_thresh_s : float
                Maximum time (in seconds) between two bouts to merge them.
            fps : int or float
                Nominal frames per second of the recording.
            leniency : float
                Fractional tolerance (e.g., 0.1 = ±10%) to accommodate
                frame rate variations or dropped frames.

            Returns
            -------
            filtered : np.ndarray
                New array with close bouts merged and short bouts removed.
            """

            arr = np.array(arr, dtype=int)
            filtered = arr.copy()

            # Convert thresholds to frames with leniency
            duration_thresh = int(round(duration_thresh_s * fps * (1 - leniency)))
            between_thresh  = int(round(between_thresh_s * fps * (1 + leniency)))

            # --- Pass 1: Merge bouts separated by short gaps ---
            ones = np.where(filtered == 1)[0]
            if len(ones) > 1:
                for i in range(len(ones) - 1):
                    gap = ones[i + 1] - ones[i] - 1
                    if 0 < gap <= between_thresh:
                        filtered[ones[i] + 1 : ones[i + 1]] = 1

            # --- Pass 2: Remove bouts shorter than minimum duration ---
            in_bout = False
            start = 0
            for i in range(len(filtered)):
                if filtered[i] == 1 and not in_bout:
                    in_bout = True
                    start = i
                elif (filtered[i] == 0 or i == len(filtered) - 1) and in_bout:
                    end = i if filtered[i] == 0 else i + 1
                    duration = end - start
                    if duration < duration_thresh:
                        filtered[start:end] = 0
                    in_bout = False

            return filtered

        def filter_bmanifest(
                bManifest,
                fps,
                duration_thresh_s=1.0,
                between_thresh_s=0.3,
                leniency=0.1,
                skip_columns=("binauralAud", "monauralAud", "rearAud",
                              "binocularVis", "monocularVis", "noVis")
        ):
            """
            Apply temporal smoothing to binary behavioral columns.
            """
            import numpy as np

            behavior_index = {
                "approach": 1,
                "follow": 2,
                "chase": 3,
                "sniff": 4,
                "hh": 5,
                "ho": 6,
                "activeAvoid": 7,
                "passiveAvoid": 8,
                "flee": 9,
                "disengage": 10,
                "stationaryProx": 11,
                "socialOrient": 12,
                "proximity": 13,
                "binauralAud": 14,
                "monauralAud": 15,
                "rearAud": 16,
                "binocularVis": 17,
                "monocularVis": 18,
                "noVis": 19
            }

            bManifest = np.array(bManifest, dtype=int)
            filtered_manifest = bManifest.copy()

            for name, col in behavior_index.items():
                if name in skip_columns:
                    continue

                filtered_col = filter_behavior(
                    bManifest[:, col],
                    duration_thresh_s=duration_thresh_s,
                    between_thresh_s=between_thresh_s,
                    fps=fps,
                    leniency=leniency
                )
                filtered_manifest[:, col] = filtered_col

            return filtered_manifest

        priorSecondFrameChunk = prior_second_to_corr_frame()
        def det_instigator(): ###ADD APPROACH ANGLES COMPARISON!!!###
            instigator = []
            for j in range(0, len(priorSecondFrameChunk)):
                # Extract positions from the chunk
                frame_0, frame_1 = priorSecondFrameChunk[j]
                frame_0 = int(math.trunc(frame_0*fps))
                frame_1 = int(math.trunc(frame_1*fps))
                if frame_1 >= frame_count:
                    frame_1 = frame_count-1
                if frame_0 >= frame_count:
                    frame_0 = frame_count-1
                # Calculate velocities or use 0 if the frame is 0
                if frame_0 != 0:
                    m1_velocity = ps_velocity(locations[frame_0, ns_index, AXIS_X, subject_track], 
                                                locations[frame_0, ns_index, AXIS_Y, subject_track], 
                                                locations[frame_1, ns_index, AXIS_X, subject_track], 
                                                locations[frame_1, ns_index, AXIS_Y, subject_track], iFPS)
                    m2_velocity = ps_velocity(locations[frame_0, ns_index, AXIS_X, stranger_track], 
                                                locations[frame_0, ns_index, AXIS_Y, stranger_track], 
                                                locations[frame_1, ns_index, AXIS_X, stranger_track], 
                                                locations[frame_1, ns_index, AXIS_Y, stranger_track], iFPS)
                else:
                    m1_velocity = m2_velocity = 0
                # Determine instigator based on velocities
                if m1_velocity > m2_velocity:
                    instigator.append("Subject")
                elif m2_velocity > m1_velocity:
                    instigator.append("Stranger")
                else:
                    instigator.append("Shared")
            return instigator

        def det_instigator(
                intervals,
                bout_type="generic",
                pre_window_s=0.75,
                min_score_frac=0.12,
                shared_frac=0.85
        ):
            """
            Determine which mouse instigated each bout.

            Instigation is defined from the PRE-BOUT window, not the full bout:
              - movement toward the partner's behavior-relevant target
              - reduction in target distance
              - orientation toward the target at onset

            Parameters
            ----------
            intervals : list of (start_time, end_time)
                Bout intervals in seconds.
            bout_type : str
                One of:
                    "proximity" -> center-mass to center-mass
                    "sniff"     -> nose to partner head centroid
                    "hh"        -> head centroid to head centroid
                    "ho"        -> nose to nose
                    "generic"   -> nose to partner center-mass
            pre_window_s : float
                Length of the pre-bout window used to determine instigation.
            min_score_frac : float
                Minimum score as fraction of BL required to call a non-shared instigator.
            shared_frac : float
                If loser_score / winner_score >= shared_frac, call "Shared".

            Returns
            -------
            instigators : list[str]
                "Subject", "Stranger", or "Shared" for each bout.
            """
            import numpy as np

            instigators = []
            n_frames = locations.shape[0]

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            min_score = min_score_frac * BL
            pre_window = max(2, int(round(pre_window_s * fps)))

            head_nodes = [ns_index, er_index, el_index]

            def head_centroid(track, f):
                return np.array([
                    np.nanmean(locations[f, head_nodes, AXIS_X, track]),
                    np.nanmean(locations[f, head_nodes, AXIS_Y, track])
                ])

            def get_source_target(track_self, track_other, f, mode):
                """
                Returns source_point, target_point, source_mode
                """
                nose_self = locations[f, ns_index, :, track_self]
                nose_other = locations[f, ns_index, :, track_other]
                cm_self = locations[f, cm_index, :, track_self]
                cm_other = locations[f, cm_index, :, track_other]
                head_self = head_centroid(track_self, f)
                head_other = head_centroid(track_other, f)

                if mode == "proximity":
                    return cm_self, cm_other, "cm"
                elif mode == "sniff":
                    return nose_self, head_other, "nose"
                elif mode == "hh":
                    return head_self, head_other, "head"
                elif mode == "ho":
                    return nose_self, nose_other, "nose"
                else:
                    return nose_self, cm_other, "nose"

            def get_heading_angle(track, f):
                return app_angle(
                    locations[f, cm_index, AXIS_X, track],
                    locations[f, cm_index, AXIS_Y, track],
                    locations[f, ns_index, AXIS_X, track],
                    locations[f, ns_index, AXIS_Y, track]
                )

            def point_angle(source_xy, target_xy):
                return app_angle(source_xy[0], source_xy[1], target_xy[0], target_xy[1])

            def approach_score(track_self, track_other, f0, f1, mode):
                """
                Score how much track_self instigates toward track_other in pre-window.
                """
                s0, t0, src_mode = get_source_target(track_self, track_other, f0, mode)
                s1, t1, _ = get_source_target(track_self, track_other, f1, mode)

                # Distance reduction (positive means moved closer)
                d0 = np.linalg.norm(s0 - t0)
                d1 = np.linalg.norm(s1 - t1)
                delta_close = d0 - d1

                # Movement projected toward initial target
                move_vec = s1 - s0
                to_target_vec = t0 - s0
                to_target_norm = np.linalg.norm(to_target_vec)
                if to_target_norm < 1e-9:
                    proj_toward = 0.0
                else:
                    proj_toward = float(np.dot(move_vec, to_target_vec / to_target_norm))

                # Orientation at bout onset
                body_ang = get_heading_angle(track_self, f1)
                target_ang = point_angle(s1, t1)
                face_err = rel_angle(body_ang, target_ang)
                orient_bonus = 0.25 * BL if face_err <= 55 else 0.0

                # Mild onset weighting: later pre-bout distance matters a little more
                onset_bonus = 0.0
                if delta_close > 0 and d1 < d0:
                    onset_bonus = 0.10 * BL

                score = (1.0 * delta_close) + (0.75 * proj_toward) + orient_bonus + onset_bonus
                return score

            for start_t, end_t in intervals:
                start_f = int(round(start_t * fps))
                start_f = max(1, min(start_f, n_frames - 1))

                f0 = max(0, start_f - pre_window)
                f1 = start_f

                if (f1 - f0) < 2:
                    instigators.append("Shared")
                    continue

                subj_score = approach_score(subject_track, stranger_track, f0, f1, bout_type)
                strg_score = approach_score(stranger_track, subject_track, f0, f1, bout_type)

                # Require meaningful evidence
                if subj_score < min_score and strg_score < min_score:
                    instigators.append("Shared")
                    continue

                winner = max(subj_score, strg_score)
                loser = min(subj_score, strg_score)

                # If both are substantial and close, call shared
                if winner > 0 and (loser / winner) >= shared_frac:
                    instigators.append("Shared")
                elif subj_score > strg_score:
                    instigators.append("Subject")
                elif strg_score > subj_score:
                    instigators.append("Stranger")
                else:
                    instigators.append("Shared")

            return instigators

        def det_instigator_boutwise(sniff_intervals, fps, sniff_array, locations,
                            subject_track, stranger_track,
                            pre_window_s=1.0):
            """
            Determine a single instigator per sniff bout based on movement
            during the 1 second *before* the bout starts, labeling all frames
            of the bout with that instigator.

            Parameters
            ----------
            sniff_intervals : list of (start_time, end_time)
                Bout start and end times in seconds.
            fps : float
                Video frame rate.
            sniff_array : array-like of shape (frames, 3)
                [frame, label, flag] from det_sniff().
            locations : np.ndarray
                Full coordinate array (frames, nodes, 2, instances).
            subject_track, stranger_track : int
                Track indices.
            pre_window_s : float
                Duration (seconds) of the pre-bout window used to infer initiator.

            Returns
            -------
            bout_instigators : list[str]
                Per-bout instigator label ("Subject", "Stranger", "Shared").
            frame_instigators : np.ndarray
                Array of per-frame instigator labels aligned with sniff_array frames.
            """
            import numpy as np

            frame_instigators = np.array(["None"] * len(sniff_array), dtype=object)
            bout_instigators = []

            pre_window = int(round(pre_window_s * fps))
            n_frames = locations.shape[0]

            for (start_t, end_t) in sniff_intervals:
                start_f = int(start_t * fps)
                end_f = int(end_t * fps)
                pre_start = max(0, start_f - pre_window)
                pre_frames = np.arange(pre_start, start_f)

                if len(pre_frames) < 2:
                    bout_instigators.append("Shared")
                    continue

                # --- Extract coordinates ---
                subj_nose = locations[pre_frames, 2, :, subject_track]
                strg_nose = locations[pre_frames, 2, :, stranger_track]
                subj_cm   = locations[pre_frames, 6, :, subject_track]
                strg_cm   = locations[pre_frames, 6, :, stranger_track]

                # --- Distance to partner's center over pre-window ---
                subj_d = np.linalg.norm(subj_nose - strg_cm, axis=1)
                strg_d = np.linalg.norm(strg_nose - subj_cm, axis=1)

                # --- Displacement magnitudes ---
                subj_speed = np.mean(np.linalg.norm(np.diff(subj_nose, axis=0), axis=1))
                strg_speed = np.mean(np.linalg.norm(np.diff(strg_nose, axis=0), axis=1))

                # --- Directional approach (negative = moving closer) ---
                subj_delta = subj_d[-1] - subj_d[0]
                strg_delta = strg_d[-1] - strg_d[0]

                # --- Decide instigator ---
                if subj_delta < strg_delta - 0.1 and subj_speed > 0.5 * strg_speed:
                    instigator = "Subject"
                elif strg_delta < subj_delta - 0.1 and strg_speed > 0.5 * subj_speed:
                    instigator = "Stranger"
                else:
                    instigator = "Shared"

                bout_instigators.append(instigator)
                frame_instigators[start_f:end_f + 1] = instigator

            return bout_instigators, frame_instigators

        def det_proximity():
            self.update_status("Proximity: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            prox_array = []
            dsr_local = dynamic_sniff_range()
            max_dist = 3 * dsr_local
            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)
                d = distance(cm_x_id1[f], cm_y_id1[f], cm_x_id2[f], cm_y_id2[f])
                flag = 1 if d <= max_dist else 0
                prox_array.append([f, 'prox' if flag else 'n', flag])
            self.update_status("#     [100%]")
            return prox_array


        def summarize_proximity(prox_array, fps):
                """
                Compute total time, bout count, and durations from det_proximity() output.
                prox_array: list of [frame, 'prox' or 'n', 1/0]
                fps: frames per second
                Returns:
                  total_time_s, num_bouts, durations_s, intervals_s
                """
                intervals = []
                in_bout = False
                start = None

                for frame, _, flag in prox_array:
                    if flag == 1 and not in_bout:
                        in_bout = True
                        start = frame
                    elif flag == 0 and in_bout:
                        in_bout = False
                        end = frame
                        intervals.append((start / fps, end / fps))

                if in_bout:  # handle final bout
                    intervals.append((start / fps, prox_array[-1][0] / fps))

                durations = [(end - start) for start, end in intervals]
                total_time = sum(durations)
                return total_time, len(intervals), durations, intervals

        prox_array = det_proximity()

        def det_vis():
            self.update_status("Vis Cone: ", add_newline=False)
            ticker = math.floor(frame_count/9)
            """
            Classify frames into visual field categories:
                - Binocular (B)
                - Monocular Left (ML)
                - Monocular Right (MR)
                - None (N)
            Gated by visual clarity: target must be within 10 × dsr.
            """

            vis_array = []
            dsr = dynamic_sniff_range()
            max_view_dist = 10 * dsr  # clarity cutoff

            for f in range(gate, frame_count):
                if f % ticker == 0:
                    self.update_status("#", add_newline=False)
                # Subject heading CM→NS
                subj_theta = app_angle(cm_x_id1[f], cm_y_id1[f],
                                        ns_x_id1[f], ns_y_id1[f])

                # Ear baseline (for left/right classification)
                left_ear = (locations[f, 1, AXIS_X, subject_track],
                            locations[f, 1, AXIS_Y, subject_track])
                right_ear = (locations[f, 2, AXIS_X, subject_track],
                                locations[f, 2, AXIS_Y, subject_track])
                ear_vec = np.array([right_ear[0] - left_ear[0],
                                    right_ear[1] - left_ear[1]])

                # Stranger nodes to test
                targets = [
                    (cm_x_id2[f], cm_y_id2[f], 'CM'),
                    (ns_x_id2[f], ns_y_id2[f], 'NS')
                ]

                best_label = 'N'
                best_weight = 0  # 0=none, 1=monocular, 2=binocular

                for (tx, ty, name) in targets:
                    if f % ticker == 0:
                        self.update_status("#", add_newline=False)
                    if np.isnan(tx) or np.isnan(ty):
                        continue

                    # Distance gate (too far = not visible)
                    d = distance(cm_x_id1[f], cm_y_id1[f], tx, ty)
                    if d > max_view_dist:
                        continue

                    # Angular relation
                    target_theta = app_angle(cm_x_id1[f], cm_y_id1[f], tx, ty)
                    angle_diff = rel_angle(subj_theta, target_theta)

                    # Left vs Right using cross product
                    cm_vec = np.array([tx - cm_x_id1[f], ty - cm_y_id1[f]])
                    cross = ear_vec[0] * cm_vec[1] - ear_vec[1] * cm_vec[0]
                    side = 'L' if cross > 0 else 'R'

                    # Assign categories (priority weighted)
                    if angle_diff <= 20 and best_weight < 2:
                        best_label = 'B'     # binocular, no L/R
                        best_weight = 2
                    elif angle_diff <= 120 and best_weight < 1:
                        best_label = 'M' + side   # monocular left/right
                        best_weight = 1

                vis_array.append([f, best_label, best_weight])
            self.update_status("#     [100%]")
            return vis_array

        dsr = dynamic_sniff_range()

        def det_aud():
            self.update_status("Aud Cone: ", add_newline=False)
            ticker = math.floor(frame_count/9)
            """
            Classify frames into auditory field categories:
                - B  = Bin-aural (front, ±60°)
                - ML = Mon-aural Left (60°–150°)
                - MR = Mon-aural Right (60°–150°)
                - R  = Rear (>150°)
                - N  = None (too far to be heard)
            Gated by distance = 10 × dsr (similar to vision).
            """

            aud_array = []
            dsr = dynamic_sniff_range()
            max_hear_dist = 10 * dsr  # approximate detection radius

            for f in range(gate, frame_count):
                if f % ticker == 0:
                    self.update_status("#", add_newline=False)
                # Subject heading vector (CM→NS)
                subj_theta = app_angle(cm_x_id1[f], cm_y_id1[f],
                                        ns_x_id1[f], ns_y_id1[f])

                # Ear baseline vector
                left_ear = (locations[f, 1, AXIS_X, subject_track],
                            locations[f, 1, AXIS_Y, subject_track])
                right_ear = (locations[f, 2, AXIS_X, subject_track],
                                locations[f, 2, AXIS_Y, subject_track])
                ear_vec = np.array([right_ear[0] - left_ear[0],
                                    right_ear[1] - left_ear[1]])

                # Stranger CM as sound source
                tx, ty = cm_x_id2[f], cm_y_id2[f]
                if np.isnan(tx) or np.isnan(ty):
                    aud_array.append([f, 'N', 0])
                    continue

                # Distance gating
                d = distance(cm_x_id1[f], cm_y_id1[f], tx, ty)
                if d > max_hear_dist:
                    aud_array.append([f, 'N', 0])
                    continue

                # Angular relation
                target_theta = app_angle(cm_x_id1[f], cm_y_id1[f], tx, ty)
                angle_diff = rel_angle(subj_theta, target_theta)

                # Left vs Right (cross product with ear baseline)
                cm_vec = np.array([tx - cm_x_id1[f], ty - cm_y_id1[f]])
                cross = ear_vec[0] * cm_vec[1] - ear_vec[1] * cm_vec[0]
                side = 'L' if cross > 0 else 'R'

                # Classification
                if angle_diff <= 60:
                    label, weight = 'B', 3
                elif angle_diff <= 150:
                    label, weight = 'M' + side, 2
                else:
                    label, weight = 'R', 1

                aud_array.append([f, label, weight])
            self.update_status("#     [100%]")
            return aud_array

        ##Determine Head-On Interaction##
        def det_hh_ho():
            """
            Improved detector for:
              HH = head-to-head / head-region engagement
              HO = head-on / nose-to-nose / mutual face-to-face

            Logic
            -----
            HH:
              - either nose is close to the other animal's head region
              - OR both head centroids are close
              - with mutual head engagement (each animal oriented toward the other)

            HO:
              - direct nose-to-nose proximity
              - stronger mutual facing requirement
              - headings should be roughly opposite (face-to-face), not parallel

            Returns
            -------
            final_hh, final_ho : list
                Per-frame arrays of [frame, label, flag]
            """
            self.update_status("HH & HO: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))

            dsr_local = dynamic_sniff_range()

            # --- thresholds ---
            # HH can be a little broader than HO
            max_dist_hh = 0.90 * dsr_local
            max_dist_ho = 0.55 * dsr_local

            # head region nodes
            # assumes your current node map:
            # ns=1, er=3, el=4
            head_nodes = [ns_index, er_index, el_index]

            subj_nose = locations[:, ns_index, :, subject_track]
            strg_nose = locations[:, ns_index, :, stranger_track]

            # head centroids
            # head centroids, explicitly preserving frame axis
            subj_head_cent = np.stack([
                np.nanmean(locations[:, head_nodes, AXIS_X, subject_track], axis=1),
                np.nanmean(locations[:, head_nodes, AXIS_Y, subject_track], axis=1)
            ], axis=1)

            strg_head_cent = np.stack([
                np.nanmean(locations[:, head_nodes, AXIS_X, stranger_track], axis=1),
                np.nanmean(locations[:, head_nodes, AXIS_Y, stranger_track], axis=1)
            ], axis=1)

            print("subj_head_cent shape:", subj_head_cent.shape)
            print("strg_head_cent shape:", strg_head_cent.shape)

            final_hh, final_ho = [], []

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                # --- core positions ---
                s_nose = subj_nose[f]
                t_nose = strg_nose[f]
                s_head = subj_head_cent[f]
                t_head = strg_head_cent[f]
                s_cm = locations[f, cm_index, :, subject_track]
                t_cm = locations[f, cm_index, :, stranger_track]

                # --- basic distances ---
                ns_ns_d = np.linalg.norm(s_nose - t_nose)
                hh_centroid_d = np.linalg.norm(s_head - t_head)

                # nose to other head-region distances (symmetric)
                s_to_t_head = [
                    np.linalg.norm(s_nose - locations[f, node, :, stranger_track])
                    for node in head_nodes
                ]
                t_to_s_head = [
                    np.linalg.norm(t_nose - locations[f, node, :, subject_track])
                    for node in head_nodes
                ]

                min_s_to_t_head = min(s_to_t_head) if len(s_to_t_head) else float("inf")
                min_t_to_s_head = min(t_to_s_head) if len(t_to_s_head) else float("inf")

                # --- headings: CM -> nose ---
                subj_ang = app_angle(
                    locations[f, cm_index, AXIS_X, subject_track],
                    locations[f, cm_index, AXIS_Y, subject_track],
                    locations[f, ns_index, AXIS_X, subject_track],
                    locations[f, ns_index, AXIS_Y, subject_track]
                )

                strg_ang = app_angle(
                    locations[f, cm_index, AXIS_X, stranger_track],
                    locations[f, cm_index, AXIS_Y, stranger_track],
                    locations[f, ns_index, AXIS_X, stranger_track],
                    locations[f, ns_index, AXIS_Y, stranger_track]
                )

                # subject facing stranger?
                subj_to_strg_ang = app_angle(s_nose[0], s_nose[1], t_nose[0], t_nose[1])
                strg_to_subj_ang = app_angle(t_nose[0], t_nose[1], s_nose[0], s_nose[1])

                subj_face_err = rel_angle(subj_ang, subj_to_strg_ang)
                strg_face_err = rel_angle(strg_ang, strg_to_subj_ang)

                # mutual facing
                mutual_head_engagement = (subj_face_err <= 70) and (strg_face_err <= 70)
                strong_mutual_facing = (subj_face_err <= 45) and (strg_face_err <= 45)

                # face-to-face bodies should usually be roughly opposite, not parallel
                heading_opposition = rel_angle(subj_ang, strg_ang)

                # --- HH logic ---
                # broader head-region engagement:
                # 1) either nose is near the other head region, OR
                # 2) head centroids are close
                hh_proximity = (
                        (min_s_to_t_head <= max_dist_hh) or
                        (min_t_to_s_head <= max_dist_hh) or
                        (hh_centroid_d <= 0.85 * dsr_local)
                )

                # allow HH when mutually engaged, not necessarily perfectly opposed
                hh_flag = 1 if (
                        hh_proximity and
                        mutual_head_engagement and
                        (heading_opposition >= 60)
                ) else 0

                # --- HO logic ---
                # stricter head-on / nose-to-nose / face-to-face
                ho_flag = 1 if (
                        (ns_ns_d <= max_dist_ho) and
                        strong_mutual_facing and
                        (heading_opposition >= 120)
                ) else 0

                # If HO is true, HH should usually also be true geometrically,
                # but keep labels separate for downstream priority handling.
                final_hh.append([f, 'hh' if hh_flag else 'n', hh_flag])
                final_ho.append([f, 'ho' if ho_flag else 'n', ho_flag])

            self.update_status("#     [100%]")
            return final_hh, final_ho

        ##Determine Social Sniffing Behavior##
        def det_sniff():
            self.update_status("Sniff: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            dsr_local = dynamic_sniff_range()
            sniff_array = []
            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)
                dist, node_idx = closest_to_nose(f)
                subj_theta = app_angle(locations[f, cm_index, AXIS_X, subject_track],
                                       locations[f, cm_index, AXIS_Y, subject_track],
                                       locations[f, ns_index, AXIS_X, subject_track],
                                       locations[f, ns_index, AXIS_Y, subject_track])
                if node_idx != -1:
                    target_theta = app_angle(locations[f, ns_index, AXIS_X, subject_track],
                                             locations[f, ns_index, AXIS_Y, subject_track],
                                             locations[f, node_idx, AXIS_X, stranger_track],
                                             locations[f, node_idx, AXIS_Y, stranger_track])
                    angle_diff = rel_angle(subj_theta, target_theta)
                else:
                    angle_diff = 999
                flag = 0
                if (dist is not None) and (not np.isnan(dist)):
                    if (dist <= ( 0.5 * dsr_local)) and (angle_diff <= 90) and (node_idx != cm_index) and (node_idx != tm_index) and (node_idx != te_index): #0.2 * dsr
                        flag = 1
                    elif (node_idx == cm_index) and (dist <= (0.7 * dsr_local)) and (angle_diff <= 90) and (node_idx != tm_index) and (node_idx != te_index): #0.7*dsr
                        flag = 1
                    elif (node_idx == tm_index) or (node_idx == te_index):
                        flag = 0
                    else:
                        flag = 0
                sniff_array.append([f, 's' if flag else 'n', flag])
            self.update_status("#     [100%]")
            return sniff_array

        import numpy as np
        def det_avoid(
                locations,
                prox_array,
                sniff_array,
                subject_track,
                stranger_track,
                fps=24,
                speed_mad_factor=1.5,
                angle_thresh=120,
                gate=0,
                hhManifest=None,
                hoManifest=None,
                followManifest=None
        ):
            """
            Improved avoidance detector.

            Definitions
            -----------
            Active avoidance:
                Subject actively retreats away from a socially relevant stranger.
                Requires:
                  - recent or current social opportunity
                  - subject moving above baseline
                  - subject velocity directed away from stranger
                  - distance increasing by a meaningful amount
                  - subject body/head oriented away from stranger

            Passive avoidance:
                Subject remains socially disengaged / withdrawn in the presence of
                continued social relevance, but without strong active retreat.
                Requires:
                  - recent/current social relevance
                  - not in direct contact now
                  - subject oriented away from stranger
                  - low displacement over a short window
                  - not approaching / not following
                This is closer to withdrawal/freezing/non-approach than generic periphery occupancy.

            Returns
            -------
            avoid_array : list of [frame, label, flag]
                label in {'a', 'p', 'n'}
                flag = 1 for either active or passive avoidance, else 0
            """
            self.update_status("Avoid: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))

            n_frames = locations.shape[0]
            start_frame = gate
            avoid_array = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(1, int(round(0.5 * fps)))  # ~0.5 s window
            immobile_win = max(2, int(round(0.5 * fps)))  # ~0.5 s for passive stillness
            recent_social_win = max(1, int(round(0.75 * fps)))  # slightly tighter than 1 s

            # --- core coordinates ---
            cm_subj = locations[:, cm_index, :, subject_track]
            cm_strg = locations[:, cm_index, :, stranger_track]
            nose_subj = locations[:, ns_index, :, subject_track]
            nose_strg = locations[:, ns_index, :, stranger_track]

            # --- vectors/distances ---
            subj_to_strg_vec = cm_strg - cm_subj
            subj_to_strg_dist = np.linalg.norm(subj_to_strg_vec, axis=1)
            nose_to_strgcm_dist = np.linalg.norm(nose_subj - cm_strg, axis=1)

            # normalize vector from subject -> stranger
            subj_to_strg_unit = subj_to_strg_vec / (np.linalg.norm(subj_to_strg_vec, axis=1, keepdims=True) + 1e-9)

            # --- subject motion ---
            subj_vel = np.zeros((n_frames, 2))
            strg_vel = np.zeros((n_frames, 2))
            subj_speed = np.zeros(n_frames)
            strg_speed = np.zeros(n_frames)

            subj_vel[1:] = np.diff(cm_subj, axis=0)
            strg_vel[1:] = np.diff(cm_strg, axis=0)
            subj_speed = np.linalg.norm(subj_vel, axis=1)
            strg_speed = np.linalg.norm(strg_vel, axis=1)

            # projected retreat velocity:
            # positive = moving away from stranger
            retreat_proj = np.einsum("ij,ij->i", subj_vel, -subj_to_strg_unit)

            # gap change over ~0.5 s
            gap_delta = np.zeros(n_frames)
            gap_delta[lag:] = nose_to_strgcm_dist[lag:] - nose_to_strgcm_dist[:-lag]

            meaningful_retreat = gap_delta > (0.20 * BL)

            # --- orientation away from stranger ---
            head_vec = nose_subj - cm_subj
            head_vec = head_vec / (np.linalg.norm(head_vec, axis=1, keepdims=True) + 1e-9)

            # angle between subject heading and direction to stranger
            # large angle = facing away
            cos_angle = np.einsum("ij,ij->i", head_vec, subj_to_strg_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            facing_away = angle_deg > angle_thresh

            # --- social flags from manifests ---
            prox_flags = np.zeros(n_frames, dtype=int)
            sniff_flags = np.zeros(n_frames, dtype=int)
            hh_flags = np.zeros(n_frames, dtype=int)
            ho_flags = np.zeros(n_frames, dtype=int)
            follow_flags = np.zeros(n_frames, dtype=int)

            for row in prox_array:
                f = int(row[0])
                if 0 <= f < n_frames:
                    prox_flags[f] = int(row[2])

            for row in sniff_array:
                f = int(row[0])
                if 0 <= f < n_frames:
                    sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < n_frames:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < n_frames:
                        ho_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < n_frames:
                        follow_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)

            # --- recent social context ---
            social_now = (prox_flags == 1) | (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)

            recent_social = np.zeros(n_frames, dtype=bool)
            contact_frames = np.where(social_now)[0]
            for f in contact_frames:
                end = min(n_frames, f + recent_social_win)
                recent_social[f:end] = True

            # --- continued social relevance ---
            # stronger than old "periphery + recent social":
            # partner still near enough to matter OR currently/recently socially engaged
            socially_relevant = (
                    (subj_to_strg_dist <= 4.0 * BL) |
                    social_now |
                    recent_social
            )

            # --- robust baseline motion from nonsocial frames ---
            nonsocial = ~(social_now.astype(bool))
            if np.any(nonsocial[start_frame:]):
                base_med = np.nanmedian(subj_speed[start_frame:][nonsocial[start_frame:]])
                base_mad = np.nanmedian(np.abs(subj_speed[start_frame:][nonsocial[start_frame:]] - base_med))
            else:
                base_med = np.nanmedian(subj_speed[start_frame:])
                base_mad = np.nanmedian(np.abs(subj_speed[start_frame:] - base_med))

            if not np.isfinite(base_med):
                base_med = 0.0
            if not np.isfinite(base_mad):
                base_mad = 0.0

            active_speed_thresh = base_med + speed_mad_factor * base_mad
            slow_speed_thresh = base_med + 0.25 * base_mad

            # --- short-window displacement for passive avoidance ---
            short_disp = np.zeros(n_frames)
            short_disp[immobile_win:] = np.linalg.norm(
                cm_subj[immobile_win:] - cm_subj[:-immobile_win],
                axis=1
            )

            low_displacement = short_disp <= (0.20 * BL)
            not_approaching = gap_delta >= (-0.05 * BL)

            # optional periphery as a weak supporting feature only
            arena_center = np.nanmean(cm_subj, axis=0)
            radial_dist = np.linalg.norm(cm_subj - arena_center, axis=1)
            periphery_thresh = np.nanpercentile(radial_dist, 75)
            in_periphery = radial_dist >= periphery_thresh

            # --- ACTIVE avoidance ---
            # must actively move away from stranger, not just let distance increase
            active_mask = (
                    socially_relevant &
                    recent_social &
                    (~direct_contact) &
                    facing_away &
                    (subj_speed > active_speed_thresh) &
                    (retreat_proj > 0.15 * BL) &
                    meaningful_retreat
            )

            # --- PASSIVE avoidance ---
            # low movement + socially relevant + oriented away + not currently engaging
            passive_mask = (
                    (~active_mask) &
                    socially_relevant &
                    recent_social &
                    (~direct_contact) &
                    (follow_flags == 0) &
                    facing_away &
                    (subj_speed <= slow_speed_thresh) &
                    low_displacement &
                    not_approaching
            )

            # tighten passive slightly:
            # either partner is still fairly near, or subject has moved to periphery
            passive_mask = passive_mask & (
                    (subj_to_strg_dist <= 3.0 * BL) | in_periphery
            )

            # --- build output ---
            for f in range(start_frame, n_frames):
                if (f - start_frame) % ticker == 0:
                    self.update_status("#", add_newline=False)

                if active_mask[f]:
                    label = 'a'
                    flag = 1
                elif passive_mask[f]:
                    label = 'p'
                    flag = 1
                else:
                    label = 'n'
                    flag = 0

                avoid_array.append([f, label, flag])

            self.update_status("#     [100%]")
            return avoid_array

        ##Determine Following Behavior##
        def det_follow(sniffManifest=None, hhManifest=None, hoManifest=None, prox_array=None):
            """
            Behaviorally stricter social-follow detector.

            Definition:
              Follow = subject is behind stranger, both are locomoting, subject is
              oriented toward stranger, and subject is either:
                (a) closing the trailing gap (pursuit onset), or
                (b) maintaining a narrow trailing band while co-directed.

            Exclusions:
              - direct contact states (sniff / hh / ho) suppress follow
              - very slow or very erratic motion suppressed
              - large lateral offset suppressed

            Returns
            -------
            following : list of [frame, 'f' or 'n', flag]
            """
            self.update_status("Follow: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            following = []

            # --- core coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]
            subj_ns = locations[:, ns_index, :, subject_track]
            strg_ns = locations[:, ns_index, :, stranger_track]

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local  # effective body-length-ish scale already used in your code

            lag = max(iFPS, 1)

            # --- velocities over ~1 s ---
            subj_speed = np.zeros(frame_count)
            strg_speed = np.zeros(frame_count)
            subj_vel_vec = np.zeros((frame_count, 2))
            strg_vel_vec = np.zeros((frame_count, 2))

            subj_speed[lag:] = np.linalg.norm(subj_cm[lag:] - subj_cm[:-lag], axis=1) / lag
            strg_speed[lag:] = np.linalg.norm(strg_cm[lag:] - strg_cm[:-lag], axis=1) / lag
            subj_vel_vec[lag:] = (subj_cm[lag:] - subj_cm[:-lag]) / lag
            strg_vel_vec[lag:] = (strg_cm[lag:] - strg_cm[:-lag]) / lag

            def unit_vec(v, eps=1e-9):
                n = np.linalg.norm(v, axis=-1, keepdims=True)
                return v / (n + eps)

            def cos_aligned(a, b, eps=1e-6):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na < eps or nb < eps:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))

            # --- motion thresholds from session statistics ---
            active_subj = subj_speed[gate:]
            active_strg = strg_speed[gate:]

            subj_move_min = np.percentile(active_subj, 35) if len(active_subj) else 0.0
            strg_move_min = np.percentile(active_strg, 35) if len(active_strg) else 0.0
            subj_move_hi = np.percentile(active_subj, 95) if len(active_subj) else np.inf
            strg_move_hi = np.percentile(active_strg, 95) if len(active_strg) else np.inf

            # --- headings from CM -> nose ---
            subj_theta = np.array([
                app_angle(locations[f, cm_index, AXIS_X, subject_track],
                          locations[f, cm_index, AXIS_Y, subject_track],
                          locations[f, ns_index, AXIS_X, subject_track],
                          locations[f, ns_index, AXIS_Y, subject_track])
                for f in range(frame_count)
            ])

            strg_theta = np.array([
                app_angle(locations[f, cm_index, AXIS_X, stranger_track],
                          locations[f, cm_index, AXIS_Y, stranger_track],
                          locations[f, ns_index, AXIS_X, stranger_track],
                          locations[f, ns_index, AXIS_Y, stranger_track])
                for f in range(frame_count)
            ])

            # --- angular velocities ---
            subj_ang_vel = np.zeros(frame_count)
            strg_ang_vel = np.zeros(frame_count)
            for f in range(lag, frame_count):
                subj_ang_vel[f] = rel_angle(subj_theta[f], subj_theta[f - lag]) * (fps / lag)
                strg_ang_vel[f] = rel_angle(strg_theta[f], strg_theta[f - lag]) * (fps / lag)

            # --- distance metrics ---
            cm_dist = np.linalg.norm(subj_cm - strg_cm, axis=1)
            subjnose_to_strgcm = np.linalg.norm(subj_ns - strg_cm, axis=1)

            # signed change in gap over ~1 s
            gap_delta = np.zeros(frame_count)
            gap_delta[lag:] = subjnose_to_strgcm[lag:] - subjnose_to_strgcm[:-lag]
            closing_gap = gap_delta < (-0.15 * BL)  # becoming meaningfully closer

            # maintenance band: narrow trailing distance with low variability
            in_follow_band = (cm_dist >= 0.8 * BL) & (cm_dist <= 3.5 * BL)

            stable_gap = np.zeros(frame_count, dtype=int)
            for f in range(thresh, frame_count):
                win = subjnose_to_strgcm[max(0, f - thresh):f]
                if len(win) > 3 and np.std(win) <= 0.25 * BL:
                    stable_gap[f] = 1

            maintaining_trail = in_follow_band & (stable_gap == 1)

            # --- relative position in stranger body frame ---
            # subject should be behind stranger, not far lateral
            behind_target = np.zeros(frame_count, dtype=int)
            for f in range(frame_count):
                dx = subj_cm[f, 0] - strg_cm[f, 0]
                dy = subj_cm[f, 1] - strg_cm[f, 1]
                th = np.radians(strg_theta[f])

                # projection into stranger heading frame
                proj_long = dx * np.cos(th) + dy * np.sin(th)
                proj_lat = -dx * np.sin(th) + dy * np.cos(th)

                # behind = negative longitudinal relative to stranger heading
                if (proj_long <= -0.2 * BL) and (abs(proj_lat) <= 1.0 * BL):
                    behind_target[f] = 1

            # --- orientation requirements ---
            # 1) subject heading aligned with stranger heading
            heading_aligned = np.array(
                [1 if rel_angle(subj_theta[f], strg_theta[f]) <= 55 else 0 for f in range(frame_count)],
                dtype=int
            )

            # 2) subject heading should also point toward stranger CM
            subj_to_strg_theta = np.array([
                app_angle(subj_cm[f, 0], subj_cm[f, 1], strg_cm[f, 0], strg_cm[f, 1])
                for f in range(frame_count)
            ])
            oriented_to_target = np.array(
                [1 if rel_angle(subj_theta[f], subj_to_strg_theta[f]) <= 50 else 0 for f in range(frame_count)],
                dtype=int
            )

            # 3) velocity alignment
            vel_aligned = np.array(
                [1 if cos_aligned(subj_vel_vec[f], strg_vel_vec[f]) >= 0.65 else 0 for f in range(frame_count)],
                dtype=int
            )

            # --- both animals must be locomoting, but not wildly ---
            subj_moving = (subj_speed >= subj_move_min) & (subj_speed <= subj_move_hi)
            strg_moving = (strg_speed >= strg_move_min) & (strg_speed <= strg_move_hi)

            smooth_motion = (subj_ang_vel <= 120) & (strg_ang_vel <= 120)

            # --- social/contact exclusions ---
            # If not provided, default to no exclusion.
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)

            # --- onset vs maintenance logic ---
            pursuit_onset = closing_gap & in_follow_band
            pursuit_maintenance = maintaining_trail

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = int(
                    subj_moving[f] and
                    strg_moving[f] and
                    smooth_motion[f] and
                    behind_target[f] and
                    heading_aligned[f] and
                    oriented_to_target[f] and
                    vel_aligned[f] and
                    (pursuit_onset[f] or pursuit_maintenance[f]) and
                    (not direct_contact[f])
                )

                following.append([f, 'f' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return following

        def det_chase(
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                followManifest=None,
                prox_array=None
        ):
            """
            Detect chasing behavior.

            Definition
            ----------
            Chase = subject is behind and pursuing the stranger at higher intensity
            than ordinary follow. Requires:
              - both animals locomoting
              - subject behind stranger
              - subject oriented toward stranger
              - subject velocity aligned with stranger velocity
              - subject moving faster than stranger (or at least not slower)
              - gap closing OR tight pursuit maintenance
              - excludes direct-contact states (sniff / hh / ho)

            Returns
            -------
            chasing : list of [frame, 'c' or 'n', flag]
            """
            self.update_status("Chase: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            chasing = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(iFPS, 1)

            # --- coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]
            subj_ns = locations[:, ns_index, :, subject_track]
            strg_ns = locations[:, ns_index, :, stranger_track]

            # --- velocity ---
            subj_vel = np.zeros((frame_count, 2))
            strg_vel = np.zeros((frame_count, 2))
            subj_speed = np.zeros(frame_count)
            strg_speed = np.zeros(frame_count)

            subj_vel[lag:] = (subj_cm[lag:] - subj_cm[:-lag]) / lag
            strg_vel[lag:] = (strg_cm[lag:] - strg_cm[:-lag]) / lag

            subj_speed = np.linalg.norm(subj_vel, axis=1)
            strg_speed = np.linalg.norm(strg_vel, axis=1)

            def cos_aligned(a, b, eps=1e-9):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na < eps or nb < eps:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))

            # --- headings ---
            subj_theta = np.array([
                app_angle(
                    locations[f, cm_index, AXIS_X, subject_track],
                    locations[f, cm_index, AXIS_Y, subject_track],
                    locations[f, ns_index, AXIS_X, subject_track],
                    locations[f, ns_index, AXIS_Y, subject_track]
                )
                for f in range(frame_count)
            ])

            strg_theta = np.array([
                app_angle(
                    locations[f, cm_index, AXIS_X, stranger_track],
                    locations[f, cm_index, AXIS_Y, stranger_track],
                    locations[f, ns_index, AXIS_X, stranger_track],
                    locations[f, ns_index, AXIS_Y, stranger_track]
                )
                for f in range(frame_count)
            ])

            subj_ang_vel = np.zeros(frame_count)
            strg_ang_vel = np.zeros(frame_count)
            for f in range(lag, frame_count):
                subj_ang_vel[f] = rel_angle(subj_theta[f], subj_theta[f - lag]) * (fps / lag)
                strg_ang_vel[f] = rel_angle(strg_theta[f], strg_theta[f - lag]) * (fps / lag)

            # --- distances ---
            cm_dist = np.linalg.norm(subj_cm - strg_cm, axis=1)
            nose_to_strgcm = np.linalg.norm(subj_ns - strg_cm, axis=1)

            gap_delta = np.zeros(frame_count)
            gap_delta[lag:] = nose_to_strgcm[lag:] - nose_to_strgcm[:-lag]

            # stronger than follow: meaningful closing
            closing_gap = gap_delta < (-0.20 * BL)

            # tight pursuit band
            in_chase_band = (cm_dist >= 0.7 * BL) & (cm_dist <= 3.0 * BL)

            stable_gap = np.zeros(frame_count, dtype=int)
            for f in range(thresh, frame_count):
                win = nose_to_strgcm[max(0, f - thresh):f]
                if len(win) > 3 and np.std(win) <= 0.30 * BL:
                    stable_gap[f] = 1
            maintaining_pursuit = in_chase_band & (stable_gap == 1)

            # --- relative position in stranger frame: subject should be behind ---
            behind_target = np.zeros(frame_count, dtype=int)
            for f in range(frame_count):
                dx = subj_cm[f, 0] - strg_cm[f, 0]
                dy = subj_cm[f, 1] - strg_cm[f, 1]
                th = np.radians(strg_theta[f])

                proj_long = dx * np.cos(th) + dy * np.sin(th)
                proj_lat = -dx * np.sin(th) + dy * np.cos(th)

                if (proj_long <= -0.15 * BL) and (abs(proj_lat) <= 1.2 * BL):
                    behind_target[f] = 1

            # --- subject oriented toward stranger ---
            subj_to_strg_theta = np.array([
                app_angle(subj_cm[f, 0], subj_cm[f, 1], strg_cm[f, 0], strg_cm[f, 1])
                for f in range(frame_count)
            ])

            oriented_to_target = np.array([
                1 if rel_angle(subj_theta[f], subj_to_strg_theta[f]) <= 45 else 0
                for f in range(frame_count)
            ], dtype=int)

            heading_aligned = np.array([
                1 if rel_angle(subj_theta[f], strg_theta[f]) <= 50 else 0
                for f in range(frame_count)
            ], dtype=int)

            vel_aligned = np.array([
                1 if cos_aligned(subj_vel[f], strg_vel[f]) >= 0.70 else 0
                for f in range(frame_count)
            ], dtype=int)

            # --- speed thresholds ---
            active_subj = subj_speed[gate:]
            active_strg = strg_speed[gate:]

            subj_move_min = np.percentile(active_subj, 45) if len(active_subj) else 0.0
            strg_move_min = np.percentile(active_strg, 35) if len(active_strg) else 0.0
            subj_fast_thresh = np.percentile(active_subj, 70) if len(active_subj) else 0.0

            subj_moving = subj_speed >= subj_move_min
            strg_moving = strg_speed >= strg_move_min

            # subject should usually be at least as fast, often faster
            speed_advantage = subj_speed >= (strg_speed * 0.95)
            strong_subject_speed = subj_speed >= subj_fast_thresh

            smooth_motion = (subj_ang_vel <= 140) & (strg_ang_vel <= 140)

            # --- direct-contact exclusions ---
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)
            prox_flags = np.zeros(frame_count, dtype=int)

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            if prox_array is not None:
                for row in prox_array:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        prox_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)

            # --- chase onset and maintenance ---
            chase_onset = (
                    closing_gap &
                    in_chase_band &
                    strong_subject_speed
            )

            chase_maintenance = (
                    maintaining_pursuit &
                    speed_advantage &
                    follow_flags.astype(bool)
            )

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = int(
                    subj_moving[f] and
                    strg_moving[f] and
                    smooth_motion[f] and
                    behind_target[f] and
                    oriented_to_target[f] and
                    heading_aligned[f] and
                    vel_aligned[f] and
                    (chase_onset[f] or chase_maintenance[f]) and
                    (not direct_contact[f])
                )

                chasing.append([f, 'c' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return chasing

        def det_flee(
                prox_array,
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                chaseManifest=None,
                followManifest=None
        ):
            """
            Detect fleeing behavior.

            Definition
            ----------
            Flee = high-intensity escape-like retreat from a socially relevant partner.

            Compared with active avoidance, fleeing is stricter:
              - stronger speed requirement
              - stronger retreat directionality
              - stronger increase in partner distance
              - usually occurs during/after direct contact, chase, or strong social pressure

            Returns
            -------
            fleeing : list of [frame, 'fl' or 'n', flag]
            """
            self.update_status("Flee: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            fleeing = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(1, int(round(0.4 * fps)))  # ~0.4 s response window
            recent_social_win = max(1, int(round(0.75 * fps)))

            # --- coordinates ---
            cm_subj = locations[:, cm_index, :, subject_track]
            cm_strg = locations[:, cm_index, :, stranger_track]
            nose_subj = locations[:, ns_index, :, subject_track]

            # vector from subject -> stranger
            subj_to_strg_vec = cm_strg - cm_subj
            subj_to_strg_dist = np.linalg.norm(subj_to_strg_vec, axis=1)
            subj_to_strg_unit = subj_to_strg_vec / (np.linalg.norm(subj_to_strg_vec, axis=1, keepdims=True) + 1e-9)

            # --- subject motion ---
            subj_vel = np.zeros((frame_count, 2))
            strg_vel = np.zeros((frame_count, 2))
            subj_speed = np.zeros(frame_count)
            strg_speed = np.zeros(frame_count)

            subj_vel[1:] = np.diff(cm_subj, axis=0)
            strg_vel[1:] = np.diff(cm_strg, axis=0)
            subj_speed = np.linalg.norm(subj_vel, axis=1)
            strg_speed = np.linalg.norm(strg_vel, axis=1)

            # projected retreat velocity: positive = subject moving away from stranger
            retreat_proj = np.einsum("ij,ij->i", subj_vel, -subj_to_strg_unit)

            # nose-to-stranger-CM distance change over short window
            nose_to_strgcm = np.linalg.norm(nose_subj - cm_strg, axis=1)
            gap_delta = np.zeros(frame_count)
            gap_delta[lag:] = nose_to_strgcm[lag:] - nose_to_strgcm[:-lag]

            # stronger retreat magnitude than active avoidance
            meaningful_escape = gap_delta > (0.30 * BL)

            # --- heading / facing away ---
            head_vec = nose_subj - cm_subj
            head_vec = head_vec / (np.linalg.norm(head_vec, axis=1, keepdims=True) + 1e-9)

            cos_angle = np.einsum("ij,ij->i", head_vec, subj_to_strg_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            facing_away = angle_deg > 130  # stricter than generic active avoidance

            # --- social flags ---
            prox_flags = np.zeros(frame_count, dtype=int)
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            chase_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)

            for row in prox_array:
                f = int(row[0])
                if 0 <= f < frame_count:
                    prox_flags[f] = int(row[2])

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if chaseManifest is not None:
                for row in chaseManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        chase_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)
            social_pressure = direct_contact | (chase_flags == 1) | (follow_flags == 1) | (prox_flags == 1)

            recent_social = np.zeros(frame_count, dtype=bool)
            social_frames = np.where(social_pressure)[0]
            for f in social_frames:
                end = min(frame_count, f + recent_social_win)
                recent_social[f:end] = True

            # --- robust speed thresholds ---
            nonsocial = ~(social_pressure.astype(bool))
            if np.any(nonsocial[gate:]):
                base_med = np.nanmedian(subj_speed[gate:][nonsocial[gate:]])
                base_mad = np.nanmedian(np.abs(subj_speed[gate:][nonsocial[gate:]] - base_med))
            else:
                base_med = np.nanmedian(subj_speed[gate:])
                base_mad = np.nanmedian(np.abs(subj_speed[gate:] - base_med))

            if not np.isfinite(base_med):
                base_med = 0.0
            if not np.isfinite(base_mad):
                base_mad = 0.0

            fast_thresh = max(
                base_med + 2.0 * base_mad,
                np.percentile(subj_speed[gate:], 75) if len(subj_speed[gate:]) else 0.0
            )

            # stranger pressure: stranger is behaviorally relevant and not far away
            partner_relevant = subj_to_strg_dist <= (3.5 * BL)

            # fleeing = stronger, escape-like active retreat
            flee_mask = (
                    recent_social &
                    partner_relevant &
                    facing_away &
                    (subj_speed >= fast_thresh) &
                    (retreat_proj > 0.20 * BL) &
                    meaningful_escape
            )

            # optional tightening:
            # prefer fleeing when triggered by direct contact or pursuit
            flee_mask = flee_mask & (
                    direct_contact |
                    (chase_flags == 1) |
                    recent_social
            )

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = 1 if flee_mask[f] else 0
                fleeing.append([f, 'fl' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return fleeing

        def det_approach(
                prox_array=None,
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                followManifest=None,
                chaseManifest=None,
                avoidManifest=None,
                fleeManifest=None
        ):
            """
            Detect directed, non-contact social approach.

            Definition
            ----------
            Approach = subject moves toward the stranger in a directed way,
            reducing social distance before direct contact.

            Requires:
              - subject oriented toward stranger
              - subject moving
              - movement projected toward stranger
              - meaningful reduction in social gap
              - not already in direct contact
              - not follow / chase / avoidance / fleeing

            Returns
            -------
            approach_array : list of [frame, 'ap' or 'n', flag]
            """
            self.update_status("Approach: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            approach_array = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(1, int(round(0.5 * fps)))  # ~0.5 s window

            # --- coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]
            subj_ns = locations[:, ns_index, :, subject_track]
            strg_ns = locations[:, ns_index, :, stranger_track]

            # --- vectors / distances ---
            subj_to_strg_vec = strg_cm - subj_cm
            subj_to_strg_dist = np.linalg.norm(subj_to_strg_vec, axis=1)
            subj_to_strg_unit = subj_to_strg_vec / (np.linalg.norm(subj_to_strg_vec, axis=1, keepdims=True) + 1e-9)

            nose_to_strgcm = np.linalg.norm(subj_ns - strg_cm, axis=1)

            # --- motion ---
            subj_vel = np.zeros((frame_count, 2))
            subj_speed = np.zeros(frame_count)
            subj_vel[1:] = np.diff(subj_cm, axis=0)
            subj_speed = np.linalg.norm(subj_vel, axis=1)

            # projected approach velocity: positive = moving toward stranger
            approach_proj = np.einsum("ij,ij->i", subj_vel, subj_to_strg_unit)

            # meaningful closing of social gap over short window
            gap_delta = np.zeros(frame_count)
            gap_delta[lag:] = nose_to_strgcm[lag:] - nose_to_strgcm[:-lag]
            closing_gap = gap_delta < (-0.18 * BL)

            # --- subject heading toward stranger ---
            head_vec = subj_ns - subj_cm
            head_vec = head_vec / (np.linalg.norm(head_vec, axis=1, keepdims=True) + 1e-9)

            cos_angle = np.einsum("ij,ij->i", head_vec, subj_to_strg_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            facing_toward = angle_deg <= 55

            # --- distance band ---
            # want social relevance, but not already in close contact
            socially_relevant = subj_to_strg_dist <= (5.0 * BL)
            not_too_far = subj_to_strg_dist >= (1.0 * BL)

            # --- speed threshold from session baseline ---
            active_subj = subj_speed[gate:]
            subj_move_min = np.percentile(active_subj, 35) if len(active_subj) else 0.0
            subj_move_hi = np.percentile(active_subj, 95) if len(active_subj) else np.inf
            subj_moving = (subj_speed >= subj_move_min) & (subj_speed <= subj_move_hi)

            # --- optional manifests / exclusions ---
            prox_flags = np.zeros(frame_count, dtype=int)
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)
            chase_flags = np.zeros(frame_count, dtype=int)
            avoid_flags = np.zeros(frame_count, dtype=int)
            flee_flags = np.zeros(frame_count, dtype=int)

            if prox_array is not None:
                for row in prox_array:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        prox_flags[f] = int(row[2])

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            if chaseManifest is not None:
                for row in chaseManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        chase_flags[f] = int(row[2])

            if avoidManifest is not None:
                for row in avoidManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        avoid_flags[f] = 1 if row[1] in ('a', 'p') else 0

            if fleeManifest is not None:
                for row in fleeManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        flee_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)
            excluded_state = (
                    (follow_flags == 1) |
                    (chase_flags == 1) |
                    (avoid_flags == 1) |
                    (flee_flags == 1)
            )

            # --- optional onset bias ---
            # approach often ends in prox/contact, so allow mild support if prox is near onset
            future_social = np.zeros(frame_count, dtype=bool)
            lookahead = max(1, int(round(0.5 * fps)))
            social_target = (prox_flags == 1) | direct_contact
            social_frames = np.where(social_target)[0]
            for f in social_frames:
                start = max(0, f - lookahead)
                future_social[start:f + 1] = True

            # --- final rule ---
            approach_mask = (
                    subj_moving &
                    facing_toward &
                    socially_relevant &
                    not_too_far &
                    (approach_proj > 0.12 * BL) &
                    closing_gap &
                    (~direct_contact) &
                    (~excluded_state)
            )

            # tighten slightly: prefer approaches that lead into social opportunity
            approach_mask = approach_mask & (future_social | (subj_to_strg_dist <= 3.5 * BL))

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = 1 if approach_mask[f] else 0
                approach_array.append([f, 'ap' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return approach_array

        def det_stationary_proximity(
                prox_array,
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                followManifest=None,
                chaseManifest=None,
                avoidManifest=None,
                fleeManifest=None
        ):
            """
            Detect stationary proximity:
            both mice are near each other, neither is moving much, and they are
            not in direct contact or other high-motion social states.

            Definition
            ----------
            Stationary proximity = quiet co-presence within social range.

            Returns
            -------
            sp_array : list of [frame, 'sp' or 'n', flag]
            """
            self.update_status("Stationary Prox: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            sp_array = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(1, int(round(0.5 * fps)))  # ~0.5 s stillness window

            # --- coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]

            cm_dist = np.linalg.norm(subj_cm - strg_cm, axis=1)

            # --- short-window displacement / speed ---
            subj_disp = np.zeros(frame_count)
            strg_disp = np.zeros(frame_count)

            subj_disp[lag:] = np.linalg.norm(subj_cm[lag:] - subj_cm[:-lag], axis=1)
            strg_disp[lag:] = np.linalg.norm(strg_cm[lag:] - strg_cm[:-lag], axis=1)

            subj_speed = np.zeros(frame_count)
            strg_speed = np.zeros(frame_count)
            subj_speed[1:] = np.linalg.norm(np.diff(subj_cm, axis=0), axis=1)
            strg_speed[1:] = np.linalg.norm(np.diff(strg_cm, axis=0), axis=1)

            # --- import flags from manifests ---
            prox_flags = np.zeros(frame_count, dtype=int)
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)
            chase_flags = np.zeros(frame_count, dtype=int)
            avoid_flags = np.zeros(frame_count, dtype=int)
            flee_flags = np.zeros(frame_count, dtype=int)

            for row in prox_array:
                f = int(row[0])
                if 0 <= f < frame_count:
                    prox_flags[f] = int(row[2])

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            if chaseManifest is not None:
                for row in chaseManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        chase_flags[f] = int(row[2])

            if avoidManifest is not None:
                for row in avoidManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        avoid_flags[f] = 1 if row[1] in ('a', 'p') else 0

            if fleeManifest is not None:
                for row in fleeManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        flee_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)

            excluded_state = (
                    (follow_flags == 1) |
                    (chase_flags == 1) |
                    (avoid_flags == 1) |
                    (flee_flags == 1)
            )

            # --- stillness thresholds ---
            # robust baseline from session-wide movement
            subj_med = np.nanmedian(subj_speed[gate:]) if len(subj_speed[gate:]) else 0.0
            strg_med = np.nanmedian(strg_speed[gate:]) if len(strg_speed[gate:]) else 0.0

            subj_mad = np.nanmedian(np.abs(subj_speed[gate:] - subj_med)) if len(subj_speed[gate:]) else 0.0
            strg_mad = np.nanmedian(np.abs(strg_speed[gate:] - strg_med)) if len(strg_speed[gate:]) else 0.0

            if not np.isfinite(subj_med):
                subj_med = 0.0
            if not np.isfinite(strg_med):
                strg_med = 0.0
            if not np.isfinite(subj_mad):
                subj_mad = 0.0
            if not np.isfinite(strg_mad):
                strg_mad = 0.0

            subj_still_speed = subj_speed <= (subj_med + 0.15 * subj_mad)
            strg_still_speed = strg_speed <= (strg_med + 0.15 * strg_mad)

            subj_low_disp = subj_disp <= (0.20 * BL)
            strg_low_disp = strg_disp <= (0.20 * BL)

            both_still = subj_still_speed & strg_still_speed & subj_low_disp & strg_low_disp

            # --- proximity requirement ---
            # prefer actual proximity flag, but also support direct distance guard
            socially_near = (prox_flags == 1) | (cm_dist <= 2.5 * dsr_local)

            # --- final rule ---
            sp_mask = (
                    socially_near &
                    both_still &
                    (~direct_contact) &
                    (~excluded_state)
            )

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = 1 if sp_mask[f] else 0
                sp_array.append([f, 'sp' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return sp_array

        def det_social_orientation(
                prox_array=None,
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                approachManifest=None,
                followManifest=None,
                chaseManifest=None,
                avoidManifest=None,
                fleeManifest=None
        ):
            """
            Detect social orientation:
            subject is oriented toward the stranger within a socially relevant range,
            regardless of whether it is moving.

            Definition
            ----------
            Social orientation = directional attention/engagement toward partner.

            Returns
            -------
            so_array : list of [frame, 'so' or 'n', flag]
            """
            self.update_status("Social Orient: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            so_array = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local

            # --- coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]
            subj_ns = locations[:, ns_index, :, subject_track]
            strg_ns = locations[:, ns_index, :, stranger_track]

            cm_dist = np.linalg.norm(subj_cm - strg_cm, axis=1)
            nose_to_strgcm = np.linalg.norm(subj_ns - strg_cm, axis=1)

            # --- subject heading vector ---
            head_vec = subj_ns - subj_cm
            head_vec = head_vec / (np.linalg.norm(head_vec, axis=1, keepdims=True) + 1e-9)

            subj_to_strg_vec = strg_cm - subj_cm
            subj_to_strg_unit = subj_to_strg_vec / (np.linalg.norm(subj_to_strg_vec, axis=1, keepdims=True) + 1e-9)

            # angle between subject heading and stranger direction
            cos_angle = np.einsum("ij,ij->i", head_vec, subj_to_strg_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            # stronger and weaker orientation bands
            strongly_oriented = angle_deg <= 45
            weakly_oriented = angle_deg <= 65

            # --- manifest flags ---
            prox_flags = np.zeros(frame_count, dtype=int)
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            approach_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)
            chase_flags = np.zeros(frame_count, dtype=int)
            avoid_flags = np.zeros(frame_count, dtype=int)
            flee_flags = np.zeros(frame_count, dtype=int)

            if prox_array is not None:
                for row in prox_array:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        prox_flags[f] = int(row[2])

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if approachManifest is not None:
                for row in approachManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        approach_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            if chaseManifest is not None:
                for row in chaseManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        chase_flags[f] = int(row[2])

            if avoidManifest is not None:
                for row in avoidManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        avoid_flags[f] = 1 if row[1] in ('a', 'p') else 0

            if fleeManifest is not None:
                for row in fleeManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        flee_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)
            active_social = (
                    (approach_flags == 1) |
                    (follow_flags == 1) |
                    (chase_flags == 1)
            )
            withdrawal_state = (avoid_flags == 1) | (flee_flags == 1)

            # --- distance gating ---
            # want partner close enough to be socially relevant
            socially_relevant = (
                    (cm_dist <= 5.0 * BL) |
                    (prox_flags == 1)
            )

            # --- final rule ---
            # Strong orientation if otherwise inactive.
            # Allow weaker orientation if already in an active social state.
            so_mask = (
                    socially_relevant &
                    (~withdrawal_state) &
                    (
                            strongly_oriented |
                            (weakly_oriented & active_social) |
                            (weakly_oriented & direct_contact)
                    )
            )

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = 1 if so_mask[f] else 0
                so_array.append([f, 'so' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return so_array

        def det_disengage(
                prox_array=None,
                sniffManifest=None,
                hhManifest=None,
                hoManifest=None,
                approachManifest=None,
                followManifest=None,
                chaseManifest=None,
                avoidManifest=None,
                fleeManifest=None,
                socialOrientationManifest=None
        ):
            """
            Detect social disengagement.

            Definition
            ----------
            Disengage = subject was recently socially engaged, then turns away and/or
            ceases directed engagement without entering intense escape.

            This is meant to capture interaction termination / social withdrawal,
            distinct from:
              - active avoidance (strong retreat)
              - fleeing (escape)
              - passive avoidance (withdrawn non-approach state)

            Returns
            -------
            disengage_array : list of [frame, 'dg' or 'n', flag]
            """
            self.update_status("Disengage: ", add_newline=False)
            ticker = max(1, math.floor((frame_count - gate) / 9))
            disengage_array = []

            dsr_local = dynamic_sniff_range()
            BL = 2.0 * dsr_local
            lag = max(1, int(round(0.5 * fps)))
            recent_win = max(1, int(round(0.75 * fps)))

            # --- coordinates ---
            subj_cm = locations[:, cm_index, :, subject_track]
            strg_cm = locations[:, cm_index, :, stranger_track]
            subj_ns = locations[:, ns_index, :, subject_track]

            cm_dist = np.linalg.norm(subj_cm - strg_cm, axis=1)
            nose_to_strgcm = np.linalg.norm(subj_ns - strg_cm, axis=1)

            # --- subject motion ---
            subj_vel = np.zeros((frame_count, 2))
            subj_speed = np.zeros(frame_count)
            subj_vel[1:] = np.diff(subj_cm, axis=0)
            subj_speed = np.linalg.norm(subj_vel, axis=1)

            # direction to stranger
            subj_to_strg_vec = strg_cm - subj_cm
            subj_to_strg_unit = subj_to_strg_vec / (np.linalg.norm(subj_to_strg_vec, axis=1, keepdims=True) + 1e-9)

            # projected retreat component; positive = moving away from partner
            retreat_proj = np.einsum("ij,ij->i", subj_vel, -subj_to_strg_unit)

            # mild increase in social gap over short window
            gap_delta = np.zeros(frame_count)
            gap_delta[lag:] = nose_to_strgcm[lag:] - nose_to_strgcm[:-lag]

            mild_separation = gap_delta > (0.05 * BL)

            # --- orientation away ---
            head_vec = subj_ns - subj_cm
            head_vec = head_vec / (np.linalg.norm(head_vec, axis=1, keepdims=True) + 1e-9)

            cos_angle = np.einsum("ij,ij->i", head_vec, subj_to_strg_unit)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            weakly_away = angle_deg >= 95
            strongly_toward = angle_deg <= 55

            # --- flags from manifests ---
            prox_flags = np.zeros(frame_count, dtype=int)
            sniff_flags = np.zeros(frame_count, dtype=int)
            hh_flags = np.zeros(frame_count, dtype=int)
            ho_flags = np.zeros(frame_count, dtype=int)
            approach_flags = np.zeros(frame_count, dtype=int)
            follow_flags = np.zeros(frame_count, dtype=int)
            chase_flags = np.zeros(frame_count, dtype=int)
            avoid_flags = np.zeros(frame_count, dtype=int)
            flee_flags = np.zeros(frame_count, dtype=int)
            social_orient_flags = np.zeros(frame_count, dtype=int)

            if prox_array is not None:
                for row in prox_array:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        prox_flags[f] = int(row[2])

            if sniffManifest is not None:
                for row in sniffManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        sniff_flags[f] = int(row[2])

            if hhManifest is not None:
                for row in hhManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        hh_flags[f] = int(row[2])

            if hoManifest is not None:
                for row in hoManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        ho_flags[f] = int(row[2])

            if approachManifest is not None:
                for row in approachManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        approach_flags[f] = int(row[2])

            if followManifest is not None:
                for row in followManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        follow_flags[f] = int(row[2])

            if chaseManifest is not None:
                for row in chaseManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        chase_flags[f] = int(row[2])

            if avoidManifest is not None:
                for row in avoidManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        avoid_flags[f] = 1 if row[1] in ('a', 'p') else 0

            if fleeManifest is not None:
                for row in fleeManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        flee_flags[f] = int(row[2])

            if socialOrientationManifest is not None:
                for row in socialOrientationManifest:
                    f = int(row[0])
                    if 0 <= f < frame_count:
                        social_orient_flags[f] = int(row[2])

            direct_contact = (sniff_flags == 1) | (hh_flags == 1) | (ho_flags == 1)
            active_engagement = (
                    (approach_flags == 1) |
                    (follow_flags == 1) |
                    (chase_flags == 1) |
                    (social_orient_flags == 1) |
                    direct_contact |
                    (prox_flags == 1)
            )

            # recent social engagement history
            recent_social = np.zeros(frame_count, dtype=bool)
            engaged_frames = np.where(active_engagement)[0]
            for f in engaged_frames:
                end = min(frame_count, f + recent_win)
                recent_social[f:end] = True

            # baseline speed
            active_subj = subj_speed[gate:]
            base_med = np.nanmedian(active_subj) if len(active_subj) else 0.0
            base_mad = np.nanmedian(np.abs(active_subj - base_med)) if len(active_subj) else 0.0

            if not np.isfinite(base_med):
                base_med = 0.0
            if not np.isfinite(base_mad):
                base_mad = 0.0

            slow_thresh = base_med + 0.35 * base_mad
            moderate_thresh = base_med + 1.2 * base_mad

            # exclude strong retreat/escape states
            not_escape = (flee_flags == 0)
            not_strong_avoid = ~(
                    (avoid_flags == 1) &
                    (subj_speed > moderate_thresh) &
                    (retreat_proj > 0.15 * BL)
            )

            # not currently socially engaging
            not_currently_engaging = (
                    (direct_contact == 0) &
                    (approach_flags == 0) &
                    (follow_flags == 0) &
                    (chase_flags == 0)
            )

            # still socially relevant enough to count as disengagement rather than unrelated exploration
            socially_relevant = (cm_dist <= 4.0 * BL) | recent_social | (prox_flags == 1)

            # main logic:
            # recent social context, no direct engagement now, turning away or losing orientation,
            # mild separation or low-motion withdrawal, but not intense escape
            disengage_mask = (
                    recent_social &
                    socially_relevant &
                    not_currently_engaging &
                    not_escape &
                    not_strong_avoid &
                    (
                            weakly_away |
                            ((social_orient_flags == 0) & (~strongly_toward))
                    ) &
                    (
                            mild_separation |
                            (subj_speed <= slow_thresh) |
                            (retreat_proj > 0.03 * BL)
                    )
            )

            for f in range(gate, frame_count):
                if (f - gate) % ticker == 0:
                    self.update_status("#", add_newline=False)

                flag = 1 if disengage_mask[f] else 0
                disengage_array.append([f, 'dg' if flag else 'n', flag])

            self.update_status("#     [100%]")
            return disengage_array

        hhManifest, hoManifest = det_hh_ho()

        vis_array = det_vis()

        aud_array = det_aud()

        sniffManifest = det_sniff()

        followManifest = det_follow(
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            prox_array=prox_array
        )

        chaseManifest = det_chase(
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            followManifest=followManifest,
            prox_array=prox_array
        )

        fleeManifest = det_flee(
            prox_array=prox_array,
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            chaseManifest=chaseManifest,
            followManifest=followManifest
        )

        avoidManifest = det_avoid(
            locations,
            prox_array,
            sniffManifest,
            subject_track,
            stranger_track,
            fps=fps,
            gate=gate,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            followManifest=followManifest
        )

        approachManifest = det_approach(
            prox_array=prox_array,
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            followManifest=followManifest,
            chaseManifest=chaseManifest,
            avoidManifest=avoidManifest,
            fleeManifest=fleeManifest
        )

        stationaryProxManifest = det_stationary_proximity(
            prox_array=prox_array,
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            followManifest=followManifest,
            chaseManifest=chaseManifest,
            avoidManifest=avoidManifest,
            fleeManifest=fleeManifest
        )

        socialOrientationManifest = det_social_orientation(
            prox_array=prox_array,
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            approachManifest=approachManifest,
            followManifest=followManifest,
            chaseManifest=chaseManifest,
            avoidManifest=avoidManifest,
            fleeManifest=fleeManifest
        )

        disengageManifest = det_disengage(
            prox_array=prox_array,
            sniffManifest=sniffManifest,
            hhManifest=hhManifest,
            hoManifest=hoManifest,
            approachManifest=approachManifest,
            followManifest=followManifest,
            chaseManifest=chaseManifest,
            avoidManifest=avoidManifest,
            fleeManifest=fleeManifest,
            socialOrientationManifest=socialOrientationManifest
        )

        def split_aud_manifest(aud_array):
            """
            Convert auditory labels into separate binary channels:
              binauralAud, monauralAud, rearAud
            """
            binaural = {}
            monaural = {}
            rear = {}

            for row in aud_array:
                f = int(row[0])
                label = row[1]

                binaural[f] = 1 if label == 'B' else 0
                monaural[f] = 1 if label in ('ML', 'MR') else 0
                rear[f] = 1 if label == 'R' else 0

            return binaural, monaural, rear

        def split_vis_manifest(vis_array):
            """
            Convert visual labels into separate binary channels:
              binocularVis, monocularVis, noVis
            """
            binocular = {}
            monocular = {}
            no_vis = {}

            for row in vis_array:
                f = int(row[0])
                label = row[1]

                binocular[f] = 1 if label == 'B' else 0
                monocular[f] = 1 if label in ('ML', 'MR') else 0
                no_vis[f] = 1 if label == 'N' else 0

            return binocular, monocular, no_vis

        def split_avoid_manifest(avoidManifest):
            """
            Convert labeled avoid manifest into separate binary maps:
              activeAvoid, passiveAvoid
            """
            active = {}
            passive = {}

            for row in avoidManifest:
                f = int(row[0])
                label = row[1]

                active[f] = 1 if label == 'a' else 0
                passive[f] = 1 if label == 'p' else 0

            return active, passive

        def set_behavior_manifest(
                approachManifest,
                followManifest,
                chaseManifest,
                sniffManifest,
                hhManifest,
                hoManifest,
                avoidManifest,
                fleeManifest,
                disengageManifest,
                stationaryProxManifest,
                socialOrientationManifest,
                aud_array,
                vis_array,
                prox_array
        ):
            """
            Build a full framewise behavior manifest.

            Column order:
            [frame,
             approach,
             follow,
             chase,
             sniff,
             hh,
             ho,
             activeAvoid,
             passiveAvoid,
             flee,
             disengage,
             stationaryProx,
             socialOrient,
             proximity,
             binauralAud,
             monauralAud,
             rearAud,
             binocularVis,
             monocularVis,
             noVis]
            """
            behaviorManifest = []

            activeAvoidMap, passiveAvoidMap = split_avoid_manifest(avoidManifest)
            binauralAudMap, monauralAudMap, rearAudMap = split_aud_manifest(aud_array)
            binocularVisMap, monocularVisMap, noVisMap = split_vis_manifest(vis_array)

            def manifest_to_flag_dict(manifest):
                d = {}
                for row in manifest:
                    f = int(row[0])
                    d[f] = int(row[2])
                return d

            approachMap = manifest_to_flag_dict(approachManifest)
            followMap = manifest_to_flag_dict(followManifest)
            chaseMap = manifest_to_flag_dict(chaseManifest)
            sniffMap = manifest_to_flag_dict(sniffManifest)
            hhMap = manifest_to_flag_dict(hhManifest)
            hoMap = manifest_to_flag_dict(hoManifest)
            fleeMap = manifest_to_flag_dict(fleeManifest)
            disengageMap = manifest_to_flag_dict(disengageManifest)
            stationaryProxMap = manifest_to_flag_dict(stationaryProxManifest)
            socialOrientMap = manifest_to_flag_dict(socialOrientationManifest)
            proxMap = manifest_to_flag_dict(prox_array)

            ticker = max(1, math.floor(frame_count / 9))

            for f in range(0, gate):
                behaviorManifest.append([
                    f,  # frame
                    0,  # approach
                    0,  # follow
                    0,  # chase
                    0,  # sniff
                    0,  # hh
                    0,  # ho
                    0,  # activeAvoid
                    0,  # passiveAvoid
                    0,  # flee
                    0,  # disengage
                    0,  # stationaryProx
                    0,  # socialOrient
                    0,  # proximity
                    0,  # binauralAud
                    0,  # monauralAud
                    0,  # rearAud
                    0,  # binocularVis
                    0,  # monocularVis
                    0  # noVis
                ])

            for f in range(gate, frame_count):
                if f % ticker == 0:
                    print("#", end="")

                behaviorManifest.append([
                    f,
                    approachMap.get(f, 0),
                    followMap.get(f, 0),
                    chaseMap.get(f, 0),
                    sniffMap.get(f, 0),
                    hhMap.get(f, 0),
                    hoMap.get(f, 0),
                    activeAvoidMap.get(f, 0),
                    passiveAvoidMap.get(f, 0),
                    fleeMap.get(f, 0),
                    disengageMap.get(f, 0),
                    stationaryProxMap.get(f, 0),
                    socialOrientMap.get(f, 0),
                    proxMap.get(f, 0),
                    binauralAudMap.get(f, 0),
                    monauralAudMap.get(f, 0),
                    rearAudMap.get(f, 0),
                    binocularVisMap.get(f, 0),
                    monocularVisMap.get(f, 0),
                    noVisMap.get(f, 0)
                ])

            behaviorManifest.append([
                frame_count,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ])

            print("#     [100%]")
            return behaviorManifest

        bManifest = set_behavior_manifest(
            approachManifest,
            followManifest,
            chaseManifest,
            sniffManifest,
            hhManifest,
            hoManifest,
            avoidManifest,
            fleeManifest,
            disengageManifest,
            stationaryProxManifest,
            socialOrientationManifest,
            aud_array,
            vis_array,
            prox_array
        )
        bManifest= filter_bmanifest(bManifest, fps)

        def set_heatmap_manifest(behaviorManifest):
            heatmapManifest = []
            ticker = max(1, math.floor(len(behaviorManifest) / 9))

            def compare_score(score, newscore):
                return score if abs(score) >= abs(newscore) else newscore

            for i, row in enumerate(behaviorManifest):
                if i % ticker == 0:
                    print("#", end="")

                frame_idx = row[0]
                score = 0

                # Column map
                approach = row[1]
                follow = row[2]
                chase = row[3]
                sniff = row[4]
                hh = row[5]
                ho = row[6]
                activeAvoid = row[7]
                passiveAvoid = row[8]
                flee = row[9]
                disengage = row[10]
                stationaryProx = row[11]
                socialOrient = row[12]
                proximity = row[13]
                binauralAud = row[14]
                monauralAud = row[15]
                rearAud = row[16]
                binocularVis = row[17]
                monocularVis = row[18]
                noVis = row[19]

                # negative / withdrawal
                if flee == 1:
                    score = compare_score(score, -10)
                if activeAvoid == 1:
                    score = compare_score(score, -7)
                if passiveAvoid == 1:
                    score = compare_score(score, -4)
                if disengage == 1:
                    score = compare_score(score, -2)

                # positive locomotor-social
                if approach == 1:
                    score = compare_score(score, 3)
                if follow == 1:
                    score = compare_score(score, 5)
                if chase == 1:
                    score = compare_score(score, 8)

                # contact social
                if sniff == 1:
                    score = compare_score(score, 6)
                if hh == 1:
                    score = compare_score(score, 9)
                if ho == 1:
                    score = compare_score(score, 10)

                # contextual social
                if stationaryProx == 1:
                    score = compare_score(score, 2)
                if socialOrient == 1:
                    score = compare_score(score, 2)
                if proximity == 1:
                    score = compare_score(score, 1)

                # sensory states can gently modulate if nothing stronger is happening
                if score == 0:
                    if binocularVis == 1:
                        score = compare_score(score, 1)
                    elif monocularVis == 1:
                        score = compare_score(score, 0.5)

                    if binauralAud == 1:
                        score = compare_score(score, 1)
                    elif monauralAud == 1:
                        score = compare_score(score, 0.5)
                    elif rearAud == 1:
                        score = compare_score(score, -0.5)

                    if noVis == 1:
                        score = compare_score(score, -0.5)

                heatmapManifest.append([i + 1 / fps, score])

            print("#     [100%]")
            return heatmapManifest

        def extract_behavior_intervals(behaviorManifest, behavior):
            behavior_index = {
                "approach": 1,
                "follow": 2,
                "chase": 3,
                "sniff": 4,
                "hh": 5,
                "ho": 6,
                "activeAvoid": 7,
                "passiveAvoid": 8,
                "flee": 9,
                "disengage": 10,
                "stationaryProx": 11,
                "socialOrient": 12,
                "proximity": 13,
                "binauralAud": 14,
                "monauralAud": 15,
                "rearAud": 16,
                "binocularVis": 17,
                "monocularVis": 18,
                "noVis": 19
            }

            if behavior not in behavior_index:
                raise ValueError(f"Unknown behavior: {behavior}")

            idx = behavior_index[behavior]
            intervals = []
            in_interval = False
            start_frame = None

            for row in behaviorManifest:
                frame = row[0]
                val = row[idx]
                is_active = val != 0

                if is_active and not in_interval:
                    in_interval = True
                    start_frame = frame
                elif not is_active and in_interval:
                    in_interval = False
                    end_frame = frame
                    intervals.append((start_frame / fps, end_frame / fps))
                    start_frame = None

            if in_interval:
                end_frame = behaviorManifest[-1][0]
                intervals.append((start_frame / fps, end_frame / fps))

            return intervals

        def prox_intervals(prox_array, fps):
            intervals, durations = [], []
            start = None
            for frame, _, flag in prox_array:
                if flag == 1 and start is None:
                    start = frame
                elif flag == 0 and start is not None:
                    end = frame - 1
                    intervals.append([start / fps, end / fps])
                    durations.append((end - start + 1) / fps)
                    start = None
            if start is not None:  # handle last bout
                end = prox_array[-1][0]
                intervals.append([start / fps, end / fps])
                durations.append((end - start + 1) / fps)
            return intervals, durations

        proxIntervals, timesTogether = prox_intervals(prox_array, fps)

        def behavior_time_from_manifest(bManifest, behavior_name):
            behavior_index = {
                "approach": 1,
                "follow": 2,
                "chase": 3,
                "sniff": 4,
                "hh": 5,
                "ho": 6,
                "activeAvoid": 7,
                "passiveAvoid": 8,
                "flee": 9,
                "disengage": 10,
                "stationaryProx": 11,
                "socialOrient": 12,
                "proximity": 13,
                "binauralAud": 14,
                "monauralAud": 15,
                "rearAud": 16,
                "binocularVis": 17,
                "monocularVis": 18,
                "noVis": 19
            }

            idx = behavior_index[behavior_name]
            f = 0
            for row in bManifest:
                if row[idx] >= 1:
                    f += 1
            return f / fps

        def get_interval_durations(intervals):
            durations = []
            for start, end in intervals:
                durations.append(end - start)
            return durations

        def time_track(manifest):
            f = 0
            for i in range(0, len(manifest)):
                if manifest[i][2] >= 1:
                    f += 1
            t = f/fps
            return t
            
        def activeAvoidTime_track(manifest):
            f = 0
            for i in range(0, len(manifest)):
                if manifest[i][2] == 'a' or manifest[i][1] == 'a/p':
                    f += 1
                t = f/fps
            return t

        def weightedTime_track(manifest, weight):
            f = 0
            for i in range(0, len(manifest)):
                if manifest[i][2] == weight:
                    f += 1
                t = f/fps
            return t

        approachIntervals = extract_behavior_intervals(bManifest, "approach")
        followIntervals = extract_behavior_intervals(bManifest, "follow")
        chaseIntervals = extract_behavior_intervals(bManifest, "chase")
        sniffIntervals = extract_behavior_intervals(bManifest, "sniff")
        hhIntervals = extract_behavior_intervals(bManifest, "hh")
        hoIntervals = extract_behavior_intervals(bManifest, "ho")
        activeAvoidIntervals = extract_behavior_intervals(bManifest, "activeAvoid")
        passiveAvoidIntervals = extract_behavior_intervals(bManifest, "passiveAvoid")
        fleeIntervals = extract_behavior_intervals(bManifest, "flee")
        disengageIntervals = extract_behavior_intervals(bManifest, "disengage")
        stationaryProxIntervals = extract_behavior_intervals(bManifest, "stationaryProx")
        socialOrientIntervals = extract_behavior_intervals(bManifest, "socialOrient")
        proxIntervals = extract_behavior_intervals(bManifest, "proximity")

        binauralAudIntervals = extract_behavior_intervals(bManifest, "binauralAud")
        monauralAudIntervals = extract_behavior_intervals(bManifest, "monauralAud")
        rearAudIntervals = extract_behavior_intervals(bManifest, "rearAud")

        binocularVisIntervals = extract_behavior_intervals(bManifest, "binocularVis")
        monocularVisIntervals = extract_behavior_intervals(bManifest, "monocularVis")
        noVisIntervals = extract_behavior_intervals(bManifest, "noVis")

        sniffInstigators = det_instigator(sniffIntervals, bout_type="sniff")
        hhInstigators = det_instigator(hhIntervals, bout_type="hh")
        hoInstigators = det_instigator(hoIntervals, bout_type="ho")
        proximityInstigators = det_instigator(proxIntervals, bout_type="proximity")

        approachTime = behavior_time_from_manifest(bManifest, "approach")
        followTime = behavior_time_from_manifest(bManifest, "follow")
        chaseTime = behavior_time_from_manifest(bManifest, "chase")

        sniffTime = behavior_time_from_manifest(bManifest, "sniff")
        hhTime = behavior_time_from_manifest(bManifest, "hh")
        hoTime = behavior_time_from_manifest(bManifest, "ho")

        activeAvoidTime = behavior_time_from_manifest(bManifest, "activeAvoid")
        passiveAvoidTime = behavior_time_from_manifest(bManifest, "passiveAvoid")
        fleeTime = behavior_time_from_manifest(bManifest, "flee")
        disengageTime = behavior_time_from_manifest(bManifest, "disengage")

        stationaryProxTime = behavior_time_from_manifest(bManifest, "stationaryProx")
        socialOrientTime = behavior_time_from_manifest(bManifest, "socialOrient")
        proximityTime = behavior_time_from_manifest(bManifest, "proximity")

        binauralAudTime = behavior_time_from_manifest(bManifest, "binauralAud")
        monauralAudTime = behavior_time_from_manifest(bManifest, "monauralAud")
        rearAudTime = behavior_time_from_manifest(bManifest, "rearAud")

        binocularVisTime = behavior_time_from_manifest(bManifest, "binocularVis")
        monocularVisTime = behavior_time_from_manifest(bManifest, "monocularVis")
        noVisTime = behavior_time_from_manifest(bManifest, "noVis")

        # Sample data
        mouse1_locations = np.array(list(zip(cm_x_id1, cm_y_id1)))
        mouse2_locations = np.array(list(zip(cm_x_id2, cm_y_id2)))

        import matplotlib.pyplot as plt

        def plot_mouse_trace_gradient(ax, locations, scores, cmap='coolwarm'):
            norm = mcolors.Normalize(vmin=-10, vmax=10)
            colormap = plt.cm.get_cmap(cmap)
            colors = colormap(norm(scores))

            segments = [
                [locations[i], locations[i + 1]]
                for i in range(gate, len(locations) - 1)
            ]
            line_colors = colors[:-1]

            lc = mc.LineCollection(segments, colors=line_colors, linewidths=2, label='Mouse 1 (gradient)')
            ax.add_collection(lc)

            ax.scatter(*locations[gate], color='gray', label='Start')
            ax.scatter(*locations[-1], color='black', label='End')

            return colormap, norm

        def plot_mouse_trace_plain(ax, locations, color='black'):
            segments = [
                [locations[i], locations[i + 1]]
                for i in range(gate, len(locations) - 1)
            ]
            lc = mc.LineCollection(segments, colors=color, linewidths=2, label='Mouse 2')
            ax.add_collection(lc)

            ax.scatter(*locations[gate], color='gray')
            ax.scatter(*locations[-1], color='black')

        import matplotlib.pyplot as plt

        def combined_ethogram(bManifest, fps, base_name):
            """
            Expanded row-based ethogram for the full behavior manifest.

            Manifest columns:
            [frame,
             approach,
             follow,
             chase,
             sniff,
             hh,
             ho,
             activeAvoid,
             passiveAvoid,
             flee,
             disengage,
             stationaryProx,
             socialOrient,
             proximity,
             binauralAud,
             monauralAud,
             rearAud,
             binocularVis,
             monocularVis,
             noVis]
            """

            data = np.array(bManifest, dtype=object)
            frames = data[:, 0].astype(float) / fps
            frame_duration = np.median(np.diff(frames)) if len(frames) > 1 else 1.0 / fps

            behavior_index = {
                "approach": 1,
                "follow": 2,
                "chase": 3,
                "sniff": 4,
                "hh": 5,
                "ho": 6,
                "activeAvoid": 7,
                "passiveAvoid": 8,
                "flee": 9,
                "disengage": 10,
                "stationaryProx": 11,
                "socialOrient": 12,
                "proximity": 13,
                "binauralAud": 14,
                "monauralAud": 15,
                "rearAud": 16,
                "binocularVis": 17,
                "monocularVis": 18,
                "noVis": 19,
            }

            categories = [
                "approach",
                "follow",
                "chase",
                "sniff",
                "hh",
                "ho",
                "activeAvoid",
                "passiveAvoid",
                "flee",
                "disengage",
                "stationaryProx",
                "socialOrient",
                "proximity",
                "binauralAud",
                "monauralAud",
                "rearAud",
                "binocularVis",
                "monocularVis",
                "noVis",
            ]

            pretty_names = {
                "approach": "Approach",
                "follow": "Follow",
                "chase": "Chase",
                "sniff": "Sniff",
                "hh": "Head-Head",
                "ho": "Head-On",
                "activeAvoid": "Active Avoid",
                "passiveAvoid": "Passive Avoid",
                "flee": "Flee",
                "disengage": "Disengage",
                "stationaryProx": "Stationary Prox",
                "socialOrient": "Social Orient",
                "proximity": "Proximity",
                "binauralAud": "Binaural Aud",
                "monauralAud": "Monaural Aud",
                "rearAud": "Rear Aud",
                "binocularVis": "Binocular Vis",
                "monocularVis": "Monocular Vis",
                "noVis": "No Vis",
            }

            base_colors = {
                "approach": "#66c2a5",
                "follow": "#377eb8",
                "chase": "#1f78b4",
                "sniff": "#984ea3",
                "hh": "#ff7f00",
                "ho": "#ffd92f",
                "activeAvoid": "#e41a1c",
                "passiveAvoid": "#fb8072",
                "flee": "#a50f15",
                "disengage": "#8da0cb",
                "stationaryProx": "#a65628",
                "socialOrient": "#4daf4a",
                "proximity": "#b15928",
                "binauralAud": "#e41a1c",
                "monauralAud": "#377eb8",
                "rearAud": "#4daf4a",
                "binocularVis": "#e41a1c",
                "monocularVis": "#377eb8",
                "noVis": "#999999",
            }

            row_height = 0.5
            margin = 2 / 72.0
            fig_height = len(categories) * (row_height + margin)
            fig, ax = plt.subplots(figsize=(16, fig_height))

            for row_idx, cat in enumerate(categories):
                y_bottom = row_idx * (row_height + margin)
                col = behavior_index[cat]

                start = None
                for i in range(len(frames)):
                    val = int(data[i, col])
                    if val == 1 and start is None:
                        start = frames[i]
                    elif (val == 0 or i == len(frames) - 1) and start is not None:
                        end = frames[i] if val == 0 else frames[i] + frame_duration
                        rect = Rectangle(
                            (start, y_bottom),
                            width=(end - start),
                            height=row_height,
                            facecolor=base_colors[cat],
                            edgecolor="none"
                        )
                        ax.add_patch(rect)
                        start = None

            ax.set_xlim(frames[0], frames[-1] + frame_duration)
            ax.set_ylim(0, len(categories) * (row_height + margin))
            ax.set_yticks([
                i * (row_height + margin) + row_height / 2 for i in range(len(categories))
            ])
            ax.set_yticklabels([pretty_names[c] for c in categories], fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_title("Combined Ethogram")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            output_dir = Path("./exports/")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{base_name}_ETHOGRAM.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✅ Row-based combined ethogram saved to {output_path}")

        def build_iManifest(
                fps,
                approachIntervals,
                followIntervals,
                chaseIntervals,
                sniffIntervals,
                hhIntervals,
                hoIntervals,
                activeAvoidIntervals,
                passiveAvoidIntervals,
                fleeIntervals,
                disengageIntervals,
                stationaryProxIntervals,
                socialOrientIntervals,
                proxIntervals,
                binauralAudIntervals,
                monauralAudIntervals,
                rearAudIntervals,
                binocularVisIntervals,
                monocularVisIntervals,
                noVisIntervals,
                sniffInstigators=None,
                hhInstigators=None,
                hoInstigators=None,
                proximityInstigators=None
        ):
            """
            Build a long-format interval manifest.

            Output rows:
            [Bout_ID, Behavior, Start_s, End_s, Duration_s, Instigator]

            Ordered by behavior type first, then by start time within behavior.
            """
            iManifest = []
            bout_counter = 1

            def add_rows(label, intervals, instigators=None):
                nonlocal bout_counter

                if intervals is None:
                    return
                if instigators is None:
                    instigators = ["" for _ in intervals]

                for (start, end), inst in zip(intervals, instigators):
                    dur = end - start
                    bout_id = f"B{bout_counter:05d}"
                    iManifest.append([
                        bout_id,
                        label,
                        round(start, 4),
                        round(end, 4),
                        round(dur, 4),
                        inst if inst is not None else ""
                    ])
                    bout_counter += 1

            add_rows("Proximity", proxIntervals, proximityInstigators)
            add_rows("StationaryProx", stationaryProxIntervals)
            add_rows("SocialOrient", socialOrientIntervals)

            add_rows("Approach", approachIntervals)
            add_rows("Follow", followIntervals)
            add_rows("Chase", chaseIntervals)

            add_rows("Sniff", sniffIntervals, sniffInstigators)
            add_rows("Head-Head", hhIntervals, hhInstigators)
            add_rows("Head-On", hoIntervals, hoInstigators)

            add_rows("Disengage", disengageIntervals)
            add_rows("PassiveAvoid", passiveAvoidIntervals)
            add_rows("ActiveAvoid", activeAvoidIntervals)
            add_rows("Flee", fleeIntervals)

            add_rows("BinauralAud", binauralAudIntervals)
            add_rows("MonauralAud", monauralAudIntervals)
            add_rows("RearAud", rearAudIntervals)

            add_rows("BinocularVis", binocularVisIntervals)
            add_rows("MonocularVis", monocularVisIntervals)
            add_rows("NoVis", noVisIntervals)

            behavior_order = {
                "Proximity": 0,
                "StationaryProx": 1,
                "SocialOrient": 2,

                "Approach": 3,
                "Follow": 4,
                "Chase": 5,

                "Sniff": 6,
                "Head-Head": 7,
                "Head-On": 8,

                "Disengage": 9,
                "PassiveAvoid": 10,
                "ActiveAvoid": 11,
                "Flee": 12,

                "BinauralAud": 13,
                "MonauralAud": 14,
                "RearAud": 15,

                "BinocularVis": 16,
                "MonocularVis": 17,
                "NoVis": 18,
            }

            # sort by behavior first, then by start time
            iManifest.sort(key=lambda row: (behavior_order.get(row[1], 999), row[2]))

            # reassign Bout_ID after sorting
            for i, row in enumerate(iManifest, start=1):
                row[0] = f"B{i:05d}"

            return iManifest

        iManifest = build_iManifest(
            fps,
            approachIntervals,
            followIntervals,
            chaseIntervals,
            sniffIntervals,
            hhIntervals,
            hoIntervals,
            activeAvoidIntervals,
            passiveAvoidIntervals,
            fleeIntervals,
            disengageIntervals,
            stationaryProxIntervals,
            socialOrientIntervals,
            proxIntervals,
            binauralAudIntervals,
            monauralAudIntervals,
            rearAudIntervals,
            binocularVisIntervals,
            monocularVisIntervals,
            noVisIntervals,
            sniffInstigators=sniffInstigators,
            hhInstigators=hhInstigators,
            hoInstigators=hoInstigators,
            proximityInstigators=proximityInstigators
        )

        def add_behavior_group_headers(iManifest):
            """
            Insert blank/header rows between behavior groups for CSV readability.
            """
            if not iManifest:
                return iManifest

            grouped = []
            current_behavior = None

            for row in iManifest:
                behavior = row[1]
                if behavior != current_behavior:
                    current_behavior = behavior
                    grouped.append(["", f"--- {behavior} ---", "", "", "", ""])
                grouped.append(row)

            return grouped

        iManifest_for_export = add_behavior_group_headers(iManifest)

        hManifest = set_heatmap_manifest(bManifest)
        mouse1_scores = []
        for i in range(0, len(hManifest)):
            mouse1_scores.append(hManifest[i][1])
        mouse1_scores = np.array(list(zip(mouse1_scores)))

        aDuration = duration - gateTime
        aFrameCount = frame_count - gate

        import matplotlib.pyplot as plt
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Mouse 2 with fixed color (not in plasma gradient)
        plot_mouse_trace_plain(ax, mouse2_locations, color='gray')

        # Mouse 1 with gradient
        colormap, norm = plot_mouse_trace_gradient(ax, mouse1_locations, np.array(mouse1_scores), cmap='coolwarm')


        # Colorbar for gradient
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Score')

        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Mouse Trace & Score")
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, frameon=False)
        plt.tight_layout()
        # Define your custom path and filename
        output_dir = Path("./exports/")  # Directory where you want to save
        output_name = (base_name + "_TRACE.png")  # Custom name for your plot file

        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full path to save the figure
        output_path = output_dir / output_name
            
        # Save the figure with the custom path and filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Adjust options as needed
        plt.close()

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        def visualize_dsr(coords, frame_idx=0, hl_index=8, hr_index=7, ns_index=1, cm_index=6,
                          percentile_range=(20, 80), show_rings=True, savepath=None):
            """
            Visualizes and validates the dynamic sniff range (dsr) calculation.

            Parameters
            ----------
            coords : np.ndarray
                Full coordinate array of shape (frames, nodes, 2).
            frame_idx : int
                Frame to visualize (should be one with both hips visible).
            hl_index, hr_index, ns_index, cm_index : int
                Node indices for hip-left, hip-right, nose, and center mass.
            percentile_range : tuple
                Percentile range for width filtering (default 20–80).
            show_rings : bool
                Whether to overlay 1×, 3×, and 6× dsr rings for visualization.
            savepath : str or None
                Optional path to save figure as PNG.
            """

            # --- Compute hip–hip widths ---
            widths = np.linalg.norm(coords[:, hl_index, :] - coords[:, hr_index, :], axis=1)
            low, high = np.percentile(widths, percentile_range)
            filtered = widths[(widths >= low) & (widths <= high)]
            dsr = np.median(filtered)
            BL = 2 * dsr

            print(f"Estimated dynamic sniff range (dsr): {dsr:.2f} px")
            print(f"Equivalent body length (BL ≈ 2×dsr): {BL:.2f} px")
            print(f"20–80% width range: {low:.2f}–{high:.2f} px")

            # --- Plot width distribution ---
            plt.figure(figsize=(6, 3))
            plt.hist(widths, bins=50, color='gray', alpha=0.7)
            plt.axvline(dsr, color='deepskyblue', linestyle='--', label='Median DSR')
            plt.title("Hip–Hip Width Distribution")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Frame count")
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, ncol = 2)
            plt.tight_layout()
            plt.close()

            # --- Visual overlay on chosen frame ---
            if show_rings:
                plt.figure(figsize=(6, 6))
                plt.gca().invert_yaxis()
                frame = coords[frame_idx]
                cm = frame[cm_index]
                nose = frame[ns_index]

                # Plot all body nodes
                plt.scatter(frame[:, 0], frame[:, 1], s=30, c='gray', label='Nodes')
                plt.plot([frame[hl_index, 0], frame[hr_index, 0]],
                         [frame[hl_index, 1], frame[hr_index, 1]],
                         c='orange', lw=2, label='Hip–Hip')

                plt.scatter(*cm, c='red', s=50, label='Center mass')
                plt.scatter(*nose, c='blue', s=50, label='Nose')

                # Draw rings around center mass
                for r, col in zip([0.5, 1, 2, 3], ['gold', 'lime', 'blue', 'red']):
                    circle = Circle(cm, radius=r * dsr, fill=False, lw=1.5, edgecolor=col)
                    plt.gca().add_patch(circle)
                    plt.text(cm[0] + r * dsr, cm[1], f"{r}×DSR", color=col,
                             fontsize=9, ha='left', va='bottom')

                plt.title(f"Frame {frame_idx} — Visualizing DSR Scaling")
                plt.axis('equal')
                plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, ncol = 3)
                plt.tight_layout()
                if savepath:
                    plt.savefig(savepath, dpi=300)
                plt.close()

            return dsr, BL
        output_path = Path("./exports/") / f"{base_name}_DSR.png"
        dsr, BL = visualize_dsr(coords, frame_idx=150, savepath = output_path)

        pManifestDraft = [
            [f"{start} - {end}" for start, end in proxIntervals],
            timesTogether,
            proximityInstigators
        ]       
        pManifest = zip_longest(*pManifestDraft, fillvalue="") #pads shorter columns

        combined_ethogram(bManifest, fps, base_name)

        def prox_sniff_instigator_plot(prox_array, sniff_array, sniff_intervals, instigators,
                               locations, subject_track, stranger_track,
                               fps, base_name):
            """
            Creates a two-panel figure:
              - Top: CM–CM distance vs time
              - Bottom: ethogram with proximity and sniffing bouts (colored by instigator)
            """

            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Patch
            from pathlib import Path

            # === Calculate CM distance trace ===
            cm_subj = locations[:, 6, :, subject_track]  # cm_index = 6
            cm_strg = locations[:, 6, :, stranger_track]
            cm_dist = np.linalg.norm(cm_subj - cm_strg, axis=1) * cm_per_px
            time_axis = np.arange(len(cm_dist)) / fps

            # === Prepare ethogram tracks ===
            prox_flags = np.array([row[2] for row in prox_array])
            sniff_flags = np.array([row[2] for row in sniff_array])

            # === Figure setup ===
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                                 gridspec_kw={'height_ratios': [2, 1]})

            # --- Top panel: distance curve ---
            ax_top.plot(time_axis, cm_dist, color="#444444", linewidth=1.5)
            ax_top.set_ylabel("Inter-mouse Distance (cm)")
            ax_top.set_title("Inter-mouse CM Distance and Sniffing Behavior")

            # Highlight proximity periods (background shading)
            in_bout = False
            start = None
            for frame, _, flag in prox_array:
                if flag == 1 and not in_bout:
                    start = frame / fps
                    in_bout = True
                elif flag == 0 and in_bout:
                    end = frame / fps
                    ax_top.axvspan(start, end, color="#a65628", alpha=0.15)
                    in_bout = False
            if in_bout:  # handle last bout
                ax_top.axvspan(start, time_axis[-1], color="#a65628", alpha=0.15)

            # --- Bottom panel: ethogram ---
            y_positions = {"Subject": 0.6, "Shared": 0.45, "Stranger": 0.3}
            colors = {"Subject": "#377eb8", "Stranger": "#e41a1c", "Shared": "#999999"}

            # Draw invisible background bars to preserve height even if empty
            for who, y in y_positions.items():
                ax_bot.barh(y, time_axis[-1], left=0, height=0.2, color="none", edgecolor="none")

            # Draw actual bouts
            for (start, end), who in zip(sniff_intervals, instigators):
                y = y_positions.get(who, 0.45)
                ax_bot.barh(y, end - start, left=start,
                            height=0.2, color=colors[who], align='center')

            ax_bot.set_yticks(list(y_positions.values()))
            ax_bot.set_yticklabels(["Subject\ninit.", "Shared", "Stranger\ninit."])
            ax_bot.set_ylim(0.1, 0.8)  # <-- ensures constant vertical spacing
            ax_bot.set_ylabel("Sniffing")
            ax_bot.set_xlabel("Time (s)")
            ax_bot.set_xlim(time_axis[0], time_axis[-1])

            # Legend
            legend_patches = [
                Patch(facecolor="#a65628", alpha=0.15, label="Proximity Period"),
                Patch(facecolor="#377eb8", label="Subject Initiated Sniff"),
                Patch(facecolor="#e41a1c", label="Stranger Initiated Sniff"),
                Patch(facecolor="#999999", label="Shared")
            ]
            ax_bot.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.25), loc='upper center', frameon=False, ncol = 4)

            plt.tight_layout()
            output_dir = Path("./exports/")
            output_dir.mkdir(parents=True, exist_ok=True)
            outpath = output_dir / f"{base_name}_PROX_SNIFF_ETHO.png"
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved proximity–sniff ethogram: {outpath}")

        def extract_simple_intervals(binary_array, fps):
            """
            Convert a boolean vector (True=active bout) into a list of (start_time, end_time) intervals.

            Parameters
            ----------
            bool_array : np.ndarray (frames)
                Boolean mask (True during active behavior).
            fps : float
                Frames per second.

            Returns
            -------
            list of (float, float)
                Start and end times (in seconds) for each continuous True segment.
            """
            intervals = []
            in_bout = False
            start_frame = None

            for i, val in enumerate(binary_array):
                if val==1 and not in_bout:
                    in_bout = True
                    start_frame = i
                elif val==0 and in_bout:
                    in_bout = False
                    end_frame = i
                    intervals.append((start_frame / fps, end_frame / fps))

            # Handle final bout
            if in_bout and start_frame is not None:
                intervals.append((start_frame / fps, len(binary_array) / fps))

            return intervals


        sniff_intervals = extract_simple_intervals(bManifest[:, 3], fps)
        bout_instigators, frame_instigators = det_instigator_boutwise(
            sniff_intervals, fps, sniffManifest, locations,
            subject_track, stranger_track, pre_window_s=1.0
        )



        # Create combined figure
        prox_sniff_instigator_plot(prox_array, sniffManifest, sniff_intervals, bout_instigators,
                           locations, subject_track, stranger_track,
                           fps, base_name)


        # Full path to file
        # Full path to file
        filepath = os.path.join(subfolder, base_name + ".csv")

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)

            # =========================
            # METADATA
            # =========================
            writer.writerow(["===METADATA==="])
            writer.writerow(["Source File", filename])
            writer.writerow(["Duration_s", duration])
            writer.writerow(["Frames", frame_count])
            writer.writerow(["Analysis Duration after Gating_s", aDuration])
            writer.writerow(["Analysis Frame Count after Gating", aFrameCount])
            writer.writerow([])

            # =========================
            # TIME SUMMARY
            # =========================
            writer.writerow(["===TIMES_SUM==="])
            writer.writerow(["Total Time Spent Together_s", sumTotalTime])

            writer.writerow(["Approach Total Time_s", approachTime])
            writer.writerow(["Follow Total Time_s", followTime])
            writer.writerow(["Chase Total Time_s", chaseTime])

            writer.writerow(["Sniff Total Time_s", sniffTime])
            writer.writerow(["Head-Head Total Time_s", hhTime])
            writer.writerow(["Head-On Total Time_s", hoTime])

            writer.writerow(["Active Avoidance Total Time_s", activeAvoidTime])
            writer.writerow(["Passive Avoidance Total Time_s", passiveAvoidTime])
            writer.writerow(["Flee Total Time_s", fleeTime])
            writer.writerow(["Disengage Total Time_s", disengageTime])

            writer.writerow(["Stationary Proximity Total Time_s", stationaryProxTime])
            writer.writerow(["Social Orientation Total Time_s", socialOrientTime])
            writer.writerow(["Proximity Total Time_s", proximityTime])

            writer.writerow(["Binaural Audio Total Time_s", binauralAudTime])
            writer.writerow(["Monaural Audio Total Time_s", monauralAudTime])
            writer.writerow(["Rear Audio Total Time_s", rearAudTime])

            writer.writerow(["Binocular Vision Total Time_s", binocularVisTime])
            writer.writerow(["Monocular Vision Total Time_s", monocularVisTime])
            writer.writerow(["No Visual Total Time_s", noVisTime])
            writer.writerow([])

            # =========================
            # BEHAVIOR INTERVALS
            # =========================
            writer.writerow(["===BEHAVIOR_INTERVALS==="])
            writer.writerow(["Bout_ID", "Behavior", "Start_s", "End_s", "Duration_s", "Instigator"])
            writer.writerows(iManifest_for_export)
            writer.writerow([])

            # =========================
            # FILTERED MATRIX
            # =========================
            writer.writerow(["===FILTERED_MATRIX==="])
            writer.writerow([
                "Frame",
                "Approach",
                "Follow",
                "Chase",
                "Sniff",
                "Head-Head",
                "Head-On",
                "ActiveAvoid",
                "PassiveAvoid",
                "Flee",
                "Disengage",
                "StationaryProx",
                "SocialOrient",
                "Proximity",
                "BinauralAud",
                "MonauralAud",
                "RearAud",
                "BinocularVis",
                "MonocularVis",
                "NoVis"
            ])
            writer.writerows(bManifest)

        self.update_status(f"File saved to: {filepath}")
        print()
        print()
        print()
        self.update_status("<<<< !!!DONE!!! >>>>    :)")
        pass


    def process_files(self):
        processing_thread = threading.Thread(target=self._process_files_thread)
        processing_thread.start()

    def _process_files_thread(self):
        try:
            self.update_status("|> Process started for all file pairs...\n")
            if not self.video_paths or not self.h5_paths:
                self.update_status("!X!X!X! No files selected.")
                return

            # Run up to the shorter of the two lists
            for i, (video_path, h5_path) in enumerate(zip(self.video_paths, self.h5_paths), 1):
                self.update_status(f"\n|> Processing pair {i}/{min(len(self.video_paths), len(self.h5_paths))}")
                self.video_path = video_path
                self.h5_path = h5_path

            # Call your current processing code here
            self._run_single_analysis(video_path, h5_path)

        # After all files processed
            self._show_completion_popup()

        except Exception as e:
            import traceback
            print("!X!X!X! An error occurred:")
            traceback.print_exc()

# Run the app
if __name__ == "__main__":
        root = tk.Tk()
        app = SinglePairApp(root)
        root.mainloop()

