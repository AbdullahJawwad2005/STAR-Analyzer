"""
test_rf.py — Synthetic end-to-end test for rf_analysis pipeline.
Run:  python test_rf.py
"""
import sys, traceback
import numpy as np
import pandas as pd
import tempfile, os
from pathlib import Path

# ── helpers ─────────────────────────────────────────────────────────────────

def _print(msg):
    print(msg, flush=True)

def _ok(label):
    print(f"  [OK]  {label}", flush=True)

def _fail(label, exc=None):
    print(f"  [FAIL] {label}", flush=True)
    if exc:
        traceback.print_exc()

# ── Build synthetic data ─────────────────────────────────────────────────────

def make_synthetic_data(n_frames=3000, n_tracks=2, fps=25.0, seed=42):
    """
    Create realistic synthetic track_arrays, pair_arrays, pair_beh that
    mimic real STAR Analyzer pipeline outputs.
    """
    rng = np.random.default_rng(seed)

    # ── track_arrays: {track_idx: {feat_name: (n_frames,)}} ──────────────────
    # Each track gets a handful of features representing every _classify_feat category

    def make_track(t_seed):
        r = np.random.default_rng(t_seed)
        d = {}
        # position
        d['nose_x'] = r.normal(320, 80, n_frames)
        d['nose_y'] = r.normal(240, 60, n_frames)
        d['body_x'] = d['nose_x'] + r.normal(0, 5, n_frames)
        d['body_y'] = d['nose_y'] + r.normal(0, 5, n_frames)
        # velocity / speed
        d['nose_speed']  = np.abs(r.normal(5, 3, n_frames))
        d['nose_vx']     = r.normal(0, 3, n_frames)
        d['nose_vy']     = r.normal(0, 3, n_frames)
        d['body_speed']  = np.abs(r.normal(5, 3, n_frames))
        # accel
        d['nose_accel']  = np.abs(r.normal(1, 0.5, n_frames))
        # jerk
        d['nose_jerk']   = r.normal(0, 0.2, n_frames)
        # angular
        d['body_heading_deg'] = r.uniform(-180, 180, n_frames)
        # shape
        d['elongation']       = 1.0 + np.abs(r.normal(0, 0.3, n_frames))
        d['eccentricity']     = np.clip(np.abs(r.normal(0.5, 0.2, n_frames)), 0, 1)
        d['compactness']      = 1.0 + np.abs(r.normal(0.1, 0.05, n_frames))
        d['circularity']      = np.clip(np.abs(r.normal(0.8, 0.1, n_frames)), 0, 1)
        d['hourglass_area']   = np.abs(r.normal(500, 100, n_frames))
        d['hourglass_ratio']  = np.abs(r.normal(1, 0.3, n_frames))
        d['path_efficiency']  = np.clip(r.normal(0.7, 0.2, n_frames), 0, 1)
        d['total_disp']       = np.cumsum(np.abs(r.normal(0, 2, n_frames)))
        # speed_accel (classified as accel because 'accel' is in name)
        d['speed_accel']      = r.normal(0, 0.5, n_frames)
        # distance-type features
        d['dist_roi_center']  = np.abs(r.normal(100, 40, n_frames))
        d['dist_roi_boundary']= np.abs(r.normal(50, 20, n_frames))
        # binary behavior flag
        d['stationary']       = (np.abs(r.normal(0, 1, n_frames)) < 0.5).astype(np.int8)
        # cumulative displacement (shape)
        d['cm_total_disp']    = np.cumsum(np.abs(r.normal(0, 2, n_frames)))
        return d

    track_arrays = {0: make_track(seed), 1: make_track(seed + 1)}

    # ── pair_arrays: {'t0_t1/feat_name': (n_frames,)} ────────────────────────
    pfx = 't0_t1'
    pair_arrays = {}
    # distance
    pair_arrays[f'{pfx}/inter_animal_dist']         = np.abs(rng.normal(80, 30, n_frames))
    pair_arrays[f'{pfx}/inter_animal_displacement'] = rng.normal(0, 5, n_frames)
    # angular
    pair_arrays[f'{pfx}/approach_angle_A']          = rng.uniform(-180, 180, n_frames)
    pair_arrays[f'{pfx}/approach_angle_B']          = rng.uniform(-180, 180, n_frames)
    # velocity-type
    pair_arrays[f'{pfx}/velocity_cos_sim']          = np.clip(rng.normal(0, 0.4, n_frames), -1, 1)
    # distance (covar/corr)
    pair_arrays[f'{pfx}/pos_covariance_x']          = rng.normal(0, 100, n_frames)
    pair_arrays[f'{pfx}/pos_covariance_y']          = rng.normal(0, 100, n_frames)
    pair_arrays[f'{pfx}/pos_correlation_x']         = np.clip(rng.normal(0, 0.3, n_frames), -1, 1)
    pair_arrays[f'{pfx}/pos_correlation_y']         = np.clip(rng.normal(0, 0.3, n_frames), -1, 1)
    # categorical — should be skipped by _classify_feat
    pair_arrays[f'{pfx}/visual_scope_A']            = np.array(['Binocular'] * n_frames)
    pair_arrays[f'{pfx}/visual_scope_B']            = np.array(['Monocular'] * n_frames)

    # ── pair_beh: {'t0_t1/BehName': (n_frames,) int8} ───────────────────────
    # We need enough bouts of each class to survive the n_splits=5 filter.
    # Strategy: inject distinct, recognisable patterns per behavior.
    from behaviors import _SECOND_ORDER_BINARY_KEYS

    pair_beh = {}
    # Directional pair (t0 is always A, t1 is always B)
    behaviors_to_inject = {
        'Follow_AtoB':    (200, 50, 8),   # 8 bouts of ~50 frames each, starting at 200
        'Follow_BtoA':    (700, 50, 8),
        'Chase_AtoB':     (1200, 40, 6),
        'Chase_BtoA':     (1600, 40, 6),
        'Approach_AtoB':  (400, 30, 7),
        'Approach_BtoA':  (900, 30, 7),
        'Flee_AtoB':      (1400, 35, 6),
        'Flee_BtoA':      (100, 35, 6),
        'StationaryProx': (2000, 60, 10),
        'Disengaged':     (50, 25, 12),
    }

    for beh_key in _SECOND_ORDER_BINARY_KEYS:
        arr = np.zeros(n_frames, dtype=np.int8)
        if beh_key in behaviors_to_inject:
            start_offset, bout_len, n_bouts = behaviors_to_inject[beh_key]
            for i in range(n_bouts):
                s = start_offset + i * (bout_len + 20)
                e = min(s + bout_len, n_frames)
                if s < n_frames:
                    arr[s:e] = 1
        pair_beh[f'{pfx}/{beh_key}'] = arr

    return track_arrays, pair_arrays, pair_beh


# ── Test 1: build_bout_table ─────────────────────────────────────────────────

def test_build_bout_table(track_arrays, pair_arrays, pair_beh, fps):
    _print("\n=== Test 1: build_bout_table ===")
    from rf_analysis import build_bout_table

    try:
        df = build_bout_table(track_arrays, pair_arrays, pair_beh,
                              track_names=['Animal0', 'Animal1'],
                              fps=fps, min_bout_frames=3)
    except Exception as e:
        _fail("build_bout_table raised exception", e)
        traceback.print_exc()
        return None

    if df.empty:
        _fail("bout_df is empty — no bouts found!")
        return None

    _ok(f"bout_df has {len(df)} rows, {len(df.columns)} columns")

    # Check metadata columns
    for col in ('bout_id', 'pair', 'behavior', 'start_frame', 'end_frame',
                'duration_s', 'group'):
        if col not in df.columns:
            _fail(f"Missing metadata column: {col}")
        else:
            _ok(f"Metadata column present: {col}")

    # Check behaviors are canonical (no _AtoB / _BtoA suffix)
    unique_beh = df['behavior'].unique()
    _ok(f"Unique behaviors: {sorted(unique_beh)}")
    for b in unique_beh:
        if b.endswith('_AtoB') or b.endswith('_BtoA'):
            _fail(f"Behavior not canonicalized: {b}")

    # Check no all-NaN feature columns
    feat_cols = [c for c in df.columns
                 if c not in {'bout_id', 'pair', 'behavior', 'start_frame',
                               'end_frame', 'duration_s', 'group'}]
    _ok(f"Feature columns: {len(feat_cols)}")
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        _fail(f"All-NaN feature columns ({len(all_nan_cols)}): {all_nan_cols[:5]}")
    else:
        _ok("No all-NaN feature columns")

    # Verify duration_s is positive and sane
    bad_dur = df[df['duration_s'] <= 0]
    if len(bad_dur):
        _fail(f"{len(bad_dur)} rows have duration_s <= 0")
    else:
        _ok("All bout durations positive")

    # Check start_frame < end_frame
    bad_range = df[df['start_frame'] > df['end_frame']]
    if len(bad_range):
        _fail(f"{len(bad_range)} rows have start_frame > end_frame")
    else:
        _ok("All frame ranges valid (start <= end)")

    # Verify pair column
    bad_pair = df[df['pair'] != 't0_t1']
    if len(bad_pair):
        _fail(f"Unexpected pair values: {bad_pair['pair'].unique()}")
    else:
        _ok("Pair column correct")

    # Per-behavior counts
    beh_counts = df['behavior'].value_counts()
    _print(f"  Bouts per behavior:\n{beh_counts.to_string()}")

    return df


# ── Test 2: run_rf_analysis ──────────────────────────────────────────────────

def test_run_rf_analysis(bout_df, fps):
    _print("\n=== Test 2: run_rf_analysis ===")
    from rf_analysis import run_rf_analysis

    msgs = []
    def status_cb(m): msgs.append(m); _print(f"  STATUS: {m}")

    try:
        results = run_rf_analysis(
            bout_df,
            n_estimators=50,    # small for speed
            n_splits=3,
            n_perm_repeats=3,
            top_n_features=10,
            status_cb=status_cb,
        )
    except Exception as e:
        _fail("run_rf_analysis raised exception")
        traceback.print_exc()
        return None

    # Validate keys
    for key in ('bout_df', 'report_str', 'confusion', 'labels',
                'importance_global', 'importance_per_class'):
        if key not in results:
            _fail(f"Missing key in results: {key}")
        else:
            _ok(f"Result key present: {key}")

    # Confusion matrix shape
    labels = results['labels']
    cm = results['confusion']
    n_cls = len(labels)
    if cm.shape != (n_cls, n_cls):
        _fail(f"Confusion matrix shape {cm.shape} != ({n_cls},{n_cls})")
    else:
        _ok(f"Confusion matrix shape OK: {cm.shape}")

    # Check cm values are non-negative ints
    if np.any(cm < 0):
        _fail("Confusion matrix has negative values")
    else:
        _ok("Confusion matrix non-negative")

    # Total predictions should equal total filtered bouts
    total_preds = cm.sum()
    total_bouts = len(results['bout_df'])
    if total_preds != total_bouts:
        _fail(f"CM total ({total_preds}) != filtered bout count ({total_bouts})")
    else:
        _ok(f"CM total matches bout count: {total_preds}")

    # classification report string
    rstr = results['report_str']
    if not isinstance(rstr, str) or len(rstr) < 50:
        _fail(f"report_str too short or wrong type: {repr(rstr[:100])}")
    else:
        _ok("classification_report string present")
    _print("  Report:\n" + rstr)

    # global importance
    imp = results['importance_global']
    if not isinstance(imp, pd.DataFrame) or len(imp) == 0:
        _fail("importance_global is empty or wrong type")
    else:
        _ok(f"importance_global has {len(imp)} rows")
    for col in ('feature', 'importance_mean', 'importance_std'):
        if col not in imp.columns:
            _fail(f"importance_global missing column: {col}")

    # per-class importance
    imp_pc = results['importance_per_class']
    if not isinstance(imp_pc, dict) or len(imp_pc) == 0:
        _fail("importance_per_class is empty or wrong type")
    else:
        _ok(f"importance_per_class has {len(imp_pc)} classes")
    for cls_lbl, cls_imp in imp_pc.items():
        if not isinstance(cls_imp, pd.DataFrame) or len(cls_imp) == 0:
            _fail(f"importance_per_class[{cls_lbl}] is empty")
        if 'feature' not in cls_imp.columns or 'importance' not in cls_imp.columns:
            _fail(f"importance_per_class[{cls_lbl}] missing columns")

    # rf_prediction column in bout_df
    if 'rf_prediction' not in results['bout_df'].columns:
        _fail("rf_prediction column missing from bout_df")
    else:
        n_valid = results['bout_df']['rf_prediction'].notna().sum()
        _ok(f"rf_prediction column present ({n_valid} valid predictions)")

    return results


# ── Test 3: write_rf_outputs ─────────────────────────────────────────────────

def test_write_rf_outputs(results):
    _print("\n=== Test 3: write_rf_outputs ===")
    from rf_analysis import write_rf_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, 'test_session')
        try:
            paths = write_rf_outputs(results, base, write_plots=True)
        except Exception as e:
            _fail("write_rf_outputs raised exception")
            traceback.print_exc()
            return

        expected_suffixes = ['_rf_bouts.csv', '_rf_report.csv',
                              '_rf_importance.csv', '_rf_analysis.pdf']
        for suf in expected_suffixes:
            expected = os.path.join(tmpdir, f'test_session{suf}')
            if not os.path.exists(expected):
                _fail(f"Expected output missing: {suf}")
            else:
                size = os.path.getsize(expected)
                _ok(f"Output present: {suf} ({size} bytes)")

        # Spot-check CSV content
        bouts_path = os.path.join(tmpdir, 'test_session_rf_bouts.csv')
        if os.path.exists(bouts_path):
            df = pd.read_csv(bouts_path)
            _ok(f"_rf_bouts.csv: {len(df)} rows, {len(df.columns)} cols")
            if 'rf_prediction' not in df.columns:
                _fail("rf_prediction column missing from _rf_bouts.csv")
            if 'behavior' not in df.columns:
                _fail("behavior column missing from _rf_bouts.csv")

        report_path = os.path.join(tmpdir, 'test_session_rf_report.csv')
        if os.path.exists(report_path):
            df = pd.read_csv(report_path)
            _ok(f"_rf_report.csv: {len(df)} rows")
            for col in ('class', 'precision', 'recall', 'f1_score', 'support'):
                if col not in df.columns:
                    _fail(f"_rf_report.csv missing column: {col}")
                else:
                    _ok(f"  report column OK: {col}")

        imp_path = os.path.join(tmpdir, 'test_session_rf_importance.csv')
        if os.path.exists(imp_path):
            df = pd.read_csv(imp_path)
            _ok(f"_rf_importance.csv: {len(df)} rows")

        pdf_path = os.path.join(tmpdir, 'test_session_rf_analysis.pdf')
        if os.path.exists(pdf_path):
            size = os.path.getsize(pdf_path)
            if size < 1000:
                _fail(f"PDF too small ({size} bytes) — likely broken")
            else:
                _ok(f"PDF looks valid ({size} bytes)")


# ── Test 4: run_full_rf_pipeline ─────────────────────────────────────────────

def test_full_pipeline(track_arrays, pair_arrays, pair_beh, fps):
    _print("\n=== Test 4: run_full_rf_pipeline ===")
    from rf_analysis import run_full_rf_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, 'session')
        try:
            paths = run_full_rf_pipeline(
                track_arrays, pair_arrays, pair_beh,
                track_names=['Animal0', 'Animal1'],
                fps=fps,
                base_path=base,
                min_bout_frames=3,
                n_estimators=50,
                n_splits=3,
                write_plots=True,
                status_cb=lambda m: _print(f"  PIPE: {m}"),
            )
        except Exception as e:
            _fail("run_full_rf_pipeline raised exception")
            traceback.print_exc()
            return

        _ok(f"run_full_rf_pipeline returned {len(paths)} paths")
        for p in paths:
            if os.path.exists(p):
                _ok(f"  File exists: {Path(p).name} ({os.path.getsize(p)} bytes)")
            else:
                _fail(f"  File missing: {p}")


# ── Edge case: empty pair_beh ─────────────────────────────────────────────────

def test_empty_behavior(fps):
    _print("\n=== Test 5: empty pair_beh (all zeros) ===")
    from rf_analysis import build_bout_table
    from behaviors import _SECOND_ORDER_BINARY_KEYS

    pair_beh_empty = {}
    pfx = 't0_t1'
    for beh_key in _SECOND_ORDER_BINARY_KEYS:
        pair_beh_empty[f'{pfx}/{beh_key}'] = np.zeros(1000, dtype=np.int8)

    try:
        df = build_bout_table({0: {}, 1: {}}, {}, pair_beh_empty,
                              ['A', 'B'], fps=fps, min_bout_frames=3)
        if df.empty:
            _ok("Returned empty DataFrame as expected for all-zero behaviors")
        else:
            _fail(f"Expected empty DataFrame but got {len(df)} rows")
    except Exception as e:
        _fail("Exception on empty input")
        traceback.print_exc()


# ── Edge case: too few bouts to CV ───────────────────────────────────────────

def test_few_bouts(fps):
    _print("\n=== Test 6: too few bouts (fewer than n_splits per class) ===")
    from rf_analysis import run_rf_analysis, build_bout_table
    from behaviors import _SECOND_ORDER_BINARY_KEYS

    n_frames = 500
    pair_beh_few = {}
    pfx = 't0_t1'
    # Only inject 2 bouts of Follow — fewer than n_splits=5
    arr = np.zeros(n_frames, dtype=np.int8)
    arr[10:20] = 1
    arr[30:40] = 1
    pair_beh_few[f'{pfx}/Follow_AtoB'] = arr
    for beh_key in _SECOND_ORDER_BINARY_KEYS:
        if beh_key != 'Follow_AtoB':
            pair_beh_few[f'{pfx}/{beh_key}'] = np.zeros(n_frames, dtype=np.int8)

    ta = {0: {'nose_speed': np.abs(np.random.randn(n_frames))},
          1: {'nose_speed': np.abs(np.random.randn(n_frames))}}
    pa = {f'{pfx}/inter_animal_dist': np.abs(np.random.randn(n_frames)) * 50}

    df = build_bout_table(ta, pa, pair_beh_few, ['A', 'B'], fps=fps)

    if df.empty:
        _ok("Empty bout table returned (as expected — only 2 bouts)")
        return

    try:
        results = run_rf_analysis(df, n_splits=5)
        _fail("Expected ValueError (all classes < n_splits) but got none")
    except ValueError as ve:
        _ok(f"Correctly raised ValueError: {ve}")
    except Exception as e:
        _fail("Unexpected exception type")
        traceback.print_exc()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _print("=" * 60)
    _print("RF Analysis Test Suite")
    _print("=" * 60)

    fps = 25.0
    _print("\n--- Building synthetic data ---")
    track_arrays, pair_arrays, pair_beh = make_synthetic_data(
        n_frames=3000, n_tracks=2, fps=fps)
    _ok(f"track_arrays: {len(track_arrays)} tracks, "
        f"{len(track_arrays[0])} features each")
    _ok(f"pair_arrays: {len(pair_arrays)} pair-features")
    _ok(f"pair_beh: {len(pair_beh)} behavior arrays")

    # Run tests
    bout_df = test_build_bout_table(track_arrays, pair_arrays, pair_beh, fps)
    if bout_df is not None:
        results = test_run_rf_analysis(bout_df, fps)
        if results is not None:
            test_write_rf_outputs(results)

    test_full_pipeline(track_arrays, pair_arrays, pair_beh, fps)
    test_empty_behavior(fps)
    test_few_bouts(fps)

    _print("\n" + "=" * 60)
    _print("Done.")
    _print("=" * 60)
