"""
rf_analysis.py — Random Forest validation of rule-based 2nd-order behavior bouts
=================================================================================
Trains a random forest on kinematic features during each detected behavior bout,
using the rule-based label as the target.  Outputs feature importance rankings
and a confusion matrix to validate that behavior definitions are internally
consistent and driven by the expected kinematic features.

No Qt dependency — pure functions only.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from behaviors import _find_bouts, _SECOND_ORDER_BINARY_KEYS
from binned_export import _classify_feat, _agg_025


# ---------------------------------------------------------------------------
# 1. Build bout-level feature table
# ---------------------------------------------------------------------------

def build_bout_table(
    track_arrays: dict,
    pair_arrays: dict,
    pair_beh: dict,
    track_names: list[str],
    fps: float,
    min_bout_frames: int = 3,
) -> pd.DataFrame:
    """
    Extract feature-aggregated rows for every second-order behavior bout.

    Parameters
    ----------
    track_arrays : dict  — {track_idx: {feat_name: (n_frames,)}} per-animal features
    pair_arrays  : dict  — {'pfx/feat_name': (n_frames,)} per-pair features
    pair_beh     : dict  — {'tA_tB/BehName': (n_frames,) int8} behavior arrays
    track_names  : list  — human-readable track names
    fps          : float — video frame rate
    min_bout_frames : int — discard bouts shorter than this

    Returns
    -------
    pd.DataFrame with one row per bout, columns = aggregated features + metadata.
    """
    rows: list[dict] = []
    bout_id = 0

    # Collect all pair prefixes present in pair_beh
    pair_prefixes: set[str] = set()
    for key in pair_beh:
        pfx = key.rsplit("/", 1)[0]
        pair_prefixes.add(pfx)

    for pfx in sorted(pair_prefixes):
        # Parse track indices from prefix like "t0_t1"
        parts = pfx.split("_")
        try:
            tA = int(parts[0][1:])
            tB = int(parts[1][1:])
        except (ValueError, IndexError):
            continue

        for beh_key in _SECOND_ORDER_BINARY_KEYS:
            full_key = f"{pfx}/{beh_key}"
            if full_key not in pair_beh:
                continue
            arr = pair_beh[full_key]
            # Skip non-binary arrays (e.g. speed floats)
            if arr.dtype not in (np.int8, np.int16, np.int32, np.int64,
                                 np.uint8, np.bool_):
                continue

            bouts = _find_bouts(arr)
            for start, end_incl in bouts:
                length = end_incl - start + 1
                if length < min_bout_frames:
                    continue

                sl = slice(start, end_incl + 1)
                row: dict = {}

                # For directional behaviors (_AtoB/_BtoA), normalise so
                # the actor is always "tA_" and the target is always "tB_".
                # _AtoB: actor=tA, target=tB (natural order)
                # _BtoA: actor=tB, target=tA (swap)
                is_btoa = beh_key.endswith("_BtoA")
                if is_btoa:
                    actor_idx, target_idx = tB, tA
                else:
                    actor_idx, target_idx = tA, tB

                # --- Track features (actor_, target_) ---
                for t_idx, t_prefix in ((actor_idx, "actor_"),
                                        (target_idx, "target_")):
                    if t_idx not in track_arrays:
                        continue
                    for feat_name, feat_arr in track_arrays[t_idx].items():
                        cat = _classify_feat(feat_name)
                        if cat == "categorical":
                            continue
                        vals = feat_arr[sl].astype(np.float64)
                        agg = _agg_025(vals, cat)
                        for suffix, val in agg.items():
                            row[f"{t_prefix}{feat_name}{suffix}"] = val

                # --- Pair features (unprefixed) ---
                for pkey, parr in pair_arrays.items():
                    if not pkey.startswith(f"{pfx}/"):
                        continue
                    feat_name = pkey.split("/", 1)[1]
                    cat = _classify_feat(feat_name)
                    if cat == "categorical":
                        continue
                    vals = parr[sl].astype(np.float64)
                    agg = _agg_025(vals, cat)
                    for suffix, val in agg.items():
                        row[f"{feat_name}{suffix}"] = val

                # --- Metadata ---
                # Merge directional variants into one class
                # (Follow_AtoB + Follow_BtoA → Follow)
                if beh_key.endswith("_AtoB"):
                    canonical_beh = beh_key[:-5]
                elif beh_key.endswith("_BtoA"):
                    canonical_beh = beh_key[:-5]
                else:
                    canonical_beh = beh_key

                row["bout_id"] = bout_id
                row["pair"] = pfx
                row["behavior"] = canonical_beh
                row["start_frame"] = int(start)
                row["end_frame"] = int(end_incl)
                row["duration_s"] = round(length / fps, 4)
                row["group"] = pfx  # for GroupKFold splitting

                rows.append(row)
                bout_id += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metadata columns to exclude from features
# ---------------------------------------------------------------------------

_META_COLS = {"bout_id", "pair", "behavior", "start_frame", "end_frame",
              "duration_s", "group"}


# ---------------------------------------------------------------------------
# 2. Run random forest analysis
# ---------------------------------------------------------------------------

def run_rf_analysis(
    bout_df: pd.DataFrame,
    n_estimators: int = 100,
    n_splits: int = 5,
    n_perm_repeats: int = 5,
    top_n_features: int = 15,
    status_cb=None,
) -> dict:
    """
    Train a random forest classifier on the bout table and return diagnostics.

    Parameters
    ----------
    bout_df        : DataFrame from build_bout_table()
    n_estimators   : number of trees
    n_splits       : CV folds
    n_perm_repeats : permutation importance repeats
    top_n_features : top N features to report per class
    status_cb      : optional callable(str) for progress messages

    Returns
    -------
    dict with keys: bout_df, report_str, confusion, labels,
                    importance_global, importance_per_class
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.metrics import (confusion_matrix, classification_report)
    from sklearn.inspection import permutation_importance

    def _status(msg):
        if status_cb:
            status_cb(msg)

    if bout_df.empty:
        raise ValueError("No bouts found — cannot run RF analysis.")

    # Prepare X, y, groups
    feat_cols = [c for c in bout_df.columns if c not in _META_COLS]
    X = bout_df[feat_cols].fillna(0).values.astype(np.float64)
    y = np.asarray(bout_df["behavior"].tolist(), dtype=object)
    groups = np.asarray(bout_df["group"].tolist(), dtype=object)

    # Drop classes with fewer samples than n_splits
    labels, counts = np.unique(y, return_counts=True)
    valid = set(labels[counts >= n_splits])
    if not valid:
        raise ValueError(
            f"All behavior classes have fewer than {n_splits} bouts. "
            "Increase data or reduce n_splits.")

    mask = np.isin(y, list(valid))
    X, y, groups = X[mask], y[mask], groups[mask]
    bout_df_filtered = bout_df.loc[mask].copy()

    labels_sorted = sorted(np.unique(y))

    # Choose CV strategy
    unique_groups = np.unique(groups)
    if len(unique_groups) >= n_splits:
        _status("RF: Using GroupKFold cross-validation…")
        cv = GroupKFold(n_splits=n_splits)
        split_args = (X, y, groups)
    else:
        _, filtered_counts = np.unique(y, return_counts=True)
        actual_splits = min(n_splits, int(filtered_counts.min()))
        _status("RF: Using StratifiedKFold cross-validation…")
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                             random_state=42)
        split_args = (X, y)

    # Out-of-fold predictions
    oof_preds = np.empty(len(y), dtype=object)
    for fold, (train_idx, test_idx) in enumerate(cv.split(*split_args)):
        _status(f"RF: Training fold {fold + 1}/{cv.get_n_splits(*split_args)}…")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof_preds[test_idx] = clf.predict(X[test_idx])

    # Confusion matrix + classification report
    cm = confusion_matrix(y, oof_preds, labels=labels_sorted)
    report_str = classification_report(y, oof_preds, labels=labels_sorted,
                                       zero_division=0)

    # Final model on all data
    _status("RF: Fitting final model on all data…")
    final_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    # Global feature importance (permutation importance)
    _status("RF: Computing permutation importance…")
    perm = permutation_importance(
        final_clf, X, y,
        n_repeats=n_perm_repeats,
        random_state=42,
        n_jobs=-1,
    )
    imp_global = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Per-class importance (top N) — derived from the final model's per-tree
    # predictions rather than training separate OVR models (much faster).
    # For each class, weight each tree's Gini importances by how well that
    # tree predicts the class (proportion of correct predictions for that class).
    _status("RF: Computing per-class feature importance…")
    imp_per_class: dict[str, pd.DataFrame] = {}
    n_features = X.shape[1]
    for cls_label in labels_sorted:
        cls_mask = y == cls_label
        # Accumulate weighted Gini importances across trees
        weighted_imp = np.zeros(n_features)
        total_weight = 0.0
        for tree in final_clf.estimators_:
            preds = tree.predict(X[cls_mask])
            weight = np.mean(preds == cls_label)  # accuracy on this class
            weighted_imp += weight * tree.feature_importances_
            total_weight += weight
        if total_weight > 0:
            weighted_imp /= total_weight
        cls_imp = pd.DataFrame({
            "feature": feat_cols,
            "importance": weighted_imp,
        }).sort_values("importance", ascending=False).head(top_n_features)
        imp_per_class[cls_label] = cls_imp.reset_index(drop=True)

    # Attach predictions to bout_df
    bout_df_filtered = bout_df_filtered.copy()
    bout_df_filtered["rf_prediction"] = oof_preds

    return {
        "bout_df": bout_df_filtered,
        "report_str": report_str,
        "confusion": cm,
        "labels": labels_sorted,
        "importance_global": imp_global,
        "importance_per_class": imp_per_class,
    }


# ---------------------------------------------------------------------------
# 3. Write outputs
# ---------------------------------------------------------------------------

def write_rf_outputs(
    results: dict,
    base_path: str,
    write_plots: bool = True,
) -> list[str]:
    """
    Write RF analysis results to CSV files and optionally a PDF.

    Parameters
    ----------
    results    : dict from run_rf_analysis()
    base_path  : file path stem (no extension)
    write_plots: if True, generate a PDF with confusion matrix + importance chart

    Returns
    -------
    list of file paths created
    """
    paths: list[str] = []
    base = Path(base_path)

    # Bout table with predictions
    bout_path = str(base.parent / f"{base.name}_rf_bouts.csv")
    results["bout_df"].to_csv(bout_path, index=False)
    paths.append(bout_path)

    # Classification report
    report_path = str(base.parent / f"{base.name}_rf_report.csv")
    _report_to_csv(results["report_str"], report_path)
    paths.append(report_path)

    # Global feature importance
    imp_path = str(base.parent / f"{base.name}_rf_importance.csv")
    results["importance_global"].to_csv(imp_path, index=False)
    paths.append(imp_path)

    # PDF plots
    if write_plots:
        pdf_path = str(base.parent / f"{base.name}_rf_analysis.pdf")
        _write_pdf(results, pdf_path)
        paths.append(pdf_path)

    return paths


def _report_to_csv(report_str: str, path: str):
    """Parse sklearn classification_report string into a CSV."""
    lines = report_str.strip().split("\n")
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            # Class row: label  precision  recall  f1  support
            # Handle multi-word labels (e.g. "Follow_AtoB")
            try:
                support = int(float(parts[-1]))
                f1 = float(parts[-2])
                recall = float(parts[-3])
                prec = float(parts[-4])
                label = " ".join(parts[:-4])
                rows.append({
                    "class": label,
                    "precision": prec,
                    "recall": recall,
                    "f1_score": f1,
                    "support": support,
                })
            except (ValueError, IndexError):
                continue
        elif len(parts) == 4:
            # accuracy row
            try:
                rows.append({
                    "class": "accuracy",
                    "precision": "",
                    "recall": "",
                    "f1_score": float(parts[-2]),
                    "support": int(float(parts[-1])),
                })
            except (ValueError, IndexError):
                continue
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pdf(results: dict, pdf_path: str):
    """Generate a two-panel PDF: confusion matrix + top feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cm = results["confusion"]
    labels = results["labels"]
    imp = results["importance_global"]

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Confusion matrix ---
        n = len(labels)
        fig_w = max(6, n * 0.8 + 2)
        fig_h = max(5, n * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Normalise rows to percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        im = ax.imshow(cm_pct, cmap="Blues", aspect="auto",
                       vmin=0, vmax=100)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (% of true class)")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = cm_pct[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%\n({cm[i,j]})",
                        ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Top feature importances (global) ---
        top = imp.head(20).iloc[::-1]  # reverse for horizontal bar
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35 + 1)))
        ax.barh(range(len(top)), top["importance_mean"].values,
                xerr=top["importance_std"].values, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=7)
        ax.set_xlabel("Permutation Importance")
        ax.set_title("Top 20 Features (Global Permutation Importance)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3+: Per-class top features ---
        for cls_label, cls_imp in results["importance_per_class"].items():
            top_cls = cls_imp.head(top_n := min(15, len(cls_imp))).iloc[::-1]
            fig, ax = plt.subplots(
                figsize=(7, max(3, len(top_cls) * 0.35 + 1)))
            ax.barh(range(len(top_cls)), top_cls["importance"].values,
                    color="#dd8452", edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(top_cls)))
            ax.set_yticklabels(top_cls["feature"].values, fontsize=7)
            ax.set_xlabel("Gini Importance (one-vs-rest)")
            ax.set_title(f"Top Features — {cls_label}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Convenience pipeline
# ---------------------------------------------------------------------------

def run_full_rf_pipeline(
    track_arrays: dict,
    pair_arrays: dict,
    pair_beh: dict,
    track_names: list[str],
    fps: float,
    base_path: str,
    min_bout_frames: int = 3,
    n_estimators: int = 100,
    n_splits: int = 5,
    write_plots: bool = True,
    status_cb=None,
) -> list[str]:
    """
    End-to-end RF analysis: build bouts → train → write outputs.

    Returns list of created file paths.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    _status("RF: Building bout feature table…")
    bout_df = build_bout_table(
        track_arrays, pair_arrays, pair_beh,
        track_names, fps,
        min_bout_frames=min_bout_frames,
    )
    if bout_df.empty:
        _status("RF: No qualifying bouts found — skipping analysis.")
        return []

    _status(f"RF: {len(bout_df)} bouts across "
            f"{bout_df['behavior'].nunique()} behavior classes.")

    results = run_rf_analysis(
        bout_df,
        n_estimators=n_estimators,
        n_splits=n_splits,
        status_cb=status_cb,
    )

    _status("RF: Writing output files…")
    paths = write_rf_outputs(results, base_path, write_plots=write_plots)

    _status(f"RF: Done — {len(paths)} files created.")
    return paths
