"""
Headless tests for ExportOptionsDialog and ProcessingOptionsDialog.
Run: python test_export_dialog.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

# Import the classes under test
from run_popup import ExportOptionsDialog, ProcessingOptionsDialog

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {name}"
    if not condition and detail:
        msg += f"\n         {detail}"
    print(msg)
    _results.append(condition)

# ---------------------------------------------------------------------------
# 1. Basic init — no crash (the _updating bug we fixed)
# ---------------------------------------------------------------------------
print("\n--- 1. Init without crash ---")
try:
    dlg = ExportOptionsDialog(graphs_available=True, default_dir="/tmp", default_name="test")
    check("ExportOptionsDialog instantiates without AttributeError", True)
except AttributeError as e:
    check("ExportOptionsDialog instantiates without AttributeError", False, str(e))

try:
    dlg2 = ExportOptionsDialog(graphs_available=False, default_dir="/tmp", default_name="test")
    check("graphs_available=False doesn't crash", True)
except Exception as e:
    check("graphs_available=False doesn't crash", False, str(e))

try:
    proc_dlg = ProcessingOptionsDialog(n_tracks=2)
    check("ProcessingOptionsDialog instantiates", True)
except Exception as e:
    check("ProcessingOptionsDialog instantiates", False, str(e))

# ---------------------------------------------------------------------------
# 2. _updating guard is initialized before widget signals can fire
# ---------------------------------------------------------------------------
print("\n--- 2. _updating guard ---")
dlg = ExportOptionsDialog()
check("_updating is False after init", dlg._updating is False)
check("_updating is bool", isinstance(dlg._updating, bool))

# ---------------------------------------------------------------------------
# 3. Default state — all checks enabled and checked
# ---------------------------------------------------------------------------
print("\n--- 3. Default checkbox state ---")
dlg = ExportOptionsDialog()
opts = dlg.options()
all_keys = (
    [k for k, _ in ExportOptionsDialog._MAIN_SHEETS] +
    [k for k, _ in ExportOptionsDialog._BINNED_SHEETS] +
    [k for k, _ in ExportOptionsDialog._GRAPH_PDFS] +
    [k for k, _ in ExportOptionsDialog._RF_ANALYSIS]
)
check("options() has all expected keys", set(opts.keys()) == set(all_keys),
      f"missing={set(all_keys)-set(opts.keys())} extra={set(opts.keys())-set(all_keys)}")
check("all options True by default", all(opts.values()),
      f"False keys: {[k for k,v in opts.items() if not v]}")
check("master checkbox is Checked", dlg._master.checkState() == Qt.Checked)

# ---------------------------------------------------------------------------
# 4. proc_opts gating — disabled keys get unchecked and disabled
# ---------------------------------------------------------------------------
print("\n--- 4. proc_opts gating ---")
proc_opts_none = {
    "proc_single_beh": False,
    "proc_pair_beh":   False,
    "proc_features":   False,
    "proc_zones":      False,
    "proc_proximity":  False,
}
dlg_gated = ExportOptionsDialog(proc_opts=proc_opts_none)
gated_opts = dlg_gated.options()

# These should all be disabled (hence unchecked)
should_be_gated = {
    "main_1st_order_behaviors", "main_2nd_order_behaviors",
    "main_behavior_summary", "main_engagement_indices",
    "main_animal_features", "main_pair_features",
    "main_zone_summary",
    "binned_animal_025", "binned_pair_025", "binned_eng_indices_025",
    "binned_animal_1s", "binned_pair_1s",
    "graph_heatmaps", "graph_distance",
    "graph_oncoplot", "graph_sync_oncoplot",
    "graph_oncoplot_clean", "graph_sync_oncoplot_clean", "graph_dist_features",
}
for key in should_be_gated:
    cb = dlg_gated._checks[key]
    check(f"  gated: {key} disabled", not cb.isEnabled())
    check(f"  gated: {key} unchecked", not cb.isChecked())

# These should NOT be gated (no proc_key mapped)
ungated = {"main_tracking_data", "main_session_info", "main_key_metrics",
           "graph_cascade", "rf_analysis", "rf_analysis_plots"}
for key in ungated:
    cb = dlg_gated._checks.get(key)
    if cb:
        check(f"  ungated: {key} still enabled", cb.isEnabled())

# ---------------------------------------------------------------------------
# 5. proc_opts partial — only pair_beh=False
# ---------------------------------------------------------------------------
print("\n--- 5. Partial gating (proc_pair_beh=False) ---")
dlg_partial = ExportOptionsDialog(proc_opts={"proc_pair_beh": False})
pair_gated = ["main_2nd_order_behaviors", "main_engagement_indices",
              "binned_eng_indices_025", "graph_sync_oncoplot", "graph_sync_oncoplot_clean"]
pair_ungated = ["main_1st_order_behaviors", "main_animal_features",
                "main_tracking_data", "graph_cascade"]
for key in pair_gated:
    cb = dlg_partial._checks[key]
    check(f"  pair_beh gated: {key}", not cb.isEnabled() and not cb.isChecked())
for key in pair_ungated:
    cb = dlg_partial._checks[key]
    check(f"  not gated: {key} still checked", cb.isChecked())

# ---------------------------------------------------------------------------
# 6. Master checkbox sync — uncheck one item -> master goes partial
# ---------------------------------------------------------------------------
print("\n--- 6. Master checkbox sync ---")
dlg = ExportOptionsDialog()
# Uncheck one item
dlg._checks["main_tracking_data"].setChecked(False)
check("master -> PartiallyChecked after one uncheck",
      dlg._master.checkState() == Qt.PartiallyChecked)

# Uncheck all items one by one
for cb in dlg._checks.values():
    cb.setChecked(False)
check("master -> Unchecked after all unchecked",
      dlg._master.checkState() == Qt.Unchecked)

# Re-check all
for cb in dlg._checks.values():
    cb.setChecked(True)
check("master -> Checked after all re-checked",
      dlg._master.checkState() == Qt.Checked)

# ---------------------------------------------------------------------------
# 7. Group "All" checkbox drives children
# ---------------------------------------------------------------------------
print("\n--- 7. Group All checkbox drives children ---")
dlg = ExportOptionsDialog()
# Find the group_all for MAIN_SHEETS (first group = index 0)
grp_all_main = dlg._group_all[0]
main_keys = [k for k, _ in ExportOptionsDialog._MAIN_SHEETS]

grp_all_main.setCheckState(Qt.Unchecked)
check("group All=Unchecked -> all children unchecked",
      all(not dlg._checks[k].isChecked() for k in main_keys))

grp_all_main.setCheckState(Qt.Checked)
check("group All=Checked -> all children checked",
      all(dlg._checks[k].isChecked() for k in main_keys))

# ---------------------------------------------------------------------------
# 8. _updating left clean after all operations
# ---------------------------------------------------------------------------
print("\n--- 8. _updating always resets to False ---")
check("_updating is False after sync ops", dlg._updating is False)

# ---------------------------------------------------------------------------
# 9. ProcessingOptionsDialog defaults
# ---------------------------------------------------------------------------
print("\n--- 9. ProcessingOptionsDialog defaults ---")
pdlg = ProcessingOptionsDialog(n_tracks=2)
opts = pdlg.options()
expected_keys = {"proc_kinematics", "proc_single_beh", "proc_pair_beh",
                 "proc_features", "proc_zones", "proc_proximity"}
check("has all 5 proc option keys", set(opts.keys()) == expected_keys,
      f"got: {set(opts.keys())}")
check("all options True by default", all(opts.values()),
      f"False: {[k for k,v in opts.items() if not v]}")

# Single track — pair options should be disabled
pdlg_single = ProcessingOptionsDialog(n_tracks=1)
opts_single = pdlg_single.options()
check("proc_pair_beh accessible (key exists)", "proc_pair_beh" in opts_single)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(_results)
passed = sum(_results)
failed = total - passed
print(f"\n{'='*50}")
print(f"  {passed}/{total} passed  |  {failed} failed")
print('='*50)
sys.exit(0 if failed == 0 else 1)
