"""
Headless tests for ExportOptionsDialog and ProcessingOutputDialog.
Run: python test_export_dialog.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from run_popup import ExportOptionsDialog, ProcessingOutputDialog

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

# All keys that can appear in ExportOptionsDialog
_ALL_KEYS = (
    [k for k, _ in ExportOptionsDialog._MAIN_SHEETS] +
    [k for k, _ in ExportOptionsDialog._KEY_METRICS_SHEET] +
    [k for k, _ in ExportOptionsDialog._BINNED_SHEETS] +
    [k for k, _ in ExportOptionsDialog._GRAPH_PDFS] +
    [k for k, _ in ExportOptionsDialog._RF_ANALYSIS]
)
_ALL_OPTS_TRUE = {k: True for k in _ALL_KEYS}

# ---------------------------------------------------------------------------
# 1. Basic init — no crash
# ---------------------------------------------------------------------------
print("\n--- 1. Init without crash ---")
try:
    dlg = ExportOptionsDialog(output_opts=_ALL_OPTS_TRUE,
                              default_dir="/tmp", default_name="test")
    check("ExportOptionsDialog(all opts) instantiates", True)
except Exception as e:
    check("ExportOptionsDialog(all opts) instantiates", False, str(e))

try:
    dlg_empty = ExportOptionsDialog()   # no opts → empty dialog
    check("ExportOptionsDialog() (no opts) instantiates", True)
except Exception as e:
    check("ExportOptionsDialog() (no opts) instantiates", False, str(e))

try:
    proc_dlg = ProcessingOutputDialog(n_tracks=2)
    check("ProcessingOutputDialog(n_tracks=2) instantiates", True)
except Exception as e:
    check("ProcessingOutputDialog(n_tracks=2) instantiates", False, str(e))

# ---------------------------------------------------------------------------
# 2. _updating guard is initialised before widget signals can fire
# ---------------------------------------------------------------------------
print("\n--- 2. _updating guard ---")
dlg = ExportOptionsDialog(output_opts=_ALL_OPTS_TRUE)
check("_updating is False after init", dlg._updating is False)
check("_updating is bool", isinstance(dlg._updating, bool))

# ---------------------------------------------------------------------------
# 3. Default state — all provided opts show up as checked
# ---------------------------------------------------------------------------
print("\n--- 3. Default checkbox state ---")
dlg = ExportOptionsDialog(output_opts=_ALL_OPTS_TRUE)
opts = dlg.options()
check("options() has all expected keys",
      set(opts.keys()) == set(_ALL_KEYS),
      f"missing={set(_ALL_KEYS)-set(opts.keys())} extra={set(opts.keys())-set(_ALL_KEYS)}")
check("all options True by default", all(opts.values()),
      f"False keys: {[k for k,v in opts.items() if not v]}")
check("master checkbox is Checked", dlg._master.checkState() == Qt.Checked)

# ---------------------------------------------------------------------------
# 4. output_opts gating — only True items appear as checkboxes
# ---------------------------------------------------------------------------
print("\n--- 4. output_opts gating ---")
only_tracking = {"main_tracking_data": True, "main_session_info": True}
dlg_gated = ExportOptionsDialog(output_opts=only_tracking)
check("only 2 checkboxes when 2 opts provided",
      len(dlg_gated._checks) == 2,
      f"got {len(dlg_gated._checks)} checks: {list(dlg_gated._checks)}")
check("main_tracking_data present", "main_tracking_data" in dlg_gated._checks)
check("main_session_info present", "main_session_info" in dlg_gated._checks)
check("main_zone_summary absent (not in opts)", "main_zone_summary" not in dlg_gated._checks)

# Empty opts → no checkboxes
dlg_none = ExportOptionsDialog(output_opts={})
check("no checkboxes when opts empty", len(dlg_none._checks) == 0)

# ---------------------------------------------------------------------------
# 5. Master checkbox sync — uncheck one item → master goes partial
# ---------------------------------------------------------------------------
print("\n--- 5. Master checkbox sync ---")
dlg = ExportOptionsDialog(output_opts=_ALL_OPTS_TRUE)
dlg._checks["main_tracking_data"].setChecked(False)
check("master -> PartiallyChecked after one uncheck",
      dlg._master.checkState() == Qt.PartiallyChecked)

for cb in dlg._checks.values():
    cb.setChecked(False)
check("master -> Unchecked after all unchecked",
      dlg._master.checkState() == Qt.Unchecked)

for cb in dlg._checks.values():
    cb.setChecked(True)
check("master -> Checked after all re-checked",
      dlg._master.checkState() == Qt.Checked)

# ---------------------------------------------------------------------------
# 6. Group "All" checkbox drives children
# ---------------------------------------------------------------------------
print("\n--- 6. Group All checkbox drives children ---")
dlg = ExportOptionsDialog(output_opts=_ALL_OPTS_TRUE)
grp_all_main = dlg._group_all[0]
main_keys = [k for k, _ in ExportOptionsDialog._MAIN_SHEETS]

grp_all_main.setCheckState(Qt.Unchecked)
check("group All=Unchecked -> all children unchecked",
      all(not dlg._checks[k].isChecked() for k in main_keys))

grp_all_main.setCheckState(Qt.Checked)
check("group All=Checked -> all children checked",
      all(dlg._checks[k].isChecked() for k in main_keys))

# ---------------------------------------------------------------------------
# 7. _updating always resets to False
# ---------------------------------------------------------------------------
print("\n--- 7. _updating always resets ---")
check("_updating is False after sync ops", dlg._updating is False)

# ---------------------------------------------------------------------------
# 8. ProcessingOutputDialog — defaults with current key names
# ---------------------------------------------------------------------------
print("\n--- 8. ProcessingOutputDialog defaults ---")
pdlg = ProcessingOutputDialog(n_tracks=2)
opts = pdlg.options()
# All output keys come from the same sheet lists as ExportOptionsDialog
all_proc_keys = set(_ALL_KEYS) | {"live_zones", "live_proximity"}
check("options() has all expected keys",
      set(opts.keys()) == all_proc_keys,
      f"missing={all_proc_keys - set(opts.keys())} extra={set(opts.keys()) - all_proc_keys}")
check("main_tracking_data key present", "main_tracking_data" in opts)
check("live_zones key present", "live_zones" in opts)
check("live_proximity key present", "live_proximity" in opts)

# ---------------------------------------------------------------------------
# 9. ProcessingOutputDialog — single-track disables pair-only options
# ---------------------------------------------------------------------------
print("\n--- 9. ProcessingOutputDialog single-track ---")
pdlg_single = ProcessingOutputDialog(n_tracks=1)
opts_single = pdlg_single.options()
# Pair-only keys should be False (disabled) for single track
pair_only_keys = [k for k, _ in ProcessingOutputDialog._MAIN_SHEETS
                  if "[pair]" in dict(ProcessingOutputDialog._MAIN_SHEETS).get(k, "")
                  or "[pair]" in next((l for ky, l in ProcessingOutputDialog._MAIN_SHEETS if ky == k), "")]
# Verify the pair checkbox is disabled
pair_cb = pdlg_single._checks.get("main_2nd_order_behaviors")
if pair_cb is not None:
    check("main_2nd_order_behaviors disabled for 1 track", not pair_cb.isEnabled())
else:
    check("main_2nd_order_behaviors key exists", False, "key not in _checks")

check("proc_pair_beh-gated key is False in opts",
      opts_single.get("main_2nd_order_behaviors") is False)

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
