import sys
import logging

# Configure logging FIRST — before any non-stdlib import.
# h5py calls logging.basicConfig(level=DEBUG) during its own import, which
# makes any later basicConfig() call a no-op (Python skips it when the root
# logger already has handlers).  Configuring here, with force=True, ensures
# our WARNING level wins regardless of import order.
_handlers = [logging.FileHandler("mosiac.log")]
# sys.stderr is None in windowed PyInstaller builds — only add StreamHandler when it exists
if sys.stderr is not None:
    _handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=_handlers,
    force=True,   # override any handler already added by h5py/matplotlib
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

# Windows: set AppUserModelID BEFORE Qt is imported so the taskbar always
# groups under our icon instead of the generic Python/PyInstaller one.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "wanglabs.MOSIAC.2.0"
        )
    except Exception:
        pass

from PySide6.QtWidgets import QApplication

from main_window import MainWindow, _make_mosiac_icon


def main():
    app = QApplication(sys.argv)
    icon = _make_mosiac_icon()
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)   # explicit set on the window for reliable taskbar icon
    window.show()
    window._center_on_screen()
    window.raise_()
    window.activateWindow()
    window.setup_debug_routing()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
