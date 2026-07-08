import sys
import logging

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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mosiac.log"),
    ],
)


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
