import sys
import logging

from PySide6.QtWidgets import QApplication

from main_window import MainWindow, _make_star_icon

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("star_analyzer.log"),
    ],
)


def main():
    # Windows: register our own App User Model ID so the taskbar shows our
    # icon instead of the generic Python icon.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "wanglabs.STARAnalyzer.1.0"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setWindowIcon(_make_star_icon())
    window = MainWindow()
    window.show()
    window._center_on_screen()
    window.raise_()
    window.activateWindow()
    window.setup_debug_routing()
    print("STAR Analyzer window opened (check taskbar / Alt+Tab).", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
