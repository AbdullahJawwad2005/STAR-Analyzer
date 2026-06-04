import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window._center_on_screen()
    window.raise_()
    window.activateWindow()
    print("STAR Analyzer window opened (check taskbar / Alt+Tab).", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
