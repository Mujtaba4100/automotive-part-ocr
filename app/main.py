"""
main.py
-------
Application entry point.

Usage::

    python main.py
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.gui.main_window import MainWindow
from app.utils.constants import APP_NAME
from app.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    """Bootstrap the Qt application and show the main window."""
    # High-DPI support (Qt 6 has it on by default, but be explicit)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")   # consistent cross-platform base style

    window = MainWindow()
    window.show()

    log.info("Application started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
