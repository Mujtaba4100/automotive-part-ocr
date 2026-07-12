"""
main_window.py
--------------
The application's single main window.  Contains only GUI construction,
signal wiring, and state management.  No OCR or image-processing logic lives
here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets import LogWidget, StatCard, make_button, make_progress_bar
from app.services.ocr_service import OCRService
from app.utils.constants import APP_NAME, APP_VERSION
from app.utils.logger import get_logger
from app.workers.folder_worker import FolderWorker

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────── #
# Stylesheet                                                           #
# ──────────────────────────────────────────────────────────────────── #

_STYLESHEET = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #1a1d27;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ── Group boxes ── */
QGroupBox {
    border: 1px solid #2d3348;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
    color: #94a3b8;
    font-size: 11px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 12px;
}

/* ── Folder path label ── */
#FolderLabel {
    background-color: #242738;
    border: 1px solid #2d3348;
    border-radius: 6px;
    padding: 6px 10px;
    color: #64748b;
    font-size: 12px;
}

/* ── Stat cards ── */
#StatCard {
    background-color: #242738;
    border: 1px solid #2d3348;
    border-radius: 8px;
}
#StatCardTitle {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}
#StatCardValue {
    color: #f1f5f9;
    font-size: 28px;
    font-weight: 700;
}

/* ── Progress bar ── */
#MainProgressBar {
    background-color: #242738;
    border: 1px solid #2d3348;
    border-radius: 5px;
    color: #f1f5f9;
    font-size: 11px;
    font-weight: 600;
}
#MainProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6
    );
    border-radius: 4px;
}

/* ── Log widget ── */
#LogWidget {
    background-color: #0f1117;
    border: 1px solid #2d3348;
    border-radius: 6px;
    color: #94a3b8;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 12px;
    selection-background-color: #3730a3;
}

/* ── Buttons ── */
QPushButton {
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
    font-size: 13px;
    border: none;
}
QPushButton:disabled {
    background-color: #1e2133;
    color: #475569;
}
#BtnSelectFolder {
    background-color: #334155;
    color: #e2e8f0;
}
#BtnSelectFolder:hover { background-color: #3d4f66; }
#BtnSelectFolder:pressed { background-color: #253344; }

#BtnStart {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6
    );
    color: #ffffff;
}
#BtnStart:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #818cf8, stop:1 #a78bfa
    );
}
#BtnStart:pressed {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f46e5, stop:1 #7c3aed
    );
}

#BtnCancel {
    background-color: #7f1d1d;
    color: #fecaca;
}
#BtnCancel:hover { background-color: #991b1b; }
#BtnCancel:pressed { background-color: #6b1212; }

/* ── Status bar ── */
QStatusBar {
    background-color: #141722;
    color: #475569;
    font-size: 11px;
    border-top: 1px solid #2d3348;
}
"""


# ──────────────────────────────────────────────────────────────────── #
# Main Window                                                          #
# ──────────────────────────────────────────────────────────────────── #

class MainWindow(QMainWindow):
    """Application main window.

    Initialises the :class:`~app.services.ocr_service.OCRService` once and
    reuses it across multiple processing runs.

    Args:
        parent: Optional parent widget (``None`` for a top-level window).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_folder: Path | None = None
        self._worker: FolderWorker | None = None
        self._ocr_service: OCRService | None = None

        self._build_ui()
        self._apply_stylesheet()
        self._update_button_states()

        # Warm up OCR in background so first run isn't slow
        self._init_ocr_service()

    # ---------------------------------------------------------------- #
    # UI construction                                                   #
    # ---------------------------------------------------------------- #

    def _build_ui(self) -> None:
        """Construct all widgets and layouts."""
        self.setWindowTitle(f"{APP_NAME}  ·  {APP_VERSION}")
        self.setMinimumSize(820, 640)
        self.resize(960, 700)

        # ── Central widget ──────────────────────────────────────── #
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 12)
        root_layout.setSpacing(14)

        # ── Folder selection group ───────────────────────────────── #
        folder_group = QGroupBox("Source Folder", central)
        folder_layout = QHBoxLayout(folder_group)
        folder_layout.setSpacing(8)

        self._folder_label = QLabel("No folder selected", folder_group)
        self._folder_label.setObjectName("FolderLabel")
        self._folder_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._folder_label.setWordWrap(False)
        self._folder_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._btn_select = make_button("📂  Select Folder", "BtnSelectFolder", folder_group)
        self._btn_select.clicked.connect(self._on_select_folder)

        folder_layout.addWidget(self._folder_label)
        folder_layout.addWidget(self._btn_select)
        root_layout.addWidget(folder_group)

        # ── Statistics row ──────────────────────────────────────── #
        stats_group = QGroupBox("Statistics", central)
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setSpacing(12)

        self._card_total = StatCard("Total Images", stats_group)
        self._card_processed = StatCard("Processed", stats_group)
        self._card_detected = StatCard("Detected", stats_group)

        for card in (self._card_total, self._card_processed, self._card_detected):
            stats_layout.addWidget(card)

        root_layout.addWidget(stats_group)

        # ── Progress bar ─────────────────────────────────────────── #
        progress_group = QGroupBox("Progress", central)
        progress_layout = QVBoxLayout(progress_group)
        self._progress_bar = make_progress_bar(progress_group)
        progress_layout.addWidget(self._progress_bar)
        root_layout.addWidget(progress_group)

        # ── Log ──────────────────────────────────────────────────── #
        log_group = QGroupBox("Processing Log", central)
        log_layout = QVBoxLayout(log_group)
        self._log_widget = LogWidget(log_group)
        log_layout.addWidget(self._log_widget)
        root_layout.addWidget(log_group, stretch=1)

        # ── Action buttons ───────────────────────────────────────── #
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_start = make_button("▶  Start", "BtnStart", central, min_width=130)
        self._btn_cancel = make_button("✖  Cancel", "BtnCancel", central, min_width=110)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_cancel)
        root_layout.addLayout(btn_row)

        # ── Status bar ───────────────────────────────────────────── #
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — select a folder to begin.")

    def _apply_stylesheet(self) -> None:
        """Apply the application-wide dark stylesheet."""
        self.setStyleSheet(_STYLESHEET)

    # ---------------------------------------------------------------- #
    # OCR initialisation                                                #
    # ---------------------------------------------------------------- #

    def _init_ocr_service(self) -> None:
        """Create the OCR service.  Runs synchronously on startup; model
        download is fast on subsequent runs (weights are cached locally)."""
        try:
            self._log("Initialising OCR engine …")
            self._ocr_service = OCRService()
            self._log("✅ OCR engine ready.")
            self._status_bar.showMessage("OCR engine ready — select a folder to begin.")
        except Exception as exc:  # noqa: BLE001
            self._log(f"❌ Failed to initialise OCR engine: {exc}")
            QMessageBox.critical(
                self,
                "OCR Initialisation Error",
                f"Could not load the PaddleOCR model.\n\n{exc}",
            )
            self._btn_start.setEnabled(False)

    # ---------------------------------------------------------------- #
    # Slot handlers                                                     #
    # ---------------------------------------------------------------- #

    @Slot()
    def _on_select_folder(self) -> None:
        """Open a folder-picker dialog and store the user's choice."""
        folder_str = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            str(self._selected_folder or Path.home()),
        )
        if not folder_str:
            return

        self._selected_folder = Path(folder_str)
        self._folder_label.setText(folder_str)
        self._folder_label.setStyleSheet("color: #e2e8f0;")
        self._log(f"Folder selected: {folder_str}")
        self._status_bar.showMessage(f"Folder: {folder_str}")
        self._update_button_states()

    @Slot()
    def _on_start(self) -> None:
        """Validate state and launch the background worker."""
        if not self._selected_folder:
            QMessageBox.warning(self, "No Folder", "Please select a source folder first.")
            return

        if self._ocr_service is None:
            QMessageBox.critical(
                self, "OCR Not Ready",
                "The OCR engine failed to initialise.  Please restart the application."
            )
            return

        # Reset UI state
        self._log_widget.clear_log()
        self._progress_bar.setValue(0)
        self._card_total.set_value(0)
        self._card_processed.set_value(0)
        self._card_detected.set_value(0)

        # Build and wire worker
        self._worker = FolderWorker(self._selected_folder, self._ocr_service)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.stats_update.connect(self._on_stats_update)
        self._worker.finished.connect(self._on_finished)

        self._worker.start()
        self._update_button_states()
        self._status_bar.showMessage("Processing …")
        log.info("Worker started for folder: %s", self._selected_folder)

    @Slot()
    def _on_cancel(self) -> None:
        """Request graceful cancellation of the running worker."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._btn_cancel.setEnabled(False)
            self._status_bar.showMessage("Cancelling …")

    # ---------------------------------------------------------------- #
    # Worker signal handlers                                            #
    # ---------------------------------------------------------------- #

    @Slot(int)
    def _on_progress(self, value: int) -> None:
        self._progress_bar.setValue(value)

    @Slot(str)
    def _on_log_message(self, message: str) -> None:
        self._log_widget.append_line(message)

    @Slot(int, int, int)
    def _on_stats_update(self, total: int, processed: int, detected: int) -> None:
        self._card_total.set_value(total)
        self._card_processed.set_value(processed)
        self._card_detected.set_value(detected)

    @Slot(bool)
    def _on_finished(self, success: bool) -> None:
        """Handle worker completion or cancellation."""
        self._worker = None
        self._update_button_states()

        if success:
            self._progress_bar.setValue(100)
            self._status_bar.showMessage("Completed successfully.")
        else:
            self._status_bar.showMessage("Cancelled or failed.")

    # ---------------------------------------------------------------- #
    # Helpers                                                           #
    # ---------------------------------------------------------------- #

    def _log(self, message: str) -> None:
        """Append *message* to the log widget."""
        self._log_widget.append_line(message)

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on the current application state."""
        running = self._worker is not None and self._worker.isRunning()
        self._btn_select.setEnabled(not running)
        self._btn_start.setEnabled(
            not running
            and self._selected_folder is not None
            and self._ocr_service is not None
        )
        self._btn_cancel.setEnabled(running)

    # ---------------------------------------------------------------- #
    # Window close                                                      #
    # ---------------------------------------------------------------- #

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Gracefully stop the worker thread before closing the window."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(5000)  # max 5 s
        super().closeEvent(event)
