"""
main_window.py
--------------
The application's single main window. Contains GUI construction, styling,
signals, and state management. No OCR or image-processing logic lives here.
"""

from __future__ import annotations

import os
import time
import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QIcon, QTextDocument
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
    QDialog,
    QPushButton,
    QStyle,
)

from app.gui.widgets import LogWidget, StatCard, make_button, make_progress_bar
from app.services.ocr_service import OCRService
from app.utils.constants import APP_NAME, APP_VERSION
from app.utils.logger import get_logger
from app.workers.folder_worker import FolderWorker

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────── #
# CSS Stylesheet                                                       #
# ──────────────────────────────────────────────────────────────────── #

_STYLESHEET = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #0a0b10;
    color: #f8fafc;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

/* ── Group boxes ── */
QGroupBox {
    border: 1px solid #1e2230;
    border-radius: 8px;
    margin-top: 16px;
    padding: 14px 14px 14px 14px;
    font-weight: 700;
    color: #64748b;
    font-size: 11px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
}

/* ── Folder Path Box ── */
#FolderLabel {
    background-color: #12131a;
    border: 1px solid #1e2230;
    border-radius: 6px;
    padding: 8px 12px;
    color: #94a3b8;
    font-size: 13px;
}

/* ── Stat cards ── */
#StatCard {
    background-color: #12131a;
    border: 1px solid #1e2230;
    border-radius: 8px;
}
#StatCardTitle {
    color: #475569;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
}
#StatCardValue {
    color: #cbd5e1;
    font-size: 24px;
    font-weight: 800;
}

/* ── Progress bar ── */
#MainProgressBar {
    background-color: #12131a;
    border: 1px solid #1e2230;
    border-radius: 6px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
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
    background-color: #050508;
    border: 1px solid #1e2230;
    border-radius: 6px;
    padding: 6px;
    color: #cbd5e1;
    selection-background-color: #4f46e5;
}

/* ── Scrollbars ── */
QScrollBar:vertical {
    border: none;
    background: #0a0b10;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #1e2230;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #2d3248;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ── Buttons ── */
QPushButton {
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #1e2230;
    background-color: #12131a;
    color: #cbd5e1;
}
QPushButton:hover {
    background-color: #1b1d28;
    border-color: #2d3248;
    color: #f8fafc;
}
QPushButton:pressed {
    background-color: #0c0d14;
}
QPushButton:disabled {
    background-color: #07080c;
    border-color: #101117;
    color: #334155;
}

#BtnStart {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6
    );
    color: #ffffff;
    border: none;
}
#BtnStart:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f46e5, stop:1 #7c3aed
    );
}
#BtnStart:pressed {
    background: #4338ca;
}

#BtnCancel {
    background-color: #991b1b;
    color: #fee2e2;
    border: none;
}
#BtnCancel:hover {
    background-color: #b91c1c;
}
#BtnCancel:pressed {
    background-color: #7f1d1d;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #050608;
    color: #475569;
    font-size: 11px;
    border-top: 1px solid #12131a;
}
"""


# ──────────────────────────────────────────────────────────────────── #
# Completion Custom Dialog                                             #
# ──────────────────────────────────────────────────────────────────── #

class CompletionDialog(QDialog):
    """Custom pop-up dialog showing execution stats summaries."""

    def __init__(self, parent: QWidget, processed: int, detected: int, elapsed_str: str, output_folder: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing Complete")
        self.setMinimumWidth(360)
        self.setStyleSheet(parent.styleSheet())
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Processing Complete", self)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #10b981;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        info_label = QLabel(
            f"<div style='font-size: 13px; color: #cbd5e1; line-height: 1.5;'>"
            f"Images processed:&nbsp;&nbsp;<b>{processed}</b><br>"
            f"OEM part numbers detected:&nbsp;&nbsp;<b style='color: #10b981;'>{detected}</b><br>"
            f"Total elapsed time:&nbsp;&nbsp;<b>{elapsed_str}</b>"
            f"</div>",
            self
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self._btn_open = QPushButton("Open Output Folder", self)
        self._btn_open.setObjectName("BtnStart")
        self._btn_open.clicked.connect(self._open_folder)

        self._btn_close = QPushButton("Close", self)
        self._btn_close.clicked.connect(self.reject)

        btn_layout.addWidget(self._btn_open)
        btn_layout.addWidget(self._btn_close)
        layout.addLayout(btn_layout)

        self._output_folder = output_folder

    def _open_folder(self) -> None:
        try:
            if self._output_folder.exists():
                os.startfile(self._output_folder)
            elif self._output_folder.parent.exists():
                os.startfile(self._output_folder.parent)
        except Exception as exc:
            log.error("Failed to open output directory: %s", exc)
        self.accept()


# ──────────────────────────────────────────────────────────────────── #
# Main Application Window                                              #
# ──────────────────────────────────────────────────────────────────── #

class MainWindow(QMainWindow):
    """Application main window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_folder: Path | None = None
        self._worker: FolderWorker | None = None
        self._ocr_service: OCRService | None = None

        # Statistics trackers
        self._total_count = 0
        self._processed_count = 0
        self._detected_count = 0
        self._start_time = 0.0

        # UI Initialization
        self._build_ui()
        self._apply_stylesheet()
        self._update_button_states()

        # Elapsed Timer Setup
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        # Warm up OCR in background
        self._init_ocr_service()

    def _build_ui(self) -> None:
        """Construct all widgets and layouts."""
        self.setWindowTitle("OEM Photo Sorter")
        self.setMinimumSize(980, 680)
        self.resize(1000, 700)

        # System Icon
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))

        # Central Widget
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 20, 20, 16)
        root_layout.setSpacing(14)

        # ── 1. Header Widget ─────────────────────────────────────── #
        header_widget = QWidget(central)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(2)

        main_title = QLabel("OEM Photo Sorter", header_widget)
        main_title.setStyleSheet("font-size: 24px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;")
        
        subtitle = QLabel("Automatically detect automotive OEM part numbers using OCR.", header_widget)
        subtitle.setStyleSheet("font-size: 13px; color: #64748b;")

        header_layout.addWidget(main_title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header_widget)

        # ── 2. Folder Selection Controls ─────────────────────────── #
        folder_group = QGroupBox("Source Folder Selection", central)
        folder_layout = QHBoxLayout(folder_group)
        folder_layout.setSpacing(8)

        self._folder_label = QLabel("No folder selected", folder_group)
        self._folder_label.setObjectName("FolderLabel")
        self._folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._btn_select = make_button("Select Folder", "BtnSelectFolder", folder_group)
        self._btn_select.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self._btn_select.clicked.connect(self._on_select_folder)

        folder_layout.addWidget(self._folder_label)
        folder_layout.addWidget(self._btn_select)
        root_layout.addWidget(folder_group)

        # ── 3. Live Dashboard Cards ──────────────────────────────── #
        stats_group = QGroupBox("Performance Statistics", central)
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.setSpacing(10)

        self._card_total = StatCard("📁 Total Images", stats_group)
        self._card_processed = StatCard("✅ Processed", stats_group)
        self._card_detected = StatCard("🔍 OEM Detected", stats_group)
        self._card_time = StatCard("⏱ Elapsed Time", stats_group)
        self._card_speed = StatCard("⚡ Images/Second", stats_group)

        for card in (self._card_total, self._card_processed, self._card_detected, self._card_time, self._card_speed):
            stats_layout.addWidget(card)

        root_layout.addWidget(stats_group)

        # ── 4. Progress Tracking Bar ─────────────────────────────── #
        progress_group = QGroupBox("Processing Progress", central)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)

        self._progress_bar = make_progress_bar(progress_group)
        
        # Details overlay panel
        self._details_label = QLabel("Pending Folder Scan...", progress_group)
        self._details_label.setStyleSheet("color: #64748b; font-size: 12px; font-weight: 500;")
        self._details_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._details_label)
        root_layout.addWidget(progress_group)

        # ── 5. HTML Processing Console log ───────────────────────── #
        log_group = QGroupBox("Activity Console Log", central)
        log_outer_layout = QVBoxLayout(log_group)
        log_outer_layout.setContentsMargins(12, 14, 12, 12)
        log_outer_layout.setSpacing(8)

        # Toolbar Row for Log Management
        log_toolbar = QHBoxLayout()
        log_toolbar.setSpacing(8)

        self._btn_save_log = make_button("Save Log", "BtnSaveLog", log_group, min_width=100)
        self._btn_save_log.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self._btn_save_log.clicked.connect(self._on_save_log)

        self._btn_clear_log = make_button("Clear Log", "BtnClearLog", log_group, min_width=100)
        self._btn_clear_log.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogDiscardButton))
        self._btn_clear_log.clicked.connect(self._on_clear_log)

        log_toolbar.addStretch()
        log_toolbar.addWidget(self._btn_save_log)
        log_toolbar.addWidget(self._btn_clear_log)
        log_outer_layout.addLayout(log_toolbar)

        self._log_widget = LogWidget(log_group)
        log_outer_layout.addWidget(self._log_widget)
        root_layout.addWidget(log_group, stretch=1)

        # ── 6. Bottom Action Controls ────────────────────────────── #
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_open_output = make_button("Open Output Folder", "BtnOpenOutput", central, min_width=180)
        self._btn_open_output.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self._btn_open_output.clicked.connect(self._on_open_output_folder)
        self._btn_open_output.setEnabled(False)

        self._btn_start = make_button("Start Process", "BtnStart", central, min_width=140)
        self._btn_start.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._btn_start.clicked.connect(self._on_start)

        self._btn_cancel = make_button("Cancel", "BtnCancel", central, min_width=120)
        self._btn_cancel.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._btn_open_output)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_cancel)
        root_layout.addLayout(btn_row)

        # ── 7. Native Status bar ─────────────────────────────────── #
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _apply_stylesheet(self) -> None:
        """Apply the global theme stylesheet."""
        self.setStyleSheet(_STYLESHEET)

    def _init_ocr_service(self) -> None:
        """Create the OCR service synchronously."""
        try:
            self._log("Initialising PaddleOCR engine …")
            self._ocr_service = OCRService()
            self._log("✓ PaddleOCR engine initialized successfully.")
            self._status_bar.showMessage("Ready")
        except Exception as exc:  # noqa: BLE001
            self._log(f"ERROR: Failed to initialise OCR engine: {exc}")
            QMessageBox.critical(
                self,
                "OCR Service Error",
                f"Could not load the PaddleOCR model.\n\n{exc}",
            )
            self._btn_start.setEnabled(False)
            self._status_bar.showMessage("Errors")

    # ---------------------------------------------------------------- #
    # Slots & Buttons Events                                            #
    # ---------------------------------------------------------------- #

    @Slot()
    def _on_select_folder(self) -> None:
        """Open folder dialog and store selection."""
        folder_str = QFileDialog.getExistingDirectory(
            self,
            "Select Automotive Parts Images Directory",
            str(self._selected_folder or Path.home()),
        )
        if not folder_str:
            return

        self._selected_folder = Path(folder_str)
        self._folder_label.setText(folder_str)
        self._folder_label.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        self._log(f"Selected working folder: {folder_str}")
        self._status_bar.showMessage("Ready")
        self._update_button_states()

    @Slot()
    def _on_start(self) -> None:
        """Launch background execution worker thread."""
        if not self._selected_folder:
            QMessageBox.warning(self, "Select Folder", "Please select a source folder first.")
            return

        if self._ocr_service is None:
            QMessageBox.critical(
                self, "OCR Service Missing",
                "The PaddleOCR engine is not initialized. Please restart."
            )
            return

        # Reset UI dashboards
        self._progress_bar.setValue(0)
        self._card_total.set_value(0)
        self._card_processed.set_value(0)
        self._card_detected.set_value(0)
        self._card_time.set_value("00:00")
        self._card_speed.set_value(0.0)
        
        self._total_count = 0
        self._processed_count = 0
        self._detected_count = 0

        self._log_widget.clear_log()
        self._btn_open_output.setEnabled(False)

        # Build Worker Thread
        self._worker = FolderWorker(self._selected_folder, self._ocr_service)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.stats_update.connect(self._on_stats_update)
        self._worker.finished.connect(self._on_finished)

        # Start timer and worker
        self._start_time = time.perf_counter()
        self._timer.start(1000)  # tick every 1s
        self._worker.start()

        self._update_button_states()
        self._status_bar.showMessage("Processing...")
        log.info("Worker thread launched for folder: %s", self._selected_folder)

    @Slot()
    def _on_cancel(self) -> None:
        """Request cancel on worker thread."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._btn_cancel.setEnabled(False)
            self._status_bar.showMessage("Cancelled")

    @Slot()
    def _on_open_output_folder(self) -> None:
        """Open copied files output location."""
        if self._selected_folder:
            output_dir = self._selected_folder / "Number_Photos"
            target_dir = output_dir if output_dir.exists() else self._selected_folder
            try:
                os.startfile(target_dir)
            except Exception as exc:
                QMessageBox.warning(self, "Folder Error", f"Failed to open directory: {exc}")

    @Slot()
    def _on_clear_log(self) -> None:
        self._log_widget.clear_log()

    @Slot()
    def _on_save_log(self) -> None:
        """Write log contents to disk file."""
        log_text = self._log_widget.toPlainText()
        if not log_text.strip():
            QMessageBox.information(self, "Empty Console", "Log console is empty. Nothing to write.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Log File",
            str(Path.home() / "oem_photo_sorter_run.log"),
            "Log Files (*.log *.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file_writer:
                    file_writer.write(log_text)
                self._log(f"✓ Saved activity console log successfully to: {file_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", f"Could not write log file:\n{exc}")

    # ---------------------------------------------------------------- #
    # Live Timer tick                                                   #
    # ---------------------------------------------------------------- #

    def _on_timer_tick(self) -> None:
        """Update live statistics elapsed time and throughput speed."""
        if self._start_time == 0.0:
            return

        elapsed = time.perf_counter() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        self._card_time.set_value(f"{mins:02d}:{secs:02d}")

        if elapsed > 0:
            speed = self._processed_count / elapsed
            self._card_speed.set_value(speed)

    # ---------------------------------------------------------------- #
    # Worker Thread Signals                                             #
    # ---------------------------------------------------------------- #

    @Slot(int)
    def _on_progress(self, value: int) -> None:
        self._progress_bar.setValue(value)

    @Slot(str)
    def _on_log_message(self, message: str) -> None:
        self._log_widget.append_line(message)

    @Slot(int, int, int)
    def _on_stats_update(self, total: int, processed: int, detected: int) -> None:
        self._total_count = total
        self._processed_count = processed
        self._detected_count = detected

        self._card_total.set_value(total)
        self._card_processed.set_value(processed)
        self._card_detected.set_value(detected)

        # Update remaining count
        remaining = max(0, total - processed)

        # Update Live text stats overlay
        self._details_label.setText(
            f"Total scanned: {total}  |  Processed: {processed}  |  "
            f"OEM Part Numbers detected: {detected}  |  Remaining: {remaining}"
        )

    @Slot(bool)
    def _on_finished(self, success: bool) -> None:
        """Background thread cleanup."""
        self._timer.stop()
        self._update_button_states()

        # Stop duration tracker
        elapsed_sec = time.perf_counter() - self._start_time
        mins, secs = divmod(int(elapsed_sec), 60)
        elapsed_str = f"{mins:02d}:{secs:02d}"

        if success:
            self._progress_bar.setValue(100)
            self._status_bar.showMessage("Completed successfully")
            
            # Show Completion popup modal report
            if self._selected_folder:
                output_dir = self._selected_folder / "Number_Photos"
                dialog = CompletionDialog(self, self._processed_count, self._detected_count, elapsed_str, output_dir)
                dialog.exec()
        else:
            self._status_bar.showMessage("Cancelled")

        self._worker = None

    # ---------------------------------------------------------------- #
    # State Helpers                                                     #
    # ---------------------------------------------------------------- #

    def _log(self, message: str) -> None:
        self._log_widget.append_line(message)

    def _update_button_states(self) -> None:
        """Sync button availability with execution state."""
        running = self._worker is not None and self._worker.isRunning()
        self._btn_select.setEnabled(not running)
        self._btn_start.setEnabled(
            not running
            and self._selected_folder is not None
            and self._ocr_service is not None
        )
        self._btn_cancel.setEnabled(running)
        self._btn_open_output.setEnabled(not running and self._selected_folder is not None)

    # ---------------------------------------------------------------- #
    # Window Close Override                                             #
    # ---------------------------------------------------------------- #

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Stop worker cleanly on exit."""
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
