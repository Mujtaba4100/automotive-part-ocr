"""
folder_worker.py
----------------
Background QThread worker that drives the complete processing pipeline
(scan → load → OCR → copy) without ever blocking the GUI event loop.

Signals emitted to the main window
------------------------------------
progress(int)           – 0-100 percentage for the progress bar
log_message(str)        – human-readable status line for the log widget
stats_update(int, int, int)
                        – (total, processed, detected) counter tuple
finished(bool)          – True = completed normally, False = cancelled
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.services.file_service import FileService
from app.services.image_service import load_and_correct, get_ocr_candidates
from app.services.ocr_service import OCRService
from app.services.oem_detector import OEMDetector
from app.utils.constants import OEM_CONFIDENCE_THRESHOLD
from app.utils.logger import get_logger

log = get_logger(__name__)


class FolderWorker(QThread):
    """QThread subclass that processes all images in a folder.

    Args:
        folder: Root directory chosen by the user.
        ocr_service: Pre-constructed :class:`~app.services.ocr_service.OCRService`
            instance so the model is not reloaded on every run.
    """

    # ---------------------------------------------------------------- #
    # Qt Signals                                                        #
    # ---------------------------------------------------------------- #
    progress: Signal = Signal(int)                 # 0-100
    log_message: Signal = Signal(str)              # text line
    stats_update: Signal = Signal(int, int, int)   # total, processed, detected
    finished: Signal = Signal(bool)                # True=ok, False=cancelled

    # ---------------------------------------------------------------- #
    # Construction                                                      #
    # ---------------------------------------------------------------- #

    def __init__(self, folder: Path, ocr_service: OCRService) -> None:
        super().__init__()
        self._folder: Path = folder
        self._ocr: OCRService = ocr_service
        self._detector: OEMDetector = OEMDetector()
        self._cancel_requested: bool = False

    # ---------------------------------------------------------------- #
    # Public control                                                    #
    # ---------------------------------------------------------------- #

    def request_cancel(self) -> None:
        """Signal the worker to stop after the current image finishes."""
        self._cancel_requested = True
        log.info("Cancellation requested.")

    # ---------------------------------------------------------------- #
    # QThread entry point                                               #
    # ---------------------------------------------------------------- #

    def run(self) -> None:
        """Main processing loop – executed in the background thread."""
        self._cancel_requested = False

        # ── 1. Scan for images ────────────────────────────────────── #
        self._emit_log("Scanning folder for images …")
        try:
            images = FileService.scan_images(self._folder)
        except Exception as exc:  # noqa: BLE001
            self._emit_log(f"ERROR scanning folder: {exc}")
            self.finished.emit(False)
            return

        total = len(images)
        if total == 0:
            self._emit_log("No supported images found in the selected folder.")
            self.stats_update.emit(0, 0, 0)
            self.finished.emit(True)
            return

        self._emit_log(f"Found {total} image(s). Starting processing …")
        self.stats_update.emit(total, 0, 0)

        # ── 2. Ensure output folder exists ────────────────────────── #
        try:
            output_folder = FileService.ensure_output_folder(self._folder)
            self._emit_log(f"Output folder: {output_folder}")
        except OSError as exc:
            self._emit_log(f"ERROR creating output folder: {exc}")
            self.finished.emit(False)
            return

        # ── 3. Process each image ─────────────────────────────────── #
        processed = 0
        detected = 0

        for index, image_path in enumerate(images):
            if self._cancel_requested:
                self._emit_log("⚠ Processing cancelled by user.")
                self.finished.emit(False)
                return

            self._emit_log(f"[{index + 1}/{total}] Processing: {image_path.name}")

            # Load & correct orientation
            image_array = load_and_correct(image_path)
            if image_array is None:
                self._emit_log(f"  ✗ Could not read image – skipping.")
                processed += 1
                self._emit_progress(processed, total)
                self.stats_update.emit(total, processed, detected)
                continue

            # Multi-attempt OCR on preprocessed/rotated candidates
            best_detected = False
            best_numbers: list[str] = []
            best_conf = 0.0
            best_desc = ""

            try:
                candidates = get_ocr_candidates(image_array)
                for cand in candidates:
                    if self._cancel_requested:
                        break

                    cand_desc = cand["desc"]
                    cand_img = cand["image"]

                    # Extract OCR text strings and confidences
                    detections = self._ocr.extract_all_text_with_confidences(cand_img)
                    
                    # Run detections through the OEM detector
                    det_ok, matched, highest_c = self._detector.detect(detections)

                    if det_ok and highest_c > best_conf:
                        best_detected = True
                        best_numbers = matched
                        best_conf = highest_c
                        best_desc = cand_desc

                        # Optimisation: if we find a very high-confidence match, escape early
                        if best_conf >= 0.92:
                            log.debug(
                                "Early exit for '%s' triggered by %s match with conf %s",
                                image_path.name, cand_desc, highest_c
                            )
                            break
            except Exception as exc:  # noqa: BLE001
                self._emit_log(f"  ✗ OCR processing pipeline error: {exc}")
                processed += 1
                self._emit_progress(processed, total)
                self.stats_update.emit(total, processed, detected)
                continue

            # Actions based on detection results
            if best_detected and best_conf >= OEM_CONFIDENCE_THRESHOLD:
                # Log results using the specified format
                oem_lines = "\n".join(f"- {num}" for num in best_numbers)
                log_message = (
                    f"  ✓ OEM detected ({best_desc})\n"
                    f"  Numbers:\n"
                    f"{oem_lines}\n"
                    f"  Confidence: {best_conf:.2f}"
                )
                self._emit_log(log_message)

                # Copy file
                copied = FileService.copy_image(image_path, output_folder)
                if copied:
                    detected += 1
                    self._emit_log(f"  → Copied to Number_Photos/")
                else:
                    self._emit_log(f"  ✗ Copy failed.")
            else:
                self._emit_log("  ✗ No OEM number detected")

            processed += 1
            self._emit_progress(processed, total)
            self.stats_update.emit(total, processed, detected)

        # ── 4. Done ───────────────────────────────────────────────── #
        self._emit_log(
            f"\n✅ Finished! Processed {processed}/{total} images. "
            f"Detected: {detected} image(s) copied to Number_Photos/."
        )
        self.finished.emit(True)

    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _emit_log(self, message: str) -> None:
        """Emit *message* via the :pyattr:`log_message` signal and also
        write it to the module logger."""
        log.info(message)
        self.log_message.emit(message)

    def _emit_progress(self, processed: int, total: int) -> None:
        """Compute a 0-100 percentage and emit :pyattr:`progress`."""
        pct = int((processed / total) * 100) if total else 0
        self.progress.emit(pct)
