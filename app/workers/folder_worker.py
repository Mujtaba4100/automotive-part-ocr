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

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.services.file_service import FileService
from app.services.image_service import load_and_correct
from app.services.ocr_service import OCRService
from app.services.oem_detector import OEMDetector
from app.utils.constants import OEM_CONFIDENCE_THRESHOLD, MAX_WORKERS
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

        # ── 3. Process each image in parallel ──────────────────────── #
        processed = 0
        detected = 0
        stats_lock = threading.Lock()

        # We submit to a ThreadPoolExecutor with MAX_WORKERS threads
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a dict mapping future objects to their index and image path
            futures = {
                executor.submit(self._process_image, img_path, output_folder): (idx, img_path)
                for idx, img_path in enumerate(images)
            }

            from concurrent.futures import as_completed
            for future in as_completed(futures):
                if self._cancel_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    self._emit_log("⚠ Processing cancelled by user.")
                    self.finished.emit(False)
                    return

                idx, img_path = futures[future]
                try:
                    res = future.result()
                    with stats_lock:
                        processed += 1
                        if res["detected"] and res["copied"]:
                            detected += 1
                        self._emit_log(f"[{processed}/{total}] {img_path.name}:\n{res['log']}")
                        self._emit_progress(processed, total)
                        self.stats_update.emit(total, processed, detected)
                except Exception as exc:
                    with stats_lock:
                        processed += 1
                        self._emit_log(f"[{processed}/{total}] {img_path.name}: ✗ Error processing image: {exc}")
                        self._emit_progress(processed, total)
                        self.stats_update.emit(total, processed, detected)

        # ── 4. Done ───────────────────────────────────────────────── #
        self._emit_log(
            f"\n✅ Finished! Processed {processed}/{total} images. "
            f"Detected: {detected} image(s) copied to Number_Photos/."
        )
        self.finished.emit(True)

    def _process_image(self, image_path: Path, output_folder: Path) -> dict:
        """Process a single image through the complete Phase 3 pipeline."""
        start_time = time.perf_counter()
        result = {
            "detected": False,
            "matched_numbers": [],
            "highest_confidence": 0.0,
            "copied": False,
            "log": ""
        }

        # Load image & correct EXIF
        image_array = load_and_correct(image_path)
        if image_array is None:
            result["log"] = "  ✗ Could not read or decode image."
            return result

        raw_detections = []
        found_confident_match = False

        from app.services.image_service import rotate_image, upscale_image, crop_and_warp, get_advanced_enhancements

        # Multi-scale & multi-rotation candidate generator
        # Try 1.0x scale (normal), then 1.5x, then 2.0x
        for scale in [1.0, 1.5, 2.0]:
            if found_confident_match or self._cancel_requested:
                break

            scaled_img = upscale_image(image_array, scale)

            # Test rotations: 0°, 180°, 90°, 270°
            for rotation in [0, 180, 90, 270]:
                if found_confident_match or self._cancel_requested:
                    break

                rotated_img = rotate_image(scaled_img, rotation)

                # A) Crop perspective-corrected region boxes (automatic region detection)
                boxes = self._ocr.detect_text_regions(rotated_img)
                for box in boxes:
                    if self._cancel_requested:
                        break

                    crop = crop_and_warp(rotated_img, box)
                    # Advanced Preprocessing (CLAHE, thresholding, denoising, sharpening, gamma, norm)
                    enhancements = get_advanced_enhancements(crop)
                    
                    for enh in enhancements:
                        if self._cancel_requested:
                            break

                        pipeline = enh["pipeline"]
                        enh_img = enh["image"]

                        # Recognition crop run
                        recs = self._ocr.recognise_region(enh_img)
                        for text, conf in recs:
                            raw_detections.append({
                                "text": text,
                                "ocr_conf": conf,
                                "rotation": rotation,
                                "preprocess": pipeline,
                                "scale": scale,
                                "is_crop": True
                            })

                # B) Fallback: Full image recognition on this rotated scale
                fallback_recs = self._ocr.extract_all_text_with_confidences(rotated_img)
                for text, conf in fallback_recs:
                    raw_detections.append({
                        "text": text,
                        "ocr_conf": conf,
                        "rotation": rotation,
                        "preprocess": "FullImage",
                        "scale": scale,
                        "is_crop": False
                    })

                # C) Fast Escape Check: Evaluate current gathered candidates
                det_ok, matched, highest_c, conf_map, meta_map = self._detector.detect_fused(raw_detections)
                if det_ok and highest_c >= 0.90:
                    found_confident_match = True
                    break

        # Final evaluation run over all collected candidates
        det_ok, matched, highest_c, conf_map, meta_map = self._detector.detect_fused(raw_detections)
        duration = time.perf_counter() - start_time

        if det_ok and highest_c >= OEM_CONFIDENCE_THRESHOLD:
            result["detected"] = True
            result["matched_numbers"] = matched
            result["highest_confidence"] = highest_c

            # Copy file
            copied = FileService.copy_image(image_path, output_folder)
            result["copied"] = copied
            copied_status = "Copied to Number_Photos/" if copied else "Copy failed"

            # Log details matching requirements
            best_num = matched[0]
            best_pipeline, best_rotation, best_scale = meta_map.get(best_num, ("Unknown", 0, 1.0))
            
            oem_lines = "\n".join(f"  - {num}" for num in matched)
            result["log"] = (
                f"  ✓ OEM detected\n"
                f"  Numbers:\n"
                f"{oem_lines}\n"
                f"  Confidence: {highest_c:.2f} (Fused)\n"
                f"  Processing time: {duration:.2f}s\n"
                f"  Pipeline: {best_pipeline} (Scale {best_scale}x)\n"
                f"  Rotation: {best_rotation}°\n"
                f"  → {copied_status}"
            )
        else:
            result["log"] = (
                f"  ✗ No OEM number detected\n"
                f"  Processing time: {duration:.2f}s"
            )

        return result

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
