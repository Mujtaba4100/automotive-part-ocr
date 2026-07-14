"""
folder_worker.py
----------------
Background QThread worker that drives the complete processing pipeline
(scan → load → OCR → copy) in parallel without blocking the GUI event loop.
Optimized for high-throughput, low-latency, and caching.
"""

from __future__ import annotations

import os
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.services.file_service import FileService
from app.services.image_service import load_and_correct
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

        # SHA256 duplicate image cache & lock
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()

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
        run_start_time = time.perf_counter()

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

        # Calculate optimal thread count: min(cpu_count, 6)
        num_workers = min(os.cpu_count() or 1, 6)
        self._emit_log(f"Starting parallel engine pool with {num_workers} threads...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
        total_time = time.perf_counter() - run_start_time
        avg_time = total_time / processed if processed > 0 else 0.0
        images_per_sec = processed / total_time if total_time > 0 else 0.0

        self._emit_log(
            f"\n✅ Finished! Processed {processed}/{total} images.\n"
            f"Detected: {detected} image(s) copied to Number_Photos/.\n"
            f"Total runtime: {total_time:.2f}s\n"
            f"Average time per image: {avg_time:.2f}s\n"
            f"Throughput: {images_per_sec:.2f} images/second"
        )
        self.finished.emit(True)

    def _process_image(self, image_path: Path, output_folder: Path) -> dict:
        """Process a single image through the optimized Phase 3 pipeline."""
        start_time = time.perf_counter()
        result = {
            "detected": False,
            "matched_numbers": [],
            "highest_confidence": 0.0,
            "copied": False,
            "log": "",
            "duration": 0.0
        }

        # ── 1. Hashing Check ───────────────────────────────────────── #
        file_hash = self._calculate_sha256(image_path)
        with self._cache_lock:
            cached = self._cache.get(file_hash)

        if cached is not None:
            if cached["detected"]:
                result["detected"] = True
                result["matched_numbers"] = cached["matched_numbers"]
                result["highest_confidence"] = cached["highest_confidence"]
                
                copied = FileService.copy_image(image_path, output_folder)
                result["copied"] = copied
                copied_status = "Copied to Number_Photos/" if copied else "Copy failed"
                
                oem_lines = "\n".join(f"  - {num}" for num in result["matched_numbers"])
                result["log"] = (
                    f"  ✓ (Cache Hit) Duplicate image\n"
                    f"  Numbers:\n"
                    f"{oem_lines}\n"
                    f"  Confidence: {result['highest_confidence']:.2f} (Fused)\n"
                    f"  → {copied_status}"
                )
            else:
                result["log"] = "  ✗ (Cache Hit) Duplicate image (No OEM number detected)"
            
            result["duration"] = time.perf_counter() - start_time
            return result

        # ── 2. Load & Correct EXIF ─────────────────────────────────── #
        image_array = load_and_correct(image_path)
        if image_array is None:
            result["log"] = "  ✗ Could not read or decode image."
            result["duration"] = time.perf_counter() - start_time
            return result

        # ── 3. Run Region Detection (Once on base image layout) ────── #
        boxes = self._ocr.detect_text_regions(image_array)

        raw_detections = []
        found_confident_match = False

        from app.services.image_service import (
            crop_and_warp, upscale_image, enhance_clahe, sharpen_image,
            adaptive_threshold, morphological_cleanup, denoise_image, adjust_gamma,
            apply_tophat, apply_unsharp_mask
        )

        # ── 4. First Pass: Original Crops & Smart Upscaling ───────── #
        from app.services.image_service import rotate_image
        for box in boxes:
            if self._cancel_requested:
                break

            crop = crop_and_warp(image_array, box)
            crop_h = crop.shape[0]
            crop_w = crop.shape[1]

            # Aspect ratio rotation check: test vertical rotations if vertical box
            rotations = [90, 270] if crop_h > crop_w else [0]
            text_thickness = crop_w if crop_h > crop_w else crop_h

            # Single-pass smart upscaling factor depending on text thickness
            if text_thickness < 15:
                scale_factor = 3.0
            elif text_thickness < 30:
                scale_factor = 2.0
            else:
                scale_factor = 1.0

            for rot in rotations:
                rotated_crop = rotate_image(crop, rot)
                scaled_crop = upscale_image(rotated_crop, scale_factor)
                
                # Apply edge sharpening to upscaled crops to clarify blurred fonts
                if scale_factor > 1.0:
                    scaled_crop = apply_unsharp_mask(scaled_crop)

                recs = self._ocr.recognise_region(scaled_crop)
                for text, conf in recs:
                    raw_detections.append({
                        "text": text,
                        "ocr_conf": conf,
                        "rotation": rot,
                        "preprocess": "Original",
                        "scale": scale_factor,
                        "is_crop": True
                    })

        # Run Fallback: Full base image OCR
        if not self._cancel_requested:
            fallback_recs = self._ocr.extract_all_text_with_confidences(image_array)
            for text, conf in fallback_recs:
                raw_detections.append({
                    "text": text,
                    "ocr_conf": conf,
                    "rotation": 0,
                    "preprocess": "FullImage",
                    "scale": 1.0,
                    "is_crop": False
                })

        # Evaluate current detections
        det_ok, matched, highest_c, conf_map, meta_map = self._detector.detect_fused(raw_detections)
        if det_ok and highest_c >= 0.90:
            found_confident_match = True

        # ── 5. Second Pass: Lazy Preprocessing Pipelines ─────────────── #
        if not found_confident_match and not self._cancel_requested:
            # We try the 4 most effective enhancements one-by-one: CLAHE, Top-Hat, Sharpen, Thresholded
            enhancement_types = [
                ("CLAHE", lambda c: enhance_clahe(c)),
                ("TopHat", lambda c: apply_tophat(c)),
                ("Sharpened", lambda c: sharpen_image(c)),
                ("Thresholded", lambda c: morphological_cleanup(adaptive_threshold(c)))
            ]

            for name, pipeline_func in enhancement_types:
                if found_confident_match or self._cancel_requested:
                    break

                for box in boxes:
                    if self._cancel_requested:
                        break

                    crop = crop_and_warp(image_array, box)
                    crop_h = crop.shape[0]
                    crop_w = crop.shape[1]

                    # Aspect ratio rotation check
                    rotations = [90, 270] if crop_h > crop_w else [0]
                    text_thickness = crop_w if crop_h > crop_w else crop_h

                    # Process enhancement
                    enhanced_crop = pipeline_func(crop)
                    
                    # Single-pass smart upscaling factor
                    if text_thickness < 15:
                        scale_factor = 3.0
                    elif text_thickness < 30:
                        scale_factor = 2.0
                    else:
                        scale_factor = 1.0

                    for rot in rotations:
                        rotated_enhanced = rotate_image(enhanced_crop, rot)
                        scaled_enhanced = upscale_image(rotated_enhanced, scale_factor)
                        
                        if scale_factor > 1.0:
                            scaled_enhanced = apply_unsharp_mask(scaled_enhanced)

                        recs = self._ocr.recognise_region(scaled_enhanced)
                        for text, conf in recs:
                            raw_detections.append({
                                "text": text,
                                "ocr_conf": conf,
                                "rotation": rot,
                                "preprocess": name,
                                "scale": scale_factor,
                                "is_crop": True
                            })

                # Re-evaluate fusion after this specific pipeline runs
                det_ok, matched, highest_c, conf_map, meta_map = self._detector.detect_fused(raw_detections)
                if det_ok and highest_c >= 0.90:
                    found_confident_match = True
                    break

        # Final evaluation run over all collected candidates
        det_ok, matched, highest_c, conf_map, meta_map = self._detector.detect_fused(raw_detections)
        duration = time.perf_counter() - start_time

        # Update cache
        cache_data = {
            "detected": det_ok and highest_c >= OEM_CONFIDENCE_THRESHOLD,
            "matched_numbers": matched,
            "highest_confidence": highest_c
        }
        with self._cache_lock:
            self._cache[file_hash] = cache_data

        # Finalize results
        if cache_data["detected"]:
            result["detected"] = True
            result["matched_numbers"] = matched
            result["highest_confidence"] = highest_c

            copied = FileService.copy_image(image_path, output_folder)
            result["copied"] = copied
            copied_status = "Copied to Number_Photos/" if copied else "Copy failed"

            best_num = matched[0]
            # pyrefly: ignore [bad-unpacking]
            best_pipeline, best_rotation, best_scale = meta_map.get(best_num, ("Original", 0, 1.0))
            
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

        result["duration"] = duration
        return result

    def _calculate_sha256(self, filepath: Path) -> str:
        """Calculate the SHA256 signature hash of a file."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as exc:
            log.warning("Failed to compute hash for %s: %s", filepath, exc)
            return str(filepath)

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
