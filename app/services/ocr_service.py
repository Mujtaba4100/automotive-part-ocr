"""
ocr_service.py
--------------
Thin wrapper around PaddleOCR 2.7.x.  The heavy model is loaded **once**
at construction time and reused for every inference call to avoid repeated
cold-start overhead.

Requires
--------
  paddleocr==2.7.3
  paddlepaddle==2.6.2

PaddlePaddle 3.x has an unresolved OneDNN / PIR incompatibility on
Windows.  Pin to the 2.x line (see requirements.txt).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from paddleocr import PaddleOCR

from app.utils.logger import get_logger
from app.utils.constants import MIN_TEXT_LENGTH

log = get_logger(__name__)


class OCRService:
    """Singleton-style OCR service that wraps PaddleOCR 2.7.x.

    Example::

        svc = OCRService()
        has_text = svc.has_meaningful_text(image_array)
    """

    def __init__(self) -> None:
        """Initialise and warm-up the PaddleOCR engine."""
        log.info("Initialising PaddleOCR engine …")
        try:
            # use_angle_cls=True  → detects rotated text automatically
            # lang='en'           → English model (smallest & fastest)
            # show_log=False      → suppress PaddleOCR's verbose output
            self._engine: PaddleOCR = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
            log.info("PaddleOCR engine ready.")
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to initialise PaddleOCR: %s", exc)
            raise

    # ---------------------------------------------------------------- #
    # Public API                                                        #
    # ---------------------------------------------------------------- #

    def has_meaningful_text(self, image: np.ndarray) -> bool:
        """Return ``True`` if *image* contains at least one text block
        with ``MIN_TEXT_LENGTH`` or more alphanumeric characters.

        Args:
            image: ``uint8`` NumPy array (H × W × 3, RGB).

        Returns:
            ``True`` when meaningful text was detected, ``False`` otherwise.
        """
        raw_results = self._run_ocr(image)
        if not raw_results:
            return False

        for line in raw_results:
            if not line:
                continue
            for detection in line:
                text = self._extract_text(detection)
                if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                    return True

        return False

    def extract_all_text(self, image: np.ndarray) -> list[str]:
        """Extract every recognised text string from *image*.

        Args:
            image: ``uint8`` NumPy array (H × W × 3, RGB).

        Returns:
            List of recognised text strings (may be empty).
        """
        raw_results = self._run_ocr(image)
        texts: list[str] = []

        if not raw_results:
            return texts

        for line in raw_results:
            if not line:
                continue
            for detection in line:
                text = self._extract_text(detection)
                if text:
                    texts.append(text.strip())

        return texts

    def extract_all_text_with_confidences(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Extract every recognised text string along with its confidence score from *image*.

        Args:
            image: ``uint8`` NumPy array (H × W × 3, RGB).

        Returns:
            List of (text, confidence) tuples.
        """
        raw_results = self._run_ocr(image)
        results: list[tuple[str, float]] = []

        if not raw_results:
            return results

        for line in raw_results:
            if not line:
                continue
            for detection in line:
                try:
                    text = str(detection[1][0])
                    conf = float(detection[1][1])
                    if text:
                        results.append((text.strip(), conf))
                except (IndexError, TypeError, ValueError):
                    continue

        return results


    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _run_ocr(self, image: np.ndarray) -> Optional[list]:
        """Execute PaddleOCR inference, returning raw results or ``None``.

        The 2.7.x result format is a nested list::

            [
              [  # per-image
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", confidence)
              ],
              …
            ]
        """
        try:
            return self._engine.ocr(image, cls=True)
        except Exception as exc:  # noqa: BLE001
            log.error("OCR inference error: %s", exc)
            return None

    @staticmethod
    def _extract_text(detection: object) -> str:
        """Pull the text string out of a single PaddleOCR detection tuple.

        PaddleOCR 2.x returns detections in the form::

            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ("text", confidence)

        Args:
            detection: A single detection item from the OCR results.

        Returns:
            The recognised text string, or an empty string on failure.
        """
        try:
            # detection[1] → ("text", confidence)
            return str(detection[1][0])
        except (IndexError, TypeError):
            return ""
