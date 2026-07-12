"""
oem_detector.py
---------------
Dedicated detector for extracting and validating automotive OEM part numbers
using regular expressions, token normalization, blacklists, and a heuristic
multi-factor confidence scoring model.
"""

from __future__ import annotations

import re
from app.utils.constants import BLACKLIST, SUPPORTED_PATTERNS, OCR_CONFIDENCE_THRESHOLD, OEM_CONFIDENCE_THRESHOLD
from app.utils.logger import get_logger

log = get_logger(__name__)


class OEMDetector:
    """Detects, extracts, and scores automotive OEM part numbers from text."""

    def __init__(self) -> None:
        # Pre-compile patterns
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in SUPPORTED_PATTERNS]
        # Pre-normalize the blacklist for fast matching
        self._normalized_blacklist = {self.normalize(w) for w in BLACKLIST}

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize input string to uppercase and strip all non-alphanumeric chars.

        Example:
            "A.2076800489"  -> "A2076800489"
            "D-A2078300054" -> "DA2078300054"
            "A207 6800489"  -> "A2076800489"
            ">PPT20<"       -> "PPT20"
        """
        # Convert to uppercase
        val = text.upper()
        # Keep only A-Z and 0-9
        return re.sub(r"[^A-Z0-9]", "", val)

    def is_blacklisted(self, normalized_text: str) -> bool:
        """Return True if the normalized candidate is present in the blacklist."""
        if not normalized_text:
            return True
        return normalized_text in self._normalized_blacklist

    def calculate_confidence(self, normalized: str, ocr_conf: float) -> float:
        """Compute a multi-factor confidence score for a normalized part number candidate.

        Scores are based on:
          1. OCR engine confidence
          2. Regular expression match quality (specific patterns vs fallback)
          3. Character composition (alphanumeric vs purely letters/digits)
          4. String length

        Returns:
            Confidence value in range [0.0, 1.0].
        """
        if not normalized or len(normalized) < 3:
            return 0.0

        if self.is_blacklisted(normalized):
            return 0.0

        # ── 1. Pattern Match Score ─────────────────────────────────── #
        pattern_score = 0.0
        matched_any = False

        # Try specific patterns first (Mercedes, VDO, VAG V1, VAG V2)
        for pattern in self._compiled_patterns[:-1]:
            if pattern.match(normalized):
                pattern_score = 1.0
                matched_any = True
                break

        # Try general/fallback pattern if specific ones failed
        if not matched_any:
            fallback_pattern = self._compiled_patterns[-1]
            if fallback_pattern.match(normalized):
                pattern_score = 0.7
                matched_any = True

        # If it doesn't match any pattern, confidence is 0.0
        if not matched_any:
            return 0.0

        # ── 2. Character Composition Score ─────────────────────────── #
        has_letters = any(c.isalpha() for c in normalized)
        has_digits = any(c.isdigit() for c in normalized)

        if has_letters and has_digits:
            comp_score = 1.0
        elif has_digits:
            comp_score = 0.6  # Purely numeric strings are less typical but possible
        else:
            comp_score = 0.2  # Purely alphabetical strings are highly unlikely

        # ── 3. Length Score ────────────────────────────────────────── #
        length = len(normalized)
        if 9 <= length <= 11:
            len_score = 1.0
        elif length == 8 or length == 12:
            len_score = 0.8
        else:
            len_score = 0.4

        # ── 4. Weighted Aggregate ──────────────────────────────────── #
        # Weights: OCR (35%), Pattern (35%), Composition (15%), Length (15%)
        conf = (ocr_conf * 0.35) + (pattern_score * 0.35) + (comp_score * 0.15) + (len_score * 0.15)
        return min(max(conf, 0.0), 1.0)

    def detect(self, detections: list[tuple[str, float]]) -> tuple[bool, list[str], float]:
        """Process OCR detections and find the most probable OEM part numbers.

        Args:
            detections: List of (text_string, ocr_confidence) tuples.

        Returns:
            A tuple of (detected, matched_numbers, highest_confidence).
        """
        matches: list[tuple[str, float]] = []

        for text, ocr_conf in detections:
            if ocr_conf < OCR_CONFIDENCE_THRESHOLD:
                continue

            # Split only by:
            # - Pipes: |
            # - Slashes/backslashes: /, \
            # - Two or more consecutive spaces: \s{2,}
            # - Newlines/tabs
            # This allows single spaces (e.g. "A207 6800489") to remain together.
            tokens = re.split(r"[\t\n\r|\\/]+|\s{2,}", text)
            
            # Also evaluate the entire raw text line as a fallback candidate
            candidates = list(tokens)
            if len(tokens) > 1:
                candidates.append(text)

            for token in candidates:
                token_clean = token.strip().strip("> < [ ] ( )")
                if not token_clean:
                    continue

                normalized = self.normalize(token_clean)
                conf = self.calculate_confidence(normalized, ocr_conf)

                if conf >= OEM_CONFIDENCE_THRESHOLD:
                    matches.append((normalized, conf))

        if not matches:
            return False, [], 0.0

        # Sort matches by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)
        unique_numbers: list[str] = []
        for num, _ in matches:
            if num not in unique_numbers:
                unique_numbers.append(num)

        highest_confidence = matches[0][1]
        return True, unique_numbers, highest_confidence
