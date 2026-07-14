"""
oem_detector.py
---------------
Dedicated detector for extracting and validating automotive OEM part numbers
using regular expressions, token normalization, blacklists, and a heuristic
multi-factor confidence fusion system.
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

    @staticmethod
    def reconstruct_oem(val: str) -> str:
        """Perform fuzzy auto-reconstruction and cleanup of prefix/suffix noise."""
        if not val:
            return val

        # 1. Strip common label prefixes (LU, HW, SW, ZB, ZP, PPT, TYP, TYPE, PART, NO, NR)
        # Loop multiple times to handle combined prefixes (e.g. PPT LU A212...)
        changed = True
        while changed:
            changed = False
            for prefix in ["LUA", "HWA", "SWA", "ZBA", "ZPA", "LU", "HW", "SW", "ZB", "ZP", "PPT", "TYP", "TYPE", "PART", "NO", "NR"]:
                if prefix in ["LUA", "HWA", "SWA", "ZBA", "ZPA"]:
                    if val.startswith(prefix):
                        val = val[len(prefix)-1:]
                        changed = True
                        break
                else:
                    if val.startswith(prefix):
                        val = val[len(prefix):]
                        changed = True
                        break

        # 2. Strip common blacklisted words from start/end of the token
        changed = True
        while changed:
            changed = False
            for word in ["GERMANY", "REAR", "FRONT", "LEFT", "RIGHT", "MADE", "ORIGINAL", "OEM", "ABS", "WARRANTY", "PARTSUT", "PARTSWUT", "90DAYS"]:
                if val.startswith(word):
                    val = val[len(word):]
                    changed = True
                    break
                if val.endswith(word):
                    val = val[:-len(word)]
                    changed = True
                    break

        # 3. Strip Mercedes revision/quality stand suffixes (e.g. Q1, Q2, Q01, Q02, or trailing digits after 10-digit sequence)
        if val.startswith("A") and len(val) > 11:
            if val[1:11].isdigit():
                suffix = val[11:]
                if suffix.startswith("Q") or suffix.isdigit():
                    val = val[:11]

        # 4. Perform Mercedes chassis code auto-reconstruction (e.g. "2048601669" -> "A2048601669")
        mercedes_chassis = {
            "124", "129", "140", "163", "164", "166", "168", "169", "170", "171", "172",
            "201", "202", "203", "204", "205", "207", "208", "209", "210", "211", "212",
            "213", "215", "216", "218", "219", "220", "221", "222", "230", "231", "245",
            "246", "251", "463", "639", "906"
        }
        if len(val) == 10 and val.isdigit():
            prefix = val[:3]
            if prefix in mercedes_chassis:
                return "A" + val

        # 5. VDO/Continental prefix auto-reconstruction (e.g. "2C53420732" -> "A2C53420732")
        if len(val) == 10 and val.startswith("2C"):
            if val[2].isdigit():
                return "A" + val

        return val

    def is_blacklisted(self, normalized_text: str) -> bool:
        """Return True if the normalized candidate is present in the blacklist."""
        if not normalized_text:
            return True
        return normalized_text in self._normalized_blacklist

    def calculate_confidence(self, normalized: str, ocr_conf: float) -> float:
        """Compute a basic multi-factor confidence score for a single detection."""
        if not normalized or len(normalized) < 3:
            return 0.0

        if self.is_blacklisted(normalized):
            return 0.0

        # ── 1. Pattern Match Score ─────────────────────────────────── #
        pattern_score = 0.0
        matched_any = False

        # Try specific patterns first (Mercedes, VDO, VAG V1, VAG V2, BMW, Toyota, Honda, Ford, Valeo)
        for pattern in self._compiled_patterns[:-1]:
            if pattern.match(normalized):
                if pattern.pattern == r"^\d{6}$":
                    pattern_score = 0.7
                else:
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
        """Legacy detection method. Keep for simple verification scripts."""
        matches: list[tuple[str, float]] = []

        for text, ocr_conf in detections:
            if ocr_conf < OCR_CONFIDENCE_THRESHOLD:
                continue

            tokens = re.split(r"[\t\n\r|\\/]+|\s{2,}", text)
            candidates = list(tokens)
            if len(tokens) > 1:
                candidates.append(text)

            for token in candidates:
                token_clean = token.strip().strip("> < [ ] ( )")
                if not token_clean:
                    continue

                normalized = self.normalize(token_clean)
                normalized = self.reconstruct_oem(normalized)
                conf = self.calculate_confidence(normalized, ocr_conf)

                if conf >= OEM_CONFIDENCE_THRESHOLD:
                    matches.append((normalized, conf))

        if not matches:
            return False, [], 0.0

        matches.sort(key=lambda x: x[1], reverse=True)
        unique_numbers: list[str] = []
        for num, _ in matches:
            if num not in unique_numbers:
                unique_numbers.append(num)

        highest_confidence = matches[0][1]
        return True, unique_numbers, highest_confidence

    def detect_fused(self, raw_detections: list[dict]) -> tuple[bool, list[str], float, dict[str, float], dict[str, tuple[str, int]]]:
        """Process multiple raw detections and apply Phase 3 Confidence Fusion.

        Args:
            raw_detections: List of dicts, each with keys:
                "text" (str)
                "ocr_conf" (float)
                "rotation" (int)
                "preprocess" (str)
                "scale" (float)
                "is_crop" (bool)

        Returns:
            Tuple: (detected: bool, matched_numbers: list[str], highest_conf: float,
                    confidences_map: dict[str, float], metadata_map: dict[str, tuple[str, int]])
        """
        candidates_map: dict[str, list[dict]] = {}

        # ── 1. Gather and Clean Candidates ─────────────────────────── #
        for det in raw_detections:
            text = det["text"]
            ocr_conf = det["ocr_conf"]
            if ocr_conf < OCR_CONFIDENCE_THRESHOLD:
                continue

            tokens = re.split(r"[\t\n\r|\\/]+|\s{2,}", text)
            candidates = list(tokens)
            if len(tokens) > 1:
                candidates.append(text)

            for token in candidates:
                token_clean = token.strip().strip("> < [ ] ( )")
                if not token_clean:
                    continue

                normalized = self.normalize(token_clean)
                normalized = self.reconstruct_oem(normalized)
                if self.is_blacklisted(normalized) or len(normalized) < 3:
                    continue

                if normalized not in candidates_map:
                    candidates_map[normalized] = []
                candidates_map[normalized].append(det)

        # ── 2. Run Confidence Fusion Scoring ───────────────────────── #
        fused_matches: list[tuple[str, float, str, int]] = []
        for normalized, detections in candidates_map.items():
            ocr_confs = [d["ocr_conf"] for d in detections]
            ocr_base = max(ocr_confs) if ocr_confs else 0.0

            # Determine regex pattern match score
            pattern_score = 0.0
            matched_any = False
            for pattern in self._compiled_patterns[:-1]:
                if pattern.match(normalized):
                    # Downgrade 6-digit Valeo pattern to prevent false positive boosts
                    if pattern.pattern == r"^\d{6}$":
                        pattern_score = 0.7
                    else:
                        pattern_score = 1.0
                    matched_any = True
                    break

            if not matched_any:
                fallback_pattern = self._compiled_patterns[-1]
                if fallback_pattern.match(normalized):
                    pattern_score = 0.7
                    matched_any = True

            # If it matches absolutely no configured pattern, reject it
            if not matched_any:
                continue

            # Agreement score (frequency of detection)
            num_detections = len(detections)
            agreement_score = min(1.0, 0.4 + (num_detections - 1) * 0.15)

            # Rotation agreement (diversity of angles)
            unique_rotations = {d["rotation"] for d in detections}
            rotation_score = min(1.0, 0.5 + (len(unique_rotations) - 1) * 0.5)

            # Preprocessing agreement (diversity of pipelines)
            unique_preprocesses = {d["preprocess"] for d in detections}
            preprocess_score = min(1.0, 0.5 + (len(unique_preprocesses) - 1) * 0.25)

            # Weights: OCR base (50%), agreement (15%), rotation (15%), preprocess (20%)
            factor_score = (ocr_base * 0.50) + (agreement_score * 0.15) + (rotation_score * 0.15) + (preprocess_score * 0.20)
            
            # Apply +0.20 boost for specific brand patterns to prevent false negatives on genuine parts
            if pattern_score == 1.0:
                factor_score += 0.20
                
            fused_conf = pattern_score * factor_score
            fused_conf = min(max(fused_conf, 0.0), 1.0)

            if fused_conf >= OEM_CONFIDENCE_THRESHOLD:
                # Find the rotation, preprocess, and scale that gave the highest base OCR confidence for logging
                best_det = max(detections, key=lambda d: d["ocr_conf"])
                best_preprocess = best_det["preprocess"]
                best_rotation = best_det["rotation"]
                best_scale = best_det["scale"]
                fused_matches.append((normalized, fused_conf, best_preprocess, best_rotation, best_scale))

        if not fused_matches:
            return False, [], 0.0, {}, {}

        # Sort matches by confidence descending
        fused_matches.sort(key=lambda x: x[1], reverse=True)

        matched_numbers = [m[0] for m in fused_matches]
        highest_confidence = fused_matches[0][1]
        
        confidences_map = {m[0]: m[1] for m in fused_matches}
        metadata_map = {m[0]: (m[2], m[3], m[4]) for m in fused_matches} # best (preprocess, rotation, scale)

        return True, matched_numbers, highest_confidence, confidences_map, metadata_map
