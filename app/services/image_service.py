"""
image_service.py
----------------
Handles all image I/O, EXIF orientation correction, and colour-space
normalisation.  Never modifies files on disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError

from app.utils.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------- #
# EXIF orientation tag value → (rotation_degrees, flip_horizontally)    #
# --------------------------------------------------------------------- #
# Reference: https://exiftool.org/TagNames/EXIF.html#:~:text=Orientation #
_EXIF_ORIENTATION_TAG: int = next(
    tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"
)

_ORIENTATION_MAP: dict[int, tuple[int, bool]] = {
    1: (0, False),    # Normal
    2: (0, True),     # Mirrored horizontal
    3: (180, False),  # Rotated 180°
    4: (180, True),   # Mirrored vertical
    5: (270, True),   # Mirrored horizontal + rotated 270°
    6: (270, False),  # Rotated 270° CW  (i.e. 90° CCW)
    7: (90, True),    # Mirrored horizontal + rotated 90°
    8: (90, False),   # Rotated 90° CW
}


def load_and_correct(image_path: Path) -> Optional[np.ndarray]:
    """Load an image from *image_path*, apply EXIF orientation correction,
    and return it as an **RGB** NumPy array suitable for PaddleOCR.

    The original file is **never** modified.

    Args:
        image_path: Absolute path to the source image.

    Returns:
        A ``uint8`` NumPy array with shape ``(H, W, 3)`` in RGB colour
        order, or ``None`` if the image could not be loaded.
    """
    try:
        pil_image = _open_pil(image_path)
        if pil_image is None:
            return None

        pil_image = _apply_exif_rotation(pil_image)
        pil_image = pil_image.convert("RGB")
        return np.array(pil_image, dtype=np.uint8)

    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load image '%s': %s", image_path, exc)
        return None


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _open_pil(image_path: Path) -> Optional[Image.Image]:
    """Open an image with Pillow, returning ``None`` on failure."""
    try:
        img = Image.open(image_path)
        img.load()          # force decode so truncated files raise here
        return img
    except (UnidentifiedImageError, OSError) as exc:
        log.warning("Cannot open image '%s': %s", image_path, exc)
        return None


def _apply_exif_rotation(image: Image.Image) -> Image.Image:
    """Read the EXIF Orientation tag and rotate/flip *image* accordingly.

    Args:
        image: Source PIL image (may or may not contain EXIF data).

    Returns:
        Orientation-corrected PIL image.
    """
    try:
        exif_data = image._getexif()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        exif_data = None

    if not exif_data:
        return image

    orientation_value: int = exif_data.get(_EXIF_ORIENTATION_TAG, 1)
    degrees, flip = _ORIENTATION_MAP.get(orientation_value, (0, False))

    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if degrees:
        # Pillow's rotate is counter-clockwise; our map stores CW degrees
        image = image.rotate(-degrees, expand=True)

    return image


# ------------------------------------------------------------------ #
# Preprocessing Pipeline (Phase 2)                                   #
# ------------------------------------------------------------------ #

def enhance_clahe(image: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as exc:
        log.warning("CLAHE enhancement failed: %s", exc)
        return image


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply edge-preserving bilateral filter to denoise the image."""
    try:
        # bilateralFilter preserves high frequency edges (like text) while smoothing textures
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    except Exception as exc:
        log.warning("Denoising failed: %s", exc)
        return image


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Apply a sharpening kernel filter to enhance text legibility."""
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    except Exception as exc:
        log.warning("Sharpening failed: %s", exc)
        return image


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Convert image to black and white using adaptive thresholding."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    except Exception as exc:
        log.warning("Adaptive thresholding failed: %s", exc)
        return image


def morphological_cleanup(image: np.ndarray) -> np.ndarray:
    """Apply morphological opening to remove small noise or cleanup binary text."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    except Exception as exc:
        log.warning("Morphological cleanup failed: %s", exc)
        return image


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 0, 90, 180, or 270 degrees clockwise."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def get_ocr_candidates(image: np.ndarray) -> list[dict[str, object]]:
    """Generate multiple rotated and preprocessed candidate versions of an image.

    To ensure high performance while maintaining quality, we build options in
    decreasing order of likelihood and quality:
      1. Original image (0° rotation)
      2. Enhanced image (0° rotation, CLAHE + denoise + sharpen)
      3. Thresholded image (0° rotation, CLAHE + denoise + sharpen + adaptive threshold + morphology)
      4. Enhanced image (180° rotation)
      5. Enhanced image (90° rotation)
      6. Enhanced image (270° rotation)
      7. Thresholded image (180° rotation)
      8. Thresholded image (90° rotation)
      9. Thresholded image (270° rotation)

    Args:
        image: Original EXIF-corrected RGB numpy array.

    Returns:
        A list of dictionaries with 'desc' (description string) and 'image' (numpy array).
    """
    candidates = []

    # 1. Base 0° candidates
    candidates.append({"desc": "Original (0°)", "image": image})

    enhanced_0 = sharpen_image(denoise_image(enhance_clahe(image)))
    candidates.append({"desc": "Enhanced (0°)", "image": enhanced_0})

    thresholded_0 = morphological_cleanup(adaptive_threshold(enhanced_0))
    candidates.append({"desc": "Thresholded (0°)", "image": thresholded_0})

    # 2. Rotations for Enhanced & Thresholded (order 180°, then 90°, then 270°)
    for angle in [180, 90, 270]:
        candidates.append({
            "desc": f"Enhanced ({angle}°)",
            "image": rotate_image(enhanced_0, angle)
        })

    for angle in [180, 90, 270]:
        candidates.append({
            "desc": f"Thresholded ({angle}°)",
            "image": rotate_image(thresholded_0, angle)
        })

    return candidates

