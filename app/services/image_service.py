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


# ------------------------------------------------------------------ #
# Advanced Image Preprocessing & Warping (Phase 3)                  #
# ------------------------------------------------------------------ #

def crop_and_warp(image: np.ndarray, box: list[list[float]]) -> np.ndarray:
    """Crop and apply perspective correction to warp a quadrilateral box into a flat rectangle.
    
    Args:
        image: Source RGB NumPy array.
        box: A list of 4 points: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
             ordered top-left, top-right, bottom-right, bottom-left.
             
    Returns:
        Warped crop as an RGB NumPy array.
    """
    try:
        pts = np.array(box, dtype=np.float32)
        # TL, TR, BR, BL
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

        # Calculate width
        width_a = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_b = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Calculate height
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        if max_width <= 0 or max_height <= 0:
            # Fallback to simple bounding box crop if warping calculations are invalid
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x0, x1 = int(min(x_coords)), int(max(x_coords))
            y0, y1 = int(min(y_coords)), int(max(y_coords))
            h, w = image.shape[:2]
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
            return image[y0:y1, x0:x1].copy()

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))
    except Exception as exc:
        log.warning("Perspective warp failed: %s", exc)
        # Return fallback bounding box crop
        try:
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x0, x1 = int(min(x_coords)), int(max(x_coords))
            y0, y1 = int(min(y_coords)), int(max(y_coords))
            h, w = image.shape[:2]
            return image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()
        except Exception:
            return image


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to enhance low-light or over-exposed text."""
    try:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    except Exception as exc:
        log.warning("Gamma correction failed: %s", exc)
        return image


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Normalize image brightness using min-max scaling."""
    try:
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    except Exception as exc:
        log.warning("Brightness normalization failed: %s", exc)
        return image


def upscale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Upscale image by a scale factor using bicubic interpolation."""
    if scale == 1.0:
        return image
    try:
        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception as exc:
        log.warning("Upscaling failed: %s", exc)
        return image


def apply_tophat(image: np.ndarray) -> np.ndarray:
    """Apply Morphological Top-Hat filtering to isolate raised molded text."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        enhanced = cv2.add(gray, tophat)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    except Exception as exc:
        log.warning("Top-Hat filter failed: %s", exc)
        return image


def apply_unsharp_mask(image: np.ndarray, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """Apply unsharp masking to reconstruct sharp character edges for small/blurred crops."""
    try:
        gaussian_3 = cv2.GaussianBlur(image, (3, 3), 0)
        sharp = cv2.addWeighted(image, 1.0 + amount, gaussian_3, -amount, 0)
        if threshold > 0:
            low_contrast_mask = np.abs(image - gaussian_3) < threshold
            np.copyto(sharp, image, where=low_contrast_mask)
        return sharp
    except Exception as exc:
        log.warning("Unsharp masking failed: %s", exc)
        return image


def get_advanced_enhancements(crop: np.ndarray) -> list[dict[str, object]]:
    """Generate multiple preprocessed variations of a cropped text region.
    
    Returns:
        A list of dictionaries containing the enhancement 'pipeline' description 
        and the processed crop image.
    """
    enhancements = []
    
    # 1. Original crop
    enhancements.append({"pipeline": "Original", "image": crop})

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    enhancements.append({"pipeline": "CLAHE", "image": enhance_clahe(crop)})

    # 3. Sharpened
    enhancements.append({"pipeline": "Sharpened", "image": sharpen_image(crop)})

    # 4. Top-Hat (Molded plastic raised text contrast enhancer)
    enhancements.append({"pipeline": "TopHat", "image": apply_tophat(crop)})

    # 5. Unsharp Mask (Edge sharpening)
    enhancements.append({"pipeline": "Unsharp", "image": apply_unsharp_mask(crop)})

    # 6. Adaptive Threshold (Binarized + Morphology)
    thresh = morphological_cleanup(adaptive_threshold(crop))
    enhancements.append({"pipeline": "Thresholded", "image": thresh})

    # 7. Bilateral Denoised
    enhancements.append({"pipeline": "BilateralFilter", "image": denoise_image(crop)})

    # 8. Brightness Normalized
    enhancements.append({"pipeline": "Normalised", "image": normalize_brightness(crop)})

    # 9. Gamma Corrected
    enhancements.append({"pipeline": "GammaHigh", "image": adjust_gamma(crop, 1.5)})
    enhancements.append({"pipeline": "GammaLow", "image": adjust_gamma(crop, 0.6)})

    return enhancements



