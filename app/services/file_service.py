"""
file_service.py
---------------
Handles all filesystem operations:
  - Recursive image discovery
  - Output-folder creation
  - Safe file copying (never moves or renames originals)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.utils.logger import get_logger
from app.utils.constants import SUPPORTED_EXTENSIONS, OUTPUT_FOLDER_NAME

log = get_logger(__name__)


class FileService:
    """Provides pure filesystem utilities with no OCR or image dependencies."""

    # ---------------------------------------------------------------- #
    # Discovery                                                         #
    # ---------------------------------------------------------------- #

    @staticmethod
    def scan_images(root_folder: Path) -> list[Path]:
        """Recursively walk *root_folder* and collect all supported image files.

        Files located inside the ``Number_Photos`` output folder are excluded
        to avoid re-processing previously copied results.

        Args:
            root_folder: Directory to scan.

        Returns:
            Sorted list of :class:`~pathlib.Path` objects for each found image.
        """
        output_folder = root_folder / OUTPUT_FOLDER_NAME
        images: list[Path] = []

        for ext in SUPPORTED_EXTENSIONS:
            # rglob matches both lower- and upper-case on case-insensitive FSes;
            # we apply our own case-fold below for cross-platform safety.
            for path in root_folder.rglob(f"*{ext}"):
                if output_folder in path.parents or path.parent == output_folder:
                    continue
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    images.append(path)

        # Deduplicate (rglob may yield duplicates on some platforms)
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in images:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        unique.sort()
        log.debug("Found %d images in '%s'", len(unique), root_folder)
        return unique

    # ---------------------------------------------------------------- #
    # Output folder                                                     #
    # ---------------------------------------------------------------- #

    @staticmethod
    def ensure_output_folder(root_folder: Path) -> Path:
        """Create the ``Number_Photos`` directory inside *root_folder* if it
        does not already exist, then return its :class:`~pathlib.Path`.

        Args:
            root_folder: The user-selected source directory.

        Returns:
            :class:`~pathlib.Path` pointing to the output folder.

        Raises:
            OSError: If the directory cannot be created.
        """
        output_folder = root_folder / OUTPUT_FOLDER_NAME
        output_folder.mkdir(parents=True, exist_ok=True)
        log.debug("Output folder ready: '%s'", output_folder)
        return output_folder

    # ---------------------------------------------------------------- #
    # Copying                                                           #
    # ---------------------------------------------------------------- #

    @staticmethod
    def copy_image(source: Path, destination_folder: Path) -> bool:
        """Copy *source* into *destination_folder* using a collision-safe name.

        If a file with the same name already exists in *destination_folder*,
        a numeric suffix is appended (e.g. ``image_1.jpg``, ``image_2.jpg``).

        The original file is **never** moved, renamed, or altered.

        Args:
            source: Path to the original image file.
            destination_folder: Directory into which the copy is placed.

        Returns:
            ``True`` if the copy succeeded, ``False`` otherwise.
        """
        try:
            dest_path = FileService._safe_destination(source, destination_folder)
            shutil.copy2(source, dest_path)   # copy2 preserves metadata
            log.debug("Copied '%s' → '%s'", source.name, dest_path)
            return True
        except (OSError, shutil.Error) as exc:
            log.error("Failed to copy '%s': %s", source, exc)
            return False

    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _safe_destination(source: Path, destination_folder: Path) -> Path:
        """Return a unique destination path, appending a counter if necessary.

        Args:
            source: Original file path (used for stem and suffix).
            destination_folder: Target directory.

        Returns:
            A :class:`~pathlib.Path` that does not yet exist on disk.
        """
        candidate = destination_folder / source.name
        if not candidate.exists():
            return candidate

        stem = source.stem
        suffix = source.suffix
        counter = 1
        while True:
            candidate = destination_folder / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
