"""
Application-wide constants.
"""

# Supported image extensions (lowercase)
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# Name of the output folder created inside the selected folder
OUTPUT_FOLDER_NAME: str = "Number_Photos"

# Minimum number of characters in a text block to be considered meaningful
MIN_TEXT_LENGTH: int = 3

# Application metadata
APP_NAME: str = "OEM Part Number Detector"
APP_VERSION: str = "1.0.0 (Phase 1 MVP)"

# Logging
LOG_DATE_FORMAT: str = "%H:%M:%S"
LOG_FORMAT: str = "%(asctime)s  [%(levelname)s]  %(message)s"

# GUI
PROGRESS_BAR_MIN: int = 0
PROGRESS_BAR_MAX: int = 100

# Thread-pool
MAX_WORKERS: int = 4

# ── Phase 2 OCR & OEM Settings ──────────────────────────────────────
# Minimum confidence score returned by the OCR engine for a single text line
OCR_CONFIDENCE_THRESHOLD: float = 0.40

# Minimum overall confidence score required to classify a photo as containing an OEM part number
OEM_CONFIDENCE_THRESHOLD: float = 0.65

# Words or terms to filter out. Evaluated in a normalized form (uppercase, alphanumeric only).
BLACKLIST: list[str] = [
    "Warranty",
    "90 Days",
    "Partsut",
    "Parts Hut",
    "Left",
    "Right",
    "Germany",
    "Mercedes",
    "Made",
    "Original",
    "OEM",
    "Front",
    "Rear",
    "ABS",
    "Plastic",
]

# Regex patterns for matching common automotive part formats (after normalization)
SUPPORTED_PATTERNS: list[str] = [
    r"^[A-Z]\d{10}$",              # Mercedes-Benz (e.g. A2076800489)
    r"^A2C\d{8}$",                 # VDO/Continental (e.g. A2C53420732)
    r"^\d[A-Z0-9]{2}\d{6}[A-Z]?$", # VAG V1 (e.g. 1K0820803J, 8K0959455N, 4F0820743A)
    r"^\d{2}[A-Z]\d{6}[A-Z]?$",    # VAG V2 (e.g. 03L906023)
    r"^\d{11}$",                   # BMW 11-digit (e.g. 51717070509)
    r"^[A-Z0-9]{9,12}$"            # General fallback alphanumeric part number structure
]
