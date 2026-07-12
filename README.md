# OEM Part Number Detector — Phase 1 MVP

A production-quality Windows desktop application that scans a folder of
automotive part images, automatically corrects image orientation, performs OCR,
and copies images that contain OEM part numbers into a dedicated output folder.

---

## Features

| Feature | Details |
|---|---|
| Recursive folder scan | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| EXIF orientation fix | All 8 EXIF orientations handled automatically |
| OCR engine | PaddleOCR (English model, angle-classifier enabled) |
| Output isolation | Copies to `Number_Photos/` — originals are never modified |
| Background processing | QThread worker — GUI never freezes |
| Live progress | Real-time progress bar + log + 3 stat counters |
| Graceful cancel | Stop at any time without corrupting files |
| Dark premium UI | PySide6 + Fusion style + custom dark stylesheet |

---

## Project Structure

```
ocr/
│
├── main.py                    ← entry point  (python main.py)
├── requirements.txt
├── README.md
│
└── app/
    ├── main.py                ← real bootstrap (called by root main.py)
    │
    ├── gui/
    │   ├── main_window.py     ← MainWindow (GUI only, no business logic)
    │   └── widgets.py         ← StatCard, LogWidget, factories
    │
    ├── workers/
    │   └── folder_worker.py   ← QThread: scan → load → OCR → copy
    │
    ├── services/
    │   ├── image_service.py   ← EXIF rotation, PIL/NumPy conversion
    │   ├── ocr_service.py     ← PaddleOCR wrapper (model loaded once)
    │   └── file_service.py    ← scan, output folder, safe copy
    │
    └── utils/
        ├── constants.py       ← app-wide constants
        └── logger.py          ← centralised logging
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or newer |
| pip | 23+ recommended |
| OS | Windows 10 / 11 (64-bit) |

> **GPU (optional)** — If you have an NVIDIA GPU with CUDA 11.x+, replace
> `paddlepaddle` with `paddlepaddle-gpu` in `requirements.txt` for
> significantly faster inference.

---

## Setup Instructions

### 1. Clone / download the project

```powershell
# If using git
git clone <repository-url>
cd ocr
```

### 2. Create a virtual environment (strongly recommended)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** The first install downloads PaddlePaddle (≈250 MB) and the
> PaddleOCR English model weights (≈40 MB).  Subsequent runs use the
> locally cached weights.

### 5. Run the application

```powershell
python main.py
```

---

## Usage

1. Click **📂 Select Folder** and choose the directory containing your
   automotive part images.
2. Click **▶ Start** — processing begins immediately in the background.
3. Watch the **Processing Log**, **Progress Bar**, and the three stat
   counters update in real time.
4. When finished, all images where text was detected are copied into
   `<your-folder>/Number_Photos/`.
5. Click **✖ Cancel** at any time to stop gracefully.

---

## Output

```
<selected-folder>/
├── img001.jpg           ← original — untouched
├── img002.jpg           ← original — untouched
└── Number_Photos/
    ├── img001.jpg       ← copy of img with detected text
    └── img003.jpg       ← copy of img with detected text
```

- Originals are **never** moved, renamed, or modified.
- If a file with the same name already exists in `Number_Photos/`, a
  numeric suffix is appended (`img001_1.jpg`, `img001_2.jpg`, …).

---

## Configuration

Edit `app/utils/constants.py` to adjust behaviour without touching
business logic:

| Constant | Default | Description |
|---|---|---|
| `SUPPORTED_EXTENSIONS` | `.jpg .jpeg .png .bmp .webp` | File types to scan |
| `OUTPUT_FOLDER_NAME` | `Number_Photos` | Name of the output directory |
| `MIN_TEXT_LENGTH` | `3` | Minimum chars to count as "meaningful text" |
| `MAX_WORKERS` | `4` | Thread-pool size (reserved for Phase 2) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'paddleocr'`
Run `pip install -r requirements.txt` inside the activated virtual environment.

### Application window is blank / crashes on startup
Ensure you are using Python 3.11+ and that PySide6 is installed:
```powershell
python -c "import PySide6; print(PySide6.__version__)"
```

### OCR model download fails
PaddleOCR downloads weights on first use.  Ensure you have an internet
connection during the first launch.  After that the app works offline.

### Very slow processing
- Use a GPU: replace `paddlepaddle` with `paddlepaddle-gpu` in
  `requirements.txt`.
- Reduce image resolution before scanning (Phase 2 will add this option).

---

## Roadmap

| Phase | Features |
|---|---|
| **1 (current)** | MVP — scan, OCR, copy |
| 2 | Confidence filtering, resolution pre-processing |
| 3 | YOLO part-region detection |
| 4 | Perspective correction |
| 5 | OEM-pattern regex validation |

---

## License

MIT — see `LICENSE` for details.
