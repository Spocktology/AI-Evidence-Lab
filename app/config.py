"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AEL_DATA_DIR", ROOT / "data")).resolve()
CASES_DIR = DATA_DIR / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("AEL_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("AEL_EFFORT", "high")
USE_FALLBACKS = os.environ.get("AEL_USE_FALLBACKS", "1") == "1"
MAX_UPLOAD_MB = int(os.environ.get("AEL_MAX_UPLOAD_MB", "4096"))

TOOL_VERSION = "1.0.0"

# --- Vision budget -----------------------------------------------------------
# Claude reads images natively up to 2576 px on the long edge and ~4784 visual
# tokens (one token per 28x28 px patch). Anything larger is downscaled by the
# API, which would throw away exactly the detail a forensic review needs. So
# every image we send is sized to fit inside this budget *ourselves*: whole
# frames are scaled down for context, and regions of interest are sent as
# native-resolution (1:1 pixel) tiles that fit the budget without resizing.
VISION_MAX_EDGE = 2576
VISION_MAX_TOKENS = 4784
PATCH = 28
# Per-image request limit is 10 MB base64; stay well under it.
MAX_IMAGE_BYTES = 7_000_000
# Hard cap on images per Claude request (keeps cost and latency predictable).
MAX_IMAGES_PER_REQUEST = 40

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif", ".bmp", ".gif", ".dng"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".3gp", ".wmv"}
