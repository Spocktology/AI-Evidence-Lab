"""Loading images and pulling frames from video, always at full resolution."""
from __future__ import annotations

import io
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import MAX_IMAGE_BYTES, PATCH, VISION_MAX_EDGE, VISION_MAX_TOKENS

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = 400_000_000  # allow very large stills (panoramas, medium format)


def load_image(path: Path) -> Image.Image:
    """Open an image as 8-bit RGB with EXIF orientation applied (what the viewer saw)."""
    im = Image.open(path)
    try:
        im.seek(0)
    except EOFError:
        pass
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = (arr / max(arr.max(), 1) * 255).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im


def to_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"))


# ----------------------------------------------------------------------------
# Vision budget helpers
# ----------------------------------------------------------------------------

def vision_tokens(w: int, h: int) -> int:
    return math.ceil(w / PATCH) * math.ceil(h / PATCH)


def fits_budget(w: int, h: int) -> bool:
    return max(w, h) <= VISION_MAX_EDGE and vision_tokens(w, h) <= VISION_MAX_TOKENS


def fit_to_budget(im: Image.Image) -> tuple[Image.Image, float]:
    """Downscale (never upscale) so Claude sees the image without further resizing."""
    w, h = im.size
    if fits_budget(w, h):
        return im, 1.0
    s = min(VISION_MAX_EDGE / max(w, h), math.sqrt(VISION_MAX_TOKENS * PATCH * PATCH / (w * h)))
    while True:
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        if fits_budget(nw, nh):
            return im.resize((nw, nh), Image.LANCZOS), nw / w
        s *= 0.98


def max_native_tile() -> int:
    """Largest square tile that fits the budget at 1:1 pixels."""
    side = int(math.sqrt(VISION_MAX_TOKENS)) * PATCH
    while vision_tokens(side, side) > VISION_MAX_TOKENS:
        side -= PATCH
    return min(side, VISION_MAX_EDGE)


def encode_for_claude(im: Image.Image, lossless: bool = True) -> tuple[bytes, str]:
    """Encode losslessly where possible so we never add compression artifacts
    the AI could mistake for evidence."""
    if lossless:
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=False, compress_level=6)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return buf.getvalue(), "image/png"
        buf = io.BytesIO()
        im.save(buf, "WEBP", lossless=True, method=4)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return buf.getvalue(), "image/webp"
    for q in (97, 93, 90, 85):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, subsampling=0)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return buf.getvalue(), "image/jpeg"
    return buf.getvalue(), "image/jpeg"


def save_preview(im: Image.Image, path: Path, max_edge: int = 1600, quality: int = 88) -> tuple[int, int]:
    """A display-only JPEG for the browser UI and the report (not used for analysis)."""
    p = im.copy()
    p.thumbnail((max_edge, max_edge), Image.LANCZOS)
    p.save(path, "JPEG", quality=quality)
    return p.size


# ----------------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------------

def extract_frame(video: Path, t: float, out_png: Path) -> bool:
    """Extract the frame at time t (seconds) as a lossless full-resolution PNG.
    ffmpeg applies the stream's rotation so the frame matches what viewers see."""
    if not shutil.which("ffmpeg"):
        return _extract_frame_cv2(video, t, out_png)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(t, 0):.4f}", "-i", str(video),
           "-frames:v", "1", "-an", "-pix_fmt", "rgb24", str(out_png)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out_png.exists() and out_png.stat().st_size > 0


def _extract_frame_cv2(video: Path, t: float, out_png: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return bool(cv2.imwrite(str(out_png), frame))


def make_browser_proxy(video: Path, out_mp4: Path, codec: str | None) -> Path:
    """Browsers can't play every codec (HEVC, ProRes, ...). Make an H.264 copy
    for on-screen playback only. All analysis uses the original."""
    if codec in ("h264", "vp8", "vp9", "av1") and video.suffix.lower() in (".mp4", ".m4v", ".webm", ".mov"):
        return video
    if out_mp4.exists():
        return out_mp4
    if not shutil.which("ffmpeg"):
        return video
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", "scale='min(1920,iw)':-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-movflags", "+faststart", str(out_mp4)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return video
    return out_mp4 if out_mp4.exists() else video


def frame_times(start: float, end: float, count: int, fps: float | None) -> list[float]:
    """Evenly spaced times in [start, end]. If the range holds fewer frames than
    requested, return every frame in the range."""
    start, end = max(0.0, start), max(start, end)
    if fps and fps > 0:
        n_in_range = int((end - start) * fps) + 1
        if n_in_range <= count:
            return [round(start + i / fps, 4) for i in range(n_in_range)]
    if count <= 1 or end <= start:
        return [round(start, 4)]
    step = (end - start) / (count - 1)
    return [round(start + i * step, 4) for i in range(count)]
