"""Pixel-level forensic preprocessing.

Everything here is measurement, not judgement: numbers and visual maps that
are handed to the AI (and printed in the report) so its conclusions rest on
evidence it could not reliably see by eye, especially in downscaled views.

    - Error Level Analysis (JPEG re-save differences)
    - Noise residual map + per-cell noise consistency
    - Frequency spectrum of the noise residual (periodic upsampling patterns
      common in AI generators and resampled/composited regions)
    - JPEG quantization tables / estimated quality / standard-vs-custom tables
    - Exposure statistics (clipping, dynamic range)
    - Per-grid-cell statistics comparing the marked region with the rest
    - Measurement of the most salient object inside the marked region
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .grid import Grid, col_name

# Standard IJG (libjpeg) luminance/chrominance tables (natural order).
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
_STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32, dtype=np.float64)


def _scaled_table(std: np.ndarray, q: int) -> np.ndarray:
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((std * scale + 50) / 100), 1, 255)


def jpeg_analysis(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            tables = getattr(im, "quantization", None) or {}
            info = {
                "progressive": bool(im.info.get("progressive") or im.info.get("progression")),
                "num_quant_tables": len(tables),
            }
            try:
                from PIL import JpegImagePlugin

                info["chroma_subsampling"] = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(
                    JpegImagePlugin.get_sampling(im), "unknown")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return None
    if not tables:
        return info
    luma = np.sort(np.array(tables[0], dtype=np.float64))
    best_q, best_err = 0, 1e9
    for q in range(1, 101):
        err = np.abs(np.sort(_scaled_table(_STD_LUMA, q)) - luma).mean()
        if err < best_err:
            best_q, best_err = q, err
    info["estimated_quality"] = best_q
    info["standard_ijg_tables"] = bool(best_err < 0.01)
    info["table_match_error"] = round(float(best_err), 3)
    info["interpretation"] = (
        "Quantization tables exactly match the standard libjpeg/IJG tables at quality "
        f"{best_q}. Many editing tools, websites and libraries use these; many camera makers use their own "
        "tables, so an exact match suggests (not proves) the file was re-saved by software."
        if info["standard_ijg_tables"] else
        f"Custom (non-IJG) quantization tables, closest to IJG quality ~{best_q}. Custom tables are typical "
        "of camera firmware and some apps."
    )
    return info


def ela_map(im: Image.Image, quality: int = 90) -> tuple[np.ndarray, float]:
    """Error Level Analysis: re-save at a known quality and look at differences.
    Regions edited after the last save often re-compress differently."""
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    orig = np.asarray(im.convert("RGB"), dtype=np.int16)
    diff = np.abs(orig - resaved).max(axis=2).astype(np.float32)
    return diff, float(diff.mean())


def noise_residual(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return g - cv2.medianBlur(gray, 3).astype(np.float32)


def local_std(x: np.ndarray, k: int = 16) -> np.ndarray:
    mean = cv2.boxFilter(x, -1, (k, k))
    sq = cv2.boxFilter(x * x, -1, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def spectrum(residual: np.ndarray, size: int = 1024) -> tuple[np.ndarray, dict[str, float]]:
    """Log-magnitude spectrum of the noise residual (centre crop). Natural
    camera noise gives a fairly smooth spectrum; generators and resampling
    often leave sharp periodic peaks (a regular dot/star pattern)."""
    h, w = residual.shape
    s = min(size, h, w)
    s -= s % 2
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = residual[y0:y0 + s, x0:x0 + s]
    win = np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(patch * win))
    mag = np.log1p(np.abs(f))
    c = s // 2
    m = mag.copy()
    m[c - 3:c + 4, :] = 0  # ignore DC cross / axes (from image borders and edges)
    m[:, c - 3:c + 4] = 0
    yy, xx = np.ogrid[:s, :s]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    m[r < s * 0.05] = 0
    body = m[m > 0]
    med = float(np.median(body)) if body.size else 1.0
    mad = float(np.median(np.abs(body - med))) if body.size else 1.0
    peaks = int(((m - med) / max(mad, 1e-6) > 12).sum())
    score = float(((m.max() - med) / max(mad, 1e-6))) if body.size else 0.0
    vis = mag - mag.min()
    vis = (vis / max(vis.max(), 1e-6) * 255).astype(np.uint8)
    return vis, {"periodic_peak_score": round(score, 1), "strong_peak_count": peaks,
                 "note": "Higher scores / many isolated peaks = periodic structure in the noise. Typical "
                         "real camera images score lower; resampling, AI generation, screen moire and "
                         "heavy compression score higher. Treat as one indicator only."}


def exposure_stats(arr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return {
        "mean_luminance": round(float(gray.mean()), 1),
        "pct_clipped_highlights": round(float((gray >= 250).mean() * 100), 2),
        "pct_crushed_shadows": round(float((gray <= 5).mean() * 100), 2),
        "luminance_p1_p99": [int(np.percentile(gray, 1)), int(np.percentile(gray, 99))],
        "mean_saturation": round(float(cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 1].mean()), 1),
    }


def cell_stats(arr: np.ndarray, grid: Grid, ela: np.ndarray | None) -> dict[str, dict[str, float]]:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    res = noise_residual(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    out = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            x0, y0, x1, y1 = grid.cell_box(c, r)
            name = f"{col_name(c)}{r + 1}"
            g = gray[y0:y1, x0:x1]
            # Noise measured only in flat areas (edges would dominate otherwise).
            rr = res[y0:y1, x0:x1]
            flat = np.abs(lap[y0:y1, x0:x1]) < 8
            noise = float(rr[flat].std()) if flat.sum() > 100 else float(rr.std())
            out[name] = {
                "luminance": round(float(g.mean()), 1),
                "noise": round(noise, 3),
                "sharpness": round(float(lap[y0:y1, x0:x1].var()), 1),
                "ela": round(float(ela[y0:y1, x0:x1].mean()), 3) if ela is not None else None,
            }
    return out


def compare_cells(stats: dict[str, dict[str, float]], selected: list[str]) -> dict[str, Any]:
    """How unusual are the marked cells compared with the whole frame?
    Robust z-scores (median/MAD). |z| > 3 is noteworthy, but context matters:
    a bright object on a dark sky will naturally differ in sharpness/ELA."""
    keys = ["noise", "sharpness", "ela", "luminance"]
    result: dict[str, Any] = {"cells": {}, "summary": {}}
    for k in keys:
        vals = np.array([s[k] for s in stats.values() if s.get(k) is not None], dtype=np.float64)
        if vals.size < 4:
            continue
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) * 1.4826 or (vals.std() or 1.0)
        zs = []
        for cell in selected:
            if cell in stats and stats[cell].get(k) is not None:
                z = (stats[cell][k] - med) / mad
                result["cells"].setdefault(cell, {})[f"{k}_z"] = round(float(z), 2)
                zs.append(z)
        if zs:
            result["summary"][k] = {"frame_median": round(float(med), 3),
                                    "selected_mean_z": round(float(np.mean(zs)), 2),
                                    "selected_max_abs_z": round(float(np.max(np.abs(zs))), 2)}
    return result


def degrees_per_pixel(facts: dict[str, Any], width: int, height: int) -> float | None:
    """Approximate angular resolution from the 35 mm-equivalent focal length
    (defined on the 43.27 mm full-frame diagonal)."""
    f35 = facts.get("focal_length_35mm")
    if not f35 or f35 <= 0:
        return None
    dfov = 2 * math.degrees(math.atan(43.27 / (2 * f35)))
    return dfov / math.hypot(width, height)


def measure_object(arr: np.ndarray, box: tuple[int, int, int, int], deg_per_px: float | None) -> dict[str, Any] | None:
    """Find the most salient blob inside the marked region and measure it."""
    x0, y0, x1, y1 = box
    roi = arr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.medianBlur(gray.astype(np.uint8), 21 if min(gray.shape) > 42 else 3).astype(np.float32)
    dev = np.abs(gray - bg)
    med = float(np.median(dev))
    mad = float(np.median(np.abs(dev - med))) * 1.4826 + 1e-3
    mask = (dev > med + max(6 * mad, 12)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return {"found": False, "note": "No distinct object stands out from the local background in the marked region."}
    idx = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA] * [dev[labels == i].mean() for i in range(1, n)]))
    bx, by, bw, bh, area = st[idx]
    obj = labels == idx
    colour = roi[obj].mean(axis=0)
    # Edge sharpness: gradient magnitude along the blob boundary vs ROI typical.
    gm = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edge = cv2.dilate(obj.astype(np.uint8), np.ones((3, 3), np.uint8)) - cv2.erode(obj.astype(np.uint8), np.ones((3, 3), np.uint8))
    edge_grad = float(gm[edge > 0].mean()) if edge.any() else 0.0
    out = {
        "found": True,
        "bbox_px_full_frame": [int(x0 + bx), int(y0 + by), int(bw), int(bh)],
        "centroid_px_full_frame": [round(float(x0 + cent[idx][0]), 1), round(float(y0 + cent[idx][1]), 1)],
        "area_px": int(area),
        "size_px": [int(bw), int(bh)],
        "mean_rgb": [int(v) for v in colour],
        "peak_luminance": int(gray[obj].max()),
        "contrast_vs_background": round(float((gray[obj] - bg[obj]).mean()), 1),
        "boundary_gradient": round(edge_grad, 1),
        "roi_median_gradient": round(float(np.median(gm)), 1),
        "other_candidate_blobs": int(n - 2),
        "saturated_core": bool((gray[obj] >= 250).mean() > 0.2),
    }
    if deg_per_px:
        out["approx_angular_size_deg"] = [round(bw * deg_per_px, 4), round(bh * deg_per_px, 4)]
        out["angular_size_note"] = ("From the 35mm-equivalent focal length. For reference the full moon is ~0.5 deg. "
                                    "Blurred/overexposed point sources appear larger than they are.")
    if area < 30:
        out["note"] = "Very small object (< 30 pixels): shape detail is not reliable at this size."
    return out


def to_rgb_map(x: np.ndarray, gain: float | None = None, colormap: int = cv2.COLORMAP_INFERNO) -> Image.Image:
    x = x.astype(np.float32)
    if gain is None:
        hi = float(np.percentile(x, 99.5)) or 1.0
        x = x / hi
    else:
        x = x * gain / 255.0
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(u8, colormap), cv2.COLOR_BGR2RGB))


def analyze_image(im: Image.Image, path: Path | None, grid: Grid, selected: list[str],
                  facts: dict[str, Any], out_dir: Path, prefix: str) -> dict[str, Any]:
    """Run the full forensic battery on one still/frame. Writes visual maps
    into out_dir and returns the numeric results."""
    arr = np.asarray(im.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    result: dict[str, Any] = {"exposure": exposure_stats(arr)}

    is_jpeg = path is not None and path.suffix.lower() in (".jpg", ".jpeg")
    ela, ela_mean = ela_map(im)
    result["ela"] = {"mean": round(ela_mean, 3),
                     "applicable": bool(is_jpeg),
                     "note": "ELA is most meaningful on JPEGs. On PNG/video frames it mostly shows edges and texture."}
    to_rgb_map(ela).save(out_dir / f"{prefix}_ela.png")

    res = noise_residual(gray)
    nmap = local_std(res)
    to_rgb_map(nmap, colormap=cv2.COLORMAP_VIRIDIS).save(out_dir / f"{prefix}_noise.png")
    spec, spec_stats = spectrum(res)
    Image.fromarray(spec).save(out_dir / f"{prefix}_fft.png")
    result["spectrum"] = spec_stats

    stats = cell_stats(arr, grid, ela)
    result["cell_comparison"] = compare_cells(stats, selected) if selected else None
    if selected:
        box = grid.union_box(selected)
        dpp = degrees_per_pixel(facts, im.width, im.height)
        result["object_measurement"] = measure_object(arr, box, dpp)
    result["maps"] = {"ela": f"{prefix}_ela.png", "noise": f"{prefix}_noise.png", "fft": f"{prefix}_fft.png"}
    return result
