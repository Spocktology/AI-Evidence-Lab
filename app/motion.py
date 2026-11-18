"""Video motion preprocessing over the user's time range.

Works on every decoded frame in the range (subsampled only for very long
ranges) at a reduced analysis resolution, and reports in full-resolution
pixel coordinates:

    - camera motion (global shift per frame) vs. object motion
    - a motion heat-map and per-grid-cell motion energy (used to *suggest*
      cells to the user, and given to the AI)
    - a track of the most salient object inside the marked cells:
      position, apparent speed, size and brightness over time
    - brightness periodicity (blink rate). Aircraft anti-collision strobes
      flash roughly 40-100 times per minute; satellites and planets don't blink
      (though they can scintillate/flare)
    - integrity signals: duplicated frames, sudden noise-level jumps (possible
      splices), and timestamp irregularities
    - a "light trail" (max-intensity stack, camera-motion compensated) image
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .grid import Grid, col_name

ANALYSIS_WIDTH = 960
MAX_ANALYSIS_FRAMES = 900


def _read_range(video: Path, start: float, end: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video for motion analysis.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = (end - start) * fps
    stride = max(1, math.ceil(total / MAX_ANALYSIS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end + 1e-3:
            break
        if i % stride == 0:
            yield t, frame
        i += 1
    cap.release()


def _orient(frame: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    """OpenCV may or may not apply rotation metadata; match the ffmpeg-extracted
    frames' orientation (full_w x full_h)."""
    h, w = frame.shape[:2]
    if (w > h) != (full_w > full_h) and w != h:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _global_motion(prev_u8: np.ndarray, cur_u8: np.ndarray, mask: np.ndarray, win) -> tuple[np.ndarray, str]:
    """Estimate camera motion between two frames as a 2x3 similarity transform.

    Primary method: track background corner features (Lucas-Kanade) with the
    user-marked object masked out, and fit with RANSAC so moving objects are
    rejected as outliers. Fallback for low-texture scenes (night sky, fog):
    phase correlation on the masked frame with bright point sources clipped,
    accepted only if the correlation peak is strong. Otherwise report
    'unreliable' and assume no camera motion, rather than mistaking the
    object's own motion for camera motion."""
    pts = cv2.goodFeaturesToTrack(prev_u8, maxCorners=500, qualityLevel=0.005, minDistance=7, mask=mask)
    if pts is not None and len(pts) >= 15:
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(prev_u8, cur_u8, pts, None, winSize=(21, 21), maxLevel=3)
        good = st.ravel() == 1
        if good.sum() >= 12:
            m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5, maxIters=2000)
            if m is not None and inl is not None and inl.sum() >= max(10, 0.4 * good.sum()):
                return m, "features"
    p = prev_u8.astype(np.float32)
    c = cur_u8.astype(np.float32)
    hi = float(np.percentile(p, 98))
    fill = float(np.median(p))
    p = np.where(mask > 0, np.minimum(p, hi), fill).astype(np.float32)
    c = np.where(mask > 0, np.minimum(c, hi), fill).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(p, c, win)
    if resp > 0.25:
        return np.float32([[1, 0, dx], [0, 1, dy]]), "phase"
    return np.float32([[1, 0, 0], [0, 1, 0]]), "unreliable"


def _h(m: np.ndarray) -> np.ndarray:
    return np.vstack([m, [0, 0, 1]]).astype(np.float64)


def analyze_motion(video: Path, start: float, end: float, full_w: int, full_h: int, grid: Grid,
                   selected: list[str], out_dir: Path, deg_per_px: float | None = None) -> dict[str, Any]:
    scale = min(1.0, ANALYSIS_WIDTH / max(full_w, full_h))
    aw, ah = int(full_w * scale), int(full_h * scale)
    inv = 1.0 / scale
    roi = None
    mask = np.full((ah, aw), 255, np.uint8)
    if selected:
        x0, y0, x1, y1 = grid.union_box(selected)
        roi = (int(x0 * scale), int(y0 * scale), max(int(x0 * scale) + 2, int(x1 * scale)),
               max(int(y0 * scale) + 2, int(y1 * scale)))
        mask[roi[1]:roi[3], roi[0]:roi[2]] = 0  # exclude the object from camera-motion estimation

    prev_u8 = None
    win = cv2.createHanningWindow((aw, ah), cv2.CV_32F)
    heat = np.zeros((ah, aw), np.float32)
    stack = None
    cum = np.eye(3)  # maps first-frame coordinates -> current-frame coordinates
    times, shifts, methods, noise_levels, dup_flags, track, peaks = [], [], [], [], [], [], []
    first_rgb = None
    for t, frame in _read_range(video, start, end):
        frame = _orient(frame, full_w, full_h)
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g32 = gray.astype(np.float32)
        if first_rgb is None:
            first_rgb = rgb
        times.append(t)
        # Robust (MAD) noise sigma: small bright objects don't move it.
        res = g32 - cv2.medianBlur(gray, 3).astype(np.float32)
        noise_levels.append(float(np.median(np.abs(res)) * 1.4826))
        if prev_u8 is not None:
            m, how = _global_motion(prev_u8, gray, mask, win)
            methods.append(how)
            cum = _h(m) @ cum
            aligned_prev = cv2.warpAffine(prev_u8, m, (aw, ah), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            heat += np.abs(g32 - aligned_prev)
            d = np.abs(g32 - prev_u8.astype(np.float32))
            dup_flags.append(float(d.mean()) < 0.02 and float(d.max()) <= 2)
            shifts.append((float(m[0, 2]), float(m[1, 2])))
        to_first = np.linalg.inv(cum)[:2].astype(np.float32)
        reg = cv2.warpAffine(rgb, to_first, (aw, ah), borderMode=cv2.BORDER_CONSTANT)
        stack = reg if stack is None else np.maximum(stack, reg)
        if roi is not None:
            pt, peak = _track_point(g32, roi, t, np.linalg.inv(cum), inv)
            track.append(pt)
            peaks.append((t, peak))
        prev_u8 = gray

    if len(times) < 2:
        return {"error": "Too few frames could be decoded in this range for motion analysis."}

    out: dict[str, Any] = {"frames_analyzed": len(times), "analysis_scale": round(scale, 4),
                           "range_s": [round(times[0], 3), round(times[-1], 3)]}

    # --- timing integrity -----------------------------------------------------
    dts = np.diff(times)
    med_dt = float(np.median(dts)) if dts.size else 0
    out["timing"] = {
        "median_frame_interval_s": round(med_dt, 5),
        "effective_fps": round(1 / med_dt, 2) if med_dt > 0 else None,
        "irregular_intervals": int((np.abs(dts - med_dt) > med_dt * 0.5).sum()) if med_dt > 0 else 0,
        "identical_consecutive_frames": int(sum(dup_flags)),
        "identical_note": "Pixel-identical consecutive frames suggest frame-rate conversion, a re-encode, a "
                          "freeze/dropped-frame fill, or editing. Some security cameras do this at low light.",
    }
    nl = np.array(noise_levels)
    jumps = np.abs(np.diff(nl))
    thr = max(float(np.median(jumps)) * 6, float(nl.mean()) * 0.25, 0.3) if jumps.size else 0
    jump_times = [round(times[i + 1], 3) for i in np.where(jumps > thr)[0]][:10]
    out["noise_continuity"] = {
        "mean_noise_sigma": round(float(nl.mean()), 3),
        "sudden_jumps_at_s": jump_times,
        "note": "Sudden changes in sensor-noise level between consecutive frames can mark a cut/splice or "
                "an exposure/ISO change by the camera. Compare with what happens on screen at those times.",
    }

    # --- camera motion ---------------------------------------------------------
    sh = np.array(shifts) * inv
    mag = np.hypot(sh[:, 0], sh[:, 1])
    unreliable = methods.count("unreliable") / max(len(methods), 1)
    out["camera_motion"] = {
        "mean_shift_px_per_frame": round(float(mag.mean()), 2),
        "max_shift_px_per_frame": round(float(mag.max()), 2),
        "total_pan_px": [round(float(cum[0, 2] * inv), 1), round(float(cum[1, 2] * inv), 1)],
        "estimation_method_counts": {k: methods.count(k) for k in set(methods)},
        "interpretation": ("Camera motion could NOT be measured reliably for most frames (scene lacks texture, "
                           "e.g. night sky). Object motion below is relative to the frame, not the scene."
                           if unreliable > 0.5 else
                           "Camera essentially static." if mag.mean() < 0.5 else
                           "Handheld shake / small movements." if mag.mean() < 4 else
                           "Significant panning or camera movement."),
    }

    # --- motion heat map + per-cell energy -------------------------------------
    heat /= max(len(times) - 1, 1)
    hm = heat / (np.percentile(heat, 99.7) or 1)
    hm_img = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blend = cv2.addWeighted(cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR), 0.45, hm_img, 0.75, 0)
    Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)).save(out_dir / "motion_heatmap.png")
    cell_energy = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            x0, y0, x1, y1 = grid.cell_box(c, r)
            sub = heat[int(y0 * scale):max(int(y0 * scale) + 1, int(y1 * scale)),
                       int(x0 * scale):max(int(x0 * scale) + 1, int(x1 * scale))]
            # 99th percentile, not mean: a small moving light occupies few pixels of a cell.
            cell_energy[f"{col_name(c)}{r + 1}"] = float(np.percentile(sub, 99))
    vals = np.array(list(cell_energy.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    ranked = sorted(cell_energy.items(), key=lambda kv: -kv[1])
    out["suggested_cells"] = [k for k, v in ranked if (v - med) / mad > 5 and v > med + 4][:12]
    out["top_motion_cells"] = [{"cell": k, "energy": round(v, 2)} for k, v in ranked[:8]]
    Image.fromarray(stack).save(out_dir / "light_trail_stack.png")
    out["images"] = {"heatmap": "motion_heatmap.png", "light_trail": "light_trail_stack.png"}

    # --- object track ----------------------------------------------------------
    if roi is not None:
        pts = [p for p in track if p]
        out["object_track"] = _summarize_track(pts, deg_per_px, peaks)
        if any(p.get("found") for p in pts):
            _draw_track(first_rgb, pts, scale, roi, out_dir / "object_track.png")
            out["images"]["track"] = "object_track.png"
    return out


def _track_point(g32: np.ndarray, roi, t: float, to_first: np.ndarray, inv: float):
    """Most salient blob in the ROI. Returns (point, peak) where peak is the
    ROI's strongest deviation from local background in every frame (used for
    blink analysis even when the light is off and no blob is found)."""
    x0, y0, x1, y1 = roi
    sub = g32[y0:y1, x0:x1]
    if sub.size < 16:
        return None, 0.0
    k = 15 if min(sub.shape) > 30 else 3
    bg = cv2.medianBlur(sub.astype(np.uint8), k).astype(np.float32)
    dev = sub - bg
    peak = float(np.percentile(dev, 99.95))
    adev = np.abs(dev)
    med = float(np.median(adev))
    mad = float(np.median(np.abs(adev - med))) * 1.4826 + 1e-3
    mask = (adev > med + max(6 * mad, 10)).astype(np.uint8)
    n, labels, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return {"t": round(t, 4), "found": False}, peak
    scores = [st[i, cv2.CC_STAT_AREA] * adev[labels == i].mean() for i in range(1, n)]
    i = 1 + int(np.argmax(scores))
    cx, cy = cent[i]
    px, py = x0 + cx, y0 + cy
    scene = to_first @ np.array([px, py, 1.0])
    obj = labels == i
    return {
        "t": round(t, 4), "found": True,
        "x": round(px * inv, 1), "y": round(py * inv, 1),
        # position in first-frame (scene) coordinates, camera motion removed
        "sx": round(float(scene[0]) * inv, 1), "sy": round(float(scene[1]) * inv, 1),
        "area": round(float(st[i, cv2.CC_STAT_AREA]) * inv * inv, 1),
        "brightness": round(float(sub[obj].mean()), 1),
    }, peak


def _summarize_track(pts: list[dict], deg_per_px: float | None, peaks: list[tuple[float, float]]) -> dict[str, Any]:
    found = [p for p in pts if p.get("found")]
    res: dict[str, Any] = {"frames": len(pts), "frames_with_object": len(found)}
    if len(found) < 3:
        res["note"] = "Object could not be followed reliably in the marked cells (too faint, too small, or it left the region)."
        return res
    t = np.array([p["t"] for p in found])
    sx = np.array([p["sx"] for p in found])
    sy = np.array([p["sy"] for p in found])
    b = np.array([p["brightness"] for p in found])
    a = np.array([p["area"] for p in found])
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    v = np.hypot(np.diff(sx), np.diff(sy)) / dt
    v = v[np.isfinite(v)]
    res.update({
        "duration_tracked_s": round(float(t[-1] - t[0]), 3),
        "scene_displacement_px": [round(float(sx[-1] - sx[0]), 1), round(float(sy[-1] - sy[0]), 1)],
        "apparent_speed_px_per_s": {"median": round(float(np.median(v)), 1) if v.size else None,
                                    "max": round(float(np.max(v)), 1) if v.size else None},
        "path_straightness": _straightness(sx, sy),
        "size_px_area": {"min": round(float(a.min()), 1), "median": round(float(np.median(a)), 1), "max": round(float(a.max()), 1)},
        "brightness": {"min": round(float(b.min()), 1), "max": round(float(b.max()), 1), "std": round(float(b.std()), 2)},
        "blink": _blink(np.array([p[0] for p in peaks]), np.array([p[1] for p in peaks])),
        "samples": [{k: p[k] for k in ("t", "x", "y", "brightness")} for p in found[:: max(1, len(found) // 24)]],
    })
    if deg_per_px and v.size:
        res["apparent_angular_speed_deg_per_s"] = round(float(np.median(v)) * deg_per_px, 4)
    if len(found) < len(pts) * 0.6:
        res["note"] = "Object detection was intermittent; it may blink, be obscured, or be near the noise floor."
    return res


def _straightness(x: np.ndarray, y: np.ndarray) -> float | None:
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    if path < 3:
        return None
    return round(float(math.hypot(x[-1] - x[0], y[-1] - y[0]) / path), 3)


def _blink(t: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if len(t) < 16 or t[-1] - t[0] < 1.5:
        return {"detected": False, "note": "Range too short to measure blink rate."}
    grid_t = np.linspace(t[0], t[-1], len(t))
    bi = np.interp(grid_t, t, b)
    bi = bi - bi.mean()
    if bi.std() < 3:
        return {"detected": False, "note": "Brightness essentially steady."}
    fs = 1 / (grid_t[1] - grid_t[0])
    spec = np.abs(np.fft.rfft(bi * np.hanning(len(bi))))
    freqs = np.fft.rfftfreq(len(bi), 1 / fs)
    band = (freqs > 0.2) & (freqs < min(8, fs / 2))
    if not band.any():
        return {"detected": False}
    k = int(np.argmax(np.where(band, spec, 0)))
    strength = float(spec[k] / (np.median(spec[band]) + 1e-6))
    detected = strength > 4
    return {"detected": bool(detected), "frequency_hz": round(float(freqs[k]), 2),
            "flashes_per_minute": round(float(freqs[k] * 60), 1), "peak_strength": round(strength, 1),
            "note": "Aircraft anti-collision strobes typically flash ~40-100 times/min. Rolling-shutter/LED "
                    "flicker can alias into apparent blinking."}


def _draw_track(rgb: np.ndarray, pts: list[dict], scale: float, roi, path: Path) -> None:
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle(roi, outline=(255, 60, 60), width=2)
    xy = [(p["x"] * scale, p["y"] * scale) for p in pts if p.get("found")]
    if len(xy) > 1:
        d.line(xy, fill=(0, 255, 255), width=2)
    for x, y in xy[:: max(1, len(xy) // 30)]:
        d.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(255, 255, 0))
    im.save(path)
