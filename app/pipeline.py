"""Intake, preprocessing and the two-pass AI analysis."""
from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from . import cases, config, forensics, metadata, motion
from .ai import AnalysisError, ImageItem, build_content, call_claude
from .grid import Grid, draw_grid, enhance, native_tiles, pixel_zoom
from .media import (extract_frame, fit_to_budget, frame_times, load_image, make_browser_proxy,
                    save_preview)
from .prompts import (FINAL_SCHEMA, PASS1_INSTRUCTIONS, PASS1_SCHEMA, PASS2_INSTRUCTIONS, SYSTEM_PROMPT,
                      TECHNICAL_CONTEXT_KEYS, WITNESS_KEYS)

DEFAULT_GRID_COLS = 8
MAX_VIDEO_FRAMES = 24
MAX_ROI_FRAMES = 16


# ----------------------------------------------------------------------------
# Intake
# ----------------------------------------------------------------------------

def intake(upload_path: Path, original_name: str, title: str | None) -> dict[str, Any]:
    ext = Path(original_name).suffix.lower()
    if ext in config.IMAGE_EXTS:
        kind = "image"
    elif ext in config.VIDEO_EXTS:
        kind = "video"
    else:
        raise ValueError(f"Unsupported file type '{ext}'.")
    case_id = cases.new_case_id()
    d = cases.create_case_dir(case_id)
    stored = cases.safe_filename(original_name)
    dest = d / "original" / stored
    shutil.move(str(upload_path), dest)
    hashes = cases.hash_file(dest)
    cases.make_read_only(dest)
    case: dict[str, Any] = {
        "id": case_id, "title": title or Path(original_name).stem, "filename": original_name,
        "stored_name": stored, "kind": kind, "size_bytes": dest.stat().st_size, "hashes": hashes,
        "created_at": cases.now_iso(), "updated_at": cases.now_iso(), "tool_version": config.TOOL_VERSION,
        "context": {}, "selection": None, "frames": [], "analysis": {"status": "new"},
    }
    cases.save(case)
    cases.log(case_id, "INTAKE", f"file={original_name} bytes={case['size_bytes']} sha256={hashes['sha256']} md5={hashes['md5']}")

    exif = metadata.run_exiftool(dest, kind == "video")
    source = "exiftool"
    if exif is None:
        exif = metadata.run_pillow_exif(dest) if kind == "image" else {}
        source = "pillow" if kind == "image" else "none"
    probe = metadata.run_ffprobe(dest) if kind == "video" else None
    facts = metadata.normalize(exif, probe, kind)
    c2pa = metadata.read_c2pa(dest)
    markers = metadata.scan_markers(dest, json.dumps(exif, default=str))
    flags = metadata.consistency_checks(facts, exif, markers, c2pa, kind, ext)
    derived = d / "derived"

    if kind == "image":
        im = load_image(dest)
        facts["width"], facts["height"] = im.width, im.height
        pw, ph = save_preview(im, derived / "preview.jpg", max_edge=2048)
        case["preview"] = {"path": "derived/preview.jpg", "width": pw, "height": ph}
        case["jpeg"] = forensics.jpeg_analysis(dest)
    else:
        proxy = make_browser_proxy(dest, derived / "proxy.mp4", facts.get("codec"))
        case["playback"] = "original" if proxy == dest else "derived/proxy.mp4"
        if proxy != dest:
            cases.log(case_id, "PROXY", "Created H.264 playback copy for the browser (analysis uses the original).")
        mid = (facts.get("duration_s") or 0) / 2
        if extract_frame(dest, mid, derived / "poster.png"):
            pim = Image.open(derived / "poster.png").convert("RGB")
            facts.setdefault("width", pim.width)
            facts.setdefault("height", pim.height)
            facts["display_width"], facts["display_height"] = pim.width, pim.height
            save_preview(pim, derived / "poster.jpg")

    case.update({
        "metadata_source": source, "exif": exif, "probe": probe, "facts": facts, "c2pa": c2pa,
        "markers": markers, "flags": flags, "questionnaire": metadata.build_questionnaire(facts, kind),
    })
    cases.save(case)
    cases.log(case_id, "METADATA", f"source={source} fields={len(exif)} flags={len(flags)} c2pa={bool(c2pa and c2pa.get('present'))}")
    return case


# ----------------------------------------------------------------------------
# Video frames for marking
# ----------------------------------------------------------------------------

def extract_frames(case_id: str, start: float, end: float, count: int, grid_cols: int) -> dict[str, Any]:
    case = cases.load(case_id)
    if case["kind"] != "video":
        raise ValueError("Frame extraction applies to videos only.")
    dur = case["facts"].get("duration_s") or end
    start, end = max(0.0, min(start, dur)), max(0.0, min(end, dur))
    if end < start:
        start, end = end, start
    count = max(1, min(int(count), MAX_VIDEO_FRAMES))
    fps = case["facts"].get("fps_average") or case["facts"].get("fps_nominal")
    times = frame_times(start, end, count, fps)
    src = cases.case_dir(case_id) / "original" / case["stored_name"]
    fdir = cases.case_dir(case_id) / "derived" / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    frames = []
    for i, t in enumerate(times):
        fid = f"F{i + 1:02d}"
        png = fdir / f"{fid}.png"
        if not extract_frame(src, t, png):
            continue
        im = Image.open(png).convert("RGB")
        pw, ph = save_preview(im, fdir / f"{fid}.jpg")
        frames.append({"id": fid, "t": t, "png": f"derived/frames/{fid}.png", "preview": f"derived/frames/{fid}.jpg",
                       "width": im.width, "height": im.height, "pw": pw, "ph": ph})
    if not frames:
        raise RuntimeError("No frames could be extracted from that time range.")
    w, h = frames[0]["width"], frames[0]["height"]
    grid = Grid.for_size(w, h, grid_cols)

    # Motion pre-scan to suggest cells (fast; uses reduced resolution).
    suggestions: list[str] = []
    try:
        scan_dir = cases.case_dir(case_id) / "derived" / "motion_scan"
        scan_dir.mkdir(exist_ok=True)
        m = motion.analyze_motion(src, start, end, w, h, grid, [], scan_dir)
        suggestions = m.get("suggested_cells", [])
    except Exception:  # noqa: BLE001  (suggestions are optional)
        pass
    case = cases.update(case_id, frames=frames, video_range={"start": start, "end": end},
                        grid_cols=grid.cols, motion_suggestions=suggestions, selection=None)
    cases.log(case_id, "FRAMES", f"range={start:.3f}-{end:.3f}s count={len(frames)} grid={grid.cols}x{grid.rows}")
    return {"frames": frames, "grid": grid.as_dict(), "suggested_cells": suggestions,
            "heatmap": "derived/motion_scan/motion_heatmap.png"
            if (cases.case_dir(case_id) / "derived" / "motion_scan" / "motion_heatmap.png").exists() else None}


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def _set_status(case_id: str, **kw: Any) -> None:
    case = cases.load(case_id)
    a = case.get("analysis", {})
    a.update(kw)
    cases.update(case_id, analysis=a)


def run_analysis(case_id: str) -> None:
    """Runs in a background thread."""
    try:
        _run(case_id)
    except AnalysisError as e:
        _set_status(case_id, status="error", error=str(e), finished_at=cases.now_iso())
        cases.log(case_id, "ERROR", str(e))
    except Exception as e:  # noqa: BLE001
        _set_status(case_id, status="error", error=f"{type(e).__name__}: {e}", finished_at=cases.now_iso())
        cases.log(case_id, "ERROR", traceback.format_exc().replace("\n", " | ")[:2000])


def _run(case_id: str) -> None:
    case = cases.load(case_id)
    d = cases.case_dir(case_id)
    adir = d / "derived" / "analysis"
    rdir = d / "derived" / "report"
    for p in (adir, rdir):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    _set_status(case_id, status="running", step="Verifying original file integrity", progress=3,
                started_at=cases.now_iso(), error=None)
    intact = cases.verify_original(case_id)
    cases.log(case_id, "VERIFY", f"original sha256 {'matches' if intact else 'DOES NOT MATCH'} intake hash")
    if not intact:
        raise AnalysisError("The original file no longer matches its intake hash. Analysis stopped.")

    src = d / "original" / case["stored_name"]
    sel = case.get("selection") or {"grid_cols": DEFAULT_GRID_COLS, "items": []}
    grid_cols = int(sel.get("grid_cols") or DEFAULT_GRID_COLS)
    facts = case.get("facts", {})
    items: list[ImageItem] = []
    pre: dict[str, Any] = {}

    _set_status(case_id, step="Preprocessing: forensic maps and full-resolution crops", progress=10)
    if case["kind"] == "image":
        im = load_image(src)
        grid = Grid.for_size(im.width, im.height, grid_cols)
        cells = next((it["cells"] for it in sel.get("items", []) if it.get("cells")), [])
        pre["grid"] = grid.as_dict()
        pre["marked_cells"] = cells
        pre["forensics"] = forensics.analyze_image(im, src, grid, cells, facts, adir, "image")
        items += _image_items(im, grid, cells, "IMG", adir, pre["forensics"], first=True)
        pre["pass2_images"] = ["IMG-OVERVIEW-GRID", "IMG-ROI-1", "IMG-TILE-1", "IMG-ROI-ENHANCED", "IMG-PIXEL-ZOOM"]
    else:
        frames = case.get("frames") or []
        if not frames:
            raise AnalysisError("No frames were extracted. Choose a time range and extract frames first.")
        by_id = {it["frame_id"]: it.get("cells", []) for it in sel.get("items", [])}
        w, h = frames[0]["width"], frames[0]["height"]
        grid = Grid.for_size(w, h, grid_cols)
        all_cells = sorted({c for v in by_id.values() for c in v})
        pre["grid"] = grid.as_dict()
        pre["marked_cells_by_frame"] = {k: v for k, v in by_id.items() if v}
        marked = [f for f in frames if by_id.get(f["id"])]
        key = marked[len(marked) // 2] if marked else frames[len(frames) // 2]
        key_im = Image.open(d / key["png"]).convert("RGB")
        pre["key_frame"] = key["id"]
        pre["forensics_key_frame"] = forensics.analyze_image(key_im, None, grid, by_id.get(key["id"], []), facts,
                                                             adir, "key")
        _set_status(case_id, step="Preprocessing: motion, tracking and frame integrity", progress=20)
        rng = case.get("video_range") or {"start": frames[0]["t"], "end": frames[-1]["t"]}
        try:
            pre["motion"] = motion.analyze_motion(src, rng["start"], rng["end"], w, h, grid, all_cells, adir,
                                                  forensics.degrees_per_pixel(facts, w, h))
        except Exception as e:  # noqa: BLE001
            pre["motion"] = {"error": str(e)}
        fixed = _image_items(key_im, grid, by_id.get(key["id"], []), f"KEY-{key['id']}", adir,
                             pre["forensics_key_frame"], first=False, roi_tiles=False)
        for name, desc in (("track", "Automatic track of the most salient object in the marked cells "
                                     "(cyan line) on the first analysed frame."),
                           ("light_trail", "Max-intensity stack of all analysed frames, camera motion removed: "
                                           "moving lights leave trails."),
                           ("heatmap", "Motion heat-map over the selected time range (camera motion removed): "
                                       "bright = movement between frames.")):
            fn = (pre.get("motion") or {}).get("images", {}).get(name)
            if fn and (adir / fn).exists():
                fixed.append(ImageItem(f"MOTION-{name.upper()}", desc,
                                       fit_to_budget(Image.open(adir / fn).convert("RGB"))[0]))
        frame_items: list[ImageItem] = []
        roi_budget = MAX_ROI_FRAMES
        for f in frames:
            fim = key_im if f["id"] == key["id"] else Image.open(d / f["png"]).convert("RGB")
            cells = by_id.get(f["id"], [])
            frame_items += _frame_items(fim, grid, cells, f, roi_budget > 0)
            if cells:
                roi_budget -= 1
        allowed = config.MAX_IMAGES_PER_REQUEST - len(fixed)
        if len(frame_items) > allowed:  # drop overviews of unmarked frames first, then thin evenly
            unmarked = {f"{f['id']}-GRID" for f in frames if not by_id.get(f["id"]) and f["id"] != key["id"]}
            frame_items = [it for it in frame_items if it.id not in unmarked] + \
                [it for it in frame_items if it.id in unmarked]
            frame_items = frame_items[:allowed]
            order = {f"{f['id']}-{suffix}": i * 2 + j
                     for i, f in enumerate(frames) for j, suffix in enumerate(("GRID", "ROI"))}
            frame_items.sort(key=lambda it: order.get(it.id, 0))
        items = frame_items + fixed
        pre["pass2_images"] = [f"{key['id']}-GRID", f"{key['id']}-ROI", "MOTION-TRACK", "MOTION-LIGHT_TRAIL"]

    items = items[:config.MAX_IMAGES_PER_REQUEST]
    for it in items:  # report copies (display JPEGs) of every image Claude sees
        save_preview(it.image, rdir / f"{it.id}.jpg", max_edge=1600)
    image_index = [{"id": it.id, "description": it.description, "file": f"derived/report/{it.id}.jpg",
                    **it.meta} for it in items]

    case_data = _case_data(case, pre)
    technical_ctx = {k: v for k, v in (case.get("context") or {}).items() if k in TECHNICAL_CONTEXT_KEYS and v}
    witness_ctx = {k: v for k, v in (case.get("context") or {}).items() if k in WITNESS_KEYS and v}

    # ---- Pass 1: blind -------------------------------------------------------
    _set_status(case_id, step="AI pass 1 of 2: blind technical examination (this can take a few minutes)", progress=35)
    before = (PASS1_INSTRUCTIONS + "\n\nCASE DATA (metadata, forensic measurements):\n"
              + json.dumps(case_data, indent=1, default=str)
              + "\n\nTECHNICAL CAPTURE CONTEXT supplied by the user:\n"
              + json.dumps(technical_ctx or {"note": "none provided"}, indent=1)
              + "\n\nIMAGES FOLLOW.")
    content, img_stats = build_content(items, before, "Now produce the PASS 1 findings as JSON.")
    cases.log(case_id, "AI_PASS1_SEND", f"model={config.MODEL} images={img_stats['images']} "
                                        f"bytes={img_stats['image_bytes']} visual_tokens~{img_stats['visual_tokens_est']}")
    pass1, usage1 = call_claude(content, PASS1_SCHEMA, SYSTEM_PROMPT,
                                on_progress=lambda n: _set_status(case_id, progress=min(60, 40 + n // 1500)))
    cases.log(case_id, "AI_PASS1_DONE", json.dumps(usage1))

    # ---- Pass 2: synthesis ------------------------------------------------------
    _set_status(case_id, step="AI pass 2 of 2: weighing explanations and writing the report", progress=65)
    wanted = pre.pop("pass2_images", [])
    key_items = [it for it in items if it.id in wanted][:4] or items[:2]
    before2 = (PASS2_INSTRUCTIONS + "\n\n(a) CASE DATA:\n" + json.dumps(case_data, indent=1, default=str)
               + "\n\n(b) YOUR PASS 1 FINDINGS (blind):\n" + json.dumps(pass1, indent=1)
               + "\n\n(c) CAPTURE CONTEXT AND WITNESS ACCOUNT:\n"
               + json.dumps({**technical_ctx, **witness_ctx} or {"note": "none provided"}, indent=1)
               + "\n\nIMAGE IDS AVAILABLE FOR ANNOTATION: " + ", ".join(it.id for it in items)
               + "\n\nKey images are repeated below for reference.")
    content2, _ = build_content(key_items, before2, "Now produce the FINAL REPORT as JSON.")
    final, usage2 = call_claude(content2, FINAL_SCHEMA, SYSTEM_PROMPT,
                                on_progress=lambda n: _set_status(case_id, progress=min(90, 70 + n // 1500)))
    cases.log(case_id, "AI_PASS2_DONE", json.dumps(usage2))

    _set_status(case_id, step="Rendering annotated stills", progress=93)
    final = _clamp_probabilities(final)
    annotated = _draw_annotations(final.get("annotations") or pass1.get("annotations") or [], rdir)
    for entry in image_index:
        if entry["id"] in annotated:
            entry["annotated_file"] = f"derived/report/{entry['id']}_annotated.jpg"

    result = {"pass1": pass1, "final": final, "preprocessing": pre, "images": image_index,
              "usage": [usage1, usage2], "image_stats": img_stats,
              "total_cost_usd": round(usage1["est_cost_usd"] + usage2["est_cost_usd"], 4),
              "model": config.MODEL, "effort": config.EFFORT, "completed_at": cases.now_iso()}
    (d / "derived" / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    _set_status(case_id, status="done", step="Complete", progress=100, finished_at=cases.now_iso())
    cases.log(case_id, "REPORT", f"verdict={final.get('authenticity', {}).get('verdict')} "
                                 f"confidence={final.get('overall_confidence', {}).get('percent')}")


def _image_items(im: Image.Image, grid: Grid, cells: list[str], prefix: str, adir: Path,
                 fx: dict[str, Any], first: bool, roi_tiles: bool = True) -> list[ImageItem]:
    """Images for one still (or the key video frame): overview, native ROI
    tiles, enhanced + pixel-zoom views, and forensic maps."""
    out: list[ImageItem] = []
    ov, s = fit_to_budget(im)
    if first or not prefix.startswith("KEY"):
        out.append(ImageItem(f"{prefix}-OVERVIEW-GRID" if prefix == "IMG" else f"{prefix}-GRID",
                             f"Overview of the whole frame, downscaled x{s:.3f}, with the labelled grid; "
                             f"marked cells outlined red: {', '.join(cells) or 'none'}.",
                             draw_grid(ov, grid, cells)))
        out.append(ImageItem(f"{prefix}-OVERVIEW", f"Same overview without grid lines (downscaled x{s:.3f}).", ov))
    if cells:
        box = grid.union_box(cells, pad_cells=0.25)
        tiles = native_tiles(im, box, max_tiles=4) if roi_tiles else []
        for i, t in enumerate(tiles, 1):
            out.append(ImageItem(f"{prefix}-ROI-{i}",
                                 ("Native full-resolution" if t["scale"] == 1 else f"Region scaled x{t['scale']}")
                                 + f" crop of the marked region (cells {', '.join(cells)}), frame pixels "
                                   f"x={t['box'][0]}-{t['box'][2]}, y={t['box'][1]}-{t['box'][3]}.",
                                 t["image"], lossless=True, meta={"box": t["box"]}))
        roi = im.crop(grid.union_box(cells, pad_cells=0.25))
        out.append(ImageItem(f"{prefix}-ROI-ENHANCED", "Enhanced (local contrast equalisation, shadow lift) "
                             "version of the marked region. Display aid only.", fit_to_budget(enhance(roi))[0]))
        obj = (fx or {}).get("object_measurement") or {}
        if obj.get("found"):
            bx, by, bw, bh = obj["bbox_px_full_frame"]
            pad = max(16, int(max(bw, bh) * 1.5))
            tight = im.crop((max(0, bx - pad), max(0, by - pad), min(im.width, bx + bw + pad), min(im.height, by + bh + pad)))
            z, f = pixel_zoom(tight)
            if f > 1:
                out.append(ImageItem(f"{prefix}-PIXEL-ZOOM", f"Nearest-neighbour x{f} pixel zoom on the detected object "
                                     f"(each original pixel is a {f}x{f} block; no detail added).",
                                     fit_to_budget(z)[0], lossless=True))
    else:
        # Nothing marked: give full-resolution coverage of the whole frame if affordable.
        tiles = native_tiles(im, (0, 0, im.width, im.height), max_tiles=6)
        if len(tiles) > 1 or tiles[0]["scale"] == 1:
            for i, t in enumerate(tiles, 1):
                out.append(ImageItem(f"{prefix}-TILE-{i}", f"Native full-resolution tile {i}/{len(tiles)}: pixels "
                                     f"x={t['box'][0]}-{t['box'][2]}, y={t['box'][1]}-{t['box'][3]}.",
                                     t["image"], lossless=True, meta={"box": t["box"]}))
    maps = (fx or {}).get("maps", {})
    for key, desc in (("ela", "Error Level Analysis map (bright = larger re-compression error)."),
                      ("noise", "Local sensor-noise map (noise residual std; inconsistent patches can indicate "
                                "pasted/generated regions)."),
                      ("fft", "Frequency spectrum of the noise residual (centre 1024 px). Sharp isolated dots/grids "
                              "= periodic patterns.")):
        if maps.get(key) and (adir / maps[key]).exists():
            out.append(ImageItem(f"{prefix}-{key.upper()}", desc, fit_to_budget(Image.open(adir / maps[key]).convert("RGB"))[0]))
    return out


def _frame_items(im: Image.Image, grid: Grid, cells: list[str], f: dict, include_roi: bool) -> list[ImageItem]:
    ov, s = fit_to_budget(im)
    out = [ImageItem(f"{f['id']}-GRID", f"Video frame {f['id']} at t={f['t']:.3f}s, whole frame downscaled x{s:.3f} "
                     f"with grid; marked cells: {', '.join(cells) or 'none'}.", draw_grid(ov, grid, cells),
                     meta={"t": f["t"]})]
    if cells and include_roi:
        t = native_tiles(im, grid.union_box(cells, pad_cells=0.25), max_tiles=1)[0]
        out.append(ImageItem(f"{f['id']}-ROI", ("Native full-resolution" if t["scale"] == 1 else f"Scaled x{t['scale']}")
                             + f" crop of marked cells {', '.join(cells)} at t={f['t']:.3f}s, frame pixels "
                               f"x={t['box'][0]}-{t['box'][2]}, y={t['box'][1]}-{t['box'][3]}.",
                             t["image"], lossless=True, meta={"t": f["t"], "box": t["box"]}))
    return out


def _case_data(case: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    exif = case.get("exif") or {}
    # Keep the full tag list but trim very long values for the prompt.
    exif_trim = {k: (str(v)[:300] if len(str(v)) > 300 else v) for k, v in exif.items()}
    if len(exif_trim) > 400:
        exif_trim = dict(list(exif_trim.items())[:400]) | {"_note": f"{len(exif) - 400} more tags omitted"}
    return {
        "file": {"name": case["filename"], "kind": case["kind"], "size_bytes": case["size_bytes"],
                 "sha256": case["hashes"]["sha256"]},
        "key_facts": case.get("facts"),
        "metadata_flags": case.get("flags"),
        "content_credentials_c2pa": case.get("c2pa"),
        "generator_or_editor_markers": case.get("markers"),
        "jpeg_structure": case.get("jpeg"),
        "preprocessing": pre,
        "all_metadata_tags": exif_trim,
    }


def _clamp_probabilities(final: dict[str, Any]) -> dict[str, Any]:
    def c(v):
        try:
            return max(1.0, min(99.0, float(v)))
        except (TypeError, ValueError):
            return None

    for k in ("ai_generated", "cgi_or_composite", "digitally_edited", "authentic_capture"):
        p = (final.get("authenticity") or {}).get(k)
        if isinstance(p, dict):
            for f in ("percent", "low", "high"):
                p[f] = c(p.get(f))
    oc = final.get("overall_confidence") or {}
    for f in ("percent", "low", "high"):
        if f in oc:
            oc[f] = c(oc[f])
    for e in final.get("explanations") or []:
        try:
            e["likelihood_percent"] = max(0.0, min(100.0, float(e.get("likelihood_percent", 0))))
        except (TypeError, ValueError):
            e["likelihood_percent"] = 0.0
    final.get("explanations", []).sort(key=lambda e: -e["likelihood_percent"])
    return final


def _draw_annotations(annotations: list[dict], rdir: Path) -> set[str]:
    by_img: dict[str, list[dict]] = {}
    for a in annotations:
        by_img.setdefault(a.get("image_id", ""), []).append(a)
    done = set()
    colours = [(255, 64, 64), (0, 220, 255), (255, 210, 0), (120, 255, 120), (255, 120, 255)]
    for img_id, anns in by_img.items():
        p = rdir / f"{img_id}.jpg"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        dr = ImageDraw.Draw(im)
        lw = max(2, im.width // 400)
        for i, a in enumerate(anns):
            col = colours[i % len(colours)]
            try:
                x, y, w, h = (float(a[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            if max(x, y, w, h) > 1.5:  # model returned pixels instead of fractions
                x, y, w, h = x / im.width, y / im.height, w / im.width, h / im.height
            box = (x * im.width, y * im.height, (x + w) * im.width, (y + h) * im.height)
            dr.rectangle(box, outline=col, width=lw)
            label = f"{i + 1}. {a.get('label', '')}"[:40]
            ty = box[1] - 16 if box[1] > 18 else box[3] + 2
            dr.rectangle((box[0], ty, box[0] + 7 * len(label) + 6, ty + 15), fill=(0, 0, 0))
            dr.text((box[0] + 3, ty + 1), label, fill=col)
        im.save(rdir / f"{img_id}_annotated.jpg", quality=90)
        done.add(img_id)
    return done
