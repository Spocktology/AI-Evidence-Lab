"""AI Evidence Lab - web server.

Run:  uvicorn app.main:app --reload   (then open http://localhost:8000)
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import cases, config, pipeline, report
from .grid import Grid, parse_cell

app = FastAPI(title="AI Evidence Lab", version=config.TOOL_VERSION)
STATIC = config.ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_running: set[str] = set()
_run_lock = threading.Lock()


def _case_or_404(case_id: str) -> dict[str, Any]:
    try:
        return cases.load(case_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "Case not found") from e


def _public(case: dict[str, Any]) -> dict[str, Any]:
    """Case data for the browser (omit bulky raw metadata)."""
    out = {k: v for k, v in case.items() if k not in ("exif", "probe")}
    out["metadata_tag_count"] = len(case.get("exif") or {})
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    import os

    return {"ok": True, "model": config.MODEL, "effort": config.EFFORT,
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
            "exiftool": bool(shutil.which("exiftool")), "ffmpeg": bool(shutil.which("ffmpeg")),
            "version": config.TOOL_VERSION}


@app.get("/api/cases")
def list_cases() -> list[dict[str, Any]]:
    return cases.list_cases()


@app.post("/api/cases")
async def create_case(file: UploadFile = File(...), title: str = Form("")) -> dict[str, Any]:
    tmp_dir = config.DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
            tmp.write(chunk)
    try:
        case = pipeline.intake(tmp_path, file.filename or "upload", title.strip() or None)
    except ValueError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e)) from e
    return _public(case)


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    return _public(_case_or_404(case_id))


@app.get("/api/cases/{case_id}/metadata")
def get_metadata(case_id: str) -> dict[str, Any]:
    case = _case_or_404(case_id)
    return {"source": case.get("metadata_source"), "tags": case.get("exif"), "ffprobe": case.get("probe")}


@app.get("/api/cases/{case_id}/custody")
def get_custody(case_id: str) -> dict[str, Any]:
    _case_or_404(case_id)
    return {"log": cases.read_log(case_id), "original_intact": cases.verify_original(case_id)}


class ContextIn(BaseModel):
    answers: dict[str, str]


@app.post("/api/cases/{case_id}/context")
def save_context(case_id: str, body: ContextIn) -> dict[str, Any]:
    case = _case_or_404(case_id)
    keys = {q["key"] for q in case.get("questionnaire", [])}
    answers = {k: str(v)[:5000] for k, v in body.answers.items() if k in keys}
    case = cases.update(case_id, context=answers)
    cases.log(case_id, "CONTEXT", f"fields={sorted(k for k, v in answers.items() if v)}")
    return _public(case)


class FramesIn(BaseModel):
    start: float
    end: float
    count: int = 12
    grid_cols: int = 8


@app.post("/api/cases/{case_id}/frames")
def frames(case_id: str, body: FramesIn) -> dict[str, Any]:
    _case_or_404(case_id)
    try:
        return pipeline.extract_frames(case_id, body.start, body.end, body.count, body.grid_cols)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e


class SelectionItem(BaseModel):
    frame_id: str
    cells: list[str]


class SelectionIn(BaseModel):
    grid_cols: int
    items: list[SelectionItem]


@app.post("/api/cases/{case_id}/selection")
def save_selection(case_id: str, body: SelectionIn) -> dict[str, Any]:
    case = _case_or_404(case_id)
    if case["kind"] == "image":
        w, h = case["facts"]["width"], case["facts"]["height"]
    else:
        if not case.get("frames"):
            raise HTTPException(400, "Extract frames first.")
        w, h = case["frames"][0]["width"], case["frames"][0]["height"]
    grid = Grid.for_size(int(w), int(h), body.grid_cols)
    valid_frames = {"IMG"} | {f["id"] for f in case.get("frames", [])}
    items = []
    for it in body.items:
        if it.frame_id not in valid_frames:
            continue
        cells = []
        for c in it.cells:
            try:
                ci, ri = parse_cell(c)
            except ValueError:
                continue
            if 0 <= ci < grid.cols and 0 <= ri < grid.rows:
                cells.append(c.upper())
        items.append({"frame_id": it.frame_id, "cells": sorted(set(cells))})
    sel = {"grid_cols": grid.cols, "grid_rows": grid.rows, "items": items}
    case = cases.update(case_id, selection=sel)
    cases.log(case_id, "SELECTION", "; ".join(f"{i['frame_id']}:{','.join(i['cells'])}" for i in items if i["cells"]) or "none")
    return _public(case)


@app.post("/api/cases/{case_id}/analyze")
def analyze(case_id: str) -> dict[str, Any]:
    case = _case_or_404(case_id)
    with _run_lock:
        if case_id in _running:
            raise HTTPException(409, "Analysis already running for this case.")
        _running.add(case_id)
    pipeline._set_status(case_id, status="queued", step="Starting", progress=1, error=None)
    cases.log(case_id, "ANALYZE", "analysis requested")

    def worker() -> None:
        try:
            pipeline.run_analysis(case_id)
        finally:
            with _run_lock:
                _running.discard(case_id)

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


@app.get("/api/cases/{case_id}/status")
def status(case_id: str) -> dict[str, Any]:
    return _case_or_404(case_id).get("analysis", {"status": "new"})


@app.get("/api/cases/{case_id}/result")
def result(case_id: str) -> JSONResponse:
    _case_or_404(case_id)
    p = cases.case_dir(case_id) / "derived" / "result.json"
    if not p.exists():
        raise HTTPException(404, "No result yet")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/cases/{case_id}/files/{path:path}")
def case_file(case_id: str, path: str) -> FileResponse:
    _case_or_404(case_id)
    base = cases.case_dir(case_id).resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


@app.get("/cases/{case_id}/report", response_class=HTMLResponse)
def report_page(case_id: str, redact_location: bool = False) -> HTMLResponse:
    _case_or_404(case_id)
    try:
        return HTMLResponse(report.render(case_id, embed=False, redact_location=redact_location))
    except FileNotFoundError as e:
        raise HTTPException(404, "No report yet - run the analysis first.") from e


@app.get("/cases/{case_id}/report/download")
def report_download(case_id: str, redact_location: bool = False) -> Response:
    case = _case_or_404(case_id)
    try:
        html = report.render(case_id, embed=True, redact_location=redact_location)
    except FileNotFoundError as e:
        raise HTTPException(404, "No report yet - run the analysis first.") from e
    cases.log(case_id, "EXPORT", f"self-contained report downloaded redact_location={redact_location}")
    name = cases.safe_filename(f"{case.get('title') or case_id}-report.html").replace(" ", "_")
    return Response(html, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
