"""End-to-end tests with synthetic media and a stubbed Claude call.

Run:  python -m pytest -q
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="ael-test-"))
os.environ["AEL_DATA_DIR"] = str(TMP)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

from fastapi.testclient import TestClient  # noqa: E402

from app import ai, pipeline  # noqa: E402
from app.main import app  # noqa: E402
from app.prompts import FINAL_SCHEMA, PASS1_SCHEMA  # noqa: E402

client = TestClient(app)
CALLS: list[dict] = []


def _fake_value(schema):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: _fake_value(v) for k, v in schema["properties"].items()}
    if t == "array":
        return [_fake_value(schema["items"])]
    if t == "number":
        return 42.0
    if t == "boolean":
        return False
    return "test"


def fake_call(content, schema, system, on_progress=None):
    images = [b for b in content if b["type"] == "image"]
    labels = [b["text"] for b in content if b["type"] == "text" and b["text"].startswith("[Image ")]
    CALLS.append({"schema": "pass1" if schema is PASS1_SCHEMA else "final", "images": len(images),
                  "labels": labels, "text": "\n".join(b["text"] for b in content if b["type"] == "text")})
    data = _fake_value(schema)
    first_id = labels[0].split("]")[0].replace("[Image ", "") if labels else "X"
    data["annotations"] = [{"image_id": first_id, "label": "object", "x": 0.4, "y": 0.4, "w": 0.1, "h": 0.1,
                            "note": "test"}]
    return data, {"model": "stub", "input_tokens": 1000, "output_tokens": 100, "est_cost_usd": 0.01,
                  "request_id": None}


@pytest.fixture(autouse=True)
def stub_claude(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(pipeline, "call_claude", fake_call)


def make_photo(path: Path):
    rng = np.random.default_rng(0)
    arr = np.full((3024, 4032, 3), (20, 30, 60), np.uint8)
    arr = np.clip(arr + rng.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    yy, xx = np.ogrid[:3024, :4032]
    arr[(yy - 900) ** 2 + (xx - 2600) ** 2 < 14 ** 2] = (255, 240, 200)
    im = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 15 Pro"
    exif[0x0131] = "17.1"
    im.save(path, "JPEG", quality=92, exif=exif)


def make_video(path: Path):
    """2 s of 640x360 at 30 fps with a small blinking light moving right."""
    frames = []
    for i in range(60):
        f = np.full((360, 640, 3), 15, np.uint8)
        f = np.clip(f + np.random.default_rng(i).normal(0, 3, f.shape), 0, 255).astype(np.uint8)
        if (i // 8) % 2 == 0:
            x = 100 + i * 6
            f[150:156, x:x + 6] = 255
        frames.append(f)
    raw = b"".join(f.tobytes() for f in frames)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360",
                    "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   input=raw, check=True)


def wait_done(case_id):
    pipeline.run_analysis(case_id)  # run synchronously in the test
    return client.get(f"/api/cases/{case_id}/status").json()


def test_photo_flow():
    p = TMP / "IMG_0001.JPG"
    make_photo(p)
    with open(p, "rb") as f:
        r = client.post("/api/cases", files={"file": ("IMG_0001.JPG", f, "image/jpeg")}, data={"title": "Light"})
    assert r.status_code == 200, r.text
    case = r.json()
    cid = case["id"]
    assert case["kind"] == "image" and case["facts"]["make"] == "Apple"
    assert case["jpeg"]["estimated_quality"] in range(85, 100)
    keys = {q["key"]: q for q in case["questionnaire"]}
    assert keys["device"]["known"] == "Apple iPhone 15 Pro"
    assert keys["capture_time"]["required"] is True  # no DateTimeOriginal in this file

    r = client.post(f"/api/cases/{cid}/context", json={"answers": {
        "provenance": "Original file from the recording device", "narrative": "SECRET-NARRATIVE",
        "capture_time": "2026-11-01 19:30 EST", "location": "Austin, TX"}})
    assert r.status_code == 200
    # Object at x=2600,y=900 in a 4032x3024 frame with 8 cols x 6 rows -> column F (2520-3024), row 2 (504-1008).
    r = client.post(f"/api/cases/{cid}/selection", json={"grid_cols": 8, "items": [{"frame_id": "IMG", "cells": ["F2"]}]})
    assert r.status_code == 200
    st = wait_done(cid)
    assert st["status"] == "done", st
    p1, p2 = CALLS
    assert "SECRET-NARRATIVE" not in p1["text"], "witness narrative must be withheld from the blind pass"
    assert "SECRET-NARRATIVE" in p2["text"]
    assert any("Native full-resolution crop" in l for l in p1["labels"])
    res = client.get(f"/api/cases/{cid}/result").json()
    obj = res["preprocessing"]["forensics"]["object_measurement"]
    assert obj["found"] and abs(obj["centroid_px_full_frame"][0] - 2600) < 5 and abs(obj["centroid_px_full_frame"][1] - 900) < 5
    html = client.get(f"/cases/{cid}/report").text
    assert "Chain of custody" in html and "Authentic camera capture" in html
    dl = client.get(f"/cases/{cid}/report/download?redact_location=true")
    assert dl.status_code == 200 and "data:image/jpeg;base64" in dl.text and "Austin, TX" not in dl.text
    custody = client.get(f"/api/cases/{cid}/custody").json()
    assert custody["original_intact"] is True
    assert [e["action"] for e in custody["log"]][:2] == ["INTAKE", "METADATA"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_video_flow():
    p = TMP / "clip.mp4"
    make_video(p)
    with open(p, "rb") as f:
        r = client.post("/api/cases", files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    case = r.json()
    cid = case["id"]
    assert case["kind"] == "video" and abs(case["facts"]["fps_average"] - 30) < 0.5
    r = client.post(f"/api/cases/{cid}/frames", json={"start": 0.2, "end": 1.8, "count": 8, "grid_cols": 8})
    assert r.status_code == 200, r.text
    fr = r.json()
    assert len(fr["frames"]) == 8 and fr["frames"][0]["width"] == 640
    assert fr["suggested_cells"], "motion should be detected"
    # Light travels along y~153 (row 4 of 5 rows at 72 px each -> row 3) from x~112 to ~400.
    cells = ["B3", "C3", "D3", "E3", "F3"]
    items = [{"frame_id": f["id"], "cells": cells} for f in fr["frames"]]
    r = client.post(f"/api/cases/{cid}/selection", json={"grid_cols": 8, "items": items})
    assert r.status_code == 200, r.text
    st = wait_done(cid)
    assert st["status"] == "done", st
    res = client.get(f"/api/cases/{cid}/result").json()
    m = res["preprocessing"]["motion"]
    assert m["camera_motion"]["mean_shift_px_per_frame"] < 1.0
    tr = m["object_track"]
    assert tr["frames_with_object"] >= 10
    assert tr["scene_displacement_px"][0] > 150  # moved right
    assert tr["blink"]["detected"] and 1.0 < tr["blink"]["frequency_hz"] < 2.6  # 8 on / 8 off at 30fps = 1.875 Hz
    assert CALLS[0]["images"] <= 40
    html = client.get(f"/cases/{cid}/report").text
    assert "Object track" in html


def teardown_module(_):
    shutil.rmtree(TMP, ignore_errors=True)
