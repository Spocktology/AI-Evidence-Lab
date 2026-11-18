"""Case storage and chain-of-custody log.

Each case lives in data/cases/<case_id>/:
    original/<filename>   the uploaded file, never modified (made read-only)
    derived/              everything the tool generates (frames, maps, crops)
    case.json             case state (metadata, context, selections, results)
    custody.log           append-only, timestamped log of every action
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CASES_DIR

_lock = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_case_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "upload")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip() or "upload"
    return name[:150]


def case_dir(case_id: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", case_id):
        raise FileNotFoundError(case_id)
    d = CASES_DIR / case_id
    if not d.is_dir():
        raise FileNotFoundError(case_id)
    return d


def create_case_dir(case_id: str) -> Path:
    d = CASES_DIR / case_id
    (d / "original").mkdir(parents=True, exist_ok=False)
    (d / "derived").mkdir()
    return d


def hash_file(path: Path) -> dict[str, str]:
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def log(case_id: str, action: str, detail: str = "") -> None:
    line = f"{now_iso()}\t{action}\t{detail}\n"
    with _lock, open(CASES_DIR / case_id / "custody.log", "a", encoding="utf-8") as f:
        f.write(line)


def read_log(case_id: str) -> list[dict[str, str]]:
    p = case_dir(case_id) / "custody.log"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            out.append({"time": parts[0], "action": parts[1], "detail": parts[2]})
    return out


def load(case_id: str) -> dict[str, Any]:
    with _lock:
        return json.loads((case_dir(case_id) / "case.json").read_text(encoding="utf-8"))


def save(case: dict[str, Any]) -> None:
    with _lock:
        d = CASES_DIR / case["id"]
        tmp = d / "case.json.tmp"
        tmp.write_text(json.dumps(case, indent=2, default=str), encoding="utf-8")
        tmp.replace(d / "case.json")


def update(case_id: str, **fields: Any) -> dict[str, Any]:
    with _lock:
        case = load(case_id)
        case.update(fields)
        case["updated_at"] = now_iso()
        save(case)
        return case


def list_cases() -> list[dict[str, Any]]:
    out = []
    for d in sorted(CASES_DIR.iterdir(), reverse=True):
        f = d / "case.json"
        if f.exists():
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({
                "id": c["id"], "title": c.get("title"), "filename": c.get("filename"),
                "kind": c.get("kind"), "created_at": c.get("created_at"),
                "status": c.get("analysis", {}).get("status", "new"),
            })
    return out


def verify_original(case_id: str) -> bool:
    """Re-hash the original and confirm it still matches the intake hash."""
    case = load(case_id)
    p = case_dir(case_id) / "original" / case["stored_name"]
    return hash_file(p)["sha256"] == case["hashes"]["sha256"]
