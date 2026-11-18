"""Render the analysis report as HTML (linked images, or fully self-contained)."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import cases, config

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html"]))

VERDICT_LABELS = {
    "authentic_capture": "Authentic camera capture",
    "authentic_but_edited": "Authentic capture, edited",
    "likely_ai_generated": "Likely AI-generated",
    "likely_cgi_or_composite": "Likely CGI / composite",
    "inconclusive": "Inconclusive",
}
LOCATION_KEYS = re.compile(r"gps|location|latitude|longitude|position|coordinates", re.I)


def _pct(v: Any) -> str:
    try:
        return f"{float(v):.0f}%"
    except (TypeError, ValueError):
        return "-"


_env.filters["pct"] = _pct
_env.filters["label"] = lambda s: str(s or "").replace("_", " ").capitalize()


def render(case_id: str, embed: bool = False, redact_location: bool = False) -> str:
    case = cases.load(case_id)
    d = cases.case_dir(case_id)
    rp = d / "derived" / "result.json"
    if not rp.exists():
        raise FileNotFoundError("result")
    result = json.loads(rp.read_text(encoding="utf-8"))

    def src(rel: str | None) -> str | None:
        if not rel:
            return None
        if embed:
            p = d / rel
            if not p.exists():
                return None
            mime = "image/png" if p.suffix == ".png" else "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")
        return f"/api/cases/{case_id}/files/{rel}"

    facts = dict(case.get("facts") or {})
    context = dict(case.get("context") or {})
    questions = {q["key"]: q["label"] for q in case.get("questionnaire", [])}
    if redact_location:
        facts = {k: ("[redacted]" if LOCATION_KEYS.search(k) else v) for k, v in facts.items()}
        context = {k: ("[redacted]" if k == "location" else v) for k, v in context.items()}

    images = result.get("images", [])
    for im in images:
        im["src"] = src(im.get("annotated_file") or im.get("file"))
        im["plain_src"] = src(im.get("file"))
    img_by_id = {im["id"]: im for im in images}
    final = result.get("final", {})
    ann_by_img: dict[str, list] = {}
    for a in final.get("annotations") or []:
        ann_by_img.setdefault(a.get("image_id"), []).append(a)

    pre = result.get("preprocessing", {})
    usage = result.get("usage", [])
    return _env.get_template("report.html").render(
        case=case, result=result, final=final, facts=facts, context=context, questions=questions,
        images=images, img_by_id=img_by_id, ann_by_img=ann_by_img, pre=pre,
        verdict_label=VERDICT_LABELS.get((final.get("authenticity") or {}).get("verdict"), "-"),
        custody=cases.read_log(case_id), intact=cases.verify_original(case_id), usage=usage,
        total_cost=result.get("total_cost_usd"), tool_version=config.TOOL_VERSION,
        redact_location=redact_location, embed=embed, json_dumps=lambda o: json.dumps(o, indent=1, default=str),
        preview=src((case.get("preview") or {}).get("path")) if case["kind"] == "image" else src("derived/poster.jpg"),
    )
