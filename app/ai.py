"""Calls to the Claude API."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
from PIL import Image

from . import config
from .media import encode_for_claude, vision_tokens

# $ per million tokens (input, output) for the cost line in the report.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-sonnet-5-5": (2.0, 10.0),
    "claude-fable-5-2": (8.0, 40.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-4-8": (5.0, 25.0),
}

# Raw bytes across all images in one request. Base64 adds ~33% and the
# request limit is 128 MB, so stay comfortably below.
REQUEST_IMAGE_BYTE_BUDGET = 80_000_000


class AnalysisError(RuntimeError):
    pass


@dataclass
class ImageItem:
    id: str
    description: str
    image: Image.Image
    lossless: bool = False  # True for native-resolution evidence crops
    meta: dict[str, Any] = field(default_factory=dict)


def build_content(items: list[ImageItem], text_before: str, text_after: str) -> tuple[list[dict], dict]:
    """Interleave a label before each image (so Claude can cite image IDs) and
    keep the total payload within the request budget: native evidence crops
    stay lossless; context images fall back to high-quality JPEG if needed."""
    content: list[dict] = [{"type": "text", "text": text_before}]
    total_bytes, total_tokens = 0, 0
    encoded = []
    for it in items:
        data, mime = encode_for_claude(it.image, lossless=it.lossless)
        encoded.append((it, data, mime))
        total_bytes += len(data)
    if total_bytes > REQUEST_IMAGE_BYTE_BUDGET:  # re-encode context images as JPEG
        total_bytes = 0
        re_enc = []
        for it, data, mime in encoded:
            if not it.lossless:
                data, mime = encode_for_claude(it.image, lossless=False)
            re_enc.append((it, data, mime))
            total_bytes += len(data)
        encoded = re_enc
    for it, data, mime in encoded:
        w, h = it.image.size
        total_tokens += vision_tokens(w, h)
        content.append({"type": "text", "text": f"[Image {it.id}] {it.description} ({w}x{h} px)"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime,
                                                     "data": base64.standard_b64encode(data).decode("ascii")}})
    content.append({"type": "text", "text": text_after})
    return content, {"images": len(items), "image_bytes": total_bytes, "visual_tokens_est": total_tokens}


def call_claude(content: list[dict], schema: dict, system: str, on_progress=None) -> tuple[dict, dict]:
    """One structured-output request. Streams (long requests), uses adaptive
    thinking, and returns (parsed_json, usage_info)."""
    try:
        client = anthropic.Anthropic(timeout=1800, max_retries=3)
    except Exception as e:  # noqa: BLE001  (no credentials configured)
        raise AnalysisError("No Anthropic API key found. Put ANTHROPIC_API_KEY in your .env file and "
                            "restart the server.") from e
    params: dict[str, Any] = dict(
        model=config.MODEL,
        max_tokens=64000,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": config.EFFORT, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    try:
        if config.USE_FALLBACKS:
            with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                             **params) as stream:
                msg = _consume(stream, on_progress)
        else:
            with client.messages.stream(**params) as stream:
                msg = _consume(stream, on_progress)
    except anthropic.AuthenticationError as e:
        raise AnalysisError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.PermissionDeniedError as e:
        raise AnalysisError(f"The API key lacks permission for this request: {e.message}") from e
    except anthropic.NotFoundError as e:
        raise AnalysisError(f"Model '{config.MODEL}' not found or not available to this key.") from e
    except anthropic.RateLimitError as e:
        raise AnalysisError("Rate limited by the Anthropic API. Wait a minute and run the analysis again.") from e
    except anthropic.BadRequestError as e:
        raise AnalysisError(f"The API rejected the request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise AnalysisError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AnalysisError("Could not reach the Anthropic API (network error).") from e

    if msg.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        raise AnalysisError("Claude declined to analyse this media"
                            + (f" ({details.category}: {details.explanation})" if details else "") + ".")
    if msg.stop_reason == "max_tokens":
        raise AnalysisError("The response was cut off (max tokens). Try fewer frames or a smaller region.")
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnalysisError("Claude's response was not valid JSON.") from e
    u = msg.usage
    in_tok = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) + \
        (getattr(u, "cache_creation_input_tokens", 0) or 0)
    out_tok = u.output_tokens or 0
    pin, pout = PRICES.get(msg.model, PRICES.get(config.MODEL, (0, 0)))
    usage = {"model": msg.model, "input_tokens": in_tok, "output_tokens": out_tok,
             "est_cost_usd": round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 4),
             "request_id": getattr(msg, "_request_id", None)}
    return data, usage


def _consume(stream, on_progress) -> Any:
    chars = 0
    for event in stream:
        if on_progress and getattr(event, "type", "") == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", "") == "text_delta":
                chars += len(delta.text)
                if chars % 2000 < len(delta.text):
                    on_progress(chars)
    return stream.get_final_message()
