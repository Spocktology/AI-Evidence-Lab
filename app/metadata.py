"""Metadata extraction: EXIF/XMP/IPTC/QuickTime (exiftool), container and stream
info (ffprobe), Content Credentials (C2PA), generator fingerprints, and
consistency checks. Produces a normalized `key_facts` dict that drives both
the AI prompt and the "what we still need to know" questionnaire.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

# Strings that generators, editors, or provenance systems leave in files.
GENERATOR_MARKERS = {
    b"trainedAlgorithmicMedia": "IPTC DigitalSourceType = trainedAlgorithmicMedia (declared AI-generated)",
    b"compositeWithTrainedAlgorithmicMedia": "IPTC DigitalSourceType = composite with AI-generated content",
    b"algorithmicMedia": "IPTC DigitalSourceType = algorithmicMedia",
    b"compositeSynthetic": "IPTC DigitalSourceType = compositeSynthetic",
    b"Stable Diffusion": "Stable Diffusion reference",
    b"ComfyUI": "ComfyUI workflow reference",
    b"Midjourney": "Midjourney reference",
    b"DALL-E": "DALL-E reference",
    b"DALL\xc2\xb7E": "DALL-E reference",
    b"Adobe Firefly": "Adobe Firefly reference",
    b"GPT-4o": "OpenAI image generation reference",
    b"gpt-image": "OpenAI image generation reference",
    b"Imagen": "Google Imagen reference",
    b"SynthID": "Google SynthID reference",
    b"Sora": "OpenAI Sora reference",
    b"Runway": "Runway reference",
    b"Kling": "Kling reference",
    b"Veo": "Google Veo reference",
    b"Pika": "Pika reference",
    b"Luma": "Luma reference",
    b"Higgsfield": "Higgsfield reference",
    b"NovelAI": "NovelAI reference",
    b"Leonardo": "Leonardo.ai reference",
    b"Ideogram": "Ideogram reference",
    b"Flux": "FLUX model reference",
    b"c2pa": "C2PA (Content Credentials) data present",
    b"jumbf": "JUMBF box (often carries C2PA) present",
    b"Photoshop": "Adobe Photoshop reference",
    b"Lightroom": "Adobe Lightroom reference",
    b"GIMP": "GIMP reference",
    b"After Effects": "Adobe After Effects reference",
    b"Premiere": "Adobe Premiere reference",
    b"DaVinci": "DaVinci Resolve reference",
    b"CapCut": "CapCut reference",
    b"Blender": "Blender reference",
    b"Unreal Engine": "Unreal Engine reference",
    b"Lavf": "FFmpeg (libavformat) remux/encode",
    b"Snapseed": "Snapseed reference",
    b"Picsart": "Picsart reference",
    b"FaceApp": "FaceApp reference",
}
# Markers that need word boundaries to avoid false hits in binary data.
_SHORT = {b"Sora", b"Veo", b"Pika", b"Luma", b"Kling", b"Flux", b"Imagen", b"Runway", b"Leonardo", b"Blender", b"Premiere"}


def _run(cmd: list[str], timeout: int = 120) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 and not r.stdout:
        return None
    return r.stdout.decode("utf-8", "replace")


def run_exiftool(path: Path, video: bool) -> dict[str, Any] | None:
    if not shutil.which("exiftool"):
        return None
    cmd = ["exiftool", "-j", "-G1", "-a", "-u", "-api", "LargeFileSupport=1"]
    if video:
        cmd.append("-ee3")  # embedded timed metadata (e.g. drone/dashcam GPS tracks)
    cmd.append(str(path))
    out = _run(cmd, timeout=300)
    if not out:
        return None
    try:
        data = json.loads(out)[0]
    except (json.JSONDecodeError, IndexError):
        return None
    # Drop binary blobs and enormous values.
    clean = {}
    for k, v in data.items():
        s = str(v)
        if "(Binary data" in s or "use -b option" in s:
            clean[k] = "[binary data omitted]"
        elif len(s) > 2000:
            clean[k] = s[:2000] + "...[truncated]"
        else:
            clean[k] = v
    return clean


def run_pillow_exif(path: Path) -> dict[str, Any]:
    """Fallback when exiftool is not installed."""
    from PIL import ExifTags, Image

    out: dict[str, Any] = {}
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(path) as im:
            out["File:ImageWidth"], out["File:ImageHeight"] = im.size
            out["File:FileType"] = im.format
            exif = im.getexif()
            for tag, val in exif.items():
                out[f"IFD0:{ExifTags.TAGS.get(tag, tag)}"] = _plain(val)
            for ifd_id, group in ((ExifTags.IFD.Exif, "ExifIFD"), (ExifTags.IFD.GPSInfo, "GPS")):
                try:
                    sub = exif.get_ifd(ifd_id)
                except Exception:  # noqa: BLE001
                    continue
                names = ExifTags.GPSTAGS if group == "GPS" else ExifTags.TAGS
                for tag, val in sub.items():
                    out[f"{group}:{names.get(tag, tag)}"] = _plain(val)
            for k, v in (im.info or {}).items():
                if isinstance(v, (str, int, float)):
                    out[f"Info:{k}"] = str(v)[:2000]
    except Exception as e:  # noqa: BLE001
        out["Error"] = f"Could not read image metadata: {e}"
    return out


def _plain(v: Any) -> Any:
    if isinstance(v, bytes):
        return f"[{len(v)} bytes]"
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, Fraction) or type(v).__name__ == "IFDRational":
        try:
            return float(v)
        except (ZeroDivisionError, ValueError):
            return None
    return v


def run_ffprobe(path: Path) -> dict[str, Any] | None:
    if not shutil.which("ffprobe"):
        return None
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def read_c2pa(path: Path) -> dict[str, Any] | None:
    try:
        import c2pa
    except ImportError:
        return None
    try:
        reader = c2pa.Reader(str(path))
    except Exception as e:  # noqa: BLE001  (no manifest is the normal case)
        msg = str(e)
        if "ManifestNotFound" in msg or "not found" in msg.lower() or "JumbfNotFound" in msg:
            return {"present": False}
        return {"present": False, "note": f"C2PA read error: {msg[:300]}"}
    try:
        data = json.loads(reader.json())
        state = None
        try:
            state = reader.get_validation_state()
        except Exception:  # noqa: BLE001
            pass
        active = data.get("active_manifest")
        manifest = (data.get("manifests") or {}).get(active, {})
        summary = {
            "present": True,
            "validation_state": str(state) if state else None,
            "claim_generator": manifest.get("claim_generator") or str(manifest.get("claim_generator_info", ""))[:300],
            "title": manifest.get("title"),
            "signature_issuer": (manifest.get("signature_info") or {}).get("issuer"),
            "signature_time": (manifest.get("signature_info") or {}).get("time"),
            "assertions": [],
            "digital_source_types": [],
        }
        for a in manifest.get("assertions", []):
            summary["assertions"].append(a.get("label"))
            blob = json.dumps(a.get("data", {}))
            for m in re.findall(r"digitalsourcetype/([A-Za-z]+)", blob):
                if m not in summary["digital_source_types"]:
                    summary["digital_source_types"].append(m)
        summary["validation_status"] = [
            s.get("code") for s in (data.get("validation_status") or [])
        ][:20]
        return summary
    finally:
        try:
            reader.close()
        except Exception:  # noqa: BLE001
            pass


def scan_markers(path: Path, metadata_text: str = "", head: int = 4 << 20, tail: int = 2 << 20) -> list[str]:
    """Look for generator / editor fingerprints.

    Long, distinctive markers are searched in the raw bytes at the start and
    end of the file (where metadata lives). Short brand names (e.g. "Veo",
    "Sora") would match random bytes in compressed data, so they are only
    searched in the already-extracted metadata text.
    """
    size = path.stat().st_size
    with open(path, "rb") as f:
        chunk = f.read(head)
        if size > head:
            f.seek(max(head, size - tail))
            chunk += f.read(tail)
    meta_bytes = metadata_text.encode("utf-8", "ignore")
    found = []
    for marker, label in GENERATOR_MARKERS.items():
        if marker in _SHORT:
            if re.search(rb"(?<![A-Za-z])" + re.escape(marker) + rb"(?![A-Za-z])", meta_bytes):
                found.append(label)
        elif marker in (b"c2pa", b"jumbf"):
            if re.search(rb"(?i)" + marker, chunk):
                found.append(label)
        elif marker in chunk or marker in meta_bytes:
            found.append(label)
    # A1111 / ComfyUI write prompts into PNG text chunks.
    if re.search(rb"tEXt(parameters|prompt|workflow)\x00", chunk):
        found.append("PNG text chunk with generation parameters/prompt (typical of Stable Diffusion UIs)")
    return sorted(set(found))


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------

def _find(meta: dict[str, Any], *names: str) -> Any:
    """Find the first tag whose name (ignoring group prefix) matches."""
    lower = {k.split(":", 1)[-1].lower(): v for k, v in meta.items()}
    for n in names:
        v = lower.get(n.lower())
        if v not in (None, "", "0000:00:00 00:00:00", "Unknown"):
            return v
    return None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(\.\d+)?(/\d+)?", str(v))
    if not m:
        return None
    try:
        return float(Fraction(m.group(0)))
    except (ValueError, ZeroDivisionError):
        return None


def _rate(s: str | None) -> float | None:
    if not s or s in ("0/0",):
        return None
    try:
        return round(float(Fraction(s)), 3)
    except (ValueError, ZeroDivisionError):
        return None


def normalize(exif: dict[str, Any], probe: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    f: dict[str, Any] = {
        "make": _find(exif, "Make", "AndroidMake", "com.apple.quicktime.make"),
        "model": _find(exif, "Model", "AndroidModel", "com.apple.quicktime.model", "CameraModelName"),
        "lens": _find(exif, "LensModel", "LensID", "Lens", "LensInfo"),
        "software": _find(exif, "Software", "CreatorTool", "HistorySoftwareAgent", "Encoder", "com.apple.quicktime.software"),
        "capture_time": _find(exif, "DateTimeOriginal", "CreationDate", "CreateDate", "MediaCreateDate", "TrackCreateDate"),
        "modify_time": _find(exif, "ModifyDate", "MetadataDate"),
        "timezone_offset": _find(exif, "OffsetTimeOriginal", "OffsetTime", "TimeZone"),
        "gps_latitude": _find(exif, "GPSLatitude"),
        "gps_longitude": _find(exif, "GPSLongitude"),
        "gps_altitude": _find(exif, "GPSAltitude"),
        "gps_direction": _find(exif, "GPSImgDirection"),
        "gps_position": _find(exif, "GPSPosition", "GPSCoordinates", "com.apple.quicktime.location.ISO6709"),
        "focal_length_mm": _num(_find(exif, "FocalLength")),
        "focal_length_35mm": _num(_find(exif, "FocalLengthIn35mmFormat", "FocalLength35efl")),
        "aperture": _num(_find(exif, "FNumber", "Aperture")),
        "exposure_time": _find(exif, "ExposureTime", "ShutterSpeed"),
        "iso": _num(_find(exif, "ISO", "ISOSpeedRatings", "PhotographicSensitivity")),
        "flash": _find(exif, "Flash"),
        "digital_zoom": _num(_find(exif, "DigitalZoomRatio")),
        "scene_type": _find(exif, "SceneCaptureType", "SceneType"),
        "orientation": _find(exif, "Orientation"),
        "width": _num(_find(exif, "ImageWidth", "ExifImageWidth", "SourceImageWidth")),
        "height": _num(_find(exif, "ImageHeight", "ExifImageHeight", "SourceImageHeight")),
        "file_type": _find(exif, "FileType", "MIMEType"),
        "hdr_or_computational": _find(exif, "HDRHeadroom", "HDRGain", "CompositeImage", "HDRImageType", "AEBracketValue"),
        "jpeg_quality_tag": _find(exif, "Quality", "JPEGQuality"),
    }
    if probe:
        v = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
        a = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
        fmt = probe.get("format", {})
        tags = {k.lower(): val for k, val in (fmt.get("tags") or {}).items()}
        if v:
            rotation = None
            for sd in v.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rotation = sd["rotation"]
            rotation = rotation if rotation is not None else (v.get("tags") or {}).get("rotate")
            r_fps, avg_fps = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
            f.update({
                "width": v.get("width") or f["width"],
                "height": v.get("height") or f["height"],
                "codec": v.get("codec_name"),
                "codec_profile": v.get("profile"),
                "pixel_format": v.get("pix_fmt"),
                "color_transfer": v.get("color_transfer"),
                "color_primaries": v.get("color_primaries"),
                "fps_nominal": r_fps,
                "fps_average": avg_fps,
                "variable_frame_rate": bool(r_fps and avg_fps and abs(r_fps - avg_fps) / max(r_fps, 1) > 0.02),
                "frame_count": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
                "video_bitrate_kbps": round(int(v["bit_rate"]) / 1000) if str(v.get("bit_rate", "")).isdigit() else None,
                "rotation": rotation,
                "audio": bool(a),
                "audio_codec": a.get("codec_name") if a else None,
            })
            if kind == "video":
                f["hdr"] = v.get("color_transfer") in ("arib-std-b67", "smpte2084")
        f["duration_s"] = _rate(fmt.get("duration")) if fmt.get("duration") else None
        f["container"] = fmt.get("format_long_name")
        f["container_brand"] = tags.get("major_brand")
        f["encoder"] = tags.get("encoder") or f.get("software")
        f["make"] = f["make"] or tags.get("com.apple.quicktime.make") or tags.get("com.android.manufacturer")
        f["model"] = f["model"] or tags.get("com.apple.quicktime.model") or tags.get("com.android.model")
        f["capture_time"] = f["capture_time"] or tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
        f["gps_position"] = f["gps_position"] or tags.get("com.apple.quicktime.location.iso6709") or tags.get("location")
    return {k: v for k, v in f.items() if v not in (None, "")}


def consistency_checks(facts: dict[str, Any], exif: dict[str, Any], markers: list[str],
                       c2pa: dict[str, Any] | None, kind: str, file_ext: str) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    def add(level: str, msg: str) -> None:
        flags.append({"level": level, "message": msg})

    if not facts.get("make") and not facts.get("model"):
        add("caution", "No camera make/model recorded. Common when a file has been shared through social media or "
                       "messaging apps (they strip metadata), screenshotted, screen-recorded, or generated/rendered. "
                       "Not suspicious by itself.")
    if not facts.get("capture_time"):
        add("caution", "No original capture timestamp present.")
    sw = str(facts.get("software", ""))
    if re.search(r"photoshop|lightroom|gimp|snapseed|picsart|facetune|after effects|premiere|capcut|davinci|"
                 r"final cut|imovie|blender|canva|affinity|pixelmator|luminar", sw, re.I):
        add("warning", f"Editing software recorded in metadata: '{sw}'. The file was opened/saved by an editor; "
                       "this does not prove content manipulation, but the file is not camera-original.")
    if facts.get("capture_time") and facts.get("modify_time") and str(facts["capture_time"])[:19] != str(facts["modify_time"])[:19]:
        add("info", f"Modify date ({facts['modify_time']}) differs from capture date ({facts['capture_time']}).")
    for m in markers:
        level = "warning" if re.search(r"AI|Diffusion|Midjourney|DALL|Firefly|Imagen|SynthID|Sora|Runway|Kling|Veo|"
                                       r"Pika|Luma|Higgsfield|NovelAI|Leonardo|Ideogram|FLUX|generation|trained|"
                                       r"synthetic|algorithmic|Blender|Unreal", m) else "info"
        add(level, f"File contains marker: {m}.")
    if c2pa and c2pa.get("present"):
        dst = c2pa.get("digital_source_types") or []
        add("warning" if any("algorithmic" in d.lower() or "synthetic" in d.lower() for d in dst) else "info",
            f"Content Credentials (C2PA) manifest found. Generator: {c2pa.get('claim_generator')}; "
            f"signed by: {c2pa.get('signature_issuer')}; digital source types: {', '.join(dst) or 'none declared'}; "
            f"validation: {c2pa.get('validation_state')}.")
    if kind == "video":
        if facts.get("variable_frame_rate"):
            add("info", f"Variable frame rate (nominal {facts.get('fps_nominal')} fps vs average "
                        f"{facts.get('fps_average')} fps). Typical of phone video; timing/speed estimates must use "
                        "per-frame timestamps.")
        enc = str(facts.get("encoder", ""))
        if "Lavf" in enc or "lavf" in enc:
            add("caution", f"Encoder tag '{enc}' indicates the file was re-encoded or remuxed with FFmpeg "
                           "(common for downloads from social platforms and for editing tools).")
        if facts.get("video_bitrate_kbps") and facts.get("width"):
            bpp = facts["video_bitrate_kbps"] * 1000 / max(1, facts["width"] * facts["height"] * (facts.get("fps_average") or 30))
            if bpp < 0.04:
                add("caution", f"Low bitrate for the resolution ({facts['video_bitrate_kbps']} kbps, "
                               f"{bpp:.3f} bits/pixel/frame): heavy compression, typical of platform re-encodes. "
                               "Fine detail and small objects will be degraded by compression artifacts.")
    if file_ext in (".png", ".webp") and kind == "image" and not facts.get("make"):
        add("caution", "PNG/WebP image without camera data: cameras rarely save these formats directly; typical of "
                       "screenshots, edited exports, and AI generators.")
    if facts.get("digital_zoom") and facts["digital_zoom"] > 1:
        add("info", f"Digital zoom {facts['digital_zoom']}x recorded: interpolated pixels, reduced true detail.")
    return flags


# ----------------------------------------------------------------------------
# Questionnaire: what the user still needs to tell us
# ----------------------------------------------------------------------------

def build_questionnaire(facts: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Return the capture-context questions. Each has `known` (prefilled from
    metadata) and `required` (we need an answer because metadata is missing)."""

    def q(key, label, help_, required=False, known=None, type_="text", options=None):
        return {"key": key, "label": label, "help": help_, "required": required and known is None,
                "known": known, "type": type_, "options": options}

    device = " ".join(str(x) for x in (facts.get("make"), facts.get("model")) if x) or None
    res = f"{int(facts['width'])}x{int(facts['height'])}" if facts.get("width") and facts.get("height") else None
    qs = [
        q("provenance", "Is this the original file straight from the device that recorded it?",
          "Downloads from social media, files sent through messaging apps, screenshots and screen recordings lose "
          "metadata and quality. This changes how the evidence is weighed.", True, None, "select",
          ["Original file from the recording device", "Copied/AirDropped from the device (unaltered)",
           "Downloaded from social media / website", "Received via text/messaging app", "Screenshot",
           "Screen recording / filmed off another screen", "Edited/cropped/trimmed version", "Unknown"]),
        q("source_chain", "Where did this file come from? (who recorded it, how it reached you)",
          "e.g. 'My neighbor recorded it on her phone and texted it to me.'", True),
        q("device", "Recording device (make & model)", "e.g. iPhone 15 Pro, Samsung S24, DJI Mini 4, Ring doorbell, "
          "trail camera, GoPro, DSLR model.", True, device),
        q("lens_zoom", "Lens / zoom used", "Which camera (wide/ultra-wide/telephoto) and any zoom level (e.g. 3x, 10x "
          "digital). Unknown is fine.", False, facts.get("lens")),
        q("resolution", "Recording resolution", "e.g. 4K, 1080p, 12MP.", True, res),
    ]
    if kind == "video":
        fps = facts.get("fps_average") or facts.get("fps_nominal")
        qs.append(q("fps", "Frame rate (FPS)", "e.g. 30, 60, 120/240 slow-motion. Check the camera settings if unsure.",
                    True, f"{fps:g}" if fps else None))
        qs.append(q("stabilization", "Stabilization / special modes", "Gimbal, action mode, cinematic mode, "
                    "time-lapse, slow-motion?", False))
    else:
        qs.append(q("exposure", "Shutter speed / ISO / modes", "Night mode, HDR, long exposure, burst, Live Photo, "
                    "portrait mode? Unknown is fine.", False,
                    ", ".join(str(x) for x in (facts.get("exposure_time"), f"ISO {int(facts['iso'])}" if facts.get('iso') else None) if x) or None))
    qs += [
        q("capture_time", "Date and time it was captured (with time zone)", "Needed for sun/moon/star positions, "
          "satellite passes and flight-tracking checks.", True, facts.get("capture_time")),
        q("location", "Location (as precise as you're comfortable sharing)", "City/area, or coordinates. Used for "
          "sky-object and flight-path checks.", True, facts.get("gps_position")),
        q("direction", "Direction the camera was pointing and elevation", "e.g. 'facing northwest, about 30 degrees "
          "above the horizon'.", False, facts.get("gps_direction")),
        q("support", "Handheld, mounted, or moving?", "Tripod, handheld, in a moving car, from an aircraft, "
          "fixed security camera...", False, None, "select",
          ["Handheld", "Tripod / fixed mount", "Security / doorbell / trail camera", "From a moving vehicle",
           "From an aircraft / drone", "Unknown"]),
        q("through_glass", "Was it captured through a window, windshield, or screen?", "Glass causes reflections "
          "and ghost images.", False, None, "select", ["No", "Yes - window", "Yes - windshield", "Yes - other", "Unknown"]),
        q("conditions", "Lighting and weather", "Daylight, dusk, night; clear, cloudy, rain, fog, wind; "
          "any lights nearby?", False),
        q("naked_eye", "What did the witness see with their own eyes (not on the screen)?", "Did the object look "
          "the same to the naked eye? Brightness, colour, size compared to the moon at arm's length, sound?", False),
        q("duration", "How long was it seen, and how did it move/leave?", "", False),
        q("narrative", "Witness account / what you believe this shows",
          "Describe the event in your own words. The AI examines the media BEFORE reading this, so your account "
          "cannot bias its technical observations.", False, None, "textarea"),
    ]
    return qs
