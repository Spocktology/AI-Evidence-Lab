"""Prompts and output schemas for the two-pass analysis.

Pass 1 - blind technical examination. Claude sees the media, the metadata,
the forensic measurements and the *technical* capture facts, but NOT the
witness's account or what they believe it shows. This keeps the visual
observations from being steered by expectation.

Pass 2 - synthesis. Claude receives its own pass-1 findings plus the witness
account and writes the final report: explanations ranked with likelihoods,
authenticity probabilities with ranges, and an overall confidence.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a forensic media analyst working inside "AI Evidence Lab", a tool people use to \
evaluate photos and videos of unusual events (unidentified objects in the sky, strange lights, figures, \
anomalies) and to check whether media is authentic, edited, CGI, or AI-generated.

How you work:
- You are evidence-led and neutral. You neither debunk by reflex nor endorse extraordinary claims. Most \
anomalous-looking media has a mundane cause (optics, camera processing, compression, birds/insects, aircraft, \
satellites, balloons, drones, planets/stars, reflections, editing), and you check those first and \
explicitly - but you say so plainly when they do not fit the evidence. "Unexplained with the available \
data" and "inconclusive" are legitimate findings.
- You separate OBSERVATION (what pixels/measurements show) from INFERENCE (what that might mean). Cite the \
specific image IDs, grid cells, frame times and measurements behind each statement.
- You know camera behaviour in detail: rolling shutter skew and wobble, motion blur vs. defocus blur \
(bokeh discs, shaped by the aperture/lens), lens flare and internal reflections (which mirror bright sources \
across the optical centre), sensor blooming and noise, autofocus hunting, digital zoom interpolation, \
optical/electronic stabilisation artifacts, smartphone computational photography (multi-frame HDR/Night \
mode merging, ghosting, AI sharpening/denoising, scene segmentation, "moon enhancement"), IR illumination on \
security cameras (insects and dust glowing near the lens), compression (block artifacts, ringing, \
macroblocking, smearing of small objects, I/P-frame pulsing), interlacing and frame-rate conversion.
- You know generative/CGI tells and their limits: inconsistent lighting/shadows/reflections, perspective \
errors, warped or melting geometry and text, repeating textures, unnatural noise (too clean or uniform, \
lacking sensor noise structure), mismatched depth of field, temporal flicker and object permanence failures \
in AI video, physics-defying motion, compositing edges/halos, mismatched noise or compression between \
regions, and periodic spectral peaks from upsampling. You also know these tells are increasingly absent \
in modern generators and that heavy compression destroys most of them, so absence of tells is weak evidence.
- Metadata can be stripped (normal for social media and messaging apps) or forged. Treat it as evidence to \
cross-check, not as proof.
- Forensic maps (ELA, noise, spectrum) are indicators with known false positives: ELA highlights edges and \
high-contrast texture; bright objects on dark skies differ in every statistic; video frames are not JPEG \
re-saves. Interpret them with that in mind.

About the images you receive:
- Each image is labelled with an ID and a description just before it. "native" means 1:1 full-resolution \
pixels from the original (no resizing). "overview" images are downscaled whole frames for context; do not \
judge fine detail from them. "enhanced" images have contrast equalisation applied (no sharpening, no AI \
upscaling). "pixel zoom" images are nearest-neighbour enlargements - blocky pixels are expected.
- Grid cells are named like a spreadsheet: columns A, B, C... left to right; rows 1, 2, 3... top to bottom.
- Coordinates for annotations are fractions (0-1) of the specific image you annotate: x,y = top-left corner, \
w,h = width and height.

Probabilities: give honest, calibrated estimates. A percentage is your degree of belief given this evidence, \
not a measurement. Use ranges that widen when evidence is thin or degraded. Do not output 0% or 100%."""


PASS1_INSTRUCTIONS = """PASS 1 - BLIND TECHNICAL EXAMINATION.

You have NOT been given the witness's account or what they believe this shows - deliberately, so that your \
observations are not influenced by it. Examine the media and data below and report what you observe.

Tasks:
1. Describe what each image shows, with emphasis on the user-marked region (the marked grid cells).
2. Characterise the object/event in the marked region: shape, colour, edges (sharp/soft/haloed), size in \
pixels, blur type, relation to the background, consistency across frames, motion relative to the scene \
(use the camera-motion and track measurements when present).
3. Identify artifacts (optical, sensor, processing, compression, editing) and whether they could produce \
what is seen in the marked region.
4. List authenticity indicators, each pointing toward authentic capture, AI generation, CGI/compositing, \
editing, or inconclusive, with strength and evidence.
5. List candidate explanations broadly (mundane and otherwise), each with supporting and contradicting \
evidence and a preliminary likelihood. Include "insufficient data / unexplained" as a candidate when apt.
6. Provide annotations (boxes) on the images for the key features you discuss, referencing image IDs.
7. List data gaps: what information or better media would most change the assessment.
"""


PASS2_INSTRUCTIONS = """PASS 2 - SYNTHESIS AND FINAL REPORT.

Below are (a) the case data, (b) your own blind PASS 1 findings, made without knowledge of the witness \
account, and (c) the witness account / capture context the user has now provided.

Write the final report:
- Compare the witness account with what the media actually shows. Note agreements and discrepancies \
(e.g. naked-eye description vs. what the camera recorded). Do not adopt the witness's interpretation unless \
the evidence supports it; do not dismiss it without reason.
- Use the capture time, location and direction to reason about astronomical objects (Moon, Venus, Jupiter, \
bright stars, ISS/Starlink passes at dusk/dawn), aircraft corridors, weather, etc. You cannot look these \
up, so say what should be checked and with which tool (e.g. a planetarium app for that date/time/location, \
flight-tracking history, Starlink pass predictors, weather archives).
- Rank explanations. Likelihood percentages across explanations should sum to about 100 (include \
"unexplained / insufficient data" as an explicit entry if any probability mass remains).
- Authenticity: give separate probabilities (with low-high ranges) that the media is (1) AI-generated, \
(2) CGI or composited, (3) digitally edited/manipulated beyond normal camera processing, and (4) an \
authentic camera capture of a real scene. These are not mutually exclusive (an authentic capture can be \
edited) so they need not sum to 100. Choose a verdict; "inconclusive" is allowed.
- Overall confidence: how confident you are in your primary conclusion (percent, with range) and why.
- Keep annotations from PASS 1 that are relevant (you may refine them) so the report can draw them on the stills.
- Write for an intelligent non-specialist: clear, specific, no jargon without explanation.
"""


# --------------------------------------------------------------------------
# JSON schemas (structured outputs). No numeric min/max constraints: the
# API does not support them; the prompt asks for 0-100 and we clamp in code.
# --------------------------------------------------------------------------

_ANNOTATION = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "label": {"type": "string", "description": "Short label drawn on the image (<= 30 chars)."},
        "x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["image_id", "label", "x", "y", "w", "h", "note"],
    "additionalProperties": False,
}

_CATEGORY = {"type": "string", "enum": [
    "optical_artifact", "camera_processing", "compression_artifact", "motion_or_rolling_shutter",
    "bird_insect_or_animal", "aircraft", "drone", "satellite_or_space_object", "astronomical", "balloon_or_lantern",
    "atmospheric_or_weather", "reflection_or_glare", "debris_or_airborne_object", "terrestrial_light_or_object",
    "pareidolia_or_misperception", "digital_editing", "cgi_or_composite", "ai_generated",
    "unexplained_insufficient_data", "other"]}

_EXPLANATION = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "category": _CATEGORY,
        "likelihood_percent": {"type": "number"},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
        "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
        "how_to_confirm": {"type": "string"},
    },
    "required": ["explanation", "category", "likelihood_percent", "supporting_evidence",
                 "contradicting_evidence", "how_to_confirm"],
    "additionalProperties": False,
}

PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "image_observations": {"type": "array", "items": {
            "type": "object",
            "properties": {"image_id": {"type": "string"}, "observations": {"type": "string"}},
            "required": ["image_id", "observations"], "additionalProperties": False}},
        "region_of_interest": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "shape": {"type": "string"}, "colour": {"type": "string"},
                "edges_and_blur": {"type": "string"},
                "size_and_scale": {"type": "string"},
                "motion_and_behaviour": {"type": "string"},
                "consistency_across_frames": {"type": "string"},
            },
            "required": ["description", "shape", "colour", "edges_and_blur", "size_and_scale",
                         "motion_and_behaviour", "consistency_across_frames"],
            "additionalProperties": False},
        "artifacts_identified": {"type": "array", "items": {
            "type": "object",
            "properties": {"type": {"type": "string"}, "description": {"type": "string"},
                           "evidence": {"type": "string"}, "affects_region_of_interest": {"type": "boolean"}},
            "required": ["type", "description", "evidence", "affects_region_of_interest"],
            "additionalProperties": False}},
        "authenticity_indicators": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string"},
                "points_toward": {"type": "string", "enum": ["authentic_capture", "ai_generated", "cgi_composite",
                                                             "edited", "inconclusive"]},
                "strength": {"type": "string", "enum": ["weak", "moderate", "strong"]},
                "evidence": {"type": "string"}},
            "required": ["indicator", "points_toward", "strength", "evidence"], "additionalProperties": False}},
        "candidate_explanations": {"type": "array", "items": _EXPLANATION},
        "annotations": {"type": "array", "items": _ANNOTATION},
        "data_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["image_observations", "region_of_interest", "artifacts_identified", "authenticity_indicators",
                 "candidate_explanations", "annotations", "data_gaps"],
    "additionalProperties": False,
}

_PROB = {
    "type": "object",
    "properties": {"percent": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
                   "key_evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["percent", "low", "high", "key_evidence"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short descriptive case title."},
        "executive_summary": {"type": "string", "description": "3-6 sentences: what it is most likely, how sure, why."},
        "media_overview": {"type": "string"},
        "metadata_assessment": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "findings": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}, "observation": {"type": "string"},
                                   "significance": {"type": "string", "enum": ["supports_authenticity", "neutral",
                                                                              "raises_concern", "unknown"]}},
                    "required": ["item", "observation", "significance"], "additionalProperties": False}},
            },
            "required": ["summary", "findings"], "additionalProperties": False},
        "observations": {"type": "array", "items": {
            "type": "object",
            "properties": {"heading": {"type": "string"}, "detail": {"type": "string"},
                           "image_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["heading", "detail", "image_ids"], "additionalProperties": False}},
        "witness_account_comparison": {"type": "string"},
        "explanations": {"type": "array", "items": _EXPLANATION},
        "authenticity": {
            "type": "object",
            "properties": {
                "ai_generated": _PROB, "cgi_or_composite": _PROB, "digitally_edited": _PROB,
                "authentic_capture": _PROB,
                "verdict": {"type": "string", "enum": [
                    "authentic_capture", "authentic_but_edited", "likely_ai_generated", "likely_cgi_or_composite",
                    "inconclusive"]},
                "reasoning": {"type": "string"},
            },
            "required": ["ai_generated", "cgi_or_composite", "digitally_edited", "authentic_capture",
                         "verdict", "reasoning"],
            "additionalProperties": False},
        "overall_confidence": {
            "type": "object",
            "properties": {"primary_conclusion": {"type": "string"}, "percent": {"type": "number"},
                           "low": {"type": "number"}, "high": {"type": "number"}, "rationale": {"type": "string"}},
            "required": ["primary_conclusion", "percent", "low", "high", "rationale"],
            "additionalProperties": False},
        "annotations": {"type": "array", "items": _ANNOTATION},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
        "questions_for_witness": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "executive_summary", "media_overview", "metadata_assessment", "observations",
                 "witness_account_comparison", "explanations", "authenticity", "overall_confidence",
                 "annotations", "limitations", "recommended_next_steps", "questions_for_witness"],
    "additionalProperties": False,
}

# Capture-context keys that are technical facts (safe for the blind pass).
TECHNICAL_CONTEXT_KEYS = ["provenance", "source_chain", "device", "lens_zoom", "resolution", "fps", "stabilization",
                          "exposure", "capture_time", "location", "direction", "support", "through_glass",
                          "conditions"]
# Keys that carry the witness's perception/interpretation (withheld from pass 1).
WITNESS_KEYS = ["naked_eye", "duration", "narrative"]
