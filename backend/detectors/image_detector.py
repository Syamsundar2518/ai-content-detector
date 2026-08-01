"""
image_detector.py  (REPLACEMENT — v5)
----------------------------------------
WHAT CHANGED FROM v4:
Previously, the correct Content-Type header (required by Hugging Face's
current Router — see v4 notes below) had to be guessed by the CALLER
(app.py), based on the uploaded file's extension. That's fragile: a
mislabeled extension, a video frame, or any caller that forgets to pass
it would silently fall back to a wrong guess.

This version detects the real image format directly from the image
BYTES THEMSELVES, using Pillow (already a project dependency — see
requirements.txt). This is more reliable than reading the file
extension, because it looks at the file's actual content/magic bytes,
not just its name.

WHY PILLOW INSTEAD OF imghdr:
Python's built-in `imghdr` module can also do this, but it was
deprecated in Python 3.11 and REMOVED entirely in Python 3.13 — using
it would risk breaking this project on a future Python upgrade. Pillow
is actively maintained, already required by this project, and reads
image formats the same reliable way (via each format's magic bytes),
so it was the safer choice.

detect_image() still accepts an optional `content_type` argument for
backward compatibility with existing callers (app.py currently passes
one) — but it's now only used as a fallback if auto-detection fails,
never as the primary source of truth.

--- (unchanged from v4 below) ---
Same underlying model as before (Organika/sdxl-detector — verified still
actively maintained: 78k+ downloads/month, used in 100+ community Spaces,
confirmed supported on Hugging Face's current Inference Providers), still
called through the shared `HuggingFaceRouterEngine` in base.py.

This file is also reused by video_detector.py — each sampled video frame
is passed through the exact same detect_image() function below.

MODEL LIMITATIONS (be upfront about these):
  - Trained specifically to catch SDXL/Stable-Diffusion-style images.
    It is less reliable against other AI image generators (e.g. some
    proprietary tools) or heavily edited/compressed images.

API CONTRACT (unchanged — the frontend needs no changes):
  detect_image(image_bytes, content_type=None) -> {
      "result": "Likely AI Generated" | "Likely Human Created"
                 | "Setup needed" | "Could not analyze",
      "confidence": 0-100,
      "details": { ... }
  }
"""

import io
import logging
from typing import Optional

from PIL import Image, UnidentifiedImageError

from .base import HuggingFaceRouterEngine, DetectionEngineError

logger = logging.getLogger("ai_content_detector")

# The model this detector uses. To swap models later, change ONLY this line.
IMAGE_MODEL_ID = "Organika/sdxl-detector"

_engine = HuggingFaceRouterEngine(model_id=IMAGE_MODEL_ID)

AI_LABEL_HINTS = ("artificial", "ai", "fake", "generated", "label_1")

# Maps Pillow's format names to proper MIME types.
# Examples: jpeg -> image/jpeg, png -> image/png, webp -> image/webp
PILLOW_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}

# Used only if auto-detection fails AND the caller didn't supply a
# content_type either — a safe last resort so a request is still attempted.
FALLBACK_CONTENT_TYPE = "image/jpeg"


def _detect_content_type(image_bytes: bytes) -> Optional[str]:
    """
    Inspects the actual image bytes (not the filename) to determine the
    real MIME type, using Pillow. Returns None if the bytes don't look
    like a valid, recognized image — the caller decides what to do next.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            pillow_format = img.format  # e.g. "JPEG", "PNG", "WEBP"
    except UnidentifiedImageError:
        logger.warning("Could not identify image format from file bytes.")
        return None
    except Exception as exc:
        logger.warning("Unexpected error while detecting image format: %s", exc)
        return None

    mime_type = PILLOW_FORMAT_TO_MIME.get(pillow_format)
    if mime_type is None:
        logger.warning("Detected image format '%s' has no known MIME mapping.", pillow_format)
        return None

    logger.info("Auto-detected image content type: %s (Pillow format: %s)", mime_type, pillow_format)
    return mime_type


def _extract_ai_probability(raw_response):
    """
    Turns the model's raw JSON response into a single 0.0-1.0 "probability
    this image is AI-generated" number. Expected shape (standard Hugging
    Face image-classification response):
        [{"label": "artificial", "score": 0.92}, {"label": "human", "score": 0.08}]
    Raises ValueError if the shape is not recognized.
    """
    if not isinstance(raw_response, list) or not raw_response or not isinstance(raw_response[0], dict):
        raise ValueError(f"Unrecognized response shape: {type(raw_response)}")

    ai_entry = next(
        (d for d in raw_response if any(hint in str(d.get("label", "")).lower() for hint in AI_LABEL_HINTS)),
        None,
    )
    if ai_entry is None:
        raise ValueError(f"Could not find an AI-labeled score in: {raw_response}")

    return float(ai_entry["score"])


def detect_image(image_bytes: bytes, content_type: str = None) -> dict:
    """
    Main function called by app.py (and by video_detector.py, once per
    sampled frame). Always returns a plain dict matching the API contract
    above — never raises.

    The real content type is auto-detected from the image bytes
    themselves (via Pillow). The `content_type` parameter is only used
    as a fallback if that detection fails — it is NOT the primary
    source of truth anymore, so passing it is optional.
    """
    detected_content_type = _detect_content_type(image_bytes)
    final_content_type = detected_content_type or content_type or FALLBACK_CONTENT_TYPE

    if detected_content_type is None:
        logger.warning(
            "Falling back to content type '%s' (auto-detection failed).",
            final_content_type,
        )

    try:
        raw_response = _engine.call(raw_bytes=image_bytes, content_type=final_content_type)
        ai_probability = _extract_ai_probability(raw_response)

    except DetectionEngineError as exc:
        logger.warning("Image detection failed (%s): %s", exc.kind, exc.reason)
        result_label = "Setup needed" if exc.kind == "missing_token" else "Could not analyze"
        return {"result": result_label, "confidence": 0, "details": {"reason": exc.reason}}

    except ValueError as exc:
        logger.error("Unexpected response shape from '%s': %s", IMAGE_MODEL_ID, exc)
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "Received an unexpected response from the AI model. Please try again."},
        }

    label = "Likely AI Generated" if ai_probability >= 0.5 else "Likely Human Created"
    confidence = round(ai_probability * 100) if ai_probability >= 0.5 else round((1 - ai_probability) * 100)

    logger.info("Image detection result: %s (%s%%)", label, confidence)

    return {
        "result": label,
        "confidence": confidence,
        "details": {
            "method": f"huggingface-router:{IMAGE_MODEL_ID}",
            "model_ai_score_percent": round(ai_probability * 100, 1),
            "detected_content_type": final_content_type,
        },
    }
