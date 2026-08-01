"""
image_detector.py  (REPLACEMENT — v4)
----------------------------------------
WHAT CHANGED FROM v3 (bugfix):
Hugging Face's current Router endpoint requires an explicit Content-Type
header on raw image uploads (the old endpoint guessed it automatically).
Without it, every image request failed with:
    HTTP 400 - "No content type provided and no default one configured."
detect_image() now accepts an optional `content_type` parameter (e.g.
"image/png") so callers that know the real file type can pass it along.
It defaults to "image/jpeg" for backward compatibility — video_detector.py
doesn't need any changes, since it always encodes frames as JPEG already.

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
  detect_image(image_bytes, content_type="image/jpeg") -> {
      "result": "Likely AI Generated" | "Likely Human Created"
                 | "Setup needed" | "Could not analyze",
      "confidence": 0-100,
      "details": { ... }
  }
"""

import logging

from .base import HuggingFaceRouterEngine, DetectionEngineError

logger = logging.getLogger("ai_content_detector")

# The model this detector uses. To swap models later, change ONLY this line.
IMAGE_MODEL_ID = "Organika/sdxl-detector"

_engine = HuggingFaceRouterEngine(model_id=IMAGE_MODEL_ID)

AI_LABEL_HINTS = ("artificial", "ai", "fake", "generated", "label_1")


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


def detect_image(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Main function called by app.py (and by video_detector.py, once per
    sampled frame). Always returns a plain dict matching the API contract
    above — never raises.

    `content_type` should match the real image format when known (e.g.
    "image/png" for a .png upload) — this is required by Hugging Face's
    current Router. Defaults to "image/jpeg" for callers that don't
    specify it (like video frame sampling, which is always JPEG).
    """
    try:
        raw_response = _engine.call(raw_bytes=image_bytes, content_type=content_type)
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
        },
    }
