"""
text_detector.py  (REPLACEMENT — v3)
--------------------------------------
WHAT CHANGED FROM v2:
  1. No longer calls Hugging Face directly — it now goes through the
     shared `HuggingFaceRouterEngine` in base.py, which uses the
     CURRENT, supported https://router.huggingface.co endpoint (the old
     one this file used before, api-inference.huggingface.co, has been
     shut down by Hugging Face and returns errors — that's why detection
     stopped working).
  2. Swapped the underlying AI model. The old model
     (Hello-SimpleAI/chatgpt-detector-roberta) is an unmaintained,
     GPT-2-era research model. It's replaced with
     desklib/ai-text-detector-v1.01 — a modern, actively-used model that
     currently leads the independent RAID Benchmark for AI-text
     detection, is MIT-licensed, and is confirmed to run on Hugging
     Face's current Inference Providers.

MODEL LIMITATIONS (be upfront about these):
  - Trained and evaluated primarily on English text.
  - Like every AI-text detector, it can be fooled by heavy paraphrasing
    or adversarial rewriting, and can occasionally flag simple, very
    formulaic human writing as AI. Treat results as an estimate.

WHY THIS DESIGN IS MODULAR:
  This file doesn't know any HTTP/networking details — that all lives in
  base.py. If Hugging Face ever changes again, or you want to switch to
  a different provider (e.g. a paid API), you only need to change
  base.py (or write a new engine with the same `call()` method) —
  nothing here or in app.py has to change.

API CONTRACT (unchanged from before — the frontend needs no changes):
  detect_text(text) -> {
      "result": "Likely AI Generated" | "Likely Human Created"
                 | "Setup needed" | "Not enough text" | "Could not analyze",
      "confidence": 0-100,
      "details": { ... }
  }
"""

import logging

from .base import HuggingFaceRouterEngine, DetectionEngineError

logger = logging.getLogger("ai_content_detector")

# The model this detector uses. To swap models later, change ONLY this line.
TEXT_MODEL_ID = "desklib/ai-text-detector-v1.01"

_engine = HuggingFaceRouterEngine(model_id=TEXT_MODEL_ID)

# Label text that this (and most similar) models use to mean "AI-written".
# Kept as a list because different model versions/providers sometimes use
# slightly different label spellings (e.g. "AI", "LABEL_1", "generated").
AI_LABEL_HINTS = ("ai", "generated", "machine", "artificial", "fake", "label_1")


def _extract_ai_probability(raw_response):
    """
    Turns the model's raw JSON response into a single 0.0-1.0 "probability
    this text is AI-generated" number. Handles the standard Hugging Face
    text-classification response shape:
        [[{"label": "...", "score": 0.9}, {"label": "...", "score": 0.1}]]
    or, less commonly, a flat list without the extra wrapping:
        [{"label": "...", "score": 0.9}, {"label": "...", "score": 0.1}]
    Raises ValueError if the shape is not recognized, so the caller can
    turn that into a clear, friendly error instead of crashing.
    """
    if isinstance(raw_response, list) and raw_response and isinstance(raw_response[0], list):
        scores = raw_response[0]
    elif isinstance(raw_response, list) and raw_response and isinstance(raw_response[0], dict):
        scores = raw_response
    else:
        raise ValueError(f"Unrecognized response shape: {type(raw_response)}")

    ai_entry = next(
        (s for s in scores if any(hint in str(s.get("label", "")).lower() for hint in AI_LABEL_HINTS)),
        None,
    )
    if ai_entry is None:
        raise ValueError(f"Could not find an AI-labeled score in: {scores}")

    return float(ai_entry["score"])


def detect_text(text: str) -> dict:
    """
    Main function called by app.py. Always returns a plain dict matching
    the API contract above — never raises, so app.py doesn't need any
    special error handling beyond what it already has.
    """
    text = (text or "").strip()

    if len(text) < 5:
        return {
            "result": "Not enough text",
            "confidence": 0,
            "details": {"reason": "Please enter at least a few sentences."},
        }

    try:
        raw_response = _engine.call(json_payload={"inputs": text[:2000]})
        ai_probability = _extract_ai_probability(raw_response)

    except DetectionEngineError as exc:
        # Errors from base.py are already safe, human-readable messages.
        logger.warning("Text detection failed (%s): %s", exc.kind, exc.reason)
        result_label = "Setup needed" if exc.kind == "missing_token" else "Could not analyze"
        return {"result": result_label, "confidence": 0, "details": {"reason": exc.reason}}

    except ValueError as exc:
        # The model responded, but not in a shape we understood.
        logger.error("Unexpected response shape from '%s': %s", TEXT_MODEL_ID, exc)
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "Received an unexpected response from the AI model. Please try again."},
        }

    label = "Likely AI Generated" if ai_probability >= 0.5 else "Likely Human Created"
    confidence = round(ai_probability * 100) if ai_probability >= 0.5 else round((1 - ai_probability) * 100)

    logger.info("Text detection result: %s (%s%%)", label, confidence)

    return {
        "result": label,
        "confidence": confidence,
        "details": {
            "method": f"huggingface-router:{TEXT_MODEL_ID}",
            "model_ai_score_percent": round(ai_probability * 100, 1),
        },
    }
