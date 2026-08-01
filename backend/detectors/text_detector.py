"""
text_detector.py  (REPLACEMENT — v5)
--------------------------------------
WHAT CHANGED FROM v4 (and why):
  Two different Hugging Face community models were tried for text
  detection, and both failed in production. This isn't a coincidence —
  Hugging Face's free serverless Inference API only reliably serves
  large, popular, "warm" models. Free/community AI-text-detector models
  are exactly the kind of niche, low-traffic models that routinely fail
  to load on it (unsupported architectures, no assigned provider, cold
  starts that exceed timeouts, etc.). Rather than guess at a third
  Hugging Face model, text detection has moved to a different kind of
  service entirely.

  NEW ENGINE: Sapling.ai AI Detector API
  Verified directly against Sapling's official developer docs
  (https://sapling.ai/docs/api/detector/) before using it:
    - It is a dedicated, purpose-built AI-text-detection PRODUCT — not a
      community model riding on shared inference infrastructure — so it
      does not have the "model fails to load" failure mode.
    - Actively maintained: their docs show updates as recent as June
      2026, and they publicly advertise detection coverage for current
      models (GPT-4o/GPT-5, Claude, Gemini).
    - Has a real free tier: a rate-limited developer API key (50,000
      characters/24 hours) is available with just a free account — no
      credit card required to get started.
    - Confirmed exact request/response format directly from their docs
      (see _extract_ai_probability below) rather than guessing.

MODEL LIMITATIONS (be upfront about these):
  - English-focused, like most detectors of this kind.
  - Free tier is rate-limited (50,000 characters/24 hours) — fine for
    personal or demo use, but a paid key would be needed for high volume.
  - No detector — free or paid — is 100% accurate. Sapling's own docs
    say the same: results are an estimate, not proof.

WHY THIS DESIGN IS MODULAR:
  All the networking, retry, and error-handling logic lives in
  detectors/base.py's SaplingTextEngine class — this file only knows
  "send text, get a 0-1 score back." If you ever want to switch
  providers again, you write one new engine class in base.py with a
  matching call() method and change ONE line here. app.py and the
  frontend never need to change.

API CONTRACT (unchanged — the frontend needs no changes):
  detect_text(text) -> {
      "result": "Likely AI Generated" | "Likely Human Created"
                 | "Setup needed" | "Not enough text" | "Could not analyze",
      "confidence": 0-100,
      "details": { ... }
  }
"""

import logging

from .base import SaplingTextEngine, DetectionEngineError

logger = logging.getLogger("ai_content_detector")

_engine = SaplingTextEngine()


def _extract_ai_probability(raw_response: dict) -> float:
    """
    Turns Sapling's raw JSON response into a single 0.0-1.0 "probability
    this text is AI-generated" number.

    Confirmed response shape from Sapling's official docs:
        {
            "score": 0.98,
            "sentence_scores": [
                {"sentence": "...", "score": 0.99},
                ...
            ]
        }
    "score" is already exactly what we need: 0 = confidently human,
    1 = confidently AI. Raises ValueError if the shape is not
    recognized, so the caller can turn that into a clear, friendly
    error instead of crashing.
    """
    if not isinstance(raw_response, dict) or "score" not in raw_response:
        raise ValueError(f"Unrecognized response shape: {raw_response}")

    try:
        return float(raw_response["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'score' field was not a number: {raw_response.get('score')}") from exc


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

    # Sapling's free-tier developer key is capped at 50,000 characters
    # per request; we trim well below that to keep requests fast.
    try:
        raw_response = _engine.call(text=text[:20000])
        ai_probability = _extract_ai_probability(raw_response)

    except DetectionEngineError as exc:
        # Errors from base.py are already safe, human-readable messages.
        logger.warning("Text detection failed (%s): %s", exc.kind, exc.reason)
        result_label = "Setup needed" if exc.kind == "missing_token" else "Could not analyze"
        return {"result": result_label, "confidence": 0, "details": {"reason": exc.reason}}

    except ValueError as exc:
        # The service responded, but not in a shape we understood.
        logger.error("Unexpected response shape from Sapling: %s", exc)
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "Received an unexpected response from the AI detection service. Please try again."},
        }

    label = "Likely AI Generated" if ai_probability >= 0.5 else "Likely Human Created"
    confidence = round(ai_probability * 100) if ai_probability >= 0.5 else round((1 - ai_probability) * 100)

    logger.info("Text detection result: %s (%s%%)", label, confidence)

    return {
        "result": label,
        "confidence": confidence,
        "details": {
            "method": "sapling-ai-detector",
            "model_ai_score_percent": round(ai_probability * 100, 1),
        },
    }
