"""
image_detector.py  (v2 — REAL AI MODEL)
------------------------------------------
WHAT CHANGED FROM BEFORE:
The old version measured pixel noise, color spread, and edge sharpness
using math. This version instead sends the image to a real, pre-trained
AI model that was specifically trained to recognize AI-generated images.

MODEL USED: Organika/sdxl-detector
  - Free, open-source model on Hugging Face.
  - Trained to tell apart real photos from AI-generated images (like
    those made with Stable Diffusion / SDXL).
  - Runs on Hugging Face's servers — not on Render — so our free-tier
    backend stays small, fast, and doesn't need to load a big model
    into memory itself.

WHY THIS MATTERS FOR RENDER'S FREE TIER:
Render's free plan only gives ~512MB RAM. AI image models are often
hundreds of MB to multiple GB — way too big to load directly. By calling
Hugging Face's API instead, our backend just sends the image bytes over
the internet and gets a small JSON answer back. Much lighter.

This file is also reused by video_detector.py — each sampled video frame
is passed through the exact same function below.
"""

import os
import time
import requests

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Organika/sdxl-detector"

MAX_RETRIES = 3
RETRY_DELAY = 6


def _call_huggingface(image_bytes: bytes, hf_token: str):
    """
    Sends raw image bytes to the Hugging Face model.
    Retries automatically if the model is "waking up" (cold start).
    """
    headers = {"Authorization": f"Bearer {hf_token}"}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                HF_MODEL_URL,
                headers=headers,
                data=image_bytes,
                timeout=25,
            )
        except requests.exceptions.RequestException:
            return None

        if response.status_code == 200:
            return response.json()

        if response.status_code == 503:
            time.sleep(RETRY_DELAY)
            continue

        return {"__error__": response.status_code}

    return {"__error__": "timeout_after_retries"}


def detect_image(image_bytes: bytes):
    """
    Main function called by app.py (and by video_detector.py per frame).
    Returns: { "result": "...", "confidence": 87, "details": {...} }
    """
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()

    if not hf_token:
        return {
            "result": "Setup needed",
            "confidence": 0,
            "details": {
                "reason": "No Hugging Face API token found. Add HUGGINGFACE_API_TOKEN "
                          "to your backend/.env file (see .env.example)."
            },
        }

    raw = _call_huggingface(image_bytes, hf_token)

    if raw is None:
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "Could not reach Hugging Face. Check your internet connection."},
        }

    if isinstance(raw, dict) and "__error__" in raw:
        code = raw["__error__"]
        if code == 401:
            reason = "Your Hugging Face API token was rejected. Double-check it in backend/.env."
        elif code == 429:
            reason = "Hugging Face's free tier rate limit was hit. Please wait a minute and try again."
        else:
            reason = f"The AI model service returned an error (code: {code}). Please try again shortly."
        return {"result": "Could not analyze", "confidence": 0, "details": {"reason": reason}}

    # Expected successful shape:
    # [{"label": "artificial", "score": 0.92}, {"label": "human", "score": 0.08}]
    try:
        if not isinstance(raw, list) or not raw:
            raise ValueError("Unexpected response shape")

        ai_entry = next(
            (d for d in raw if any(k in d["label"].lower() for k in ("artificial", "ai", "fake", "generated"))),
            None,
        )
        if not ai_entry:
            raise ValueError("Unexpected response shape")

        ai_probability = ai_entry["score"]
    except Exception:
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "Received an unexpected response from the AI model. Please try again."},
        }

    label = "Likely AI Generated" if ai_probability >= 0.5 else "Likely Human Created"
    confidence = round(ai_probability * 100) if ai_probability >= 0.5 else round((1 - ai_probability) * 100)

    return {
        "result": label,
        "confidence": confidence,
        "details": {
            "method": "huggingface-sdxl-detector",
            "model_ai_score_percent": round(ai_probability * 100, 1),
        },
    }
