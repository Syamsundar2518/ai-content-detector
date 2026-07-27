"""
text_detector.py  (v2 — REAL AI MODEL)
----------------------------------------
WHAT CHANGED FROM BEFORE:
The old version guessed using math (sentence length patterns, word
repetition, etc). This version instead asks a real, pre-trained AI model
to make the decision. We do this by sending the text to Hugging Face's
free "Inference API" — think of it like a web request to a robot that
has already been trained to spot AI writing.

MODEL USED: Hello-SimpleAI/chatgpt-detector-roberta
  - This is a free, open-source model made specifically to detect
    ChatGPT-style AI text vs human text.
  - It runs on Hugging Face's servers, not your computer or Render — so
    your app stays small and fast (important for Render's free tier).

WHAT THIS FILE DOES, STEP BY STEP:
  1. Reads your free Hugging Face API token from the .env file.
  2. Sends the text to the model's API endpoint.
  3. Handles two special situations gracefully:
       a) The model is "cold" (asleep) and needs ~20 seconds to wake up
          — we wait and retry automatically instead of failing.
       b) The API is temporarily unavailable — we return a clear,
          honest error message instead of crashing.
  4. Turns the model's raw score into our friendly label + percentage.
"""

import os
import time
import requests

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"

# How many times we'll retry if the model is still "waking up"
MAX_RETRIES = 3
# How long to wait (seconds) between retries
RETRY_DELAY = 6


def _call_huggingface(text: str, hf_token: str):
    """
    Sends the text to the Hugging Face model and returns the raw response,
    automatically retrying if the model is still loading ("cold start").
    """
    headers = {"Authorization": f"Bearer {hf_token}"}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                HF_MODEL_URL,
                headers=headers,
                json={"inputs": text[:2000]},  # trim very long text to keep requests fast
                timeout=25,
            )
            print("Status Code:", response.status_code)
            print("Response:", response.text)
        except requests.exceptions.RequestException as e:
            print("Hugging Face Request Error:", e)
            return None  # network problem — no internet, DNS issue, etc.

        # 200 = success
        if response.status_code == 200:
            return response.json()

        # 503 usually means "model is loading, try again shortly"
        if response.status_code == 503:
            time.sleep(RETRY_DELAY)
            continue

        # 401 = bad/missing token, 429 = rate limited, or anything else
        return {"__error__": response.status_code}

    return {"__error__": "timeout_after_retries"}


def detect_text(text: str):
    """
    Main function called by app.py.
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

    if len(text.strip()) < 5:
        return {
            "result": "Not enough text",
            "confidence": 0,
            "details": {"reason": "Please enter at least a few sentences."},
        }

    raw = _call_huggingface(text, hf_token)

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
    # [[{"label": "ChatGPT", "score": 0.9}, {"label": "Human", "score": 0.1}]]
    try:
        scores = raw[0] if isinstance(raw, list) else None
        ai_entry = next(
            (s for s in scores if s["label"].lower() in ("chatgpt", "fake", "ai", "generated")),
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
            "method": "huggingface-chatgpt-detector-roberta",
            "model_ai_score_percent": round(ai_probability * 100, 1),
        },
    }
