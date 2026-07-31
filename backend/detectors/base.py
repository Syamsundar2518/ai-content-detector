"""
detectors/base.py  (NEW FILE)
-------------------------------
Shared infrastructure for calling AI detection models through Hugging
Face's CURRENT Router API (https://router.huggingface.co).

WHY THIS FILE EXISTS
Hugging Face deleted its old endpoint (https://api-inference.huggingface.co
now returns HTTP 410 Gone) and replaced it with a new one:
https://router.huggingface.co. This file contains the ONLY code in the
whole project that knows that URL and how to call it.

Both text_detector.py and image_detector.py need the exact same things:
  - Read an API token from environment variables
  - Send a request to a Hugging Face model through the Router
  - Retry automatically if the model is still "waking up" (cold start)
  - Turn network/API problems into clear, typed, human-readable errors
  - Log what happened, so problems are easy to diagnose later

Putting this ONCE here (instead of copy-pasting it into every detector)
is what makes the system modular: if you ever want to use a different
AI provider (a paid API, a self-hosted model, etc.), you write one new
small class with the same `call()` method and swap it in. Nothing in
app.py, the detector functions' signatures, or the frontend needs to
change.
"""

import os
import time
import logging
import requests

logger = logging.getLogger("ai_content_detector")

# Hugging Face's current, supported endpoint for classic "pipeline" style
# models (text-classification, image-classification, etc.) served through
# their free hf-inference provider.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 6
REQUEST_TIMEOUT_SECONDS = 30


class DetectionEngineError(Exception):
    """
    Raised whenever a detection engine cannot produce a result.

    `reason` is always a short, safe-to-display-to-the-user message.
    `kind` categorizes the failure so calling code can react differently
    (e.g. show a "Setup needed" message for a missing token, vs a
    "Could not analyze" message for a temporary API problem).
    """
    def __init__(self, reason: str, kind: str = "api_error", status_code: int = None):
        self.reason = reason
        self.kind = kind  # "missing_token" | "network_error" | "timeout" | "auth_error" | "rate_limited" | "api_error"
        self.status_code = status_code
        super().__init__(reason)


class HuggingFaceRouterEngine:
    """
    A small, reusable client for Hugging Face's current Router API.

    Usage:
        engine = HuggingFaceRouterEngine(model_id="some-org/some-model")
        raw_json = engine.call(json_payload={"inputs": "some text"})   # for text
        raw_json = engine.call(raw_bytes=image_bytes)                  # for images

    To point at a different model, just change `model_id` — no other
    code changes. To swap to a different AI provider entirely, write a
    new class with the same `call()` method and use it instead.
    """

    def __init__(self, model_id: str, token_env_var: str = "HUGGINGFACE_API_TOKEN"):
        self.model_id = model_id
        self.token_env_var = token_env_var
        self.api_url = f"{HF_ROUTER_BASE_URL}/{model_id}"

    def _get_token(self) -> str:
        token = os.getenv(self.token_env_var, "").strip()
        if not token:
            raise DetectionEngineError(
                reason=(
                    f"No Hugging Face API token found. Set {self.token_env_var} "
                    f"in your backend/.env file (see .env.example)."
                ),
                kind="missing_token",
            )
        return token

    def call(self, *, json_payload: dict = None, raw_bytes: bytes = None):
        """
        Sends a request to the configured model. Provide EITHER
        json_payload (for text-style inputs) OR raw_bytes (for images).
        Retries automatically if the model is still loading on Hugging
        Face's servers. Raises DetectionEngineError on any failure —
        callers should catch this and turn it into a user-facing message.
        """
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(
                "Calling Hugging Face model '%s' (attempt %s/%s)",
                self.model_id, attempt, MAX_RETRIES,
            )
            try:
                if raw_bytes is not None:
                    response = requests.post(
                        self.api_url, headers=headers, data=raw_bytes,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                else:
                    response = requests.post(
                        self.api_url, headers=headers, json=json_payload,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
            except requests.exceptions.Timeout:
                logger.warning("Request to model '%s' timed out.", self.model_id)
                raise DetectionEngineError(
                    reason="The AI model took too long to respond. Please try again.",
                    kind="timeout",
                )
            except requests.exceptions.RequestException as exc:
                logger.error("Network error calling model '%s': %s", self.model_id, exc)
                raise DetectionEngineError(
                    reason="Could not reach the AI detection service. Check your internet connection.",
                    kind="network_error",
                )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 503:
                # The model is "cold" and Hugging Face is loading it — worth retrying.
                logger.info(
                    "Model '%s' is still loading, retrying in %ss...",
                    self.model_id, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            if response.status_code == 401:
                logger.error("Hugging Face rejected the API token (401) for model '%s'.", self.model_id)
                raise DetectionEngineError(
                    reason="Your Hugging Face API token was rejected. Double-check it in backend/.env.",
                    kind="auth_error",
                    status_code=401,
                )

            if response.status_code == 429:
                logger.warning("Hugging Face rate limit hit (429) for model '%s'.", self.model_id)
                raise DetectionEngineError(
                    reason="The free usage limit was reached. Please wait a minute and try again.",
                    kind="rate_limited",
                    status_code=429,
                )

            # Any other status (404 = model not found/unsupported on this
            # provider, 400 = bad request, 500 = server-side problem, etc.)
            logger.error(
                "Hugging Face returned status %s for model '%s': %s",
                response.status_code, self.model_id, response.text[:300],
            )
            raise DetectionEngineError(
                reason=(
                    f"The AI detection service returned an unexpected error "
                    f"(code {response.status_code}). Please try again shortly."
                ),
                kind="api_error",
                status_code=response.status_code,
            )

        logger.error("Model '%s' did not become ready after %s retries.", self.model_id, MAX_RETRIES)
        raise DetectionEngineError(
            reason="The AI model is taking unusually long to start. Please try again in a minute.",
            kind="timeout",
        )
