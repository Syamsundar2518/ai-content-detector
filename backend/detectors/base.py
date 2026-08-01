"""
detectors/base.py  (REPLACEMENT — v2)
-------------------------------
Shared infrastructure for calling AI detection services.

WHAT CHANGED IN THIS UPDATE:
Added a second engine, `SaplingTextEngine`, alongside the existing
`HuggingFaceRouterEngine`. This is used ONLY by text_detector.py now.
image_detector.py (and therefore video_detector.py, which reuses it)
still uses HuggingFaceRouterEngine — that path was never reported broken,
so it was left untouched.

WHY TEXT DETECTION MOVED OFF HUGGING FACE:
Hugging Face's free serverless Inference API only reliably serves
large, popular, "warm" models. Community-uploaded text-classification
models (which is what free AI-text detectors are) routinely fail to
load on it — this project hit that wall twice with two different
models. Rather than guess at a third Hugging Face model, text detection
now uses Sapling.ai's dedicated AI Detector API
(https://sapling.ai/docs/api/detector/) — a purpose-built product for
exactly this task, not a community model riding on shared inference
infrastructure. It is actively maintained (documented updates as recent
as June 2026) and offers a free, rate-limited developer API key (50,000
characters/24 hours) requiring only a free account — no credit card.

WHY THIS FILE EXISTS
This file contains ALL the networking/retry/error-handling code for
every detection engine used in the project. Putting it here (instead of
copy-pasting into every detector) is what makes the system modular: if
you ever want to swap providers again, you write one new class with a
`call()`-style method and swap it in. Nothing in app.py, the detector
functions' signatures, or the frontend needs to change.
"""

import os
import time
import logging
import requests

logger = logging.getLogger("ai_content_detector")

# Hugging Face's current, supported endpoint for classic "pipeline" style
# models (text-classification, image-classification, etc.) served through
# their free hf-inference provider. Still used by image_detector.py.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models"

# Sapling.ai's dedicated AI-text-detection endpoint. Confirmed against
# their official docs at https://sapling.ai/docs/api/detector/
SAPLING_DETECT_URL = "https://api.sapling.ai/api/v1/aidetect"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 6
REQUEST_TIMEOUT_SECONDS = 30


class DetectionEngineError(Exception):
    """
    Raised whenever a detection engine cannot produce a result.

    `reason` is always a short, safe-to-display-to-the-user message.
    `kind` categorizes the failure so calling code can react differently
    (e.g. show a "Setup needed" message for a missing key, vs a
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
    Used by image_detector.py (and, through it, video_detector.py).

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


class SaplingTextEngine:
    """
    A small, reusable client for Sapling.ai's AI Detector API.
    Used by text_detector.py.

    This is a dedicated, purpose-built AI-text-detection product (not a
    community model on shared inference infrastructure), so it doesn't
    have the "model fails to load" failure mode that Hugging Face did.

    Usage:
        engine = SaplingTextEngine()
        raw_json = engine.call(text="some text to check")
        # raw_json looks like: {"score": 0.93, "sentence_scores": [...]}
    """

    def __init__(self, key_env_var: str = "SAPLING_API_KEY"):
        self.key_env_var = key_env_var

    def _get_key(self) -> str:
        key = os.getenv(self.key_env_var, "").strip()
        if not key:
            raise DetectionEngineError(
                reason=(
                    f"No Sapling API key found. Set {self.key_env_var} "
                    f"in your backend/.env file (see .env.example). Get a free "
                    f"key at https://sapling.ai/."
                ),
                kind="missing_token",
            )
        return key

    def call(self, *, text: str):
        """
        Sends text to Sapling's AI Detector endpoint. Raises
        DetectionEngineError on any failure — callers should catch this
        and turn it into a user-facing message. Retries on transient
        server errors (5xx) and timeouts, but not on 4xx client errors
        (those won't succeed on retry).
        """
        key = self._get_key()
        payload = {"key": key, "text": text}

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info("Calling Sapling AI Detector (attempt %s/%s)", attempt, MAX_RETRIES)
            try:
                response = requests.post(
                    SAPLING_DETECT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout:
                logger.warning("Request to Sapling timed out (attempt %s).", attempt)
                if attempt == MAX_RETRIES:
                    raise DetectionEngineError(
                        reason="The AI detection service took too long to respond. Please try again.",
                        kind="timeout",
                    )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            except requests.exceptions.RequestException as exc:
                logger.error("Network error calling Sapling: %s", exc)
                raise DetectionEngineError(
                    reason="Could not reach the AI detection service. Check your internet connection.",
                    kind="network_error",
                )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 401 or response.status_code == 403:
                logger.error("Sapling rejected the API key (%s).", response.status_code)
                raise DetectionEngineError(
                    reason="Your Sapling API key was rejected. Double-check it in backend/.env.",
                    kind="auth_error",
                    status_code=response.status_code,
                )

            if response.status_code == 429:
                logger.warning("Sapling rate limit hit (429).")
                raise DetectionEngineError(
                    reason="The free usage limit was reached. Please wait a bit and try again.",
                    kind="rate_limited",
                    status_code=429,
                )

            if 500 <= response.status_code < 600:
                # Server-side problem on Sapling's end — worth a brief retry.
                logger.warning(
                    "Sapling returned server error %s (attempt %s/%s), retrying...",
                    response.status_code, attempt, MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    raise DetectionEngineError(
                        reason="The AI detection service is temporarily unavailable. Please try again shortly.",
                        kind="api_error",
                        status_code=response.status_code,
                    )
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            # Any other 4xx — bad request, text too long, etc. Not worth retrying.
            logger.error("Sapling returned status %s: %s", response.status_code, response.text[:300])
            raise DetectionEngineError(
                reason=(
                    f"The AI detection service rejected the request "
                    f"(code {response.status_code}). Please try again with different text."
                ),
                kind="api_error",
                status_code=response.status_code,
            )

        # Should not normally reach here, but keep a safe fallback.
        raise DetectionEngineError(
            reason="The AI detection service is temporarily unavailable. Please try again shortly.",
            kind="timeout",
        )
