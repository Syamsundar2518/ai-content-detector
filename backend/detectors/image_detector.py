"""
image_detector.py
-------------------
Decides whether an IMAGE looks "AI-generated" or "human/camera-created".

Same two-method approach as text_detector.py:

1. ONLINE (optional): if a Hugging Face API token is set, we send the image
   to a free public AI-image-detection model.

2. OFFLINE (always available): we look at basic image statistics that tend
   to differ between AI-generated images and real photos:

   - Noise pattern: real camera photos have natural sensor noise. AI images
     are often "too clean" or have unusual, very uniform noise.
   - Color histogram smoothness: AI-generated images sometimes have unusually
     smooth / evenly distributed color transitions compared to real photos.
   - Edge sharpness consistency: real photos have varied focus; some AI
     images are uniformly sharp or uniformly soft across the whole frame.

These are rough signals, not proof. We are honest about that in the API
response and the UI.
"""

import os
import io
import base64
import requests
import numpy as np
from PIL import Image

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Organika/sdxl-detector"


def _offline_analysis(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Not a valid image", 0, {"reason": "Could not read this image file."}

    img = img.resize((256, 256))  # normalize size for consistent analysis
    arr = np.array(img).astype(np.float32)

    # 1. Noise estimate: look at pixel-to-pixel differences.
    #    Very low noise across a photo-like image can hint at AI smoothing.
    diff_x = np.abs(np.diff(arr, axis=1))
    diff_y = np.abs(np.diff(arr, axis=0))
    noise_level = float((diff_x.mean() + diff_y.mean()) / 2)

    # 2. Color histogram spread (standard deviation of pixel values).
    #    AI images sometimes cluster colors more tightly (lower spread).
    color_std = float(arr.std())

    # 3. High-frequency detail ratio (rough edge/texture measure) using a
    #    simple gradient magnitude, no external CV model needed.
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_energy = float((gx.mean() + gy.mean()) / 2)

    # ---- Combine into an AI-probability score (0 to 1) ----
    ai_score = 0.5

    # Very smooth / low noise -> more AI-like
    if noise_level < 4.0:
        ai_score += 0.20
    elif noise_level > 9.0:
        ai_score -= 0.15

    # Tight color distribution -> more AI-like
    if color_std < 55:
        ai_score += 0.15
    elif color_std > 75:
        ai_score -= 0.10

    # Unusually uniform edge energy (too "perfect") -> more AI-like
    if edge_energy < 3.5:
        ai_score += 0.10

    ai_score = max(0.05, min(0.95, ai_score))

    label = "Likely AI Generated" if ai_score >= 0.5 else "Likely Human Created"
    confidence = round(ai_score * 100) if ai_score >= 0.5 else round((1 - ai_score) * 100)

    details = {
        "method": "offline-pixel-statistics",
        "noise_level": round(noise_level, 2),
        "color_spread": round(color_std, 2),
        "edge_energy": round(edge_energy, 2),
    }
    return label, confidence, details


def _online_analysis(image_bytes: bytes, hf_token: str):
    try:
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = requests.post(HF_MODEL_URL, headers=headers, data=image_bytes, timeout=20)
        if response.status_code != 200:
            return None

        data = response.json()
        # Expected shape: [{"label": "artificial", "score": 0.9}, {"label": "human", "score": 0.1}]
        if not isinstance(data, list) or not data:
            return None

        ai_entry = next((d for d in data if "artificial" in d["label"].lower()
                          or "ai" in d["label"].lower() or "fake" in d["label"].lower()), None)
        if not ai_entry:
            return None

        score = ai_entry["score"]
        label = "Likely AI Generated" if score >= 0.5 else "Likely Human Created"
        confidence = round(score * 100) if score >= 0.5 else round((1 - score) * 100)
        return label, confidence, {"method": "huggingface-sdxl-detector"}

    except Exception:
        return None


def detect_image(image_bytes: bytes):
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()

    if hf_token:
        result = _online_analysis(image_bytes, hf_token)
        if result:
            label, confidence, details = result
            return {"result": label, "confidence": confidence, "details": details}

    label, confidence, details = _offline_analysis(image_bytes)
    return {"result": label, "confidence": confidence, "details": details}
