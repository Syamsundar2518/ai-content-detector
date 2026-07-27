"""
video_detector.py  (v2 — REAL AI MODEL, via frame sampling)
---------------------------------------------------------------
WHAT CHANGED FROM BEFORE:
The overall STRATEGY is the same as before (pull out a few sample frames,
analyze each one as an image, then combine the results) — that part of
the design was already good and honest. What changed is that each frame
is now sent to the REAL Hugging Face image model (via image_detector.py's
detect_image function) instead of being analyzed with pixel-math guesses.

HONEST LIMITATION (same as before, still true):
There is no reliable free "AI video detector" model. Frame-by-frame image
analysis is the best free, explainable approach available — it will not
catch every deepfake or AI video, especially ones that don't rely on
diffusion-style image generation for individual frames.

WHY WE ONLY SAMPLE 4 FRAMES (reduced from 6):
Each frame now triggers a real network call to Hugging Face, which can
take a few seconds (or up to ~20 seconds if the model needs to "wake up").
On Render's free tier, requests that take too long can time out. Sampling
fewer frames keeps total analysis time reasonable while still giving a
meaningful multi-point estimate.
"""

import cv2
import numpy as np
import tempfile
import os

from .image_detector import detect_image

FRAMES_TO_SAMPLE = 4


def _extract_sample_frames(video_path: str, num_frames: int = FRAMES_TO_SAMPLE):
    """Pulls `num_frames` evenly spaced frames out of the video as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly spaced frame indexes, avoiding the very first/last frame
    indexes = np.linspace(0, total_frames - 1, num_frames + 2)[1:-1].astype(int)

    frames_bytes = []
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            frames_bytes.append(buffer.tobytes())

    cap.release()
    return frames_bytes


def detect_video(video_bytes: bytes, original_filename: str = "upload.mp4"):
    """
    Main function called by app.py.
    Saves the uploaded video to a temporary file (OpenCV needs a real file
    path), extracts a few sample frames, sends EACH one to the real AI
    image model, then averages the results into one final verdict.
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

    suffix = os.path.splitext(original_filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = _extract_sample_frames(tmp_path)
    finally:
        os.remove(tmp_path)  # always clean up the temp file, even if something fails

    if not frames:
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "We couldn't read any frames from this video file."},
        }

    ai_scores = []
    human_scores = []
    failed_frames = 0

    for frame_bytes in frames:
        frame_result = detect_image(frame_bytes)

        # If a single frame's request fails (e.g. rate limit), skip it rather
        # than failing the whole video — we just use whichever frames worked.
        if frame_result["result"] in ("Could not analyze", "Setup needed"):
            failed_frames += 1
            continue

        confidence = frame_result["confidence"]
        if frame_result["result"] == "Likely AI Generated":
            ai_scores.append(confidence)
        else:
            human_scores.append(confidence)

    total_analyzed = len(ai_scores) + len(human_scores)

    if total_analyzed == 0:
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "All frame analysis requests failed. Please try again shortly."},
        }

    if len(ai_scores) >= len(human_scores):
        label = "Likely AI Generated"
        confidence = round(sum(ai_scores) / len(ai_scores))
    else:
        label = "Likely Human Created"
        confidence = round(sum(human_scores) / len(human_scores))

    details = {
        "method": "frame-sampling + huggingface-sdxl-detector",
        "frames_analyzed": total_analyzed,
        "frames_flagged_ai": len(ai_scores),
        "frames_flagged_human": len(human_scores),
    }
    if failed_frames:
        details["frames_skipped"] = failed_frames

    return {"result": label, "confidence": confidence, "details": details}
