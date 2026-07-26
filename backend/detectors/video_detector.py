"""
video_detector.py
-------------------
Decides whether a VIDEO looks "AI-generated" or "human/camera-created".

HONEST LIMITATION: Free, reliable AI-video/deepfake detectors are very rare.
Instead of pretending to have one, we use a reasonable and explainable
beginner-friendly approach:

1. We open the video with OpenCV.
2. We pull out a handful of evenly-spaced frames (e.g. 6 frames across
   the video's length) - like taking 6 screenshots.
3. We run EACH frame through our existing image_detector.py logic
   (the same one used for the "Detect Image" feature).
4. We average the results across all frames to get one final answer.

This means video detection is only as strong as our image detection,
applied multiple times. It will not catch every deepfake, but it is a
transparent, working, free approach - which is exactly what we promised.
"""

import cv2
import numpy as np
import tempfile
import os

from .image_detector import _offline_analysis as image_offline_analysis


def _extract_sample_frames(video_path: str, num_frames: int = 6):
    """Pulls `num_frames` evenly spaced frames out of the video as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly spaced frame indexes (avoid the very first/last frame,
    # which are sometimes black or a fade-in/out)
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
    Saves the uploaded video bytes to a temporary file (OpenCV needs a
    real file path, it can't read video straight from memory), extracts
    frames, and analyzes each one.
    """
    suffix = os.path.splitext(original_filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = _extract_sample_frames(tmp_path, num_frames=6)
    finally:
        os.remove(tmp_path)  # always clean up, even if something goes wrong

    if not frames:
        return {
            "result": "Could not analyze",
            "confidence": 0,
            "details": {"reason": "We couldn't read any frames from this video file."},
        }

    ai_confidences = []
    human_confidences = []

    for frame_bytes in frames:
        label, confidence, _ = image_offline_analysis(frame_bytes)
        if label == "Likely AI Generated":
            ai_confidences.append(confidence)
        elif label == "Likely Human Created":
            human_confidences.append(confidence)

    ai_votes = len(ai_confidences)
    human_votes = len(human_confidences)

    if ai_votes >= human_votes:
        label = "Likely AI Generated"
        confidence = round(sum(ai_confidences) / ai_votes) if ai_votes else 50
    else:
        label = "Likely Human Created"
        confidence = round(sum(human_confidences) / human_votes) if human_votes else 50

    details = {
        "method": "frame-sampling + offline-pixel-statistics",
        "frames_analyzed": len(frames),
        "frames_flagged_ai": ai_votes,
        "frames_flagged_human": human_votes,
    }
    return {"result": label, "confidence": confidence, "details": details}
