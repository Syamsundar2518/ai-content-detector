"""
app.py
------
This is the "brain" of our backend. It is a Flask web server that:

  1. Listens for requests coming from the frontend (the website in the browser).
  2. Runs the right detector (text, image, or video).
  3. Sends the result back as JSON, e.g. {"result": "Likely AI Generated", "confidence": 87}

How to run this file:
  1. Open a terminal in the "backend" folder.
  2. Install requirements:      pip install -r requirements.txt
  3. Copy .env.example to .env (optional, for the Hugging Face token).
  4. Run the server:            python app.py
  5. It will start at:          http://127.0.0.1:5000
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from detectors.text_detector import detect_text
from detectors.image_detector import detect_image
from detectors.video_detector import detect_video

# Load variables from the .env file (like HUGGINGFACE_API_TOKEN) into memory
load_dotenv()

app = Flask(__name__)

# CORS lets our frontend (which runs on a different address, e.g.
# http://127.0.0.1:5500) send requests to this backend without the
# browser blocking it for security reasons.
CORS(app)

# ---- File upload rules ----
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi"}
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 100


def _get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@app.route("/")
def home():
    """A simple route just to confirm the server is alive."""
    return jsonify({"status": "ok", "message": "AI Content Detector backend is running."})


@app.route("/api/detect/text", methods=["POST"])
def api_detect_text():
    """
    Expects JSON body: { "text": "some text here" }
    Returns: { "result": "...", "confidence": 87, "details": {...} }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text was provided."}), 400

    if len(text) > 20000:
        return jsonify({"error": "Text is too long. Please limit to 20,000 characters."}), 400

    result = detect_text(text)
    return jsonify(result)


@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """
    Expects a multipart/form-data request with a file field named "file".
    Returns: { "result": "...", "confidence": 87, "details": {...} }
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = _get_extension(file.filename)
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": f"Unsupported image type '.{ext}'. Allowed: jpg, jpeg, png."}), 400

    file_bytes = file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return jsonify({"error": f"Image is too large ({size_mb:.1f}MB). Max is {MAX_IMAGE_SIZE_MB}MB."}), 400

    result = detect_image(file_bytes)
    return jsonify(result)


@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    """
    Expects a multipart/form-data request with a file field named "file".
    Returns: { "result": "...", "confidence": 87, "details": {...} }
    """
    if "file" not in request.files:
        return jsonify({"error": "No video file was uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = _get_extension(file.filename)
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": f"Unsupported video type '.{ext}'. Allowed: mp4, mov, avi."}), 400

    file_bytes = file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        return jsonify({"error": f"Video is too large ({size_mb:.1f}MB). Max is {MAX_VIDEO_SIZE_MB}MB."}), 400

    result = detect_video(file_bytes, original_filename=file.filename)
    return jsonify(result)


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "The uploaded file is too large."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
