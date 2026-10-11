"""
app.py
------
Flask web app for the DeepFake Detection System.

Usage:
    python app/app.py
Then open http://localhost:5000
"""
import os
import sys
import uuid
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.inference import DeepFakePredictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

predictor = DeepFakePredictor()


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", model_loaded=predictor.model_loaded)


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. "
                                  f"Allowed: {', '.join(config.ALLOWED_VIDEO_EXTENSIONS)}"}), 400

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(config.UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    try:
        result = predictor.predict(save_path)
    finally:
        # Clean up the uploaded file after analysis
        if os.path.exists(save_path):
            os.remove(save_path)

    status_code = 400 if "error" in result else 200
    return jsonify(result), status_code


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.model_loaded})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
