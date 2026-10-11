"""
inference.py
------------
Wraps the full pipeline (face/blink extraction -> model forward pass)
for a single uploaded video, used by the Flask app.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.face_extractor import FaceExtractor
from src.preprocessing.blink_detector import blink_feature_vector, count_blinks
from src.models.cnn_lstm import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DeepFakePredictor:
    """Loads the model once and reuses it across requests."""

    def __init__(self, checkpoint_path=config.MODEL_CHECKPOINT):
        self.extractor = FaceExtractor()
        self.model = build_model()
        self.model_loaded = False

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_loaded = True
        self.model.eval()

    def predict(self, video_path):
        """Returns a dict with the prediction, confidence, and blink stats,
        or an error message if no face could be detected."""
        faces, ears, success = self.extractor.extract_sequence(video_path)

        if not success:
            return {"error": "No face could be detected in this video. "
                              "Try a clip with a clearer, front-facing view."}

        if not self.model_loaded:
            return {"error": "No trained model checkpoint found at "
                              f"{config.MODEL_CHECKPOINT}. Train a model first (see README.md)."}

        frame_tensors = torch.stack([_transform(f) for f in faces]).unsqueeze(0).to(config.DEVICE)
        blink_feats = torch.from_numpy(blink_feature_vector(ears)).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = self.model(frame_tensors, blink_feats)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        fake_prob = float(probs[1])
        real_prob = float(probs[0])
        label = "FAKE" if fake_prob > real_prob else "REAL"

        return {
            "label": label,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(max(fake_prob, real_prob), 4),
            "blink_count": int(count_blinks(ears)),
            "num_frames_analyzed": int(len(ears)),
        }
