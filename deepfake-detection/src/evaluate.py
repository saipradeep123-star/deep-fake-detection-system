"""
evaluate.py
-----------
Loads the best checkpoint and reports accuracy, F1, confusion matrix,
and ROC-AUC on the held-out test split.

Usage:
    python -m src.evaluate
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report,
)
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.models.cnn_lstm import build_model


def main():
    _, _, test_loader = get_dataloaders()

    model = build_model()
    checkpoint = torch.load(config.MODEL_CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            frames = batch["frames"].to(config.DEVICE)
            blink_feats = batch["blink_features"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            logits = model(frames, blink_feats)
            probs = F.softmax(logits, dim=1)[:, 1]  # probability of "fake"
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test F1-score : {f1:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred) [real, fake]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
