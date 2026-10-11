"""
train.py
--------
Trains the CNN-LSTM deepfake classifier.

Usage:
    python -m src.train
"""
import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import get_dataloaders
from src.models.cnn_lstm import build_model


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if is_train else "eval", leave=False):
            frames = batch["frames"].to(config.DEVICE)
            blink_feats = batch["blink_features"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(frames, blink_feats)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * frames.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def main():
    print(f"Using device: {config.DEVICE}")
    train_loader, val_loader, _ = get_dataloaders()

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, config.MODEL_CHECKPOINT)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        # Optional: unfreeze the backbone halfway through training for fine-tuning
        if epoch == config.NUM_EPOCHS // 2 and config.FREEZE_CNN_BACKBONE:
            print("Unfreezing CNN backbone for fine-tuning...")
            model.unfreeze_backbone(num_last_blocks=20)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.LEARNING_RATE * 0.1, weight_decay=config.WEIGHT_DECAY,
            )

    print(f"Training complete. Best val_loss={best_val_loss:.4f}. "
          f"Checkpoint saved to {config.MODEL_CHECKPOINT}")


if __name__ == "__main__":
    main()
