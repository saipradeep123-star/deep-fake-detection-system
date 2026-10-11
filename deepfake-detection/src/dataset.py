"""
dataset.py
----------
PyTorch Dataset that reads the .npz files produced by
src/preprocessing/preprocess_dataset.py (face sequences + EAR values)
and returns normalized tensors ready for the CNN-LSTM model.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.blink_detector import blink_feature_vector

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepFakeSequenceDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.records = pd.read_csv(csv_path)

        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                *base_transforms,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                *base_transforms,
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        data = np.load(row["path"])
        faces = data["faces"]      # (T, H, W, 3) uint8
        ears = data["ears"]        # (T,)
        label = int(row["label"])

        frame_tensors = torch.stack([self.transform(f) for f in faces])  # (T, 3, H, W)
        blink_feats = torch.from_numpy(blink_feature_vector(ears))

        return {
            "frames": frame_tensors,
            "blink_features": blink_feats,
            "label": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    from torch.utils.data import DataLoader

    train_ds = DeepFakeSequenceDataset(os.path.join(config.SPLITS_DIR, "train.csv"), augment=True)
    val_ds = DeepFakeSequenceDataset(os.path.join(config.SPLITS_DIR, "val.csv"), augment=False)
    test_ds = DeepFakeSequenceDataset(os.path.join(config.SPLITS_DIR, "test.csv"), augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
